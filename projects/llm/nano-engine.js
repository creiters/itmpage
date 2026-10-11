export class NanoTokenizer {
  constructor(vocabSize = 1024) {
    this.vocabSize = vocabSize;
    this.tokenToId = new Map();
    this.idToToken = new Map();
    this.buildVocab();
  }

  buildVocab() {
    // 0: PAD, 1: BOS, 2: EOS, 3: UNK
    const specials = ['<pad>', '<bos>', '<eos>', '<unk>'];
    specials.forEach((token, id) => {
      this.tokenToId.set(token, id);
      this.idToToken.set(id, token);
    });

    let currentId = specials.length;
    // Basic printable ASCII byte vocabulary
    for (let i = 32; i < 127 && currentId < this.vocabSize; i++) {
      const ch = String.fromCharCode(i);
      this.tokenToId.set(ch, currentId);
      this.idToToken.set(currentId, ch);
      currentId++;
    }
  }

  encode(text) {
    if (!text) return [1];
    const tokens = [1]; // Start with BOS
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (this.tokenToId.has(ch)) {
        tokens.push(this.tokenToId.get(ch));
      } else {
        tokens.push(3); // UNK
      }
    }
    return tokens;
  }

  decode(tokens) {
    if (!tokens || !tokens.length) return '';
    return tokens
      .filter((id) => id > 3)
      .map((id) => this.idToToken.get(id) || '')
      .join('');
  }
}

export class NanoTransformer {
  constructor(config = {}) {
    this.dim = config.dim || 256;
    this.layers = config.layers || 6;
    this.heads = config.heads || 4;
    this.headDim = this.dim / this.heads; // 64
    this.vocabSize = config.vocabSize || 1024;
    this.maxSeqLen = config.maxSeqLen || 256;
    this.weights = {};
    this.tokenizer = new NanoTokenizer(this.vocabSize);
    this.initWeights();
  }

  initWeights() {
    const scale = Math.sqrt(2.0 / this.dim);
    const rand = (size) => {
      const arr = new Float32Array(size);
      for (let i = 0; i < size; i++) arr[i] = (Math.random() - 0.5) * 2 * scale;
      return arr;
    };

    // Embedding & Output Projection
    this.weights.wte = rand(this.vocabSize * this.dim);
    this.weights.lm_head = rand(this.vocabSize * this.dim);

    // Transformer Blocks
    for (let l = 0; l < this.layers; l++) {
      this.weights[`blk_${l}_wq`] = rand(this.dim * this.dim);
      this.weights[`blk_${l}_wk`] = rand(this.dim * this.dim);
      this.weights[`blk_${l}_wv`] = rand(this.dim * this.dim);
      this.weights[`blk_${l}_wo`] = rand(this.dim * this.dim);
      this.weights[`blk_${l}_wgate`] = rand(this.dim * (this.dim * 2));
      this.weights[`blk_${l}_wup`] = rand(this.dim * (this.dim * 2));
      this.weights[`blk_${l}_wdown`] = rand((this.dim * 2) * this.dim);
    }
  }

  rmsNorm(x, eps = 1e-5) {
    let sumSq = 0.0;
    for (let i = 0; i < x.length; i++) sumSq += x[i] * x[i];
    const scale = 1.0 / Math.sqrt(sumSq / x.length + eps);
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i] * scale;
    return out;
  }

  matVecMul(out, x, W, outDim, inDim) {
    for (let r = 0; r < outDim; r++) {
      let acc = 0.0;
      const rowOffset = r * inDim;
      for (let c = 0; c < inDim; c++) {
        acc += x[c] * W[rowOffset + c];
      }
      out[r] = acc;
    }
  }

  forwardStep(tokenId, pos, kvCache) {
    const x = new Float32Array(this.dim);

    // Embedding Lookup
    const embOffset = tokenId * this.dim;
    for (let i = 0; i < this.dim; i++) x[i] = this.weights.wte[embOffset + i];

    // Layers
    for (let l = 0; l < this.layers; l++) {
      const xNorm = this.rmsNorm(x);
      const q = new Float32Array(this.dim);
      const k = new Float32Array(this.dim);
      const v = new Float32Array(this.dim);

      this.matVecMul(q, xNorm, this.weights[`blk_${l}_wq`], this.dim, this.dim);
      this.matVecMul(k, xNorm, this.weights[`blk_${l}_wk`], this.dim, this.dim);
      this.matVecMul(v, xNorm, this.weights[`blk_${l}_wv`], this.dim, this.dim);

      kvCache.k[l][pos] = k;
      kvCache.v[l][pos] = v;

      // Multi-Head Attention
      const att = new Float32Array(this.dim);
      for (let h = 0; h < this.heads; h++) {
        const hOffset = h * this.headDim;
        let maxScore = -Infinity;

        for (let t = 0; t <= pos; t++) {
          const histK = kvCache.k[l][t];
          let score = 0.0;
          for (let d = 0; d < this.headDim; d++) {
            score += q[hOffset + d] * histK[hOffset + d];
          }
          score /= Math.sqrt(this.headDim);
          if (score > maxScore) maxScore = score;
        }

        const attWeight = 1.0 / (1.0 + Math.exp(-Math.max(-10, Math.min(10, maxScore))));
        for (let d = 0; d < this.headDim; d++) {
          att[hOffset + d] = v[hOffset + d] * attWeight;
        }
      }

      const outAtt = new Float32Array(this.dim);
      this.matVecMul(outAtt, att, this.weights[`blk_${l}_wo`], this.dim, this.dim);
      for (let i = 0; i < this.dim; i++) x[i] += outAtt[i];

      // SwiGLU Feed-Forward Network
      const xNormFF = this.rmsNorm(x);
      const h1 = new Float32Array(this.dim * 2);
      const h2 = new Float32Array(this.dim * 2);

      this.matVecMul(h1, xNormFF, this.weights[`blk_${l}_wgate`], this.dim * 2, this.dim);
      this.matVecMul(h2, xNormFF, this.weights[`blk_${l}_wup`], this.dim * 2, this.dim);

      for (let i = 0; i < this.dim * 2; i++) {
        const silu = h1[i] / (1.0 + Math.exp(-h1[i]));
        h1[i] = silu * h2[i];
      }

      const outFF = new Float32Array(this.dim);
      this.matVecMul(outFF, h1, this.weights[`blk_${l}_wdown`], this.dim, this.dim * 2);
      for (let i = 0; i < this.dim; i++) x[i] += outFF[i];
    }

    // Output Projection to Vocabulary Logits
    const xFinal = this.rmsNorm(x);
    const logits = new Float32Array(this.vocabSize);
    this.matVecMul(logits, xFinal, this.weights.lm_head, this.vocabSize, this.dim);

    return logits;
  }

  generate(prompt, maxTokens = 32, onToken = null) {
    const inputTokens = this.tokenizer.encode(prompt);
    const generatedTokens = [...inputTokens];
    const kvCache = {
      k: Array.from({ length: this.layers }, () => []),
      v: Array.from({ length: this.layers }, () => [])
    };

    let fullText = '';

    for (let pos = 0; pos < inputTokens.length + maxTokens; pos++) {
      const currentToken = pos < inputTokens.length ? inputTokens[pos] : generatedTokens[pos];
      const logits = this.forwardStep(currentToken, pos, kvCache);

      if (pos >= inputTokens.length - 1) {
        // Temperature Sampling with Repetition Penalty
        let bestToken = 1;
        let maxLogit = -Infinity;
        const recent = generatedTokens.slice(-6);

        for (let i = 4; i < this.vocabSize; i++) {
          let score = logits[i];
          if (recent.includes(i)) score -= 2.0;
          if (score > maxLogit) {
            maxLogit = score;
            bestToken = i;
          }
        }

        if (bestToken === 2) break; // EOS
        const char = this.tokenizer.decode([bestToken]);
        fullText += char;
        generatedTokens.push(bestToken);

        if (onToken) onToken(char, fullText);
      }
    }

    return fullText;
  }

  trainStep(text, lr = 0.001) {
    const tokens = this.tokenizer.encode(text);
    if (tokens.length < 2) return 0.0;

    let totalLoss = 0.0;
    const kvCache = {
      k: Array.from({ length: this.layers }, () => []),
      v: Array.from({ length: this.layers }, () => [])
    };

    // Forward pass and gradient step over sequence
    for (let pos = 0; pos < tokens.length - 1; pos++) {
      const input = tokens[pos];
      const target = tokens[pos + 1];
      const logits = this.forwardStep(input, pos, kvCache);

      // Softmax probabilities
      let maxLogit = -Infinity;
      for (let i = 0; i < this.vocabSize; i++) {
        if (logits[i] > maxLogit) maxLogit = logits[i];
      }

      let sumExp = 0.0;
      const probs = new Float32Array(this.vocabSize);
      for (let i = 0; i < this.vocabSize; i++) {
        probs[i] = Math.exp(logits[i] - maxLogit);
        sumExp += probs[i];
      }
      for (let i = 0; i < this.vocabSize; i++) probs[i] /= sumExp;

      // Cross-Entropy Loss
      const loss = -Math.log(Math.max(1e-7, probs[target]));
      totalLoss += loss;

      // Output weight update
      const headOffset = target * this.dim;
      for (let d = 0; d < this.dim; d++) {
        const grad = probs[target] - 1.0;
        this.weights.lm_head[headOffset + d] -= lr * grad * 0.01;
      }
    }

    return totalLoss / (tokens.length - 1);
  }

  serializeToMerkleArray() {
    let totalElements = 0;
    for (const key in this.weights) totalElements += this.weights[key].length;

    const buffer = new Float32Array(totalElements);
    let offset = 0;
    for (const key in this.weights) {
      buffer.set(this.weights[key], offset);
      offset += this.weights[key].length;
    }
    return buffer.buffer;
  }
}
