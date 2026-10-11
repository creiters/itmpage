export class NanoTokenizer {
  constructor(vocabSize = 1024) {
    this.vocabSize = vocabSize;
    this.tokenToId = new Map();
    this.idToToken = new Map();
    this.buildVocab();
  }

  buildVocab() {
    const specials = ['<pad>', '<bos>', '<eos>', '<unk>'];
    specials.forEach((token, id) => {
      this.tokenToId.set(token, id);
      this.idToToken.set(id, token);
    });

    let currentId = specials.length;
    for (let i = 32; i < 127 && currentId < this.vocabSize; i++) {
      const ch = String.fromCharCode(i);
      this.tokenToId.set(ch, currentId);
      this.idToToken.set(currentId, ch);
      currentId++;
    }
  }

  encode(text) {
    if (!text) return [1];
    const tokens = [1];
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      tokens.push(this.tokenToId.has(ch) ? this.tokenToId.get(ch) : 3);
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
    this.dim = config.dim || 128;
    this.layers = config.layers || 4;
    this.heads = config.heads || 4;
    this.headDim = this.dim / this.heads; // 32
    this.vocabSize = config.vocabSize || 1024;
    this.maxSeqLen = config.maxSeqLen || 64;

    this.weights = {};
    this.grads = {};
    this.adamM = {};
    this.adamV = {};
    this.stepCount = 0;

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

    const registerParam = (name, size) => {
      this.weights[name] = rand(size);
      this.grads[name] = new Float32Array(size);
      this.adamM[name] = new Float32Array(size);
      this.adamV[name] = new Float32Array(size);
    };

    registerParam('wte', this.vocabSize * this.dim);
    registerParam('lm_head', this.vocabSize * this.dim);

    for (let l = 0; l < this.layers; l++) {
      registerParam(`blk_${l}_wq`, this.dim * this.dim);
      registerParam(`blk_${l}_wk`, this.dim * this.dim);
      registerParam(`blk_${l}_wv`, this.dim * this.dim);
      registerParam(`blk_${l}_wo`, this.dim * this.dim);
      registerParam(`blk_${l}_wgate`, this.dim * (this.dim * 2));
      registerParam(`blk_${l}_wup`, this.dim * (this.dim * 2));
      registerParam(`blk_${l}_wdown`, (this.dim * 2) * this.dim);
    }
  }

  zeroGrads() {
    for (const key in this.grads) {
      this.grads[key].fill(0.0);
    }
  }

  rmsNorm(x, eps = 1e-5) {
    let sumSq = 0.0;
    for (let i = 0; i < x.length; i++) sumSq += x[i] * x[i];
    const invScale = 1.0 / Math.sqrt(sumSq / x.length + eps);
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i] * invScale;
    return { out, invScale };
  }

  forwardStepWithTape(tokenId, pos, kvCache) {
    const tape = { pos, tokenId, layers: [] };
    const x = new Float32Array(this.dim);

    const embOffset = tokenId * this.dim;
    for (let i = 0; i < this.dim; i++) x[i] = this.weights.wte[embOffset + i];
    tape.xInit = new Float32Array(x);

    for (let l = 0; l < this.layers; l++) {
      const layerTape = {};
      const norm1 = this.rmsNorm(x);
      layerTape.norm1 = norm1;

      const q = new Float32Array(this.dim);
      const k = new Float32Array(this.dim);
      const v = new Float32Array(this.dim);

      for (let r = 0; r < this.dim; r++) {
        let sq = 0, sk = 0, sv = 0;
        const off = r * this.dim;
        for (let c = 0; c < this.dim; c++) {
          sq += norm1.out[c] * this.weights[`blk_${l}_wq`][off + c];
          sk += norm1.out[c] * this.weights[`blk_${l}_wk`][off + c];
          sv += norm1.out[c] * this.weights[`blk_${l}_wv`][off + c];
        }
        q[r] = sq;
        k[r] = sk;
        v[r] = sv;
      }

      layerTape.q = q;
      layerTape.k = k;
      layerTape.v = v;

      kvCache.k[l][pos] = k;
      kvCache.v[l][pos] = v;

      const att = new Float32Array(this.dim);
      layerTape.attWeights = [];

      for (let h = 0; h < this.heads; h++) {
        const hOffset = h * this.headDim;
        let maxScore = -Infinity;
        const scores = new Float32Array(pos + 1);

        for (let t = 0; t <= pos; t++) {
          const histK = kvCache.k[l][t];
          let score = 0.0;
          for (let d = 0; d < this.headDim; d++) {
            score += q[hOffset + d] * histK[hOffset + d];
          }
          score /= Math.sqrt(this.headDim);
          scores[t] = score;
          if (score > maxScore) maxScore = score;
        }

        let sumExp = 0.0;
        const weights = new Float32Array(pos + 1);
        for (let t = 0; t <= pos; t++) {
          weights[t] = Math.exp(scores[t] - maxScore);
          sumExp += weights[t];
        }
        for (let t = 0; t <= pos; t++) {
          weights[t] /= sumExp;
          const histV = kvCache.v[l][t];
          for (let d = 0; d < this.headDim; d++) {
            att[hOffset + d] += histV[hOffset + d] * weights[t];
          }
        }
        layerTape.attWeights[h] = weights;
      }

      layerTape.att = att;

      const outAtt = new Float32Array(this.dim);
      for (let r = 0; r < this.dim; r++) {
        let sum = 0.0;
        const off = r * this.dim;
        for (let c = 0; c < this.dim; c++) sum += att[c] * this.weights[`blk_${l}_wo`][off + c];
        outAtt[r] = sum;
        x[r] += sum;
      }
      layerTape.xMid = new Float32Array(x);

      const norm2 = this.rmsNorm(x);
      layerTape.norm2 = norm2;

      const h1 = new Float32Array(this.dim * 2);
      const h2 = new Float32Array(this.dim * 2);
      const siluH1 = new Float32Array(this.dim * 2);

      for (let r = 0; r < this.dim * 2; r++) {
        let sGate = 0, sUp = 0;
        const off = r * this.dim;
        for (let c = 0; c < this.dim; c++) {
          sGate += norm2.out[c] * this.weights[`blk_${l}_wgate`][off + c];
          sUp += norm2.out[c] * this.weights[`blk_${l}_wup`][off + c];
        }
        h1[r] = sGate;
        h2[r] = sUp;
        const sig = 1.0 / (1.0 + Math.exp(-sGate));
        siluH1[r] = sGate * sig;
      }

      layerTape.h1 = h1;
      layerTape.h2 = h2;
      layerTape.siluH1 = siluH1;

      const mlpAct = new Float32Array(this.dim * 2);
      for (let i = 0; i < this.dim * 2; i++) mlpAct[i] = siluH1[i] * h2[i];
      layerTape.mlpAct = mlpAct;

      const outFF = new Float32Array(this.dim);
      for (let r = 0; r < this.dim; r++) {
        let sum = 0.0;
        const off = r * (this.dim * 2);
        for (let c = 0; c < this.dim * 2; c++) sum += mlpAct[c] * this.weights[`blk_${l}_wdown`][off + c];
        outFF[r] = sum;
        x[r] += sum;
      }

      tape.layers.push(layerTape);
    }

    const normFinal = this.rmsNorm(x);
    tape.normFinal = normFinal;

    const logits = new Float32Array(this.vocabSize);
    for (let r = 0; r < this.vocabSize; r++) {
      let sum = 0.0;
      const off = r * this.dim;
      for (let c = 0; c < this.dim; c++) sum += normFinal.out[c] * this.weights.lm_head[off + c];
      logits[r] = sum;
    }

    return { logits, tape };
  }

  backwardStep(targetTokenId, logits, tape) {
    let maxLogit = -Infinity;
    for (let i = 0; i < this.vocabSize; i++) if (logits[i] > maxLogit) maxLogit = logits[i];

    let sumExp = 0.0;
    const probs = new Float32Array(this.vocabSize);
    for (let i = 0; i < this.vocabSize; i++) {
      probs[i] = Math.exp(logits[i] - maxLogit);
      sumExp += probs[i];
    }
    for (let i = 0; i < this.vocabSize; i++) probs[i] /= sumExp;

    const loss = -Math.log(Math.max(1e-7, probs[targetTokenId]));

    const dLogits = new Float32Array(probs);
    dLogits[targetTokenId] -= 1.0;

    const dNormFinal = new Float32Array(this.dim);
    for (let r = 0; r < this.vocabSize; r++) {
      const dL = dLogits[r];
      if (Math.abs(dL) < 1e-9) continue;
      const off = r * this.dim;
      for (let c = 0; c < this.dim; c++) {
        this.grads.lm_head[off + c] += dL * tape.normFinal.out[c];
        dNormFinal[c] += dL * this.weights.lm_head[off + c];
      }
    }

    const dx = new Float32Array(this.dim);
    const scaleFinal = tape.normFinal.invScale;
    for (let i = 0; i < this.dim; i++) dx[i] = dNormFinal[i] * scaleFinal;

    for (let l = this.layers - 1; l >= 0; l--) {
      const layerTape = tape.layers[l];

      const dMlpAct = new Float32Array(this.dim * 2);
      for (let r = 0; r < this.dim; r++) {
        const off = r * (this.dim * 2);
        const dxi = dx[r];
        for (let c = 0; c < this.dim * 2; c++) {
          this.grads[`blk_${l}_wdown`][off + c] += dxi * layerTape.mlpAct[c];
          dMlpAct[c] += dxi * this.weights[`blk_${l}_wdown`][off + c];
        }
      }

      const dNorm2 = new Float32Array(this.dim);
      for (let r = 0; r < this.dim * 2; r++) {
        const dAct = dMlpAct[r];
        const h1 = layerTape.h1[r];
        const h2 = layerTape.h2[r];
        const sig = 1.0 / (1.0 + Math.exp(-h1));

        const dSilu = dAct * h2;
        const dh2 = dAct * layerTape.siluH1[r];
        const dh1 = dSilu * (sig + h1 * sig * (1.0 - sig));

        const off = r * this.dim;
        for (let c = 0; c < this.dim; c++) {
          this.grads[`blk_${l}_wgate`][off + c] += dh1 * layerTape.norm2.out[c];
          this.grads[`blk_${l}_wup`][off + c] += dh2 * layerTape.norm2.out[c];
          dNorm2[c] += dh1 * this.weights[`blk_${l}_wgate`][off + c] + dh2 * this.weights[`blk_${l}_wup`][off + c];
        }
      }

      for (let i = 0; i < this.dim; i++) dx[i] += dNorm2[i] * layerTape.norm2.invScale;

      const dAtt = new Float32Array(this.dim);
      for (let r = 0; r < this.dim; r++) {
        const off = r * this.dim;
        const dxi = dx[r];
        for (let c = 0; c < this.dim; c++) {
          this.grads[`blk_${l}_wo`][off + c] += dxi * layerTape.att[c];
          dAtt[c] += dxi * this.weights[`blk_${l}_wo`][off + c];
        }
      }

      const dNorm1 = new Float32Array(this.dim);
      for (let r = 0; r < this.dim; r++) {
        const off = r * this.dim;
        const dQ = dAtt[r] * 0.5;
        for (let c = 0; c < this.dim; c++) {
          this.grads[`blk_${l}_wq`][off + c] += dQ * layerTape.norm1.out[c];
          this.grads[`blk_${l}_wk`][off + c] += dQ * layerTape.norm1.out[c];
          this.grads[`blk_${l}_wv`][off + c] += dQ * layerTape.norm1.out[c];
          dNorm1[c] += dQ * (this.weights[`blk_${l}_wq`][off + c] + this.weights[`blk_${l}_wv`][off + c]);
        }
      }

      for (let i = 0; i < this.dim; i++) dx[i] += dNorm1[i] * layerTape.norm1.invScale;
    }

    const embOffset = tape.tokenId * this.dim;
    for (let i = 0; i < this.dim; i++) {
      this.grads.wte[embOffset + i] += dx[i];
    }

    return loss;
  }

  stepAdamW(lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0.01) {
    this.stepCount++;
    const clipVal = 1.0;

    for (const key in this.weights) {
      const W = this.weights[key];
      const G = this.grads[key];
      const M = this.adamM[key];
      const V = this.adamV[key];

      const len = W.length;
      const b1 = beta1;
      const b2 = beta2;
      const step = this.stepCount;

      const biasCorr1 = 1.0 - Math.pow(b1, step);
      const biasCorr2 = 1.0 - Math.pow(b2, step);

      for (let i = 0; i < len; i++) {
        let g = G[i];
        if (g > clipVal) g = clipVal;
        else if (g < -clipVal) g = -clipVal;

        W[i] -= lr * weightDecay * W[i];

        M[i] = b1 * M[i] + (1.0 - b1) * g;
        V[i] = b2 * V[i] + (1.0 - b2) * (g * g);

        const mHat = M[i] / biasCorr1;
        const vHat = V[i] / biasCorr2;

        W[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
      }
    }
  }

  trainSequence(tokenIds, lr = 0.001) {
    if (tokenIds.length < 2) return 0.0;
    this.zeroGrads();

    let seqLoss = 0.0;
    const seqLen = Math.min(tokenIds.length - 1, this.maxSeqLen);

    const kvCache = {
      k: Array.from({ length: this.layers }, () => []),
      v: Array.from({ length: this.layers }, () => [])
    };

    for (let pos = 0; pos < seqLen; pos++) {
      const inputId = tokenIds[pos];
      const targetId = tokenIds[pos + 1];

      const { logits, tape } = this.forwardStepWithTape(inputId, pos, kvCache);
      const stepLoss = this.backwardStep(targetId, logits, tape);
      seqLoss += stepLoss;
    }

    this.stepAdamW(lr);
    return seqLoss / seqLen;
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
      const { logits } = this.forwardStepWithTape(currentToken, pos, kvCache);

      if (pos >= inputTokens.length - 1) {
        let bestToken = 1;
        let maxLogit = -Infinity;
        const recent = generatedTokens.slice(-6);

        for (let i = 4; i < this.vocabSize; i++) {
          let score = logits[i];
          if (recent.includes(i)) score -= 2.5;
          if (score > maxLogit) {
            maxLogit = score;
            bestToken = i;
          }
        }

        if (bestToken === 2) break;
        const char = this.tokenizer.decode([bestToken]);
        fullText += char;
        generatedTokens.push(bestToken);

        if (onToken) onToken(char, fullText);
      }
    }

    return fullText;
  }

  serializeToSegmentArray() {
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