/**
 * nano-engine.js - Word-Level Micro-Transformer with Dynamic Vocab Expansion & Autograd
 */

export class NanoTokenizer {
  constructor() {
    this.tokenToId = new Map();
    this.idToToken = new Map();
    this.specials = ['<pad>', '<bos>', '<eos>', '<unk>', '<user>', '<bot>'];
    this.buildVocab();
  }

  buildVocab() {
    this.specials.forEach((tok, id) => {
      this.tokenToId.set(tok, id);
      this.idToToken.set(id, tok);
    });

    const baseWords = [
      "Hello", "Hi", "Hey", "How", "are", "you", "I", "am", "doing", "well",
      "great", "fine", "good", "What", "is", "your", "name", "My", "JSONLLM",
      "an", "offline", "browser", "AI", "assistant", "Yes", "No", "can", "help",
      "with", "code", "HTML", "C", "and", "dialogue", "today", "Who", "created",
      "runs", "entirely", "on", "device", "Thanks", "Thank", "welcome", "Goodbye",
      "Bye", "ready", "System", "Status", "OK", "Please", "ask", "a", "question",
      ".", ",", "!", "?", ":", ";", "'", "\"", "\n"
    ];

    let id = this.specials.length;
    for (const w of baseWords) {
      if (!this.tokenToId.has(w)) {
        this.tokenToId.set(w, id);
        this.idToToken.set(id, w);
        id++;
      }
    }
    this.vocabSize = id;
  }

  registerWord(word) {
    if (!this.tokenToId.has(word)) {
      const id = this.vocabSize++;
      this.tokenToId.set(word, id);
      this.idToToken.set(id, word);
      return id;
    }
    return this.tokenToId.get(word);
  }

  encode(text) {
    if (!text) return [1];
    const tokens = [1]; // <bos>
    const regex = /(<user>|<bot>|<eos>|<bos>|[a-zA-Z0-9]+|[.,!?:;'"\n])/g;
    const matches = text.match(regex) || [];

    for (const piece of matches) {
      if (this.tokenToId.has(piece)) {
        tokens.push(this.tokenToId.get(piece));
      } else {
        const newId = this.registerWord(piece);
        tokens.push(newId);
      }
    }
    return tokens;
  }

  decode(tokens) {
    if (!tokens || !tokens.length) return '';
    const parts = [];
    for (const id of tokens) {
      if (id >= this.specials.length) {
        parts.push(this.idToToken.get(id) || '');
      }
    }
    return parts.join(' ').replace(/ ([.,!?:;'])/g, '$1');
  }
}

export class NanoTransformer {
  constructor(config = {}) {
    this.dim = config.dim || 64;
    this.layers = config.layers || 2;
    this.heads = config.heads || 2;
    this.headDim = this.dim / this.heads;
    this.maxSeqLen = config.maxSeqLen || 64;
    this.stepCount = config.stepCount || 0;

    this.tokenizer = new NanoTokenizer();
    this.vocabSize = Math.max(config.vocabSize || 128, this.tokenizer.vocabSize);

    this.weights = {};
    this.grads = {};
    this.adamM = {};
    this.adamV = {};

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

  ensureVocabCapacity(requiredSize) {
    if (requiredSize <= this.vocabSize) return;
    const oldSize = this.vocabSize;
    this.vocabSize = requiredSize;

    const expand = (name) => {
      const oldArr = this.weights[name];
      const newArr = new Float32Array(requiredSize * this.dim);
      newArr.set(oldArr);
      const scale = Math.sqrt(2.0 / this.dim);
      for (let i = oldArr.length; i < newArr.length; i++) {
        newArr[i] = (Math.random() - 0.5) * 2 * scale;
      }
      this.weights[name] = newArr;
      this.grads[name] = new Float32Array(requiredSize * this.dim);
      this.adamM[name] = new Float32Array(requiredSize * this.dim);
      this.adamV[name] = new Float32Array(requiredSize * this.dim);
    };

    expand('wte');
    expand('lm_head');
  }

  rmsNorm(x, eps = 1e-5) {
    let ss = 0.0;
    for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
    const invScale = 1.0 / Math.sqrt(ss / x.length + eps);
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i] * invScale;
    return { out, invScale };
  }

  forwardStepWithTape(tokenId, pos, kvCache) {
    const tape = { pos, tokenId, layers: [] };
    const x = new Float32Array(this.dim);

    const safeTok = Math.max(0, Math.min(tokenId, this.vocabSize - 1));
    const embOffset = safeTok * this.dim;
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
      for (let h = 0; h < this.heads; h++) {
        const hOffset = h * this.headDim;
        let maxScore = -Infinity;
        const scores = new Float32Array(pos + 1);

        for (let t = 0; t <= pos; t++) {
          const histK = kvCache.k[l][t];
          if (!histK) continue;
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
          weights[t] /= (sumExp || 1.0);
          const histV = kvCache.v[l][t];
          if (!histV) continue;
          for (let d = 0; d < this.headDim; d++) {
            att[hOffset + d] += histV[hOffset + d] * weights[t];
          }
        }
      }

      layerTape.att = att;
      for (let r = 0; r < this.dim; r++) {
        let sum = 0.0;
        const off = r * this.dim;
        for (let c = 0; c < this.dim; c++) sum += att[c] * this.weights[`blk_${l}_wo`][off + c];
        x[r] += sum;
      }

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
        siluH1[r] = sGate / (1.0 + Math.exp(-sGate));
      }

      layerTape.h1 = h1;
      layerTape.h2 = h2;
      layerTape.siluH1 = siluH1;

      const mlpAct = new Float32Array(this.dim * 2);
      for (let i = 0; i < this.dim * 2; i++) mlpAct[i] = siluH1[i] * h2[i];
      layerTape.mlpAct = mlpAct;

      for (let r = 0; r < this.dim; r++) {
        let sum = 0.0;
        const off = r * (this.dim * 2);
        for (let c = 0; c < this.dim; c++) sum += mlpAct[c] * this.weights[`blk_${l}_wdown`][off + c];
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

  trainPair(inputSeq, targetToken, kvCache, lr = 0.003) {
    const pos = inputSeq.length - 1;
    const lastTok = inputSeq[pos];
    const { logits, tape } = this.forwardStepWithTape(lastTok, pos, kvCache);

    let maxLogit = -Infinity;
    for (let i = 0; i < this.vocabSize; i++) if (logits[i] > maxLogit) maxLogit = logits[i];

    let sumExp = 0.0;
    const probs = new Float32Array(this.vocabSize);
    for (let i = 0; i < this.vocabSize; i++) {
      probs[i] = Math.exp(logits[i] - maxLogit);
      sumExp += probs[i];
    }
    for (let i = 0; i < this.vocabSize; i++) probs[i] /= (sumExp || 1.0);

    const safeTarget = Math.min(targetToken, this.vocabSize - 1);
    const loss = -Math.log(Math.max(1e-7, probs[safeTarget]));
    probs[safeTarget] -= 1.0;

    const dNormFinal = new Float32Array(this.dim);
    for (let r = 0; r < this.vocabSize; r++) {
      const dL = probs[r];
      if (Math.abs(dL) < 1e-8) continue;
      const off = r * this.dim;
      for (let c = 0; c < this.dim; c++) {
        this.weights.lm_head[off + c] -= lr * (dL * tape.normFinal.out[c]);
        dNormFinal[c] += dL * this.weights.lm_head[off + c];
      }
    }

    const embOffset = Math.min(lastTok, this.vocabSize - 1) * this.dim;
    for (let i = 0; i < this.dim; i++) {
      this.weights.wte[embOffset + i] -= lr * (dNormFinal[i] * tape.normFinal.invScale);
    }

    return loss;
  }
}
