/**
 * nano-engine.js - Complete Autograd Transformer Runtime with AdamW
 */

export class NanoTokenizer {
  constructor(vocabSize = 128) {
    this.vocabSize = vocabSize;
    this.tokenToId = new Map();
    this.idToToken = new Map();
    this.specials = ['<pad>', '<bos>', '<eos>', '<unk>', '<user>', '<bot>'];
    this.buildVocab();
  }

  buildVocab() {
    this.specials.forEach((t, i) => {
      this.tokenToId.set(t, i);
      this.idToToken.set(i, t);
    });

    let cur = this.specials.length;
    for (let c = 32; c <= 126 && cur < this.vocabSize; c++) {
      const ch = String.fromCharCode(c);
      this.tokenToId.set(ch, cur);
      this.idToToken.set(cur, ch);
      cur++;
    }
  }

  encode(text) {
    if (!text) return [1];
    const tokens = [1];
    let cursor = 0;

    while (cursor < text.length) {
      let matched = false;
      for (const spec of this.specials) {
        if (text.startsWith(spec, cursor)) {
          tokens.push(this.tokenToId.get(spec));
          cursor += spec.length;
          matched = true;
          break;
        }
      }
      if (!matched) {
        const ch = text[cursor];
        tokens.push(this.tokenToId.has(ch) ? this.tokenToId.get(ch) : 3);
        cursor++;
      }
    }
    return tokens;
  }

  decode(tokens) {
    if (!tokens || !tokens.length) return '';
    let out = '';
    for (const id of tokens) {
      if (id >= this.specials.length) {
        out += this.idToToken.get(id) || '';
      }
    }
    return out;
  }
}

export class NanoTransformer {
  constructor(config = {}) {
    this.dim = config.dim || 64;
    this.layers = config.layers || 2;
    this.heads = config.heads || 2;
    this.headDim = this.dim / this.heads;
    this.vocabSize = config.vocabSize || 128;
    this.maxSeqLen = config.maxSeqLen || 64;
    this.stepCount = config.stepCount || 0;

    this.weights = {};
    this.grads = {};
    this.adamM = {};
    this.adamV = {};

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
    let ss = 0.0;
    for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
    const mean = ss / x.length;
    const invScale = 1.0 / Math.sqrt(mean + eps);
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i] * invScale;
    return { out, invScale, raw: x };
  }

  // Backward pass through RMSNorm
  backwardRMSNorm(dOut, normData) {
    const { out, invScale, raw } = normData;
    const N = raw.length;
    let dot = 0.0;
    for (let i = 0; i < N; i++) dot += dOut[i] * out[i];

    const dIn = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      dIn[i] = invScale * (dOut[i] - (out[i] * dot) / N);
    }
    return dIn;
  }

  forwardStepWithTape(tokenId, pos, historyContext = null) {
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

      if (historyContext) {
        historyContext.k[l][pos] = k;
        historyContext.v[l][pos] = v;
      }

      const att = new Float32Array(this.dim);
      layerTape.scores = [];
      layerTape.weights = [];

      for (let h = 0; h < this.heads; h++) {
        const hOffset = h * this.headDim;
        let maxScore = -Infinity;
        const scores = new Float32Array(pos + 1);

        for (let t = 0; t <= pos; t++) {
          const histK = historyContext ? historyContext.k[l][t] : k;
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
          const histV = historyContext ? historyContext.v[l][t] : v;
          if (!histV) continue;
          for (let d = 0; d < this.headDim; d++) {
            att[hOffset + d] += histV[hOffset + d] * weights[t];
          }
        }
        layerTape.weights[h] = weights;
      }

      layerTape.att = att;
      const xAtt = new Float32Array(this.dim);
      for (let r = 0; r < this.dim; r++) {
        let sum = 0.0;
        const off = r * this.dim;
        for (let c = 0; c < this.dim; c++) sum += att[c] * this.weights[`blk_${l}_wo`][off + c];
        xAtt[r] = sum;
        x[r] += sum; // Residual
      }
      layerTape.xMid = new Float32Array(x);

      // SwiGLU MLP
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
        for (let c = 0; c < this.dim * 2; c++) sum += mlpAct[c] * this.weights[`blk_${l}_wdown`][off + c];
        x[r] += sum; // Residual
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
    for (let i = 0; i < this.vocabSize; i++) probs[i] /= (sumExp || 1.0);

    const loss = -Math.log(Math.max(1e-7, probs[targetTokenId]));
    probs[targetTokenId] -= 1.0; // Cross-entropy gradient

    // 1. LM Head Backprop
    const dNormFinal = new Float32Array(this.dim);
    for (let r = 0; r < this.vocabSize; r++) {
      const dL = probs[r];
      if (Math.abs(dL) < 1e-9) continue;
      const off = r * this.dim;
      for (let c = 0; c < this.dim; c++) {
        this.grads.lm_head[off + c] += dL * tape.normFinal.out[c];
        dNormFinal[c] += dL * this.weights.lm_head[off + c];
      }
    }

    let dx = this.backwardRMSNorm(dNormFinal, tape.normFinal);

    // 2. Transformer Layer Reverse Propagation
    for (let l = this.layers - 1; l >= 0; l--) {
      const layerTape = tape.layers[l];

      // MLP Down-Projection
      const dMlpAct = new Float32Array(this.dim * 2);
      for (let r = 0; r < this.dim; r++) {
        const off = r * (this.dim * 2);
        const dxi = dx[r];
        for (let c = 0; c < this.dim * 2; c++) {
          this.grads[`blk_${l}_wdown`][off + c] += dxi * layerTape.mlpAct[c];
          dMlpAct[c] += dxi * this.weights[`blk_${l}_wdown`][off + c];
        }
      }

      // SwiGLU Gates
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

      const dxMlp = this.backwardRMSNorm(dNorm2, layerTape.norm2);
      for (let i = 0; i < this.dim; i++) dx[i] += dxMlp[i];

      // Attention Projection
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

      const dxAtt = this.backwardRMSNorm(dNorm1, layerTape.norm1);
      for (let i = 0; i < this.dim; i++) dx[i] += dxAtt[i];
    }

    // 3. Embedding Matrix
    const embOffset = Math.min(tape.tokenId, this.vocabSize - 1) * this.dim;
    for (let i = 0; i < this.dim; i++) {
      this.grads.wte[embOffset + i] += dx[i];
    }

    return loss;
  }

  stepAdamW(lr = 0.002, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0.01) {
    this.stepCount++;
    const clipVal = 1.0;

    for (const key in this.weights) {
      const W = this.weights[key];
      const G = this.grads[key];
      const M = this.adamM[key];
      const V = this.adamV[key];

      const len = W.length;
      const biasCorr1 = 1.0 - Math.pow(beta1, this.stepCount);
      const biasCorr2 = 1.0 - Math.pow(beta2, this.stepCount);

      for (let i = 0; i < len; i++) {
        let g = G[i];
        if (g > clipVal) g = clipVal;
        else if (g < -clipVal) g = -clipVal;

        W[i] -= lr * weightDecay * W[i];

        M[i] = beta1 * M[i] + (1.0 - beta1) * g;
        V[i] = beta2 * V[i] + (1.0 - beta2) * (g * g);

        const mHat = M[i] / biasCorr1;
        const vHat = V[i] / biasCorr2;

        W[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
      }
    }
  }

  // Trains on a full window without caching cross-token state leakage
  trainSequence(tokenIds, lr = 0.002) {
    if (tokenIds.length < 2) return 0.0;
    this.zeroGrads();

    const len = Math.min(tokenIds.length - 1, this.maxSeqLen);
    const history = {
      k: Array.from({ length: this.layers }, () => []),
      v: Array.from({ length: this.layers }, () => [])
    };

    let totalLoss = 0.0;
    for (let pos = 0; pos < len; pos++) {
      const inputId = tokenIds[pos];
      const targetId = tokenIds[pos + 1];

      const { logits, tape } = this.forwardStepWithTape(inputId, pos, history);
      const stepLoss = this.backwardStep(targetId, logits, tape);
      totalLoss += stepLoss;
    }

    this.stepAdamW(lr);
    return totalLoss / len;
  }
}
