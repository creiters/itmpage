/**
 * nano-engine.js - 4M Parameter Transformer with Hardware WASM SIMD Acceleration
 */

import { HalfFloatArray } from './jsonllm-codec.js';
import { WasiRuntime } from './wasm-engine.js';

export class NanoTokenizer {
  constructor(vocabSize = 1024) {
    this.vocabSize = vocabSize;
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

    const foundational = [
      "Hello", "Hi", "Hey", "How", "are", "you", "I", "am", "doing", "well",
      "great", "fine", "good", "What", "is", "your", "name", "My", "vimtoLLM",
      "Miriam", "wife", "partner", "woman", "love", "darling", "think", "help",
      "together", "an", "offline", "browser", "AI", "assistant", "Yes", "No",
      "code", "HTML", "C", "space", "inside", "minus", "84", "10", "WASM", "SIMD",
      ".", ",", "!", "?", ":", ";", "'", "\"", "\n", "-", "(", ")"
    ];

    let id = this.specials.length;
    for (const w of foundational) {
      if (!this.tokenToId.has(w)) {
        this.tokenToId.set(w, id);
        this.idToToken.set(id, w);
        id++;
      }
    }
  }

  encode(text) {
    if (!text) return [1];
    const tokens = [1];
    const regex = /(<user>|<bot>|<eos>|<bos>|[a-zA-Z0-9_-]+|[.,!?:;'"()\n])/g;
    const matches = text.match(regex) || [];

    for (const piece of matches) {
      if (this.tokenToId.has(piece)) {
        tokens.push(this.tokenToId.get(piece));
      } else {
        const id = this.tokenToId.size;
        this.tokenToId.set(piece, id);
        this.idToToken.set(id, piece);
        tokens.push(id);
      }
    }
    return tokens;
  }

  decode(tokens) {
    if (!tokens || !tokens.length) return '';
    return tokens
      .filter((id) => id >= this.specials.length)
      .map((id) => this.idToToken.get(id) || '')
      .join(' ')
      .replace(/ ([.,!?:;'")])/g, '$1');
  }
}

export class NanoTransformer {
  constructor(config = {}) {
    this.dim = config.dim || 192;
    this.layers = config.layers || 8;
    this.heads = config.heads || 6;
    this.headDim = this.dim / this.heads; // 32
    this.ffnDim = config.ffnDim || 512;
    this.vocabSize = config.vocabSize || 1024;
    this.maxSeqLen = config.maxSeqLen || 128;
    this.stepCount = config.stepCount || 0;

    this.tokenizer = new NanoTokenizer(this.vocabSize);
    this.weights = {};
    this.grads = {};
    this.adamM = {};
    this.adamV = {};

    this.wasi = new WasiRuntime(256, 2048);
    this.hasWasm = false;

    this.allocateScratchpads();
    this.initWeights();
  }

  async initWasm(wasmBinary) {
    try {
      await this.wasi.compileAndInstantiate(wasmBinary, console.log);
      this.hasWasm = !!this.wasi.exports;
      return this.hasWasm;
    } catch {
      this.hasWasm = false;
      return false;
    }
  }

  allocateScratchpads() {
    this.x = new HalfFloatArray(this.dim);
    this.xb = new HalfFloatArray(this.dim);
    this.q = new HalfFloatArray(this.dim);
    this.k = new HalfFloatArray(this.dim);
    this.v = new HalfFloatArray(this.dim);
    this.att = new HalfFloatArray(this.dim);
    this.h1 = new HalfFloatArray(this.ffnDim);
    this.h2 = new HalfFloatArray(this.ffnDim);
    this.siluH1 = new HalfFloatArray(this.ffnDim);
    this.mlpAct = new HalfFloatArray(this.ffnDim);
    this.logits = new HalfFloatArray(this.vocabSize);
  }

  initWeights() {
    const scale = Math.sqrt(2.0 / this.dim);
    const randFP16 = (size) => {
      const arr = new HalfFloatArray(size);
      for (let i = 0; i < size; i++) arr[i] = (Math.random() - 0.5) * 2 * scale;
      return arr;
    };

    const register = (name, size) => {
      this.weights[name] = randFP16(size);
      this.grads[name] = new HalfFloatArray(size);
      this.adamM[name] = new HalfFloatArray(size);
      this.adamV[name] = new HalfFloatArray(size);
    };

    register('wte', this.vocabSize * this.dim);
    register('wpe', this.maxSeqLen * this.dim);
    register('lm_head', this.vocabSize * this.dim);

    for (let l = 0; l < this.layers; l++) {
      register(`blk_${l}_wq`, this.dim * this.dim);
      register(`blk_${l}_wk`, this.dim * this.dim);
      register(`blk_${l}_wv`, this.dim * this.dim);
      register(`blk_${l}_wo`, this.dim * this.dim);
      register(`blk_${l}_wgate`, this.dim * this.ffnDim);
      register(`blk_${l}_wup`, this.dim * this.ffnDim);
      register(`blk_${l}_wdown`, this.ffnDim * this.dim);
    }
  }

  zeroGrads() {
    for (const key in this.grads) {
      this.grads[key].fill(0.0);
    }
  }

  rmsNorm(out, src, eps = 1e-5) {
    let ss = 0.0;
    for (let i = 0; i < src.length; i++) ss += src[i] * src[i];
    const invScale = 1.0 / Math.sqrt(ss / src.length + eps);
    for (let i = 0; i < src.length; i++) out[i] = src[i] * invScale;
    return invScale;
  }

  forwardStepWithTape(tokenId, pos, historyContext = null) {
    const tape = { pos, tokenId, layers: [] };
    const safeTok = Math.max(0, Math.min(tokenId, this.vocabSize - 1));
    const embOffset = safeTok * this.dim;
    const posOffset = Math.min(pos, this.maxSeqLen - 1) * this.dim;

    for (let i = 0; i < this.dim; i++) {
      this.x[i] = this.weights.wte[embOffset + i] + this.weights.wpe[posOffset + i];
    }
    tape.xInit = new HalfFloatArray(this.x);

    for (let l = 0; l < this.layers; l++) {
      const layerTape = {};
      const invScale1 = this.rmsNorm(this.xb, this.x);
      layerTape.norm1 = { out: new HalfFloatArray(this.xb), invScale: invScale1 };

      // Matrix Projections: Q, K, V
      for (let r = 0; r < this.dim; r++) {
        let sq = 0.0, sk = 0.0, sv = 0.0;
        const off = r * this.dim;
        for (let c = 0; c < this.dim; c++) {
          sq += this.xb[c] * this.weights[`blk_${l}_wq`][off + c];
          sk += this.xb[c] * this.weights[`blk_${l}_wk`][off + c];
          sv += this.xb[c] * this.weights[`blk_${l}_wv`][off + c];
        }
        this.q[r] = sq;
        this.k[r] = sk;
        this.v[r] = sv;
      }

      layerTape.q = new HalfFloatArray(this.q);
      layerTape.k = new HalfFloatArray(this.k);
      layerTape.v = new HalfFloatArray(this.v);

      if (historyContext) {
        historyContext.k[l][pos] = new HalfFloatArray(this.k);
        historyContext.v[l][pos] = new HalfFloatArray(this.v);
      }

      this.att.fill(0.0);
      for (let h = 0; h < this.heads; h++) {
        const hOffset = h * this.headDim;
        let maxScore = -Infinity;
        const scores = new HalfFloatArray(pos + 1);

        for (let t = 0; t <= pos; t++) {
          const histK = historyContext ? historyContext.k[l][t] : this.k;
          if (!histK) continue;
          let score = 0.0;
          for (let d = 0; d < this.headDim; d++) {
            score += this.q[hOffset + d] * histK[hOffset + d];
          }
          score /= Math.sqrt(this.headDim);
          scores[t] = score;
          if (score > maxScore) maxScore = score;
        }

        let sumExp = 0.0;
        const weights = new HalfFloatArray(pos + 1);
        for (let t = 0; t <= pos; t++) {
          weights[t] = Math.exp(scores[t] - maxScore);
          sumExp += weights[t];
        }
        for (let t = 0; t <= pos; t++) {
          weights[t] /= (sumExp || 1.0);
          const histV = historyContext ? historyContext.v[l][t] : this.v;
          if (!histV) continue;
          for (let d = 0; d < this.headDim; d++) {
            this.att[hOffset + d] += histV[hOffset + d] * weights[t];
          }
        }
      }

      layerTape.att = new HalfFloatArray(this.att);
      for (let r = 0; r < this.dim; r++) {
        let sum = 0.0;
        const off = r * this.dim;
        for (let c = 0; c < this.dim; c++) sum += this.att[c] * this.weights[`blk_${l}_wo`][off + c];
        this.x[r] += sum;
      }

      // SwiGLU Gated Feed-Forward Block
      const invScale2 = this.rmsNorm(this.xb, this.x);
      layerTape.norm2 = { out: new HalfFloatArray(this.xb), invScale: invScale2 };

      for (let r = 0; r < this.ffnDim; r++) {
        let sGate = 0.0, sUp = 0.0;
        const off = r * this.dim;
        for (let c = 0; c < this.dim; c++) {
          sGate += this.xb[c] * this.weights[`blk_${l}_wgate`][off + c];
          sUp += this.xb[c] * this.weights[`blk_${l}_wup`][off + c];
        }
        this.h1[r] = sGate;
        this.h2[r] = sUp;
        this.siluH1[r] = sGate / (1.0 + Math.exp(-sGate));
        this.mlpAct[r] = this.siluH1[r] * sUp;
      }

      layerTape.h1 = new HalfFloatArray(this.h1);
      layerTape.h2 = new HalfFloatArray(this.h2);
      layerTape.siluH1 = new HalfFloatArray(this.siluH1);
      layerTape.mlpAct = new HalfFloatArray(this.mlpAct);

      for (let r = 0; r < this.dim; r++) {
        let sum = 0.0;
        const off = r * this.ffnDim;
        for (let c = 0; c < this.ffnDim; c++) sum += this.mlpAct[c] * this.weights[`blk_${l}_wdown`][off + c];
        this.x[r] += sum;
      }

      tape.layers.push(layerTape);
    }

    const invScaleFinal = this.rmsNorm(this.xb, this.x);
    tape.normFinal = { out: new HalfFloatArray(this.xb), invScale: invScaleFinal };

    for (let r = 0; r < this.vocabSize; r++) {
      let sum = 0.0;
      const off = r * this.dim;
      for (let c = 0; c < this.dim; c++) sum += this.xb[c] * this.weights.lm_head[off + c];
      this.logits[r] = sum;
    }

    return { logits: this.logits, tape };
  }

  backwardStep(targetTokenId, logits, tape) {
    let maxLogit = -Infinity;
    for (let i = 0; i < this.vocabSize; i++) if (logits[i] > maxLogit) maxLogit = logits[i];

    let sumExp = 0.0;
    const probs = new HalfFloatArray(this.vocabSize);
    for (let i = 0; i < this.vocabSize; i++) {
      probs[i] = Math.exp(logits[i] - maxLogit);
      sumExp += probs[i];
    }
    for (let i = 0; i < this.vocabSize; i++) probs[i] /= (sumExp || 1.0);

    const safeTarget = Math.min(targetTokenId, this.vocabSize - 1);
    const loss = -Math.log(Math.max(1e-7, probs[safeTarget]));
    probs[safeTarget] -= 1.0;

    // Output Head Backprop
    const dNormFinal = new HalfFloatArray(this.dim);
    for (let r = 0; r < this.vocabSize; r++) {
      const dL = probs[r];
      if (Math.abs(dL) < 1e-8) continue;
      const off = r * this.dim;
      for (let c = 0; c < this.dim; c++) {
        this.grads.lm_head[off + c] += dL * tape.normFinal.out[c];
        dNormFinal[c] += dL * this.weights.lm_head[off + c];
      }
    }

    const dx = new HalfFloatArray(this.dim);
    for (let i = 0; i < this.dim; i++) dx[i] = dNormFinal[i] * tape.normFinal.invScale;

    // Transformer Blocks Reverse Propagation
    for (let l = this.layers - 1; l >= 0; l--) {
      const layerTape = tape.layers[l];

      const dMlpAct = new HalfFloatArray(this.ffnDim);
      for (let r = 0; r < this.dim; r++) {
        const off = r * this.ffnDim;
        const dxi = dx[r];
        for (let c = 0; c < this.ffnDim; c++) {
          this.grads[`blk_${l}_wdown`][off + c] += dxi * layerTape.mlpAct[c];
          dMlpAct[c] += dxi * this.weights[`blk_${l}_wdown`][off + c];
        }
      }

      const dNorm2 = new HalfFloatArray(this.dim);
      for (let r = 0; r < this.ffnDim; r++) {
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

      const dAtt = new HalfFloatArray(this.dim);
      for (let r = 0; r < this.dim; r++) {
        const off = r * this.dim;
        const dxi = dx[r];
        for (let c = 0; c < this.dim; c++) {
          this.grads[`blk_${l}_wo`][off + c] += dxi * layerTape.att[c];
          dAtt[c] += dxi * this.weights[`blk_${l}_wo`][off + c];
        }
      }

      const dNorm1 = new HalfFloatArray(this.dim);
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

    const embOffset = Math.min(tape.tokenId, this.vocabSize - 1) * this.dim;
    const posOffset = Math.min(tape.pos, this.maxSeqLen - 1) * this.dim;
    for (let i = 0; i < this.dim; i++) {
      this.grads.wte[embOffset + i] += dx[i];
      this.grads.wpe[posOffset + i] += dx[i];
    }

    return loss;
  }

  trainSequence(tokenIds, lr = 0.003) {
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
      totalLoss += this.backwardStep(targetId, logits, tape);
    }

    this.stepAdamW(lr);
    return totalLoss / len;
  }

  stepAdamW(lr = 0.003, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0.01) {
    this.stepCount++;

    for (const key in this.weights) {
      const W = this.weights[key];
      const G = this.grads[key];
      const M = this.adamM[key];
      const V = this.adamV[key];

      // Dispatch to SIMD WASM Kernel when available
      if (this.hasWasm && W.length >= 1024) {
        this.wasi.resetHeap();
        const len = W.length;
        const wPtr = this.wasi.allocate(len);
        const gPtr = this.wasi.allocate(len);
        const mPtr = this.wasi.allocate(len);
        const vPtr = this.wasi.allocate(len);

        this.wasi.writeFloats(wPtr, W);
        this.wasi.writeFloats(gPtr, G);
        this.wasi.writeFloats(mPtr, M);
        this.wasi.writeFloats(vPtr, V);

        this.wasi.exports.wasm_adamw_step(wPtr, gPtr, mPtr, vPtr, len, lr, beta1, beta2, eps, weightDecay, this.stepCount);

        W.set(this.wasi.readFloats(wPtr, len));
        M.set(this.wasi.readFloats(mPtr, len));
        V.set(this.wasi.readFloats(vPtr, len));
        continue;
      }

      // High-precision typed fallback
      const len = W.length;
      const b1 = beta1;
      const b2 = beta2;
      const b1Pow = Math.pow(b1, this.stepCount);
      const b2Pow = Math.pow(b2, this.stepCount);
      const biasCorr1 = 1.0 - b1Pow;
      const biasCorr2 = 1.0 - b2Pow;

      for (let i = 0; i < len; i++) {
        let g = G[i];
        if (g > 1.0) g = 1.0;
        else if (g < -1.0) g = -1.0;

        W[i] -= lr * weightDecay * W[i];

        M[i] = b1 * M[i] + (1.0 - b1) * g;
        V[i] = b2 * V[i] + (1.0 - b2) * (g * g);

        const mHat = M[i] / biasCorr1;
        const vHat = V[i] / biasCorr2;

        W[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
      }
    }
  }
}
