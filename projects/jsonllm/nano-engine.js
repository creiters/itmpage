export class NanoTransformer {
  constructor(config = {}) {
    this.dim = config.dim || 64;
    this.layers = config.layers || 2;
    this.heads = config.heads || 2;
    this.headDim = this.dim / this.heads;
    this.vocabSize = config.vocabSize || 96;
    this.maxSeqLen = config.maxSeqLen || 48;

    this.weights = {};
    this.m = {};
    this.v = {};
    this.step = 0;
    this.initWeights();
  }

  initWeights() {
    const scale = 1.0 / Math.sqrt(this.dim);
    const rand = (size) => {
      const arr = new Float32Array(size);
      for (let i = 0; i < size; i++) arr[i] = (Math.random() * 2 - 1) * scale;
      return arr;
    };

    this.weights['wte'] = rand(this.vocabSize * this.dim);
    this.weights['lm_head'] = rand(this.vocabSize * this.dim);

    for (let l = 0; l < this.layers; l++) {
      this.weights[`l${l}_wq`] = rand(this.dim * this.dim);
      this.weights[`l${l}_wk`] = rand(this.dim * this.dim);
      this.weights[`l${l}_wv`] = rand(this.dim * this.dim);
      this.weights[`l${l}_wo`] = rand(this.dim * this.dim);
      this.weights[`l${l}_wgate`] = rand(this.dim * (this.dim * 2));
      this.weights[`l${l}_wup`] = rand(this.dim * (this.dim * 2));
      this.weights[`l${l}_wdown`] = rand((this.dim * 2) * this.dim);
    }

    for (const k of Object.keys(this.weights)) {
      this.m[k] = new Float32Array(this.weights[k].length);
      this.v[k] = new Float32Array(this.weights[k].length);
    }
  }

  swapWeights(config, weights) {
    this.dim = config.dim;
    this.layers = config.layers;
    this.heads = config.heads;
    this.vocabSize = config.vocabSize;
    this.maxSeqLen = config.maxSeqLen;
    this.weights = weights;
  }

  rmsNorm(out, x, size, eps = 1e-5) {
    let ss = 0.0;
    for (let i = 0; i < size; i++) ss += x[i] * x[i];
    const s = 1.0 / Math.sqrt(ss / size + eps);
    for (let i = 0; i < size; i++) out[i] = x[i] * s;
  }

  matVec(out, x, W, rows, cols) {
    for (let r = 0; r < rows; r++) {
      let sum = 0.0;
      const off = r * cols;
      for (let c = 0; c < cols; c++) sum += x[c] * W[off + c];
      out[r] = sum;
    }
  }

  forward(tokens) {
    const lastTok = tokens[tokens.length - 1];
    const x = new Float32Array(this.dim);
    const embOff = lastTok * this.dim;
    for (let i = 0; i < this.dim; i++) x[i] = this.weights['wte'][embOff + i];

    const xb = new Float32Array(this.dim);
    const q = new Float32Array(this.dim);
    const k = new Float32Array(this.dim);
    const v = new Float32Array(this.dim);
    const att = new Float32Array(this.dim);
    const h1 = new Float32Array(this.dim * 2);
    const h2 = new Float32Array(this.dim * 2);
    const outFF = new Float32Array(this.dim);

    for (let l = 0; l < this.layers; l++) {
      this.rmsNorm(xb, x, this.dim);
      this.matVec(q, xb, this.weights[`l${l}_wq`], this.dim, this.dim);
      this.matVec(k, xb, this.weights[`l${l}_wk`], this.dim, this.dim);
      this.matVec(v, xb, this.weights[`l${l}_wv`], this.dim, this.dim);

      let score = 0.0;
      for (let i = 0; i < this.dim; i++) score += q[i] * k[i];
      score = 1.0 / (1.0 + Math.exp(-score / Math.sqrt(this.dim)));

      for (let i = 0; i < this.dim; i++) att[i] = v[i] * score;
      this.matVec(xb, att, this.weights[`l${l}_wo`], this.dim, this.dim);
      for (let i = 0; i < this.dim; i++) x[i] += xb[i];

      // SwiGLU MLP
      this.rmsNorm(xb, x, this.dim);
      this.matVec(h1, xb, this.weights[`l${l}_wgate`], this.dim * 2, this.dim);
      this.matVec(h2, xb, this.weights[`l${l}_wup`], this.dim * 2, this.dim);

      const mlpAct = new Float32Array(this.dim * 2);
      for (let i = 0; i < this.dim * 2; i++) {
        const silu = h1[i] / (1.0 + Math.exp(-h1[i]));
        mlpAct[i] = silu * h2[i];
      }
      this.matVec(outFF, mlpAct, this.weights[`l${l}_wdown`], this.dim, this.dim * 2);
      for (let i = 0; i < this.dim; i++) x[i] += outFF[i];
    }

    const logits = new Float32Array(this.vocabSize);
    this.rmsNorm(xb, x, this.dim);
    this.matVec(logits, xb, this.weights['lm_head'], this.vocabSize, this.dim);
    return logits;
  }

  trainStep(tokenIds, lr = 0.003) {
    if (tokenIds.length < 2) return 0;
    this.step++;
    let totalLoss = 0;

    for (let i = 0; i < tokenIds.length - 1; i++) {
      const inputSeq = tokenIds.slice(0, i + 1);
      const target = tokenIds[i + 1];
      const logits = this.forward(inputSeq);

      let maxL = -Infinity;
      for (let v = 0; v < this.vocabSize; v++) if (logits[v] > maxL) maxL = logits[v];
      let sumExp = 0;
      for (let v = 0; v < this.vocabSize; v++) sumExp += Math.exp(logits[v] - maxL);
      const prob = Math.exp(logits[target] - maxL) / sumExp;

      totalLoss += -Math.log(Math.max(prob, 1e-12));

      // AdamW Embedding Optimization
      const grad = prob - 1.0;
      const embOff = tokenIds[i] * this.dim;
      const b1 = 0.9, b2 = 0.999, eps = 1e-8;

      for (let d = 0; d < this.dim; d++) {
        const g = grad * 0.05;
        const idx = embOff + d;
        this.m['wte'][idx] = b1 * this.m['wte'][idx] + (1 - b1) * g;
        this.v['wte'][idx] = b2 * this.v['wte'][idx] + (1 - b2) * (g * g);

        const mHat = this.m['wte'][idx] / (1 - Math.pow(b1, this.step));
        const vHat = this.v['wte'][idx] / (1 - Math.pow(b2, this.step));
        this.weights['wte'][idx] -= lr * (mHat / (Math.sqrt(vHat) + eps));
      }
    }
    return totalLoss / (tokenIds.length - 1);
  }
}
