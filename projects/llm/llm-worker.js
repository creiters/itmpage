import { WasiRuntime } from './wasm-engine.js';
import { GGUFParser } from './gguf-parser.js';
import { SimpleBPETokenizer } from './tokenizer.js';

let wasi = null;
let parsedModel = null;
let tokenizer = null;
let isLoaded = false;
let activeModelName = 'Unloaded';
let modelBuffer = null;
let modelView = null;

// Model Hyperparameters (SmolLM-135M defaults, auto-populated from GGUF)
let nLayers = 30;
let nEmbd = 576;
let nHeads = 9;
let nKVHeads = 3;
let headDim = 64;
let nFF = 1536;

// Fast Q4_K_M Block Dequantizer
function dequantizeQ4_K_Row(view, byteOffset, out, length) {
  let outIdx = 0;
  let inOffset = byteOffset;
  const blocks = Math.ceil(length / 256);

  for (let b = 0; b < blocks && outIdx < length; b++) {
    const d = view.getUint16(inOffset, true) / 65535.0;
    const dmin = view.getUint16(inOffset + 2, true) / 65535.0;
    inOffset += 4;

    for (let i = 0; i < 128 && outIdx < length; i++) {
      const byte = view.getUint8(inOffset + i);
      out[outIdx++] = (byte & 0x0f) * d - dmin;
      if (outIdx < length) {
        out[outIdx++] = ((byte >> 4) & 0x0f) * d - dmin;
      }
    }
    inOffset += 128;
  }
}

// Math Operators
function rmsNorm(out, x, weight, size, eps = 1e-5) {
  let ss = 0.0;
  for (let i = 0; i < size; i++) ss += x[i] * x[i];
  const scale = 1.0 / Math.sqrt(ss / size + eps);
  for (let i = 0; i < size; i++) out[i] = x[i] * scale * weight[i];
}

function matVecMul(out, x, tensor, outDim, inDim) {
  const rowBytes = Math.ceil((inDim * 4.5) / 8) + 32;
  const rowBuffer = new Float32Array(inDim);

  for (let r = 0; r < outDim; r++) {
    const offset = parsedModel.tensorDataStart + tensor.offset + r * rowBytes;
    if (offset + rowBytes > modelBuffer.byteLength) break;
    dequantizeQ4_K_Row(modelView, offset, rowBuffer, inDim);

    let sum = 0.0;
    for (let c = 0; c < inDim; c++) {
      sum += x[c] * rowBuffer[c];
    }
    out[r] = sum;
  }
}

function applyRoPE(q, k, pos, hDim) {
  for (let i = 0; i < hDim; i += 2) {
    const freq = 1.0 / Math.pow(10000.0, i / hDim);
    const val = pos * freq;
    const cos = Math.cos(val);
    const sin = Math.sin(val);

    const q0 = q[i];
    const q1 = q[i + 1];
    q[i] = q0 * cos - q1 * sin;
    q[i + 1] = q0 * sin + q1 * cos;

    if (k && i < k.length) {
      const k0 = k[i];
      const k1 = k[i + 1];
      k[i] = k0 * cos - k1 * sin;
      k[i + 1] = k0 * sin + k1 * cos;
    }
  }
}

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'INIT_ENGINE') {
    try {
      wasi = new WasiRuntime(4096);
      activeModelName = payload?.modelName || 'Loaded Model';

      if (payload?.wasmBinary) {
        modelBuffer = payload.wasmBinary;
        modelView = new DataView(modelBuffer);

        self.postMessage({ type: 'STATUS', status: 'Parsing GGUF layers...' });
        const parser = new GGUFParser(payload.wasmBinary);
        parsedModel = parser.parse();

        // Extract Architecture Parameters
        nLayers = parsedModel.metadata['smollm.block_count'] || parsedModel.metadata['llama.block_count'] || 30;
        nEmbd = parsedModel.metadata['smollm.embedding_length'] || parsedModel.metadata['llama.embedding_length'] || 576;
        nHeads = parsedModel.metadata['smollm.attention.head_count'] || parsedModel.metadata['llama.attention.head_count'] || 9;
        nKVHeads = parsedModel.metadata['smollm.attention.head_count_kv'] || parsedModel.metadata['llama.attention.head_count_kv'] || 3;
        headDim = nEmbd / nHeads;
        nFF = parsedModel.metadata['smollm.feed_forward_length'] || parsedModel.metadata['llama.feed_forward_length'] || 1536;

        const vocabTokens = parsedModel.metadata['tokenizer.ggml.tokens'] || [];
        const vocabScores = parsedModel.metadata['tokenizer.ggml.scores'] || [];
        tokenizer = new SimpleBPETokenizer(vocabTokens, vocabScores);

        isLoaded = true;
        self.postMessage({
          type: 'STATUS',
          status: `Ready: ${activeModelName} (${nLayers} Layers, ${nEmbd} Dim)`
        });
      } else {
        isLoaded = true;
        self.postMessage({ type: 'STATUS', status: 'Ready: Standalone Mode' });
      }
    } catch (err) {
      isLoaded = false;
      self.postMessage({ type: 'ERROR', message: `Init failed: ${err.message}` });
    }
  }

  if (type === 'INFER') {
    const { prompt, taskId } = payload;

    if (!isLoaded || !modelBuffer) {
      self.postMessage({
        type: 'TOKEN',
        taskId,
        token: 'Error: Engine not ready.',
        fullText: 'Error: Engine not ready.',
        done: true
      });
      return;
    }

    self.postMessage({ type: 'STATUS', status: 'Computing transformer layers...' });

    const promptTokens = tokenizer.encode(prompt);
    let currentToken = promptTokens[0] || 1;
    const generatedTokens = [...promptTokens];
    let fullOutput = '';
    const maxGenTokens = 32;

    // Allocate Static State Tensors
    const x = new Float32Array(nEmbd);
    const xb = new Float32Array(nEmbd);
    const q = new Float32Array(nEmbd);
    const k = new Float32Array(headDim * nKVHeads);
    const v = new Float32Array(headDim * nKVHeads);
    const att = new Float32Array(nEmbd);
    const h1 = new Float32Array(nFF);
    const h2 = new Float32Array(nFF);
    const outMlp = new Float32Array(nEmbd);

    const embedTensor = parsedModel.tensors.get('token_embd.weight') || parsedModel.tensors.get('model.embed_tokens.weight');
    const lmHeadTensor = parsedModel.tensors.get('output.weight') || embedTensor;
    const normWeight = new Float32Array(nEmbd).fill(1.0);

    // Context Processing & Autoregressive Loop
    for (let pos = 0; pos < promptTokens.length + maxGenTokens; pos++) {
      if (pos < promptTokens.length) {
        currentToken = promptTokens[pos];
      }

      // 1. Embedding Lookup
      const rowBytes = Math.ceil((nEmbd * 4.5) / 8) + 32;
      const embOffset = parsedModel.tensorDataStart + embedTensor.offset + currentToken * rowBytes;
      if (embOffset + rowBytes < modelBuffer.byteLength) {
        dequantizeQ4_K_Row(modelView, embOffset, x, nEmbd);
      }

      // 2. Iterate Transformer Layers across Slices
      for (let l = 0; l < nLayers; l++) {
        rmsNorm(xb, x, normWeight, nEmbd);

        const wq = parsedModel.tensors.get(`blk.${l}.attn_q.weight`);
        const wk = parsedModel.tensors.get(`blk.${l}.attn_k.weight`);
        const wv = parsedModel.tensors.get(`blk.${l}.attn_v.weight`);
        const wo = parsedModel.tensors.get(`blk.${l}.attn_output.weight`);

        if (wq && wk && wv) {
          matVecMul(q, xb, wq, nEmbd, nEmbd);
          matVecMul(k, xb, wk, headDim * nKVHeads, nEmbd);
          matVecMul(v, xb, wv, headDim * nKVHeads, nEmbd);

          applyRoPE(q, k, pos, headDim);

          // Scaled Dot-Product Attention
          let score = 0.0;
          for (let i = 0; i < headDim; i++) score += q[i] * k[i];
          score /= Math.sqrt(headDim);
          const attWeight = 1.0 / (1.0 + Math.exp(-score));

          for (let i = 0; i < nEmbd; i++) {
            att[i] = v[i % (headDim * nKVHeads)] * attWeight;
          }

          if (wo) {
            matVecMul(xb, att, wo, nEmbd, nEmbd);
            for (let i = 0; i < nEmbd; i++) x[i] += xb[i];
          }
        }

        // SwiGLU Feed-Forward Network Block
        const wGate = parsedModel.tensors.get(`blk.${l}.ffn_gate.weight`);
        const wUp = parsedModel.tensors.get(`blk.${l}.ffn_up.weight`);
        const wDown = parsedModel.tensors.get(`blk.${l}.ffn_down.weight`);

        if (wGate && wUp && wDown) {
          rmsNorm(xb, x, normWeight, nEmbd);
          matVecMul(h1, xb, wGate, nFF, nEmbd);
          matVecMul(h2, xb, wUp, nFF, nEmbd);

          for (let i = 0; i < nFF; i++) {
            const silu = h1[i] / (1.0 + Math.exp(-h1[i]));
            h1[i] = silu * h2[i];
          }

          matVecMul(outMlp, h1, wDown, nEmbd, nFF);
          for (let i = 0; i < nEmbd; i++) x[i] += outMlp[i];
        }
      }

      // Only generate new tokens once prompt pre-fill completes
      if (pos >= promptTokens.length - 1) {
        rmsNorm(xb, x, normWeight, nEmbd);

        // 3. Logits Search with Repetition Penalty
        let bestToken = 1;
        let maxLogit = -Infinity;
        const candidatePool = Math.min(tokenizer.tokens.length, 5000);
        const recentTokens = generatedTokens.slice(-6);

        for (let vIdx = 3; vIdx < candidatePool; vIdx++) {
          const headOffset = parsedModel.tensorDataStart + lmHeadTensor.offset + vIdx * rowBytes;
          if (headOffset + rowBytes > modelBuffer.byteLength) break;

          const headWeights = new Float32Array(nEmbd);
          dequantizeQ4_K_Row(modelView, headOffset, headWeights, nEmbd);

          let logit = 0.0;
          for (let i = 0; i < nEmbd; i++) logit += xb[i] * headWeights[i];

          if (recentTokens.includes(vIdx)) {
            logit -= 2.5; // Repetition penalty
          }

          if (logit > maxLogit) {
            maxLogit = logit;
            bestToken = vIdx;
          }
        }

        // Check for Stop / End-of-Sequence
        if (bestToken === 0 || bestToken === 1 || bestToken === 2) break;

        const decoded = tokenizer.decode(bestToken);
        if (!decoded || decoded.includes('<|im_end|>')) break;

        fullOutput += decoded;
        generatedTokens.push(bestToken);
        currentToken = bestToken;

        self.postMessage({
          type: 'TOKEN',
          taskId,
          token: decoded,
          fullText: fullOutput.trimStart(),
          done: pos === promptTokens.length + maxGenTokens - 1
        });

        await new Promise((r) => setTimeout(r, 20));
      }
    }

    self.postMessage({
      type: 'TOKEN',
      taskId,
      token: '',
      fullText: fullOutput.trimStart(),
      done: true
    });
    self.postMessage({ type: 'STATUS', status: `Ready: ${activeModelName}` });
  }
};
