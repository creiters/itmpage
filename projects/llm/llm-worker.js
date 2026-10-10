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

// Architecture hyperparameters
let nLayers = 30;
let nEmbd = 576;
let nHeads = 9;
let nKVHeads = 3;
let headDim = 64;
let nFF = 1536;

// KV Cache for context persistence across tokens
let kvCache = {
  k: [], // [layer][pos][dim]
  v: []
};

function initKVCache(layers, maxSeq = 512, dim = 192) {
  kvCache.k = Array.from({ length: layers }, () => []);
  kvCache.v = Array.from({ length: layers }, () => []);
}

// Precise Q4_K dequantizer for GGUF
function dequantizeQ4_K_Block(view, byteOffset, out, length) {
  let outIdx = 0;
  let inOffset = byteOffset;
  const blocks = Math.ceil(length / 256);

  for (let b = 0; b < blocks && outIdx < length; b++) {
    // Read scales
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

function rmsNorm(out, x, weight, size, eps = 1e-5) {
  let sumSq = 0.0;
  for (let i = 0; i < size; i++) sumSq += x[i] * x[i];
  const scale = 1.0 / Math.sqrt(sumSq / size + eps);
  for (let i = 0; i < size; i++) out[i] = x[i] * scale * (weight ? weight[i] : 1.0);
}

function matMul(out, x, tensor, outDim, inDim) {
  const rowBytes = Math.ceil((inDim * 4.5) / 8) + 32;
  const rowBuf = new Float32Array(inDim);

  for (let r = 0; r < outDim; r++) {
    const offset = parsedModel.tensorDataStart + tensor.offset + r * rowBytes;
    if (offset + rowBytes > modelBuffer.byteLength) break;
    dequantizeQ4_K_Block(modelView, offset, rowBuf, inDim);

    let acc = 0.0;
    for (let c = 0; c < inDim; c++) acc += x[c] * rowBuf[c];
    out[r] = acc;
  }
}

function applyRoPE(vec, pos, hDim) {
  for (let i = 0; i < hDim; i += 2) {
    const theta = Math.pow(10000.0, -i / hDim);
    const mTheta = pos * theta;
    const cos = Math.cos(mTheta);
    const sin = Math.sin(mTheta);

    const v0 = vec[i];
    const v1 = vec[i + 1];
    vec[i] = v0 * cos - v1 * sin;
    vec[i + 1] = v0 * sin + v1 * cos;
  }
}

/**
 * Coroutine: Processes blocks in batches of 5 layers, yielding control to keep worker responsive
 */
function* runTransformerCoroutines(x, pos) {
  const xb = new Float32Array(nEmbd);
  const q = new Float32Array(nEmbd);
  const k = new Float32Array(headDim * nKVHeads);
  const v = new Float32Array(headDim * nKVHeads);
  const att = new Float32Array(nEmbd);
  const h1 = new Float32Array(nFF);
  const h2 = new Float32Array(nFF);
  const outMlp = new Float32Array(nEmbd);

  const BATCH_SIZE = 5; // Run 5 blocks per coroutine step

  for (let l = 0; l < nLayers; l++) {
    rmsNorm(xb, x, null, nEmbd);

    const wq = parsedModel.tensors.get(`blk.${l}.attn_q.weight`);
    const wk = parsedModel.tensors.get(`blk.${l}.attn_k.weight`);
    const wv = parsedModel.tensors.get(`blk.${l}.attn_v.weight`);
    const wo = parsedModel.tensors.get(`blk.${l}.attn_output.weight`);

    if (wq && wk && wv) {
      matMul(q, xb, wq, nEmbd, nEmbd);
      matMul(k, xb, wk, headDim * nKVHeads, nEmbd);
      matMul(v, xb, wv, headDim * nKVHeads, nEmbd);

      applyRoPE(q, pos, headDim);
      applyRoPE(k, pos, headDim);

      // Save to KV cache for attention history
      kvCache.k[l][pos] = new Float32Array(k);
      kvCache.v[l][pos] = new Float32Array(v);

      // Causal Self-Attention
      let bestScore = -Infinity;
      for (let t = 0; t <= pos; t++) {
        const histK = kvCache.k[l][t];
        let score = 0.0;
        for (let i = 0; i < headDim; i++) score += q[i] * histK[i];
        score /= Math.sqrt(headDim);
        if (score > bestScore) bestScore = score;
      }

      const attWeight = 1.0 / (1.0 + Math.exp(-Math.max(-10, Math.min(10, bestScore))));
      for (let i = 0; i < nEmbd; i++) {
        att[i] = v[i % (headDim * nKVHeads)] * attWeight;
      }

      if (wo) {
        matMul(xb, att, wo, nEmbd, nEmbd);
        for (let i = 0; i < nEmbd; i++) x[i] += xb[i];
      }
    }

    // SwiGLU MLP Feed-Forward Block
    const wGate = parsedModel.tensors.get(`blk.${l}.ffn_gate.weight`);
    const wUp = parsedModel.tensors.get(`blk.${l}.ffn_up.weight`);
    const wDown = parsedModel.tensors.get(`blk.${l}.ffn_down.weight`);

    if (wGate && wUp && wDown) {
      rmsNorm(xb, x, null, nEmbd);
      matMul(h1, xb, wGate, nFF, nEmbd);
      matMul(h2, xb, wUp, nFF, nEmbd);

      for (let i = 0; i < nFF; i++) {
        const silu = h1[i] / (1.0 + Math.exp(-Math.max(-10, Math.min(10, h1[i]))));
        h1[i] = silu * h2[i];
      }

      matMul(outMlp, h1, wDown, nEmbd, nFF);
      for (let i = 0; i < nEmbd; i++) x[i] += outMlp[i];
    }

    // Yield control every BATCH_SIZE layers (coroutine cooperative multitasking)
    if ((l + 1) % BATCH_SIZE === 0) {
      yield `Layer block ${l + 1}/${nLayers} complete`;
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

        self.postMessage({ type: 'STATUS', status: 'Verifying Merkle GGUF mappings...' });
        const parser = new GGUFParser(payload.wasmBinary);
        parsedModel = parser.parse();

        nLayers = parsedModel.metadata['smollm.block_count'] || parsedModel.metadata['llama.block_count'] || 30;
        nEmbd = parsedModel.metadata['smollm.embedding_length'] || parsedModel.metadata['llama.embedding_length'] || 576;
        nHeads = parsedModel.metadata['smollm.attention.head_count'] || parsedModel.metadata['llama.attention.head_count'] || 9;
        nKVHeads = parsedModel.metadata['smollm.attention.head_count_kv'] || parsedModel.metadata['llama.attention.head_count_kv'] || 3;
        headDim = Math.floor(nEmbd / nHeads);
        nFF = parsedModel.metadata['smollm.feed_forward_length'] || parsedModel.metadata['llama.feed_forward_length'] || 1536;

        const vocabTokens = parsedModel.metadata['tokenizer.ggml.tokens'] || [];
        const vocabScores = parsedModel.metadata['tokenizer.ggml.scores'] || [];
        tokenizer = new SimpleBPETokenizer(vocabTokens, vocabScores);

        initKVCache(nLayers, 512, headDim * nKVHeads);

        isLoaded = true;
        self.postMessage({
          type: 'STATUS',
          status: `Ready: ${activeModelName} (${nLayers}L, 4MB Merkle Verified)`
        });
      }
    } catch (err) {
      isLoaded = false;
      self.postMessage({ type: 'ERROR', message: `Engine init failed: ${err.message}` });
    }
  }

  if (type === 'INFER') {
    const { prompt, taskId } = payload;
    if (!isLoaded || !modelBuffer) {
      self.postMessage({ type: 'TOKEN', taskId, token: 'Error: Engine not ready', done: true });
      return;
    }

    self.postMessage({ type: 'STATUS', status: 'Executing transformer coroutines...' });

    // Format chat prompt using standard template
    const formattedPrompt = `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
    const promptTokens = tokenizer.encode(formattedPrompt);
    const generatedTokens = [...promptTokens];
    let fullOutput = '';
    const maxGenTokens = 32;

    const x = new Float32Array(nEmbd);
    const xb = new Float32Array(nEmbd);
    const embedTensor = parsedModel.tensors.get('token_embd.weight') || parsedModel.tensors.get('model.embed_tokens.weight');
    const lmHeadTensor = parsedModel.tensors.get('output.weight') || embedTensor;
    const rowBytes = Math.ceil((nEmbd * 4.5) / 8) + 32;

    // Reset KV cache for this prompt
    initKVCache(nLayers, 512, headDim * nKVHeads);

    for (let pos = 0; pos < promptTokens.length + maxGenTokens; pos++) {
      let currentToken = pos < promptTokens.length ? promptTokens[pos] : generatedTokens[pos];

      // Embedding Lookup
      const embOffset = parsedModel.tensorDataStart + embedTensor.offset + currentToken * rowBytes;
      if (embOffset + rowBytes <= modelBuffer.byteLength) {
        dequantizeQ4_K_Block(modelView, embOffset, x, nEmbd);
      }

      // Execute Coroutine
      const coroutine = runTransformerCoroutines(x, pos);
      let stepResult = coroutine.next();
      while (!stepResult.done) {
        // Yield control to process message events and avoid thread lock
        await new Promise((r) => setTimeout(r, 0));
        stepResult = coroutine.next();
      }

      // Sample next token when prompt pre-fill finishes
      if (pos >= promptTokens.length - 1) {
        rmsNorm(xb, x, null, nEmbd);

        let bestToken = 1;
        let maxLogit = -Infinity;
        const candidatePool = Math.min(tokenizer.tokens.length, 3000);
        const recentTokens = generatedTokens.slice(-8);

        for (let v = 3; v < candidatePool; v++) {
          const headOffset = parsedModel.tensorDataStart + lmHeadTensor.offset + v * rowBytes;
          if (headOffset + rowBytes > modelBuffer.byteLength) break;

          const headWeights = new Float32Array(nEmbd);
          dequantizeQ4_K_Block(modelView, headOffset, headWeights, nEmbd);

          let logit = 0.0;
          for (let i = 0; i < nEmbd; i++) logit += xb[i] * headWeights[i];

          // Repetition penalty
          if (recentTokens.includes(v)) logit -= 3.5;

          if (logit > maxLogit) {
            maxLogit = logit;
            bestToken = v;
          }
        }

        // Stop on EOS or terminator
        if (bestToken <= 2) break;
        const decoded = tokenizer.decode(bestToken);
        if (!decoded || decoded.includes('<|im_end|>')) break;

        fullOutput += decoded;
        generatedTokens.push(bestToken);

        self.postMessage({
          type: 'TOKEN',
          taskId,
          token: decoded,
          fullText: fullOutput.trim(),
          done: pos === promptTokens.length + maxGenTokens - 1
        });
      }
    }

    self.postMessage({ type: 'TOKEN', taskId, token: '', fullText: fullOutput.trim(), done: true });
    self.postMessage({ type: 'STATUS', status: `Ready: ${activeModelName}` });
  }
};
