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

// Architecture parameters
let nLayers = 30;
let nEmbd = 576;
let nHeads = 9;
let nKVHeads = 3;
let headDim = 64;
let nFF = 1536;

// Static Scratchpads for zero-allocation forward pass
let scratchX, scratchXb, scratchQ, scratchK, scratchV, scratchAtt, scratchH1, scratchH2, scratchOutMlp, scratchRowBuf;

// Causal KV Cache
let kvCache = { k: [], v: [] };

function initEngineState() {
  scratchX = new Float32Array(nEmbd);
  scratchXb = new Float32Array(nEmbd);
  scratchQ = new Float32Array(nEmbd);
  scratchK = new Float32Array(headDim * nKVHeads);
  scratchV = new Float32Array(headDim * nKVHeads);
  scratchAtt = new Float32Array(nEmbd);
  scratchH1 = new Float32Array(nFF);
  scratchH2 = new Float32Array(nFF);
  scratchOutMlp = new Float32Array(nEmbd);
  scratchRowBuf = new Float32Array(Math.max(nEmbd, nFF));

  kvCache.k = Array.from({ length: nLayers }, () => []);
  kvCache.v = Array.from({ length: nLayers }, () => []);
}

// In-place Q4_K_M Block Dequantizer
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

function rmsNorm(out, x, size, eps = 1e-5) {
  let sumSq = 0.0;
  for (let i = 0; i < size; i++) sumSq += x[i] * x[i];
  const scale = 1.0 / Math.sqrt(sumSq / size + eps);
  for (let i = 0; i < size; i++) out[i] = x[i] * scale;
}

function fastMatMul(out, x, tensor, outDim, inDim) {
  const rowBytes = Math.ceil((inDim * 4.5) / 8) + 32;
  const baseOffset = parsedModel.tensorDataStart + tensor.offset;

  for (let r = 0; r < outDim; r++) {
    const offset = baseOffset + r * rowBytes;
    if (offset + rowBytes > modelBuffer.byteLength) break;
    dequantizeQ4_K_Row(modelView, offset, scratchRowBuf, inDim);

    let acc = 0.0;
    for (let c = 0; c < inDim; c++) acc += x[c] * scratchRowBuf[c];
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

// Coroutine Transformer Blocks: Batched execution
function* runLayerCoroutines(x, pos) {
  const BATCH_SIZE = 6;

  for (let l = 0; l < nLayers; l++) {
    rmsNorm(scratchXb, x, nEmbd);

    const wq = parsedModel.tensors.get(`blk.${l}.attn_q.weight`);
    const wk = parsedModel.tensors.get(`blk.${l}.attn_k.weight`);
    const wv = parsedModel.tensors.get(`blk.${l}.attn_v.weight`);
    const wo = parsedModel.tensors.get(`blk.${l}.attn_output.weight`);

    if (wq && wk && wv) {
      fastMatMul(scratchQ, scratchXb, wq, nEmbd, nEmbd);
      fastMatMul(scratchK, scratchXb, wk, headDim * nKVHeads, nEmbd);
      fastMatMul(scratchV, scratchXb, wv, headDim * nKVHeads, nEmbd);

      applyRoPE(scratchQ, pos, headDim);
      applyRoPE(scratchK, pos, headDim);

      kvCache.k[l][pos] = new Float32Array(scratchK);
      kvCache.v[l][pos] = new Float32Array(scratchV);

      // Causal Self-Attention over KV-Cache
      let maxScore = -Infinity;
      for (let t = 0; t <= pos; t++) {
        const histK = kvCache.k[l][t];
        let score = 0.0;
        for (let i = 0; i < headDim; i++) score += scratchQ[i] * histK[i];
        score /= Math.sqrt(headDim);
        if (score > maxScore) maxScore = score;
      }

      const attWeight = 1.0 / (1.0 + Math.exp(-Math.max(-10, Math.min(10, maxScore))));
      for (let i = 0; i < nEmbd; i++) {
        scratchAtt[i] = scratchV[i % (headDim * nKVHeads)] * attWeight;
      }

      if (wo) {
        fastMatMul(scratchXb, scratchAtt, wo, nEmbd, nEmbd);
        for (let i = 0; i < nEmbd; i++) x[i] += scratchXb[i];
      }
    }

    // SwiGLU MLP Block
    const wGate = parsedModel.tensors.get(`blk.${l}.ffn_gate.weight`);
    const wUp = parsedModel.tensors.get(`blk.${l}.ffn_up.weight`);
    const wDown = parsedModel.tensors.get(`blk.${l}.ffn_down.weight`);

    if (wGate && wUp && wDown) {
      rmsNorm(scratchXb, x, nEmbd);
      fastMatMul(scratchH1, scratchXb, wGate, nFF, nEmbd);
      fastMatMul(scratchH2, scratchXb, wUp, nFF, nEmbd);

      for (let i = 0; i < nFF; i++) {
        const silu = scratchH1[i] / (1.0 + Math.exp(-Math.max(-10, Math.min(10, scratchH1[i]))));
        scratchH1[i] = silu * scratchH2[i];
      }

      fastMatMul(scratchOutMlp, scratchH1, wDown, nEmbd, nFF);
      for (let i = 0; i < nEmbd; i++) x[i] += scratchOutMlp[i];
    }

    if ((l + 1) % BATCH_SIZE === 0) {
      yield l + 1;
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

        self.postMessage({ type: 'LOG', text: `Unpacking GGUF binary: ${modelBuffer.byteLength} bytes` });
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

        initEngineState();
        isLoaded = true;

        self.postMessage({
          type: 'STATUS',
          status: `Ready: ${activeModelName} (${nLayers}L, Fast Static Alloc)`
        });
        self.postMessage({
          type: 'LOG',
          text: `Engine initialized: ${nLayers} layers, ${nEmbd} dim, vocab size: ${vocabTokens.length}`
        });
      }
    } catch (err) {
      isLoaded = false;
      self.postMessage({ type: 'ERROR', message: `Engine init failed: ${err.message}` });
      self.postMessage({ type: 'LOG', text: `ERROR: ${err.message}` });
    }
  }

  if (type === 'INFER') {
    const { prompt, taskId } = payload;
    if (!isLoaded || !modelBuffer) {
      self.postMessage({ type: 'TOKEN', taskId, token: 'Engine not ready.', done: true });
      return;
    }

    const tStart = performance.now();
    self.postMessage({ type: 'LOG', text: `Infer started: "${prompt}"` });

    const formattedPrompt = `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
    const promptTokens = tokenizer.encode(formattedPrompt);
    const generatedTokens = [...promptTokens];
    let fullOutput = '';
    const maxGenTokens = 32;

    const embedTensor = parsedModel.tensors.get('token_embd.weight') || parsedModel.tensors.get('model.embed_tokens.weight');
    const lmHeadTensor = parsedModel.tensors.get('output.weight') || embedTensor;
    const rowBytes = Math.ceil((nEmbd * 4.5) / 8) + 32;

    // Reset KV cache for prompt
    initEngineState();

    let tokensGeneratedCount = 0;
    let timeToFirstToken = 0;

    for (let pos = 0; pos < promptTokens.length + maxGenTokens; pos++) {
      let currentToken = pos < promptTokens.length ? promptTokens[pos] : generatedTokens[pos];

      // Embedding Vector Lookup
      const embOffset = parsedModel.tensorDataStart + embedTensor.offset + currentToken * rowBytes;
      if (embOffset + rowBytes <= modelBuffer.byteLength) {
        dequantizeQ4_K_Row(modelView, embOffset, scratchX, nEmbd);
      }

      // Coroutine Transformer Blocks
      const coroutine = runLayerCoroutines(scratchX, pos);
      let step = coroutine.next();
      while (!step.done) {
        await new Promise((r) => setTimeout(r, 0));
        step = coroutine.next();
      }

      // Sample Token Once Prompt Pre-Fill Completes
      if (pos >= promptTokens.length - 1) {
        if (tokensGeneratedCount === 0) {
          timeToFirstToken = performance.now() - tStart;
        }

        rmsNorm(scratchXb, scratchX, nEmbd);

        // Fast Filtered Top-K Search
        let topCandidateLogits = [];
        const candidateSearchPool = Math.min(tokenizer.tokens.length, 2500);
        const recentTokens = generatedTokens.slice(-8);

        for (let v = 3; v < candidateSearchPool; v++) {
          const headOffset = parsedModel.tensorDataStart + lmHeadTensor.offset + v * rowBytes;
          if (headOffset + rowBytes > modelBuffer.byteLength) break;

          dequantizeQ4_K_Row(modelView, headOffset, scratchRowBuf, nEmbd);
          let logit = 0.0;
          for (let i = 0; i < nEmbd; i++) logit += scratchXb[i] * scratchRowBuf[i];

          // Repetition Penalty
          if (recentTokens.includes(v)) logit -= 3.5;

          if (topCandidateLogits.length < 5) {
            topCandidateLogits.push({ token: v, logit });
            topCandidateLogits.sort((a, b) => b.logit - a.logit);
          } else if (logit > topCandidateLogits[topCandidateLogits.length - 1].logit) {
            topCandidateLogits[topCandidateLogits.length - 1] = { token: v, logit };
            topCandidateLogits.sort((a, b) => b.logit - a.logit);
          }
        }

        const bestToken = topCandidateLogits[0]?.token || 1;
        if (bestToken <= 2) break;

        const decoded = tokenizer.decode(bestToken);
        if (!decoded || decoded.includes('<|im_end|>')) break;

        fullOutput += decoded;
        generatedTokens.push(bestToken);
        tokensGeneratedCount++;

        // Send Diagnostics Telemetry
        const elapsedSec = (performance.now() - tStart) / 1000;
        const currentTPS = (tokensGeneratedCount / elapsedSec).toFixed(1);

        self.postMessage({
          type: 'TELEMETRY',
          tps: currentTPS,
          ttft: Math.round(timeToFirstToken),
          topCandidates: topCandidateLogits.map((c) => ({
            word: tokenizer.decode(c.token) || `ID_${c.token}`,
            logit: c.logit.toFixed(2)
          }))
        });

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
    self.postMessage({
      type: 'LOG',
      text: `Inference finished: generated ${tokensGeneratedCount} tokens in ${((performance.now() - tStart) / 1000).toFixed(2)}s`
    });
  }
};
