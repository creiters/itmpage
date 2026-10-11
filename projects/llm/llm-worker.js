import { WasiRuntime } from './wasm-engine.js';
import { GGUFParser } from './gguf-parser.js';
import { SimpleBPETokenizer } from './tokenizer.js';
import { GPUKernel } from './gpu-kernel.js';

let wasi = null;
let wasmInstance = null;
let gpuEngine = null;
let parsedModel = null;
let tokenizer = null;
let isLoaded = false;
let activeModelName = 'Unloaded';
let executionDevice = 'CPU'; // 'CPU' | 'GPU'
let modelBuffer = null;
let chunkManifest = [];

let xPtr = 0;
let xbPtr = 0;
let weightsBasePtr = 0;

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'SET_DEVICE') {
    const target = payload.device;
    if (target === 'GPU') {
      if (!gpuEngine) gpuEngine = new GPUKernel();
      const ok = await gpuEngine.init();
      if (ok) {
        executionDevice = 'GPU';
        self.postMessage({ type: 'STATUS', status: `Ready: ${activeModelName} (WebGPU Active)` });
        self.postMessage({ type: 'LOG', category: 'DEVICE', text: 'Hardware backend switched to GPU (WebGPU).' });
      } else {
        executionDevice = 'CPU';
        self.postMessage({ type: 'LOG', category: 'DEVICE', text: 'WebGPU not available on this device. Fallback to CPU.' });
      }
    } else {
      executionDevice = 'CPU';
      self.postMessage({ type: 'STATUS', status: `Ready: ${activeModelName} (CPU WASI/SIMD Active)` });
      self.postMessage({ type: 'LOG', category: 'DEVICE', text: 'Hardware backend switched to CPU (WASI SIMD).' });
    }
  }

  if (type === 'INIT_ENGINE') {
    try {
      wasi = new WasiRuntime(4096, 8192);
      activeModelName = payload?.modelName || 'Loaded Model';
      chunkManifest = payload?.manifest || [];

      // Check GPU capability
      gpuEngine = new GPUKernel();
      const hasGPU = await gpuEngine.init();
      executionDevice = hasGPU ? 'GPU' : 'CPU';
      wasi.setDevice(executionDevice);

      if (payload?.engineWasmBinary && payload?.modelBinary) {
        modelBuffer = payload.modelBinary;

        self.postMessage({ type: 'LOG', category: 'WASM', text: 'Instantiating WASI C-WASM module...' });
        wasmInstance = await wasi.compileAndInstantiate(payload.engineWasmBinary, (log) => {
          self.postMessage({ type: 'LOG', category: 'WASM_OUT', text: log });
        });

        const exports = wasmInstance.exports;
        exports.init_engine();
        xPtr = exports.get_x_ptr();
        xbPtr = exports.get_xb_ptr();

        self.postMessage({ type: 'LOG', category: 'GGUF', text: 'Parsing GGUF metadata...' });
        const parser = new GGUFParser(payload.modelBinary);
        parsedModel = parser.parse();

        const vocabTokens = parsedModel.metadata['tokenizer.ggml.tokens'] || [];
        const vocabScores = parsedModel.metadata['tokenizer.ggml.scores'] || [];
        tokenizer = new SimpleBPETokenizer(vocabTokens, vocabScores);

        weightsBasePtr = 16 * 1024 * 1024;
        const targetView = new Uint8Array(wasi.memory.buffer, weightsBasePtr, modelBuffer.byteLength);
        targetView.set(new Uint8Array(modelBuffer));

        isLoaded = true;
        self.postMessage({
          type: 'STATUS',
          status: `Ready: ${activeModelName} (${executionDevice} Mode)`
        });
        self.postMessage({
          type: 'LOG',
          category: 'INIT',
          text: `Engine initialized in ${executionDevice} mode with ${chunkManifest.length} DAG slices.`
        });
      }
    } catch (err) {
      isLoaded = false;
      self.postMessage({ type: 'ERROR', message: `Init failed: ${err.message}` });
      self.postMessage({ type: 'LOG', category: 'ERROR', text: err.message });
    }
  }

  if (type === 'INFER') {
    const { prompt, taskId } = payload;
    if (!isLoaded || !wasmInstance) {
      self.postMessage({ type: 'TOKEN', taskId, token: 'Engine not ready.', done: true });
      return;
    }

    const tStart = performance.now();
    self.postMessage({ type: 'LOG', category: 'INFERENCE', text: `Inferencing via ${executionDevice}: "${prompt}"` });

    const formattedPrompt = `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
    const promptTokens = tokenizer.encode(formattedPrompt);
    const generatedTokens = [...promptTokens];
    let fullOutput = '';
    const maxGenTokens = 32;

    const exports = wasmInstance.exports;
    exports.init_engine();

    const embedTensor = parsedModel.tensors.get('token_embd.weight') || parsedModel.tensors.get('model.embed_tokens.weight');
    const lmHeadTensor = parsedModel.tensors.get('output.weight') || embedTensor;
    const rowBytes = Math.ceil((576 * 4.5) / 8) + 32;
    const tensorStart = weightsBasePtr + parsedModel.tensorDataStart;

    let tokensGeneratedCount = 0;
    let timeToFirstToken = 0;

    for (let pos = 0; pos < promptTokens.length + maxGenTokens; pos++) {
      let currentToken = pos < promptTokens.length ? promptTokens[pos] : generatedTokens[pos];

      // Embedding Vector Lookup
      const embOffset = tensorStart + embedTensor.offset + (currentToken * rowBytes);
      const wasmFloatView = new Float32Array(wasi.memory.buffer);
      const wasmByteView = new Uint8Array(wasi.memory.buffer);

      for (let i = 0; i < 576; i++) {
        wasmFloatView[(xPtr / 4) + i] = 0;
      }
      let outIdx = 0;
      let inOffset = embOffset + 4;
      while (outIdx < 576) {
        const byte = wasmByteView[inOffset++];
        wasmFloatView[(xPtr / 4) + outIdx++] = (byte & 0x0F) - 8;
        if (outIdx < 576) wasmFloatView[(xPtr / 4) + outIdx++] = ((byte >> 4) & 0x0F) - 8;
      }

      // 30 Transformer Blocks
      for (let l = 0; l < 30; l++) {
        const wq = tensorStart + parsedModel.tensors.get(`blk.${l}.attn_q.weight`).offset;
        const wk = tensorStart + parsedModel.tensors.get(`blk.${l}.attn_k.weight`).offset;
        const wv = tensorStart + parsedModel.tensors.get(`blk.${l}.attn_v.weight`).offset;
        const wo = tensorStart + parsedModel.tensors.get(`blk.${l}.attn_output.weight`).offset;
        const wGate = tensorStart + parsedModel.tensors.get(`blk.${l}.ffn_gate.weight`).offset;
        const wUp = tensorStart + parsedModel.tensors.get(`blk.${l}.ffn_up.weight`).offset;
        const wDown = tensorStart + parsedModel.tensors.get(`blk.${l}.ffn_down.weight`).offset;

        exports.run_layer(l, pos, wq, wk, wv, wo, wGate, wUp, wDown);
      }

      // Sample Token
      if (pos >= promptTokens.length - 1) {
        if (tokensGeneratedCount === 0) {
          timeToFirstToken = performance.now() - tStart;
        }

        exports.rms_norm(xbPtr, xPtr, 576, 1e-5);

        let topCandidates = [];
        const candidatePool = Math.min(tokenizer.tokens.length, 1500);
        const recentTokens = generatedTokens.slice(-8);

        for (let v = 3; v < candidatePool; v++) {
          const rowPtr = tensorStart + lmHeadTensor.offset + (v * rowBytes);
          let logit = exports.fused_dot_q4_k(xbPtr, rowPtr, 576);

          if (recentTokens.includes(v)) logit -= 3.5;

          if (topCandidates.length < 5) {
            topCandidates.push({ token: v, logit });
            topCandidates.sort((a, b) => b.logit - a.logit);
          } else if (logit > topCandidates[topCandidates.length - 1].logit) {
            topCandidates[topCandidates.length - 1] = { token: v, logit };
            topCandidates.sort((a, b) => b.logit - a.logit);
          }
        }

        const bestToken = topCandidates[0]?.token || 1;
        if (bestToken <= 2) break;

        const decoded = tokenizer.decode(bestToken);
        if (!decoded || decoded.includes('<|im_end|>')) break;

        fullOutput += decoded;
        generatedTokens.push(bestToken);
        tokensGeneratedCount++;

        const elapsedSec = (performance.now() - tStart) / 1000;
        const currentTPS = (tokensGeneratedCount / elapsedSec).toFixed(1);

        self.postMessage({
          type: 'TELEMETRY',
          tps: currentTPS,
          ttft: Math.round(timeToFirstToken),
          device: executionDevice,
          topCandidates: topCandidates.map((c) => ({
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

        await new Promise((r) => setTimeout(r, 0));
      }
    }

    self.postMessage({ type: 'TOKEN', taskId, token: '', fullText: fullOutput.trim(), done: true });
    self.postMessage({
      type: 'LOG',
      category: 'INFERENCE_COMPLETE',
      text: `Inference finished on ${executionDevice}: ${tokensGeneratedCount} tokens in ${((performance.now() - tStart) / 1000).toFixed(2)}s`
    });
  }
};
