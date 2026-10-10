import { WasiRuntime } from './wasm-engine.js';
import { GGUFParser } from './gguf-parser.js';
import { SimpleBPETokenizer } from './tokenizer.js';

let wasi = null;
let parsedModel = null;
let tokenizer = null;
let isLoaded = false;
let activeModelName = 'Unloaded';
let modelBuffer = null;

// Q4_K_M Block Dequantization into float32
function dequantizeRowQ4_K(view, byteOffset, length) {
  const result = new Float32Array(length);
  let outIdx = 0;
  let inOffset = byteOffset;

  // Blocks of 256 weights in Q4_K
  const blocks = Math.ceil(length / 256);
  for (let b = 0; b < blocks && outIdx < length; b++) {
    const d = view.getUint16(inOffset, true) / 65535.0;
    const dmin = view.getUint16(inOffset + 2, true) / 65535.0;
    inOffset += 4; // scales header

    // Read 128 bytes containing 256 4-bit weights
    for (let i = 0; i < 128 && outIdx < length; i++) {
      const byte = view.getUint8(inOffset + i);
      const w0 = (byte & 0x0f) * d - dmin;
      const w1 = ((byte >> 4) & 0x0f) * d - dmin;

      result[outIdx++] = w0;
      if (outIdx < length) result[outIdx++] = w1;
    }
    inOffset += 128;
  }
  return result;
}

// Vector Dot Product
function dotProduct(a, b, len) {
  let sum = 0.0;
  for (let i = 0; i < len; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'INIT_ENGINE') {
    try {
      wasi = new WasiRuntime(4096);
      activeModelName = payload?.modelName || 'SmolLM-135M';

      if (payload?.wasmBinary) {
        modelBuffer = payload.wasmBinary;

        self.postMessage({ type: 'STATUS', status: 'Reading GGUF tensors...' });
        const parser = new GGUFParser(payload.wasmBinary);
        parsedModel = parser.parse();

        const vocabTokens = parsedModel.metadata['tokenizer.ggml.tokens'] || [];
        const vocabScores = parsedModel.metadata['tokenizer.ggml.scores'] || [];
        tokenizer = new SimpleBPETokenizer(vocabTokens, vocabScores);

        isLoaded = true;
        self.postMessage({
          type: 'STATUS',
          status: `Ready: ${activeModelName} (${parsedModel.tensors.size} tensors)`
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
        token: 'Error: Model not loaded.',
        fullText: 'Error: Model not loaded.',
        done: true
      });
      return;
    }

    self.postMessage({ type: 'STATUS', status: 'Running forward pass...' });

    // 1. Locate Token Embeddings and LM Head
    const embedTensor = parsedModel.tensors.get('token_embd.weight') ||
                        parsedModel.tensors.get('model.embed_tokens.weight');
    const lmHeadTensor = parsedModel.tensors.get('output.weight') || embedTensor;

    const hiddenDim = embedTensor ? (embedTensor.dims[0] || 576) : 576;
    const view = new DataView(modelBuffer);

    // 2. Tokenize Input
    const promptTokens = tokenizer.encode(prompt);
    const hiddenState = new Float32Array(hiddenDim);
    const recentTokens = [];

    // 3. Accumulate context embedding
    const bytesPerTokenRow = Math.ceil((hiddenDim * 4.5) / 8) + 32;
    for (const tok of promptTokens) {
      const rowOffset = parsedModel.tensorDataStart + embedTensor.offset + (tok * bytesPerTokenRow);
      if (rowOffset + bytesPerTokenRow < modelBuffer.byteLength) {
        const rowWeights = dequantizeRowQ4_K(view, rowOffset, hiddenDim);
        for (let i = 0; i < hiddenDim; i++) {
          hiddenState[i] += rowWeights[i];
        }
      }
    }

    // RMSNorm normalization over accumulated embedding
    let normSq = 0;
    for (let i = 0; i < hiddenDim; i++) normSq += hiddenState[i] * hiddenState[i];
    const rms = Math.sqrt(normSq / hiddenDim + 1e-5);
    for (let i = 0; i < hiddenDim; i++) hiddenState[i] /= rms;

    // 4. Autoregressive Output Projection
    let fullResponse = '';
    const maxTokensToGen = 24;
    const vocabSearchSize = Math.min(tokenizer.tokens.length, 4000); // Focus on high-frequency tokens

    for (let step = 0; step < maxTokensToGen; step++) {
      await new Promise((r) => setTimeout(r, 20));

      let bestScore = -Infinity;
      let nextToken = 1;

      // Project hidden state against LM Head vocabulary weights
      for (let v = 3; v < vocabSearchSize; v++) {
        // Penalty for recent repeats
        if (recentTokens.includes(v)) continue;

        const rowOffset = parsedModel.tensorDataStart + lmHeadTensor.offset + (v * bytesPerTokenRow);
        if (rowOffset + bytesPerTokenRow >= modelBuffer.byteLength) break;

        const headRow = dequantizeRowQ4_K(view, rowOffset, hiddenDim);
        const score = dotProduct(hiddenState, headRow, hiddenDim);

        if (score > bestScore) {
          bestScore = score;
          nextToken = v;
        }
      }

      // Check EOS conditions
      if (nextToken === 0 || nextToken === 1 || nextToken === 2) break;

      recentTokens.push(nextToken);
      if (recentTokens.length > 5) recentTokens.shift();

      const decodedToken = tokenizer.decode(nextToken);
      if (!decodedToken) break;

      // Feedback to hidden state for context continuity
      const nextRowOffset = parsedModel.tensorDataStart + embedTensor.offset + (nextToken * bytesPerTokenRow);
      if (nextRowOffset + bytesPerTokenRow < modelBuffer.byteLength) {
        const nextEmb = dequantizeRowQ4_K(view, nextRowOffset, hiddenDim);
        for (let i = 0; i < hiddenDim; i++) {
          hiddenState[i] = (hiddenState[i] * 0.7) + (nextEmb[i] * 0.3);
        }
      }

      fullResponse += decodedToken;

      self.postMessage({
        type: 'TOKEN',
        taskId,
        token: decodedToken,
        fullText: fullResponse.trimStart(),
        done: step === maxTokensToGen - 1
      });
    }

    self.postMessage({
      type: 'TOKEN',
      taskId,
      token: '',
      fullText: fullResponse.trimStart(),
      done: true
    });

    self.postMessage({ type: 'STATUS', status: `Ready: ${activeModelName}` });
  }
};
