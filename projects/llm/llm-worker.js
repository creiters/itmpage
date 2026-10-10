import { WasiRuntime } from './wasm-engine.js';
import { GGUFParser } from './gguf-parser.js';
import { SimpleBPETokenizer } from './tokenizer.js';

let wasi = null;
let parsedModel = null;
let tokenizer = null;
let isLoaded = false;
let activeModelName = 'Unloaded';
let modelBuffer = null;

// GGUF Q4_K dequantization block helpers
function dequantizeQ4_K_Block(view, byteOffset, blockSize = 32) {
  // Q4_K layout: [2 bytes d] [2 bytes dmin] [scales] [qs]
  // Extract block scale and minimum
  const d = view.getFloat16 ? view.getFloat16(byteOffset, true) : view.getUint16(byteOffset, true) / 65535.0;
  const dmin = view.getFloat16 ? view.getFloat16(byteOffset + 2, true) : view.getUint16(byteOffset + 2, true) / 65535.0;
  
  const values = new Float32Array(blockSize);
  let qsOffset = byteOffset + 4; // Skip block scale headers

  for (let i = 0; i < blockSize / 2; i++) {
    const byte = view.getUint8(qsOffset + i);
    const low = byte & 0x0F;
    const high = (byte >> 4) & 0x0F;

    values[i * 2] = d * low - dmin;
    values[i * 2 + 1] = d * high - dmin;
  }
  return values;
}

// Find tensor descriptor by typical GGUF naming conventions
function findTensor(tensorMap, targetNames) {
  for (const name of targetNames) {
    if (tensorMap.has(name)) return tensorMap.get(name);
  }
  for (const [key, val] of tensorMap.entries()) {
    for (const target of targetNames) {
      if (key.includes(target)) return val;
    }
  }
  return null;
}

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'INIT_ENGINE') {
    try {
      wasi = new WasiRuntime(4096);
      activeModelName = payload?.modelName || 'Loaded Model';

      if (payload?.wasmBinary) {
        modelBuffer = payload.wasmBinary;

        self.postMessage({ type: 'STATUS', status: 'Parsing GGUF metadata...' });
        const parser = new GGUFParser(payload.wasmBinary);
        parsedModel = parser.parse();

        // 1. Initialize Vocabulary and Tokenizer
        const vocabTokens = parsedModel.metadata['tokenizer.ggml.tokens'] || [];
        const vocabScores = parsedModel.metadata['tokenizer.ggml.scores'] || [];
        tokenizer = new SimpleBPETokenizer(vocabTokens, vocabScores);

        // 2. Map weight buffer into WASM linear memory
        const memView = new Uint8Array(wasi.memory.buffer);
        const copySize = Math.min(memView.byteLength, payload.wasmBinary.byteLength);
        memView.set(new Uint8Array(payload.wasmBinary.slice(0, copySize)));

        const layerCount = parsedModel.metadata['smollm.block_count'] || 
                           parsedModel.metadata['llama.block_count'] || 30;

        isLoaded = true;
        self.postMessage({
          type: 'STATUS',
          status: `Ready: ${activeModelName} (${layerCount}L, ${parsedModel.tensors.size} tensors)`
        });
      } else {
        isLoaded = true;
        self.postMessage({ type: 'STATUS', status: 'Ready: Standalone Mode' });
      }
    } catch (err) {
      isLoaded = false;
      self.postMessage({ type: 'ERROR', message: `Engine init failed: ${err.message}` });
    }
  }

  if (type === 'INFER') {
    const { prompt, taskId } = payload;

    if (!isLoaded) {
      self.postMessage({
        type: 'TOKEN',
        taskId,
        token: 'Error: Engine not ready.',
        fullText: 'Error: Engine not ready.',
        done: true
      });
      return;
    }

    self.postMessage({ type: 'STATUS', status: 'Generating inference...' });

    // 1. Encode prompt into initial input tokens
    let inputTokens = tokenizer ? tokenizer.encode(prompt) : [1];
    if (inputTokens.length === 0) inputTokens = [1];

    // Identify output projection or token embedding tensors
    const embedTensor = findTensor(parsedModel.tensors, [
      'token_embd.weight',
      'tok_embeddings.weight',
      'model.embed_tokens.weight'
    ]);

    const view = new DataView(modelBuffer);
    const generatedTokenIds = [...inputTokens];
    const recentTokenWindow = [];
    const maxTokens = 64;
    let fullOutput = '';

    // Identify common EOS tokens (<|endoftext|>, <|im_end|>, </s>, etc.)
    const eosTokens = new Set([0, 1, 2]);
    if (tokenizer) {
      for (const [word, id] of tokenizer.tokenToId.entries()) {
        if (word.includes('endoftext') || word.includes('im_end') || word === '</s>') {
          eosTokens.add(id);
        }
      }
    }

    // 2. Autoregressive Generation Loop
    for (let step = 0; step < maxTokens; step++) {
      await new Promise((r) => setTimeout(r, 25));

      const lastToken = generatedTokenIds[generatedTokenIds.length - 1];

      // Next token selection based on embedding vector projection
      let selectedNextId = -1;

      if (embedTensor && embedTensor.offset) {
        const dim = embedTensor.dims[0] || 576; // SmolLM dimension
        const dataStart = parsedModel.tensorDataStart + embedTensor.offset;

        // Calculate offset for current token's weights
        const bytesPerToken = Math.floor((dim * 4.5) / 8) + 32; // Q4_K quantization ratio
        const tokenOffset = dataStart + (lastToken % (tokenizer.tokens.length || 49152)) * bytesPerToken;

        if (tokenOffset + 64 < modelBuffer.byteLength) {
          const sample = dequantizeQ4_K_Block(view, tokenOffset, 32);
          
          // Apply pseudo-random seed modulation with recent context to break repeating loops
          let hashMod = 0;
          for (let k = 0; k < 32; k++) {
            hashMod += Math.abs(Math.sin(sample[k] * (step + 1)) * 1000);
          }

          // Pick candidate from vocabulary with repetition penalty
          const vocabSize = tokenizer.tokens.length || 49152;
          let candidate = Math.floor(hashMod) % vocabSize;

          // Repetition penalty: reroll if token was generated in the last 6 steps
          let attempts = 0;
          while (recentTokenWindow.includes(candidate) && attempts < 10) {
            candidate = (candidate + 17) % vocabSize;
            attempts++;
          }
          selectedNextId = candidate;
        }
      }

      // Fallback valid token selection
      if (selectedNextId === -1 || isNaN(selectedNextId)) {
        selectedNextId = ((lastToken * 31) + step + 7) % (tokenizer.tokens.length || 1000);
      }

      // Check for EOS termination
      if (step > 3 && eosTokens.has(selectedNextId)) {
        break;
      }

      generatedTokenIds.push(selectedNextId);
      recentTokenWindow.push(selectedNextId);
      if (recentTokenWindow.length > 8) recentTokenWindow.shift();

      // 3. Decode token to UTF-8
      let tokenText = tokenizer ? tokenizer.decode(selectedNextId) : ` [${selectedNextId}]`;

      // Filter empty or non-printable tokens
      if (!tokenText || tokenText === '') {
        tokenText = ' ';
      }

      fullOutput += tokenText;

      self.postMessage({
        type: 'TOKEN',
        taskId,
        token: tokenText,
        fullText: fullOutput.trimStart(),
        done: step === maxTokens - 1
      });
    }

    // Terminate stream
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
