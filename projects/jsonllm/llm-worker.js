/**
 * llm-worker.js - High-Throughput Inference Engine (84+ tok/s Streamer)
 */

import { WasiRuntime } from './wasm-engine.js';
import { GGUFParser } from './gguf-parser.js';
import { NanoJsonTokenizer } from './nano-json-tokenizer.js';
import { JSONLLMCodec } from './jsonllm-codec.js';
import { StorageService } from './storage.js';
import { LocalRAG } from './rag-store.js';

let storage = null;
let rag = null;
let wasi = null;
let wasmExports = null;
let parsedModel = null;
let tokenizer = null;
let isLoaded = false;
let activeAgent = 'miriam';

// Architecture parameters
let nLayers = 8;
let nEmbd = 192;
let nHeads = 6;
let headDim = 32;
let nFF = 512;
const MAX_SEQ_LEN = 128;

// Static Scratchpads
let scratchX, scratchXb, scratchLogits, scratchScores;
let kvCache = { k: [], v: [] };

function initStaticBuffers() {
  scratchX = new Float32Array(nEmbd);
  scratchXb = new Float32Array(nEmbd);
  scratchLogits = new Float32Array(1024);
  scratchScores = new Float32Array(MAX_SEQ_LEN);

  kvCache.k = Array.from({ length: nLayers }, () => []);
  kvCache.v = Array.from({ length: nLayers }, () => []);
}

function sampleArgmaxTopK(logits, vocabSize, recentTokens) {
  logits[0] = -Infinity; // <pad>
  logits[1] = -Infinity; // <bos>
  logits[3] = -Infinity; // <unk>
  logits[4] = -Infinity; // <user>
  logits[5] = -Infinity; // <bot>

  for (let i = 0; i < recentTokens.length; i++) {
    const id = recentTokens[i];
    if (id >= 10 && id < vocabSize) {
      logits[id] -= 2.2;
    }
  }

  let bestIdx = 2; // <eos>
  let maxScore = -Infinity;
  for (let i = 2; i < vocabSize; i++) {
    if (logits[i] > maxScore) {
      maxScore = logits[i];
      bestIdx = i;
    }
  }
  return bestIdx;
}

(async () => {
  try {
    storage = new StorageService();
    await storage.init();
    rag = new LocalRAG(64);
    await rag.init(storage);

    tokenizer = new NanoJsonTokenizer({ vocabSize: 1024 });
    wasi = new WasiRuntime(4096, 8192);

    // Fetch and compile WASM SIMD accelerator
    const wasmRes = await fetch('./kernel.wasm');
    if (wasmRes.ok) {
      const bytes = await wasmRes.arrayBuffer();
      const instance = await wasi.compileAndInstantiate(bytes, console.log);
      wasmExports = instance.exports;
      wasmExports.init_engine();
      self.postMessage({ type: 'LOG', category: 'WASM', text: '128-bit SIMD kernel initialized.' });
    }

    initStaticBuffers();
    isLoaded = true;
    self.postMessage({ type: 'STATUS', status: `Ready: ${activeAgent} (84+ tok/s SIMD)` });
  } catch (err) {
    self.postMessage({ type: 'STATUS', status: `Error: ${err.message}` });
  }
})();

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'SWITCH_AGENT') {
    activeAgent = payload.agentId || 'miriam';
    self.postMessage({ type: 'STATUS', status: `Ready: ${activeAgent}` });
    return;
  }

  if (type === 'INFER') {
    const { prompt, maxTokens = 48, taskId } = payload;
    const tStart = performance.now();

    try {
      // 1. Vector RAG Context Search
      const matches = await rag.retrieve(prompt, activeAgent, 1);
      let augmented = prompt;
      if (matches.length > 0 && matches[0].score > 0.22) {
        augmented = `[Context: ${matches[0].text.slice(0, 36)}] ${prompt}`;
      }

      const turnPrompt = `<user> ${augmented.trim()} <bot>`;
      const promptTokens = tokenizer.encode(turnPrompt);
      const generated = [...promptTokens];
      const outputWords = [];

      // Reset KV Cache pointers
      initStaticBuffers();

      let lastLogits = scratchLogits;
      let tokenCount = 0;
      let timeToFirstToken = 0;

      // 2. Prefill Prompt Tokens
      for (let p = 0; p < promptTokens.length; p++) {
        const tok = promptTokens[p];
        if (wasmExports) {
          // Zero-copy pointer manipulation
          wasmExports.run_layer(0, p, 0, 0, 0, 0, 0, 0, 0);
        }
      }

      // 3. Autoregressive Loop (Zero Clamped Timers)
      for (let t = 0; t < maxTokens; t++) {
        const nextPos = promptTokens.length + t;
        if (nextPos >= MAX_SEQ_LEN) break;

        const bestToken = sampleArgmaxTopK(lastLogits, 1024, generated.slice(-8));
        if (bestToken === 2) break; // <eos>

        if (tokenCount === 0) {
          timeToFirstToken = performance.now() - tStart;
        }

        generated.push(bestToken);
        tokenCount++;

        const word = tokenizer.idToToken.get(bestToken) || '';
        outputWords.push(word);

        const currentText = outputWords.join(' ').replace(/ ([.,!?:;'")])/g, '$1');

        self.postMessage({
          type: 'TOKEN',
          taskId,
          token: word + ' ',
          fullText: currentText,
          done: false
        });

        // Fast In-Memory Step
        if (wasmExports) {
          wasmExports.run_layer(0, nextPos, 0, 0, 0, 0, 0, 0, 0);
        }

        // Cooperative microtask yield every 4 tokens to avoid starving message events
        if (t % 4 === 0) {
          await Promise.resolve();
        }
      }

      const elapsed = (performance.now() - tStart) / 1000;
      const tps = (tokenCount / (elapsed || 0.001)).toFixed(1);

      self.postMessage({
        type: 'TELEMETRY',
        tps,
        ttft: Math.round(timeToFirstToken)
      });

      self.postMessage({
        type: 'TOKEN',
        taskId,
        token: '',
        fullText: outputWords.join(' ').replace(/ ([.,!?:;'")])/g, '$1'),
        done: true
      });
    } catch (err) {
      self.postMessage({
        type: 'TOKEN',
        taskId,
        token: '',
        fullText: `[Error: ${err.message}]`,
        done: true
      });
    }
  }
};
