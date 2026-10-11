/**
 * llm-worker.js - 128M Dialogue Agent Inference Worker
 */

import { NanoTransformer } from './nano-engine.js';
import { JSONLLMCodec } from './jsonllm-codec.js';
import { StorageService } from './storage.js';
import { LocalRAG } from './rag-store.js';

let storage = null;
let rag = null;
let engine = null;
let activeModelName = 'dialogue-128m';

// 128M Conversational Matrix
const SEED_CORPUS = [
  { u: "Hi", b: "Hello! How can I assist you with your project today?" },
  { u: "Hello", b: "Hi there! I am your offline 128M dialogue agent." },
  { u: "Who are you?", b: "I am an autonomous 128M browser language model running locally in JavaScript." },
  { u: "What is your architecture?", b: "I am a 128M parameter transformer with 16 layers, 12 heads, and SwiGLU MLP." },
  { u: "Are you online?", b: "No, all 128M weights run strictly offline inside your browser tab." },
  { u: "Can you help me with code?", b: "Yes, I understand C, HTML5, WebAssembly, and systems programming." },
  { u: "How are weights stored?", b: "Weights are compressed in 4-bit Q4_0 format and stored in IndexedDB." },
  { u: "What is WebMCP?", b: "WebMCP is the in-browser Model Context Protocol interface for tool calling." },
  { u: "Thank you", b: "You are very welcome! Let me know if you need anything else." }
];

async function trainBaseline(transformer, epochs = 24) {
  for (let ep = 0; ep < epochs; ep++) {
    for (const pair of SEED_CORPUS) {
      const fullText = `<user> ${pair.u} <bot> ${pair.b} <eos>`;
      const tokens = transformer.tokenizer.encode(fullText);
      transformer.trainSequence(tokens, 0.003);
    }
  }
}

function sampleNextToken(logits, vocabSize, specialsCount, recentTokens = []) {
  const clone = new Float32Array(logits);

  // Mask control tokens
  clone[0] = -Infinity; // <pad>
  clone[1] = -Infinity; // <bos>
  clone[3] = -Infinity; // <unk>
  clone[4] = -Infinity; // <user>
  clone[5] = -Infinity; // <bot>
  clone[6] = -Infinity; // <context>

  // Repetition penalty
  for (const id of recentTokens.slice(-8)) {
    if (id >= specialsCount && id < vocabSize) {
      clone[id] -= 2.6;
    }
  }

  let maxL = -Infinity;
  for (let i = 2; i < vocabSize; i++) {
    if (clone[i] > maxL) maxL = clone[i];
  }

  const temp = 0.55;
  const candidates = [];
  let sumExp = 0.0;

  for (let i = 2; i < vocabSize; i++) {
    if (clone[i] === -Infinity) continue;
    const v = Math.exp((clone[i] - maxL) / temp);
    candidates.push({ id: i, val: v });
    sumExp += v;
  }

  for (const c of candidates) c.prob = c.val / (sumExp || 1.0);
  candidates.sort((a, b) => b.prob - a.prob);

  // Top-K (4) with Top-P (0.85) Cutoff
  const kPool = candidates.slice(0, Math.min(4, candidates.length));
  let cumSum = 0.0;
  const pPool = [];
  for (const c of kPool) {
    cumSum += c.prob;
    pPool.push(c);
    if (cumSum >= 0.85) break;
  }

  let r = Math.random() * cumSum;
  for (const c of pPool) {
    r -= c.prob;
    if (r <= 0) return c.id;
  }
  return pPool[0]?.id || 2;
}

async function initModel() {
  self.postMessage({ type: 'STATUS', status: 'Allocating 128M parameter memory pool...' });

  engine = new NanoTransformer({
    dim: 768,
    layers: 16,
    heads: 12,
    ffnDim: 2048,
    vocabSize: 4096,
    maxSeqLen: 128
  });

  const cached = await storage.getModel(activeModelName);
  if (cached) {
    try {
      const dec = await JSONLLMCodec.importFromJSONLLM(cached);
      engine.weights = dec.weights;
      self.postMessage({ type: 'STATUS', status: `Ready: ${activeModelName} (128M Cached)` });
      return;
    } catch {
      // Fall through to initial training
    }
  }

  self.postMessage({ type: 'STATUS', status: 'Training 128M baseline dialogue syntax...' });
  await trainBaseline(engine, 28);

  const jsonStr = await JSONLLMCodec.exportToJSONLLM(
    { dim: engine.dim, layers: engine.layers, heads: engine.heads, ffnDim: engine.ffnDim, vocabSize: engine.vocabSize, maxSeqLen: engine.maxSeqLen },
    engine.tokenizer,
    engine.weights
  );
  await storage.putModel(activeModelName, jsonStr);
  self.postMessage({ type: 'STATUS', status: `Ready: ${activeModelName} (128M Initialized)` });
}

(async () => {
  storage = new StorageService();
  await storage.init();
  rag = new LocalRAG(64);
  await rag.init(storage);

  await initModel();
})();

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'INFER') {
    const { prompt, maxTokens = 36 } = payload;
    const tStart = performance.now();

    try {
      // Vector RAG Fact Matching
      const matches = await rag.retrieve(prompt, 1);
      let augmented = prompt;
      if (matches.length > 0 && matches[0].score > 0.18) {
        augmented = `[Fact: ${matches[0].text.slice(0, 36)}] ${prompt}`;
        self.postMessage({ type: 'LOG', category: 'INFER', text: `RAG Context Applied: "${matches[0].text}"` });
      }

      const turnPrompt = `<user> ${augmented.trim()} <bot>`;
      const promptTokens = engine.tokenizer.encode(turnPrompt);

      const generated = [...promptTokens];
      const outputWords = [];

      const kvCache = {
        k: Array.from({ length: engine.layers }, () => []),
        v: Array.from({ length: engine.layers }, () => [])
      };

      // 1. Prefill all prompt tokens
      let lastLogits = null;
      for (let p = 0; p < promptTokens.length; p++) {
        const res = engine.forwardStepWithTape(promptTokens[p], p, kvCache);
        lastLogits = res.logits;
      }

      // 2. Autoregressive token generation
      for (let t = 0; t < maxTokens; t++) {
        const nextPos = promptTokens.length + t;
        if (nextPos >= engine.maxSeqLen) break;

        const bestToken = sampleNextToken(
          lastLogits,
          engine.vocabSize,
          engine.tokenizer.specials.length,
          generated
        );

        if (bestToken === 2) break; // <eos>

        generated.push(bestToken);
        const word = engine.tokenizer.idToToken.get(bestToken) || '';
        outputWords.push(word);

        const currentText = outputWords.join(' ').replace(/ ([.,!?:;'")])/g, '$1');
        const elapsed = (performance.now() - tStart) / 1000;
        const tps = ((t + 1) / (elapsed || 0.001)).toFixed(1);

        self.postMessage({
          type: 'TELEMETRY',
          tps,
          ttft: Math.round(performance.now() - tStart),
          topCandidates: [{ word, logit: '1.0' }]
        });

        self.postMessage({
          type: 'TOKEN',
          token: word + ' ',
          fullText: currentText,
          done: false
        });

        const nextRes = engine.forwardStepWithTape(bestToken, nextPos, kvCache);
        lastLogits = nextRes.logits;
        await new Promise((r) => setTimeout(r, 12));
      }

      self.postMessage({
        type: 'TOKEN',
        token: '',
        fullText: outputWords.join(' ').replace(/ ([.,!?:;'")])/g, '$1'),
        done: true
      });

    } catch (err) {
      self.postMessage({
        type: 'TOKEN',
        token: '',
        fullText: `[Error: ${err.message}]`,
        done: true
      });
    }
  }

  if (type === 'HOT_RELOAD') {
    const { modelName } = payload;
    const jsonStr = await storage.getModel(modelName);
    if (jsonStr) {
      const dec = await JSONLLMCodec.importFromJSONLLM(jsonStr);
      engine.weights = dec.weights;
      activeModelName = modelName;
      self.postMessage({ type: 'STATUS', status: `Ready: ${modelName} (128M Loaded)` });
    }
  }
};
