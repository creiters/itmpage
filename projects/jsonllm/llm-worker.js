/**
 * llm-worker.js - Word-Level Autoregressive Inference Worker
 */

import { NanoTransformer } from './nano-engine.js';
import { JSONLLMCodec } from './jsonllm-codec.js';
import { StorageService } from './storage.js';
import { LocalRAG } from './rag-store.js';

let storage = null;
let rag = null;
let engine = null;
let activeModelName = 'agent-dialog';

const SEED_PAIRS = [
  { u: "Hi", b: "Hello! How can I help you today?" },
  { u: "Hello", b: "Hi there! I am your offline local AI." },
  { u: "How are you?", b: "I am doing great and running offline in your browser." },
  { u: "What is your name?", b: "My name is JSONLLM." },
  { u: "Your name?", b: "My name is JSONLLM." },
  { u: "Name?", b: "My name is JSONLLM." },
  { u: "Who are you?", b: "I am an offline browser AI." },
  { u: "Can you help me?", b: "Yes, I can assist with dialogue, HTML, and C code." },
  { u: "My name is Sasha", b: "Nice to meet you Sasha! How can I help you?" },
  { u: "Sasha", b: "Hello Sasha! What are we working on today?" }
];

async function trainBaseline(transformer, epochs = 32) {
  for (let ep = 0; ep < epochs; ep++) {
    for (const pair of SEED_PAIRS) {
      const fullText = `<user> ${pair.u} <bot> ${pair.b} <eos>`;
      const tokens = transformer.tokenizer.encode(fullText);
      transformer.ensureVocabCapacity(transformer.tokenizer.vocabSize);
      transformer.trainSequence(tokens, 0.004);
    }
  }
}

function sampleNextToken(logits, vocabSize, specialsCount, recentTokens = []) {
  const clone = new Float32Array(logits);

  // Mask control tokens so they can never be selected into user output
  clone[0] = -Infinity; // <pad>
  clone[1] = -Infinity; // <bos>
  clone[3] = -Infinity; // <unk>
  clone[4] = -Infinity; // <user>
  clone[5] = -Infinity; // <bot>

  // Repetition penalty on recent word tokens
  for (const id of recentTokens.slice(-6)) {
    if (id >= specialsCount && id < vocabSize) {
      clone[id] -= 2.5;
    }
  }

  let maxL = -Infinity;
  for (let i = 2; i < vocabSize; i++) {
    if (clone[i] > maxL) maxL = clone[i];
  }

  const temp = 0.5; // Lower temperature = more precise grammar
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

  const kPool = candidates.slice(0, Math.min(3, candidates.length));
  let cumSum = 0.0;
  for (const c of kPool) cumSum += c.prob;

  let r = Math.random() * cumSum;
  for (const c of kPool) {
    r -= c.prob;
    if (r <= 0) return c.id;
  }
  return kPool[0]?.id || 2;
}

async function initFreshModel() {
  self.postMessage({ type: 'STATUS', status: 'Training baseline language weights...' });
  await trainBaseline(engine, 36);
  const jsonStr = await JSONLLMCodec.exportToJSONLLM(
    { dim: engine.dim, layers: engine.layers, heads: engine.heads, vocabSize: engine.vocabSize, maxSeqLen: engine.maxSeqLen },
    engine.tokenizer,
    engine.weights
  );
  await storage.putModel(activeModelName, jsonStr);
  self.postMessage({ type: 'STATUS', status: `Ready: ${activeModelName}` });
}

(async () => {
  storage = new StorageService();
  await storage.init();
  rag = new LocalRAG(64);
  await rag.init(storage);

  engine = new NanoTransformer({ dim: 64, layers: 2, heads: 2, maxSeqLen: 64 });

  const cached = await storage.getModel(activeModelName);
  if (cached) {
    try {
      const dec = await JSONLLMCodec.importFromJSONLLM(cached);
      engine.dim = dec.config.dim;
      engine.layers = dec.config.layers;
      engine.weights = dec.weights;
      if (dec.tokenizer) {
        engine.tokenizer.tokenToId = new Map(Object.entries(dec.tokenizer.tokenToId).map(([k, v]) => [k, Number(v)]));
        engine.tokenizer.idToToken = new Map(Object.entries(dec.tokenizer.idToToken).map(([k, v]) => [Number(k), v]));
        engine.tokenizer.vocabSize = engine.tokenizer.tokenToId.size;
        engine.vocabSize = engine.tokenizer.vocabSize;
      }
      self.postMessage({ type: 'STATUS', status: `Ready: ${activeModelName}` });
    } catch {
      await initFreshModel();
    }
  } else {
    await initFreshModel();
  }
})();

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'INFER') {
    const { prompt, maxTokens = 20 } = payload;
    const tStart = performance.now();

    try {
      const matches = await rag.retrieve(prompt, 1);
      let augmented = prompt;
      if (matches.length > 0 && matches[0].score > 0.18) {
        augmented = `[Context: ${matches[0].text.slice(0, 30)}] ${prompt}`;
        self.postMessage({ type: 'LOG', category: 'INFER', text: `RAG Context Applied: "${matches[0].text}"` });
      }

      const turnPrompt = `<user> ${augmented.trim()} <bot>`;
      const promptTokens = engine.tokenizer.encode(turnPrompt);
      engine.ensureVocabCapacity(engine.tokenizer.vocabSize);

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

      // 2. Generate autoregressively
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

        const currentText = outputWords.join(' ').replace(/ ([.,!?:;'])/g, '$1');

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
        await new Promise((r) => setTimeout(r, 16));
      }

      self.postMessage({
        type: 'TOKEN',
        token: '',
        fullText: outputWords.join(' ').replace(/ ([.,!?:;'])/g, '$1'),
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
      if (dec.tokenizer) {
        engine.tokenizer.tokenToId = new Map(Object.entries(dec.tokenizer.tokenToId).map(([k, v]) => [k, Number(v)]));
        engine.tokenizer.idToToken = new Map(Object.entries(dec.tokenizer.idToToken).map(([k, v]) => [Number(k), v]));
        engine.tokenizer.vocabSize = engine.tokenizer.tokenToId.size;
        engine.vocabSize = engine.tokenizer.vocabSize;
      }
      activeModelName = modelName;
      self.postMessage({ type: 'STATUS', status: `Ready: ${modelName}` });
    }
  }
};
