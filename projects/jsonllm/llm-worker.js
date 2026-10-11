import { NanoTransformer } from './nano-engine.js';
import { JSONLLMCodec } from './jsonllm-codec.js';
import { StorageService } from './storage.js';
import { LocalRAG } from './rag-store.js';

let storage = null;
let rag = null;
let engine = null;
let activeModelName = 'agent-dialog';

// Default baseline corpus to train on startup if no model is stored
const DEFAULT_SEED_CORPUS = 
  "<user> Hi <bot> Hello! How can I help you today? <eos> " +
  "<user> Who are you? <bot> I am your offline local JSONLLM model. <eos> " +
  "<user> How are you? <bot> I am operating offline in your browser. <eos>";

(async () => {
  storage = new StorageService();
  await storage.init();
  rag = new LocalRAG(64);
  await rag.init(storage);

  // Initialize micro-transformer with 128 vocab slots
  engine = new NanoTransformer({ dim: 64, layers: 2, heads: 2, vocabSize: 128, maxSeqLen: 64 });

  // Load active model if exists in IndexedDB, or warm up default weights
  const cached = await storage.getModel(activeModelName);
  if (cached) {
    try {
      const dec = await JSONLLMCodec.importFromJSONLLM(cached);
      engine.dim = dec.config.dim;
      engine.layers = dec.config.layers;
      engine.weights = dec.weights;
      self.postMessage({ type: 'STATUS', status: `Ready: ${activeModelName}` });
    } catch (err) {
      self.postMessage({ type: 'LOG', category: 'ERROR', text: `Failed to load cache: ${err.message}` });
    }
  } else {
    // Warm up baseline weights so initial responses are coherent
    const tokens = engine.tokenizer.encode(DEFAULT_SEED_CORPUS);
    for (let e = 0; e < 4; e++) {
      engine.trainSequence(tokens, 0.003);
    }
    const jsonStr = await JSONLLMCodec.exportToJSONLLM(
      { dim: engine.dim, layers: engine.layers, heads: engine.heads, vocabSize: engine.vocabSize, maxSeqLen: engine.maxSeqLen },
      engine.tokenizer,
      engine.weights
    );
    await storage.putModel(activeModelName, jsonStr);
    self.postMessage({ type: 'STATUS', status: `Ready: ${activeModelName} (Initialized)` });
  }
})();

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'INFER') {
    const { prompt, maxTokens = 48 } = payload;
    const tStart = performance.now();

    try {
      if (!engine) {
        throw new Error('Engine not initialized yet.');
      }

      // 1. Query RAG context
      const matches = await rag.retrieve(prompt, 1);
      let augmented = prompt;
      if (matches.length > 0 && matches[0].score > 0.15) {
        augmented = `[Context: ${matches[0].text.slice(0, 40)}] ${prompt}`;
        self.postMessage({ type: 'LOG', category: 'INFER', text: `RAG Context Applied: "${matches[0].text}"` });
      }

      // Format prompt into standard dialogue turn
      const formattedInput = `<user> ${augmented} <bot> `;
      const tokens = engine.tokenizer.encode(formattedInput);
      const gen = [...tokens];
      let full = '';

      // Initialize persistent KV cache
      const kvCache = {
        k: Array.from({ length: engine.layers }, () => []),
        v: Array.from({ length: engine.layers }, () => [])
      };

      // 2. Prefill Phase: Run all prompt tokens to populate kvCache sequentially
      let currentLogits = null;
      for (let p = 0; p < tokens.length; p++) {
        const res = engine.forwardStepWithTape(tokens[p], p, kvCache);
        currentLogits = res.logits;
      }

      // 3. Autoregressive Generation Loop
      for (let t = 0; t < maxTokens; t++) {
        // Apply repetition penalty on recent tokens
        const recent = gen.slice(-8);
        for (const id of recent) {
          if (id > 3 && id < engine.vocabSize) {
            currentLogits[id] -= 2.2;
          }
        }

        // Top Candidate extraction for telemetry
        const scored = [];
        for (let i = 4; i < engine.vocabSize; i++) {
          scored.push({ id: i, word: engine.tokenizer.decode([i]) || String.fromCharCode(i), logit: currentLogits[i] });
        }
        scored.sort((a, b) => b.logit - a.logit);

        const topCandidates = scored.slice(0, 3).map((c) => ({
          word: c.word,
          logit: c.logit.toFixed(2)
        }));

        const bestToken = scored[0]?.id ?? 2;

        // Stop on <eos> (2) or role tag <user> (4)
        if (bestToken === 2 || bestToken === 4) {
          break;
        }

        gen.push(bestToken);
        const piece = engine.tokenizer.decode([bestToken]);
        full += piece;

        const elapsed = (performance.now() - tStart) / 1000;
        const tps = ((t + 1) / (elapsed || 0.001)).toFixed(1);

        self.postMessage({
          type: 'TELEMETRY',
          tps,
          ttft: Math.round(performance.now() - tStart),
          topCandidates
        });

        self.postMessage({
          type: 'TOKEN',
          token: piece,
          fullText: full,
          done: false
        });

        const nextPos = tokens.length + t;
        if (nextPos >= engine.maxSeqLen) break;

        // Forward pass with the newly generated token
        const nextRes = engine.forwardStepWithTape(bestToken, nextPos, kvCache);
        currentLogits = nextRes.logits;

        await new Promise((r) => setTimeout(r, 10)); // Cooperative micro-yield
      }

      self.postMessage({
        type: 'TOKEN',
        token: '',
        fullText: full,
        done: true
      });

    } catch (err) {
      self.postMessage({ type: 'LOG', category: 'ERROR', text: `Inference failed: ${err.message}` });
      self.postMessage({
        type: 'TOKEN',
        token: `\n[Error: ${err.message}]`,
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
      self.postMessage({ type: 'STATUS', status: `Ready: ${modelName}` });
    }
  }
};
