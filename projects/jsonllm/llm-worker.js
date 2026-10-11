import { NanoTransformer } from './nano-engine.js';
import { JSONLLMCodec } from './jsonllm-codec.js';
import { StorageService } from './storage.js';
import { LocalRAG } from './rag-store.js';

let storage = null;
let rag = null;
let engine = null;
let activeModelName = 'agent-dialog';

(async () => {
  storage = new StorageService();
  await storage.init();
  rag = new LocalRAG(64);
  await rag.init(storage);

  engine = new NanoTransformer({ dim: 64, layers: 2, heads: 2, vocabSize: 96, maxSeqLen: 48 });

  // Load active model if exists in IndexedDB
  const cached = await storage.getModel(activeModelName);
  if (cached) {
    const dec = await JSONLLMCodec.importFromJSONLLM(cached);
    engine.dim = dec.config.dim;
    engine.layers = dec.config.layers;
    engine.weights = dec.weights;
    self.postMessage({ type: 'STATUS', status: `Ready: ${activeModelName}` });
  } else {
    self.postMessage({ type: 'STATUS', status: 'Ready: Standalone' });
  }
})();

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'INFER') {
    const { prompt, maxTokens = 32 } = payload;
    const tStart = performance.now();

    // Query RAG context
    const matches = await rag.retrieve(prompt, 1);
    let augmented = prompt;
    if (matches.length > 0 && matches[0].score > 0.15) {
      augmented = `[Fact: ${matches[0].text.slice(0, 40)}] ${prompt}`;
      self.postMessage({ type: 'LOG', category: 'INFER', text: `RAG Context Ingested: "${matches[0].text}"` });
    }

    const tokens = engine.tokenizer.encode(augmented);
    const gen = [...tokens];
    let full = '';

    const kvCache = {
      k: Array.from({ length: engine.layers }, () => []),
      v: Array.from({ length: engine.layers }, () => [])
    };

    for (let t = 0; t < maxTokens; t++) {
      const curTok = gen[gen.length - 1];
      const pos = gen.length - 1;

      const { logits } = engine.forwardStepWithTape(curTok, pos, kvCache);

      // Repetition penalty
      const recent = gen.slice(-6);
      for (const id of recent) {
        if (id > 3) logits[id] -= 2.5;
      }

      // Collect top-3 candidates for telemetry visualization
      const scored = [];
      for (let i = 4; i < engine.vocabSize; i++) {
        scored.push({ word: engine.tokenizer.decode([i]) || `#${i}`, logit: logits[i] });
      }
      scored.sort((a, b) => b.logit - a.logit);
      const topCandidates = scored.slice(0, 3).map((c) => ({
        word: c.word,
        logit: c.logit.toFixed(2)
      }));

      const best = scored[0] ? engine.tokenizer.encode(scored[0].word)[1] : 2;
      if (best === 2) break; // <eos>

      gen.push(best);
      const piece = engine.tokenizer.decode([best]);
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
        done: t === maxTokens - 1
      });

      await new Promise((r) => setTimeout(r, 12));
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
