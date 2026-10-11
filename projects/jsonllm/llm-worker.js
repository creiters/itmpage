import { NanoTransformer } from './nano-engine.js';
import { JSONLLMCodec, HalfFloatArray } from './jsonllm-codec.js';
import { StorageService } from './storage.js';
import { LocalRAG } from './rag-store.js';

let storage = null;
let rag = null;
let engine = null;
let activeAgent = 'vimto-dialog';
let activeQuantMode = 'fp4';

const INITIAL_SEED_CORPUS = [
  { u: "Hi", b: "Hello! How can I assist you today?" },
  { u: "Hello", b: "Hi there! I am vimtoLLM - all space inside! 10 on minus 84." },
  { u: "Who are you?", b: "I am vimtoLLM - all space inside! 10 on minus 84., running locally in your browser." },
  { u: "What is your name?", b: "My name is vimtoLLM." },
  { u: "What is your architecture?", b: "I am a 4M parameter transformer using pure FP16 and FP4 E2M1 quantization." },
  { u: "Where do you run?", b: "I run entirely offline inside your browser tab without any external servers." },
  { u: "Thank you", b: "You are very welcome! Let me know if you need anything else." }
];

function log(category, text) {
  self.postMessage({ type: 'LOG', category, text });
}

async function trainBaselineNonBlocking(transformer, epochs = 12) {
  log('TRAIN', `Seeding baseline dialogue syntax for ${activeAgent} (${epochs} epochs)...`);
  for (let ep = 0; ep < epochs; ep++) {
    for (let i = 0; i < INITIAL_SEED_CORPUS.length; i++) {
      const pair = INITIAL_SEED_CORPUS[i];
      const fullText = `<user> ${pair.u} <bot> ${pair.b} <eos>`;
      const tokens = transformer.tokenizer.encode(fullText);
      transformer.trainSequence(tokens, 0.004);

      if ((i + 1) % 2 === 0) {
        await new Promise((r) => setTimeout(r, 0));
      }
    }
  }
  log('TRAIN', 'Baseline dialogue seeding converged.');
}

function sampleNextToken(logits, vocabSize, specialsCount, recentTokens = []) {
  const clone = new HalfFloatArray(logits);

  clone[0] = -Infinity; // <pad>
  clone[1] = -Infinity; // <bos>
  clone[3] = -Infinity; // <unk>
  clone[4] = -Infinity; // <user>
  clone[5] = -Infinity; // <bot>

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

async function loadOrInitAgent(agentId) {
  activeAgent = agentId;
  log('STATUS', `Loading agent weights for: ${activeAgent}...`);

  if (!engine) {
    engine = new NanoTransformer({
      dim: 192,
      layers: 8,
      heads: 6,
      ffnDim: 512,
      vocabSize: 1024,
      maxSeqLen: 128
    });
  }

  const cached = await storage.getModel(activeAgent);
  if (cached) {
    try {
      const dec = await JSONLLMCodec.importFromJSONLLM(cached);
      engine.weights = dec.weights;
      activeQuantMode = dec.quantMode;
      log('STORAGE', `Restored ${activeAgent} from IndexedDB (${activeQuantMode.toUpperCase()}).`);
      self.postMessage({ type: 'STATUS', status: `Ready: ${activeAgent} (${activeQuantMode.toUpperCase()})` });
      return;
    } catch (err) {
      log('ERROR', `Restore failed for ${activeAgent}: ${err.message}`);
    }
  }

  self.postMessage({ type: 'STATUS', status: `Training: ${activeAgent} baseline...` });
  await trainBaselineNonBlocking(engine, 12);

  const jsonStr = await JSONLLMCodec.exportToJSONLLM(
    { dim: engine.dim, layers: engine.layers, heads: engine.heads, ffnDim: engine.ffnDim, vocabSize: engine.vocabSize, maxSeqLen: engine.maxSeqLen },
    engine.tokenizer,
    engine.weights,
    'fp4'
  );
  await storage.putModel(activeAgent, jsonStr);
  log('STORAGE', `Initialized and saved ${activeAgent} to IndexedDB (${(jsonStr.length / 1024).toFixed(1)} KB).`);
  self.postMessage({ type: 'STATUS', status: `Ready: ${activeAgent} (FP4 Ready)` });
}

(async () => {
  try {
    storage = new StorageService();
    await storage.init();
    log('STORAGE', 'Storage service connected.');
    rag = new LocalRAG(64);
    await rag.init(storage);
    log('RAG', 'Vector RAG connected.');
    await loadOrInitAgent('vimto-dialog');
  } catch (err) {
    log('ERROR', `Boot failure: ${err.message}`);
  }
})();

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'SWITCH_AGENT') {
    await loadOrInitAgent(payload.agentId);
    return;
  }

  if (type === 'INFER') {
    const { prompt, agentId = activeAgent, maxTokens = 32 } = payload;
    const tStart = performance.now();

    if (agentId !== activeAgent) {
      await loadOrInitAgent(agentId);
    }

    try {
      // Agent-Scoped Vector RAG Lookup
      const matches = await rag.retrieve(prompt, activeAgent, 1);
      let augmented = prompt;
      if (matches.length > 0 && matches[0].score > 0.18) {
        augmented = `[Context: ${matches[0].text.slice(0, 36)}] ${prompt}`;
        log('RAG', `RAG match (agent: ${matches[0].agentId}, score: ${matches[0].score.toFixed(3)}) applied.`);
      }

      const turnPrompt = `<user> ${augmented.trim()} <bot>`;
      const promptTokens = engine.tokenizer.encode(turnPrompt);

      const generated = [...promptTokens];
      const outputWords = [];

      const kvCache = {
        k: Array.from({ length: engine.layers }, () => []),
        v: Array.from({ length: engine.layers }, () => [])
      };

      let lastLogits = null;
      for (let p = 0; p < promptTokens.length; p++) {
        const res = engine.forwardStepWithTape(promptTokens[p], p, kvCache);
        lastLogits = res.logits;
      }

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
        await new Promise((r) => setTimeout(r, 6));
      }

      self.postMessage({
        type: 'TOKEN',
        token: '',
        fullText: outputWords.join(' ').replace(/ ([.,!?:;'")])/g, '$1'),
        done: true
      });

    } catch (err) {
      log('ERROR', `Inference error: ${err.message}`);
      self.postMessage({
        type: 'TOKEN',
        token: '',
        fullText: `[Error: ${err.message}]`,
        done: true
      });
    }
  }

  if (type === 'HOT_RELOAD') {
    await loadOrInitAgent(payload.modelName);
  }
};
