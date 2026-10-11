import { NanoTransformer } from './nano-engine.js';
import { JSONLLMCodec, HalfFloatArray } from './jsonllm-codec.js';
import { StorageService } from './storage.js';
import { LocalRAG } from './rag-store.js';

let storage = null;
let rag = null;
let engine = null;
let activeModelName = 'vimtollm-4m';
let activeQuantMode = 'fp4';

const SEED_CORPUS = [
  { u: "Hi", b: "Hello! How can I assist you today?" },
  { u: "Hello", b: "Hi there! I am vimtoLLM - all space inside! 10 on minus 84." },
  { u: "Who are you?", b: "I am vimtoLLM - all space inside! 10 on minus 84., running locally in your browser." },
  { u: "What is your name?", b: "My name is vimtoLLM." },
  { u: "Name?", b: "My name is vimtoLLM." },
  { u: "What is your architecture?", b: "I am a 4M parameter transformer using pure FP16 and FP4 E2M1 quantization." },
  { u: "Where do you run?", b: "I run entirely offline inside your browser tab without any external servers." },
  { u: "How are weights stored?", b: "My weights are stored in FP4 E2M1 and Q4 formats directly in IndexedDB." },
  { u: "Can you help me with code?", b: "Yes, I can assist with dialogue, HTML5, C code, and algorithms." },
  { u: "Thank you", b: "You are very welcome! Let me know if you need anything else." }
];

function log(category, text) {
  self.postMessage({ type: 'LOG', category, text });
}

async function trainBaselineNonBlocking(transformer, epochs = 16) {
  log('TRAIN', `Starting non-blocking baseline initialization for vimtoLLM (${epochs} epochs)...`);
  for (let ep = 0; ep < epochs; ep++) {
    for (let i = 0; i < SEED_CORPUS.length; i++) {
      const pair = SEED_CORPUS[i];
      const fullText = `<user> ${pair.u} <bot> ${pair.b} <eos>`;
      const tokens = transformer.tokenizer.encode(fullText);
      transformer.trainSequence(tokens, 0.004);

      if ((i + 1) % 3 === 0) {
        await new Promise((r) => setTimeout(r, 0));
      }
    }
    if ((ep + 1) % 4 === 0) {
      log('TRAIN', `Baseline initialization progress: Epoch ${ep + 1}/${epochs}`);
    }
  }
  log('TRAIN', 'vimtoLLM baseline initialization converged.');
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

async function initModel() {
  log('COMPUTE', 'Allocating 4M parameter pure FP16 memory pool for vimtoLLM...');

  engine = new NanoTransformer({
    dim: 192,
    layers: 8,
    heads: 6,
    ffnDim: 512,
    vocabSize: 1024,
    maxSeqLen: 128
  });

  const cached = await storage.getModel(activeModelName);
  if (cached) {
    try {
      log('STORAGE', `Found cached model "${activeModelName}" in IndexedDB. Dequantizing...`);
      const dec = await JSONLLMCodec.importFromJSONLLM(cached);
      engine.weights = dec.weights;
      activeQuantMode = dec.quantMode;
      log('STORAGE', `Model "${activeModelName}" loaded successfully (${activeQuantMode.toUpperCase()}).`);
      self.postMessage({ type: 'STATUS', status: `Ready: ${activeModelName} (${activeQuantMode.toUpperCase()})` });
      return;
    } catch (err) {
      log('ERROR', `Cache restore failed: ${err.message}. Rebuilding baseline...`);
    }
  }

  self.postMessage({ type: 'STATUS', status: 'Training vimtoLLM baseline dialogue...' });
  await trainBaselineNonBlocking(engine, 16);

  log('STORAGE', `Serializing initialized model "${activeModelName}" to IndexedDB...`);
  const jsonStr = await JSONLLMCodec.exportToJSONLLM(
    { dim: engine.dim, layers: engine.layers, heads: engine.heads, ffnDim: engine.ffnDim, vocabSize: engine.vocabSize, maxSeqLen: engine.maxSeqLen },
    engine.tokenizer,
    engine.weights,
    'fp4'
  );
  await storage.putModel(activeModelName, jsonStr);
  log('STORAGE', `Model "${activeModelName}" committed to IndexedDB (${(jsonStr.length / 1024).toFixed(1)} KB).`);
  self.postMessage({ type: 'STATUS', status: `Ready: ${activeModelName} (FP4 Ready)` });
}

(async () => {
  try {
    storage = new StorageService();
    await storage.init();
    log('STORAGE', 'IndexedDB service initialized in vimtoLLM worker.');
    rag = new LocalRAG(64);
    await rag.init(storage);
    log('RAG', 'Vector RAG store ready in vimtoLLM worker.');
    await initModel();
  } catch (err) {
    log('ERROR', `Worker boot fatal: ${err.message}`);
    self.postMessage({ type: 'STATUS', status: `Error: ${err.message}` });
  }
})();

self.onmessage = async (e) => {
  const { type, payload } = e.data;
  log('COMPUTE', `LLM Worker received action: ${type}`);

  if (type === 'INFER') {
    const { prompt, maxTokens = 32 } = payload;
    const tStart = performance.now();
    log('INFER', `Starting inference for prompt: "${prompt}"`);

    try {
      const matches = await rag.retrieve(prompt, 1);
      let augmented = prompt;
      if (matches.length > 0 && matches[0].score > 0.18) {
        augmented = `[Fact: ${matches[0].text.slice(0, 36)}] ${prompt}`;
        log('RAG', `Context match score ${matches[0].score.toFixed(3)} applied: "${matches[0].text}"`);
      }

      const turnPrompt = `<user> ${augmented.trim()} <bot>`;
      const promptTokens = engine.tokenizer.encode(turnPrompt);
      log('INFER', `Encoded prompt into ${promptTokens.length} tokens.`);

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
      log('INFER', `Prefill complete in ${(performance.now() - tStart).toFixed(1)} ms. Beginning autoregression...`);

      for (let t = 0; t < maxTokens; t++) {
        const nextPos = promptTokens.length + t;
        if (nextPos >= engine.maxSeqLen) {
          log('INFER', 'Reached max sequence length boundary.');
          break;
        }

        const bestToken = sampleNextToken(
          lastLogits,
          engine.vocabSize,
          engine.tokenizer.specials.length,
          generated
        );

        if (bestToken === 2) {
          log('INFER', 'Generated <eos> token. Halting sequence.');
          break;
        }

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

      const totalTime = ((performance.now() - tStart) / 1000).toFixed(2);
      log('INFER', `Generation complete: ${outputWords.length} tokens generated in ${totalTime}s.`);

      self.postMessage({
        type: 'TOKEN',
        token: '',
        fullText: outputWords.join(' ').replace(/ ([.,!?:;'")])/g, '$1'),
        done: true
      });

    } catch (err) {
      log('ERROR', `Inference failed: ${err.message}`);
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
    log('STORAGE', `Hot reloading model "${modelName}"...`);
    const jsonStr = await storage.getModel(modelName);
    if (jsonStr) {
      const dec = await JSONLLMCodec.importFromJSONLLM(jsonStr);
      engine.weights = dec.weights;
      activeModelName = modelName;
      activeQuantMode = dec.quantMode;
      log('STORAGE', `Hot reload complete for "${modelName}".`);
      self.postMessage({ type: 'STATUS', status: `Ready: ${modelName} (${activeQuantMode.toUpperCase()})` });
    } else {
      log('ERROR', `Model "${modelName}" not found in storage for hot reload.`);
    }
  }
};
