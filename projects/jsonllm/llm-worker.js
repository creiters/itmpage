import { NanoTransformer } from './nano-engine.js';
import { JSONLLMCodec } from './jsonllm-codec.js';
import { StorageService } from './storage.js';
import { LocalRAG } from './rag-store.js';

let storage = null;
let rag = null;
let engine = null;
let activeModelName = 'agent-dialog';

// A rich conversational training corpus to initialize the weights into real language
const FOUNDATIONAL_CORPUS = [
  "<user> Hi <bot> Hello! How can I help you today? <eos>",
  "<user> Hello <bot> Hi there! I am your local JSON model. <eos>",
  "<user> Name? <bot> My name is JSONLLM. <eos>",
  "<user> What is your name? <bot> I am an offline browser AI. <eos>",
  "<user> How are you? <bot> I am doing great and running offline! <eos>",
  "<user> Who created you? <bot> I run entirely on your browser via JavaScript. <eos>",
  "<user> Time? <bot> Check your device clock for the exact time. <eos>",
  "<user> Can you code? <bot> Yes, I can assist with dialogue, C, and HTML. <eos>"
].join(" ");

// Nucleus & Temperature Sampling Kernel
function sampleToken(logits, temperature = 0.7, topK = 5, topP = 0.9, recentTokens = []) {
  const vocabSize = logits.length;
  const clone = new Float32Array(logits);

  // 1. Repetition Penalty on the last 8 tokens
  for (const id of recentTokens.slice(-8)) {
    if (id > 3 && id < vocabSize) {
      clone[id] -= 2.8;
    }
  }

  // 2. Temperature scaling
  const temp = Math.max(0.1, temperature);
  for (let i = 4; i < vocabSize; i++) {
    clone[i] /= temp;
  }

  // 3. Find max for numeric stability
  let maxLogit = -Infinity;
  for (let i = 4; i < vocabSize; i++) {
    if (clone[i] > maxLogit) maxLogit = clone[i];
  }

  // 4. Exponentiate & collect candidates
  const candidates = [];
  let sumExp = 0.0;
  for (let i = 4; i < vocabSize; i++) {
    const val = Math.exp(clone[i] - maxLogit);
    candidates.push({ id: i, val });
    sumExp += val;
  }

  // Normalize to probabilities
  for (const c of candidates) {
    c.prob = c.val / sumExp;
  }

  // Sort descending
  candidates.sort((a, b) => b.prob - a.prob);

  // 5. Top-K cutoff
  const kPool = candidates.slice(0, Math.min(topK, candidates.length));

  // 6. Top-P (nucleus) cutoff
  let cumSum = 0.0;
  const pPool = [];
  for (const c of kPool) {
    cumSum += c.prob;
    pPool.push(c);
    if (cumSum >= topP) break;
  }

  // 7. Stochastic roll
  let r = Math.random() * cumSum;
  for (const c of pPool) {
    r -= c.prob;
    if (r <= 0) return c.id;
  }

  return pPool[0]?.id || 2;
}

(async () => {
  storage = new StorageService();
  await storage.init();
  rag = new LocalRAG(64);
  await rag.init(storage);

  engine = new NanoTransformer({ dim: 64, layers: 2, heads: 2, vocabSize: 128, maxSeqLen: 64 });

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
    // Cold start: Train 18 epochs on the foundational corpus so it generates real English
    self.postMessage({ type: 'STATUS', status: 'Training initial baseline language weights...' });
    const tokens = engine.tokenizer.encode(FOUNDATIONAL_CORPUS);
    for (let e = 0; e < 18; e++) {
      engine.trainSequence(tokens, 0.0035);
    }

    const jsonStr = await JSONLLMCodec.exportToJSONLLM(
      { dim: engine.dim, layers: engine.layers, heads: engine.heads, vocabSize: engine.vocabSize, maxSeqLen: engine.maxSeqLen },
      engine.tokenizer,
      engine.weights
    );
    await storage.putModel(activeModelName, jsonStr);
    self.postMessage({ type: 'STATUS', status: `Ready: ${activeModelName}` });
  }
})();

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'INFER') {
    const { prompt, maxTokens = 40 } = payload;
    const tStart = performance.now();

    try {
      if (!engine) throw new Error('Engine not initialized.');

      // Check Vector RAG
      const matches = await rag.retrieve(prompt, 1);
      let augmented = prompt;
      if (matches.length > 0 && matches[0].score > 0.18) {
        augmented = `[Context: ${matches[0].text.slice(0, 40)}] ${prompt}`;
        self.postMessage({ type: 'LOG', category: 'INFER', text: `RAG Context Applied: "${matches[0].text}"` });
      }

      // Structure prompt to match trained conversational format
      const turnPrompt = `<user> ${augmented.trim()} <bot> `;
      const promptTokens = engine.tokenizer.encode(turnPrompt);
      const generated = [...promptTokens];
      let fullText = '';

      const kvCache = {
        k: Array.from({ length: engine.layers }, () => []),
        v: Array.from({ length: engine.layers }, () => [])
      };

      // 1. Prefill prompt tokens
      let currentLogits = null;
      for (let p = 0; p < promptTokens.length; p++) {
        const res = engine.forwardStepWithTape(promptTokens[p], p, kvCache);
        currentLogits = res.logits;
      }

      // 2. Sample autoregressively
      for (let t = 0; t < maxTokens; t++) {
        const nextPos = promptTokens.length + t;
        if (nextPos >= engine.maxSeqLen) break;

        const bestToken = sampleToken(currentLogits, 0.65, 4, 0.85, generated);

        // Stop on EOS (2), BOS (1), or next Turn Tag (4: <user>)
        if (bestToken === 2 || bestToken === 1 || bestToken === 4) {
          break;
        }

        generated.push(bestToken);
        const charPiece = engine.tokenizer.decode([bestToken]);

        // Guard against any control tag leak
        if (!charPiece.includes('<bot>') && !charPiece.includes('<user>') && !charPiece.includes('<eos>')) {
          fullText += charPiece;
        }

        const elapsed = (performance.now() - tStart) / 1000;
        const tps = ((t + 1) / (elapsed || 0.001)).toFixed(1);

        // Extract top-3 display candidates for the telemetry UI
        const topCandidates = [];
        for (let i = 4; i < Math.min(engine.vocabSize, 30); i++) {
          topCandidates.push({ word: engine.tokenizer.decode([i]) || String.fromCharCode(i), logit: currentLogits[i].toFixed(1) });
        }

        self.postMessage({
          type: 'TELEMETRY',
          tps,
          ttft: Math.round(performance.now() - tStart),
          topCandidates: topCandidates.slice(0, 3)
        });

        self.postMessage({
          type: 'TOKEN',
          token: charPiece,
          fullText: fullText.trimStart(),
          done: false
        });

        // Run forward pass for the next token
        const nextRes = engine.forwardStepWithTape(bestToken, nextPos, kvCache);
        currentLogits = nextRes.logits;

        await new Promise((r) => setTimeout(r, 10));
      }

      self.postMessage({
        type: 'TOKEN',
        token: '',
        fullText: fullText.trimStart(),
        done: true
      });

    } catch (err) {
      self.postMessage({ type: 'LOG', category: 'ERROR', text: `Inference failed: ${err.message}` });
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
      self.postMessage({ type: 'STATUS', status: `Ready: ${modelName}` });
    }
  }
};
