/**
 * llm-worker.js - Small JSONLLM Inference & RAG Retrieval Engine
 */

import { DOMAIN_SPECS, DomainTokenizer } from './domain-specs.js';
import { NanoTransformer } from './nano-engine.js';
import { JSONLLMCodec } from './jsonllm-codec.js';
import { StorageService } from './storage.js';
import { LocalRAG } from './rag-store.js';

let storage = null;
let rag = null;
let currentDomain = 'dialog';
let tokenizer = null;
let engine = null;

// Intent classification heuristic
function classifyIntent(text) {
  if (/<(!doctype|html|div|p|body|head|span|button|main)/i.test(text)) return 'html5';
  if (/(#include|int\s+main|printf|void|char\*|struct\b)/.test(text)) return 'clang';
  return 'dialog';
}

async function switchModel(domain) {
  currentDomain = domain;
  const spec = DOMAIN_SPECS[domain];
  tokenizer = new DomainTokenizer(spec);

  const saved = await storage.getModel(spec.modelName);
  if (saved) {
    const dec = await JSONLLMCodec.importFromJSONLLM(saved);
    if (!engine) engine = new NanoTransformer({ ...spec, vocabSize: tokenizer.vocabSize });
    engine.swapWeights({ ...spec, vocabSize: tokenizer.vocabSize }, dec.weights);
  } else {
    engine = new NanoTransformer({ ...spec, vocabSize: tokenizer.vocabSize });
    const corpus = domain === 'dialog'
      ? "<user> Hello <bot> Greetings! I am your small browser model. <eos>"
      : domain === 'clang'
      ? "#include <stdio.h>\nint main() { printf(\"OK\"); return 0; }"
      : "<!DOCTYPE html><html><body><main></main></body></html>";
    const toks = tokenizer.encode(corpus);
    for (let e = 0; e < 4; e++) engine.trainStep(toks, 0.003);

    const jsonStr = await JSONLLMCodec.exportToJSONLLM(spec, engine.weights);
    await storage.putModel(spec.modelName, jsonStr);
  }
}

(async () => {
  storage = new StorageService();
  await storage.init();
  rag = new LocalRAG(64);
  await rag.init(storage);

  await rag.indexDocument('rag_init', 'Small browser-native models learn local syntax and dialogue.');
  await switchModel('dialog');

  self.postMessage({ type: 'STATUS', status: 'Ready: agent-dialog' });
})();

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'INFER') {
    const { prompt, maxTokens = 32 } = payload;
    const target = classifyIntent(prompt);

    if (target !== currentDomain) {
      await switchModel(target);
      self.postMessage({ type: 'DOMAIN_SWITCH', domain: target });
    }

    const matches = await rag.retrieve(prompt, 1);
    let augmented = prompt;
    if (matches.length > 0 && matches[0].score > 0.15) {
      augmented = `[Context: ${matches[0].text.slice(0, 50)}] ${prompt}`;
    }

    const tokens = tokenizer.encode(augmented);
    const gen = [...tokens];
    let full = '';

    for (let t = 0; t < maxTokens; t++) {
      const logits = engine.forward(gen.slice(-engine.maxSeqLen));

      let best = 3, maxL = -Infinity;
      for (let v = 3; v < engine.vocabSize; v++) {
        if (logits[v] > maxL) { maxL = logits[v]; best = v; }
      }

      if (best === 2) break; // <eos>
      gen.push(best);
      const piece = tokenizer.decode([best]);
      full += piece;

      self.postMessage({
        type: 'TOKEN',
        token: piece,
        fullText: full,
        done: t === maxTokens - 1
      });
    }
  }
};
