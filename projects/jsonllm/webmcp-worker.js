/**
 * webmcp-worker.js - 4M Training & WebMCP Server Worker with FP4/Q4 Serialization
 */

import { NanoTransformer } from './nano-engine.js';
import { StorageService } from './storage.js';
import { LocalRAG } from './rag-store.js';
import { JSONLLMCodec } from './jsonllm-codec.js';
import { WebMCPServer } from './webmcp-server.js';

let engine = null;
let storage = null;
let rag = null;
let server = null;
let isTrainingActive = false;

(async () => {
  storage = new StorageService();
  await storage.init();
  rag = new LocalRAG(64);
  await rag.init(storage);

  engine = new NanoTransformer({
    dim: 192,
    layers: 8,
    heads: 6,
    ffnDim: 512,
    vocabSize: 1024,
    maxSeqLen: 128
  });

  server = new WebMCPServer(engine, storage, rag, JSONLLMCodec);
  self.postMessage({ type: 'STATUS', status: 'WebMCP Ready (4M Pure FP16/FP4)' });
})();

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'WEBMCP_REQUEST') {
    const res = await server.handleRequest(payload);
    self.postMessage({ type: 'WEBMCP_RESPONSE', payload: res });
    return;
  }

  if (type === 'START_TRAIN') {
    const { text, epochs = 5, lr = 0.003, seqLen = 48 } = payload;
    isTrainingActive = true;

    const allTokens = engine.tokenizer.encode(text);
    if (allTokens.length < 4) {
      self.postMessage({ type: 'LOG', category: 'ERROR', text: 'Text too short (min 4 tokens).' });
      return;
    }

    const sequences = [];
    const stride = Math.max(4, Math.floor(seqLen * 0.6));
    for (let i = 0; i < allTokens.length - seqLen; i += stride) {
      sequences.push(allTokens.slice(i, i + seqLen + 1));
    }
    if (sequences.length === 0) sequences.push(allTokens);

    self.postMessage({
      type: 'LOG',
      category: 'TRAIN',
      text: `Training 4M Model on ${allTokens.length} tokens across ${sequences.length} batches (${epochs} epochs)...`
    });

    const tStart = performance.now();
    let totalSteps = 0;
    let epLossSum = 0.0;

    for (let ep = 1; ep <= epochs && isTrainingActive; ep++) {
      for (let i = sequences.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [sequences[i], sequences[j]] = [sequences[j], sequences[i]];
      }

      epLossSum = 0.0;
      for (let b = 0; b < sequences.length && isTrainingActive; b++) {
        const loss = engine.trainSequence(sequences[b], lr);
        epLossSum += loss;
        totalSteps++;

        if ((b + 1) % 2 === 0 || b === sequences.length - 1) {
          const avgLoss = epLossSum / (b + 1);
          const ppl = Math.exp(Math.min(10, avgLoss));
          const elapsedSec = (performance.now() - tStart) / 1000;
          const tokensPerSec = ((totalSteps * seqLen) / (elapsedSec || 0.001)).toFixed(1);
          const progressPct = Math.round(
            (((ep - 1) * sequences.length + (b + 1)) / (epochs * sequences.length)) * 100
          );

          self.postMessage({
            type: 'TRAIN_PROGRESS',
            payload: {
              epoch: ep,
              totalEpochs: epochs,
              loss: avgLoss,
              perplexity: ppl,
              tokensPerSec,
              progressPct
            }
          });
          await new Promise((r) => setTimeout(r, 0));
        }
      }
    }

    isTrainingActive = false;
    self.postMessage({
      type: 'TRAIN_DONE',
      loss: (epLossSum / sequences.length).toFixed(4),
      ppl: Math.exp(Math.min(10, epLossSum / sequences.length)).toFixed(2)
    });
    return;
  }

  if (type === 'GENERATE_JSONLLM') {
    const modelName = payload?.modelName || 'dialogue-4m';
    const mode = payload?.mode || 'fp4'; // 'fp4' or 'q4'

    self.postMessage({ type: 'LOG', category: 'STORAGE', text: `Quantizing 4M weights into ${mode.toUpperCase()} Deflate format...` });

    const jsonString = await JSONLLMCodec.exportToJSONLLM(
      { dim: engine.dim, layers: engine.layers, heads: engine.heads, ffnDim: engine.ffnDim, vocabSize: engine.vocabSize, maxSeqLen: engine.maxSeqLen },
      engine.tokenizer,
      engine.weights,
      mode
    );

    await storage.putModel(modelName, jsonString);

    self.postMessage({
      type: 'GENERATE_JSONLLM_SUCCESS',
      payload: {
        modelName,
        sizeKb: (jsonString.length / 1024).toFixed(1),
        mode
      }
    });
  }
};
