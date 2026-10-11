/**
 * webmcp-worker.js
 * Dedicated Worker for On-Device Sequence Training, WebMCP RPC, and Checkpoint Packaging
 */

import { NanoTransformer } from './nano-engine.js';
import { StorageService } from './storage.js';
import { LocalRAG } from './rag-store.js';
import { ChunkIndexer } from './chunk-indexer.js';
import { JSONLLMCodec } from './jsonllm-codec.js';
import { WebMCPServer } from './webmcp-server.js';

let engine = null;
let storage = null;
let rag = null;
let server = null;
let isTrainingActive = false;

(async () => {
  engine = new NanoTransformer({
    dim: 128,
    layers: 4,
    heads: 4,
    vocabSize: 1024,
    maxSeqLen: 64
  });

  storage = new StorageService();
  await storage.init();
  await storage.requestPersistence();

  rag = new LocalRAG(64);
  await rag.init(storage);

  server = new WebMCPServer(engine, storage, rag, ChunkIndexer, JSONLLMCodec);

  // Restore existing checkpoint from IndexedDB if available
  try {
    const existing = await JSONLLMCodec.loadFromIndexedDB(storage, 'browser-nano-llm');
    engine.loadFromJSONLLMPayload(existing);
    self.postMessage({ type: 'STATUS', status: 'Loaded Checkpoint from IndexedDB' });
  } catch {
    self.postMessage({ type: 'STATUS', status: 'Engine Initialized & Ready' });
  }
})();

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  // 1. WebMCP Protocol Handling
  if (type === 'WEBMCP_REQUEST') {
    if (!server) return;
    const response = await server.handleRequest(payload);
    self.postMessage({ type: 'WEBMCP_RESPONSE', payload: response });
    return;
  }

  // 2. Multi-Epoch Sliding-Window Dataset Training Loop
  if (type === 'START_DATASET_TRAINING') {
    const { datasetText, epochs = 5, lr = 0.001, seqLen = 48 } = payload;
    isTrainingActive = true;

    self.postMessage({
      type: 'LOG',
      category: 'TRAIN',
      text: `Starting training: ${datasetText.length} characters, ${epochs} epochs, window=${seqLen}, lr=${lr}`
    });

    const allTokens = engine.tokenizer.encode(datasetText);
    if (allTokens.length < 8) {
      self.postMessage({ type: 'ERROR', message: 'Dataset is too short. Provide at least 20 words.' });
      isTrainingActive = false;
      return;
    }

    // Window packaging with 25% overlap
    const sequences = [];
    const stride = Math.max(4, Math.floor(seqLen * 0.75));
    for (let i = 0; i < allTokens.length - seqLen; i += stride) {
      sequences.push(allTokens.slice(i, i + seqLen + 1));
    }
    if (sequences.length === 0) {
      sequences.push(allTokens);
    }

    const tStart = performance.now();
    let totalSteps = 0;

    for (let epoch = 1; epoch <= epochs && isTrainingActive; epoch++) {
      // Stochastic shuffle per epoch
      for (let i = sequences.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [sequences[i], sequences[j]] = [sequences[j], sequences[i]];
      }

      let epochLossSum = 0.0;
      let batchLossSum = 0.0;

      for (let b = 0; b < sequences.length && isTrainingActive; b++) {
        const seq = sequences[b];
        const stepLoss = engine.trainSequence(seq, lr);
        batchLossSum += stepLoss;
        epochLossSum += stepLoss;
        totalSteps++;

        // Dispatch progress telemetry every 4 steps
        if ((b + 1) % 4 === 0 || b === sequences.length - 1) {
          const avgLoss = batchLossSum / ((b % 4) + 1);
          const perplexity = Math.exp(Math.min(10, avgLoss));
          const elapsedSec = (performance.now() - tStart) / 1000;
          const tokensPerSec = ((totalSteps * seqLen) / (elapsedSec || 0.001)).toFixed(1);
          const progressPct = Math.round(
            (((epoch - 1) * sequences.length + (b + 1)) / (epochs * sequences.length)) * 100
          );

          self.postMessage({
            type: 'TRAIN_PROGRESS',
            payload: {
              epoch,
              totalEpochs: epochs,
              step: b + 1,
              totalStepsInEpoch: sequences.length,
              loss: avgLoss,
              perplexity,
              tokensPerSec,
              progressPct
            }
          });

          batchLossSum = 0.0;
          await new Promise((r) => setTimeout(r, 0)); // Prevent thread starvation
        }
      }

      self.postMessage({
        type: 'LOG',
        category: 'TRAIN',
        text: `Epoch ${epoch}/${epochs} completed. Mean Loss: ${(epochLossSum / sequences.length).toFixed(4)}`
      });
    }

    isTrainingActive = false;
    self.postMessage({
      type: 'TRAIN_FINISHED',
      payload: {
        totalSteps,
        elapsedSec: ((performance.now() - tStart) / 1000).toFixed(2)
      }
    });
    return;
  }

  // 3. Interrupt Training
  if (type === 'STOP_TRAINING') {
    isTrainingActive = false;
    self.postMessage({ type: 'LOG', category: 'TRAIN', text: 'Training interrupted by user.' });
    return;
  }

  // 4. Quantize, Pack, and Store to IndexedDB
  if (type === 'CONVERT_TO_JSONLLM_AND_STORE') {
    const modelName = payload?.modelName || 'browser-nano-llm';

    try {
      self.postMessage({ type: 'LOG', category: 'STORAGE', text: `Quantizing ${modelName} to 4-bit JSONLLM...` });
      const t0 = performance.now();

      const { jsonString, manifest, segments } = await JSONLLMCodec.convertAndStoreToIndexedDB(
        engine,
        storage,
        modelName
      );

      const elapsed = (performance.now() - t0).toFixed(1);
      self.postMessage({
        type: 'LOG',
        category: 'STORAGE',
        text: `Stored ${segments.length} SI-Radix segments in IndexedDB (${(jsonString.length / 1024).toFixed(1)} KB) in ${elapsed} ms`
      });

      self.postMessage({
        type: 'CONVERT_JSONLLM_SUCCESS',
        payload: {
          modelName,
          jsonString,
          manifest,
          segmentsCount: segments.length,
          compressedSizeBytes: jsonString.length
        }
      });
    } catch (err) {
      self.postMessage({ type: 'ERROR', message: `Checkpoint failed: ${err.message}` });
    }
    return;
  }

  // 5. Load Stored Model from IndexedDB
  if (type === 'LOAD_JSONLLM_FROM_INDEXEDDB') {
    const modelName = payload?.modelName || 'browser-nano-llm';
    try {
      const restored = await JSONLLMCodec.loadFromIndexedDB(storage, modelName);
      engine.loadFromJSONLLMPayload(restored);
      self.postMessage({
        type: 'STATUS',
        status: `Loaded: ${modelName}`
      });
      self.postMessage({
        type: 'LOAD_JSONLLM_SUCCESS',
        payload: { modelName, config: restored.config }
      });
    } catch (err) {
      self.postMessage({ type: 'ERROR', message: `Load failed: ${err.message}` });
    }
  }
};
