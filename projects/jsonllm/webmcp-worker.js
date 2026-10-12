/**
 * webmcp-worker.js - Concurrent Coroutine Training & WebMCP Server Worker
 */

import { NanoTransformer, gc } from './nano-engine.js';
import { StorageService } from './storage.js';
import { LocalRAG } from './rag-store.js';
import { JSONLLMCodec } from './jsonllm-codec.js';
import { WebMCPServer } from './webmcp-server.js';

let engine = null;
let storage = null;
let rag = null;
let server = null;
let isTrainingActive = false;

function log(category, text) {
  self.postMessage({ type: 'LOG', category, text });
}

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
  self.postMessage({ type: 'STATUS', status: 'WebMCP & Coroutine Scheduler Ready' });
})();

function* createEpochCoroutine(epochIdx, sequences, lr, onBatchComplete) {
  const seqs = [...sequences];
  for (let i = seqs.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [seqs[i], seqs[j]] = [seqs[j], seqs[i]];
  }

  let lossSum = 0.0;
  for (let b = 0; b < seqs.length; b++) {
    const loss = engine.trainSequence(seqs[b], lr);
    lossSum += loss;
    onBatchComplete(epochIdx, b + 1, seqs.length, loss);

    yield {
      epoch: epochIdx,
      batch: b + 1,
      totalBatches: seqs.length,
      stepLoss: loss,
      meanLoss: lossSum / (b + 1)
    };
  }
  return { epoch: epochIdx, finalLoss: lossSum / seqs.length };
}

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'WEBMCP_REQUEST') {
    const res = await server.handleRequest(payload);
    self.postMessage({ type: 'WEBMCP_RESPONSE', payload: res });
    return;
  }

  if (type === 'START_TRAIN_JSON') {
    const { targetAgent = 'miriam', pairs, epochs = 12, lr = 0.003, concurrency = 3 } = payload;
    isTrainingActive = true;

    log('TRAIN', `Launching ${epochs} epochs (concurrency: ${concurrency}) for ${targetAgent}...`);

    const formatted = pairs.map((p) => `<user> ${p.u} <bot> ${p.b} <eos>`).join(' ');
    const allTokens = engine.tokenizer.encode(formatted);

    const seqLen = 48;
    const sequences = [];
    const stride = Math.max(4, Math.floor(seqLen * 0.6));
    for (let i = 0; i < allTokens.length - seqLen; i += stride) {
      sequences.push(allTokens.slice(i, i + seqLen + 1));
    }
    if (sequences.length === 0) sequences.push(allTokens);

    const tStart = performance.now();
    let totalSteps = 0;
    const totalRequired = epochs * sequences.length;
    let accumulatedLoss = 0.0;

    const activeCoroutines = new Map();
    let nextEpoch = 1;

    while (nextEpoch <= epochs && activeCoroutines.size < concurrency) {
      const ep = nextEpoch++;
      activeCoroutines.set(
        ep,
        createEpochCoroutine(ep, sequences, lr, () => { totalSteps++; })
      );
    }

    while (activeCoroutines.size > 0 && isTrainingActive) {
      for (const [ep, routine] of Array.from(activeCoroutines.entries())) {
        const step = routine.next();
        if (step.value) accumulatedLoss += step.value.stepLoss;

        if (step.done) {
          activeCoroutines.delete(ep);
          gc.purgeIdlePools();

          if (nextEpoch <= epochs) {
            const nextEp = nextEpoch++;
            activeCoroutines.set(
              nextEp,
              createEpochCoroutine(nextEp, sequences, lr, () => { totalSteps++; })
            );
          }
        }
      }

      const elapsed = (performance.now() - tStart) / 1000;
      const currentAvgLoss = totalSteps > 0 ? accumulatedLoss / totalSteps : 0.0;
      const tps = ((totalSteps * seqLen) / (elapsed || 0.001)).toFixed(1);
      const progressPct = Math.min(100, Math.round((totalSteps / totalRequired) * 100));

      self.postMessage({
        type: 'TRAIN_PROGRESS',
        payload: {
          epoch: Math.min(epochs, nextEpoch - 1),
          totalEpochs: epochs,
          activeStreams: activeCoroutines.size,
          loss: currentAvgLoss,
          perplexity: Math.exp(Math.min(10, currentAvgLoss)),
          tokensPerSec: tps,
          progressPct
        }
      });

      await new Promise((r) => setTimeout(r, 0));
    }

    isTrainingActive = false;
    const finalLoss = totalSteps > 0 ? (accumulatedLoss / totalSteps).toFixed(4) : "0.0000";

    self.postMessage({
      type: 'TRAIN_DONE',
      targetAgent,
      loss: finalLoss,
      ppl: Math.exp(Math.min(10, parseFloat(finalLoss))).toFixed(2)
    });
  }

  if (type === 'STOP_TRAINING') {
    isTrainingActive = false;
    gc.purgeIdlePools();
    log('TRAIN', 'Training stream interrupted. Memory pools cleared.');
  }

  if (type === 'SAVE_COMPACT_DIALOGUE') {
    const { datasetName, agentId = 'miriam', pairs } = payload;
    const compactJSON = await JSONLLMCodec.exportDialogueDataset(engine.tokenizer, pairs);
    await storage.putDialogueDataset(datasetName, agentId, compactJSON);

    self.postMessage({
      type: 'SAVE_DIALOGUE_COMPLETE',
      payload: {
        datasetName,
        agentId,
        pairCount: pairs.length,
        bytes: compactJSON.length,
        sizeKb: (compactJSON.length / 1024).toFixed(2)
      }
    });
  }

  if (type === 'GENERATE_JSONLLM') {
    const { modelName = 'miriam', mode = 'fp4' } = payload;
    log('STORAGE', `Compiling ${modelName} into clean JSONLLM envelope...`);

    const jsonString = await JSONLLMCodec.exportToJSONLLM(
      { dim: engine.dim, layers: engine.layers, heads: engine.heads, ffnDim: engine.ffnDim, vocabSize: engine.vocabSize, maxSeqLen: engine.maxSeqLen },
      engine.tokenizer,
      engine.weights,
      mode
    );

    await storage.putModel(modelName, jsonString);
    gc.purgeIdlePools();

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
