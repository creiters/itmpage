/**
 * webmcp-worker.js - Concurrent Epoch Coroutines, WebMCP RPC & GC Scheduler
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
  self.postMessage({ type: 'STATUS', status: 'WebMCP Ready (Coroutine & GC Engine)' });
})();

/**
 * Generator Coroutine for a Single Training Epoch
 * Yields periodically to allow interleaved concurrency with other active epoch streams
 */
function* createEpochCoroutine(epochIndex, sequences, lr, onBatchComplete) {
  // Stochastic shuffle inside epoch stream
  const seqs = [...sequences];
  for (let i = seqs.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [seqs[i], seqs[j]] = [seqs[j], seqs[i]];
  }

  let epochLossSum = 0.0;
  for (let b = 0; b < seqs.length; b++) {
    const loss = engine.trainSequence(seqs[b], lr);
    epochLossSum += loss;

    onBatchComplete(epochIndex, b + 1, seqs.length, loss);

    // Yield control after every training sequence step
    yield {
      epoch: epochIndex,
      batch: b + 1,
      totalBatches: seqs.length,
      stepLoss: loss,
      meanLoss: epochLossSum / (b + 1)
    };
  }

  return { epoch: epochIndex, finalEpochLoss: epochLossSum / seqs.length };
}

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'WEBMCP_REQUEST') {
    const res = await server.handleRequest(payload);
    self.postMessage({ type: 'WEBMCP_RESPONSE', payload: res });
    return;
  }

  // Multi-Epoch Concurrent Training with Coroutines and GC
  if (type === 'START_TRAIN_JSON' || type === 'START_DATASET_TRAINING') {
    const targetAgent = payload.targetAgent || 'vimto-dialog';
    const epochs = payload.epochs || 5;
    const lr = payload.lr || 0.003;
    const concurrency = Math.min(epochs, payload.concurrency || 3); // Concurrent active epoch streams

    isTrainingActive = true;
    log('TRAIN', `Launching ${epochs} epochs (concurrency factor: ${concurrency}) with GC tracking...`);

    let pairs = payload.pairs;
    if (!pairs && payload.datasetText) {
      pairs = [{ u: "Context", b: payload.datasetText }];
    }

    const formattedText = pairs.map((p) => `<user> ${p.u} <bot> ${p.b} <eos>`).join(' ');
    const allTokens = engine.tokenizer.encode(formattedText);
    const seqLen = 48;
    const sequences = [];
    const stride = Math.max(4, Math.floor(seqLen * 0.6));

    for (let i = 0; i < allTokens.length - seqLen; i += stride) {
      sequences.push(allTokens.slice(i, i + seqLen + 1));
    }
    if (sequences.length === 0) sequences.push(allTokens);

    const tStart = performance.now();
    let totalCompletedSteps = 0;
    const totalRequiredSteps = epochs * sequences.length;
    let accumulatedLoss = 0.0;

    // Active coroutine scheduler pool
    const activeCoroutines = new Map();
    let nextEpochToLaunch = 1;

    // Fill initial concurrency pool
    while (nextEpochToLaunch <= epochs && activeCoroutines.size < concurrency) {
      const ep = nextEpochToLaunch++;
      activeCoroutines.set(
        ep,
        createEpochCoroutine(ep, sequences, lr, (epochIdx, b, totalB, loss) => {
          totalCompletedSteps++;
          accumulatedLoss += loss;
        })
      );
    }

    // Interleaved cooperative event loop
    while (activeCoroutines.size > 0 && isTrainingActive) {
      for (const [ep, coroutine] of Array.from(activeCoroutines.entries())) {
        const step = coroutine.next();

        if (step.done) {
          activeCoroutines.delete(ep);
          log('TRAIN', `Epoch ${ep}/${epochs} finished (Mean Loss: ${step.value.finalEpochLoss.toFixed(4)}).`);

          // Sweep garbage collector pools between finished epochs
          const freedBytes = gc.purgePools();
          if (freedBytes > 0) {
            log('STORAGE', `GC swept ${(freedBytes / 1024).toFixed(1)} KB idle scratchpad buffers.`);
          }

          // Launch next queued epoch into the pool
          if (nextEpochToLaunch <= epochs) {
            const nextEp = nextEpochToLaunch++;
            activeCoroutines.set(
              nextEp,
              createEpochCoroutine(nextEp, sequences, lr, (epochIdx, b, totalB, loss) => {
                totalCompletedSteps++;
                accumulatedLoss += loss;
              })
            );
          }
        }
      }

      // Telemetry emission
      const elapsedSec = (performance.now() - tStart) / 1000;
      const currentAvgLoss = totalCompletedSteps > 0 ? accumulatedLoss / totalCompletedSteps : 0.0;
      const tps = ((totalCompletedSteps * seqLen) / (elapsedSec || 0.001)).toFixed(1);
      const progressPct = Math.min(100, Math.round((totalCompletedSteps / totalRequiredSteps) * 100));

      self.postMessage({
        type: 'TRAIN_PROGRESS',
        payload: {
          epoch: Math.min(epochs, nextEpochToLaunch - 1),
          totalEpochs: epochs,
          activeConcurrentStreams: activeCoroutines.size,
          loss: currentAvgLoss,
          perplexity: Math.exp(Math.min(10, currentAvgLoss)),
          tokensPerSec: tps,
          progressPct
        }
      });

      // Cooperative micro-yield prevents thread starvation
      await new Promise((r) => setTimeout(r, 0));
    }

    isTrainingActive = false;
    const finalMeanLoss = totalCompletedSteps > 0 ? (accumulatedLoss / totalCompletedSteps).toFixed(4) : "0.0000";

    self.postMessage({
      type: 'TRAIN_DONE',
      targetAgent,
      loss: finalMeanLoss,
      ppl: Math.exp(Math.min(10, parseFloat(finalMeanLoss))).toFixed(2)
    });
    return;
  }

  // Interrupt Training
  if (type === 'STOP_TRAINING') {
    isTrainingActive = false;
    gc.purgePools();
    log('TRAIN', 'Training interrupted by user. Memory pools purged.');
    return;
  }

  // Save Compressed Dialogue Package
  if (type === 'SAVE_COMPACT_DIALOGUE') {
    const { datasetName, agentId = 'vimto-dialog', pairs } = payload;
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
    return;
  }

  // Quantize, Pack, and Store Checkpoint
  if (type === 'GENERATE_JSONLLM' || type === 'CONVERT_TO_JSONLLM_AND_STORE') {
    const modelName = payload?.modelName || 'vimtollm-4m';
    const mode = payload?.mode || 'fp4';

    log('STORAGE', `Quantizing ${modelName} into ${mode.toUpperCase()} Deflate envelope...`);

    const jsonString = await JSONLLMCodec.exportToJSONLLM(
      { dim: engine.dim, layers: engine.layers, heads: engine.heads, ffnDim: engine.ffnDim, vocabSize: engine.vocabSize, maxSeqLen: engine.maxSeqLen },
      engine.tokenizer,
      engine.weights,
      mode
    );

    await storage.putModel(modelName, jsonString);
    gc.purgePools();

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
