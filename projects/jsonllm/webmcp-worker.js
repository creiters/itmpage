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
  self.postMessage({ type: 'STATUS', status: 'WebMCP Ready' });
})();

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'WEBMCP_REQUEST') {
    const res = await server.handleRequest(payload);
    self.postMessage({ type: 'WEBMCP_RESPONSE', payload: res });
    return;
  }

  // Train with Strict JSON Array of Dialogue Turns
  if (type === 'START_TRAIN_JSON') {
    const { targetAgent = 'vimto-dialog', pairs, epochs = 5, lr = 0.003, seqLen = 48 } = payload;
    isTrainingActive = true;

    log('TRAIN', `Preparing JSON training for ${targetAgent} on ${pairs.length} dialogue turns...`);

    // Format strictly into sequence
    const formattedText = pairs.map((p) => `<user> ${p.u} <bot> ${p.b} <eos>`).join(' ');
    const allTokens = engine.tokenizer.encode(formattedText);

    if (allTokens.length < 4) {
      log('ERROR', 'Dataset too short (min 4 tokens).');
      return;
    }

    const sequences = [];
    const stride = Math.max(4, Math.floor(seqLen * 0.6));
    for (let i = 0; i < allTokens.length - seqLen; i += stride) {
      sequences.push(allTokens.slice(i, i + seqLen + 1));
    }
    if (sequences.length === 0) sequences.push(allTokens);

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
      targetAgent,
      loss: (epLossSum / sequences.length).toFixed(4),
      ppl: Math.exp(Math.min(10, epLossSum / sequences.length)).toFixed(2)
    });
    return;
  }

  // Save Agent-Scoped Compact Dialogue Dataset (< 2 KB)
  if (type === 'SAVE_COMPACT_DIALOGUE') {
    const { datasetName, agentId = 'vimto-dialog', pairs } = payload;
    log('STORAGE', `Compressing dialogue package "${datasetName}" for ${agentId}...`);
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

  // Generate and Save Model
  if (type === 'GENERATE_JSONLLM') {
    const { modelName = 'vimto-dialog', mode = 'fp4' } = payload;
    log('STORAGE', `Quantizing ${modelName} weights to ${mode.toUpperCase()}...`);

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
