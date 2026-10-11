import { WebMCPServer } from './webmcp-server.js';
import { ChunkIndexer } from './chunk-indexer.js';
import { JSONLLMCodec } from './jsonllm-codec.js';

let server = null;
let isTrainingActive = false;

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'INIT_WEBMCP') {
    server = new WebMCPServer({ name: 'browser-native-llm', version: '1.0.0' });
    await server.init();
    self.postMessage({ type: 'STATUS', status: 'WebMCP JSONLLM Server Ready' });
    return;
  }

  if (type === 'WEBMCP_REQUEST') {
    if (!server) {
      self.postMessage({
        type: 'WEBMCP_RESPONSE',
        payload: { jsonrpc: '2.0', id: payload?.id, error: { code: -32000, message: 'Server not ready' } }
      });
      return;
    }
    const response = await server.handleMessage(payload);
    if (response) {
      self.postMessage({ type: 'WEBMCP_RESPONSE', payload: response });
    }
    return;
  }

  if (type === 'START_DATASET_TRAINING') {
    if (!server) return;
    const { datasetText, epochs = 3, batchSize = 16, lr = 0.001, seqLen = 48 } = payload;
    isTrainingActive = true;

    self.postMessage({
      type: 'LOG',
      category: 'TRAIN',
      text: `Training started: ${datasetText.length} characters, ${epochs} epochs, lr=${lr}.`
    });

    const tokenizer = server.engine.tokenizer;
    const allTokens = tokenizer.encode(datasetText);

    if (allTokens.length < 8) {
      self.postMessage({ type: 'ERROR', message: 'Dataset too small. Provide at least 20 words.' });
      return;
    }

    const sequences = [];
    const stride = Math.floor(seqLen * 0.75);
    for (let i = 0; i < allTokens.length - seqLen; i += stride) {
      sequences.push(allTokens.slice(i, i + seqLen + 1));
    }

    const tStart = performance.now();
    let totalSteps = 0;

    for (let epoch = 1; epoch <= epochs && isTrainingActive; epoch++) {
      for (let i = sequences.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [sequences[i], sequences[j]] = [sequences[j], sequences[i]];
      }

      let epochLoss = 0.0;
      let batchLoss = 0.0;

      for (let b = 0; b < sequences.length && isTrainingActive; b++) {
        const seq = sequences[b];
        const loss = server.engine.trainSequence(seq, lr);
        batchLoss += loss;
        epochLoss += loss;
        totalSteps++;

        if ((b + 1) % 5 === 0 || b === sequences.length - 1) {
          const avgLoss = batchLoss / ((b % 5) + 1);
          const perplexity = Math.exp(Math.min(10, avgLoss));
          const elapsedSec = (performance.now() - tStart) / 1000;
          const tokensPerSec = ((totalSteps * seqLen) / elapsedSec).toFixed(1);

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
              progressPct: (((epoch - 1) * sequences.length + (b + 1)) / (epochs * sequences.length) * 100).toFixed(1)
            }
          });

          batchLoss = 0.0;
          await new Promise((r) => setTimeout(r, 0));
        }
      }

      self.postMessage({
        type: 'LOG',
        category: 'TRAIN',
        text: `Epoch ${epoch}/${epochs} completed. Mean Loss: ${(epochLoss / sequences.length).toFixed(4)}`
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

  if (type === 'STOP_TRAINING') {
    isTrainingActive = false;
    self.postMessage({ type: 'LOG', category: 'TRAIN', text: 'Training interrupted.' });
    return;
  }

  /**
   * Converts in-memory weights to JSONLLM and stores directly into IndexedDB
   */
  if (type === 'CONVERT_TO_JSONLLM_AND_STORE') {
    if (!server) return;
    const modelName = payload?.modelName || 'browser-nano-llm';

    try {
      self.postMessage({ type: 'LOG', category: 'STORAGE', text: `Converting "${modelName}" to 4-bit compressed JSONLLM...` });
      const t0 = performance.now();

      const { jsonString, manifest, segments } = await JSONLLMCodec.convertAndStoreToIndexedDB(
        server.engine,
        server.storage,
        modelName
      );

      const elapsed = (performance.now() - t0).toFixed(1);
      self.postMessage({
        type: 'LOG',
        category: 'STORAGE',
        text: `JSONLLM conversion complete in ${elapsed} ms. Stored ${segments.length} SI-Radix 4MB segments in IndexedDB.`
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
      self.postMessage({ type: 'ERROR', message: `Conversion failed: ${err.message}` });
    }
    return;
  }

  /**
   * Restores an active model from an IndexedDB-stored JSONLLM record
   */
  if (type === 'LOAD_JSONLLM_FROM_INDEXEDDB') {
    if (!server) return;
    const modelName = payload?.modelName || 'browser-nano-llm';

    try {
      self.postMessage({ type: 'LOG', category: 'STORAGE', text: `Loading JSONLLM "${modelName}" from IndexedDB...` });
      const restored = await JSONLLMCodec.loadFromIndexedDB(server.storage, modelName);
      server.engine.loadFromJSONLLMPayload(restored);

      self.postMessage({
        type: 'STATUS',
        status: `Ready: ${modelName} (Loaded from IndexedDB JSONLLM)`
      });
      self.postMessage({
        type: 'LOAD_JSONLLM_SUCCESS',
        payload: {
          modelName,
          config: restored.config
        }
      });
    } catch (err) {
      self.postMessage({ type: 'ERROR', message: `Failed to load JSONLLM: ${err.message}` });
    }
  }
};
