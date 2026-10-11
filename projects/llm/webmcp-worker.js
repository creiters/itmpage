import { WebMCPServer } from './webmcp-server.js';
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
        text: `Epoch ${epoch}/${epochs} finished. Mean Loss: ${(epochLoss / sequences.length).toFixed(4)}`
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
    self.postMessage({ type: 'LOG', category: 'TRAIN', text: 'Training cancelled.' });
    return;
  }

  // Export current engine to JSONLLM and partition to SI-Radix segments
  if (type === 'EXPORT_JSONLLM') {
    if (!server) return;
    const modelName = payload?.modelName || 'browser-nano-llm';

    self.postMessage({ type: 'LOG', category: 'STORAGE', text: `Packing model to compressed JSONLLM container...` });
    const { jsonString, manifest, segments } = await JSONLLMCodec.exportToJSONLLM(server.engine, modelName);

    await server.storage.saveManifest(manifest);
    await server.storage.batchPutChunks(segments);

    self.postMessage({
      type: 'EXPORT_JSONLLM_COMPLETE',
      payload: {
        jsonString,
        manifest,
        segmentsCount: segments.length
      }
    });
    return;
  }

  // Import JSONLLM file and load into engine
  if (type === 'IMPORT_JSONLLM') {
    if (!server) return;
    try {
      self.postMessage({ type: 'LOG', category: 'STORAGE', text: 'Decoding and dequantizing JSONLLM file...' });
      const imported = await JSONLLMCodec.importFromJSONLLM(payload.jsonString);
      server.engine.loadFromJSONLLMPayload(imported);

      self.postMessage({
        type: 'IMPORT_JSONLLM_COMPLETE',
        payload: {
          config: imported.config,
          modelName: payload.modelName || 'custom-jsonllm'
        }
      });
    } catch (err) {
      self.postMessage({ type: 'ERROR', message: `Import failed: ${err.message}` });
    }
  }
};
