import { WebMCPServer } from './webmcp-server.js';
import { ChunkIndexer } from './chunk-indexer.js';

let server = null;
let isTrainingActive = false;

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'INIT_WEBMCP') {
    server = new WebMCPServer({ name: 'browser-native-llm', version: '1.0.0' });
    await server.init();
    self.postMessage({ type: 'STATUS', status: 'WebMCP Server Ready' });
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
      text: `Starting dataset training: ${datasetText.length} characters, ${epochs} epochs, batch size ${batchSize}.`
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

  if (type === 'SAVE_MODEL_TO_SI_RADIX') {
    if (!server) return;
    const modelBinary = server.engine.serializeToSegmentArray();
    const modelBlob = new Blob([modelBinary], { type: 'application/octet-stream' });
    const modelName = payload?.modelName || 'browser-nano-llm';

    self.postMessage({ type: 'LOG', category: 'STORAGE', text: 'Building SI-Radix Segment Interval Index...' });

    const { manifest, chunks } = await ChunkIndexer.splitAndIndex(
      modelBlob,
      modelName,
      (progress) => self.postMessage({ type: 'INDEX_PROGRESS', progress })
    );

    await server.storage.saveManifest(manifest);
    await server.storage.batchPutChunks(chunks);

    self.postMessage({
      type: 'SAVE_COMPLETE',
      payload: { manifest, chunksCount: chunks.length }
    });
  }
};
