import { NanoTransformer } from './nano-engine.js';
import { StorageService } from './storage.js';
import { LocalRAG } from './rag-store.js';
import { JSONLLMCodec } from './jsonllm-codec.js';
import { WebMCPServer } from './webmcp-server.js';

let engine = null;
let storage = null;
let rag = null;
let server = null;

(async () => {
  storage = new StorageService();
  await storage.init();
  rag = new LocalRAG(64);
  await rag.init(storage);

  engine = new NanoTransformer({ dim: 64, layers: 2, heads: 2, vocabSize: 96, maxSeqLen: 48 });
  server = new WebMCPServer(engine, storage, rag, JSONLLMCodec);

  self.postMessage({ type: 'STATUS', status: 'WebMCP Ready' });
})();

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'WEBMCP_REQUEST') {
    const res = await server.handleRequest(payload);
    self.postMessage({ type: 'WEBMCP_RESPONSE', payload: res });
  }

  if (type === 'START_TRAIN') {
    const { text, epochs = 5, lr = 0.003 } = payload;
    const tokens = engine.tokenizer.encode(text);

    self.postMessage({ type: 'LOG', category: 'TRAIN', text: `Training ${tokens.length} tokens for ${epochs} epochs...` });

    let loss = 0;
    for (let ep = 1; ep <= epochs; ep++) {
      loss = engine.trainSequence(tokens, lr);

      self.postMessage({
        type: 'TRAIN_PROGRESS',
        payload: {
          epoch: ep,
          totalEpochs: epochs,
          loss,
          perplexity: Math.exp(Math.min(10, loss)),
          progressPct: Math.round((ep / epochs) * 100)
        }
      });
      await new Promise((r) => setTimeout(r, 0));
    }

    self.postMessage({
      type: 'TRAIN_DONE',
      loss: loss.toFixed(4),
      ppl: Math.exp(Math.min(10, loss)).toFixed(2)
    });
  }

  if (type === 'GENERATE_JSONLLM') {
    const modelName = payload?.modelName || 'agent-dialog';
    const jsonString = await JSONLLMCodec.exportToJSONLLM(
      { dim: engine.dim, layers: engine.layers, heads: engine.heads, vocabSize: engine.vocabSize, maxSeqLen: engine.maxSeqLen },
      engine.tokenizer,
      engine.weights
    );

    await storage.putModel(modelName, jsonString);

    self.postMessage({
      type: 'GENERATE_JSONLLM_SUCCESS',
      payload: {
        modelName,
        jsonString,
        sizeKb: (jsonString.length / 1024).toFixed(1)
      }
    });
  }

  if (type === 'IMPORT_JSONLLM') {
    const { jsonString, modelName } = payload;
    const dec = await JSONLLMCodec.importFromJSONLLM(jsonString);
    engine.dim = dec.config.dim;
    engine.layers = dec.config.layers;
    engine.weights = dec.weights;
    await storage.putModel(modelName, jsonString);

    self.postMessage({
      type: 'IMPORT_JSONLLM_SUCCESS',
      payload: { modelName, sizeKb: (jsonString.length / 1024).toFixed(1) }
    });
  }
};
