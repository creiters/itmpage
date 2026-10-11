/**
 * webmcp-worker.js - On-Device Training & WebMCP Server Worker
 */

import { NanoTransformer } from './nano-engine.js';
import { StorageService } from './storage.js';
import { LocalRAG } from './rag-store.js';
import { ChunkIndexer } from './chunk-indexer.js';
import { JSONLLMCodec } from './jsonllm-codec.js';
import { WebMCPServer } from './webmcp-server.js';
import { DOMAIN_SPECS, DomainTokenizer } from './domain-specs.js';

let engine = null;
let storage = null;
let rag = null;
let server = null;

(async () => {
  const spec = DOMAIN_SPECS.dialog;
  const tokenizer = new DomainTokenizer(spec);
  engine = new NanoTransformer({ ...spec, vocabSize: tokenizer.vocabSize });

  storage = new StorageService();
  await storage.init();
  rag = new LocalRAG(64);
  await rag.init(storage);

  server = new WebMCPServer(engine, storage, rag, ChunkIndexer, JSONLLMCodec);
  self.postMessage({ type: 'STATUS', status: 'WebMCP Server Ready' });
})();

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'WEBMCP_REQUEST') {
    const res = await server.handleRequest(payload);
    self.postMessage({ type: 'WEBMCP_RESPONSE', payload: res });
  }

  if (type === 'TRAIN_CORPUS') {
    const { domain, text, epochs = 5, lr = 0.003 } = payload;
    const spec = DOMAIN_SPECS[domain] || DOMAIN_SPECS.dialog;
    const tok = new DomainTokenizer(spec);
    engine.swapWeights(spec, engine.weights);

    const tokens = tok.encode(text);
    let loss = 0;

    for (let ep = 1; ep <= epochs; ep++) {
      loss = engine.trainStep(tokens, lr);
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
      await new Promise((r) => setTimeout(r, 0)); // Yield to event loop
    }

    const jsonStr = await JSONLLMCodec.exportToJSONLLM(spec, engine.weights);
    const slices = ChunkIndexer.slicePayload(jsonStr, spec.modelName);
    await storage.batchSaveSegments(slices);
    await storage.putModel(spec.modelName, jsonStr);

    self.postMessage({
      type: 'TRAIN_DONE',
      loss: loss.toFixed(4),
      ppl: Math.exp(Math.min(10, loss)).toFixed(2),
      sizeKb: (jsonStr.length / 1024).toFixed(1)
    });
  }

  if (type === 'EXPORT_JSONLLM') {
    const { modelName } = payload;
    const jsonString = await storage.getModel(modelName);
    self.postMessage({
      type: 'EXPORT_JSONLLM_COMPLETE',
      payload: { modelName, jsonString }
    });
  }

  if (type === 'IMPORT_JSONLLM') {
    const { jsonString, modelName } = payload;
    const decoded = await JSONLLMCodec.importFromJSONLLM(jsonString);
    await storage.putModel(modelName, jsonString);
    const slices = ChunkIndexer.slicePayload(jsonString, modelName);
    await storage.batchSaveSegments(slices);

    self.postMessage({
      type: 'IMPORT_JSONLLM_COMPLETE',
      payload: { modelName, config: decoded.config }
    });
  }
};
