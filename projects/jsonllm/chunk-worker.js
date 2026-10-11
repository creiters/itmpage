import { StorageService } from './storage.js';
import { ChunkIndexer } from './chunk-indexer.js';

let storage = null;
let computePort = null;

(async () => {
  storage = new StorageService();
  await storage.init();
})();

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'CONNECT_COMPUTE_PORT') {
    computePort = e.ports[0];
    computePort.onmessage = async (pe) => {
      const { modelName, index } = pe.data;
      const segments = await storage.getSegmentsByModel(modelName);
      const target = segments.find((s) => s.index === index);
      if (target) {
        computePort.postMessage({ type: 'CHUNK_DATA', segment: target });
      }
    };
    return;
  }

  if (type === 'SLICE_AND_PERSIST') {
    const { payloadString, modelName } = payload;
    try {
      const segments = ChunkIndexer.slicePayload(payloadString, modelName);
      await storage.batchSaveSegments(segments);
      self.postMessage({
        type: 'SLICE_COMPLETE',
        payload: { modelName, count: segments.length }
      });
    } catch (err) {
      self.postMessage({ type: 'ERROR', message: err.message });
    }
  }
};
