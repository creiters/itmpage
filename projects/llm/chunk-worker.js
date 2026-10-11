import { ChunkIndexer } from './chunk-indexer.js';
import { StorageService } from './storage.js';

let storage = null;
let computeWorkerPort = null;
let activeManifest = null;
let activeModelName = '';

async function ensureStorage() {
  if (!storage) {
    storage = new StorageService();
    await storage.init();
  }
}

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'CONNECT_COMPUTE_PORT') {
    computeWorkerPort = e.ports[0];
    computeWorkerPort.onmessage = async (event) => {
      const req = event.data;
      if (req.type === 'REQUEST_SEGMENT_FOR_LAYER') {
        await handleLayerSegmentRequest(req.layerIndex, req.modelName);
      }
    };
    self.postMessage({ type: 'LOG', category: 'STORAGE', text: 'Fast Segment Port bound to Compute Worker.' });
    return;
  }

  await ensureStorage();

  if (type === 'SPLIT_AND_STORE_CHUNKS' || type === 'SPLIT_AND_STORE_SEGMENTS') {
    const { modelBlob, modelName } = payload;
    activeModelName = modelName;

    try {
      self.postMessage({ type: 'LOG', category: 'STORAGE', text: `Purging previous segments for ${modelName}...` });
      await storage.deleteModelChunks(modelName);

      self.postMessage({ type: 'LOG', category: 'STORAGE', text: 'Generating Segment Interval Map & Murmur3 Checksums...' });
      const t0 = performance.now();

      const { manifest, segments } = await ChunkIndexer.splitAndIndex(
        modelBlob,
        modelName,
        (progress) => self.postMessage({ type: 'CHUNK_PROGRESS', progress })
      );

      const indexElapsed = (performance.now() - t0).toFixed(1);
      self.postMessage({
        type: 'LOG',
        category: 'STORAGE',
        text: `Indexed ${segments.length} segments in ${indexElapsed} ms (${(modelBlob.size / 1024 / 1024).toFixed(1)} MB).`
      });

      await storage.saveManifest(manifest);
      await storage.batchPutChunks(segments);

      activeManifest = manifest;
      const unifiedBinary = ChunkIndexer.reassemble(segments, manifest);

      self.postMessage({
        type: 'SPLIT_COMPLETE',
        payload: {
          modelName,
          manifest,
          chunkCount: segments.length
        }
      });

      if (computeWorkerPort) {
        computeWorkerPort.postMessage({
          type: 'LOAD_MODEL_PAYLOAD',
          payload: {
            modelName,
            manifest,
            modelBinary: unifiedBinary
          }
        }, [unifiedBinary]);
      }
    } catch (err) {
      self.postMessage({ type: 'ERROR', message: `Segment indexing failed: ${err.message}` });
    }
  }

  if (type === 'LOAD_CACHED_CHUNKS' || type === 'LOAD_CACHED_SEGMENTS') {
    const { modelName } = payload;
    activeModelName = modelName;

    try {
      const manifest = await storage.getManifest(modelName);
      const segments = await storage.getModelChunks(modelName);

      if (!manifest || segments.length === 0) {
        self.postMessage({ type: 'CHUNKS_EMPTY', payload: { modelName } });
        return;
      }

      self.postMessage({
        type: 'LOG',
        category: 'STORAGE',
        text: `Verifying ${segments.length} segments via Fast Checksum Map...`
      });

      const t0 = performance.now();
      const unifiedBinary = ChunkIndexer.reassemble(segments, manifest);
      const verifyTime = (performance.now() - t0).toFixed(1);

      self.postMessage({
        type: 'LOG',
        category: 'STORAGE',
        text: `All ${segments.length} segments verified and reassembled in ${verifyTime} ms.`
      });

      activeManifest = manifest;

      self.postMessage({
        type: 'CHUNKS_READY',
        payload: {
          modelName,
          manifest,
          chunkCount: segments.length
        }
      });

      if (computeWorkerPort) {
        computeWorkerPort.postMessage({
          type: 'LOAD_MODEL_PAYLOAD',
          payload: {
            modelName,
            manifest,
            modelBinary: unifiedBinary
          }
        }, [unifiedBinary]);
      }
    } catch (err) {
      self.postMessage({ type: 'ERROR', message: `Segment verification failed: ${err.message}` });
    }
  }
};

async function handleLayerSegmentRequest(layerIndex, modelName) {
  if (!activeManifest || !computeWorkerPort) return;
  const segmentIndices = ChunkIndexer.findSegmentsByLayer(activeManifest, layerIndex);

  for (const idx of segmentIndices) {
    const chunkId = `${modelName}_seg_${idx}`;
    const segment = await storage.getChunk(chunkId);
    if (segment) {
      computeWorkerPort.postMessage({
        type: 'DELIVER_PAGED_CHUNK',
        payload: {
          index: idx,
          metadata: segment.metadata,
          binary: segment.binary
        }
      }, [segment.binary]);
    }
  }
}

