import { MerkleChunker } from './merkle-chunker.js';
import { StorageService } from './storage.js';

let storage = null;
let computeWorkerPort = null;
let activeManifest = [];
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
      if (req.type === 'REQUEST_CHUNK_BY_INDEX') {
        await handleComputeChunkRequest(req.index, req.modelName);
      }
    };
    self.postMessage({ type: 'LOG', category: 'STORAGE', text: 'Direct MessageChannel linked to Compute Worker.' });
    return;
  }

  await ensureStorage();

  if (type === 'SPLIT_AND_STORE_CHUNKS') {
    const { modelBlob, modelName } = payload;
    activeModelName = modelName;

    try {
      // 1. Quota Pre-Check
      const quota = await storage.getStorageQuota();
      if (quota.quotaBytes > 0 && quota.usageBytes + modelBlob.size > quota.quotaBytes) {
        throw new Error(`Insufficient storage: ${quota.usageMB}MB used of ${quota.quotaMB}MB quota. Need ${((modelBlob.size)/(1024*1024)).toFixed(1)}MB.`);
      }

      self.postMessage({ type: 'LOG', category: 'STORAGE', text: `Storage check passed. Quota used: ${quota.percentUsed}% (${quota.usageMB}/${quota.quotaMB} MB).` });
      self.postMessage({ type: 'LOG', category: 'STORAGE', text: `Building 4MB Merkle tree for ${modelName}...` });

      const { rootHash, chunks } = await MerkleChunker.splitToMerkleChunks(
        modelBlob,
        modelName,
        (progress) => {
          self.postMessage({ type: 'CHUNK_PROGRESS', progress });
        }
      );

      await storage.setSystemKey(`${modelName}_merkle_root`, rootHash);

      // 2. Batch Transaction Write
      self.postMessage({ type: 'LOG', category: 'STORAGE', text: `Writing ${chunks.length} 4MB chunks via batch transaction...` });
      await storage.batchPutChunks(chunks, (completed, total) => {
        if (completed % 5 === 0 || completed === total) {
          self.postMessage({
            type: 'LOG',
            category: 'STORAGE',
            text: `Batch commit: stored ${completed}/${total} chunks`
          });
        }
      });

      activeManifest = chunks.map((c, i) => ({
        chunkId: c.metadata.chunkId,
        index: i,
        layerRange: c.metadata.context.layerRange,
        byteOffset: c.metadata.byteOffset,
        byteLength: c.binary.byteLength,
        hash: c.metadata.chunkHash
      }));

      const unifiedBinary = await MerkleChunker.reassemble(chunks, rootHash);

      self.postMessage({
        type: 'SPLIT_COMPLETE',
        payload: {
          modelName,
          rootHash,
          manifest: activeManifest,
          chunkCount: chunks.length
        }
      });

      if (computeWorkerPort) {
        computeWorkerPort.postMessage({
          type: 'LOAD_MODEL_PAYLOAD',
          payload: {
            modelName,
            manifest: activeManifest,
            modelBinary: unifiedBinary
          }
        }, [unifiedBinary]);
      }
    } catch (err) {
      self.postMessage({ type: 'ERROR', message: `Chunk worker failure: ${err.message}` });
    }
  }

  if (type === 'LOAD_CACHED_CHUNKS') {
    const { modelName } = payload;
    activeModelName = modelName;

    try {
      const chunks = await storage.getModelChunks(modelName);
      if (chunks.length === 0) {
        self.postMessage({ type: 'CHUNKS_EMPTY', payload: { modelName } });
        return;
      }

      const storedRoot = await storage.getSystemKey(`${modelName}_merkle_root`);
      self.postMessage({ type: 'LOG', category: 'STORAGE', text: `Validating Merkle DAG for ${chunks.length} cached chunks...` });

      const unifiedBinary = await MerkleChunker.reassemble(chunks, storedRoot);

      activeManifest = chunks.map((c, i) => ({
        chunkId: c.metadata.chunkId,
        index: i,
        layerRange: c.metadata.context.layerRange,
        byteOffset: c.metadata.byteOffset,
        byteLength: c.binary.byteLength,
        hash: c.metadata.chunkHash
      }));

      self.postMessage({
        type: 'CHUNKS_READY',
        payload: {
          modelName,
          rootHash: storedRoot,
          manifest: activeManifest,
          chunkCount: chunks.length
        }
      });

      if (computeWorkerPort) {
        computeWorkerPort.postMessage({
          type: 'LOAD_MODEL_PAYLOAD',
          payload: {
            modelName,
            manifest: activeManifest,
            modelBinary: unifiedBinary
          }
        }, [unifiedBinary]);
      }
    } catch (err) {
      self.postMessage({ type: 'ERROR', message: `Cache load failed: ${err.message}` });
    }
  }
};

async function handleComputeChunkRequest(index, modelName) {
  try {
    const chunkId = `${modelName}_chunk_4mb_${index}`;
    const chunk = await storage.getChunk(chunkId);
    if (!chunk || !computeWorkerPort) return;

    computeWorkerPort.postMessage({
      type: 'DELIVER_PAGED_CHUNK',
      payload: {
        index,
        metadata: chunk.metadata,
        binary: chunk.binary
      }
    }, [chunk.binary]);
  } catch (err) {
    self.postMessage({ type: 'ERROR', message: `Paging failed for chunk #${index}: ${err.message}` });
  }
}
