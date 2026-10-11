import { MerkleChunker } from './merkle-chunker.js';
import { StorageService } from './storage.js';

let storage = null;
let computeWorkerPort = null;
let activeManifest = [];
let activeModelName = '';

// Initialize storage inside worker context
async function ensureStorage() {
  if (!storage) {
    storage = new StorageService();
    await storage.init();
  }
}

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  // 1. Establish direct link to the LLM compute worker
  if (type === 'CONNECT_COMPUTE_PORT') {
    computeWorkerPort = e.ports[0];
    computeWorkerPort.onmessage = async (event) => {
      const req = event.data;
      if (req.type === 'REQUEST_CHUNK_BY_INDEX') {
        await handleComputeChunkRequest(req.index, req.modelName);
      }
    };
    self.postMessage({ type: 'LOG', category: 'STORAGE', text: 'Direct MessageChannel linked to LLM Compute Worker.' });
    return;
  }

  await ensureStorage();

  // 2. Background Chunk Splitting & Storage
  if (type === 'SPLIT_AND_STORE_CHUNKS') {
    const { modelBlob, modelName } = payload;
    activeModelName = modelName;

    try {
      self.postMessage({ type: 'LOG', category: 'STORAGE', text: `Deleting older slices for ${modelName}...` });
      await storage.deleteModelChunks(modelName);

      self.postMessage({ type: 'LOG', category: 'STORAGE', text: `Building 4MB Merkle tree for ${modelName}...` });
      const { rootHash, chunks } = await MerkleChunker.splitToMerkleChunks(
        modelBlob,
        modelName,
        (progress) => {
          self.postMessage({ type: 'CHUNK_PROGRESS', progress });
        }
      );

      await storage.setSystemKey(`${modelName}_merkle_root`, rootHash);

      self.postMessage({ type: 'LOG', category: 'STORAGE', text: `Saving ${chunks.length} 4MB chunks to IndexedDB in background...` });
      for (let i = 0; i < chunks.length; i++) {
        await storage.putChunk(chunks[i]);
        if ((i + 1) % 5 === 0 || i === chunks.length - 1) {
          self.postMessage({
            type: 'LOG',
            category: 'STORAGE',
            text: `Stored chunk #${i + 1}/${chunks.length} in IndexedDB`
          });
        }
      }

      // Build manifest metadata
      activeManifest = chunks.map((c, i) => ({
        chunkId: c.metadata.chunkId,
        index: i,
        layerRange: c.metadata.context.layerRange,
        byteOffset: c.metadata.byteOffset,
        byteLength: c.binary.byteLength,
        hash: c.metadata.chunkHash
      }));

      // Assemble unified binary for initial model load
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

      // Transfer assembled binary directly to the compute worker if connected
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
      self.postMessage({ type: 'ERROR', message: `Chunk worker split failed: ${err.message}` });
    }
  }

  // 3. Rehydrate and Reassemble Cached Chunks from IndexedDB
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
      self.postMessage({ type: 'LOG', category: 'STORAGE', text: `Reassembling ${chunks.length} chunks via Merkle DAG...` });

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

  // 4. Wipe Model Storage in Background
  if (type === 'PURGE_MODEL_STORAGE') {
    const { modelName } = payload;
    try {
      await storage.deleteModelChunks(modelName);
      self.postMessage({ type: 'LOG', category: 'STORAGE', text: `Purged chunk store for ${modelName}` });
    } catch (err) {
      self.postMessage({ type: 'ERROR', message: err.message });
    }
  }
};

// Handle real-time on-demand chunk paging for compute worker
async function handleComputeChunkRequest(index, modelName) {
  try {
    const chunkId = `${modelName}_chunk_4mb_${index}`;
    const chunk = await storage.getChunk(chunkId);

    if (!chunk || !computeWorkerPort) return;

    // Send sliced buffer zero-copy directly to compute worker
    computeWorkerPort.postMessage({
      type: 'DELIVER_PAGED_CHUNK',
      payload: {
        index,
        metadata: chunk.metadata,
        binary: chunk.binary
      }
    }, [chunk.binary]);

    self.postMessage({
      type: 'LOG',
      category: 'PAGING',
      text: `Paged 4MB slice #${index} directly to Compute Worker.`
    });
  } catch (err) {
    self.postMessage({ type: 'ERROR', message: `Dynamic chunk page failed: ${err.message}` });
  }
}
