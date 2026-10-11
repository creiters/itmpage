import { WebMCPServer } from './webmcp-server.js';
import { MerkleChunker } from './merkle-chunker.js';

let server = null;

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'INIT_WEBMCP') {
    server = new WebMCPServer({ name: 'browser-nano-llm', version: '1.0.0' });
    await server.init();
    self.postMessage({ type: 'STATUS', status: 'WebMCP Server Ready' });
    return;
  }

  // Handle standard JSON-RPC WebMCP call
  if (type === 'WEBMCP_REQUEST') {
    if (!server) {
      self.postMessage({
        type: 'WEBMCP_RESPONSE',
        payload: { jsonrpc: '2.0', id: payload?.id, error: { code: -32000, message: 'Server not initialized' } }
      });
      return;
    }

    const response = await server.handleMessage(payload);
    if (response) {
      self.postMessage({ type: 'WEBMCP_RESPONSE', payload: response });
    }
    return;
  }

  // Model Training Dispatch
  if (type === 'TRAIN_STEP') {
    if (!server) return;
    const { text, lr = 0.001 } = payload;
    const loss = server.engine.trainStep(text, lr);

    self.postMessage({
      type: 'TRAIN_RESULT',
      payload: { loss, textLength: text.length }
    });
    return;
  }

  // Persist Current Weights as 4MB Merkle Slices
  if (type === 'SAVE_MODEL_TO_MERKLE') {
    if (!server) return;
    const modelBinary = server.engine.serializeToMerkleArray();
    const modelBlob = new Blob([modelBinary], { type: 'application/octet-stream' });
    const modelName = payload?.modelName || 'browser-nano-llm';

    const { rootHash, chunks } = await MerkleChunker.splitToMerkleChunks(
      modelBlob,
      modelName,
      (progress) => self.postMessage({ type: 'MERKLE_PROGRESS', progress })
    );

    await server.storage.setSystemKey(`${modelName}_merkle_root`, rootHash);
    await server.storage.batchPutChunks(chunks);

    self.postMessage({
      type: 'SAVE_COMPLETE',
      payload: { rootHash, chunksCount: chunks.length }
    });
  }
};
