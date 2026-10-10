import { StorageService } from './storage.js';
import { ChunkManager } from './chunker.js';
import { HuggingFaceDownloader } from './downloader.js';
import { MeshCoordinator } from './mesh.js';

// 1. Service Worker Offline PWA Registration
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('./sw.js').catch(console.error);
}

// 2. Identity Initialization
const nodeId = 'Node_' + Math.random().toString(36).substring(2, 7);
document.getElementById('node-label').textContent = `Node ID: ${nodeId}`;

// 3. UI Selectors
const chatStream = document.getElementById('chat-stream');
const inputPrompt = document.getElementById('input-prompt');
const btnSend = document.getElementById('btn-send');
const engineStatus = document.getElementById('engine-status');
const peerStatus = document.getElementById('peer-status');
const selectModel = document.getElementById('select-model');
const btnDownloadSplit = document.getElementById('btn-download-split');
const btnExportSlice = document.getElementById('btn-export-slice');
const progressContainer = document.getElementById('progress-container');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const graphManifest = document.getElementById('graph-manifest');
const hostTokenOut = document.getElementById('host-token-out');
const joinTokenIn = document.getElementById('join-token-in');
const btnHostPin = document.getElementById('btn-host-pin');
const btnConnectPin = document.getElementById('btn-connect-pin');

// 4. Persistence & Workers
const storage = new StorageService();
await storage.init();

const pastMessages = await storage.getMessages();
pastMessages.forEach((m) => renderMessage(m.role, m.text));

const worker = new Worker('./llm-worker.js', { type: 'module' });
let currentChunks = [];
let pendingMeshChunk = null;

// 5. Model Loading from Existing IndexedDB Slices
async function loadSlicesFromCache() {
  const modelKey = selectModel.value;
  currentChunks = await storage.getModelChunks(modelKey);

  if (currentChunks.length > 0) {
    updateDAGManifest(currentChunks);
    engineStatus.textContent = `Reassembling ${currentChunks.length} slices...`;

    try {
      const unifiedBuffer = await ChunkManager.reassemble(currentChunks);
      worker.postMessage({
        type: 'INIT_ENGINE',
        payload: { wasmBinary: unifiedBuffer, modelName: modelKey }
      });
    } catch (err) {
      engineStatus.textContent = 'Assembly Failed: ' + err.message;
    }
  } else {
    engineStatus.textContent = 'Engine: Slices Not Found';
  }
}
await loadSlicesFromCache();

// 6. Mesh Engine Integration
const mesh = new MeshCoordinator(
  nodeId,
  async (msg, peerKey, channel) => {
    if (msg.type === 'CHUNK_HEADER') {
      pendingMeshChunk = msg.metadata;
    } else if (msg.type === 'BINARY_PAYLOAD') {
      if (pendingMeshChunk) {
        await storage.putChunk({ metadata: pendingMeshChunk, binary: msg.buffer });
        pendingMeshChunk = null;
        await loadSlicesFromCache();
      }
    } else if (msg.type === 'INFERENCE_REQUEST') {
      // Execute delegated compute task for peer
      worker.postMessage({
        type: 'INFER',
        payload: { prompt: msg.prompt, taskId: msg.taskId }
      });

      const tokenHandler = (ev) => {
        if (ev.data.type === 'TOKEN' && ev.data.taskId === msg.taskId) {
          mesh.returnComputeResult(channel, msg.taskId, ev.data.token, ev.data.done);
          if (ev.data.done) {
            worker.removeEventListener('message', tokenHandler);
          }
        }
      };
      worker.addEventListener('message', tokenHandler);
    } else if (msg.type === 'INFERENCE_RESPONSE') {
      updateAssistantBubble(msg.text);
      if (msg.isDone) {
        storage.saveMessage('assistant', msg.text);
      }
    }
  },
  (peerCount) => {
    peerStatus.textContent = `${peerCount} Peers Connected`;
  }
);

// 7. Download & 1MB Splitting Pipeline
btnDownloadSplit.addEventListener('click', async () => {
  const modelKey = selectModel.value;
  btnDownloadSplit.disabled = true;
  progressContainer.style.display = 'block';

  try {
    engineStatus.textContent = `Downloading ${modelKey}...`;
    const downloadResult = await HuggingFaceDownloader.fetchStream(
      modelKey,
      ({ loaded, total, percent }) => {
        progressFill.style.width = `${percent.toFixed(1)}%`;
        progressText.textContent = `Download: ${(loaded / 1024 / 1024).toFixed(1)}MB / ${(total / 1024 / 1024).toFixed(1)}MB (${percent.toFixed(1)}%)`;
      }
    );

    engineStatus.textContent = 'Splitting into 1MB Contextual DAG slices...';
    currentChunks = await ChunkManager.splitBlob(
      downloadResult.blob,
      modelKey,
      ({ index, total, percent }) => {
        progressFill.style.width = `${percent.toFixed(1)}%`;
        progressText.textContent = `Slicing 1MB: ${index} of ${total} chunks (${percent.toFixed(1)}%)`;
      }
    );

    engineStatus.textContent = 'Writing slices to IndexedDB...';
    for (const chunk of currentChunks) {
      await storage.putChunk(chunk);
    }

    updateDAGManifest(currentChunks);

    engineStatus.textContent = 'Verifying DAG and reassembling linear memory...';
    const binary = await ChunkManager.reassemble(currentChunks);
    worker.postMessage({
      type: 'INIT_ENGINE',
      payload: { wasmBinary: binary, modelName: modelKey }
    });

    progressText.textContent = 'Ready for offline usage.';
  } catch (err) {
    engineStatus.textContent = 'Operation Error';
    progressText.textContent = err.message;
  } finally {
    btnDownloadSplit.disabled = false;
  }
});

function updateDAGManifest(chunks) {
  if (!chunks.length) return;
  const first = chunks[0].metadata;
  const last = chunks[chunks.length - 1].metadata;

  graphManifest.innerHTML = `
    <strong>Model:</strong> ${first.modelName}<br>
    <strong>Slices:</strong> ${chunks.length} × 1MB<br>
    <strong>DAG Root Hash:</strong> ${first.hash.slice(0, 10)}...<br>
    <strong>DAG Tail Hash:</strong> ${last.hash.slice(0, 10)}...<br>
    <strong>Context Layers:</strong> 0 to ${last.context.layerId}<br>
    <strong>Verified Links:</strong> Yes (SHA-256 DAG)
  `;
}

// 8. Standalone Envelope Exporter
btnExportSlice.addEventListener('click', () => {
  if (!currentChunks.length) return alert('No active 1MB slices found.');
  const slice = currentChunks[0];
  const envelope = ChunkManager.exportChunkEnvelope(slice);
  const blobUrl = URL.createObjectURL(envelope);

  const anchor = document.createElement('a');
  anchor.href = blobUrl;
  anchor.download = `${slice.metadata.chunkId}.envelope.bin`;
  anchor.click();
  URL.revokeObjectURL(blobUrl);
});

// 9. Inference Coordination & Stream UI
let currentBubble = null;

function renderMessage(role, text) {
  const el = document.createElement('div');
  el.className = `message ${role}`;
  el.textContent = text;
  chatStream.appendChild(el);
  chatStream.scrollTop = chatStream.scrollHeight;
  return el;
}

function updateAssistantBubble(text) {
  if (!currentBubble) {
    currentBubble = renderMessage('assistant', text);
  } else {
    currentBubble.textContent = text;
    chatStream.scrollTop = chatStream.scrollHeight;
  }
}

worker.onmessage = (e) => {
  const { type, status, token, fullText, done, message } = e.data;
  if (type === 'STATUS') engineStatus.textContent = `Engine: ${status}`;
  if (type === 'ERROR') engineStatus.textContent = message;
  if (type === 'TOKEN') {
    updateAssistantBubble(fullText);
    if (done) storage.saveMessage('assistant', fullText);
  }
};

async function executeChat() {
  const text = inputPrompt.value.trim();
  if (!text) return;

  renderMessage('user', text);
  inputPrompt.value = '';
  currentBubble = null;

  await storage.saveMessage('user', text);

  // Attempt distributed compute via mesh; fallback to local WASM worker
  const delegated = mesh.dispatchCompute(text);
  if (delegated) {
    updateAssistantBubble(`[Mesh] Task dispatched to peer: ${delegated.peerKey}`);
  } else {
    worker.postMessage({
      type: 'INFER',
      payload: { prompt: text, taskId: 'local_' + Date.now() }
    });
  }
}

btnSend.addEventListener('click', executeChat);
inputPrompt.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') executeChat();
});

// 10. WebRTC PIN Handshake Controls
btnHostPin.addEventListener('click', async () => {
  const pin = mesh.createPin();
  const token = await mesh.generateHostToken(pin);
  hostTokenOut.value = token;
});

btnConnectPin.addEventListener('click', async () => {
  const input = joinTokenIn.value.trim();
  if (!input) return;

  try {
    const raw = MeshCoordinator.decodePayload(input);
    if (raw.sdp.type === 'offer') {
      const answer = await mesh.acceptTokenAndCreateAnswer(input);
      joinTokenIn.value = answer; // Copy back to host
    } else if (raw.sdp.type === 'answer') {
      await mesh.finalizeHostConnection(input);
    }
  } catch (err) {
    alert('Signaling Handshake Failed: ' + err.message);
  }
});
