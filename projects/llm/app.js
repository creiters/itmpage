import { StorageService } from './storage.js';
import { MerkleChunker } from './merkle-chunker.js';
import { HuggingFaceDownloader } from './downloader.js';
import { MeshCoordinator } from './mesh.js';

// 1. Service Worker & Application Update Lifecycle
let newWorkerWaiting = null;
let refreshing = false;

const updateToast = document.getElementById('update-toast');
const btnApplyUpdate = document.getElementById('btn-apply-update');
const btnDismissUpdate = document.getElementById('btn-dismiss-update');
const btnCheckUpdate = document.getElementById('btn-check-update');
const updateStatusLabel = document.getElementById('update-status-label');

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('./sw.js').then((reg) => {
    if (reg.waiting) {
      newWorkerWaiting = reg.waiting;
      if (updateToast) updateToast.classList.add('show');
    }

    reg.addEventListener('updatefound', () => {
      const installingWorker = reg.installing;
      installingWorker.addEventListener('statechange', () => {
        if (installingWorker.state === 'installed' && navigator.serviceWorker.controller) {
          newWorkerWaiting = installingWorker;
          if (updateToast) updateToast.classList.add('show');
        }
      });
    });

    if (btnCheckUpdate) {
      btnCheckUpdate.addEventListener('click', async () => {
        btnCheckUpdate.disabled = true;
        if (updateStatusLabel) updateStatusLabel.textContent = 'Checking server for revisions...';
        try {
          await reg.update();
          setTimeout(() => {
            if (!newWorkerWaiting && updateStatusLabel) {
              updateStatusLabel.textContent = 'No updates found. Running latest version.';
            }
            btnCheckUpdate.disabled = false;
          }, 800);
        } catch (err) {
          if (updateStatusLabel) updateStatusLabel.textContent = 'Failed to check: ' + err.message;
          btnCheckUpdate.disabled = false;
        }
      });
    }
  }).catch(console.error);

  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (!refreshing) {
      refreshing = true;
      window.location.reload();
    }
  });
}

if (btnApplyUpdate) {
  btnApplyUpdate.addEventListener('click', () => {
    if (newWorkerWaiting) newWorkerWaiting.postMessage({ type: 'SKIP_WAITING' });
  });
}

if (btnDismissUpdate) {
  btnDismissUpdate.addEventListener('click', () => {
    if (updateToast) updateToast.classList.remove('show');
  });
}

// 2. Identity & DOM References
const nodeId = 'Node_' + Math.random().toString(36).substring(2, 7);
const nodeLabel = document.getElementById('node-label');
if (nodeLabel) nodeLabel.textContent = `Node ID: ${nodeId}`;

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
const btnClearCache = document.getElementById('btn-clear-cache');
const clearStatusLabel = document.getElementById('clear-status-label');

// 3. Tab Routing
const navTabs = document.querySelectorAll('.nav-tab');
const appViews = document.querySelectorAll('.app-view');

navTabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    const targetId = tab.getAttribute('data-target');
    navTabs.forEach((t) => t.classList.remove('active'));
    tab.classList.add('active');

    appViews.forEach((view) => {
      if (view.id === targetId) view.classList.add('active');
      else view.classList.remove('active');
    });

    if (targetId === 'view-chat' && chatStream) {
      chatStream.scrollTop = chatStream.scrollHeight;
    }
  });
});

// 4. Persistence & Dedicated Worker Initialization
const storage = new StorageService();
await storage.init();

const pastMessages = await storage.getMessages();
pastMessages.forEach((m) => renderMessage(m.role, m.text));

const worker = new Worker('./llm-worker.js', { type: 'module' });
let currentChunks = [];
let pendingMeshChunk = null;

// 5. Cache Validation and Engine Bootstrap
async function loadSlicesFromCache() {
  const modelKey = selectModel.value;
  currentChunks = await storage.getModelChunks(modelKey);

  if (currentChunks.length > 0) {
    const expectedTotal = currentChunks[0].metadata.totalChunks;
    const storedRoot = await storage.getSystemKey(`${modelKey}_merkle_root`);

    if (currentChunks.length !== expectedTotal) {
      if (engineStatus) {
        engineStatus.textContent = `Incomplete: ${currentChunks.length}/${expectedTotal} chunks. Please download again.`;
      }
      return;
    }

    updateDAGManifest(currentChunks, storedRoot);
    if (engineStatus) engineStatus.textContent = `Reassembling ${currentChunks.length} chunks via Merkle DAG...`;

    try {
      const unifiedBuffer = await MerkleChunker.reassemble(currentChunks, storedRoot);
      worker.postMessage({
        type: 'INIT_ENGINE',
        payload: { wasmBinary: unifiedBuffer, modelName: modelKey }
      });
    } catch (err) {
      if (engineStatus) engineStatus.textContent = 'Assembly Failed: ' + err.message;
      console.error(err);
    }
  } else {
    if (engineStatus) engineStatus.textContent = 'Engine: Standalone Mode (No Slices)';
    worker.postMessage({
      type: 'INIT_ENGINE',
      payload: { modelName: 'Standalone Engine' }
    });
  }
}
await loadSlicesFromCache();

selectModel.addEventListener('change', async () => {
  await loadSlicesFromCache();
});

// 6. WebRTC Mesh Protocol Integration
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
    if (peerStatus) peerStatus.textContent = `${peerCount} Peers Connected`;
  }
);

// 7. Download & 4 MB Merkle Chunking Pipeline
btnDownloadSplit.addEventListener('click', async () => {
  const modelKey = selectModel.value;
  btnDownloadSplit.disabled = true;
  if (progressContainer) progressContainer.style.display = 'block';

  try {
    // Purge stale/mixed chunks before downloading new ones
    await storage.deleteModelChunks(modelKey);

    if (engineStatus) engineStatus.textContent = `Downloading ${modelKey}...`;
    const downloadResult = await HuggingFaceDownloader.fetchStream(
      modelKey,
      ({ loaded, total, percent }) => {
        if (progressFill) progressFill.style.width = `${percent.toFixed(1)}%`;
        if (progressText) {
          progressText.textContent = `Download: ${(loaded / 1024 / 1024).toFixed(1)}MB / ${(total / 1024 / 1024).toFixed(1)}MB (${percent.toFixed(1)}%)`;
        }
      }
    );

    if (engineStatus) engineStatus.textContent = 'Building 4MB Merkle Tree & Context Links...';
    const { rootHash, chunks } = await MerkleChunker.splitToMerkleChunks(
      downloadResult.blob,
      modelKey,
      ({ phase, current, total }) => {
        const pct = ((current / total) * 100).toFixed(0);
        if (progressFill) progressFill.style.width = `${pct}%`;
        if (progressText) progressText.textContent = `Merkle ${phase}: ${current}/${total} (4MB blocks)`;
      }
    );

    await storage.setSystemKey(`${modelKey}_merkle_root`, rootHash);

    if (engineStatus) engineStatus.textContent = 'Saving 4MB chunks to IndexedDB...';
    for (const chunk of chunks) {
      await storage.putChunk(chunk);
    }

    currentChunks = chunks;
    updateDAGManifest(chunks, rootHash);

    if (engineStatus) engineStatus.textContent = 'Verifying Merkle root & reassembling memory...';
    const binary = await MerkleChunker.reassemble(chunks, rootHash);

    worker.postMessage({
      type: 'INIT_ENGINE',
      payload: { wasmBinary: binary, modelName: modelKey }
    });

    if (progressText) progressText.textContent = `Ready! Merkle Root: ${rootHash.slice(0, 10)}... (4MB verified)`;
  } catch (err) {
    if (engineStatus) engineStatus.textContent = 'Operation Error: ' + err.message;
    if (progressText) progressText.textContent = err.message;
    console.error(err);
  } finally {
    btnDownloadSplit.disabled = false;
  }
});

function updateDAGManifest(chunks, rootHash) {
  if (!chunks.length || !graphManifest) return;
  const first = chunks[0].metadata;
  graphManifest.innerHTML = `
    <strong>Model:</strong> ${first.modelName}<br>
    <strong>Merkle Root:</strong> ${rootHash ? rootHash.slice(0, 16) + '...' : 'Verified'}<br>
    <strong>Blocks:</strong> ${chunks.length} × 4MB Nodes<br>
    <strong>Context Links:</strong> ${first.context.targetNextContext}<br>
    <strong>Layer Spans:</strong> 0 to 30 mapped<br>
    <strong>Engine:</strong> Multi-Block Coroutine
  `;
}

// 8. Standalone Envelope Exporter
if (btnExportSlice) {
  btnExportSlice.addEventListener('click', () => {
    if (!currentChunks.length) return alert('No active 4MB slices found.');
    const slice = currentChunks[0];
    const envelope = MerkleChunker.exportChunkEnvelope(slice);
    const blobUrl = URL.createObjectURL(envelope);

    const anchor = document.createElement('a');
    anchor.href = blobUrl;
    anchor.download = `${slice.metadata.chunkId}.envelope.bin`;
    anchor.click();
    URL.revokeObjectURL(blobUrl);
  });
}

// 9. Clear All Cache & Reset Storage
if (btnClearCache) {
  btnClearCache.addEventListener('click', async () => {
    const confirmPurge = confirm(
      'Are you sure you want to clear all cache?\n\nThis will remove all downloaded 4MB model slices, IndexedDB history, and offline service worker caches.'
    );
    if (!confirmPurge) return;

    btnClearCache.disabled = true;
    if (clearStatusLabel) clearStatusLabel.textContent = 'Purging storage...';

    try {
      await storage.clearAll();

      if ('caches' in window) {
        const cacheNames = await caches.keys();
        await Promise.all(cacheNames.map((name) => caches.delete(name)));
      }

      if (chatStream) chatStream.innerHTML = '';
      currentChunks = [];
      if (graphManifest) graphManifest.innerHTML = 'No 4MB slices found in IndexedDB.';

      worker.postMessage({
        type: 'INIT_ENGINE',
        payload: { modelName: 'Standalone Engine' }
      });

      if (clearStatusLabel) {
        clearStatusLabel.style.color = '#10b981';
        clearStatusLabel.textContent = 'Cache cleared successfully. Reloading...';
      }

      setTimeout(() => window.location.reload(), 750);
    } catch (err) {
      if (clearStatusLabel) {
        clearStatusLabel.style.color = '#ef4444';
        clearStatusLabel.textContent = 'Failed to clear cache: ' + err.message;
      }
      btnClearCache.disabled = false;
    }
  });
}

// 10. Chat Stream Coordination
let currentBubble = null;

function renderMessage(role, text) {
  if (!chatStream) return null;
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
    if (chatStream) chatStream.scrollTop = chatStream.scrollHeight;
  }
}

worker.onmessage = (e) => {
  const { type, status, fullText, done, message } = e.data;
  if (type === 'STATUS' && engineStatus) engineStatus.textContent = `Engine: ${status}`;
  if (type === 'ERROR' && engineStatus) engineStatus.textContent = message;
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

  const delegated = mesh.dispatchCompute(text);
  if (delegated) {
    updateAssistantBubble(`[Mesh] Dispatched to peer: ${delegated.peerKey}`);
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

// 11. WebRTC Mesh PIN Controls
if (btnHostPin) {
  btnHostPin.addEventListener('click', async () => {
    const pin = mesh.createPin();
    const token = await mesh.generateHostToken(pin);
    if (hostTokenOut) hostTokenOut.value = token;
  });
}

if (btnConnectPin) {
  btnConnectPin.addEventListener('click', async () => {
    const input = joinTokenIn ? joinTokenIn.value.trim() : '';
    if (!input) return;

    try {
      const raw = MeshCoordinator.decodePayload(input);
      if (raw.sdp.type === 'offer') {
        const answer = await mesh.acceptTokenAndCreateAnswer(input);
        if (joinTokenIn) joinTokenIn.value = answer;
      } else if (raw.sdp.type === 'answer') {
        await mesh.finalizeHostConnection(input);
      }
    } catch (err) {
      alert('Signaling Handshake Failed: ' + err.message);
    }
  });
}
