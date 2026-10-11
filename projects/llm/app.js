import { StorageService } from './storage.js';
import { HuggingFaceDownloader } from './downloader.js';
import { MeshCoordinator } from './mesh.js';

// 1. Service Worker & Update Handlers
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
        if (updateStatusLabel) updateStatusLabel.textContent = 'Checking server for updates...';
        try {
          await reg.update();
          setTimeout(() => {
            if (!newWorkerWaiting && updateStatusLabel) {
              updateStatusLabel.textContent = 'No updates found. Running latest version.';
            }
            btnCheckUpdate.disabled = false;
          }, 800);
        } catch (err) {
          if (updateStatusLabel) updateStatusLabel.textContent = 'Failed: ' + err.message;
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

// Training Controls
const trainTextInput = document.getElementById('train-text-input');
const btnTrainStep = document.getElementById('btn-train-step');
const btnSaveWeights = document.getElementById('btn-save-weights');
const trainMetricsBadge = document.getElementById('train-metrics-badge');

// WebMCP Controls
const mcpRequestJson = document.getElementById('mcp-request-json');
const mcpResponseJson = document.getElementById('mcp-response-json');
const btnSendMcp = document.getElementById('btn-send-mcp');
const btnMcpInit = document.getElementById('btn-mcp-init');
const btnMcpTools = document.getElementById('btn-mcp-tools');
const btnMcpResources = document.getElementById('btn-mcp-resources');
const btnMcpSample = document.getElementById('btn-mcp-sample');

// Storage & DB Controls
const selectModel = document.getElementById('select-model');
const btnDownloadSplit = document.getElementById('btn-download-split');
const progressContainer = document.getElementById('progress-container');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const graphManifest = document.getElementById('graph-manifest');
const btnClearCache = document.getElementById('btn-clear-cache');
const clearStatusLabel = document.getElementById('clear-status-label');

const btnRefreshDb = document.getElementById('btn-refresh-db');
const dbChunksTbody = document.getElementById('db-chunks-tbody');
const dbSummaryBadge = document.getElementById('db-summary-badge');
const metricStorageUsed = document.getElementById('metric-storage-used');
const metricStorageQuota = document.getElementById('metric-storage-quota');
const metricStorageStatus = document.getElementById('metric-storage-status');
const btnRequestPersist = document.getElementById('btn-request-persist');

// Hardware Controls
const btnDeviceCpu = document.getElementById('btn-device-cpu');
const btnDeviceGpu = document.getElementById('btn-device-gpu');
const activeDeviceLabel = document.getElementById('active-device-label');

// WebRTC Controls
const peerStatus = document.getElementById('peer-status');
const hostTokenOut = document.getElementById('host-token-out');
const joinTokenIn = document.getElementById('join-token-in');
const btnHostPin = document.getElementById('btn-host-pin');
const btnConnectPin = document.getElementById('btn-connect-pin');

// Telemetry Controls
const metricTPS = document.getElementById('metric-tps');
const metricTTFT = document.getElementById('metric-ttft');
const metricActiveSlices = document.getElementById('metric-active-slices');
const tokenTopKBars = document.getElementById('token-topk-bars');
const diagnosticsLog = document.getElementById('diagnostics-log');
const btnClearLogs = document.getElementById('btn-clear-logs');

// Navigation Tabs
const navTabs = document.querySelectorAll('.nav-tab');
const appViews = document.querySelectorAll('.app-view');

navTabs.forEach((tab) => {
  tab.addEventListener('click', async () => {
    const targetId = tab.getAttribute('data-target');
    navTabs.forEach((t) => t.classList.remove('active'));
    tab.classList.add('active');

    appViews.forEach((view) => {
      if (view.id === targetId) view.classList.add('active');
      else view.classList.remove('active');
    });

    if (targetId === 'view-db') {
      await renderDatabaseView();
      await updateStorageMetrics();
    }
  });
});

// 3. Storage Service Initialization
const storage = new StorageService();
await storage.init();

async function appendLog(category, text) {
  await storage.logEvent(category, text);
  if (!diagnosticsLog) return;
  const time = new Date().toLocaleTimeString();
  const el = document.createElement('div');
  el.style.marginBottom = '3px';
  el.innerHTML = `<span style="color:var(--text-muted);">[${time}]</span> <span class="log-tag ${category}">${category}</span> ${text}`;
  diagnosticsLog.appendChild(el);
  diagnosticsLog.scrollTop = diagnosticsLog.scrollHeight;
}

async function hydrateLogs() {
  const previousLogs = await storage.getRecentLogs(50);
  previousLogs.reverse().forEach((log) => {
    const time = new Date(log.timestamp).toLocaleTimeString();
    const el = document.createElement('div');
    el.style.marginBottom = '3px';
    el.innerHTML = `<span style="color:var(--text-muted);">[${time}]</span> <span class="log-tag ${log.category}">${log.category}</span> ${log.message}`;
    diagnosticsLog.appendChild(el);
  });
  diagnosticsLog.scrollTop = diagnosticsLog.scrollHeight;
}
await hydrateLogs();

if (btnClearLogs) {
  btnClearLogs.addEventListener('click', async () => {
    await storage.clearLogs();
    if (diagnosticsLog) diagnosticsLog.innerHTML = '';
  });
}

// 4. Background Workers & Inter-Worker Channel
const chunkWorker = new Worker('./chunk-worker.js', { type: 'module' });
const computeWorker = new Worker('./llm-worker.js', { type: 'module' });
const webmcpWorker = new Worker('./webmcp-worker.js', { type: 'module' });

const channel = new MessageChannel();
chunkWorker.postMessage({ type: 'CONNECT_COMPUTE_PORT' }, [channel.port1]);
computeWorker.postMessage({ type: 'CONNECT_CHUNK_PORT' }, [channel.port2]);

webmcpWorker.postMessage({ type: 'INIT_WEBMCP' });

async function initComputeEngineWasm() {
  try {
    const res = await fetch('./engine.wasm');
    if (!res.ok) return;
    const wasmBinary = await res.arrayBuffer();
    computeWorker.postMessage({
      type: 'INIT_ENGINE',
      payload: { engineWasmBinary: wasmBinary, modelName: selectModel.value }
    });
  } catch (err) {
    console.warn('WASM engine load bypassed:', err.message);
  }
}
await initComputeEngineWasm();

// Event Listener for Chunk Worker
chunkWorker.onmessage = async (e) => {
  const { type, payload, message, category, text, progress } = e.data;

  if (type === 'LOG') {
    await appendLog(category || 'STORAGE', text);
  } else if (type === 'ERROR') {
    await appendLog('ERROR', message);
  } else if (type === 'CHUNK_PROGRESS') {
    const pct = ((progress.current / progress.total) * 100).toFixed(0);
    if (progressFill) progressFill.style.width = `${pct}%`;
    if (progressText) progressText.textContent = `Merkle ${progress.phase}: ${progress.current}/${progress.total} (4MB blocks)`;
  } else if (type === 'SPLIT_COMPLETE' || type === 'CHUNKS_READY') {
    if (engineStatus) engineStatus.textContent = `Engine: Loaded (${payload.chunkCount} Slices)`;
    if (progressText) progressText.textContent = `Verified Merkle Root: ${payload.rootHash.slice(0, 10)}...`;
    updateDAGManifest(payload.manifest, payload.rootHash, payload.modelName);
    if (metricActiveSlices) metricActiveSlices.textContent = `${payload.chunkCount}`;
    await renderDatabaseView();
    await updateStorageMetrics();
    btnDownloadSplit.disabled = false;
  } else if (type === 'CHUNKS_EMPTY') {
    if (engineStatus) engineStatus.textContent = 'Engine: Standalone (No Slices)';
    await appendLog('STORAGE', `No stored slices found for ${payload.modelName}. Please download.`);
  }
};

// Event Listener for WebMCP Worker
webmcpWorker.onmessage = async (e) => {
  const { type, status, payload } = e.data;

  if (type === 'STATUS' && engineStatus) {
    engineStatus.textContent = status;
  }

  if (type === 'WEBMCP_RESPONSE') {
    if (mcpResponseJson) {
      mcpResponseJson.textContent = JSON.stringify(payload, null, 2);
    }
    if (payload?.result?.content?.text) {
      updateAssistantBubble(payload.result.content.text);
      storage.saveMessage('assistant', payload.result.content.text);
    }
  }

  if (type === 'TRAIN_RESULT' && trainMetricsBadge) {
    trainMetricsBadge.textContent = `Loss: ${payload.loss.toFixed(4)} (${payload.textLength} chars)`;
    btnTrainStep.disabled = false;
  }

  if (type === 'SAVE_COMPLETE') {
    alert(`Model weights partitioned into ${payload.chunksCount} chunks! Merkle Root: ${payload.rootHash.slice(0, 10)}...`);
    await renderDatabaseView();
    await updateStorageMetrics();
  }
};

// 5. Database & Storage View Management
async function updateStorageMetrics() {
  const quota = await storage.getStorageQuota();
  const isPersisted = await storage.isPersistenceGranted();

  if (metricStorageUsed) metricStorageUsed.textContent = `${quota.usageMB} MB`;
  if (metricStorageQuota) metricStorageQuota.textContent = `${quota.quotaMB} MB`;
  if (metricStorageStatus) {
    metricStorageStatus.textContent = isPersisted ? 'Persisted' : 'Best-Effort';
    metricStorageStatus.style.color = isPersisted ? 'var(--status-ready)' : 'var(--text-muted)';
  }
}

if (btnRequestPersist) {
  btnRequestPersist.addEventListener('click', async () => {
    btnRequestPersist.disabled = true;
    const granted = await storage.requestPersistence();
    await appendLog('STORAGE', granted ? 'Persistent storage granted.' : 'Persistent storage rejected.');
    await updateStorageMetrics();
    btnRequestPersist.disabled = false;
  });
}

if (storage.broadcast) {
  storage.broadcast.onmessage = async (event) => {
    const { type, modelName, count } = event.data;
    if (type === 'CHUNKS_UPDATED') {
      await appendLog('STORAGE', `Storage sync: ${count} chunks updated for ${modelName}.`);
      await updateStorageMetrics();
      await renderDatabaseView();
    } else if (type === 'ALL_STORAGE_CLEARED') {
      window.location.reload();
    }
  };
}

async function renderDatabaseView() {
  const modelKey = selectModel.value;
  let summary = await storage.getChunkSummary(modelKey);
  if (!summary.length) {
    summary = await storage.getChunkSummary('browser-nano-llm');
  }

  if (dbSummaryBadge) {
    dbSummaryBadge.textContent = `${summary.length} chunks stored in IndexedDB (${(summary.length * 4).toFixed(0)} MB)`;
  }

  if (!dbChunksTbody) return;
  if (!summary.length) {
    dbChunksTbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No records found.</td></tr>';
    return;
  }

  dbChunksTbody.innerHTML = summary
    .map(
      (c) => `
    <tr>
      <td>${c.chunkId}</td>
      <td>L${c.layerRange[0]}-L${c.layerRange[1]}</td>
      <td>${(c.sizeBytes / 1024 / 1024).toFixed(2)} MB</td>
      <td>${c.hash.slice(0, 8)}...</td>
    </tr>
  `
    )
    .join('');
}

if (btnRefreshDb) {
  btnRefreshDb.addEventListener('click', async () => {
    await renderDatabaseView();
    await updateStorageMetrics();
  });
}

// 6. Model Loading & Download Actions
function requestLoadCachedModel() {
  const modelKey = selectModel.value;
  chunkWorker.postMessage({
    type: 'LOAD_CACHED_CHUNKS',
    payload: { modelName: modelKey }
  });
}
requestLoadCachedModel();

selectModel.addEventListener('change', () => {
  requestLoadCachedModel();
  renderDatabaseView();
});

btnDownloadSplit.addEventListener('click', async () => {
  const modelKey = selectModel.value;
  btnDownloadSplit.disabled = true;
  if (progressContainer) progressContainer.style.display = 'block';

  try {
    await appendLog('STORAGE', `Fetching model: ${modelKey}...`);
    const downloadResult = await HuggingFaceDownloader.fetchStream(
      modelKey,
      ({ loaded, total, percent }) => {
        if (progressFill) progressFill.style.width = `${percent.toFixed(1)}%`;
        if (progressText) {
          progressText.textContent = `Download: ${(loaded / 1024 / 1024).toFixed(1)}MB / ${(total / 1024 / 1024).toFixed(1)}MB (${percent.toFixed(1)}%)`;
        }
      }
    );

    await appendLog('STORAGE', 'Download complete. Passing blob to Background Chunk Worker...');
    chunkWorker.postMessage({
      type: 'SPLIT_AND_STORE_CHUNKS',
      payload: {
        modelBlob: downloadResult.blob,
        modelName: modelKey
      }
    });
  } catch (err) {
    if (engineStatus) engineStatus.textContent = 'Download Error: ' + err.message;
    await appendLog('ERROR', err.message);
    btnDownloadSplit.disabled = false;
  }
});

function updateDAGManifest(manifest, rootHash, modelName) {
  if (!manifest || !graphManifest) return;
  graphManifest.innerHTML = `
    <strong>Model:</strong> ${modelName}<br>
    <strong>Merkle Root:</strong> ${rootHash ? rootHash.slice(0, 16) + '...' : 'Verified'}<br>
    <strong>Blocks:</strong> ${manifest.length} × 4MB Nodes<br>
    <strong>Worker Architecture:</strong> Background Chunk Worker<br>
    <strong>Data Path:</strong> Zero-Copy MessageChannel
  `;
}

// 7. Training & WebMCP Client Operations
if (btnTrainStep) {
  btnTrainStep.addEventListener('click', () => {
    const text = trainTextInput.value.trim();
    if (!text) return alert('Enter training text.');
    btnTrainStep.disabled = true;
    trainMetricsBadge.textContent = 'Training step in progress...';
    webmcpWorker.postMessage({ type: 'TRAIN_STEP', payload: { text, lr: 0.001 } });
  });
}

if (btnSaveWeights) {
  btnSaveWeights.addEventListener('click', () => {
    webmcpWorker.postMessage({ type: 'SAVE_MODEL_TO_MERKLE', payload: { modelName: 'browser-nano-llm' } });
  });
}

if (btnMcpInit) {
  btnMcpInit.addEventListener('click', () => {
    mcpRequestJson.value = JSON.stringify({
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: { protocolVersion: '2024-11-05', clientInfo: { name: 'browser-ui', version: '1.0' } }
    }, null, 2);
  });
}

if (btnMcpTools) {
  btnMcpTools.addEventListener('click', () => {
    mcpRequestJson.value = JSON.stringify({
      jsonrpc: '2.0',
      id: 2,
      method: 'tools/list',
      params: {}
    }, null, 2);
  });
}

if (btnMcpResources) {
  btnMcpResources.addEventListener('click', () => {
    mcpRequestJson.value = JSON.stringify({
      jsonrpc: '2.0',
      id: 3,
      method: 'resources/list',
      params: {}
    }, null, 2);
  });
}

if (btnMcpSample) {
  btnMcpSample.addEventListener('click', () => {
    mcpRequestJson.value = JSON.stringify({
      jsonrpc: '2.0',
      id: 4,
      method: 'sampling/createMessage',
      params: {
        messages: [{ role: 'user', content: { type: 'text', text: 'Hello model' } }],
        maxTokens: 32
      }
    }, null, 2);
  });
}

if (btnSendMcp) {
  btnSendMcp.addEventListener('click', () => {
    try {
      const parsed = JSON.parse(mcpRequestJson.value);
      webmcpWorker.postMessage({ type: 'WEBMCP_REQUEST', payload: parsed });
    } catch (err) {
      alert('Invalid JSON: ' + err.message);
    }
  });
}

// 8. Hardware Selection
if (btnDeviceCpu) {
  btnDeviceCpu.addEventListener('click', () => {
    computeWorker.postMessage({ type: 'SET_DEVICE', payload: { device: 'CPU' } });
    if (activeDeviceLabel) activeDeviceLabel.textContent = 'Active Backend: CPU (WASI SIMD)';
    btnDeviceCpu.classList.remove('secondary-btn');
    btnDeviceGpu.classList.add('secondary-btn');
  });
}

if (btnDeviceGpu) {
  btnDeviceGpu.addEventListener('click', () => {
    computeWorker.postMessage({ type: 'SET_DEVICE', payload: { device: 'GPU' } });
    if (activeDeviceLabel) activeDeviceLabel.textContent = 'Active Backend: GPU (WebGPU WGSL)';
    btnDeviceGpu.classList.remove('secondary-btn');
    btnDeviceCpu.classList.add('secondary-btn');
  });
}

// 9. Storage Cleanup
if (btnClearCache) {
  btnClearCache.addEventListener('click', async () => {
    const confirmPurge = confirm('Clear all storage, chunks, logs, and caches?');
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
      if (graphManifest) graphManifest.innerHTML = 'No 4MB slices found in IndexedDB.';

      if (clearStatusLabel) {
        clearStatusLabel.style.color = '#10b981';
        clearStatusLabel.textContent = 'Purge complete. Reloading...';
      }
      setTimeout(() => window.location.reload(), 750);
    } catch (err) {
      if (clearStatusLabel) {
        clearStatusLabel.style.color = '#ef4444';
        clearStatusLabel.textContent = 'Error: ' + err.message;
      }
      btnClearCache.disabled = false;
    }
  });
}

// 10. Chat & Telemetry Streaming
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

computeWorker.onmessage = async (e) => {
  const { type, status, fullText, done, message, text, category, tps, ttft, device, topCandidates } = e.data;

  if (type === 'STATUS' && engineStatus) engineStatus.textContent = `Engine: ${status}`;
  if (type === 'ERROR' && engineStatus) engineStatus.textContent = message;
  if (type === 'LOG') await appendLog(category || 'COMPUTE', text);

  if (type === 'TELEMETRY') {
    if (metricTPS) metricTPS.textContent = tps;
    if (metricTTFT) metricTTFT.textContent = `${ttft} ms`;
    if (activeDeviceLabel) activeDeviceLabel.textContent = `Active Backend: ${device}`;

    if (tokenTopKBars && topCandidates) {
      tokenTopKBars.innerHTML = topCandidates
        .map(
          (c) => `
        <div class="topk-bar-row">
          <span class="topk-label">${c.word.replace(/\s+/g, '␣')}</span>
          <div class="topk-track">
            <div class="topk-fill" style="width: ${Math.max(10, Math.min(100, (parseFloat(c.logit) + 20) * 2.5))}%"></div>
          </div>
          <span style="font-size: 0.65rem; color: var(--text-muted); width: 35px; text-align: right;">${c.logit}</span>
        </div>
      `
        )
        .join('');
    }
  }

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

  // Distribute over mesh if connected, otherwise execute locally via WebMCP
  const delegated = mesh.dispatchCompute(text);
  if (delegated) {
    updateAssistantBubble(`[Mesh] Task dispatched to peer: ${delegated.peerKey}`);
    await appendLog('WEBRTC', `Task dispatched to peer ${delegated.peerKey}`);
  } else {
    webmcpWorker.postMessage({
      type: 'WEBMCP_REQUEST',
      payload: {
        jsonrpc: '2.0',
        id: Date.now(),
        method: 'sampling/createMessage',
        params: {
          messages: [{ role: 'user', content: { type: 'text', text } }],
          maxTokens: 64
        }
      }
    });
  }
}

btnSend.addEventListener('click', executeChat);
inputPrompt.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') executeChat();
});

// 11. Mesh Network Setup
const mesh = new MeshCoordinator(
  nodeId,
  async (msg, peerKey, channel) => {
    if (msg.type === 'INFERENCE_REQUEST') {
      await appendLog('WEBRTC', `Compute requested by peer: ${peerKey}`);
      webmcpWorker.postMessage({
        type: 'WEBMCP_REQUEST',
        payload: {
          jsonrpc: '2.0',
          id: msg.taskId,
          method: 'sampling/createMessage',
          params: { messages: [{ role: 'user', content: { type: 'text', text: msg.prompt } }], maxTokens: 64 }
        }
      });
    } else if (msg.type === 'INFERENCE_RESPONSE') {
      updateAssistantBubble(msg.text);
      if (msg.isDone) storage.saveMessage('assistant', msg.text);
    }
  },
  (peerCount) => {
    if (peerStatus) peerStatus.textContent = `${peerCount} Peers Connected`;
  }
);

if (btnHostPin) {
  btnHostPin.addEventListener('click', async () => {
    const pin = mesh.createPin();
    const token = await mesh.generateHostToken(pin);
    if (hostTokenOut) hostTokenOut.value = token;
    await appendLog('WEBRTC', `PIN generated: ${pin}`);
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
        await appendLog('WEBRTC', 'Offer accepted; generated answer token.');
      } else if (raw.sdp.type === 'answer') {
        await mesh.finalizeHostConnection(input);
        await appendLog('WEBRTC', 'Peer connected via PIN.');
      }
    } catch (err) {
      alert('Handshake failed: ' + err.message);
    }
  });
}

// Initial storage metrics evaluation
updateStorageMetrics();
