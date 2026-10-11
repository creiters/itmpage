/**
 * app.js - Main Thread UI, Training, RAG, WebMCP & WebRTC Coordinator
 */

import { StorageService } from './storage.js';
import { MeshCoordinator } from './mesh.js';

let newWorkerWaiting = null;
let refreshing = false;

// 1. PWA Service Worker & Safe Lifecycle Updates
const updateToast = document.getElementById('update-toast');
const btnApplyUpdate = document.getElementById('btn-apply-update');
const btnDismissUpdate = document.getElementById('btn-dismiss-update');

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
  }).catch(console.error);

  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (!refreshing) {
      refreshing = true;
      window.location.reload();
    }
  });
}

btnApplyUpdate?.addEventListener('click', () => {
  if (newWorkerWaiting) newWorkerWaiting.postMessage({ type: 'SKIP_WAITING' });
});

btnDismissUpdate?.addEventListener('click', () => {
  if (updateToast) updateToast.classList.remove('show');
});

// 2. Storage & Identity Initialization
const storage = new StorageService();
await storage.init();

const nodeId = 'Node_' + Math.random().toString(36).substring(2, 7);
const nodeLabel = document.getElementById('node-label');
if (nodeLabel) nodeLabel.textContent = `Node ID: ${nodeId}`;

// UI Handles - Chat
const chatStream = document.getElementById('chat-stream');
const inputPrompt = document.getElementById('input-prompt');
const btnSend = document.getElementById('btn-send');
const engineStatus = document.getElementById('engine-status');

// UI Handles - Training
const dropZone = document.getElementById('drop-zone');
const fileDatasetInput = document.getElementById('file-dataset-input');
const btnBrowseFile = document.getElementById('btn-browse-file');
const datasetSizeBadge = document.getElementById('dataset-size-badge');
const trainTextInput = document.getElementById('train-text-input');
const selectAgentTarget = document.getElementById('select-agent-target');
const inputEpochs = document.getElementById('input-epochs');
const btnStartTraining = document.getElementById('btn-start-training');
const btnStopTraining = document.getElementById('btn-stop-training');
const trainProgressFill = document.getElementById('train-progress-fill');
const trainProgressText = document.getElementById('train-progress-text');
const metricTrainLoss = document.getElementById('metric-train-loss');
const metricTrainPpl = document.getElementById('metric-train-ppl');
const metricModelSize = document.getElementById('metric-model-size');
const btnExportJsonllm = document.getElementById('btn-export-jsonllm');
const btnImportJsonllmBtn = document.getElementById('btn-import-jsonllm-btn');
const fileJsonllmImport = document.getElementById('file-jsonllm-import');

// UI Handles - RAG
const ragDocId = document.getElementById('rag-doc-id');
const ragDocText = document.getElementById('rag-doc-text');
const btnIndexRAG = document.getElementById('btn-index-rag');
const ragIndexStatus = document.getElementById('rag-index-status');

// UI Handles - WebMCP
const mcpRequestJson = document.getElementById('mcp-request-json');
const mcpResponseJson = document.getElementById('mcp-response-json');
const btnSendMcp = document.getElementById('btn-send-mcp');
const btnMcpInit = document.getElementById('btn-mcp-init');
const btnMcpTools = document.getElementById('btn-mcp-tools');
const btnMcpResources = document.getElementById('btn-mcp-resources');

// UI Handles - DB & Quota
const metricStorageUsed = document.getElementById('metric-storage-used');
const metricStorageQuota = document.getElementById('metric-storage-quota');
const metricStorageStatus = document.getElementById('metric-storage-status');
const btnRequestPersist = document.getElementById('btn-request-persist');
const dbSummaryBadge = document.getElementById('db-summary-badge');
const dbChunksTbody = document.getElementById('db-chunks-tbody');
const btnRefreshDb = document.getElementById('btn-refresh-db');
const btnClearCache = document.getElementById('btn-clear-cache');
const clearStatusLabel = document.getElementById('clear-status-label');

// UI Handles - WebRTC Mesh
const btnHostPin = document.getElementById('btn-host-pin');
const hostTokenOut = document.getElementById('host-token-out');
const joinTokenIn = document.getElementById('join-token-in');
const btnConnectPin = document.getElementById('btn-connect-pin');
const peerStatus = document.getElementById('peer-status');

// UI Handles - Logs
const diagnosticsLog = document.getElementById('diagnostics-log');
const btnClearLogs = document.getElementById('btn-clear-logs');

// 3. Tab Routing
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

async function logEvent(category, text) {
  await storage.logEvent(category, text);
  if (!diagnosticsLog) return;
  const time = new Date().toLocaleTimeString();
  const el = document.createElement('div');
  el.style.marginBottom = '3px';
  el.innerHTML = `<span style="color:var(--text-muted);">[${time}]</span> <span class="log-tag ${category}">${category}</span> ${text}`;
  diagnosticsLog.appendChild(el);
  diagnosticsLog.scrollTop = diagnosticsLog.scrollHeight;
}

// 4. Background Workers & Inter-Worker Channel
const chunkWorker = new Worker('./chunk-worker.js', { type: 'module' });
const llmWorker = new Worker('./llm-worker.js', { type: 'module' });
const webmcpWorker = new Worker('./webmcp-worker.js', { type: 'module' });

const channel = new MessageChannel();
chunkWorker.postMessage({ type: 'CONNECT_COMPUTE_PORT' }, [channel.port1]);
llmWorker.postMessage({ type: 'CONNECT_CHUNK_PORT' }, [channel.port2]);

// Worker Message Dispatchers
llmWorker.onmessage = async (e) => {
  const { type, status, domain, token, fullText, done } = e.data;

  if (type === 'STATUS' && engineStatus) {
    engineStatus.textContent = status;
  }
  if (type === 'DOMAIN_SWITCH' && engineStatus) {
    engineStatus.textContent = `Active: agent-${domain}`;
    await logEvent('COMPUTE', `Swapped active model to agent-${domain}`);
  }
  if (type === 'TOKEN') {
    let lastMsg = chatStream.querySelector('.message.assistant:last-child');
    if (!lastMsg || lastMsg.dataset.done === 'true') {
      lastMsg = document.createElement('div');
      lastMsg.className = 'message assistant';
      chatStream.appendChild(lastMsg);
    }
    lastMsg.textContent = fullText;
    if (done) {
      lastMsg.dataset.done = 'true';
      await storage.saveMessage('assistant', fullText);
    }
    chatStream.scrollTop = chatStream.scrollHeight;
  }
};

webmcpWorker.onmessage = async (e) => {
  const { type, status, payload, loss, ppl, sizeKb } = e.data;

  if (type === 'STATUS' && engineStatus) {
    engineStatus.textContent = status;
  }
  if (type === 'WEBMCP_RESPONSE' && mcpResponseJson) {
    mcpResponseJson.textContent = JSON.stringify(payload, null, 2);
  }
  if (type === 'TRAIN_PROGRESS') {
    if (metricTrainLoss) metricTrainLoss.textContent = payload.loss.toFixed(4);
    if (metricTrainPpl) metricTrainPpl.textContent = payload.perplexity.toFixed(1);
    if (trainProgressFill) trainProgressFill.style.width = `${payload.progressPct}%`;
    if (trainProgressText) {
      trainProgressText.textContent = `Epoch ${payload.epoch}/${payload.totalEpochs} — ${payload.progressPct}%`;
    }
  }
  if (type === 'TRAIN_DONE') {
    if (metricTrainLoss) metricTrainLoss.textContent = loss;
    if (metricTrainPpl) metricTrainPpl.textContent = ppl;
    if (metricModelSize) metricModelSize.textContent = `${sizeKb} KB`;
    if (btnStartTraining) btnStartTraining.disabled = false;
    if (btnStopTraining) btnStopTraining.disabled = true;
    if (trainProgressText) trainProgressText.textContent = `Training complete. Model size: ${sizeKb} KB`;

    await logEvent('TRAIN', `Trained agent on-device. Loss: ${loss}, Model size: ${sizeKb} KB`);
    await renderDatabaseView();
    await updateStorageMetrics();
  }
  if (type === 'EXPORT_JSONLLM_COMPLETE') {
    const blob = new Blob([payload.jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${payload.modelName}.jsonllm`;
    a.click();
    URL.revokeObjectURL(url);
    await logEvent('STORAGE', `Exported JSONLLM model: ${payload.modelName}`);
  }
  if (type === 'IMPORT_JSONLLM_COMPLETE') {
    alert(`Imported ${payload.modelName} successfully!`);
    await logEvent('STORAGE', `Imported ${payload.modelName} into IndexedDB`);
    await renderDatabaseView();
  }
};

// 5. Chat Interaction
function executeChat() {
  const text = inputPrompt.value.trim();
  if (!text) return;

  const msg = document.createElement('div');
  msg.className = 'message user';
  msg.textContent = text;
  chatStream.appendChild(msg);
  inputPrompt.value = '';
  chatStream.scrollTop = chatStream.scrollHeight;

  // Delegate compute if peers are connected; otherwise run on local micro-engine
  const delegated = mesh.dispatchCompute(text);
  if (delegated) {
    logEvent('WEBRTC', `Dispatched compute task to peer: ${delegated.peerKey}`);
  } else {
    llmWorker.postMessage({ type: 'INFER', payload: { prompt: text, maxTokens: 48 } });
  }
}

btnSend?.addEventListener('click', executeChat);
inputPrompt?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') executeChat();
});

// 6. Dataset Drag-and-Drop & Training Handlers
btnBrowseFile?.addEventListener('click', () => fileDatasetInput?.click());
fileDatasetInput?.addEventListener('change', async (e) => {
  await handleLoadedFiles(e.target.files);
});

dropZone?.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.style.borderColor = 'var(--accent)';
});

dropZone?.addEventListener('dragleave', () => {
  dropZone.style.borderColor = 'var(--border)';
});

dropZone?.addEventListener('drop', async (e) => {
  e.preventDefault();
  dropZone.style.borderColor = 'var(--border)';
  if (e.dataTransfer && e.dataTransfer.files) {
    await handleLoadedFiles(e.dataTransfer.files);
  }
});

async function handleLoadedFiles(files) {
  if (!files || !files.length) return;
  let textBuffer = '';
  for (const file of files) {
    textBuffer += (await file.text()) + '\n';
  }
  trainTextInput.value = (trainTextInput.value + '\n' + textBuffer).trim();
  datasetSizeBadge.textContent = `${files.length} file(s) loaded (~${Math.round(trainTextInput.value.length / 4)} tokens)`;
  await logEvent('TRAIN', `Loaded ${files.length} training file(s) into buffer`);
}

btnStartTraining?.addEventListener('click', () => {
  const text = trainTextInput.value.trim();
  if (!text) {
    alert('Please enter or upload training text.');
    return;
  }
  const domain = selectAgentTarget.value;
  const epochs = parseInt(inputEpochs.value, 10) || 5;

  btnStartTraining.disabled = true;
  btnStopTraining.disabled = false;
  trainProgressFill.style.width = '0%';
  trainProgressText.textContent = 'Training in background worker...';

  webmcpWorker.postMessage({
    type: 'TRAIN_CORPUS',
    payload: { domain, text, epochs, lr: 0.003 }
  });
});

btnExportJsonllm?.addEventListener('click', () => {
  const domain = selectAgentTarget.value;
  webmcpWorker.postMessage({
    type: 'EXPORT_JSONLLM',
    payload: { modelName: `agent-${domain}` }
  });
});

btnImportJsonllmBtn?.addEventListener('click', () => fileJsonllmImport?.click());
fileJsonllmImport?.addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  const jsonString = await file.text();
  webmcpWorker.postMessage({
    type: 'IMPORT_JSONLLM',
    payload: { jsonString, modelName: file.name.replace(/\.[^/.]+$/, '') }
  });
});

// 7. Vector RAG Indexing
btnIndexRAG?.addEventListener('click', () => {
  const docId = ragDocId.value.trim();
  const text = ragDocText.value.trim();
  if (!docId || !text) return;

  webmcpWorker.postMessage({
    type: 'WEBMCP_REQUEST',
    payload: {
      jsonrpc: '2.0',
      id: Date.now(),
      method: 'tools/call',
      params: { name: 'rag_index_document', arguments: { docId, text } }
    }
  });

  ragIndexStatus.textContent = `Indexed document: ${docId}`;
  logEvent('RAG', `Indexed knowledge: ${docId}`);
});

// 8. WebMCP Client Buttons
btnMcpInit?.addEventListener('click', () => {
  mcpRequestJson.value = JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'initialize', params: {} }, null, 2);
});
btnMcpTools?.addEventListener('click', () => {
  mcpRequestJson.value = JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'tools/list', params: {} }, null, 2);
});
btnMcpResources?.addEventListener('click', () => {
  mcpRequestJson.value = JSON.stringify({ jsonrpc: '2.0', id: 3, method: 'resources/list', params: {} }, null, 2);
});
btnSendMcp?.addEventListener('click', () => {
  try {
    const parsed = JSON.parse(mcpRequestJson.value);
    webmcpWorker.postMessage({ type: 'WEBMCP_REQUEST', payload: parsed });
  } catch (err) {
    alert('Invalid JSON: ' + err.message);
  }
});

// 9. Database View & Quotas
async function updateStorageMetrics() {
  if (navigator.storage && navigator.storage.estimate) {
    const est = await navigator.storage.estimate();
    const used = ((est.usage || 0) / (1024 * 1024)).toFixed(1);
    const quota = ((est.quota || 0) / (1024 * 1024)).toFixed(1);
    if (metricStorageUsed) metricStorageUsed.textContent = `${used} MB`;
    if (metricStorageQuota) metricStorageQuota.textContent = `${quota} MB`;
  }
  if (navigator.storage && navigator.storage.persisted) {
    const persisted = await navigator.storage.persisted();
    if (metricStorageStatus) {
      metricStorageStatus.textContent = persisted ? 'Persisted' : 'Best-Effort';
      metricStorageStatus.style.color = persisted ? 'var(--status-ready)' : 'var(--text-muted)';
    }
  }
}

btnRequestPersist?.addEventListener('click', async () => {
  if (navigator.storage && navigator.storage.persist) {
    const granted = await navigator.storage.persist();
    await logEvent('STORAGE', granted ? 'Persistent storage granted' : 'Persistence denied');
    await updateStorageMetrics();
  }
});

async function renderDatabaseView() {
  const segments = await storage.getAllSegments();
  if (dbSummaryBadge) dbSummaryBadge.textContent = `${segments.length} segment(s)`;

  if (!dbChunksTbody) return;
  if (!segments.length) {
    dbChunksTbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:var(--text-muted);">No segments cached.</td></tr>';
    return;
  }

  dbChunksTbody.innerHTML = segments.map((s) => `
    <tr>
      <td>${s.segmentId}</td>
      <td>${s.modelName}</td>
      <td>${(s.payload.length / 1024).toFixed(1)} KB</td>
      <td><code>${s.checksum}</code></td>
    </tr>
  `).join('');
}

btnRefreshDb?.addEventListener('click', async () => {
  await renderDatabaseView();
  await updateStorageMetrics();
});

btnClearCache?.addEventListener('click', async () => {
  if (!confirm('Clear all stored models, segments, and logs?')) return;
  await storage.clearLogs();
  if (chatStream) chatStream.innerHTML = '';
  window.location.reload();
});

// 10. WebRTC PIN Mesh
const mesh = new MeshCoordinator(
  nodeId,
  (msg) => {
    if (msg.type === 'MESH_INFER') {
      llmWorker.postMessage({ type: 'INFER', payload: { prompt: msg.prompt, maxTokens: 48 } });
    }
  },
  (count) => {
    if (peerStatus) peerStatus.textContent = `${count} Peers Connected`;
  }
);

btnHostPin?.addEventListener('click', async () => {
  const pin = mesh.createPin();
  const token = await mesh.generateHostToken(pin);
  if (hostTokenOut) hostTokenOut.value = token;
  await logEvent('WEBRTC', `Host PIN generated: ${pin}`);
});

btnConnectPin?.addEventListener('click', async () => {
  const tok = joinTokenIn?.value.trim();
  if (!tok) return;

  try {
    const raw = MeshCoordinator.decodePayload(tok);
    if (raw.sdp.type === 'offer') {
      const answer = await mesh.acceptTokenAndCreateAnswer(tok);
      if (joinTokenIn) joinTokenIn.value = answer;
      await logEvent('WEBRTC', 'Accepted offer; return answer token to host');
    } else if (raw.sdp.type === 'answer') {
      await mesh.finalizeHostConnection(tok);
      await logEvent('WEBRTC', 'Peer connected via PIN');
    }
  } catch (err) {
    alert('Handshake error: ' + err.message);
  }
});

btnClearLogs?.addEventListener('click', async () => {
  await storage.clearLogs();
  if (diagnosticsLog) diagnosticsLog.innerHTML = '';
});

// Boot State
renderDatabaseView();
updateStorageMetrics();
