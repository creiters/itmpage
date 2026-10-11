/**
 * app.js
 * Clean UI, WebMCP, WebRTC PIN Mesh & Single-Record JSON Storage Coordinator
 */

import { StorageService } from './storage.js';
import { MeshCoordinator } from './mesh.js';

let newWorkerWaiting = null;
let refreshing = false;

// 1. Service Worker Registration
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('./sw.js').then((reg) => {
    if (reg.waiting) newWorkerWaiting = reg.waiting;
    reg.addEventListener('updatefound', () => {
      const installing = reg.installing;
      installing.addEventListener('statechange', () => {
        if (installing.state === 'installed' && navigator.serviceWorker.controller) {
          newWorkerWaiting = installing;
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

// 2. Storage & Node ID
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
const trainTextInput = document.getElementById('train-text-input');
const selectAgentTarget = document.getElementById('select-agent-target');
const inputEpochs = document.getElementById('input-epochs');
const btnStartTraining = document.getElementById('btn-start-training');
const btnSaveJson = document.getElementById('btn-save-json');
const trainProgressFill = document.getElementById('train-progress-fill');
const trainProgressText = document.getElementById('train-progress-text');
const metricTrainLoss = document.getElementById('metric-train-loss');
const metricTrainPpl = document.getElementById('metric-train-ppl');
const metricModelSize = document.getElementById('metric-model-size');
const btnExportFile = document.getElementById('btn-export-file');
const btnImportFileBtn = document.getElementById('btn-import-file-btn');
const fileImportInput = document.getElementById('file-import-input');

// UI Handles - Vector RAG
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

// UI Handles - DB View
const dbModelsTbody = document.getElementById('db-models-tbody');
const btnRefreshDb = document.getElementById('btn-refresh-db');
const btnClearDb = document.getElementById('btn-clear-db');

// UI Handles - WebRTC Mesh
const btnHostPin = document.getElementById('btn-host-pin');
const hostTokenOut = document.getElementById('host-token-out');
const joinTokenIn = document.getElementById('join-token-in');
const btnConnectPin = document.getElementById('btn-connect-pin');
const peerStatus = document.getElementById('peer-status');

// UI Handles - Diagnostics
const diagnosticsLog = document.getElementById('diagnostics-log');
const btnClearLogs = document.getElementById('btn-clear-logs');

// 3. Tab Navigation
const navTabs = document.querySelectorAll('.nav-tab');
const appViews = document.querySelectorAll('.app-view');

navTabs.forEach((tab) => {
  tab.addEventListener('click', async () => {
    const targetId = tab.getAttribute('data-target');
    navTabs.forEach((t) => t.classList.remove('active'));
    tab.classList.add('active');

    appViews.forEach((view) => {
      view.classList.toggle('active', view.id === targetId);
    });

    if (targetId === 'view-db') await renderDatabaseView();
  });
});

async function logEvent(category, text) {
  await storage.logEvent(category, text);
  if (!diagnosticsLog) return;
  const time = new Date().toLocaleTimeString();
  const el = document.createElement('div');
  el.style.marginBottom = '3px';
  el.innerHTML = `<span style="color:var(--text-muted);">[${time}]</span> <span class="badge ${category}">${category}</span> ${text}`;
  diagnosticsLog.appendChild(el);
  diagnosticsLog.scrollTop = diagnosticsLog.scrollHeight;
}

// 4. Background Workers
const llmWorker = new Worker('./llm-worker.js', { type: 'module' });
const webmcpWorker = new Worker('./webmcp-worker.js', { type: 'module' });

llmWorker.onmessage = async (e) => {
  const { type, status, domain, token, fullText, done } = e.data;

  if (type === 'STATUS' && engineStatus) engineStatus.textContent = status;
  if (type === 'DOMAIN_SWITCH' && engineStatus) {
    engineStatus.textContent = `Active: agent-${domain}`;
    await logEvent('MODEL', `Swapped to agent-${domain}`);
  }
  if (type === 'TOKEN') {
    let last = chatStream.querySelector('.message.assistant:last-child');
    if (!last || last.dataset.done === 'true') {
      last = document.createElement('div');
      last.className = 'message assistant';
      chatStream.appendChild(last);
    }
    last.textContent = fullText;
    if (done) last.dataset.done = 'true';
    chatStream.scrollTop = chatStream.scrollHeight;
  }
};

webmcpWorker.onmessage = async (e) => {
  const { type, status, payload, loss, ppl, sizeKb } = e.data;

  if (type === 'STATUS' && engineStatus) engineStatus.textContent = status;
  if (type === 'WEBMCP_RESPONSE' && mcpResponseJson) {
    mcpResponseJson.textContent = JSON.stringify(payload, null, 2);
  }
  if (type === 'TRAIN_PROGRESS') {
    if (metricTrainLoss) metricTrainLoss.textContent = payload.loss.toFixed(4);
    if (metricTrainPpl) metricTrainPpl.textContent = payload.perplexity.toFixed(1);
    if (trainProgressFill) trainProgressFill.style.width = `${payload.progressPct}%`;
    if (trainProgressText) trainProgressText.textContent = `Epoch ${payload.epoch}/${payload.totalEpochs} — ${payload.progressPct}%`;
  }
  if (type === 'TRAIN_DONE') {
    if (metricTrainLoss) metricTrainLoss.textContent = loss;
    if (metricTrainPpl) metricTrainPpl.textContent = ppl;
    if (metricModelSize) metricModelSize.textContent = `${sizeKb} KB`;
    if (btnStartTraining) btnStartTraining.disabled = false;
    if (trainProgressText) trainProgressText.textContent = `Training complete! Model size: ${sizeKb} KB`;

    await logEvent('TRAIN', `Trained model. Loss: ${loss}, Size: ${sizeKb} KB`);
    await renderDatabaseView();
  }
  if (type === 'SAVE_JSON_COMPLETE') {
    alert(`Model "${payload.modelName}" saved as clean JSON in IndexedDB (${payload.sizeKb} KB)!`);
    await renderDatabaseView();
  }
  if (type === 'EXPORT_JSONLLM_COMPLETE') {
    const blob = new Blob([payload.jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${payload.modelName}.jsonllm`;
    a.click();
    URL.revokeObjectURL(url);
    await logEvent('STORAGE', `Exported JSON model file: ${payload.modelName}`);
  }
  if (type === 'IMPORT_JSONLLM_COMPLETE') {
    alert(`Model "${payload.modelName}" successfully imported into IndexedDB!`);
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

  const delegated = mesh.dispatchCompute(text);
  if (delegated) {
    logEvent('WEBRTC', `Dispatched compute to peer ${delegated.peerKey}`);
  } else {
    llmWorker.postMessage({ type: 'INFER', payload: { prompt: text, maxTokens: 48 } });
  }
}

btnSend.addEventListener('click', executeChat);
inputPrompt.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') executeChat();
});

// 6. Training & JSON Model Storage Handlers
btnStartTraining.addEventListener('click', () => {
  const text = trainTextInput.value.trim();
  if (!text) {
    alert('Please enter training text.');
    return;
  }
  const domain = selectAgentTarget.value;
  const epochs = parseInt(inputEpochs.value, 10) || 5;

  btnStartTraining.disabled = true;
  trainProgressFill.style.width = '0%';
  trainProgressText.textContent = 'Training in background worker...';

  webmcpWorker.postMessage({
    type: 'TRAIN_CORPUS',
    payload: { domain, text, epochs, lr: 0.003 }
  });
});

btnSaveJson.addEventListener('click', () => {
  const domain = selectAgentTarget.value;
  webmcpWorker.postMessage({
    type: 'SAVE_JSON_MODEL',
    payload: { domain }
  });
});

btnExportFile.addEventListener('click', () => {
  const domain = selectAgentTarget.value;
  webmcpWorker.postMessage({
    type: 'EXPORT_JSONLLM',
    payload: { modelName: `agent-${domain}` }
  });
});

btnImportFileBtn.addEventListener('click', () => fileImportInput.click());
fileImportInput.addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  const jsonString = await file.text();
  webmcpWorker.postMessage({
    type: 'IMPORT_JSONLLM',
    payload: { jsonString, modelName: file.name.replace(/\.[^/.]+$/, '') }
  });
});

// 7. Vector RAG Handlers
btnIndexRAG.addEventListener('click', () => {
  const docId = ragDocId.value.trim();
  const text = ragDocText.value.trim();
  if (!docId || !text) return;

  webmcpWorker.postMessage({
    type: 'WEBMCP_REQUEST',
    payload: {
      jsonrpc: '2.0',
      id: Date.now(),
      method: 'tools/call',
      params: { name: 'rag_index_fact', arguments: { docId, text } }
    }
  });

  ragIndexStatus.textContent = `Indexed fact: ${docId}`;
  logEvent('RAG', `Indexed fact: ${docId}`);
});

// 8. WebMCP Hub Handlers
btnMcpInit.addEventListener('click', () => {
  mcpRequestJson.value = JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'initialize', params: {} }, null, 2);
});
btnMcpTools.addEventListener('click', () => {
  mcpRequestJson.value = JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'tools/list', params: {} }, null, 2);
});
btnMcpResources.addEventListener('click', () => {
  mcpRequestJson.value = JSON.stringify({ jsonrpc: '2.0', id: 3, method: 'resources/list', params: {} }, null, 2);
});
btnSendMcp.addEventListener('click', () => {
  try {
    const parsed = JSON.parse(mcpRequestJson.value);
    webmcpWorker.postMessage({ type: 'WEBMCP_REQUEST', payload: parsed });
  } catch (err) {
    alert('Invalid JSON: ' + err.message);
  }
});

// 9. Database View Renderer
async function renderDatabaseView() {
  const models = await storage.getAllModels();
  if (!dbModelsTbody) return;

  if (!models.length) {
    dbModelsTbody.innerHTML = '<tr><td colspan="4" style="text-align: center;">No models stored.</td></tr>';
    return;
  }

  dbModelsTbody.innerHTML = models.map((m) => `
    <tr>
      <td><strong>${m.modelName}</strong></td>
      <td>${(m.bytes / 1024).toFixed(1)} KB</td>
      <td><code>clean-jsonllm</code></td>
      <td>${new Date(m.updated).toLocaleTimeString()}</td>
    </tr>
  `).join('');
}

btnRefreshDb.addEventListener('click', renderDatabaseView);
btnClearDb.addEventListener('click', async () => {
  if (!confirm('Clear all stored JSON models and IndexedDB data?')) return;
  await storage.clearAll();
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
    peerStatus.textContent = `${count} Peers Connected`;
  }
);

btnHostPin.addEventListener('click', async () => {
  const pin = mesh.createPin();
  const token = await mesh.generateHostToken(pin);
  hostTokenOut.value = token;
  logEvent('WEBRTC', `Host PIN generated: ${pin}`);
});

btnConnectPin.addEventListener('click', async () => {
  const tok = joinTokenIn.value.trim();
  if (!tok) return;

  try {
    const raw = MeshCoordinator.decodePayload(tok);
    if (raw.sdp.type === 'offer') {
      const answer = await mesh.acceptTokenAndCreateAnswer(tok);
      joinTokenIn.value = answer;
      logEvent('WEBRTC', 'Accepted offer; return answer token to host');
    } else if (raw.sdp.type === 'answer') {
      await mesh.finalizeHostConnection(tok);
      logEvent('WEBRTC', 'Peer connected via PIN');
    }
  } catch (err) {
    alert('Handshake error: ' + err.message);
  }
});

btnClearLogs.addEventListener('click', () => {
  diagnosticsLog.innerHTML = '';
});

// Initial boot
renderDatabaseView();
