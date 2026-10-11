import { StorageService } from './storage.js';
import { MeshCoordinator } from './mesh.js';

let newWorkerWaiting = null;
let refreshing = false;

// 1. Safe Service Worker Hot-Update
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

// 2. Storage & Identity
const storage = new StorageService();
await storage.init();

const nodeId = 'Node_' + Math.random().toString(36).substring(2, 7);
const nodeLabel = document.getElementById('node-label');
if (nodeLabel) nodeLabel.textContent = `Node ID: ${nodeId}`;

// UI Elements
const chatStream = document.getElementById('chat-stream');
const inputPrompt = document.getElementById('input-prompt');
const btnSend = document.getElementById('btn-send');
const engineStatus = document.getElementById('engine-status');

const trainTextInput = document.getElementById('train-text-input');
const inputEpochs = document.getElementById('input-epochs');
const btnStartTraining = document.getElementById('btn-start-training');
const btnGenerateJsonllm = document.getElementById('btn-generate-jsonllm');
const trainProgressFill = document.getElementById('train-progress-fill');
const trainProgressText = document.getElementById('train-progress-text');
const metricTrainLoss = document.getElementById('metric-train-loss');
const metricTrainPpl = document.getElementById('metric-train-ppl');
const metricModelSize = document.getElementById('metric-model-size');
const btnExportFile = document.getElementById('btn-export-file');
const btnImportFileBtn = document.getElementById('btn-import-file-btn');
const fileImportInput = document.getElementById('file-import-input');

const ragDocId = document.getElementById('rag-doc-id');
const ragDocText = document.getElementById('rag-doc-text');
const btnIndexRAG = document.getElementById('btn-index-rag');
const ragIndexStatus = document.getElementById('rag-index-status');
const metricRagHits = document.getElementById('metric-rag-hits');

const mcpRequestJson = document.getElementById('mcp-request-json');
const mcpResponseJson = document.getElementById('mcp-response-json');
const btnSendMcp = document.getElementById('btn-send-mcp');
const btnMcpInit = document.getElementById('btn-mcp-init');
const btnMcpTools = document.getElementById('btn-mcp-tools');
const btnMcpResources = document.getElementById('btn-mcp-resources');

const dbModelsTbody = document.getElementById('db-models-tbody');
const btnRefreshDb = document.getElementById('btn-refresh-db');
const btnClearDb = document.getElementById('btn-clear-db');

const btnHostPin = document.getElementById('btn-host-pin');
const hostTokenOut = document.getElementById('host-token-out');
const joinTokenIn = document.getElementById('join-token-in');
const btnConnectPin = document.getElementById('btn-connect-pin');
const peerStatus = document.getElementById('peer-status');

const metricTPS = document.getElementById('metric-tps');
const metricTTFT = document.getElementById('metric-ttft');
const tokenTopKBars = document.getElementById('token-topk-bars');
const diagnosticsLog = document.getElementById('diagnostics-log');
const btnClearLogs = document.getElementById('btn-clear-logs');

// 3. Tab Routing
const navTabs = document.querySelectorAll('.nav-tab');
const appViews = document.querySelectorAll('.app-view');

navTabs.forEach((tab) => {
  tab.addEventListener('click', async () => {
    const target = tab.getAttribute('data-target');
    navTabs.forEach((t) => t.classList.remove('active'));
    tab.classList.add('active');

    appViews.forEach((v) => v.classList.toggle('active', v.id === target));
    if (target === 'view-db') await renderDatabaseView();
  });
});

async function appendLog(category, text) {
  await storage.logEvent(category, text);
  if (!diagnosticsLog) return;
  const time = new Date().toLocaleTimeString();
  const el = document.createElement('div');
  el.className = 'log-entry';
  el.innerHTML = `<span style="color:var(--text-muted);">[${time}]</span> <span class="log-tag ${category}">${category}</span> <span>${text}</span>`;
  diagnosticsLog.appendChild(el);
  diagnosticsLog.scrollTop = diagnosticsLog.scrollHeight;
}

// 4. Background Workers
const llmWorker = new Worker('./llm-worker.js', { type: 'module' });
const webmcpWorker = new Worker('./webmcp-worker.js', { type: 'module' });

llmWorker.onmessage = async (e) => {
  const { type, status, token, fullText, done, tps, ttft, topCandidates, category, text } = e.data;

  if (type === 'STATUS' && engineStatus) engineStatus.textContent = status;
  if (type === 'LOG') await appendLog(category || 'INFER', text);

  if (type === 'TELEMETRY') {
    if (metricTPS) metricTPS.textContent = tps;
    if (metricTTFT) metricTTFT.textContent = `${ttft} ms`;

    // Inside llmWorker.onmessage handler for type === 'TELEMETRY':
if (tokenTopKBars && topCandidates) {
  tokenTopKBars.innerHTML = topCandidates.map((c) => {
    const fillWidth = Math.max(10, Math.min(100, (parseFloat(c.logit) + 15) * 3));
    return `
      <div class="topk-bar-row">
        <span class="topk-label">${c.word}</span>
        <div class="topk-track">
          <div class="topk-fill" style="width: ${fillWidth}%;"></div>
        </div>
        <span class="topk-score">${c.logit}</span>
      </div>
    `;
  }).join('');
}

  }

  if (type === 'TOKEN') {
    let last = chatStream.querySelector('.message.assistant:last-child');
    if (!last || last.dataset.done === 'true') {
      last = document.createElement('div');
      last.className = 'message assistant';
      chatStream.appendChild(last);
    }
    last.textContent = fullText;
    if (done) {
      last.dataset.done = 'true';
      await storage.saveMessage('assistant', fullText);
    }
    chatStream.scrollTop = chatStream.scrollHeight;
  }
};

webmcpWorker.onmessage = async (e) => {
  const { type, status, payload, loss, ppl, category, text } = e.data;

  if (type === 'STATUS' && engineStatus) engineStatus.textContent = status;
  if (type === 'LOG') await appendLog(category || 'WEBMCP', text);

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
    btnStartTraining.disabled = false;
    trainProgressText.textContent = `Training complete! Ready to generate JSONLLM.`;
    await appendLog('TRAIN', `Training converged. Final loss: ${loss}, PPL: ${ppl}`);
  }

  if (type === 'GENERATE_JSONLLM_SUCCESS') {
    if (metricModelSize) metricModelSize.textContent = `${payload.sizeKb} KB`;
    btnGenerateJsonllm.disabled = false;
    alert(`Generated JSONLLM for ${payload.modelName} (${payload.sizeKb} KB) and saved to IndexedDB!`);
    await appendLog('STORAGE', `Generated & saved JSONLLM: ${payload.modelName} (${payload.sizeKb} KB)`);
    llmWorker.postMessage({ type: 'HOT_RELOAD', payload: { modelName: payload.modelName } });
    await renderDatabaseView();
  }

  if (type === 'IMPORT_JSONLLM_SUCCESS') {
    alert(`Imported ${payload.modelName} (${payload.sizeKb} KB) into IndexedDB!`);
    await appendLog('STORAGE', `Imported ${payload.modelName} JSONLLM`);
    llmWorker.postMessage({ type: 'HOT_RELOAD', payload: { modelName: payload.modelName } });
    await renderDatabaseView();
  }
};

// 5. Chat Trigger
// Replace function executeChat() in app.js:
function executeChat() {
  const text = inputPrompt.value.trim();
  if (!text) return;

  const msg = document.createElement('div');
  msg.className = 'message user';
  msg.textContent = text;
  chatStream.appendChild(msg);
  inputPrompt.value = '';
  chatStream.scrollTop = chatStream.scrollHeight;

  // Always invoke local LLM worker unless a remote peer actively requests computation
  llmWorker.postMessage({ type: 'INFER', payload: { prompt: text, maxTokens: 48 } });
}


btnSend.addEventListener('click', executeChat);
inputPrompt.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') executeChat();
});

// 6. Training & JSONLLM Generation Triggers
btnStartTraining.addEventListener('click', () => {
  const text = trainTextInput.value.trim();
  if (!text) return alert('Enter training text.');
  const epochs = parseInt(inputEpochs.value, 10) || 5;

  btnStartTraining.disabled = true;
  trainProgressFill.style.width = '0%';
  trainProgressText.textContent = 'Running forward tape & AdamW autograd...';

  webmcpWorker.postMessage({
    type: 'START_TRAIN',
    payload: { text, epochs, lr: 0.003 }
  });
});

btnGenerateJsonllm.addEventListener('click', () => {
  btnGenerateJsonllm.disabled = true;
  webmcpWorker.postMessage({
    type: 'GENERATE_JSONLLM',
    payload: { modelName: 'agent-dialog' }
  });
});

btnExportFile.addEventListener('click', async () => {
  const jsonStr = await storage.getModel('agent-dialog');
  if (!jsonStr) return alert('Generate a JSONLLM model first.');
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'agent-dialog.jsonllm';
  a.click();
  URL.revokeObjectURL(url);
  await appendLog('STORAGE', 'Exported agent-dialog.jsonllm file to disk');
});

btnImportFileBtn.addEventListener('click', () => fileImportInput.click());
fileImportInput.addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  const jsonString = await file.text();
  const modelName = file.name.replace(/\.[^/.]+$/, '');
  webmcpWorker.postMessage({
    type: 'IMPORT_JSONLLM',
    payload: { jsonString, modelName }
  });
});

// 7. Vector RAG Trigger
btnIndexRAG.addEventListener('click', async () => {
  const id = ragDocId.value.trim();
  const text = ragDocText.value.trim();
  if (!id || !text) return;

  webmcpWorker.postMessage({
    type: 'WEBMCP_REQUEST',
    payload: {
      jsonrpc: '2.0',
      id: Date.now(),
      method: 'tools/call',
      params: { name: 'rag_index_document', arguments: { docId: id, text } }
    }
  });

  ragIndexStatus.textContent = `Indexed document: ${id}`;
  await appendLog('STORAGE', `Indexed document "${id}" into Vector RAG`);
  const docs = await storage.getAllRAGDocs();
  if (metricRagHits) metricRagHits.textContent = `${docs.length}`;
});

// 8. WebMCP Client Triggers
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

// 9. DB View Renderer
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
      <td><code>jsonllm/v2</code></td>
      <td>${new Date(m.updated).toLocaleTimeString()}</td>
    </tr>
  `).join('');
}

btnRefreshDb.addEventListener('click', renderDatabaseView);
btnClearDb.addEventListener('click', async () => {
  if (!confirm('Clear all stored JSONLLM models and IndexedDB data?')) return;
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
  await appendLog('WEBRTC', `Host PIN generated: ${pin}`);
});

btnConnectPin.addEventListener('click', async () => {
  const tok = joinTokenIn.value.trim();
  if (!tok) return;

  try {
    const raw = MeshCoordinator.decodePayload(tok);
    if (raw.sdp.type === 'offer') {
      const answer = await mesh.acceptTokenAndCreateAnswer(tok);
      joinTokenIn.value = answer;
      await appendLog('WEBRTC', 'Accepted offer; return answer token to host');
    } else if (raw.sdp.type === 'answer') {
      await mesh.finalizeHostConnection(tok);
      await appendLog('WEBRTC', 'Peer connected via PIN');
    }
  } catch (err) {
    alert('Handshake error: ' + err.message);
  }
});

btnClearLogs.addEventListener('click', async () => {
  await storage.clearLogs();
  diagnosticsLog.innerHTML = '';
});

// Hydrate previous event logs from storage
async function hydrateLogs() {
  const previousLogs = await storage.getRecentLogs(40);
  previousLogs.forEach((log) => {
    const time = new Date(log.timestamp).toLocaleTimeString();
    const el = document.createElement('div');
    el.className = 'log-entry';
    el.innerHTML = `<span style="color:var(--text-muted);">[${time}]</span> <span class="log-tag ${log.category}">${log.category}</span> <span>${log.message}</span>`;
    diagnosticsLog.appendChild(el);
  });
  diagnosticsLog.scrollTop = diagnosticsLog.scrollHeight;
}

await hydrateLogs();
await renderDatabaseView();
const initDocs = await storage.getAllRAGDocs();
if (metricRagHits) metricRagHits.textContent = `${initDocs.length}`;
