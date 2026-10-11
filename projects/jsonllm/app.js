/**
 * app.js - Master Application Coordinator with Compact Dialogue Serialization
 */

import { StorageService } from './storage.js';
import { MeshCoordinator } from './mesh.js';

const storage = new StorageService();
await storage.init();

const nodeId = 'Node_' + Math.random().toString(36).substring(2, 7);
document.getElementById('node-label').textContent = `Node ID: ${nodeId}`;

// UI Handles
const chatStream = document.getElementById('chat-stream');
const inputPrompt = document.getElementById('input-prompt');
const btnSend = document.getElementById('btn-send');
const engineStatus = document.getElementById('engine-status');

const trainTextInput = document.getElementById('train-text-input');
const selectQuantMode = document.getElementById('select-quant-mode');
const inputEpochs = document.getElementById('input-epochs');
const btnStartTraining = document.getElementById('btn-start-training');
const btnGenerateJsonllm = document.getElementById('btn-generate-jsonllm');
const trainProgressFill = document.getElementById('train-progress-fill');
const trainProgressText = document.getElementById('train-progress-text');
const metricTrainLoss = document.getElementById('metric-train-loss');
const metricTrainPpl = document.getElementById('metric-train-ppl');
const metricModelSize = document.getElementById('metric-model-size');

// Dialogue Studio Handles
const inputDatasetName = document.getElementById('input-dataset-name');
const dialogPairsInput = document.getElementById('dialog-pairs-input');
const btnSaveDialogDataset = document.getElementById('btn-save-dialog-dataset');
const btnExportDialogFile = document.getElementById('btn-export-dialog-file');
const dialogSaveStatus = document.getElementById('dialog-save-status');
const dbDialogsTbody = document.getElementById('db-dialogs-tbody');
const btnRefreshDialogList = document.getElementById('btn-refresh-dialog-list');

// RAG Handles
const ragDocId = document.getElementById('rag-doc-id');
const ragDocText = document.getElementById('rag-doc-text');
const btnIndexRAG = document.getElementById('btn-index-rag');
const ragIndexStatus = document.getElementById('rag-index-status');
const metricRagHits = document.getElementById('metric-rag-hits');

// WebMCP Handles
const mcpRequestJson = document.getElementById('mcp-request-json');
const mcpResponseJson = document.getElementById('mcp-response-json');
const btnSendMcp = document.getElementById('btn-send-mcp');
const btnMcpInit = document.getElementById('btn-mcp-init');
const btnMcpTools = document.getElementById('btn-mcp-tools');
const btnMcpResources = document.getElementById('btn-mcp-resources');

// DB & Telemetry Handles
const dbModelsTbody = document.getElementById('db-models-tbody');
const btnRefreshDb = document.getElementById('btn-refresh-db');
const btnClearDb = document.getElementById('btn-clear-db');
const metricTPS = document.getElementById('metric-tps');
const metricTTFT = document.getElementById('metric-ttft');
const tokenTopKBars = document.getElementById('token-topk-bars');
const diagnosticsLog = document.getElementById('diagnostics-log');
const btnClearLogs = document.getElementById('btn-clear-logs');

// WebRTC Handles
const btnHostPin = document.getElementById('btn-host-pin');
const hostTokenOut = document.getElementById('host-token-out');
const joinTokenIn = document.getElementById('join-token-in');
const btnConnectPin = document.getElementById('btn-connect-pin');
const peerStatus = document.getElementById('peer-status');

// Tab Routing
const navTabs = document.querySelectorAll('.nav-tab');
const appViews = document.querySelectorAll('.app-view');

navTabs.forEach((tab) => {
  tab.addEventListener('click', async () => {
    const target = tab.getAttribute('data-target');
    navTabs.forEach((t) => t.classList.remove('active'));
    tab.classList.add('active');

    appViews.forEach((v) => v.classList.toggle('active', v.id === target));

    if (target === 'view-db') await renderDatabaseView();
    if (target === 'view-dialog-save') await renderDialogueListView();
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

// Background Workers
const llmWorker = new Worker('./llm-worker.js', { type: 'module' });
const webmcpWorker = new Worker('./webmcp-worker.js', { type: 'module' });

llmWorker.onmessage = async (e) => {
  const { type, status, token, fullText, done, tps, ttft, topCandidates } = e.data;

  if (type === 'STATUS') engineStatus.textContent = status;

  if (type === 'TELEMETRY') {
    if (metricTPS) metricTPS.textContent = tps;
    if (metricTTFT) metricTTFT.textContent = `${ttft} ms`;

    if (tokenTopKBars && topCandidates) {
      tokenTopKBars.innerHTML = topCandidates.map((c) => {
        const fillW = Math.max(10, Math.min(100, (parseFloat(c.logit) + 15) * 3));
        return `
          <div class="topk-bar-row">
            <span class="topk-label">${c.word}</span>
            <div class="topk-track"><div class="topk-fill" style="width:${fillW}%;"></div></div>
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
  const { type, status, payload, loss, ppl } = e.data;

  if (type === 'STATUS') engineStatus.textContent = status;
  if (type === 'WEBMCP_RESPONSE' && mcpResponseJson) {
    mcpResponseJson.textContent = JSON.stringify(payload, null, 2);
  }

  if (type === 'TRAIN_PROGRESS') {
    metricTrainLoss.textContent = payload.loss.toFixed(4);
    metricTrainPpl.textContent = payload.perplexity.toFixed(1);
    trainProgressFill.style.width = `${payload.progressPct}%`;
    trainProgressText.textContent = `Epoch ${payload.epoch}/${payload.totalEpochs} — ${payload.progressPct}%`;
  }

  if (type === 'TRAIN_DONE') {
    metricTrainLoss.textContent = loss;
    metricTrainPpl.textContent = ppl;
    btnStartTraining.disabled = false;
    trainProgressText.textContent = 'Training finished. Ready to generate JSONLLM.';
    await appendLog('TRAIN', `Training converged. Loss: ${loss}, PPL: ${ppl}`);
  }

  if (type === 'GENERATE_JSONLLM_SUCCESS') {
    metricModelSize.textContent = `${payload.sizeKb} KB`;
    btnGenerateJsonllm.disabled = false;
    alert(`Generated 4M ${payload.mode.toUpperCase()} Model (${payload.sizeKb} KB) and saved to IndexedDB!`);
    await appendLog('STORAGE', `Saved 4M ${payload.mode.toUpperCase()} model (${payload.sizeKb} KB)`);
    llmWorker.postMessage({ type: 'HOT_RELOAD', payload: { modelName: payload.modelName } });
    await renderDatabaseView();
  }

  if (type === 'SAVE_DIALOGUE_COMPLETE') {
    dialogSaveStatus.textContent = `Saved ${payload.datasetName}: ${payload.pairCount} pairs in ${payload.sizeKb} KB!`;
    await appendLog('STORAGE', `Saved compact dialogue dataset "${payload.datasetName}" (${payload.sizeKb} KB)`);
    await renderDialogueListView();
  }
};

// Chat Trigger
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
    appendLog('WEBRTC', `Dispatched compute to peer ${delegated.peerKey}`);
  } else {
    llmWorker.postMessage({ type: 'INFER', payload: { prompt: text, maxTokens: 32 } });
  }
}
btnSend.addEventListener('click', executeChat);
inputPrompt.addEventListener('keydown', (e) => { if (e.key === 'Enter') executeChat(); });

// Training Triggers
btnStartTraining.addEventListener('click', () => {
  const text = trainTextInput.value.trim();
  if (!text) return alert('Enter training text.');
  const epochs = parseInt(inputEpochs.value, 10) || 5;

  btnStartTraining.disabled = true;
  trainProgressFill.style.width = '0%';
  trainProgressText.textContent = 'Training with FP16 autograd & AdamW...';

  webmcpWorker.postMessage({
    type: 'START_TRAIN',
    payload: { text, epochs, lr: 0.003 }
  });
});

btnGenerateJsonllm.addEventListener('click', () => {
  btnGenerateJsonllm.disabled = true;
  webmcpWorker.postMessage({
    type: 'GENERATE_JSONLLM',
    payload: { modelName: 'dialogue-4m', mode: selectQuantMode.value }
  });
});

// Dialogue Studio Triggers (< 2 KB Data Saving)
btnSaveDialogDataset.addEventListener('click', () => {
  const name = inputDatasetName.value.trim() || 'dialogue_dataset';
  try {
    const pairs = JSON.parse(dialogPairsInput.value.trim());
    if (!Array.isArray(pairs)) throw new Error('Data must be an array of {"u": "...", "b": "..."} objects.');
    dialogSaveStatus.textContent = 'Compressing dialogue dataset...';
    webmcpWorker.postMessage({
      type: 'SAVE_COMPACT_DIALOGUE',
      payload: { datasetName: name, pairs }
    });
  } catch (err) {
    alert('Invalid JSON pairs: ' + err.message);
  }
});

btnExportDialogFile.addEventListener('click', async () => {
  const name = inputDatasetName.value.trim() || 'dialogue_dataset';
  const rawJSON = await storage.getDialogueDataset(name);
  if (!rawJSON) return alert(`Dataset "${name}" not found in IndexedDB.`);

  const blob = new Blob([rawJSON], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${name}.dialog.jsonllm`;
  a.click();
  URL.revokeObjectURL(url);
  await appendLog('STORAGE', `Exported ${name}.dialog.jsonllm file to disk`);
});

async function renderDialogueListView() {
  const list = await storage.getAllDialogueDatasets();
  if (!dbDialogsTbody) return;

  if (!list.length) {
    dbDialogsTbody.innerHTML = '<tr><td colspan="4" class="status-label">No datasets stored.</td></tr>';
    return;
  }

  dbDialogsTbody.innerHTML = list.map((d) => `
    <tr>
      <td><strong>${d.datasetName}</strong></td>
      <td>Multi-turn</td>
      <td>${(d.bytes / 1024).toFixed(2)} KB</td>
      <td><code>token-delta</code></td>
    </tr>
  `).join('');
}
btnRefreshDialogList.addEventListener('click', renderDialogueListView);

// RAG Triggers
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

// WebMCP Triggers
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

// DB View Renderer
async function renderDatabaseView() {
  const models = await storage.getAllModels();
  if (!dbModelsTbody) return;

  if (!models.length) {
    dbModelsTbody.innerHTML = '<tr><td colspan="4" class="status-label">No models stored.</td></tr>';
    return;
  }

  dbModelsTbody.innerHTML = models.map((m) => `
    <tr>
      <td><strong>${m.modelName}</strong></td>
      <td>${(m.bytes / 1024).toFixed(1)} KB</td>
      <td><code>jsonllm/v3</code></td>
      <td>FP16 / FP4</td>
    </tr>
  `).join('');
}

btnRefreshDb.addEventListener('click', renderDatabaseView);
btnClearDb.addEventListener('click', async () => {
  if (!confirm('Clear all stored models, datasets, and IndexedDB caches?')) return;
  await storage.clearAll();
  chatStream.innerHTML = '';
  window.location.reload();
});

// WebRTC PIN Mesh
const mesh = new MeshCoordinator(
  nodeId,
  (msg) => {
    if (msg.type === 'MESH_INFER') {
      llmWorker.postMessage({ type: 'INFER', payload: { prompt: msg.prompt, maxTokens: 32 } });
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

btnClearLogs.addEventListener('click', () => {
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
await renderDialogueListView();
