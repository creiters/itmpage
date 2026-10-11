import { StorageService } from './storage.js';
import { MeshCoordinator } from './mesh.js';

let currentLogFilter = 'ALL';

const storage = new StorageService();
await storage.init();

const nodeId = 'Node_' + Math.random().toString(36).substring(2, 7);
document.getElementById('node-label').textContent = `Node ID: ${nodeId}`;

const diagnosticsLog = document.getElementById('diagnostics-log');
const btnClearLogs = document.getElementById('btn-clear-logs');
const btnFilterLogs = document.getElementById('btn-filter-logs');

async function appendLog(category, text) {
  await storage.logEvent(category, text);
  if (!diagnosticsLog) return;

  if (currentLogFilter !== 'ALL' && currentLogFilter !== category) {
    return;
  }

  const time = new Date().toLocaleTimeString();
  const el = document.createElement('div');
  el.className = 'log-entry';
  el.innerHTML = `<span style="color:var(--text-muted); flex-shrink:0;">[${time}]</span> <span class="log-tag ${category}">${category}</span> <span>${text}</span>`;
  diagnosticsLog.appendChild(el);
  diagnosticsLog.scrollTop = diagnosticsLog.scrollHeight;
}

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

const inputDatasetName = document.getElementById('input-dataset-name');
const dialogPairsInput = document.getElementById('dialog-pairs-input');
const btnSaveDialogDataset = document.getElementById('btn-save-dialog-dataset');
const btnLoadIntoTraining = document.getElementById('btn-load-into-training');
const btnExportDialogFile = document.getElementById('btn-export-dialog-file');
const dialogSaveStatus = document.getElementById('dialog-save-status');
const dbDialogsTbody = document.getElementById('db-dialogs-tbody');

const ragDocId = document.getElementById('rag-doc-id');
const ragDocText = document.getElementById('rag-doc-text');
const btnIndexRAG = document.getElementById('btn-index-rag');
const btnTestRagSearch = document.getElementById('btn-test-rag-search');
const ragIndexStatus = document.getElementById('rag-index-status');
const metricRagHits = document.getElementById('metric-rag-hits');
const dbModelsTbody = document.getElementById('db-models-tbody');
const btnRefreshDb = document.getElementById('btn-refresh-db');
const btnClearDb = document.getElementById('btn-clear-db');

const btnHostPin = document.getElementById('btn-host-pin');
const hostTokenOut = document.getElementById('host-token-out');
const joinTokenIn = document.getElementById('join-token-in');
const btnConnectPin = document.getElementById('btn-connect-pin');
const peerStatus = document.getElementById('peer-status');

const mcpRequestJson = document.getElementById('mcp-request-json');
const mcpResponseJson = document.getElementById('mcp-response-json');
const btnSendMcp = document.getElementById('btn-send-mcp');
const btnMcpInit = document.getElementById('btn-mcp-init');
const btnMcpTools = document.getElementById('btn-mcp-tools');
const btnMcpResources = document.getElementById('btn-mcp-resources');

const metricTPS = document.getElementById('metric-tps');
const metricTTFT = document.getElementById('metric-ttft');
const tokenTopKBars = document.getElementById('token-topk-bars');

const navTabs = document.querySelectorAll('.nav-tab');
const appViews = document.querySelectorAll('.app-view');

navTabs.forEach((tab) => {
  tab.addEventListener('click', async () => {
    const targetId = tab.getAttribute('data-target');
    navTabs.forEach((t) => t.classList.remove('active'));
    tab.classList.add('active');

    appViews.forEach((v) => v.classList.toggle('active', v.id === targetId));
    appendLog('UI', `Navigated to tab: ${targetId}`);

    if (targetId === 'view-db') await renderDatabaseView();
    if (targetId === 'view-train') await renderDialogueListView();
  });
});

const llmWorker = new Worker('./llm-worker.js', { type: 'module' });
const webmcpWorker = new Worker('./webmcp-worker.js', { type: 'module' });

llmWorker.onmessage = async (e) => {
  const { type, status, token, fullText, done, tps, ttft, topCandidates, category, text } = e.data;

  if (type === 'STATUS') {
    engineStatus.textContent = status;
    appendLog('STATUS', `LLM Worker status: ${status}`);
  }

  if (type === 'LOG') {
    await appendLog(category || 'INFER', text);
  }

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
  const { type, status, payload, loss, ppl, category, text } = e.data;

  if (type === 'STATUS') {
    engineStatus.textContent = status;
    appendLog('STATUS', `WebMCP Worker status: ${status}`);
  }

  if (type === 'LOG') {
    await appendLog(category || 'WEBMCP', text);
  }

  if (type === 'WEBMCP_RESPONSE' && mcpResponseJson) {
    mcpResponseJson.textContent = JSON.stringify(payload, null, 2);
    appendLog('WEBMCP', `JSON-RPC response received for id=${payload.id}`);
  }

  if (type === 'TRAIN_PROGRESS') {
    metricTrainLoss.textContent = payload.loss.toFixed(4);
    metricTrainPpl.textContent = payload.perplexity.toFixed(1);
    trainProgressFill.style.width = `${payload.progressPct}%`;
    trainProgressText.textContent = `Epoch ${payload.epoch}/${payload.totalEpochs} — ${payload.progressPct}% (${payload.tokensPerSec} tok/s)`;
  }

  if (type === 'TRAIN_DONE') {
    metricTrainLoss.textContent = loss;
    metricTrainPpl.textContent = ppl;
    btnStartTraining.disabled = false;
    trainProgressText.textContent = 'Training finished. Ready to generate JSONLLM.';
    await appendLog('TRAIN', `Training converged successfully. Loss: ${loss}, PPL: ${ppl}`);
  }

  if (type === 'GENERATE_JSONLLM_SUCCESS') {
    metricModelSize.textContent = `${payload.sizeKb} KB`;
    btnGenerateJsonllm.disabled = false;
    alert(`Generated 4M ${payload.mode.toUpperCase()} Model (${payload.sizeKb} KB) and saved to IndexedDB!`);
    await appendLog('STORAGE', `Saved 4M ${payload.mode.toUpperCase()} model: ${payload.modelName} (${payload.sizeKb} KB)`);
    llmWorker.postMessage({ type: 'HOT_RELOAD', payload: { modelName: payload.modelName } });
    await renderDatabaseView();
  }

  if (type === 'SAVE_DIALOGUE_COMPLETE') {
    dialogSaveStatus.textContent = `Saved ${payload.datasetName}: ${payload.pairCount} pairs in ${payload.sizeKb} KB!`;
    await appendLog('STORAGE', `Compressed dialogue dataset "${payload.datasetName}" stored (${payload.sizeKb} KB)`);
    await renderDialogueListView();
  }
};

function executeChat() {
  const text = inputPrompt.value.trim();
  if (!text) return;

  const msg = document.createElement('div');
  msg.className = 'message user';
  msg.textContent = text;
  chatStream.appendChild(msg);
  inputPrompt.value = '';
  chatStream.scrollTop = chatStream.scrollHeight;

  appendLog('INFER', `User dispatched prompt: "${text}"`);

  const delegated = mesh.dispatchCompute(text);
  if (delegated) {
    appendLog('WEBRTC', `Compute delegated to connected peer: ${delegated.peerKey}`);
  } else {
    llmWorker.postMessage({ type: 'INFER', payload: { prompt: text, maxTokens: 32 } });
  }
}
btnSend.addEventListener('click', executeChat);
inputPrompt.addEventListener('keydown', (e) => { if (e.key === 'Enter') executeChat(); });

btnStartTraining.addEventListener('click', () => {
  const text = trainTextInput.value.trim();
  if (!text) return alert('Enter training text.');
  const epochs = parseInt(inputEpochs.value, 10) || 5;

  btnStartTraining.disabled = true;
  trainProgressFill.style.width = '0%';
  trainProgressText.textContent = 'Training with FP16 autograd & AdamW...';

  appendLog('TRAIN', `Dispatched training task: ${text.length} chars, ${epochs} epochs`);
  webmcpWorker.postMessage({
    type: 'START_TRAIN',
    payload: { text, epochs, lr: 0.003 }
  });
});

btnGenerateJsonllm.addEventListener('click', () => {
  btnGenerateJsonllm.disabled = true;
  appendLog('STORAGE', `Requested JSONLLM generation with mode: ${selectQuantMode.value}`);
  webmcpWorker.postMessage({
    type: 'GENERATE_JSONLLM',
    payload: { modelName: 'vimtollm-4m', mode: selectQuantMode.value }
  });
});

btnSaveDialogDataset.addEventListener('click', () => {
  const name = inputDatasetName.value.trim() || 'dialogue_dataset';
  try {
    const pairs = JSON.parse(dialogPairsInput.value.trim());
    if (!Array.isArray(pairs)) throw new Error('Data must be an array of {"u": "...", "b": "..."} objects.');
    dialogSaveStatus.textContent = 'Compressing dialogue dataset...';
    appendLog('STORAGE', `Compressing dialogue dataset "${name}" with ${pairs.length} pairs...`);
    webmcpWorker.postMessage({
      type: 'SAVE_COMPACT_DIALOGUE',
      payload: { datasetName: name, pairs }
    });
  } catch (err) {
    alert('Invalid JSON pairs: ' + err.message);
    appendLog('ERROR', `Failed parsing dialogue JSON: ${err.message}`);
  }
});

btnLoadIntoTraining.addEventListener('click', () => {
  try {
    const pairs = JSON.parse(dialogPairsInput.value.trim());
    const formatted = pairs.map((p) => `<user> ${p.u} <bot> ${p.b} <eos>`).join('\n');
    trainTextInput.value = formatted;
    appendLog('UI', `Converted ${pairs.length} dialogue pairs into training text.`);
  } catch (err) {
    alert('Could not convert pairs: ' + err.message);
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
  appendLog('STORAGE', `Exported portable dataset file: ${name}.dialog.jsonllm`);
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
      <td>${(d.bytes / 1024).toFixed(2)} KB</td>
      <td><code>token-delta</code></td>
      <td><button class="secondary-btn small-btn" data-load-name="${d.datasetName}">Load</button></td>
    </tr>
  `).join('');

  dbDialogsTbody.querySelectorAll('button[data-load-name]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const name = btn.getAttribute('data-load-name');
      const rec = await storage.getDialogueDataset(name);
      if (rec) {
        dialogPairsInput.value = rec;
        inputDatasetName.value = name;
        appendLog('UI', `Loaded stored dataset "${name}" into editor.`);
      }
    });
  });
}

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
  await appendLog('STORAGE', `Indexed document "${id}" into Vector RAG.`);
  const docs = await storage.getAllRAGDocs();
  if (metricRagHits) metricRagHits.textContent = `${docs.length}`;
});

btnTestRagSearch.addEventListener('click', async () => {
  const text = ragDocText.value.trim();
  if (!text) return alert('Enter text to test RAG similarity.');
  appendLog('RAG', `Executing test similarity query on: "${text}"`);
  llmWorker.postMessage({ type: 'INFER', payload: { prompt: text, maxTokens: 1 } });
});

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
  appendLog('STORAGE', 'IndexedDB completely purged.');
  window.location.reload();
});

const mesh = new MeshCoordinator(
  nodeId,
  (msg) => {
    appendLog('WEBRTC', `Received P2P message: ${msg.type}`);
    if (msg.type === 'MESH_INFER') {
      llmWorker.postMessage({ type: 'INFER', payload: { prompt: msg.prompt, maxTokens: 32 } });
    }
  },
  (count) => {
    peerStatus.textContent = `${count} Peers Connected`;
    appendLog('WEBRTC', `Peer connection topology updated: ${count} active peer(s).`);
  }
);

btnHostPin.addEventListener('click', async () => {
  const pin = mesh.createPin();
  const token = await mesh.generateHostToken(pin);
  hostTokenOut.value = token;
  await appendLog('WEBRTC', `Host PIN created: ${pin}`);
});

btnConnectPin.addEventListener('click', async () => {
  const tok = joinTokenIn.value.trim();
  if (!tok) return;

  try {
    const raw = MeshCoordinator.decodePayload(tok);
    if (raw.sdp.type === 'offer') {
      const answer = await mesh.acceptTokenAndCreateAnswer(tok);
      joinTokenIn.value = answer;
      appendLog('WEBRTC', 'Accepted remote offer; answer token generated.');
    } else if (raw.sdp.type === 'answer') {
      await mesh.finalizeHostConnection(tok);
      appendLog('WEBRTC', 'Connected to peer host via PIN.');
    }
  } catch (err) {
    alert('Handshake error: ' + err.message);
    appendLog('ERROR', `WebRTC handshake error: ${err.message}`);
  }
});

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
    appendLog('WEBMCP', `Dispatched JSON-RPC method: ${parsed.method}`);
    webmcpWorker.postMessage({ type: 'WEBMCP_REQUEST', payload: parsed });
  } catch (err) {
    alert('Invalid JSON: ' + err.message);
    appendLog('ERROR', `Invalid WebMCP JSON: ${err.message}`);
  }
});

btnClearLogs.addEventListener('click', async () => {
  await storage.clearLogs();
  diagnosticsLog.innerHTML = '';
});

const logFilters = ['ALL', 'INFER', 'TRAIN', 'STORAGE', 'RAG', 'WEBRTC', 'WEBMCP', 'ERROR'];
btnFilterLogs.addEventListener('click', () => {
  const curIdx = logFilters.indexOf(currentLogFilter);
  currentLogFilter = logFilters[(curIdx + 1) % logFilters.length];
  btnFilterLogs.textContent = currentLogFilter === 'ALL' ? 'All Logs' : `${currentLogFilter} Only`;
  diagnosticsLog.innerHTML = '';
  hydrateLogs();
});

async function hydrateLogs() {
  const previousLogs = await storage.getRecentLogs(60);
  previousLogs.forEach((log) => {
    if (currentLogFilter !== 'ALL' && currentLogFilter !== log.category) return;
    const time = new Date(log.timestamp).toLocaleTimeString();
    const el = document.createElement('div');
    el.className = 'log-entry';
    el.innerHTML = `<span style="color:var(--text-muted); flex-shrink:0;">[${time}]</span> <span class="log-tag ${log.category}">${log.category}</span> <span>${log.message}</span>`;
    diagnosticsLog.appendChild(el);
  });
  diagnosticsLog.scrollTop = diagnosticsLog.scrollHeight;
}

await hydrateLogs();
await renderDatabaseView();
await renderDialogueListView();
const initDocs = await storage.getAllRAGDocs();
if (metricRagHits) metricRagHits.textContent = `${initDocs.length}`;
