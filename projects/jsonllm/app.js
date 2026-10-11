import { StorageService } from './storage.js';
import { MeshCoordinator } from './mesh.js';

const storage = new StorageService();
await storage.init();

// Register Service Worker
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('./sw.js').catch(console.error);
}

// UI Selectors
const chatStream = document.getElementById('chat-stream');
const inputPrompt = document.getElementById('input-prompt');
const btnSend = document.getElementById('btn-send');
const engineStatus = document.getElementById('engine-status');

const trainTextInput = document.getElementById('train-text-input');
const selectAgentTarget = document.getElementById('select-agent-target');
const btnTrainStep = document.getElementById('btn-train-step');
const btnSaveWeights = document.getElementById('btn-save-weights');
const metricTrainLoss = document.getElementById('metric-train-loss');
const metricTrainPpl = document.getElementById('metric-train-ppl');
const metricModelSize = document.getElementById('metric-model-size');

const ragDocId = document.getElementById('rag-doc-id');
const ragDocText = document.getElementById('rag-doc-text');
const btnIndexRAG = document.getElementById('btn-index-rag');
const ragIndexStatus = document.getElementById('rag-index-status');

const mcpRequestJson = document.getElementById('mcp-request-json');
const mcpResponseJson = document.getElementById('mcp-response-json');
const btnSendMcp = document.getElementById('btn-send-mcp');
const btnMcpInit = document.getElementById('btn-mcp-init');
const btnMcpTools = document.getElementById('btn-mcp-tools');
const btnMcpResources = document.getElementById('btn-mcp-resources');

const dbSummaryBadge = document.getElementById('db-summary-badge');
const dbChunksTbody = document.getElementById('db-chunks-tbody');
const btnRefreshDb = document.getElementById('btn-refresh-db');

const btnHostPin = document.getElementById('btn-host-pin');
const hostTokenOut = document.getElementById('host-token-out');
const joinTokenIn = document.getElementById('join-token-in');
const btnConnectPin = document.getElementById('btn-connect-pin');
const peerStatus = document.getElementById('peer-status');

const diagnosticsLog = document.getElementById('diagnostics-log');
const btnClearLogs = document.getElementById('btn-clear-logs');

// Tabs
const navTabs = document.querySelectorAll('.nav-tab');
const appViews = document.querySelectorAll('.app-view');

navTabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    const target = tab.getAttribute('data-target');
    navTabs.forEach((t) => t.classList.remove('active'));
    tab.classList.add('active');
    appViews.forEach((v) => v.classList.toggle('active', v.id === target));
    if (target === 'view-db') renderDatabaseView();
  });
});

async function log(category, msg) {
  await storage.logEvent(category, msg);
  const time = new Date().toLocaleTimeString();
  const el = document.createElement('div');
  el.textContent = `[${time}] [${category}] ${msg}`;
  diagnosticsLog.appendChild(el);
  diagnosticsLog.scrollTop = diagnosticsLog.scrollHeight;
}

// Background Workers
const llmWorker = new Worker('./llm-worker.js', { type: 'module' });
const webmcpWorker = new Worker('./webmcp-worker.js', { type: 'module' });

llmWorker.onmessage = async (e) => {
  const { type, status, domain, token, fullText, done } = e.data;
  if (type === 'STATUS') engineStatus.textContent = status;
  if (type === 'DOMAIN_SWITCH') engineStatus.textContent = `Active: agent-${domain}`;

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
  const { type, payload, loss, ppl, sizeKb } = e.data;
  if (type === 'WEBMCP_RESPONSE') {
    mcpResponseJson.textContent = JSON.stringify(payload, null, 2);
  }
  if (type === 'TRAIN_DONE') {
    metricTrainLoss.textContent = loss;
    metricTrainPpl.textContent = ppl;
    metricModelSize.textContent = `${sizeKb} KB`;
    btnTrainStep.disabled = false;
    log('TRAIN', `Training step done. Loss: ${loss}, Size: ${sizeKb} KB`);
    renderDatabaseView();
  }
};

// 1. Chat Execution
function executeChat() {
  const text = inputPrompt.value.trim();
  if (!text) return;

  const msg = document.createElement('div');
  msg.className = 'message user';
  msg.textContent = text;
  chatStream.appendChild(msg);
  inputPrompt.value = '';

  const delegated = mesh.dispatchCompute(text);
  if (delegated) {
    log('MESH', `Task dispatched to peer: ${delegated.peerKey}`);
  } else {
    llmWorker.postMessage({ type: 'INFER', payload: { prompt: text, maxTokens: 48 } });
  }
}
btnSend.addEventListener('click', executeChat);
inputPrompt.addEventListener('keydown', (e) => { if (e.key === 'Enter') executeChat(); });

// 2. Training Execution
btnTrainStep.addEventListener('click', () => {
  const text = trainTextInput.value.trim();
  if (!text) return alert('Enter training text.');
  const domain = selectAgentTarget.value;
  btnTrainStep.disabled = true;
  webmcpWorker.postMessage({
    type: 'TRAIN_CORPUS',
    payload: { domain, text, epochs: 5, lr: 0.003 }
  });
});

btnSaveWeights.addEventListener('click', () => {
  const text = trainTextInput.value.trim() || 'dialogue system ready.';
  const domain = selectAgentTarget.value;
  webmcpWorker.postMessage({
    type: 'TRAIN_CORPUS',
    payload: { domain, text, epochs: 1, lr: 0.001 }
  });
});

// 3. Vector RAG Indexing
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
  log('RAG', `Indexed knowledge: ${id}`);
});

// 4. WebMCP Buttons
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

// 5. Database View
async function renderDatabaseView() {
  const segments = await storage.getAllSegments();
  dbSummaryBadge.textContent = `${segments.length} segment(s)`;

  if (!segments.length) {
    dbChunksTbody.innerHTML = '<tr><td colspan="4" style="text-align: center;">No segments cached.</td></tr>';
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
btnRefreshDb.addEventListener('click', renderDatabaseView);

// 6. WebRTC Mesh
const nodeId = 'Node_' + Math.random().toString(36).substring(2, 6);
document.getElementById('node-label').textContent = `Node: ${nodeId}`;

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
  log('MESH', `Host PIN generated: ${pin}`);
});

btnConnectPin.addEventListener('click', async () => {
  const tok = joinTokenIn.value.trim();
  if (!tok) return;
  try {
    const raw = MeshCoordinator.decodePayload(tok);
    if (raw.sdp.type === 'offer') {
      const answer = await mesh.acceptTokenAndCreateAnswer(tok);
      joinTokenIn.value = answer;
      log('MESH', 'Accepted offer; return answer token to host.');
    } else if (raw.sdp.type === 'answer') {
      await mesh.finalizeHostConnection(tok);
      log('MESH', 'Peer connected via PIN.');
    }
  } catch (err) {
    alert('Mesh error: ' + err.message);
  }
});

btnClearLogs.addEventListener('click', async () => {
  await storage.clearLogs();
  diagnosticsLog.innerHTML = '';
});

renderDatabaseView();
