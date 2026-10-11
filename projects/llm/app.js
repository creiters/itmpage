import { StorageService } from './storage.js';
import { MeshCoordinator } from './mesh.js';

const nodeId = 'Node_' + Math.random().toString(36).substring(2, 7);
const nodeLabel = document.getElementById('node-label');
if (nodeLabel) nodeLabel.textContent = `Node ID: ${nodeId}`;

const chatStream = document.getElementById('chat-stream');
const inputPrompt = document.getElementById('input-prompt');
const btnSend = document.getElementById('btn-send');
const engineStatus = document.getElementById('engine-status');

// Training UI Selectors
const trainTextInput = document.getElementById('train-text-input');
const btnTrainStep = document.getElementById('btn-train-step');
const btnSaveWeights = document.getElementById('btn-save-weights');
const trainMetricsBadge = document.getElementById('train-metrics-badge');

// WebMCP UI Selectors
const mcpRequestJson = document.getElementById('mcp-request-json');
const mcpResponseJson = document.getElementById('mcp-response-json');
const btnSendMcp = document.getElementById('btn-send-mcp');
const btnMcpInit = document.getElementById('btn-mcp-init');
const btnMcpTools = document.getElementById('btn-mcp-tools');
const btnMcpResources = document.getElementById('btn-mcp-resources');
const btnMcpSample = document.getElementById('btn-mcp-sample');

// Database View Selectors
const btnRefreshDb = document.getElementById('btn-refresh-db');
const dbChunksTbody = document.getElementById('db-chunks-tbody');
const dbSummaryBadge = document.getElementById('db-summary-badge');

// WebRTC Selectors
const peerStatus = document.getElementById('peer-status');
const hostTokenOut = document.getElementById('host-token-out');
const joinTokenIn = document.getElementById('join-token-in');
const btnHostPin = document.getElementById('btn-host-pin');
const btnConnectPin = document.getElementById('btn-connect-pin');

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

    if (targetId === 'view-db') await renderDatabaseView();
  });
});

// Initialize Storage and WebMCP Worker
const storage = new StorageService();
await storage.init();

const webmcpWorker = new Worker('./webmcp-worker.js', { type: 'module' });
webmcpWorker.postMessage({ type: 'INIT_WEBMCP' });

webmcpWorker.onmessage = async (e) => {
  const { type, status, payload } = e.data;

  if (type === 'STATUS' && engineStatus) {
    engineStatus.textContent = status;
  }

  if (type === 'WEBMCP_RESPONSE') {
    if (mcpResponseJson) {
      mcpResponseJson.textContent = JSON.stringify(payload, null, 2);
    }
    // Forward sampling results to chat if active
    if (payload?.result?.content?.text) {
      updateAssistantBubble(payload.result.content.text);
      storage.saveMessage('assistant', payload.result.content.text);
    }
  }

  if (type === 'TRAIN_RESULT' && trainMetricsBadge) {
    trainMetricsBadge.textContent = `Loss: ${payload.loss.toFixed(4)} (${payload.textLength} chars processed)`;
    btnTrainStep.disabled = false;
  }

  if (type === 'SAVE_COMPLETE') {
    alert(`Model weights partitioned and saved! Merkle Root: ${payload.rootHash.slice(0, 10)}...`);
    await renderDatabaseView();
  }
};

// WebMCP Preset Buttons
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
      alert('Invalid JSON in request: ' + err.message);
    }
  });
}

// On-Device Training Triggers
if (btnTrainStep) {
  btnTrainStep.addEventListener('click', () => {
    const text = trainTextInput.value.trim();
    if (!text) return alert('Enter text to train the local model.');
    btnTrainStep.disabled = true;
    trainMetricsBadge.textContent = 'Computing forward & backward pass...';
    webmcpWorker.postMessage({ type: 'TRAIN_STEP', payload: { text, lr: 0.001 } });
  });
}

if (btnSaveWeights) {
  btnSaveWeights.addEventListener('click', () => {
    webmcpWorker.postMessage({ type: 'SAVE_MODEL_TO_MERKLE', payload: { modelName: 'browser-nano-llm' } });
  });
}

// Render Database Viewer
async function renderDatabaseView() {
  const summary = await storage.getChunkSummary('browser-nano-llm');
  if (dbSummaryBadge) {
    dbSummaryBadge.textContent = `${summary.length} chunks stored in IndexedDB`;
  }
  if (!dbChunksTbody) return;
  if (!summary.length) {
    dbChunksTbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No records found. Train or save a model first.</td></tr>';
    return;
  }
  dbChunksTbody.innerHTML = summary
    .map((c) => `
      <tr>
        <td>${c.chunkId}</td>
        <td>L${c.layerRange[0]}-L${c.layerRange[1]}</td>
        <td>${(c.sizeBytes / 1024 / 1024).toFixed(2)} MB</td>
        <td>${c.hash.slice(0, 8)}...</td>
      </tr>
    `)
    .join('');
}

if (btnRefreshDb) btnRefreshDb.addEventListener('click', renderDatabaseView);

// Chat Coordination
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
    chatStream.scrollTop = chatStream.scrollHeight;
  }
}

async function executeChat() {
  const text = inputPrompt.value.trim();
  if (!text) return;

  renderMessage('user', text);
  inputPrompt.value = '';
  currentBubble = null;
  await storage.saveMessage('user', text);

  // Send request via standard WebMCP JSON-RPC Sampling method
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

btnSend.addEventListener('click', executeChat);
inputPrompt.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') executeChat();
});

// WebRTC Peer Mesh Setup
const mesh = new MeshCoordinator(
  nodeId,
  async (msg, peerKey, channel) => {
    // Process JSON-RPC messages across WebRTC data channels
    if (msg.jsonrpc === '2.0') {
      webmcpWorker.postMessage({ type: 'WEBMCP_REQUEST', payload: msg });
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
      alert('Handshake failed: ' + err.message);
    }
  });
}
