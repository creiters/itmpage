import { MeshiDB } from './db.js';
import { MeshiSignaler } from './signaling.js';
import { MeshiMeshSocket } from './mesh_socket.js';
import { registerWebMCP } from './webmcp.js';

class MeshiApp {
  constructor() {
    this.nodeId = `sshanet_${Math.random().toString(36).substring(2, 7)}`;
    this.db = new MeshiDB();
    this.activeSocket = null;
    this.pendingPc = null;

    // Element references
    this.nodeStatusEl = document.getElementById('node-status');
    this.signalTokenInput = document.getElementById('signal-token');
    this.btnCreateOffer = document.getElementById('btn-create-offer');
    this.btnAcceptOffer = document.getElementById('btn-accept-offer');
    this.btnFinalize = document.getElementById('btn-finalize');
    this.payloadInput = document.getElementById('payload-input');
    this.btnSend = document.getElementById('btn-send');
    this.terminalBody = document.getElementById('term-body');
  }

  async init() {
    await this.db.open();
    this.bindEvents();
    registerWebMCP(this, this.db);
    this.initPWA();
    this.appendLog(`SYSTEM: Node ${this.nodeId} initialized. IndexedDB ready.`);
    await this.loadInitialMetadata();
  }

  async loadInitialMetadata() {
    try {
      const res = await fetch('./data/meta.json');
      const meta = await res.json();
      document.title = meta.title;
      this.appendLog("META: Loaded SSHAnet node descriptors.");
    } catch {
      this.appendLog("META: Using fallback configuration.");
    }
  }

  bindEvents() {
    // 1. Host creates Offer
    this.btnCreateOffer.addEventListener('click', async () => {
      try {
        const { pc, dc, token } = await MeshiSignaler.createOffer();
        this.pendingPc = pc;
        this.signalTokenInput.value = token;
        this.mountSocket(new MeshiMeshSocket('peer-responder', dc));
        this.appendLog("OFFER: Token generated. Transmit to Responder node.");
      } catch (err) {
        this.appendLog(`OFFER_ERROR: ${err.message}`);
      }
    });

    // 2. Responder accepts Offer and creates Answer
    this.btnAcceptOffer.addEventListener('click', async () => {
      try {
        const token = this.signalTokenInput.value.trim();
        if (!token) throw new Error("Token field is empty.");
        const { pc, dcPromise, token: answerToken } = await MeshiSignaler.acceptOffer(token);
        this.signalTokenInput.value = answerToken;
        const dc = await dcPromise;
        this.mountSocket(new MeshiMeshSocket('peer-initiator', dc));
        this.appendLog("ANSWER: Token generated. Send back to Host node.");
      } catch (err) {
        this.appendLog(`ANSWER_ERROR: ${err.message}`);
      }
    });

    // 3. Host finalizes link
    this.btnFinalize.addEventListener('click', async () => {
      try {
        const token = this.signalTokenInput.value.trim();
        if (!token || !this.pendingPc) throw new Error("Missing answer token or invalid state.");
        await MeshiSignaler.finalizeHandshake(this.pendingPc, token);
        this.appendLog("HANDSHAKE: Handshake dispatched. Awaiting DataChannel open...");
      } catch (err) {
        this.appendLog(`FINALIZE_ERROR: ${err.message}`);
      }
    });

    // Dispatch payload
    this.btnSend.addEventListener('click', async () => {
      const content = this.payloadInput.value.trim();
      if (!content) return;
      if (!this.activeSocket || this.activeSocket.readyState !== MeshiMeshSocket.OPEN) {
        this.appendLog("TRANSMIT_ERROR: No active WebRTC DataChannel connection.");
        return;
      }

      const doc = await this.db.putDocument({
        id: `meshi_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
        content,
        sender: this.nodeId
      });

      this.activeSocket.send({ type: 'DOC_SYNC', payload: doc });
      this.appendLog(`TX: Dispatched record -> ${doc.id}`);
      this.payloadInput.value = '';
    });
  }

  mountSocket(socket) {
    this.activeSocket = socket;

    socket.onopen = async () => {
      this.nodeStatusEl.textContent = 'STATUS: MESH_ACTIVE (RTC)';
      this.nodeStatusEl.classList.add('badge-online');
      this.appendLog("NET: Direct P2P RTCDataChannel link active.");

      // Sync initial local dataset to peer
      const localDocs = await this.db.getAllDocuments();
      socket.send({ type: 'BATCH_SYNC', payload: localDocs });
      this.appendLog(`TX_SYNC: Transmitted ${localDocs.length} local records to peer.`);
    };

    socket.onmessage = async (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === 'BATCH_SYNC') {
          let mergedCount = 0;
          for (const item of msg.payload) {
            const { merged } = await this.db.mergeRemoteDocument(item);
            if (merged) mergedCount++;
          }
          this.appendLog(`RX_SYNC: Merged ${mergedCount}/${msg.payload.length} incoming records.`);
        } else if (msg.type === 'DOC_SYNC') {
          const { merged, doc } = await this.db.mergeRemoteDocument(msg.payload);
          if (merged) {
            this.appendLog(`RX_UPDATE: [${doc.id}] "${doc.content}"`);
          }
        }
      } catch {
        this.appendLog(`RX_RAW: ${e.data}`);
      }
    };

    socket.onclose = () => {
      this.nodeStatusEl.textContent = 'STATUS: STANDALONE';
      this.nodeStatusEl.classList.remove('badge-online');
      this.appendLog("NET: RTCDataChannel disconnected.");
    };
  }

  appendLog(line) {
    const p = document.createElement('p');
    p.className = 'terminal-log-entry';
    p.textContent = `[${new Date().toLocaleTimeString()}] ${line}`;
    this.terminalBody.appendChild(p);
    this.terminalBody.scrollTop = this.terminalBody.scrollHeight;
  }

  initPWA() {
    if ('serviceWorker' in navigator) {
      window.addEventListener('load', () => {
        navigator.serviceWorker.register('./sw.js').catch((err) => console.warn('SW error:', err));
      });
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new MeshiApp().init();
});
