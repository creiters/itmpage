import { MeshiCrypto } from './crypto.js';
import { MeshiDB } from './db.js';
import { MeshiSignaler } from './signaling.js';
import { MeshiMeshSocket } from './mesh_socket.js';
import { registerWebMCP } from './webmcp.js';

class MeshiApp {
  constructor() {
    this.nodeId = `sshanet_${Math.random().toString(36).substring(2, 7)}`;
    this.db = new MeshiDB();
    this.crypto = null;
    this.activeSocket = null;
    this.pendingPc = null;

    // UI Nodes
    this.statusBadge = document.getElementById('node-status');
    this.authBadge = document.getElementById('auth-status');
    this.localKeyDisplay = document.getElementById('local-key');
    this.tokenArea = document.getElementById('signal-token');
    this.peerKeyInput = document.getElementById('peer-key-input');
    this.btnTrustKey = document.getElementById('btn-trust-key');
    this.btnCreateOffer = document.getElementById('btn-create-offer');
    this.btnAcceptOffer = document.getElementById('btn-accept-offer');
    this.btnFinalize = document.getElementById('btn-finalize');
    this.inputPayload = document.getElementById('payload-input');
    this.btnSend = document.getElementById('btn-send');
    this.terminal = document.getElementById('term-log');
  }

  async init() {
    await this.db.open();
    this.crypto = await MeshiCrypto.generateIdentity();
    this.localKeyDisplay.value = this.crypto.publicKeyBase64;

    // Self-trust local key for loopback testing
    await this.db.addTrustedPeer(this.crypto.publicKeyBase64, 'Local Loopback Node');

    this.bindEvents();
    registerWebMCP(this, this.db);
    this.initPWA();
    await this.loadMetadata();
    this.appendLog(`INIT: Node ${this.nodeId} active. ECDSA P-256 key generated.`);
  }

  async loadMetadata() {
    try {
      const res = await fetch('./data/meta.json');
      const meta = await res.json();
      document.title = meta.title;

      const script = document.createElement('script');
      script.type = 'application/ld+json';
      script.textContent = JSON.stringify(meta.schema);
      document.head.appendChild(script);
      this.appendLog("META: SSHAnet schema and GEO descriptors mounted.");
    } catch {
      this.appendLog("META: Using standalone fallback.");
    }
  }

  bindEvents() {
    this.btnTrustKey.addEventListener('click', async () => {
      const key = this.peerKeyInput.value.trim();
      if (!key) return;
      await this.db.addTrustedPeer(key, 'Operator Whitelisted');
      this.appendLog(`AUTH: Whitelisted peer key: ${key.substring(0, 24)}...`);
      this.peerKeyInput.value = '';
    });

    this.btnCreateOffer.addEventListener('click', async () => {
      try {
        const { pc, dc, token } = await MeshiSignaler.createOffer();
        this.pendingPc = pc;
        this.tokenArea.value = token;
        this.mountSocket(new MeshiMeshSocket('remote-peer', dc, this.crypto, this.db));
        this.appendLog("OFFER: Generated Offer. Copy and transmit to Responder.");
      } catch (err) {
        this.appendLog(`OFFER_ERR: ${err.message}`);
      }
    });

    this.btnAcceptOffer.addEventListener('click', async () => {
      try {
        const token = this.tokenArea.value.trim();
        if (!token) throw new Error("Token field empty.");
        const { pc, dcPromise, token: answerToken } = await MeshiSignaler.acceptOffer(token);
        this.tokenArea.value = answerToken;
        const dc = await dcPromise;
        this.mountSocket(new MeshiMeshSocket('remote-peer', dc, this.crypto, this.db));
        this.appendLog("ANSWER: Generated Answer. Transmit back to Host node.");
      } catch (err) {
        this.appendLog(`ANSWER_ERR: ${err.message}`);
      }
    });

    this.btnFinalize.addEventListener('click', async () => {
      try {
        const token = this.tokenArea.value.trim();
        if (!token || !this.pendingPc) throw new Error("Missing token or invalid peer connection.");
        await MeshiSignaler.finalizeHandshake(this.pendingPc, token);
        this.appendLog("HANDSHAKE: Handshake applied. Awaiting RTCDataChannel open state...");
      } catch (err) {
        this.appendLog(`FINALIZE_ERR: ${err.message}`);
      }
    });

    this.btnSend.addEventListener('click', async () => {
      const text = this.inputPayload.value.trim();
      if (!text) return;
      if (!this.activeSocket || this.activeSocket.readyState !== MeshiMeshSocket.OPEN || !this.activeSocket.isAuthorized) {
        this.appendLog("ERROR: Channel is not open and mutually authorized.");
        return;
      }

      const doc = await this.db.putDocument({
        id: `meshi_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
        content: text,
        sender: this.nodeId
      });

      this.activeSocket.send({ type: 'DOC_SYNC', payload: doc });
      this.appendLog(`TX: Document persisted & dispatched [Clock: ${doc.clock}] -> ${doc.id}`);
      this.inputPayload.value = '';
    });
  }

  mountSocket(socket) {
    this.activeSocket = socket;

    socket.onopen = () => {
      this.statusBadge.textContent = 'LINK: CONNECTED';
      this.statusBadge.classList.add('badge-online');
      this.appendLog("NET: Raw WebRTC DataChannel established. Initiating mutual auth...");
    };

    socket.addEventListener('meshi:authorized', (e) => {
      this.authBadge.textContent = 'AUTH: TRUSTED';
      this.authBadge.classList.add('badge-auth');
      this.appendLog(`AUTH: Mutual ECDSA challenge verified. Key: ${e.detail.remoteKey.substring(0, 18)}...`);
    });

    socket.addEventListener('meshi:synced', (e) => {
      this.appendLog(`SYNC: Catch-up completed. Merged ${e.detail.count} missing records.`);
    });

    socket.addEventListener('meshi:net-restored', () => {
      this.appendLog("NET: Device back online. Querying peer for delta updates...");
    });

    socket.onmessage = async (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === 'DOC_SYNC') {
          const { merged, doc } = await this.db.mergeRemoteDocument(msg.payload);
          if (merged) {
            this.appendLog(`RX: Merged [Clock: ${doc.clock}] ${doc.id} -> "${doc.content}"`);
          }
        }
      } catch {
        this.appendLog(`RX_RAW: ${e.data}`);
      }
    };

    socket.onclose = () => {
      this.statusBadge.textContent = 'LINK: CLOSED';
      this.statusBadge.classList.remove('badge-online');
      this.authBadge.textContent = 'AUTH: UNVERIFIED';
      this.authBadge.classList.remove('badge-auth');
      this.appendLog("NET: Peer connection dropped.");
    };
  }

  appendLog(text) {
    const entry = document.createElement('div');
    entry.className = 'term-line';
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
    this.terminal.appendChild(entry);
    this.terminal.scrollTop = this.terminal.scrollHeight;
  }

  initPWA() {
    if ('serviceWorker' in navigator) {
      window.addEventListener('load', () => {
        navigator.serviceWorker.register('./sw.js').catch((err) => console.warn('SW failed:', err));
      });
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new MeshiApp().init();
});
