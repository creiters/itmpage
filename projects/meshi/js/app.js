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
    this.isReconnecting = false;

    // DOM Elements
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
    this.crypto = await MeshiCrypto.init(this.db);
    this.localKeyDisplay.value = this.crypto.publicKeyBase64;

    await this.db.addTrustedPeer(this.crypto.publicKeyBase64, 'Self Loopback');

    this.bindEvents();
    this.setupServiceWorkerWatchdog();
    registerWebMCP(this, this.db);
    await this.loadMetadata();

    this.appendLog(`INIT: Node ${this.nodeId} initialized.`);
    this.appendLog(`KEY: ECDSA SPKI identity persistent across reloads.`);

    // Automatically recover active connection if cached
    await this.attemptResumeLastSession();
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
    } catch {
      this.appendLog("META: Using standalone fallback.");
    }
  }

  bindEvents() {
    this.btnTrustKey.addEventListener('click', async () => {
      const key = this.peerKeyInput.value.trim();
      if (!key) return;
      await this.db.addTrustedPeer(key, 'Whitelisted Peer');
      this.appendLog(`AUTH: Peer key whitelisted.`);
      this.peerKeyInput.value = '';
    });

    this.btnCreateOffer.addEventListener('click', async () => {
      try {
        const { pc, dc, token } = await MeshiSignaler.createOffer(this.crypto);
        this.pendingPc = pc;
        this.tokenArea.value = token;
        await this.db.setSystemKey('last_offer_token', token);
        this.mountSocket(new MeshiMeshSocket('remote-peer', dc, this.crypto, this.db, () => this.handleSocketDropout()));
        this.appendLog("OFFER: Generated & signed with WebCrypto. Copy to responder.");
      } catch (err) {
        this.appendLog(`OFFER_ERR: ${err.message}`);
      }
    });

    this.btnAcceptOffer.addEventListener('click', async () => {
      try {
        const token = this.tokenArea.value.trim();
        if (!token) throw new Error("Token field empty.");
        const { pc, dcPromise, token: answerToken, remotePublicKey } = await MeshiSignaler.acceptOffer(token, this.crypto);
        
        await this.db.addTrustedPeer(remotePublicKey, 'Auto-Approved Signaling Peer');
        this.tokenArea.value = answerToken;
        await this.db.setSystemKey('last_peer_key', remotePublicKey);
        
        const dc = await dcPromise;
        this.mountSocket(new MeshiMeshSocket('remote-peer', dc, this.crypto, this.db, () => this.handleSocketDropout()));
        this.appendLog("ANSWER: Generated & signed. Send back to initiator.");
      } catch (err) {
        this.appendLog(`ANSWER_ERR: ${err.message}`);
      }
    });

    this.btnFinalize.addEventListener('click', async () => {
      try {
        const token = this.tokenArea.value.trim();
        if (!token || !this.pendingPc) throw new Error("Missing token or invalid peer connection.");
        const remotePubKey = await MeshiSignaler.finalizeHandshake(this.pendingPc, token, this.crypto);
        await this.db.addTrustedPeer(remotePubKey, 'Auto-Approved Signaling Peer');
        await this.db.setSystemKey('last_peer_key', remotePubKey);
        this.appendLog("HANDSHAKE: Verified & finalized via WebCrypto.");
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
      this.appendLog(`TX: Dispatched [Clock: ${doc.clock}] -> ${doc.id}`);
      this.inputPayload.value = '';
    });

    window.addEventListener('online', () => {
      this.appendLog("NET: Device online event detected. Running reconnection...");
      this.handleSocketDropout();
    });
  }

  mountSocket(socket) {
    this.activeSocket = socket;

    socket.dc.addEventListener('open', () => {
      this.statusBadge.textContent = 'LINK: CONNECTED';
      this.statusBadge.classList.add('badge-online');
      this.appendLog("NET: WebRTC DataChannel established.");
    });

    socket.addEventListener('meshi:authorized', (e) => {
      this.authBadge.textContent = 'AUTH: TRUSTED';
      this.authBadge.classList.add('badge-auth');
      this.appendLog(`AUTH: Mutual ECDSA verification succeeded.`);
    });

    socket.addEventListener('meshi:synced', (e) => {
      this.appendLog(`SYNC: Reconnection delta resolved. Merged ${e.detail.count} records.`);
    });

    socket.addEventListener('message', async (e) => {
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
    });

    socket.addEventListener('close', () => {
      this.statusBadge.textContent = 'LINK: DISCONNECTED';
      this.statusBadge.classList.remove('badge-online');
      this.authBadge.textContent = 'AUTH: UNVERIFIED';
      this.authBadge.classList.remove('badge-auth');
      this.appendLog("NET: Channel disconnected.");
    });
  }

  async handleSocketDropout() {
    if (this.isReconnecting) return;
    this.isReconnecting = true;
    this.appendLog("RECONNECT: Attempting session restoration...");

    try {
      const lastPeer = await this.db.getSystemKey('last_peer_key');
      if (lastPeer && (!this.activeSocket || this.activeSocket.readyState !== MeshiMeshSocket.OPEN)) {
        // Regenerate connection offer to re-anchor local subnet candidates
        const { pc, dc, token } = await MeshiSignaler.createOffer(this.crypto);
        this.pendingPc = pc;
        this.mountSocket(new MeshiMeshSocket('remote-peer', dc, this.crypto, this.db, () => this.handleSocketDropout()));
        
        // Broadcast renegotiation beacon to background worker & sibling tabs
        navigator.serviceWorker?.controller?.postMessage({
          type: 'MESHI_BROADCAST_SIGNAL',
          token,
          targetPeer: lastPeer
        });
      }
    } catch (err) {
      console.warn("Dropout recovery warning:", err);
    } finally {
      this.isReconnecting = false;
    }
  }

  async attemptResumeLastSession() {
    const lastPeer = await this.db.getSystemKey('last_peer_key');
    if (lastPeer) {
      this.appendLog(`RESUME: Restoring previous session for peer: ${lastPeer.substring(0, 16)}...`);
      this.handleSocketDropout();
    }
  }

  setupServiceWorkerWatchdog() {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('./sw.js').then(async (reg) => {
        // Register periodic background sync if available
        if ('periodicSync' in reg) {
          try {
            await reg.periodicSync.register('meshi-keepalive', { minInterval: 60 * 1000 });
          } catch (e) {
            console.warn('Periodic sync not permitted:', e);
          }
        }
        // Register standard background sync for offline recovery
        if ('sync' in reg) {
          try {
            await reg.sync.register('meshi-reconnect-sync');
          } catch (e) {
            console.warn('Sync registration failed:', e);
          }
        }
      });

      navigator.serviceWorker.addEventListener('message', (event) => {
        if (event.data?.type === 'MESHI_BACKGROUND_NET_STABLE') {
          this.appendLog("SW: Background sync network restoration heartbeat received.");
          if (!this.activeSocket || this.activeSocket.readyState !== MeshiMeshSocket.OPEN) {
            this.handleSocketDropout();
          }
        }
      });
    }
  }

  appendLog(text) {
    const entry = document.createElement('div');
    entry.className = 'term-line';
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
    this.terminal.appendChild(entry);
    this.terminal.scrollTop = this.terminal.scrollHeight;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new MeshiApp().init();
});
