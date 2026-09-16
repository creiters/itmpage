import { MeshiSignaler, MeshiMeshSocket } from './mesh_socket.js';
import { MeshiDB } from './db.js';

class MeshiApp {
  constructor() {
    this.db = new MeshiDB();
    this.activeSocket = null;
    this.pendingPc = null;

    // Elements
    this.statusBadge = document.getElementById('node-status');
    this.tokenBox = document.getElementById('signal-token');
    this.btnOffer = document.getElementById('btn-create-offer');
    this.btnAnswer = document.getElementById('btn-create-answer');
    this.btnFinalize = document.getElementById('btn-finalize');
    this.btnSend = document.getElementById('btn-broadcast');
    this.inputPayload = document.getElementById('payload-input');
    this.logContainer = document.getElementById('mesh-logs');
  }

  async init() {
    await this.db.init();
    this.bindEvents();
    this.renderExistingRecords();
    this.appendLog('MESHI // Method 2 RTCDataChannel socket engine initialized.');
  }

  bindEvents() {
    // 1. Host creates Offer
    this.btnOffer.addEventListener('click', async () => {
      try {
        const { pc, dc, token } = await MeshiSignaler.createOffer();
        this.pendingPc = pc;
        this.tokenBox.value = token;
        this.setupSocket(new MeshiMeshSocket('remote-peer', dc));
        this.appendLog('Generated Offer Token. Transmit to Responder via subnet or copy/paste.');
      } catch (err) {
        this.appendLog(`Error generating offer: ${err.message}`);
      }
    });

    // 2. Client consumes Offer and creates Answer
    this.btnAnswer.addEventListener('click', async () => {
      try {
        const offerToken = this.tokenBox.value.trim();
        if (!offerToken) throw new Error('Token field is empty.');
        const { pc, answerToken, dcPromise } = await MeshiSignaler.acceptOffer(offerToken);
        this.tokenBox.value = answerToken;
        const dc = await dcPromise;
        this.setupSocket(new MeshiMeshSocket('remote-peer', dc));
        this.appendLog('Generated Answer Token. Transmit back to Initiator.');
      } catch (err) {
        this.appendLog(`Error creating answer: ${err.message}`);
      }
    });

    // 3. Host consumes Answer and finalizes link
    this.btnFinalize.addEventListener('click', async () => {
      try {
        const answerToken = this.tokenBox.value.trim();
        if (!answerToken || !this.pendingPc) throw new Error('Invalid state or missing answer token.');
        await MeshiSignaler.finalizeConnection(this.pendingPc, answerToken);
        this.appendLog('Handshake finalized. Awaiting DataChannel open state...');
      } catch (err) {
        this.appendLog(`Finalize error: ${err.message}`);
      }
    });

    // Send payload
    this.btnSend.addEventListener('click', () => {
      const val = this.inputPayload.value.trim();
      if (!val) return;
      if (!this.activeSocket || this.activeSocket.readyState !== MeshiMeshSocket.OPEN) {
        this.appendLog('ERROR: Duplex socket is not open.');
        return;
      }

      const packet = {
        id: `meshi_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
        data: val,
        timestamp: Date.now()
      };

      this.activeSocket.send(packet);
      this.db.save(packet);
      this.appendLog(`[TX] Sent packet: ${packet.id}`);
      this.inputPayload.value = '';
    });
  }

  setupSocket(socket) {
    this.activeSocket = socket;

    socket.onopen = async () => {
      this.statusBadge.textContent = 'LINK: CONNECTED (RTC)';
      this.statusBadge.classList.add('badge-online');
      this.appendLog('[NET] Offline WebRTC DataChannel connected.');

      // Push initial synchronization
      const localDocs = await this.db.listAll();
      socket.send({ type: 'BATCH_SYNC', payload: localDocs });
    };

    socket.onmessage = async (e) => {
      try {
        const message = JSON.parse(e.data);
        if (message.type === 'BATCH_SYNC') {
          for (const item of message.payload) await this.db.save(item);
          this.appendLog(`[RX] Synced batch of ${message.payload.length} records.`);
        } else if (message.id) {
          await this.db.save(message);
          this.appendLog(`[RX] Received record: ${message.id} -> ${message.data}`);
        }
      } catch {
        this.appendLog(`[RX Raw]: ${e.data}`);
      }
    };

    socket.onclose = () => {
      this.statusBadge.textContent = 'LINK: DISCONNECTED';
      this.statusBadge.classList.remove('badge-online');
      this.appendLog('[NET] WebRTC DataChannel closed.');
    };
  }

  async renderExistingRecords() {
    const records = await this.db.listAll();
    this.appendLog(`[DB] ${records.length} records loaded from local IndexedDB.`);
  }

  appendLog(msg) {
    const p = document.createElement('p');
    p.className = 'terminal-entry';
    p.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    this.logContainer.appendChild(p);
    this.logContainer.scrollTop = this.logContainer.scrollHeight;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new MeshiApp().init();
});
