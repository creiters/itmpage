import { MeshiSignaler } from './signaling.js';

export class MeshiMeshSocket extends EventTarget {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  constructor(peerId, dataChannel, authContext, db) {
    super();
    this.peerId = peerId;
    this.dc = dataChannel;
    this.auth = authContext;
    this.db = db;
    
    this.readyState = MeshiMeshSocket.CONNECTING;
    this.isAuthorized = false;
    this.currentChallenge = null;

    this.onopen = null;
    this.onmessage = null;
    this.onclose = null;
    this.onauthorized = null;

    this._bindChannel();
    this._initNetworkWatchdog();
  }

  _bindChannel() {
    this.dc.onopen = () => {
      this.readyState = MeshiMeshSocket.OPEN;
      // Step 1: Challenge remote peer upon physical RTC link open
      this.currentChallenge = `CHAL_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
      this.dc.send(JSON.stringify({
        type: 'AUTH_CHALLENGE',
        nonce: this.currentChallenge,
        fromKey: this.auth.publicKeyBase64
      }));
    };

    this.dc.onmessage = async (event) => {
      try {
        const msg = JSON.parse(event.data);
        await this._handleAuthAndSync(msg);
      } catch (err) {
        console.error('Meshi Protocol Error:', err);
      }
    };

    this.dc.onclose = () => {
      this.readyState = MeshiMeshSocket.CLOSED;
      this.isAuthorized = false;
      this.dispatchEvent(new CloseEvent('close'));
    };
  }

  async _handleAuthAndSync(msg) {
    // 1. Peer challenges local node -> sign challenge
    if (msg.type === 'AUTH_CHALLENGE') {
      const signature = await this.auth.signChallenge(msg.nonce);
      this.dc.send(JSON.stringify({
        type: 'AUTH_RESPONSE',
        signature,
        publicKey: this.auth.publicKeyBase64
      }));
      return;
    }

    // 2. Peer responds to our challenge -> verify against whitelist
    if (msg.type === 'AUTH_RESPONSE') {
      const ok = await this.auth.verifyPeer(msg.publicKey, this.currentChallenge, msg.signature);
      if (ok) {
        this.isAuthorized = true;
        this.dc.send(JSON.stringify({ type: 'AUTH_SUCCESS' }));
        this._triggerSyncCatchUp();
      } else {
        this.dc.send(JSON.stringify({ type: 'AUTH_FAIL', reason: 'UNAUTHORIZED_KEY' }));
        this.close();
      }
      return;
    }

    // 3. Remote peer validated our signature
    if (msg.type === 'AUTH_SUCCESS') {
      this.isAuthorized = true;
      if (typeof this.onauthorized === 'function') this.onauthorized();
      this._triggerSyncCatchUp();
      return;
    }

    // 4. Reject unverified commands
    if (!this.isAuthorized) {
      console.warn('Meshi: Received data prior to authorization. Dropping.');
      return;
    }

    // 5. Delta sync handshake: remote sends current Lamport watermark
    if (msg.type === 'SYNC_REQ') {
      const delta = await this.db.getRecordsSince(msg.sinceClock);
      this.dc.send(JSON.stringify({ type: 'SYNC_DELTA', payload: delta }));
      return;
    }

    // 6. Incoming catch-up sync batch
    if (msg.type === 'SYNC_DELTA') {
      let mergedCount = 0;
      for (const item of msg.payload) {
        const { merged } = await this.db.mergeRemoteDocument(item);
        if (merged) mergedCount++;
      }
      this.dispatchEvent(new CustomEvent('meshi:synced', { detail: { mergedCount } }));
      return;
    }

    // Pass standard payload to app listeners
    const msgEvent = new MessageEvent('message', { data: JSON.stringify(msg) });
    if (typeof this.onmessage === 'function') this.onmessage(msgEvent);
    this.dispatchEvent(msgEvent);
  }

  async _triggerSyncCatchUp() {
    // Solicit missing deltas using local clock
    const currentClock = this.db.clock || 0;
    this.dc.send(JSON.stringify({ type: 'SYNC_REQ', sinceClock: currentClock }));
  }

  _initNetworkWatchdog() {
    window.addEventListener('online', async () => {
      // If physical network restores and link dropped, re-trigger local reconnect
      if (this.readyState === MeshiMeshSocket.CLOSED) {
        this.dispatchEvent(new CustomEvent('meshi:network-online'));
      } else if (this.isAuthorized) {
        // If link stayed alive across network interface toggles, poll catch-up
        this._triggerSyncCatchUp();
      }
    });
  }

  send(data) {
    if (this.readyState !== MeshiMeshSocket.OPEN || !this.isAuthorized) {
      throw new DOMException('Socket is not connected or not authorized', 'InvalidStateError');
    }
    const payload = typeof data === 'object' && !(data instanceof ArrayBuffer)
      ? JSON.stringify(data)
      : data;
    this.dc.send(payload);
  }

  close() {
    this.readyState = MeshiMeshSocket.CLOSING;
    this.dc.close();
  }
}
