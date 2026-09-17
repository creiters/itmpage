import { MeshiCrypto } from './crypto.js';

export class MeshiMeshSocket extends EventTarget {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  constructor(peerId, dataChannel, cryptoContext, db) {
    super();
    this.peerId = peerId;
    this.dc = dataChannel;
    this.crypto = cryptoContext;
    this.db = db;

    this.readyState = MeshiMeshSocket.CONNECTING;
    this.isAuthorized = false;
    this.currentChallenge = null;
    this.remotePublicKey = null;

    this.onopen = null;
    this.onmessage = null;
    this.onclose = null;
    this.onauthorized = null;

    this._bindEvents();
    this._watchNetworkState();
  }

  _bindEvents() {
    this.dc.onopen = () => {
      this.readyState = MeshiMeshSocket.OPEN;
      // Step 1: Challenge remote peer upon link open
      this.currentChallenge = `AUTH_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
      this.dc.send(JSON.stringify({
        type: 'AUTH_CHALLENGE',
        nonce: this.currentChallenge,
        fromKey: this.crypto.publicKeyBase64
      }));
    };

    this.dc.onmessage = async (event) => {
      try {
        const msg = JSON.parse(event.data);
        await this._handleProtocolPacket(msg);
      } catch (err) {
        console.error('Meshi Parse Error:', err);
      }
    };

    this.dc.onclose = () => {
      this.readyState = MeshiMeshSocket.CLOSED;
      this.isAuthorized = false;
      const ev = new CloseEvent('close', { wasClean: true });
      if (typeof this.onclose === 'function') this.onclose(ev);
      this.dispatchEvent(ev);
    };
  }

  async _handleProtocolPacket(msg) {
    // Protocol Step 1: Inbound challenge -> sign and return public key
    if (msg.type === 'AUTH_CHALLENGE') {
      const signature = await this.crypto.signChallenge(msg.nonce);
      this.dc.send(JSON.stringify({
        type: 'AUTH_RESPONSE',
        signature,
        publicKey: this.crypto.publicKeyBase64
      }));
      return;
    }

    // Protocol Step 2: Inbound challenge response -> verify signature and local trust store
    if (msg.type === 'AUTH_RESPONSE') {
      const isValidSig = await MeshiCrypto.verifyPeer(msg.publicKey, this.currentChallenge, msg.signature);
      const isTrusted = await this.db.isPeerTrusted(msg.publicKey);

      if (isValidSig && isTrusted) {
        this.isAuthorized = true;
        this.remotePublicKey = msg.publicKey;
        this.dc.send(JSON.stringify({ type: 'AUTH_OK' }));
        this._dispatchAuthorized();
        this.triggerCatchUpSync();
      } else {
        this.dc.send(JSON.stringify({
          type: 'AUTH_DENIED',
          reason: !isValidSig ? 'BAD_SIGNATURE' : 'UNAUTHORIZED_KEY'
        }));
        this.close();
      }
      return;
    }

    // Protocol Step 3: Peer validated our signature
    if (msg.type === 'AUTH_OK') {
      this.isAuthorized = true;
      this._dispatchAuthorized();
      this.triggerCatchUpSync();
      return;
    }

    if (msg.type === 'AUTH_DENIED') {
      console.warn('Meshi link authorization rejected:', msg.reason);
      this.close();
      return;
    }

    // Drop all unauthenticated operational packets
    if (!this.isAuthorized) {
      console.warn('Meshi: Received data prior to mutual authorization. Dropped.');
      return;
    }

    // Protocol Step 4: Catch-up synchronization
    if (msg.type === 'SYNC_WATERMARK_REQ') {
      const deltas = await this.db.getDeltaSince(msg.sinceClock);
      this.dc.send(JSON.stringify({ type: 'SYNC_DELTA_BATCH', payload: deltas }));
      return;
    }

    if (msg.type === 'SYNC_DELTA_BATCH') {
      let mergedCount = 0;
      for (const item of msg.payload) {
        const { merged } = await this.db.mergeRemoteDocument(item);
        if (merged) mergedCount++;
      }
      this.dispatchEvent(new CustomEvent('meshi:synced', { detail: { count: mergedCount } }));
      return;
    }

    // Normal application-level packet
    const msgEvent = new MessageEvent('message', { data: JSON.stringify(msg) });
    if (typeof this.onmessage === 'function') this.onmessage(msgEvent);
    this.dispatchEvent(msgEvent);
  }

  _dispatchAuthorized() {
    if (typeof this.onauthorized === 'function') this.onauthorized();
    this.dispatchEvent(new CustomEvent('meshi:authorized', { detail: { remoteKey: this.remotePublicKey } }));
  }

  triggerCatchUpSync() {
    if (this.readyState === MeshiMeshSocket.OPEN && this.isAuthorized) {
      this.dc.send(JSON.stringify({
        type: 'SYNC_WATERMARK_REQ',
        sinceClock: this.db.clock || 0
      }));
    }
  }

  _watchNetworkState() {
    // When device reconnects to Wi-Fi / hotspot subnet
    window.addEventListener('online', () => {
      this.dispatchEvent(new CustomEvent('meshi:net-restored'));
      if (this.readyState === MeshiMeshSocket.OPEN && this.isAuthorized) {
        this.triggerCatchUpSync();
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
