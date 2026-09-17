export class MeshiMeshSocket extends EventTarget {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  constructor(peerId, dataChannel, cryptoContext, db, reconnectHandler) {
    super();
    this.peerId = peerId;
    this.dc = dataChannel;
    this.crypto = cryptoContext;
    this.db = db;
    this.reconnectHandler = reconnectHandler;

    this.readyState = MeshiMeshSocket.CONNECTING;
    this.isAuthorized = false;
    this.currentChallenge = null;
    this.remotePublicKey = null;
    this.heartbeatTimer = null;

    this._bindEvents();
    this._startHeartbeat();
  }

  _bindEvents() {
    this.dc.onopen = () => {
      this.readyState = MeshiMeshSocket.OPEN;
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
        console.error('Meshi Protocol Error:', err);
      }
    };

    this.dc.onclose = () => {
      this._teardown();
      this.dispatchEvent(new CloseEvent('close', { wasClean: true }));
      this._attemptAutoReconnect();
    };

    this.dc.onerror = () => {
      this._teardown();
      this._attemptAutoReconnect();
    };
  }

  async _handleProtocolPacket(msg) {
    if (msg.type === 'HEARTBEAT_PING') {
      this.dc.send(JSON.stringify({ type: 'HEARTBEAT_PONG' }));
      return;
    }
    if (msg.type === 'HEARTBEAT_PONG') return;

    if (msg.type === 'AUTH_CHALLENGE') {
      const signature = await this.crypto.sign(msg.nonce);
      this.dc.send(JSON.stringify({
        type: 'AUTH_RESPONSE',
        signature,
        publicKey: this.crypto.publicKeyBase64
      }));
      return;
    }

    if (msg.type === 'AUTH_RESPONSE') {
      const isValidSig = await this.crypto.constructor.verify(msg.publicKey, this.currentChallenge, msg.signature);
      const isTrusted = await this.db.isPeerTrusted(msg.publicKey);

      if (isValidSig && isTrusted) {
        this.isAuthorized = true;
        this.remotePublicKey = msg.publicKey;
        this.dc.send(JSON.stringify({ type: 'AUTH_OK' }));
        this._dispatchAuthorized();
        this.triggerCatchUpSync();
      } else {
        this.dc.send(JSON.stringify({ type: 'AUTH_DENIED' }));
        this.close();
      }
      return;
    }

    if (msg.type === 'AUTH_OK') {
      this.isAuthorized = true;
      this._dispatchAuthorized();
      this.triggerCatchUpSync();
      return;
    }

    if (!this.isAuthorized) return;

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

    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(msg) }));
  }

  _dispatchAuthorized() {
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

  _startHeartbeat() {
    this.heartbeatTimer = setInterval(() => {
      if (this.readyState === MeshiMeshSocket.OPEN) {
        try {
          this.dc.send(JSON.stringify({ type: 'HEARTBEAT_PING' }));
        } catch {
          this._attemptAutoReconnect();
        }
      }
    }, 5000);
  }

  _teardown() {
    clearInterval(this.heartbeatTimer);
    this.readyState = MeshiMeshSocket.CLOSED;
    this.isAuthorized = false;
  }

  _attemptAutoReconnect() {
    if (typeof this.reconnectHandler === 'function') {
      setTimeout(() => this.reconnectHandler(), 2000);
    }
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
    this._teardown();
    this.dc.close();
  }
}
