/**
 * Project Meshi Core Peer Engine
 * Provides WebRTC Local DataChannels, Chunked File Transfers, and CRDT Sync.
 */
export class MeshiEngine {
  constructor(options = {}) {
    this.nodeId = options.nodeId || `node-${crypto.randomUUID().slice(0, 8)}`;
    this.connection = null;
    this.channel = null;

    // Direct WebRTC config: Empty iceServers enforces LAN / Hotspot host candidates only
    this.rtcConfig = { iceServers: options.iceServers || [] };

    // State Store (CRDT Replica)
    this.store = new Map();
    this.vectorClock = new Map([[this.nodeId, 0]]);

    // Binary Streaming Settings
    this.CHUNK_SIZE = 16 * 1024; // 16 KB chunks to keep browser buffers lean
    this.transfers = new Map();

    // Event Hooks
    this.onStateUpdated = options.onStateUpdated || (() => {});
    this.onFileReceived = options.onFileReceived || (() => {});
    this.onProgress = options.onProgress || (() => {});
    this.onLog = options.onLog || console.log;
    this.onStatus = options.onStatus || (() => {});
  }

  // --- 1. LOCAL SIGNALING HANDSHAKE ---

  _initPeerConnection() {
    if (this.connection) return;

    this.connection = new RTCPeerConnection(this.rtcConfig);

    this.connection.oniceconnectionstatechange = () => {
      const state = this.connection.iceConnectionState;
      this.onStatus(state);
      this.onLog(`[MESHI ICE] Connection state changed: ${state.toUpperCase()}`);
    };

    this.connection.ondatachannel = (event) => {
      this.onLog('[MESHI RTC] Inbound DataChannel detected.');
      this._bindChannel(event.channel);
    };
  }

  /**
   * Host peer creates the offer envelope.
   */
  async createOfferEnvelope() {
    this._initPeerConnection();

    // Primary reliable ordered channel for deltas and byte transfers
    const dc = this.connection.createDataChannel('meshi-core-pipe', { ordered: true });
    this._bindChannel(dc);

    const offer = await this.connection.createOffer();
    await this.connection.setLocalDescription(offer);
    await this._awaitIceGathering();

    return btoa(JSON.stringify(this.connection.localDescription));
  }

  /**
   * Remote peer accepts the host's offer and generates an answer envelope.
   */
  async acceptOfferEnvelope(base64Offer) {
    this._initPeerConnection();

    const offerDesc = JSON.parse(atob(base64Offer));
    await this.connection.setRemoteDescription(new RTCSessionDescription(offerDesc));

    const answer = await this.connection.createAnswer();
    await this.connection.setLocalDescription(answer);
    await this._awaitIceGathering();

    return btoa(JSON.stringify(this.connection.localDescription));
  }

  /**
   * Host finalizes the connection by applying the answer.
   */
  async finalizeEnvelope(base64Answer) {
    const answerDesc = JSON.parse(atob(base64Answer));
    await this.connection.setRemoteDescription(new RTCSessionDescription(answerDesc));
    this.onLog('[MESHI RTC] Remote answer applied. DataChannel pipe opening...');
  }

  _awaitIceGathering() {
    return new Promise((resolve) => {
      if (this.connection.iceGatheringState === 'complete') {
        resolve();
      } else {
        const handler = () => {
          if (this.connection.iceGatheringState === 'complete') {
            this.connection.removeEventListener('icegatheringstatechange', handler);
            resolve();
          }
        };
        this.connection.addEventListener('icegatheringstatechange', handler);
      }
    });
  }

  // --- 2. DATA CHANNEL MANAGEMENT ---

  _bindChannel(channel) {
    this.channel = channel;
    this.channel.binaryType = 'arraybuffer';

    this.channel.onopen = () => {
      this.onStatus('connected');
      this.onLog('[MESHI LINK] Pipe OPEN. Handshaking vector clocks...');
      this._sendPacket({
        type: 'HELO',
        sender: this.nodeId,
        vectorClock: Object.fromEntries(this.vectorClock)
      });
    };

    this.channel.onclose = () => {
      this.onStatus('disconnected');
      this.onLog('[MESHI LINK] Pipe CLOSED.');
    };

    this.channel.onmessage = (event) => {
      if (typeof event.data === 'string') {
        this._dispatchControlPacket(JSON.parse(event.data));
      } else if (event.data instanceof ArrayBuffer) {
        this._processBinaryChunk(event.data);
      }
    };
  }

  _sendPacket(obj) {
    if (this.channel && this.channel.readyState === 'open') {
      this.channel.send(JSON.stringify(obj));
    }
  }

  // --- 3. CRDT STATE REPLICATION ---

  /**
   * Sets a key-value pair and broadcasts the delta to peers.
   */
  set(key, value) {
    const currentClock = (this.vectorClock.get(this.nodeId) || 0) + 1;
    this.vectorClock.set(this.nodeId, currentClock);

    const record = {
      value,
      origin: this.nodeId,
      clock: currentClock,
      timestamp: Date.now()
    };

    this.store.set(key, record);
    this.onStateUpdated(this.getStateSnapshot());

    this._sendPacket({
      type: 'DELTA',
      key,
      record
    });

    this.onLog(`[MESHI CRDT] Transmitted delta: [${key}]`);
  }

  getStateSnapshot() {
    const out = {};
    for (const [k, v] of this.store.entries()) {
      out[k] = v.value;
    }
    return out;
  }

  _dispatchControlPacket(pkt) {
    switch (pkt.type) {
      case 'HELO':
        this.onLog(`[MESHI HELO] Connected to remote peer: ${pkt.sender}`);
        break;

      case 'DELTA':
        this._applyDelta(pkt.key, pkt.record);
        break;

      case 'FILE_HEADER':
        this.transfers.set(pkt.transferId, {
          meta: pkt,
          chunks: [],
          receivedBytes: 0
        });
        this.onLog(`[MESHI STREAM] Receiving incoming artifact: ${pkt.name} (${pkt.size} bytes)`);
        break;

      case 'FILE_TRAILER':
        this._completeFileTransfer(pkt.transferId);
        break;
    }
  }

  _applyDelta(key, incomingRecord) {
    const existing = this.store.get(key);

    // Conflict resolution: Higher clock wins; tie-breaker: higher timestamp
    const shouldOverwrite =
      !existing ||
      incomingRecord.clock > existing.clock ||
      (incomingRecord.clock === existing.clock && incomingRecord.timestamp > existing.timestamp);

    if (shouldOverwrite) {
      this.store.set(key, incomingRecord);
      this.vectorClock.set(
        incomingRecord.origin,
        Math.max(this.vectorClock.get(incomingRecord.origin) || 0, incomingRecord.clock)
      );
      this.onStateUpdated(this.getStateSnapshot());
      this.onLog(`[MESHI CRDT] Synced key '${key}' from ${incomingRecord.origin}`);
    }
  }

  // --- 4. BINARY FILE ENGINE & CHUNKING ---

  async sendFile(file) {
    if (!this.channel || this.channel.readyState !== 'open') {
      throw new Error('DataChannel pipe is not active.');
    }

    const transferId = crypto.randomUUID();
    const arrayBuffer = await file.arrayBuffer();
    const totalBytes = arrayBuffer.byteLength;

    // 1. Send Header
    this._sendPacket({
      type: 'FILE_HEADER',
      transferId,
      name: file.name,
      mimeType: file.type || 'application/octet-stream',
      size: totalBytes
    });

    // 2. Stream Binary Chunks
    const prefix = new TextEncoder().encode(transferId); // 36-byte UUID prefix
    let offset = 0;

    while (offset < totalBytes) {
      // DataChannel backpressure management
      if (this.channel.bufferedAmount > 8 * 1024 * 1024) {
        await new Promise((resolve) => setTimeout(resolve, 25));
        continue;
      }

      const chunk = arrayBuffer.slice(offset, offset + this.CHUNK_SIZE);
      const packet = new Uint8Array(prefix.byteLength + chunk.byteLength);
      packet.set(prefix, 0);
      packet.set(new Uint8Array(chunk), prefix.byteLength);

      this.channel.send(packet.buffer);
      offset += this.CHUNK_SIZE;

      this.onProgress(Math.min(100, Math.round((offset / totalBytes) * 100)));
    }

    // 3. Send Trailer
    this._sendPacket({ type: 'FILE_TRAILER', transferId });
    this.onLog(`[MESHI STREAM] Dispatched artifact: ${file.name}`);
  }

  _processBinaryChunk(buffer) {
    const idDecoder = new TextDecoder();
    const transferId = idDecoder.decode(new Uint8Array(buffer, 0, 36));
    const chunkData = buffer.slice(36);

    const record = this.transfers.get(transferId);
    if (!record) return;

    record.chunks.push(chunkData);
    record.receivedBytes += chunkData.byteLength;

    const progress = Math.min(100, Math.round((record.receivedBytes / record.meta.size) * 100));
    this.onProgress(progress);
  }

  _completeFileTransfer(transferId) {
    const record = this.transfers.get(transferId);
    if (!record) return;

    const blob = new Blob(record.chunks, { type: record.meta.mimeType });
    this.transfers.delete(transferId);
    this.onProgress(0);

    this.onFileReceived({
      name: record.meta.name,
      size: record.meta.size,
      blob
    });

    this.onLog(`[MESHI STREAM] Artifact assembly complete: ${record.meta.name}`);
  }
}
