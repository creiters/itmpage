Here is the complete codebase for Factoroid, unifying:
 * WASI Linear Memory & Syscall Core: Zero-allocation ring buffer, Big-Endian S7 decoding, and custom host-imported WebRTC mesh syscalls (wasi_webrtc_*).
 * WebRTC Local P2P Mesh Network: Peer failover and automatic buffering across LAN stations when an edge station drops out.
 * Solana DePIN Commitment Engine: Merkle digest generation and batch anchoring to secure data availability rewards.
 * Direct Sockets API: Low-latency ISO-on-TCP and S7comm protocol engine.
 * Background Threading & Service Worker: Non-blocking UI execution, IndexedDB multi-origin state, and WebMCP agent tools.
Project File Structure
factoroid/
├── manifest.webmanifest
├── index.html
├── sw.js
├── icon-64.svg
├── icon-256.svg
├── src/
│   ├── mesh_core.c
│   ├── db.js
│   ├── s7client.js
│   ├── wasi_host.js
│   ├── solana.js
│   ├── worker.js
│   ├── mcp.js
│   └── app.js
└── dist/
    └── mesh_core.wasm

1. manifest.webmanifest
{
  "name": "Factoroid - Industrial S7 DePIN Station",
  "short_name": "Factoroid",
  "version": "1.3.0",
  "start_url": "/index.html",
  "display": "standalone",
  "background_color": "#0d1117",
  "theme_color": "#161b22",
  "icons": [
    {
      "src": "/icon-64.svg",
      "sizes": "64x64",
      "type": "image/svg+xml"
    },
    {
      "src": "/icon-256.svg",
      "sizes": "256x256",
      "type": "image/svg+xml"
    }
  ],
  "isolated_web_app_permissions": {
    "direct-sockets": {},
    "periodic-background-sync": {}
  },
  "permissions_policy": {
    "direct-sockets": ["self"],
    "periodic-background-sync": ["self"]
  }
}

2. icon-64.svg & icon-256.svg
icon-64.svg:
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="64" height="64" shape-rendering="crispEdges">
  <rect x="7" y="1" width="1" height="2" fill="#7ee787"/>
  <rect x="8" y="1" width="1" height="1" fill="#aff5b4"/>
  <path fill="#0b2818" d="M 5 3 h 5 v 1 h -5 z M 4 4 h 1 v 1 h -1 z M 10 4 h 1 v 1 h -1 z M 3 5 h 1 v 6 h -1 z M 11 5 h 1 v 6 h -1 z M 4 11 h 1 v 1 h -1 z M 10 11 h 1 v 1 h -1 z M 5 12 h 5 v 1 h -5 z"/>
  <path fill="#1b4d2e" d="M 5 4 h 5 v 1 h -5 z M 4 5 h 7 v 6 h -7 z M 5 11 h 5 v 1 h -5 z"/>
  <rect x="5" y="4" width="1" height="1" fill="#79d98e"/>
  <rect x="5" y="6" width="1" height="2" fill="#79d98e"/>
  <rect x="5" y="9" width="1" height="2" fill="#aff5b4"/>
  <rect x="7" y="4" width="1" height="2" fill="#aff5b4"/>
  <rect x="8" y="5" width="1" height="2" fill="#79d98e"/>
  <rect x="7" y="7" width="1" height="2" fill="#79d98e"/>
  <rect x="8" y="8" width="1" height="2" fill="#aff5b4"/>
  <rect x="7" y="10" width="1" height="2" fill="#aff5b4"/>
  <rect x="10" y="5" width="1" height="2" fill="#79d98e"/>
  <rect x="9" y="7" width="1" height="2" fill="#aff5b4"/>
  <rect x="10" y="9" width="1" height="2" fill="#79d98e"/>
  <rect x="5" y="5" width="1" height="1" fill="#ffffff" opacity="0.85"/>
</svg>

icon-256.svg:
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="256" height="256" shape-rendering="crispEdges">
  <rect x="7" y="1" width="1" height="2" fill="#7ee787"/>
  <rect x="8" y="1" width="1" height="1" fill="#aff5b4"/>
  <path fill="#0b2818" d="M 5 3 h 5 v 1 h -5 z M 4 4 h 1 v 1 h -1 z M 10 4 h 1 v 1 h -1 z M 3 5 h 1 v 6 h -1 z M 11 5 h 1 v 6 h -1 z M 4 11 h 1 v 1 h -1 z M 10 11 h 1 v 1 h -1 z M 5 12 h 5 v 1 h -5 z"/>
  <path fill="#1b4d2e" d="M 5 4 h 5 v 1 h -5 z M 4 5 h 7 v 6 h -7 z M 5 11 h 5 v 1 h -5 z"/>
  <rect x="5" y="4" width="1" height="1" fill="#79d98e"/>
  <rect x="5" y="6" width="1" height="2" fill="#79d98e"/>
  <rect x="5" y="9" width="1" height="2" fill="#aff5b4"/>
  <rect x="7" y="4" width="1" height="2" fill="#aff5b4"/>
  <rect x="8" y="5" width="1" height="2" fill="#79d98e"/>
  <rect x="7" y="7" width="1" height="2" fill="#79d98e"/>
  <rect x="8" y="8" width="1" height="2" fill="#aff5b4"/>
  <rect x="7" y="10" width="1" height="2" fill="#aff5b4"/>
  <rect x="10" y="5" width="1" height="2" fill="#79d98e"/>
  <rect x="9" y="7" width="1" height="2" fill="#aff5b4"/>
  <rect x="10" y="9" width="1" height="2" fill="#79d98e"/>
  <rect x="5" y="5" width="1" height="1" fill="#ffffff" opacity="0.85"/>
</svg>

3. src/mesh_core.c (WASI Core Engine)
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#define RING_BUFFER_CAPACITY 64
#define S7_RX_BUF_SIZE 2048

// Imported Host Syscalls (Provided by WasiWebRTCHost in JavaScript)
__attribute__((import_module("wasi_webrtc"), import_name("send_mesh_broadcast")))
extern void wasi_webrtc_send_broadcast(const uint8_t *data, int32_t len);

__attribute__((import_module("wasi_webrtc"), import_name("notify_peer_timeout")))
extern void wasi_webrtc_notify_peer_timeout(int32_t peer_index);

typedef struct {
    uint64_t timestamp;
    uint32_t station_id;
    uint16_t db_num;
    uint16_t offset;
    float value;
    uint8_t is_replicated; // 0 = local, 1 = remote peer replica
} TelemetryFrame;

static TelemetryFrame g_ring_buffer[RING_BUFFER_CAPACITY];
static uint32_t g_ring_head = 0;
static uint32_t g_ring_count = 0;
static uint8_t g_s7_rx_buffer[S7_RX_BUF_SIZE];

uint8_t* get_s7_rx_buffer(void) {
    return g_s7_rx_buffer;
}

int32_t get_s7_rx_buffer_capacity(void) {
    return S7_RX_BUF_SIZE;
}

float wasi_parse_s7_real(int32_t offset) {
    if (offset < 0 || offset + 4 > S7_RX_BUF_SIZE) return 0.0f;
    uint32_t val = ((uint32_t)g_s7_rx_buffer[offset] << 24) |
                   ((uint32_t)g_s7_rx_buffer[offset + 1] << 16) |
                   ((uint32_t)g_s7_rx_buffer[offset + 2] << 8) |
                   ((uint32_t)g_s7_rx_buffer[offset + 3]);
    float res;
    memcpy(&res, &val, sizeof(res));
    return res;
}

int16_t wasi_parse_s7_int(int32_t offset) {
    if (offset < 0 || offset + 2 > S7_RX_BUF_SIZE) return 0;
    return (int16_t)(((uint16_t)g_s7_rx_buffer[offset] << 8) | g_s7_rx_buffer[offset + 1]);
}

int32_t wasi_parse_s7_dint(int32_t offset) {
    if (offset < 0 || offset + 4 > S7_RX_BUF_SIZE) return 0;
    return (int32_t)(((uint32_t)g_s7_rx_buffer[offset] << 24) |
                     ((uint32_t)g_s7_rx_buffer[offset + 1] << 16) |
                     ((uint32_t)g_s7_rx_buffer[offset + 2] << 8) |
                     g_s7_rx_buffer[offset + 3]);
}

uint8_t wasi_parse_s7_byte(int32_t offset) {
    if (offset < 0 || offset >= S7_RX_BUF_SIZE) return 0;
    return g_s7_rx_buffer[offset];
}

int32_t wasi_push_local_sample(uint64_t timestamp, uint32_t station_id, uint16_t db, uint16_t off, float val) {
    uint32_t idx = (g_ring_head + g_ring_count) % RING_BUFFER_CAPACITY;
    g_ring_buffer[idx].timestamp = timestamp;
    g_ring_buffer[idx].station_id = station_id;
    g_ring_buffer[idx].db_num = db;
    g_ring_buffer[idx].offset = off;
    g_ring_buffer[idx].value = val;
    g_ring_buffer[idx].is_replicated = 0;

    if (g_ring_count < RING_BUFFER_CAPACITY) {
        g_ring_count++;
    } else {
        g_ring_head = (g_ring_head + 1) % RING_BUFFER_CAPACITY;
    }

    // Binary payload: [Timestamp 8B][StationID 4B][DB 2B][Offset 2B][Value 4B][ReplicaFlag 1B] = 21 Bytes
    uint8_t serialized[21];
    memcpy(&serialized[0], &timestamp, 8);
    memcpy(&serialized[8], &station_id, 4);
    memcpy(&serialized[12], &db, 2);
    memcpy(&serialized[14], &off, 2);
    memcpy(&serialized[16], &val, 4);
    serialized[20] = 1;

    wasi_webrtc_send_broadcast(serialized, 21);
    return idx;
}

int32_t wasi_receive_mesh_frame(const uint8_t *data, int32_t len) {
    if (len < 21) return -1;

    uint32_t idx = (g_ring_head + g_ring_count) % RING_BUFFER_CAPACITY;
    memcpy(&g_ring_buffer[idx].timestamp, &data[0], 8);
    memcpy(&g_ring_buffer[idx].station_id, &data[8], 4);
    memcpy(&g_ring_buffer[idx].db_num, &data[12], 2);
    memcpy(&g_ring_buffer[idx].offset, &data[14], 2);
    memcpy(&g_ring_buffer[idx].value, &data[16], 4);
    g_ring_buffer[idx].is_replicated = 1;

    if (g_ring_count < RING_BUFFER_CAPACITY) {
        g_ring_count++;
    } else {
        g_ring_head = (g_ring_head + 1) % RING_BUFFER_CAPACITY;
    }

    return idx;
}

int32_t wasi_get_ring_count(void) {
    return g_ring_count;
}

TelemetryFrame* wasi_get_frame_ptr(int32_t relative_idx) {
    if (relative_idx < 0 || (uint32_t)relative_idx >= g_ring_count) return NULL;
    uint32_t actual_idx = (g_ring_head + relative_idx) % RING_BUFFER_CAPACITY;
    return &g_ring_buffer[actual_idx];
}

void wasi_flush_ring(void) {
    g_ring_head = 0;
    g_ring_count = 0;
}

Build Command:
clang --target=wasm32-wasi -O3 -nostdlib \
  -Wl,--no-entry \
  -Wl,--export=get_s7_rx_buffer \
  -Wl,--export=get_s7_rx_buffer_capacity \
  -Wl,--export=wasi_parse_s7_real \
  -Wl,--export=wasi_parse_s7_int \
  -Wl,--export=wasi_parse_s7_dint \
  -Wl,--export=wasi_parse_s7_byte \
  -Wl,--export=wasi_push_local_sample \
  -Wl,--export=wasi_receive_mesh_frame \
  -Wl,--export=wasi_get_ring_count \
  -Wl,--export=wasi_get_frame_ptr \
  -Wl,--export=wasi_flush_ring \
  -Wl,--allow-undefined \
  src/mesh_core.c -o dist/mesh_core.wasm

4. src/db.js (IndexedDB Engine)
const DB_NAME = 'FactoroidLocalDB';
const DB_VERSION = 2;
const STORE_NAME = 'telemetry';

let dbInstance = null;

export function openDatabase() {
  if (dbInstance) return Promise.resolve(dbInstance);

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      let store;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        store = db.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true });
      } else {
        store = event.target.transaction.objectStore(STORE_NAME);
      }
      if (!store.indexNames.contains('timestamp')) store.createIndex('timestamp', 'timestamp', { unique: false });
      if (!store.indexNames.contains('tagId')) store.createIndex('tagId', 'tagId', { unique: false });
      if (!store.indexNames.contains('originNode')) store.createIndex('originNode', 'originNode', { unique: false });
      if (!store.indexNames.contains('anchored')) store.createIndex('anchored', 'anchored', { unique: false });
    };

    request.onsuccess = (event) => {
      dbInstance = event.target.result;
      resolve(dbInstance);
    };

    request.onerror = (event) => reject(new Error(`IndexedDB open failed: ${event.target.error}`));
  });
}

export async function storeTelemetry(tagId, dbNumber, value, originNode = 'self') {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([STORE_NAME], 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const req = store.add({
      timestamp: Date.now(),
      tagId,
      dbNumber,
      value,
      originNode,
      anchored: 0
    });
    req.onsuccess = (e) => resolve(e.target.result);
    req.onerror = (e) => reject(e.target.error);
  });
}

export async function getUnanchoredRecords(limit = 32) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([STORE_NAME], 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const index = store.index('anchored');
    const range = IDBKeyRange.only(0);
    const results = [];

    const req = index.openCursor(range);
    req.onsuccess = (e) => {
      const cursor = e.target.result;
      if (cursor && results.length < limit) {
        results.push(cursor.value);
        cursor.continue();
      } else {
        resolve(results);
      }
    };
    req.onerror = (e) => reject(e.target.error);
  });
}

export async function markAsAnchored(ids, txSignature) {
  if (!ids || ids.length === 0) return;
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([STORE_NAME], 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    tx.oncomplete = () => resolve();
    tx.onerror = (e) => reject(e.target.error);

    for (const id of ids) {
      const req = store.get(id);
      req.onsuccess = (e) => {
        const item = e.target.result;
        if (item) {
          item.anchored = 1;
          item.solanaTx = txSignature;
          store.put(item);
        }
      };
    }
  });
}

export async function getTelemetryHistory(tagId, timeWindowMs = 3600000) {
  const db = await openDatabase();
  const since = Date.now() - timeWindowMs;

  return new Promise((resolve, reject) => {
    const tx = db.transaction([STORE_NAME], 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const index = store.index('timestamp');
    const range = IDBKeyRange.lowerBound(since);
    const results = [];

    const req = index.openCursor(range);
    req.onsuccess = (e) => {
      const cursor = e.target.result;
      if (cursor) {
        if (cursor.value.tagId === tagId) results.push(cursor.value);
        cursor.continue();
      } else {
        resolve(results);
      }
    };
    req.onerror = (e) => reject(e.target.error);
  });
}

export async function purgeOldRecords(retentionDays = 7) {
  const db = await openDatabase();
  const threshold = Date.now() - (retentionDays * 24 * 60 * 60 * 1000);

  return new Promise((resolve, reject) => {
    const tx = db.transaction([STORE_NAME], 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const index = store.index('timestamp');
    const range = IDBKeyRange.upperBound(threshold);
    let count = 0;

    const req = index.openCursor(range);
    req.onsuccess = (e) => {
      const cursor = e.target.result;
      if (cursor) {
        cursor.delete();
        count++;
        cursor.continue();
      } else {
        resolve(count);
      }
    };
    req.onerror = (e) => reject(e.target.error);
  });
}

5. src/s7client.js (Direct Sockets API Client)
export class S7DirectClient {
  constructor(ip, rack = 0, slot = 1) {
    this.ip = ip;
    this.rack = rack;
    this.slot = slot;
    this.socket = null;
    this.reader = null;
    this.writer = null;
    this.connected = false;
  }

  async connect() {
    if (this.socket) await this.disconnect();

    this.socket = new TCPSocket(this.ip, { remotePort: 102 });
    const { readable, writable } = await this.socket.opened;
    this.reader = readable.getReader();
    this.writer = writable.getWriter();

    // 1. COTP Connection Request (ISO-on-TCP)
    const cotpReq = new Uint8Array([
      0x03, 0x00, 0x00, 0x16, 0x11, 0xe0, 0x00, 0x00, 0x00, 0x01, 0x00,
      0xc1, 0x02, 0x01, 0x00, 0xc2, 0x02, 0x01, (this.rack * 32) + this.slot,
      0xc0, 0x01, 0x0a
    ]);
    await this.writer.write(cotpReq);
    let res = await this.reader.read();
    if (!res.value || res.value[5] !== 0xd0) {
      throw new Error("COTP Handshake rejected by PLC");
    }

    // 2. S7comm Setup Communication
    const s7Setup = new Uint8Array([
      0x03, 0x00, 0x00, 0x19, 0x02, 0xf0, 0x80, 0x32, 0x01, 0x00, 0x00,
      0x00, 0x01, 0x00, 0x08, 0x00, 0x00, 0xf0, 0x00, 0x00, 0x01, 0x00,
      0x01, 0x01, 0xe0
    ]);
    await this.writer.write(s7Setup);
    res = await this.reader.read();
    if (!res.value || res.value[7] !== 0x32) {
      throw new Error("S7comm Setup Communication failed");
    }

    this.connected = true;
  }

  async readDB(dbNumber, byteOffset, byteLength) {
    if (!this.connected) throw new Error("PLC not connected");

    const bitAddress = byteOffset * 8;
    const req = new Uint8Array([
      0x03, 0x00, 0x00, 0x1f,
      0x02, 0xf0, 0x80,
      0x32, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x0e, 0x00, 0x00,
      0x04, 0x01,
      0x12, 0x0a, 0x10,
      0x02,
      (byteLength >> 8) & 0xff, byteLength & 0xff,
      (dbNumber >> 8) & 0xff, dbNumber & 0xff,
      0x84,
      (bitAddress >> 16) & 0xff,
      (bitAddress >> 8) & 0xff,
      bitAddress & 0xff
    ]);

    await this.writer.write(req);
    const { value, done } = await this.reader.read();
    if (done) throw new Error("PLC disconnected");
    return value;
  }

  async disconnect() {
    this.connected = false;
    try {
      if (this.reader) await this.reader.cancel();
      if (this.writer) await this.writer.close();
      if (this.socket) await this.socket.close();
    } catch {
      // Ignore disconnect errors
    }
  }
}

6. src/wasi_host.js (WASI Syscall Bridge & WebRTC Mesh)
export class WasiWebRTCHost {
  constructor(nodeId, onFramePersist) {
    this.nodeId = nodeId;
    this.numericNodeId = Math.abs(this.hashCode(nodeId));
    this.onFramePersist = onFramePersist;
    this.peers = new Map(); // peerId -> { pc, dc }
    this.heartbeats = new Map();
    this.signaler = new BroadcastChannel('factoroid_wasi_mesh_signaling');
    this.wasmInstance = null;
    this.initSignaler();
  }

  hashCode(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash) + str.charCodeAt(i);
      hash |= 0;
    }
    return hash;
  }

  getImportObject() {
    return {
      wasi_snapshot_preview1: {
        proc_exit: (code) => console.warn(`WASI Exit: ${code}`),
        fd_write: () => 0,
        fd_close: () => 0,
        fd_seek: () => 0
      },
      wasi_webrtc: {
        send_mesh_broadcast: (ptr, len) => {
          if (!this.wasmInstance) return;
          const memory = new Uint8Array(this.wasmInstance.exports.memory.buffer);
          const slice = memory.subarray(ptr, ptr + len);
          this.broadcast(slice);
        },
        notify_peer_timeout: (peerIdx) => {
          console.warn(`[WASI Mesh] Peer station #${peerIdx} timed out`);
        }
      }
    };
  }

  setInstance(instance) {
    this.wasmInstance = instance;
  }

  initSignaler() {
    this.signaler.onmessage = async (e) => {
      const { from, to, type, data } = e.data;
      if (from === this.nodeId || (to && to !== this.nodeId)) return;

      if (type === 'ANNOUNCE' && !this.peers.has(from)) {
        await this.initiatePeer(from);
      } else if (type === 'OFFER') {
        await this.handleOffer(from, data);
      } else if (type === 'ANSWER') {
        const peer = this.peers.get(from);
        if (peer?.pc) await peer.pc.setRemoteDescription(new RTCSessionDescription(data));
      } else if (type === 'CANDIDATE') {
        const peer = this.peers.get(from);
        if (peer?.pc) await peer.pc.addIceCandidate(new RTCIceCandidate(data));
      }
    };

    this.signaler.postMessage({ from: this.nodeId, type: 'ANNOUNCE' });
  }

  async initiatePeer(peerId) {
    const pc = new RTCPeerConnection({ iceServers: [] });
    const dc = pc.createDataChannel('wasi_mesh_sync', { ordered: true });
    this.setupDataChannel(peerId, dc);

    pc.onicecandidate = (e) => {
      if (e.candidate) {
        this.signaler.postMessage({ from: this.nodeId, to: peerId, type: 'CANDIDATE', data: e.candidate });
      }
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    this.peers.set(peerId, { pc, dc });
    this.signaler.postMessage({ from: this.nodeId, to: peerId, type: 'OFFER', data: offer });
  }

  async handleOffer(fromPeerId, offer) {
    const pc = new RTCPeerConnection({ iceServers: [] });
    pc.ondatachannel = (e) => this.setupDataChannel(fromPeerId, e.channel);

    pc.onicecandidate = (e) => {
      if (e.candidate) {
        this.signaler.postMessage({ from: this.nodeId, to: fromPeerId, type: 'CANDIDATE', data: e.candidate });
      }
    };

    await pc.setRemoteDescription(new RTCSessionDescription(offer));
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);

    this.peers.set(fromPeerId, { pc, dc: null });
    this.signaler.postMessage({ from: this.nodeId, to: fromPeerId, type: 'ANSWER', data: answer });
  }

  setupDataChannel(peerId, dc) {
    const peer = this.peers.get(peerId) || {};
    peer.dc = dc;
    this.peers.set(peerId, peer);

    dc.binaryType = 'arraybuffer';
    dc.onopen = () => this.heartbeats.set(peerId, Date.now());
    dc.onmessage = (event) => {
      this.heartbeats.set(peerId, Date.now());
      if (event.data instanceof ArrayBuffer && this.wasmInstance) {
        const rawBytes = new Uint8Array(event.data);
        const rxPtr = this.wasmInstance.exports.get_s7_rx_buffer();
        const memory = new Uint8Array(this.wasmInstance.exports.memory.buffer);
        memory.set(rawBytes, rxPtr);

        const insertedIdx = this.wasmInstance.exports.wasi_receive_mesh_frame(rxPtr, rawBytes.byteLength);
        if (insertedIdx >= 0) {
          this.onFramePersist(rawBytes, peerId);
        }
      }
    };
  }

  broadcast(bufferSlice) {
    for (const [_, peer] of this.peers) {
      if (peer.dc && peer.dc.readyState === 'open') {
        peer.dc.send(bufferSlice);
      }
    }
  }

  checkLiveness(timeoutMs = 4000, onPeerDown) {
    const now = Date.now();
    for (const [peerId, lastSeen] of this.heartbeats.entries()) {
      if (now - lastSeen > timeoutMs) {
        onPeerDown(peerId);
        this.heartbeats.delete(peerId);
        const p = this.peers.get(peerId);
        if (p?.pc) p.pc.close();
        this.peers.delete(peerId);
      }
    }
  }
}

7. src/solana.js (Solana DePIN Commitment Engine)
export class SolanaDePINClient {
  constructor(rpcUrl, programIdStr, secretKeyBytes = null) {
    this.rpcUrl = rpcUrl;
    this.programIdStr = programIdStr;
    this.keypair = secretKeyBytes
      ? solanaWeb3.Keypair.fromSecretKey(new Uint8Array(secretKeyBytes))
      : solanaWeb3.Keypair.generate();
    this.connection = new solanaWeb3.Connection(this.rpcUrl, 'confirmed');
  }

  getPublicKey() {
    return this.keypair.publicKey.toBase58();
  }

  async computeBatchHash(records) {
    const digestParts = records
      .map(r => `${r.timestamp}:${r.originNode}:${r.tagId}:${r.value}`)
      .join(';');
    const enc = new TextEncoder();
    const digestBuffer = await crypto.subtle.digest('SHA-256', enc.encode(digestParts));
    return new Uint8Array(digestBuffer);
  }

  async anchorBatch(records) {
    if (!records || records.length === 0) return null;

    const merkleRoot = await this.computeBatchHash(records);
    const startTs = records[0].timestamp;
    const endTs = records[records.length - 1].timestamp;

    // Anchor Layout: [Opcode 1B][StartTs 8B][EndTs 8B][Count 4B][MerkleRoot 32B]
    const payload = new Uint8Array(1 + 8 + 8 + 4 + 32);
    const dv = new DataView(payload.buffer);
    payload[0] = 0x01;
    dv.setBigUint64(1, BigInt(startTs), true);
    dv.setBigUint64(9, BigInt(endTs), true);
    dv.setUint32(17, records.length, true);
    payload.set(merkleRoot, 21);

    const programPubkey = new solanaWeb3.PublicKey(this.programIdStr);
    const [stationPda] = solanaWeb3.PublicKey.findProgramAddressSync(
      [new TextEncoder().encode('factoroid_depin'), this.keypair.publicKey.toBuffer()],
      programPubkey
    );

    const instruction = new solanaWeb3.TransactionInstruction({
      keys: [
        { pubkey: stationPda, isSigner: false, isWritable: true },
        { pubkey: this.keypair.publicKey, isSigner: true, isWritable: true }
      ],
      programId: programPubkey,
      data: payload
    });

    const tx = new solanaWeb3.Transaction().add(instruction);
    tx.feePayer = this.keypair.publicKey;
    const { blockhash } = await this.connection.getLatestBlockhash();
    tx.recentBlockhash = blockhash;

    tx.sign(this.keypair);
    const signature = await this.connection.sendRawTransaction(tx.serialize(), {
      skipPreflight: false,
      preflightCommitment: 'confirmed'
    });

    await this.connection.confirmTransaction(signature, 'confirmed');
    return signature;
  }
}

8. src/worker.js (Background Thread Coordinator)
import { openDatabase, storeTelemetry, getUnanchoredRecords, markAsAnchored } from './db.js';
import { S7DirectClient } from './s7client.js';
import { WasiWebRTCHost } from './wasi_host.js';
import { SolanaDePINClient } from './solana.js';

let s7Client = null;
let wasiHost = null;
let wasmExports = null;
let solana = null;
let cycleTimer = null;

const nodeId = `station_${Math.random().toString(36).substring(2, 9)}`;
const channel = new BroadcastChannel('factoroid_channel');

let target = {
  dbName: 'DB_Sensors',
  dbNum: 5,
  offset: 0,
  type: 'REAL',
  byteLength: 4
};

function getByteLengthForType(type) {
  switch (type) {
    case 'REAL':
    case 'DINT': return 4;
    case 'INT':  return 2;
    case 'BYTE': return 1;
    default:     return 4;
  }
}

async function initWasiModule() {
  wasiHost = new WasiWebRTCHost(nodeId, async (replicatedRawBytes, originPeer) => {
    const dv = new DataView(replicatedRawBytes.buffer);
    const ts = Number(dv.getBigUint64(0, true));
    const stationId = dv.getUint32(8, true);
    const dbNum = dv.getUint16(12, true);
    const off = dv.getUint16(14, true);
    const val = dv.getFloat32(16, true);

    const tagIdentifier = `mesh_replica_st${stationId}_db${dbNum}_off${off}`;
    await storeTelemetry(tagIdentifier, dbNum, val, `peer_${originPeer}`);

    channel.postMessage({
      type: 'MESH_REPLICA_BUFFERED',
      peerId: originPeer,
      tag: tagIdentifier,
      value: val,
      timestamp: ts
    });
  });

  const res = await fetch('../dist/mesh_core.wasm');
  const wasmBinary = await res.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(wasmBinary, wasiHost.getImportObject());

  wasiHost.setInstance(instance);
  wasmExports = instance.exports;
}

async function cycleRead() {
  if (!s7Client || !s7Client.connected) return;

  try {
    const rawResponse = await s7Client.readDB(target.dbNum, target.offset, target.byteLength);

    if (rawResponse.length >= 25 && rawResponse[21] === 0xff) {
      const payloadStart = 25;
      const rxPtr = wasmExports.get_s7_rx_buffer();
      const memory = new Uint8Array(wasmExports.memory.buffer);
      memory.set(rawResponse, rxPtr);

      let parsedVal = 0;
      switch (target.type) {
        case 'REAL': parsedVal = wasmExports.wasi_parse_s7_real(payloadStart); break;
        case 'INT':  parsedVal = wasmExports.wasi_parse_s7_int(payloadStart); break;
        case 'DINT': parsedVal = wasmExports.wasi_parse_s7_dint(payloadStart); break;
        case 'BYTE': parsedVal = wasmExports.wasi_parse_s7_byte(payloadStart); break;
      }

      const ts = Date.now();
      wasmExports.wasi_push_local_sample(
        BigInt(ts),
        wasiHost.numericNodeId,
        target.dbNum,
        target.offset,
        parsedVal
      );

      const tagIdentifier = `${target.dbName}.DB${target.dbNum}.${target.type}_OFF${target.offset}`;
      await storeTelemetry(tagIdentifier, target.dbNum, parsedVal, nodeId);

      channel.postMessage({
        type: 'TELEMETRY_UPDATE',
        tag: tagIdentifier,
        value: parsedVal,
        timestamp: ts,
        nodeId
      });
    }
  } catch (err) {
    channel.postMessage({ type: 'STATUS_ERROR', message: err.message });
  }
}

async function commitDePINBatch() {
  if (!solana) return;
  try {
    const unanchored = await getUnanchoredRecords(32);
    if (unanchored.length > 0) {
      const txSig = await solana.anchorBatch(unanchored);
      if (txSig) {
        await markAsAnchored(unanchored.map(r => r.id), txSig);
        channel.postMessage({
          type: 'DEPIN_ANCHORED',
          batchSize: unanchored.length,
          signature: txSig
        });
      }
    }
  } catch (err) {
    channel.postMessage({ type: 'DEPIN_ERROR', message: err.message });
  }
}

channel.onmessage = async (e) => {
  const { action, payload } = e.data;
  switch (action) {
    case 'CONNECT':
      if (cycleTimer) clearInterval(cycleTimer);
      try {
        s7Client = new S7DirectClient(payload.ip, payload.rack, payload.slot);
        await s7Client.connect();
        channel.postMessage({ type: 'STATUS_CONNECTED', ip: payload.ip });
        cycleTimer = setInterval(cycleRead, 500);
      } catch (err) {
        channel.postMessage({ type: 'STATUS_ERROR', message: err.message });
      }
      break;

    case 'SET_TARGET':
      target = { ...payload, byteLength: getByteLengthForType(payload.type) };
      cycleRead();
      break;

    case 'INIT_SOLANA':
      try {
        solana = new SolanaDePINClient(payload.rpcUrl, payload.programId);
        channel.postMessage({ type: 'SOLANA_INITIALIZED', pubkey: solana.getPublicKey() });
      } catch (err) {
        channel.postMessage({ type: 'STATUS_ERROR', message: `Solana Init Failed: ${err.message}` });
      }
      break;

    case 'TRIGGER_READ':
      await cycleRead();
      break;
  }
};

(async () => {
  await openDatabase();
  await initWasiModule();

  setInterval(() => wasiHost.checkLiveness(4000, (deadPeer) => {
    channel.postMessage({
      type: 'PEER_FAILED',
      peerId: deadPeer,
      message: `Station ${deadPeer} unreachable. WebRTC failover active.`
    });
  }), 2000);

  setInterval(commitDePINBatch, 10000);
  channel.postMessage({ type: 'WORKER_READY', nodeId });
})();

9. src/mcp.js (WebMCP Server)
import { getTelemetryHistory, purgeOldRecords, getUnanchoredRecords } from './db.js';

export class WebMCPServer {
  constructor(telemetryStateMap, s7TriggerFn, solanaStateMap) {
    this.telemetryState = telemetryStateMap;
    this.triggerRead = s7TriggerFn;
    this.solanaState = solanaStateMap;
  }

  async handleJsonRpc(request) {
    const { jsonrpc, id, method, params } = request;
    if (jsonrpc !== "2.0") {
      return { jsonrpc: "2.0", id, error: { code: -32600, message: "Invalid Request: Must be JSON-RPC 2.0" } };
    }

    switch (method) {
      case "resources/list":
        return {
          jsonrpc: "2.0",
          id,
          result: {
            resources: [
              { uri: "mcp://plc/live", name: "Currently Observed S7 Record", mimeType: "application/json" },
              { uri: "mcp://storage/history", name: "Local IndexedDB Historical Telemetry", mimeType: "application/json" },
              { uri: "mcp://depin/solana", name: "Solana DePIN State & Anchor History", mimeType: "application/json" }
            ]
          }
        };

      case "resources/read":
        if (params?.uri === "mcp://plc/live") {
          return {
            jsonrpc: "2.0",
            id,
            result: {
              contents: [{
                uri: params.uri,
                mimeType: "application/json",
                text: JSON.stringify(Object.fromEntries(this.telemetryState))
              }]
            }
          };
        }
        if (params?.uri === "mcp://depin/solana") {
          return {
            jsonrpc: "2.0",
            id,
            result: {
              contents: [{
                uri: params.uri,
                mimeType: "application/json",
                text: JSON.stringify(Object.fromEntries(this.solanaState))
              }]
            }
          };
        }
        return { jsonrpc: "2.0", id, error: { code: -32602, message: "Resource not found" } };

      case "tools/list":
        return {
          jsonrpc: "2.0",
          id,
          result: {
            tools: [
              {
                name: "trigger_plc_read",
                description: "Triggers a single-shot execution of the configured S7 PLC read cycle"
              },
              {
                name: "get_history",
                description: "Retrieves time-series telemetry from local IndexedDB for a given tag",
                parameters: {
                  type: "object",
                  properties: {
                    tagId: { type: "string" },
                    windowMinutes: { type: "number", default: 60 }
                  },
                  required: ["tagId"]
                }
              },
              {
                name: "check_depin_backlog",
                description: "Inspects unanchored records queued for Solana consensus batching"
              },
              {
                name: "purge_cache",
                description: "Deletes cached telemetry records older than retention threshold"
              }
            ]
          }
        };

      case "tools/call":
        if (params?.name === "trigger_plc_read") {
          await this.triggerRead();
          return { jsonrpc: "2.0", id, result: { content: [{ type: "text", text: "PLC single-shot read initiated" }] } };
        }
        if (params?.name === "get_history") {
          const windowMs = (params.arguments?.windowMinutes || 60) * 60 * 1000;
          const history = await getTelemetryHistory(params.arguments?.tagId, windowMs);
          return { jsonrpc: "2.0", id, result: { content: [{ type: "text", text: JSON.stringify(history) }] } };
        }
        if (params?.name === "check_depin_backlog") {
          const backlog = await getUnanchoredRecords(100);
          return { jsonrpc: "2.0", id, result: { content: [{ type: "text", text: `Unanchored records count: ${backlog.length}` }] } };
        }
        if (params?.name === "purge_cache") {
          const deleted = await purgeOldRecords(params.arguments?.retentionDays || 7);
          return { jsonrpc: "2.0", id, result: { content: [{ type: "text", text: `Deleted ${deleted} expired records` }] } };
        }
        return { jsonrpc: "2.0", id, error: { code: -32601, message: "Tool not found" } };

      default:
        return { jsonrpc: "2.0", id, error: { code: -32601, message: "Method not found" } };
    }
  }
}

10. sw.js (Service Worker)
const CACHE_NAME = 'factoroid-wasi-depin-v1.3';
const ASSETS = [
  '/',
  '/index.html',
  '/icon-64.svg',
  '/icon-256.svg',
  '/src/app.js',
  '/src/db.js',
  '/src/s7client.js',
  '/src/wasi_host.js',
  '/src/solana.js',
  '/src/worker.js',
  '/src/mcp.js',
  '/dist/mesh_core.wasm',
  '/manifest.webmanifest'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    Promise.all([
      self.clients.claim(),
      caches.keys().then((keys) =>
        Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
      )
    ])
  );
});

self.addEventListener('periodicsync', (event) => {
  if (event.tag === 'factoroid-db-purge') {
    event.waitUntil(
      importScripts('/src/db.js').then(() => purgeOldRecords(7))
    );
  }
});

11. src/app.js (Main UI Orchestrator)
import { purgeOldRecords } from './db.js';
import { WebMCPServer } from './mcp.js';

const channel = new BroadcastChannel('factoroid_channel');
const liveState = new Map();
const solanaState = new Map();
let backgroundWorker = null;

async function registerServiceWorker() {
  if ('serviceWorker' in navigator) {
    try {
      const reg = await navigator.serviceWorker.register('./sw.js', { type: 'module' });
      if ('periodicSync' in reg) {
        const status = await navigator.permissions.query({ name: 'periodic-background-sync' });
        if (status.state === 'granted') {
          await reg.periodicSync.register('factoroid-db-purge', { minInterval: 12 * 60 * 60 * 1000 });
        }
      }
    } catch (err) {
      console.warn('SW registration skipped:', err);
    }
  }
}

function initBackgroundWorker() {
  backgroundWorker = new Worker('./src/worker.js', { type: 'module' });

  channel.onmessage = (event) => {
    const data = event.data;
    const plcStatusEl = document.getElementById('plc-status');
    const meshStatusEl = document.getElementById('mesh-status');
    const depinStatusEl = document.getElementById('depin-status');

    switch (data.type) {
      case 'WORKER_READY':
        plcStatusEl.innerText = `WASI Runtime Ready (Node: ${data.nodeId}). Enter PLC settings and connect.`;
        break;

      case 'STATUS_CONNECTED':
        plcStatusEl.style.borderLeftColor = '#238636';
        plcStatusEl.innerText = `Connected: S7 PLC Active (${data.ip}:102)`;
        break;

      case 'STATUS_ERROR':
        plcStatusEl.style.borderLeftColor = '#da3633';
        plcStatusEl.innerText = `PLC Error: ${data.message}`;
        break;

      case 'TELEMETRY_UPDATE':
        liveState.set(data.tag, { value: data.value, timestamp: data.timestamp });
        renderUiRecord(data.tag, data.value, data.timestamp);
        break;

      case 'SOLANA_INITIALIZED':
        solanaState.set('nodeWallet', data.pubkey);
        depinStatusEl.innerText = `Solana DePIN Wallet: ${data.pubkey.slice(0, 4)}...${data.pubkey.slice(-4)}`;
        break;

      case 'DEPIN_ANCHORED':
        solanaState.set('lastAnchorTx', data.signature);
        solanaState.set('lastAnchorCount', data.batchSize);
        depinStatusEl.innerHTML = `Anchored <b>${data.batchSize}</b> records to Solana: <a href="https://explorer.solana.com/tx/${data.signature}?cluster=devnet" target="_blank" style="color: #58a6ff;">${data.signature.slice(0, 8)}...</a>`;
        break;

      case 'DEPIN_ERROR':
        depinStatusEl.innerText = `DePIN Error: ${data.message}`;
        break;

      case 'MESH_REPLICA_BUFFERED':
        meshStatusEl.innerText = `WebRTC Mesh Active: Failover buffer synced from ${data.peerId}`;
        break;

      case 'PEER_FAILED':
        meshStatusEl.style.borderLeftColor = '#da3633';
        meshStatusEl.innerText = data.message;
        break;
    }
  };
}

function renderUiRecord(tagName, val, timestamp) {
  document.getElementById('card-tag-name').innerText = tagName;
  document.getElementById('card-tag-value').innerText =
    typeof val === 'number' && !Number.isInteger(val) ? val.toFixed(2) : val;
  document.getElementById('card-tag-time').innerText =
    `Last local update: ${new Date(timestamp).toLocaleTimeString()}.${String(timestamp % 1000).padStart(3, '0')}`;
}

function bindEvents() {
  document.getElementById('btn-connect').addEventListener('click', () => {
    const ip = document.getElementById('plc-ip').value.trim();
    const rack = parseInt(document.getElementById('plc-rack').value, 10);
    const slot = parseInt(document.getElementById('plc-slot').value, 10);
    document.getElementById('plc-status').innerText = `Connecting to ${ip} via Direct Sockets...`;

    channel.postMessage({
      action: 'CONNECT',
      payload: { ip, rack, slot }
    });
  });

  document.getElementById('btn-init-solana').addEventListener('click', () => {
    const rpcUrl = document.getElementById('solana-rpc').value.trim();
    const programId = document.getElementById('solana-prog-id').value.trim();
    channel.postMessage({
      action: 'INIT_SOLANA',
      payload: { rpcUrl, programId }
    });
  });

  document.getElementById('btn-apply-record').addEventListener('click', () => {
    const payload = {
      dbName: document.getElementById('db-name').value.trim() || 'DB',
      dbNum: parseInt(document.getElementById('db-num').value, 10),
      offset: parseInt(document.getElementById('rec-offset').value, 10),
      type: document.getElementById('rec-type').value
    };

    document.getElementById('card-tag-name').innerText =
      `${payload.dbName}.DB${payload.dbNum}.${payload.type}_OFF${payload.offset}`;
    document.getElementById('card-tag-value').innerText = '--';

    channel.postMessage({ action: 'SET_TARGET', payload });
  });
}

(async function bootstrap() {
  await registerServiceWorker();
  initBackgroundWorker();
  bindEvents();

  window.mcp = new WebMCPServer(liveState, () => {
    channel.postMessage({ action: 'TRIGGER_READ' });
  }, solanaState);

  setInterval(() => purgeOldRecords(7), 86400000);
})();

12. index.html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="icon" type="image/svg+xml" href="icon-64.svg">
  <link rel="manifest" href="manifest.webmanifest">
  <title>Factoroid - S7 Edge DePIN Station</title>
  <script src="https://unpkg.com/@solana/web3.js@latest/lib/index.iife.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background-color: #0d1117; color: #c9d1d9; font-family: monospace; padding: 24px; }
    .header-bar { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
    .header-icon { width: 36px; height: 36px; display: block; image-rendering: pixelated; }
    h1 { color: #58a6ff; font-size: 1.3rem; }
    .panel { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 16px; margin-bottom: 20px; }
    .row { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; margin-bottom: 12px; }
    .col { display: flex; flex-direction: column; gap: 6px; }
    label { font-size: 0.8rem; color: #8b949e; }
    input, select { background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; padding: 8px 10px; border-radius: 4px; font-family: monospace; }
    button { background: #238636; color: #fff; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: bold; }
    button:hover { background: #2ea043; }
    .status-strip { padding: 10px 14px; background: #161b22; border-left: 4px solid #8b949e; margin-bottom: 12px; font-size: 0.85rem; border-radius: 2px; }
    .display-card { background: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 6px; max-width: 440px; margin-top: 10px; }
    .tag-title { color: #8b949e; font-size: 0.85rem; display: block; word-break: break-all; }
    .tag-value { font-size: 2.4rem; color: #58a6ff; font-weight: bold; margin-top: 10px; }
    .tag-time { font-size: 0.75rem; color: #6e7681; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="header-bar">
    <img src="icon-64.svg" alt="Factoroid Icon" class="header-icon">
    <h1>FACTOROID // WASI S7 DePIN Station</h1>
  </div>

  <!-- Solana Configuration -->
  <div class="panel">
    <div class="row">
      <div class="col">
        <label>Solana RPC Endpoint:</label>
        <input type="text" id="solana-rpc" value="https://api.devnet.solana.com" style="width: 260px;">
      </div>
      <div class="col">
        <label>DePIN Program ID:</label>
        <input type="text" id="solana-prog-id" value="DePIN11111111111111111111111111111111111111" style="width: 320px;">
      </div>
      <button id="btn-init-solana" style="background: #a371f7;">Init Solana Node</button>
    </div>
  </div>

  <!-- PLC Configuration -->
  <div class="panel">
    <div class="row">
      <div class="col">
        <label>PLC IP Address:</label>
        <input type="text" id="plc-ip" value="192.168.1.100" style="width: 140px;">
      </div>
      <div class="col">
        <label>Rack / Slot:</label>
        <div style="display: flex; gap: 4px;">
          <input type="number" id="plc-rack" value="0" style="width: 50px;">
          <input type="number" id="plc-slot" value="1" style="width: 50px;">
        </div>
      </div>
      <button id="btn-connect">Connect PLC</button>
    </div>

    <hr style="border: 0; border-top: 1px solid #30363d; margin: 15px 0;">

    <div class="row">
      <div class="col">
        <label>DB Name:</label>
        <input type="text" id="db-name" value="DB_Sensors" style="width: 120px;">
      </div>
      <div class="col">
        <label>DB Number:</label>
        <input type="number" id="db-num" value="5" min="1" style="width: 65px;">
      </div>
      <div class="col">
        <label>Byte Offset:</label>
        <input type="number" id="rec-offset" value="0" min="0" style="width: 75px;">
      </div>
      <div class="col">
        <label>Record Type:</label>
        <select id="rec-type">
          <option value="REAL">REAL (Float 32)</option>
          <option value="INT">INT (16-bit)</option>
          <option value="DINT">DINT (32-bit)</option>
          <option value="BYTE">BYTE (8-bit)</option>
        </select>
      </div>
      <button id="btn-apply-record" style="background: #1f6feb;">Track Record</button>
    </div>
  </div>

  <!-- Diagnostic Status Strips -->
  <div class="status-strip" id="plc-status">WASI Initializing...</div>
  <div class="status-strip" id="mesh-status" style="border-left-color: #388bfd;">WebRTC WASI Mesh: Listening for peers...</div>
  <div class="status-strip" id="depin-status" style="border-left-color: #a371f7;">Solana DePIN: Idle</div>

  <!-- Live Record Display -->
  <div class="display-card">
    <span class="tag-title" id="card-tag-name">Awaiting configuration...</span>
    <div class="tag-value" id="card-tag-value">--</div>
    <div class="tag-time" id="card-tag-time">Last update: never</div>
  </div>

  <script type="module" src="src/app.js"></script>
</body>
</html>

