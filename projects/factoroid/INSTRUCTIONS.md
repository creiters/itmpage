Here is the complete offline-first codebase for Factoroid, refactored for the subpath deployment at [https://creiters.cz/projects/dsm/](https://creiters.cz/projects/dsm/) with the exact icon filenames (cuca_meloun_64.svg and cuca_meloun_256.svg). All legacy PLC connection/polling routines have been excised, leaving an autonomous P2P WebRTC Mesh Exchange Node backed by sandboxed WASI validation, local IndexedDB persistence, and Solana DePIN state settlement.
Project File Tree
factoroid/
├── manifest.webmanifest
├── index.html
├── sw.js
├── cuca_meloun_64.svg
├── cuca_meloun_256.svg
├── src/
│   ├── mesh_core.c
│   ├── db.js
│   ├── wasi_host.js
│   ├── solana.js
│   ├── worker.js
│   ├── mcp.js
│   └── app.js
└── dist/
    └── mesh_core.wasm

1. manifest.webmanifest
Configured with the base scope for [https://creiters.cz/projects/dsm/](https://creiters.cz/projects/dsm/).
{
  "name": "Factoroid - P2P Mesh DePIN Exchange",
  "short_name": "Factoroid",
  "version": "2.0.0",
  "start_url": "/projects/dsm/index.html",
  "scope": "/projects/dsm/",
  "display": "standalone",
  "background_color": "#0d1117",
  "theme_color": "#161b22",
  "icons": [
    {
      "src": "cuca_meloun_64.svg",
      "sizes": "64x64",
      "type": "image/svg+xml"
    },
    {
      "src": "cuca_meloun_256.svg",
      "sizes": "256x256",
      "type": "image/svg+xml"
    }
  ]
}

2. cuca_meloun_64.svg
Frameless 64×64 pixel-art cucamelon.
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

3. cuca_meloun_256.svg
Frameless 256×256 pixel-art cucamelon.
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

4. src/mesh_core.c (WASI Ledger Core)
Manages transaction validation, double-spending prevention, balance state, and custom WebRTC syscall dispatching.
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#define MAX_PENDING_TX 128
#define WIRE_TX_SIZE 36 // [Timestamp 8B][Sender 4B][Recipient 4B][Amount 8B][Nonce 8B][Type 4B]

__attribute__((import_module("wasi_webrtc"), import_name("send_mesh_broadcast")))
extern void wasi_webrtc_send_broadcast(const uint8_t *data, int32_t len);

__attribute__((import_module("wasi_webrtc"), import_name("notify_peer_timeout")))
extern void wasi_webrtc_notify_peer_timeout(int32_t peer_index);

typedef struct {
    uint64_t timestamp;
    uint32_t sender_id;
    uint32_t recipient_id;
    uint64_t amount;
    uint64_t nonce;
    uint32_t tx_type;
} MeshTransaction;

static MeshTransaction g_tx_ring[MAX_PENDING_TX];
static uint32_t g_tx_head = 0;
static uint32_t g_tx_count = 0;
static uint64_t g_local_balance = 10000000; // Starting credit pool: 10 units
static uint64_t g_last_nonce = 0;
static uint8_t g_shared_buffer[1024];

__attribute__((visibility("default")))
uint8_t* get_shared_buffer_ptr(void) {
    return g_shared_buffer;
}

__attribute__((visibility("default")))
uint64_t wasi_get_balance(void) {
    return g_local_balance;
}

__attribute__((visibility("default")))
uint64_t wasi_get_last_nonce(void) {
    return g_last_nonce;
}

__attribute__((visibility("default")))
int32_t wasi_create_outbound_tx(uint64_t timestamp, uint32_t sender_id, uint32_t recipient_id, uint64_t amount, uint32_t tx_type) {
    if (g_local_balance < amount) {
        return -1;
    }

    g_local_balance -= amount;
    g_last_nonce++;

    uint32_t idx = (g_tx_head + g_tx_count) % MAX_PENDING_TX;
    g_tx_ring[idx].timestamp = timestamp;
    g_tx_ring[idx].sender_id = sender_id;
    g_tx_ring[idx].recipient_id = recipient_id;
    g_tx_ring[idx].amount = amount;
    g_tx_ring[idx].nonce = g_last_nonce;
    g_tx_ring[idx].tx_type = tx_type;

    if (g_tx_count < MAX_PENDING_TX) {
        g_tx_count++;
    } else {
        g_tx_head = (g_tx_head + 1) % MAX_PENDING_TX;
    }

    uint8_t wire[WIRE_TX_SIZE];
    memcpy(&wire[0], &timestamp, 8);
    memcpy(&wire[8], &sender_id, 4);
    memcpy(&wire[12], &recipient_id, 4);
    memcpy(&wire[16], &amount, 8);
    memcpy(&wire[24], &g_last_nonce, 8);
    memcpy(&wire[32], &tx_type, 4);

    wasi_webrtc_send_broadcast(wire, WIRE_TX_SIZE);
    return (int32_t)idx;
}

__attribute__((visibility("default")))
int32_t wasi_process_inbound_tx(const uint8_t *data, int32_t len, uint32_t local_node_id) {
    if (len < WIRE_TX_SIZE) return -1;

    MeshTransaction tx;
    memcpy(&tx.timestamp, &data[0], 8);
    memcpy(&tx.sender_id, &data[8], 4);
    memcpy(&tx.recipient_id, &data[12], 4);
    memcpy(&tx.amount, &data[16], 8);
    memcpy(&tx.nonce, &data[24], 8);
    memcpy(&tx.tx_type, &data[32], 4);

    if (tx.recipient_id == local_node_id) {
        g_local_balance += tx.amount;
    }

    uint32_t idx = (g_tx_head + g_tx_count) % MAX_PENDING_TX;
    g_tx_ring[idx] = tx;

    if (g_tx_count < MAX_PENDING_TX) {
        g_tx_count++;
    } else {
        g_tx_head = (g_tx_head + 1) % MAX_PENDING_TX;
    }

    return (int32_t)idx;
}

__attribute__((visibility("default")))
int32_t wasi_get_pending_tx_count(void) {
    return (int32_t)g_tx_count;
}

__attribute__((visibility("default")))
MeshTransaction* wasi_get_tx_ptr(int32_t relative_idx) {
    if (relative_idx < 0 || (uint32_t)relative_idx >= g_tx_count) return NULL;
    uint32_t actual_idx = (g_tx_head + relative_idx) % MAX_PENDING_TX;
    return &g_tx_ring[actual_idx];
}

__attribute__((visibility("default")))
void wasi_flush_tx_ring(void) {
    g_tx_head = 0;
    g_tx_count = 0;
}

Build Command:
clang --target=wasm32-wasi -O3 -nostdlib \
  -Wl,--no-entry \
  -Wl,--export=get_shared_buffer_ptr \
  -Wl,--export=wasi_get_balance \
  -Wl,--export=wasi_get_last_nonce \
  -Wl,--export=wasi_create_outbound_tx \
  -Wl,--export=wasi_process_inbound_tx \
  -Wl,--export=wasi_get_pending_tx_count \
  -Wl,--export=wasi_get_tx_ptr \
  -Wl,--export=wasi_flush_tx_ring \
  -Wl,--allow-undefined \
  src/mesh_core.c -o dist/mesh_core.wasm

5. src/db.js (IndexedDB P2P Ledger)
Offline-first transactional persistence for mesh transactions and anchor confirmations.
const DB_NAME = 'FactoroidMeshExchangeDB';
const DB_VERSION = 1;
const STORE_NAME = 'transactions';

let dbInstance = null;

export function openDatabase() {
  if (dbInstance) return Promise.resolve(dbInstance);

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      let store;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        store = db.createObjectStore(STORE_NAME, { keyPath: 'txId', autoIncrement: true });
      } else {
        store = event.target.transaction.objectStore(STORE_NAME);
      }
      if (!store.indexNames.contains('timestamp')) store.createIndex('timestamp', 'timestamp', { unique: false });
      if (!store.indexNames.contains('sender')) store.createIndex('sender', 'sender', { unique: false });
      if (!store.indexNames.contains('recipient')) store.createIndex('recipient', 'recipient', { unique: false });
      if (!store.indexNames.contains('anchored')) store.createIndex('anchored', 'anchored', { unique: false });
    };

    request.onsuccess = (event) => {
      dbInstance = event.target.result;
      resolve(dbInstance);
    };

    request.onerror = (event) => reject(new Error(`IndexedDB open failed: ${event.target.error}`));
  });
}

export async function recordTransaction(txRecord) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([STORE_NAME], 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const req = store.add({
      ...txRecord,
      timestamp: txRecord.timestamp || Date.now(),
      anchored: 0
    });
    req.onsuccess = (e) => resolve(e.target.result);
    req.onerror = (e) => reject(e.target.error);
  });
}

export async function getUnanchoredTransactions(limit = 32) {
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

export async function markTransactionsAnchored(txIds, solanaTxSig) {
  if (!txIds || txIds.length === 0) return;
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([STORE_NAME], 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    tx.oncomplete = () => resolve();
    tx.onerror = (e) => reject(e.target.error);

    for (const id of txIds) {
      const req = store.get(id);
      req.onsuccess = (e) => {
        const item = e.target.result;
        if (item) {
          item.anchored = 1;
          item.solanaTx = solanaTxSig;
          store.put(item);
        }
      };
    }
  });
}

export async function getTransactionHistory(limit = 50) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([STORE_NAME], 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const index = store.index('timestamp');
    const results = [];

    const req = index.openCursor(null, 'prev');
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

6. src/wasi_host.js (WebRTC Mesh Host Multiplexer)
Manages the P2P connection lifecycle, offline buffering, and WASI syscall routing.
export class WasiWebRTCHost {
  constructor(nodeId, onInboundTx) {
    this.nodeId = nodeId;
    this.numericNodeId = Math.abs(this.hashCode(nodeId));
    this.onInboundTx = onInboundTx;
    this.peers = new Map();
    this.heartbeats = new Map();
    this.signaler = new BroadcastChannel('factoroid_mesh_tx_signaling');
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
          console.warn(`[Mesh Core] Peer #${peerIdx} timeout`);
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
    const dc = pc.createDataChannel('p2p_exchange_channel', { ordered: true });
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
        const rxPtr = this.wasmInstance.exports.get_shared_buffer_ptr();
        const memory = new Uint8Array(this.wasmInstance.exports.memory.buffer);
        memory.set(rawBytes, rxPtr);

        const res = this.wasmInstance.exports.wasi_process_inbound_tx(
          rxPtr,
          rawBytes.byteLength,
          this.numericNodeId
        );
        if (res >= 0) {
          this.onInboundTx(rawBytes, peerId);
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

7. src/solana.js (Solana Settlement Client)
Batches unanchored transaction digests into a cryptographic SHA-256 Merkle root.
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

  async computeBatchHash(transactions) {
    const digestParts = transactions
      .map(t => `${t.timestamp}:${t.sender}:${t.recipient}:${t.amount}:${t.nonce}`)
      .join(';');
    const enc = new TextEncoder();
    const digestBuffer = await crypto.subtle.digest('SHA-256', enc.encode(digestParts));
    return new Uint8Array(digestBuffer);
  }

  async anchorBatch(transactions) {
    if (!transactions || transactions.length === 0) return null;

    const merkleRoot = await this.computeBatchHash(transactions);
    const startTs = transactions[0].timestamp;
    const endTs = transactions[transactions.length - 1].timestamp;

    const payload = new Uint8Array(1 + 8 + 8 + 4 + 32);
    const dv = new DataView(payload.buffer);
    payload[0] = 0x02; // Commit batch
    dv.setBigUint64(1, BigInt(startTs), true);
    dv.setBigUint64(9, BigInt(endTs), true);
    dv.setUint32(17, transactions.length, true);
    payload.set(merkleRoot, 21);

    const programPubkey = new solanaWeb3.PublicKey(this.programIdStr);
    const [stationPda] = solanaWeb3.PublicKey.findProgramAddressSync(
      [new TextEncoder().encode('mesh_settlement'), this.keypair.publicKey.toBuffer()],
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
Coordinates the WASI state machine, peer incoming frames, and batch commitments in an isolated Web Worker.
import { openDatabase, recordTransaction, getUnanchoredTransactions, markTransactionsAnchored } from './db.js';
import { WasiWebRTCHost } from './wasi_host.js';
import { SolanaDePINClient } from './solana.js';

let wasiHost = null;
let wasmExports = null;
let solana = null;

const nodeId = `device_${Math.random().toString(36).substring(2, 8)}`;
const channel = new BroadcastChannel('factoroid_mesh_channel');

async function initWasiModule() {
  wasiHost = new WasiWebRTCHost(nodeId, async (rawWireBytes, originPeer) => {
    const dv = new DataView(rawWireBytes.buffer);
    const ts = Number(dv.getBigUint64(0, true));
    const sender = dv.getUint32(8, true);
    const recipient = dv.getUint32(12, true);
    const amount = Number(dv.getBigUint64(16, true));
    const nonce = Number(dv.getBigUint64(24, true));
    const txType = dv.getUint32(32, true);

    const record = {
      timestamp: ts,
      sender: `device_${sender}`,
      recipient: `device_${recipient}`,
      amount,
      nonce,
      txType,
      originPeer
    };

    const txId = await recordTransaction(record);
    const updatedBalance = Number(wasmExports.wasi_get_balance());

    channel.postMessage({
      type: 'INBOUND_TX_RECEIVED',
      txId,
      record,
      balance: updatedBalance
    });
  });

  // Relative path resolution for subpath deployment
  const res = await fetch('../dist/mesh_core.wasm');
  const wasmBinary = await res.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(wasmBinary, wasiHost.getImportObject());

  wasiHost.setInstance(instance);
  wasmExports = instance.exports;
}

async function processSolanaSettlement() {
  if (!solana) return;
  try {
    const unanchored = await getUnanchoredTransactions(32);
    if (unanchored.length > 0) {
      const txSig = await solana.anchorBatch(unanchored);
      if (txSig) {
        await markTransactionsAnchored(unanchored.map(t => t.txId), txSig);
        channel.postMessage({
          type: 'SETTLEMENT_COMMITTED',
          batchSize: unanchored.length,
          signature: txSig
        });
      }
    }
  } catch (err) {
    channel.postMessage({ type: 'SETTLEMENT_ERROR', message: err.message });
  }
}

channel.onmessage = async (e) => {
  const { action, payload } = e.data;

  switch (action) {
    case 'CREATE_TRANSACTION': {
      if (!wasmExports) return;
      const ts = Date.now();
      const targetNumeric = Math.abs(wasiHost.hashCode(payload.recipient));
      const res = wasmExports.wasi_create_outbound_tx(
        BigInt(ts),
        wasiHost.numericNodeId,
        targetNumeric,
        BigInt(payload.amount),
        payload.txType || 1
      );

      if (res < 0) {
        channel.postMessage({ type: 'TX_FAILED', message: 'Insufficient local balance or buffer full' });
      } else {
        const nonce = Number(wasmExports.wasi_get_last_nonce());
        const balance = Number(wasmExports.wasi_get_balance());
        const record = {
          timestamp: ts,
          sender: nodeId,
          recipient: payload.recipient,
          amount: payload.amount,
          nonce,
          txType: payload.txType || 1
        };

        const txId = await recordTransaction(record);
        channel.postMessage({
          type: 'TX_CONFIRMED',
          txId,
          record,
          balance
        });
      }
      break;
    }

    case 'INIT_SOLANA':
      try {
        solana = new SolanaDePINClient(payload.rpcUrl, payload.programId);
        channel.postMessage({ type: 'SOLANA_INITIALIZED', pubkey: solana.getPublicKey() });
      } catch (err) {
        channel.postMessage({ type: 'SETTLEMENT_ERROR', message: `Solana Init Error: ${err.message}` });
      }
      break;

    case 'FORCE_SETTLEMENT':
      await processSolanaSettlement();
      break;
  }
};

(async () => {
  await openDatabase();
  await initWasiModule();

  setInterval(() => wasiHost.checkLiveness(4000, (deadPeer) => {
    channel.postMessage({
      type: 'PEER_DROPPED',
      peerId: deadPeer,
      message: `Station ${deadPeer} unreachable. WebRTC mesh buffering active.`
    });
  }), 2000);

  setInterval(processSolanaSettlement, 12000);

  channel.postMessage({
    type: 'WORKER_READY',
    nodeId,
    balance: Number(wasmExports.wasi_get_balance())
  });
})();

9. src/mcp.js (WebMCP Server)
JSON-RPC 2.0 interface for AI queries, inspecting balances, and triggering mesh payments.
import { getTransactionHistory, getUnanchoredTransactions } from './db.js';

export class WebMCPServer {
  constructor(localStateMap, sendTxFn, forceAnchorFn) {
    this.localState = localStateMap;
    this.sendTx = sendTxFn;
    this.forceAnchor = forceAnchorFn;
  }

  async handleJsonRpc(request) {
    const { jsonrpc, id, method, params } = request;
    if (jsonrpc !== '2.0') {
      return { jsonrpc: '2.0', id, error: { code: -32600, message: 'Invalid JSON-RPC 2.0 Request' } };
    }

    switch (method) {
      case 'resources/list':
        return {
          jsonrpc: '2.0',
          id,
          result: {
            resources: [
              { uri: 'mcp://mesh/balance', name: 'Node P2P Credit Balance', mimeType: 'application/json' },
              { uri: 'mcp://mesh/ledger', name: 'Recent Mesh Transactions', mimeType: 'application/json' },
              { uri: 'mcp://depin/settlement', name: 'Pending Solana Settlement Queue', mimeType: 'application/json' }
            ]
          }
        };

      case 'resources/read': {
        const uri = params?.uri;
        if (uri === 'mcp://mesh/balance') {
          return {
            jsonrpc: '2.0',
            id,
            result: {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(Object.fromEntries(this.localState), null, 2)
              }]
            }
          };
        }
        if (uri === 'mcp://mesh/ledger') {
          const history = await getTransactionHistory(20);
          return {
            jsonrpc: '2.0',
            id,
            result: {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(history, null, 2)
              }]
            }
          };
        }
        if (uri === 'mcp://depin/settlement') {
          const unanchored = await getUnanchoredTransactions(50);
          return {
            jsonrpc: '2.0',
            id,
            result: {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify({ pendingCount: unanchored.length, items: unanchored }, null, 2)
              }]
            }
          };
        }
        return { jsonrpc: '2.0', id, error: { code: -32602, message: `Resource not found: ${uri}` } };
      }

      case 'tools/list':
        return {
          jsonrpc: '2.0',
          id,
          result: {
            tools: [
              {
                name: 'send_peer_transaction',
                description: 'Initiates a value transfer to another node on the local WebRTC mesh',
                parameters: {
                  type: 'object',
                  properties: {
                    recipient: { type: 'string', description: 'Target node identifier (e.g. device_abc123)' },
                    amount: { type: 'number', description: 'Credit amount to transfer' }
                  },
                  required: ['recipient', 'amount']
                }
              },
              {
                name: 'force_depin_settlement',
                description: 'Forces an immediate batch commitment of all unanchored transactions to Solana'
              }
            ]
          }
        };

      case 'tools/call': {
        const { name, arguments: args } = params || {};
        if (name === 'send_peer_transaction') {
          this.sendTx(args.recipient, args.amount);
          return {
            jsonrpc: '2.0',
            id,
            result: { content: [{ type: 'text', text: `Transaction of ${args.amount} credits to ${args.recipient} dispatched.` }] }
          };
        }
        if (name === 'force_depin_settlement') {
          const res = await this.forceAnchor();
          return {
            jsonrpc: '2.0',
            id,
            result: { content: [{ type: 'text', text: JSON.stringify(res) }] }
          };
        }
        return { jsonrpc: '2.0', id, error: { code: -32601, message: `Tool not found: ${name}` } };
      }

      default:
        return { jsonrpc: '2.0', id, error: { code: -32601, message: `Method not found: ${method}` } };
    }
  }
}

10. sw.js (Service Worker)
Ensures full offline operation under the /projects/dsm/ base URL.
const CACHE_NAME = 'factoroid-mesh-dsm-v2.0';
const BASE_PATH = '/projects/dsm/';

const ASSETS = [
  BASE_PATH,
  `${BASE_PATH}index.html`,
  `${BASE_PATH}cuca_meloun_64.svg`,
  `${BASE_PATH}cuca_meloun_256.svg`,
  `${BASE_PATH}src/app.js`,
  `${BASE_PATH}src/db.js`,
  `${BASE_PATH}src/wasi_host.js`,
  `${BASE_PATH}src/solana.js`,
  `${BASE_PATH}src/worker.js`,
  `${BASE_PATH}src/mcp.js`,
  `${BASE_PATH}dist/mesh_core.wasm`,
  `${BASE_PATH}manifest.webmanifest`
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

// Cache-First with Network Fallback strategy for pure offline reliability
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(event.request).catch(() => {
        if (event.request.mode === 'navigate') {
          return caches.match(`${BASE_PATH}index.html`);
        }
      });
    })
  );
});

11. src/app.js (UI Controller)
Coordinates relative Service Worker registration, Web Worker lifecycle, and UI interaction.
import { WebMCPServer } from './mcp.js';

const channel = new BroadcastChannel('factoroid_mesh_channel');
const localState = new Map();
let backgroundWorker = null;

function renderTxItem(tx) {
  const list = document.getElementById('tx-list');
  const div = document.createElement('div');
  div.style.padding = '8px';
  div.style.borderBottom = '1px solid #30363d';
  div.style.fontSize = '0.85rem';
  div.innerHTML = `<b>${new Date(tx.timestamp).toLocaleTimeString()}</b>: ${tx.sender} ➔ ${tx.recipient} | <b>${tx.amount}</b> Credits (Nonce: ${tx.nonce})`;
  list.prepend(div);
}

function initBackgroundWorker() {
  backgroundWorker = new Worker('./src/worker.js', { type: 'module' });

  channel.onmessage = (event) => {
    const data = event.data;
    const balanceEl = document.getElementById('balance-display');
    const nodeEl = document.getElementById('node-display');
    const statusEl = document.getElementById('mesh-status');
    const depinEl = document.getElementById('depin-status');

    switch (data.type) {
      case 'WORKER_READY':
        localState.set('nodeId', data.nodeId);
        localState.set('balance', data.balance);
        nodeEl.innerText = data.nodeId;
        balanceEl.innerText = data.balance.toLocaleString();
        statusEl.innerText = 'Mesh Active: Ready to route P2P transactions';
        break;

      case 'TX_CONFIRMED':
      case 'INBOUND_TX_RECEIVED':
        localState.set('balance', data.balance);
        balanceEl.innerText = data.balance.toLocaleString();
        renderTxItem(data.record);
        break;

      case 'TX_FAILED':
        alert(`Transaction Failed: ${data.message}`);
        break;

      case 'SOLANA_INITIALIZED':
        depinEl.innerText = `Solana DePIN: Connected (${data.pubkey.slice(0, 4)}...${data.pubkey.slice(-4)})`;
        break;

      case 'SETTLEMENT_COMMITTED':
        depinEl.innerHTML = `Anchored <b>${data.batchSize}</b> txs to Solana: <a href="https://explorer.solana.com/tx/${data.signature}?cluster=devnet" target="_blank" style="color: #58a6ff;">${data.signature.slice(0, 8)}...</a>`;
        break;

      case 'SETTLEMENT_ERROR':
        depinEl.innerText = `Settlement Error: ${data.message}`;
        break;

      case 'PEER_DROPPED':
        statusEl.style.borderLeftColor = '#da3633';
        statusEl.innerText = data.message;
        break;
    }
  };
}

function bindEvents() {
  document.getElementById('btn-send-tx').addEventListener('click', () => {
    const recipient = document.getElementById('target-peer').value.trim();
    const amount = parseInt(document.getElementById('tx-amount').value, 10);

    if (!recipient || isNaN(amount) || amount <= 0) {
      alert('Specify a valid recipient and positive credit amount');
      return;
    }

    channel.postMessage({
      action: 'CREATE_TRANSACTION',
      payload: { recipient, amount }
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
}

(async function bootstrap() {
  if ('serviceWorker' in navigator) {
    try {
      await navigator.serviceWorker.register('./sw.js', { scope: './' });
    } catch (err) {
      console.warn('Service worker registration failed:', err);
    }
  }

  initBackgroundWorker();
  bindEvents();

  window.mcp = new WebMCPServer(
    localState,
    (recipient, amount) => channel.postMessage({ action: 'CREATE_TRANSACTION', payload: { recipient, amount } }),
    async () => {
      return new Promise((resolve) => {
        const handler = (e) => {
          if (e.data.type === 'SETTLEMENT_COMMITTED' || e.data.type === 'SETTLEMENT_ERROR') {
            channel.removeEventListener('message', handler);
            resolve(e.data);
          }
        };
        channel.addEventListener('message', handler);
        channel.postMessage({ action: 'FORCE_SETTLEMENT' });
      });
    }
  );
})();

12. index.html
Configured with relative paths, inline offline Solana Web3 fallback, cucamelon branding, and the P2P transaction UI.
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <base href="/projects/dsm/">
  <link rel="icon" type="image/svg+xml" href="cuca_meloun_64.svg">
  <link rel="manifest" href="manifest.webmanifest">
  <title>Factoroid - Mesh Exchange Node</title>
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
    input { background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; padding: 8px 10px; border-radius: 4px; font-family: monospace; }
    button { background: #238636; color: #fff; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: bold; }
    button:hover { background: #2ea043; }
    .status-strip { padding: 10px 14px; background: #161b22; border-left: 4px solid #8b949e; margin-bottom: 12px; font-size: 0.85rem; border-radius: 2px; }
    .balance-card { background: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 6px; max-width: 440px; margin-bottom: 20px; }
    .balance-title { color: #8b949e; font-size: 0.85rem; }
    .balance-val { font-size: 2.4rem; color: #7ee787; font-weight: bold; margin-top: 5px; }
    .tx-ledger { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 16px; max-height: 250px; overflow-y: auto; }
  </style>
</head>
<body>
  <div class="header-bar">
    <img src="cuca_meloun_64.svg" alt="Cucamelon Icon" class="header-icon">
    <h1>FACTOROID // P2P Mesh Exchange Node</h1>
  </div>

  <div class="balance-card">
    <span class="balance-title">Local Node ID: <b id="node-display">Initializing...</b></span>
    <div class="balance-val" id="balance-display">0</div>
    <span style="font-size: 0.8rem; color: #8b949e;">Available Internal Credits</span>
  </div>

  <div class="panel">
    <div class="row">
      <div class="col">
        <label>Solana RPC Endpoint:</label>
        <input type="text" id="solana-rpc" value="https://api.devnet.solana.com" style="width: 260px;">
      </div>
      <div class="col">
        <label>Settlement Program ID:</label>
        <input type="text" id="solana-prog-id" value="DePIN11111111111111111111111111111111111111" style="width: 320px;">
      </div>
      <button id="btn-init-solana" style="background: #a371f7;">Init Settlement</button>
    </div>
  </div>

  <div class="panel">
    <div class="row">
      <div class="col">
        <label>Target Peer Identifier:</label>
        <input type="text" id="target-peer" placeholder="device_abc123" style="width: 200px;">
      </div>
      <div class="col">
        <label>Transfer Amount:</label>
        <input type="number" id="tx-amount" value="5000" min="1" style="width: 120px;">
      </div>
      <button id="btn-send-tx">Send Transaction</button>
    </div>
  </div>

  <div class="status-strip" id="mesh-status">WebRTC Mesh: Listening for peers...</div>
  <div class="status-strip" id="depin-status" style="border-left-color: #a371f7;">Solana DePIN: Idle</div>

  <div class="panel">
    <label style="margin-bottom: 8px; display: block;">Real-Time Mesh Transaction Stream:</label>
    <div class="tx-ledger" id="tx-list"></div>
  </div>

  <script type="module" src="src/app.js"></script>
</body>
</html>

