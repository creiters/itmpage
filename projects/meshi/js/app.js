import { MeshiDB } from './db.js';
import { MeshiAuth } from './auth.js';
import { MeshiMeshSocket } from './mesh_socket.js';
import { MeshiSignaler } from './signaling.js';

// 1. Initialize identity & known authorized public keys
const identity = await MeshiAuth.generateIdentity();
const trustedKeys = new Set([
  identity.publicKeyBase64 // Self-authorized loopback or pre-shared keys
]);

const auth = new MeshiAuth(identity.keyPair, trustedKeys);
auth.publicKeyBase64 = identity.publicKeyBase64;

const db = new MeshiDB();
await db.open();

// 2. Establish connection (e.g., Initiator accepts offer)
const { pc, dc } = await MeshiSignaler.createOffer();
const meshSocket = new MeshiMeshSocket('node_peer_1', dc, auth, db);

// 3. Reactive synchronization events
meshSocket.onauthorized = () => {
  console.log('[MESHI] Remote node authorized. Triggering sync...');
};

meshSocket.addEventListener('meshi:synced', (e) => {
  console.log(`[MESHI] Auto-sync completed. Merged ${e.detail.mergedCount} records.`);
});

meshSocket.addEventListener('meshi:network-online', () => {
  console.log('[MESHI] Network state restored. Re-negotiating ICE channels...');
});
