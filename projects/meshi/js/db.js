export class MeshiDB {
  constructor(dbName = 'meshi_cluster_db', version = 1) {
    this.dbName = dbName;
    this.version = version;
    this.db = null;
    this.clock = 0;
  }

  async open() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(this.dbName, this.version);
      req.onupgradeneeded = (e) => {
        const d = e.target.result;
        if (!d.objectStoreNames.contains('documents')) {
          const docs = d.createObjectStore('documents', { keyPath: 'id' });
          docs.createIndex('clock', 'clock');
          docs.createIndex('updatedAt', 'updatedAt');
        }
        if (!d.objectStoreNames.contains('trusted_peers')) {
          d.createObjectStore('trusted_peers', { keyPath: 'publicKey' });
        }
      };
      req.onsuccess = () => {
        this.db = req.result;
        resolve(this.db);
      };
      req.onerror = () => reject(req.error);
    });
  }

  async addTrustedPeer(publicKey, alias = 'Subnet Peer') {
    if (!this.db) await this.open();
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('trusted_peers', 'readwrite');
      tx.objectStore('trusted_peers').put({ publicKey, alias, addedAt: Date.now() });
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async isPeerTrusted(publicKey) {
    if (!this.db) await this.open();
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('trusted_peers', 'readonly');
      const req = tx.objectStore('trusted_peers').get(publicKey);
      req.onsuccess = () => resolve(!!req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async listTrustedPeers() {
    if (!this.db) await this.open();
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('trusted_peers', 'readonly');
      const req = tx.objectStore('trusted_peers').getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }

  async putDocument(doc) {
    if (!this.db) await this.open();
    this.clock = Math.max(this.clock, doc.clock || 0) + 1;
    const record = {
      ...doc,
      clock: this.clock,
      updatedAt: Date.now()
    };
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('documents', 'readwrite');
      tx.objectStore('documents').put(record);
      tx.oncomplete = () => resolve(record);
      tx.onerror = () => reject(tx.error);
    });
  }

  async mergeRemoteDocument(remoteDoc) {
    if (!this.db) await this.open();
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('documents', 'readwrite');
      const store = tx.objectStore('documents');
      const req = store.get(remoteDoc.id);

      req.onsuccess = () => {
        const local = req.result;
        if (!local || remoteDoc.clock > local.clock || 
           (remoteDoc.clock === local.clock && remoteDoc.updatedAt > local.updatedAt)) {
          this.clock = Math.max(this.clock, remoteDoc.clock) + 1;
          store.put(remoteDoc);
          resolve({ merged: true, doc: remoteDoc });
        } else {
          resolve({ merged: false, doc: local });
        }
      };
      req.onerror = () => reject(req.error);
    });
  }

  async getDeltaSince(sinceClock = 0) {
    if (!this.db) await this.open();
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('documents', 'readonly');
      const index = tx.objectStore('documents').index('clock');
      const req = index.getAll(IDBKeyRange.lowerBound(sinceClock, true));
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }

  async getAllDocuments() {
    return this.getDeltaSince(-1);
  }
}
