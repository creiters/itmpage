export class MeshiDB {
  constructor(dbName = 'meshi_mesh_db', version = 1) {
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
          const store = d.createObjectStore('documents', { keyPath: 'id' });
          store.createIndex('clock', 'clock');
          store.createIndex('updatedAt', 'updatedAt');
        }
      };
      req.onsuccess = () => {
        this.db = req.result;
        resolve(this.db);
      };
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
      const getReq = store.get(remoteDoc.id);

      getReq.onsuccess = () => {
        const localDoc = getReq.result;
        // Last-Write-Wins based on logical clock, timestamp tie-breaker
        if (!localDoc || remoteDoc.clock > localDoc.clock || 
           (remoteDoc.clock === localDoc.clock && remoteDoc.updatedAt > localDoc.updatedAt)) {
          this.clock = Math.max(this.clock, remoteDoc.clock) + 1;
          store.put(remoteDoc);
          resolve({ merged: true, doc: remoteDoc });
        } else {
          resolve({ merged: false, doc: localDoc });
        }
      };
      getReq.onerror = () => reject(getReq.error);
    });
  }

  async getAllDocuments() {
    if (!this.db) await this.open();
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('documents', 'readonly');
      const req = tx.objectStore('documents').getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }
}
