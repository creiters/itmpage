export class MeshiDB {
  constructor(name = 'meshi_p2p_store', version = 1) {
    this.name = name;
    this.version = version;
    this.db = null;
  }

  async init() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(this.name, this.version);
      req.onupgradeneeded = (e) => {
        const d = e.target.result;
        if (!d.objectStoreNames.contains('payloads')) {
          const store = d.createObjectStore('payloads', { keyPath: 'id' });
          store.createIndex('timestamp', 'timestamp');
        }
      };
      req.onsuccess = () => {
        this.db = req.result;
        resolve();
      };
      req.onerror = () => reject(req.error);
    });
  }

  async save(doc) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('payloads', 'readwrite');
      tx.objectStore('payloads').put(doc);
      tx.oncomplete = () => resolve(doc);
      tx.onerror = () => reject(tx.error);
    });
  }

  async listAll() {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('payloads', 'readonly');
      const req = tx.objectStore('payloads').getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }
}

