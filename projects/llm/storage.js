export class StorageService {
  constructor(dbName = 'SharedLLM_Store', version = 2) {
    this.dbName = dbName;
    this.version = version;
    this.db = null;
  }

  async init() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        if (!db.objectStoreNames.contains('chat_history')) {
          db.createObjectStore('chat_history', { keyPath: 'id', autoIncrement: true });
        }

        if (!db.objectStoreNames.contains('model_chunks')) {
          const chunkStore = db.createObjectStore('model_chunks', { keyPath: 'chunkId' });
          chunkStore.createIndex('by_model', 'metadata.modelName', { unique: false });
          chunkStore.createIndex('by_index', 'metadata.index', { unique: false });
          chunkStore.createIndex('by_hash', 'metadata.chunkHash', { unique: false });
        }

        if (!db.objectStoreNames.contains('system_state')) {
          db.createObjectStore('system_state', { keyPath: 'key' });
        }
      };

      request.onsuccess = () => {
        this.db = request.result;
        resolve(this);
      };

      request.onerror = () => reject(request.error);
    });
  }

  async saveMessage(role, text) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('chat_history', 'readwrite');
      const store = tx.objectStore('chat_history');
      const req = store.add({ role, text, timestamp: Date.now() });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async getMessages() {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('chat_history', 'readonly');
      const store = tx.objectStore('chat_history');
      const req = store.getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }

  async putChunk(chunkRecord) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('model_chunks', 'readwrite');
      const store = tx.objectStore('model_chunks');
      const req = store.put({
        chunkId: chunkRecord.metadata.chunkId,
        metadata: chunkRecord.metadata,
        binary: chunkRecord.binary
      });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async getModelChunks(modelName) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('model_chunks', 'readonly');
      const store = tx.objectStore('model_chunks');
      const index = store.index('by_model');
      const req = index.getAll(modelName);

      req.onsuccess = () => {
        const records = req.result || [];
        records.sort((a, b) => a.metadata.index - b.metadata.index);

        // Defensive clone to prevent buffer detachment
        const cleanRecords = records.map((r) => {
          let buf = r.binary;
          if (buf instanceof Uint8Array) {
            buf = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
          } else if (buf instanceof ArrayBuffer) {
            buf = buf.slice(0);
          }
          return {
            chunkId: r.chunkId,
            metadata: r.metadata,
            binary: buf
          };
        });

        resolve(cleanRecords);
      };
      req.onerror = () => reject(req.error);
    });
  }

  async deleteModelChunks(modelName) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('model_chunks', 'readwrite');
      const store = tx.objectStore('model_chunks');
      const index = store.index('by_model');
      const req = index.openKeyCursor(IDBKeyRange.only(modelName));

      req.onsuccess = (e) => {
        const cursor = e.target.result;
        if (cursor) {
          store.delete(cursor.primaryKey);
          cursor.continue();
        } else {
          resolve(true);
        }
      };
      req.onerror = () => reject(req.error);
    });
  }

  async setSystemKey(key, value) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('system_state', 'readwrite');
      const store = tx.objectStore('system_state');
      const req = store.put({ key, value });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async getSystemKey(key) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('system_state', 'readonly');
      const store = tx.objectStore('system_state');
      const req = store.get(key);
      req.onsuccess = () => resolve(req.result ? req.result.value : null);
      req.onerror = () => reject(req.error);
    });
  }

  async clearAll() {
    return new Promise((resolve, reject) => {
      const storeNames = ['chat_history', 'model_chunks', 'system_state'];
      const tx = this.db.transaction(storeNames, 'readwrite');

      storeNames.forEach((name) => {
        if (this.db.objectStoreNames.contains(name)) {
          tx.objectStore(name).clear();
        }
      });

      tx.oncomplete = () => resolve(true);
      tx.onerror = () => reject(tx.error);
    });
  }
}
