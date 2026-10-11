export class StorageService {
  constructor(dbName = 'SharedLLM_Store', version = 3) {
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

        if (!db.objectStoreNames.contains('event_logs')) {
          const logStore = db.createObjectStore('event_logs', { keyPath: 'id', autoIncrement: true });
          logStore.createIndex('by_timestamp', 'timestamp', { unique: false });
          logStore.createIndex('by_category', 'category', { unique: false });
        }
      };

      request.onsuccess = () => {
        this.db = request.result;
        resolve(this);
      };

      request.onerror = () => reject(request.error);
    });
  }

  async logEvent(category, message, details = null) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('event_logs', 'readwrite');
      const store = tx.objectStore('event_logs');
      const entry = {
        timestamp: Date.now(),
        category,
        message,
        details: details ? JSON.stringify(details) : null
      };
      const req = store.add(entry);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async getRecentLogs(limit = 100) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('event_logs', 'readonly');
      const store = tx.objectStore('event_logs');
      const req = store.getAll();

      req.onsuccess = () => {
        const logs = req.result || [];
        logs.sort((a, b) => b.timestamp - a.timestamp);
        resolve(logs.slice(0, limit));
      };
      req.onerror = () => reject(req.error);
    });
  }

  async clearLogs() {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('event_logs', 'readwrite');
      const store = tx.objectStore('event_logs');
      const req = store.clear();
      req.onsuccess = () => resolve(true);
      req.onerror = () => reject(req.error);
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

  async getChunk(chunkId) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('model_chunks', 'readonly');
      const store = tx.objectStore('model_chunks');
      const req = store.get(chunkId);
      req.onsuccess = () => {
        if (!req.result) return resolve(null);
        let buf = req.result.binary;
        if (buf instanceof Uint8Array) {
          buf = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
        } else if (buf instanceof ArrayBuffer) {
          buf = buf.slice(0);
        }
        resolve({
          chunkId: req.result.chunkId,
          metadata: req.result.metadata,
          binary: buf
        });
      };
      req.onerror = () => reject(req.error);
    });
  }

  async getModelChunks(modelName) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('model_chunks', 'readonly');
      const store = tx.objectStore('model_chunks');
      const req = store.getAll();

      req.onsuccess = () => {
        const allRecords = req.result || [];
        const records = allRecords.filter((r) => {
          const m = r.metadata?.modelName || '';
          return m === modelName || m.startsWith(modelName) || modelName.startsWith(m);
        });

        records.sort((a, b) => (a.metadata?.index || 0) - (b.metadata?.index || 0));

        const cleanRecords = records.map((r, idx) => {
          let buf = r.binary;
          if (buf instanceof Uint8Array) {
            buf = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
          } else if (buf instanceof ArrayBuffer) {
            buf = buf.slice(0);
          }

          const total = r.metadata?.totalChunks || records.length;
          const startLayer = Math.floor((idx / Math.max(1, total)) * 30);
          const endLayer = Math.floor(((idx + 1) / Math.max(1, total)) * 30);

          const metadata = {
            ...r.metadata,
            index: r.metadata?.index ?? idx,
            totalChunks: total,
            modelName: r.metadata?.modelName || modelName,
            chunkHash: r.metadata?.chunkHash || r.metadata?.hash || 'LEGACY_HASH',
            context: r.metadata?.context || {
              layerRange: [startLayer, endLayer],
              canYieldNext: idx + 1 < total,
              targetNextContext: idx + 1 < total ? `context://chunk/${idx + 1}` : 'context://tail'
            }
          };

          return { chunkId: r.chunkId, metadata, binary: buf };
        });

        resolve(cleanRecords);
      };
      req.onerror = () => reject(req.error);
    });
  }

  async getChunkSummary(modelName) {
    const chunks = await this.getModelChunks(modelName);
    return chunks.map((c) => ({
      chunkId: c.chunkId,
      index: c.metadata.index,
      sizeBytes: c.binary.byteLength,
      hash: c.metadata.chunkHash,
      layerRange: c.metadata.context.layerRange,
      nextChunkId: c.metadata.nextChunkId
    }));
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
      const storeNames = ['chat_history', 'model_chunks', 'system_state', 'event_logs'];
      const tx = this.db.transaction(storeNames, 'readwrite');
      storeNames.forEach((name) => {
        if (this.db.objectStoreNames.contains(name)) tx.objectStore(name).clear();
      });
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => reject(tx.error);
    });
  }
}
