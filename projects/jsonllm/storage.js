export class StorageService {
  constructor(dbName = 'Shared_LLM_DB', version = 3) {
    this.dbName = dbName;
    this.version = version;
    this.db = null;
    this.broadcast = typeof BroadcastChannel !== 'undefined'
      ? new BroadcastChannel('shared-storage-sync')
      : null;
  }

  async init() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);

      request.onupgradeneeded = (e) => {
        const db = e.target.result;

        if (!db.objectStoreNames.contains('model_chunks')) {
          const chunkStore = db.createObjectStore('model_chunks', { keyPath: 'metadata.chunkId' });
          chunkStore.createIndex('by_model', 'metadata.modelName', { unique: false });
          chunkStore.createIndex('by_index', 'metadata.index', { unique: false });
        }

        if (!db.objectStoreNames.contains('model_manifests')) {
          db.createObjectStore('model_manifests', { keyPath: 'modelName' });
        }

        if (!db.objectStoreNames.contains('chat_messages')) {
          db.createObjectStore('chat_messages', { keyPath: 'id', autoIncrement: true });
        }

        if (!db.objectStoreNames.contains('event_logs')) {
          const logStore = db.createObjectStore('event_logs', { keyPath: 'id', autoIncrement: true });
          logStore.createIndex('by_timestamp', 'timestamp', { unique: false });
        }

        if (!db.objectStoreNames.contains('system_kv')) {
          db.createObjectStore('system_kv', { keyPath: 'key' });
        }
      };

      request.onsuccess = () => {
        this.db = request.result;
        resolve(this);
      };

      request.onerror = () => reject(request.error);
    });
  }

  async requestPersistence() {
    if (navigator.storage && navigator.storage.persist) {
      return await navigator.storage.persist();
    }
    return false;
  }

  async isPersistenceGranted() {
    if (navigator.storage && navigator.storage.persisted) {
      return await navigator.storage.persisted();
    }
    return false;
  }

  async getStorageQuota() {
    if (navigator.storage && navigator.storage.estimate) {
      const est = await navigator.storage.estimate();
      return {
        usageMB: ((est.usage || 0) / (1024 * 1024)).toFixed(1),
        quotaMB: ((est.quota || 0) / (1024 * 1024)).toFixed(1)
      };
    }
    return { usageMB: '0', quotaMB: '0' };
  }

  async putChunk(chunkRecord) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('model_chunks', 'readwrite');
      const store = tx.objectStore('model_chunks');
      const req = store.put(chunkRecord);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async batchSaveChunks(chunks, modelName) {
    if (navigator.locks) {
      return await navigator.locks.request('db-chunks-write-lock', async () => {
        return this._executeBatchSave(chunks, modelName);
      });
    }
    return this._executeBatchSave(chunks, modelName);
  }

  async _executeBatchSave(chunks, modelName) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('model_chunks', 'readwrite');
      const store = tx.objectStore('model_chunks');
      for (const c of chunks) {
        store.put(c);
      }
      tx.oncomplete = () => {
        if (this.broadcast) {
          this.broadcast.postMessage({ type: 'CHUNKS_UPDATED', modelName, count: chunks.length });
        }
        resolve(true);
      };
      tx.onerror = () => reject(tx.error);
    });
  }

  async getModelChunks(modelName) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('model_chunks', 'readonly');
      const store = tx.objectStore('model_chunks');
      const index = store.index('by_model');
      const req = index.getAll(modelName);

      req.onsuccess = () => {
        const sorted = (req.result || []).sort((a, b) => a.metadata.index - b.metadata.index);
        resolve(sorted);
      };
      req.onerror = () => reject(req.error);
    });
  }

  async deleteModelChunks(modelName) {
    const chunks = await this.getModelChunks(modelName);
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('model_chunks', 'readwrite');
      const store = tx.objectStore('model_chunks');
      for (const c of chunks) {
        store.delete(c.metadata.chunkId);
      }
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => reject(tx.error);
    });
  }

  async getChunkSummary(modelName) {
    const chunks = await this.getModelChunks(modelName);
    return chunks.map((c) => ({
      chunkId: c.metadata.chunkId,
      index: c.metadata.index,
      layerRange: c.metadata.context?.layerRange || [0, 0],
      sizeBytes: c.binary.byteLength,
      hash: c.metadata.checksum || c.metadata.chunkHash || 'SI_RADIX'
    }));
  }

  async saveManifest(manifest) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('model_manifests', 'readwrite');
      const store = tx.objectStore('model_manifests');
      const req = store.put(manifest);
      req.onsuccess = () => resolve(true);
      req.onerror = () => reject(req.error);
    });
  }

  async getManifest(modelName) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('model_manifests', 'readonly');
      const store = tx.objectStore('model_manifests');
      const req = store.get(modelName);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  }

  async logEvent(category, message) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('event_logs', 'readwrite');
      const store = tx.objectStore('event_logs');
      const req = store.add({ category, message, timestamp: Date.now() });
      req.onsuccess = () => resolve(true);
      req.onerror = () => reject(req.error);
    });
  }

  async getRecentLogs(limit = 100) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('event_logs', 'readonly');
      const store = tx.objectStore('event_logs');
      const req = store.getAll();
      req.onsuccess = () => {
        const list = req.result || [];
        resolve(list.slice(-limit));
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
      const tx = this.db.transaction('chat_messages', 'readwrite');
      const store = tx.objectStore('chat_messages');
      const req = store.add({ role, text, timestamp: Date.now() });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async getMessages() {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('chat_messages', 'readonly');
      const store = tx.objectStore('chat_messages');
      const req = store.getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }

  async setSystemKey(key, value) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('system_kv', 'readwrite');
      const store = tx.objectStore('system_kv');
      const req = store.put({ key, value });
      req.onsuccess = () => resolve(true);
      req.onerror = () => reject(req.error);
    });
  }

  async getSystemKey(key) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('system_kv', 'readonly');
      const store = tx.objectStore('system_kv');
      const req = store.get(key);
      req.onsuccess = () => resolve(req.result ? req.result.value : null);
      req.onerror = () => reject(req.error);
    });
  }

  async clearAll() {
    return new Promise((resolve, reject) => {
      const stores = ['model_chunks', 'model_manifests', 'chat_messages', 'event_logs', 'system_kv'];
      const tx = this.db.transaction(stores, 'readwrite');
      for (const s of stores) {
        tx.objectStore(s).clear();
      }
      tx.oncomplete = () => {
        if (this.broadcast) this.broadcast.postMessage({ type: 'ALL_STORAGE_CLEARED' });
        resolve(true);
      };
      tx.onerror = () => reject(tx.error);
    });
  }
}
