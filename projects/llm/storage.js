export class StorageService {
  constructor(dbName = 'SharedLLM_Store', version = 4) {
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

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        if (!db.objectStoreNames.contains('chat_history')) {
          db.createObjectStore('chat_history', { keyPath: 'id', autoIncrement: true });
        }

        if (!db.objectStoreNames.contains('model_chunks')) {
          const chunkStore = db.createObjectStore('model_chunks', { keyPath: 'chunkId' });
          chunkStore.createIndex('by_model', 'metadata.modelName', { unique: false });
          chunkStore.createIndex('by_index', 'metadata.index', { unique: false });
          chunkStore.createIndex('by_checksum', 'metadata.checksum', { unique: false });
        }

        if (!db.objectStoreNames.contains('si_radix_manifests')) {
          db.createObjectStore('si_radix_manifests', { keyPath: 'modelName' });
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
      const { usage, quota } = await navigator.storage.estimate();
      return {
        usageBytes: usage || 0,
        quotaBytes: quota || 0,
        usageMB: ((usage || 0) / (1024 * 1024)).toFixed(1),
        quotaMB: ((quota || 0) / (1024 * 1024)).toFixed(1),
        percentUsed: quota ? ((usage / quota) * 100).toFixed(1) : '0'
      };
    }
    return { usageBytes: 0, quotaBytes: 0, usageMB: '0', quotaMB: '0', percentUsed: '0' };
  }

  async withLock(lockName, fn) {
    if (navigator.locks) {
      return await navigator.locks.request(lockName, fn);
    }
    return await fn();
  }

  async saveManifest(manifest) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('si_radix_manifests', 'readwrite');
      const store = tx.objectStore('si_radix_manifests');
      const req = store.put(manifest);
      req.onsuccess = () => resolve(true);
      req.onerror = () => reject(req.error);
    });
  }

  async getManifest(modelName) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('si_radix_manifests', 'readonly');
      const store = tx.objectStore('si_radix_manifests');
      const req = store.get(modelName);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  }

  async batchPutChunks(chunks, onProgress) {
    return this.withLock('db-write-lock', async () => {
      return new Promise((resolve, reject) => {
        const tx = this.db.transaction('model_chunks', 'readwrite');
        const store = tx.objectStore('model_chunks');
        let completed = 0;

        tx.oncomplete = () => {
          if (this.broadcast) {
            this.broadcast.postMessage({
              type: 'SEGMENTS_UPDATED',
              count: chunks.length,
              modelName: chunks[0]?.metadata?.modelName
            });
          }
          resolve(true);
        };

        tx.onerror = () => reject(tx.error);
        tx.onabort = () => reject(new Error('Batch SI-Radix segment commit aborted.'));

        for (const chunk of chunks) {
          const req = store.put({
            chunkId: chunk.metadata.chunkId,
            metadata: chunk.metadata,
            binary: chunk.binary
          });

          req.onsuccess = () => {
            completed++;
            if (onProgress) onProgress(completed, chunks.length);
          };
        }
      });
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

        const clean = records.map((r) => {
          let buf = r.binary;
          if (buf instanceof Uint8Array) {
            buf = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
          } else if (buf instanceof ArrayBuffer) {
            buf = buf.slice(0);
          }
          return { chunkId: r.chunkId, metadata: r.metadata, binary: buf };
        });

        resolve(clean);
      };
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
        resolve({ chunkId: req.result.chunkId, metadata: req.result.metadata, binary: buf });
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
      checksum: c.metadata.checksum || 'N/A',
      layerRange: c.metadata.layerRange || [0, 0],
      nextChunkId: c.metadata.contextPointers?.next || null
    }));
  }

  async deleteModelChunks(modelName) {
    return this.withLock('db-write-lock', async () => {
      return new Promise((resolve, reject) => {
        const tx = this.db.transaction(['model_chunks', 'si_radix_manifests'], 'readwrite');
        const store = tx.objectStore('model_chunks');
        const manifestStore = tx.objectStore('si_radix_manifests');
        manifestStore.delete(modelName);

        const index = store.index('by_model');
        const req = index.openKeyCursor(IDBKeyRange.only(modelName));

        req.onsuccess = (e) => {
          const cursor = e.target.result;
          if (cursor) {
            store.delete(cursor.primaryKey);
            cursor.continue();
          } else {
            if (this.broadcast) {
              this.broadcast.postMessage({ type: 'MODEL_PURGED', modelName });
            }
            resolve(true);
          }
        };
        req.onerror = () => reject(req.error);
      });
    });
  }

  async logEvent(category, message, details = null) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('event_logs', 'readwrite');
      const store = tx.objectStore('event_logs');
      store.add({
        timestamp: Date.now(),
        category,
        message,
        details: details ? JSON.stringify(details) : null
      });
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => reject(tx.error);
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
      tx.objectStore('event_logs').clear();
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => reject(tx.error);
    });
  }

  async saveMessage(role, text) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('chat_history', 'readwrite');
      tx.objectStore('chat_history').add({ role, text, timestamp: Date.now() });
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => reject(tx.error);
    });
  }

  async getMessages() {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('chat_history', 'readonly');
      const req = tx.objectStore('chat_history').getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }

  async setSystemKey(key, value) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('system_state', 'readwrite');
      tx.objectStore('system_state').put({ key, value });
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => reject(tx.error);
    });
  }

  async getSystemKey(key) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('system_state', 'readonly');
      const req = tx.objectStore('system_state').get(key);
      req.onsuccess = () => resolve(req.result ? req.result.value : null);
      req.onerror = () => reject(req.error);
    });
  }

  async clearAll() {
    return this.withLock('db-write-lock', async () => {
      return new Promise((resolve, reject) => {
        const storeNames = ['chat_history', 'model_chunks', 'si_radix_manifests', 'system_state', 'event_logs'];
        const tx = this.db.transaction(storeNames, 'readwrite');
        storeNames.forEach((name) => tx.objectStore(name).clear());
        tx.oncomplete = () => {
          if (this.broadcast) this.broadcast.postMessage({ type: 'ALL_STORAGE_CLEARED' });
          resolve(true);
        };
        tx.onerror = () => reject(tx.error);
      });
    });
  }
}
