export class StorageService {
  constructor(dbName = 'SharedLLM_Store', version = 1) {
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
          chunkStore.createIndex('by_hash', 'metadata.hash', { unique: true });
          chunkStore.createIndex('by_prev', 'metadata.prevHash', { unique: false });
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
      req.onsuccess = () => resolve(req.result);
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
        resolve(records);
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
}
