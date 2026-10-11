export class StorageService {
  constructor(dbName = 'Custom_JSONLLM_Store', version = 1) {
    this.dbName = dbName;
    this.version = version;
    this.db = null;
    this.broadcast = typeof BroadcastChannel !== 'undefined'
      ? new BroadcastChannel('jsonllm-storage-sync')
      : null;
  }

  async init() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(this.dbName, this.version);

      req.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains('models')) {
          db.createObjectStore('models', { keyPath: 'modelName' });
        }
        if (!db.objectStoreNames.contains('model_segments')) {
          const segStore = db.createObjectStore('model_segments', { keyPath: 'segmentId' });
          segStore.createIndex('by_model', 'modelName', { unique: false });
        }
        if (!db.objectStoreNames.contains('rag_documents')) {
          db.createObjectStore('rag_documents', { keyPath: 'docId' });
        }
        if (!db.objectStoreNames.contains('event_logs')) {
          db.createObjectStore('event_logs', { keyPath: 'id', autoIncrement: true });
        }
      };

      req.onsuccess = async () => {
        this.db = req.result;
        if (navigator.storage && navigator.storage.persist) {
          await navigator.storage.persist();
        }
        resolve(this);
      };
      req.onerror = () => reject(req.error);
    });
  }

  async putModel(modelName, jsonString) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('models', 'readwrite');
      const store = tx.objectStore('models');
      const req = store.put({ modelName, jsonString, timestamp: Date.now() });
      req.onsuccess = () => resolve(true);
      req.onerror = () => reject(req.error);
    });
  }

  async getModel(modelName) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('models', 'readonly');
      const store = tx.objectStore('models');
      const req = store.get(modelName);
      req.onsuccess = () => resolve(req.result ? req.result.jsonString : null);
      req.onerror = () => reject(req.error);
    });
  }

  async batchSaveSegments(segments) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('model_segments', 'readwrite');
      const store = tx.objectStore('model_segments');
      for (const seg of segments) {
        store.put(seg);
      }
      tx.oncomplete = () => {
        if (this.broadcast) this.broadcast.postMessage({ type: 'SEGMENTS_UPDATED' });
        resolve(true);
      };
      tx.onerror = () => reject(tx.error);
    });
  }

  async getSegmentsByModel(modelName) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('model_segments', 'readonly');
      const store = tx.objectStore('model_segments');
      const index = store.index('by_model');
      const req = index.getAll(modelName);
      req.onsuccess = () => resolve((req.result || []).sort((a, b) => a.index - b.index));
      req.onerror = () => reject(req.error);
    });
  }

  async getAllSegments() {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('model_segments', 'readonly');
      const store = tx.objectStore('model_segments');
      const req = store.getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }

  async putRAGDoc(record) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('rag_documents', 'readwrite');
      const store = tx.objectStore('rag_documents');
      const req = store.put(record);
      req.onsuccess = () => resolve(true);
      req.onerror = () => reject(req.error);
    });
  }

  async getAllRAGDocs() {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('rag_documents', 'readonly');
      const store = tx.objectStore('rag_documents');
      const req = store.getAll();
      req.onsuccess = () => resolve(req.result || []);
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

  async getRecentLogs(limit = 60) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('event_logs', 'readonly');
      const store = tx.objectStore('event_logs');
      const req = store.getAll();
      req.onsuccess = () => resolve((req.result || []).slice(-limit));
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
}
