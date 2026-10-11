export class StorageService {
  constructor(dbName = 'Clean_JSONLLM_Store', version = 3) {
    this.dbName = dbName;
    this.version = version;
    this.db = null;
  }

  async init() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(this.dbName, this.version);

      req.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains('json_models')) {
          db.createObjectStore('json_models', { keyPath: 'modelName' });
        }
        if (!db.objectStoreNames.contains('dialogue_datasets')) {
          db.createObjectStore('dialogue_datasets', { keyPath: 'datasetName' });
        }
        if (!db.objectStoreNames.contains('rag_docs')) {
          db.createObjectStore('rag_docs', { keyPath: 'docId' });
        }
        if (!db.objectStoreNames.contains('chat_history')) {
          db.createObjectStore('chat_history', { keyPath: 'id', autoIncrement: true });
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
      const tx = this.db.transaction('json_models', 'readwrite');
      const store = tx.objectStore('json_models');
      const req = store.put({
        modelName,
        jsonString,
        bytes: jsonString.length,
        updated: Date.now()
      });
      req.onsuccess = () => resolve(true);
      req.onerror = () => reject(req.error);
    });
  }

  async getModel(modelName) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('json_models', 'readonly');
      const store = tx.objectStore('json_models');
      const req = store.get(modelName);
      req.onsuccess = () => resolve(req.result ? req.result.jsonString : null);
      req.onerror = () => reject(req.error);
    });
  }

  async getAllModels() {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('json_models', 'readonly');
      const store = tx.objectStore('json_models');
      const req = store.getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }

  async putDialogueDataset(datasetName, agentId, jsonPayload) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('dialogue_datasets', 'readwrite');
      const store = tx.objectStore('dialogue_datasets');
      const req = store.put({
        datasetName,
        agentId,
        jsonPayload,
        bytes: jsonPayload.length,
        updated: Date.now()
      });
      req.onsuccess = () => resolve(true);
      req.onerror = () => reject(req.error);
    });
  }

  async getDialogueDataset(datasetName) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('dialogue_datasets', 'readonly');
      const store = tx.objectStore('dialogue_datasets');
      const req = store.get(datasetName);
      req.onsuccess = () => resolve(req.result ? req.result.jsonPayload : null);
      req.onerror = () => reject(req.error);
    });
  }

  async getAllDialogueDatasets() {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('dialogue_datasets', 'readonly');
      const store = tx.objectStore('dialogue_datasets');
      const req = store.getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }

  async putRAGDoc(record) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('rag_docs', 'readwrite');
      const store = tx.objectStore('rag_docs');
      const req = store.put(record);
      req.onsuccess = () => resolve(true);
      req.onerror = () => reject(req.error);
    });
  }

  async getAllRAGDocs() {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('rag_docs', 'readonly');
      const store = tx.objectStore('rag_docs');
      const req = store.getAll();
      req.onsuccess = () => resolve(req.result || []);
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

  async clearAll() {
    return new Promise((resolve, reject) => {
      const stores = ['json_models', 'dialogue_datasets', 'rag_docs', 'chat_history', 'event_logs'];
      const tx = this.db.transaction(stores, 'readwrite');
      for (const s of stores) tx.objectStore(s).clear();
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => reject(tx.error);
    });
  }
}
