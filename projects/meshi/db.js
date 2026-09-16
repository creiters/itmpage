export class LocalDB {
    constructor(dbName = 'SSHAnetCoreDB', version = 1) {
        this.dbName = dbName;
        this.version = version;
        this.db = null;
    }

    async init() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.version);
            request.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains('telemetry_logs')) {
                    db.createObjectStore('telemetry_logs', { keyPath: 'id', autoIncrement: true });
                }
            };
            request.onsuccess = (e) => {
                this.db = e.target.result;
                resolve(this.db);
            };
            request.onerror = (e) => reject(e.target.error);
        });
    }

    async addRecord(dataType, content) {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction(['telemetry_logs'], 'readwrite');
            const store = transaction.objectStore(['telemetry_logs']);
            const record = {
                timestamp: new Date().toISOString(),
                type: dataType,
                payload: content
            };
            const request = store.add(record);
            request.onsuccess = () => resolve(true);
            request.onerror = (e) => reject(e.target.error);
        });
    }

    async getAllRecords() {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction(['telemetry_logs'], 'readonly');
            const store = transaction.objectStore(['telemetry_logs']);
            const request = store.getAll();
            request.onsuccess = () => resolve(request.result);
            request.onerror = (e) => reject(e.target.error);
        });
    }
}