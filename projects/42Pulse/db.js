/**
 * db.js - Native IndexedDB & WebCrypto Storage
 * Standard: ES8
 */
const DB_NAME = '42PulseOfflineDB';
const DB_VERSION = 1;

// 1. IndexedDB Helper
const openDB = () => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains('secure_auth')) {
        db.createObjectStore('secure_auth', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('campus_stats')) {
        db.createObjectStore('campus_stats', { keyPath: 'key' });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

const idbPut = async (storeName, data) => {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readwrite');
    const store = tx.objectStore(storeName);
    const req = store.put(data);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
};

const idbGet = async (storeName, key) => {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readonly');
    const store = tx.objectStore(storeName);
    const req = store.get(key);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
};

// 2. Native WebCrypto AES-GCM & PBKDF2 Operations
const CryptoEngine = {
  async deriveKey(pin, salt) {
    const enc = new TextEncoder();
    const keyMaterial = await crypto.subtle.importKey(
      'raw',
      enc.encode(pin),
      { name: 'PBKDF2' },
      false,
      ['deriveKey']
    );

    return crypto.subtle.deriveKey(
      {
        name: 'PBKDF2',
        salt: salt,
        iterations: 100000,
        hash: 'SHA-256'
      },
      keyMaterial,
      { name: 'AES-GCM', length: 256 },
      false,
      ['encrypt', 'decrypt']
    );
  },

  async encrypt(text, pin) {
    const enc = new TextEncoder();
    const salt = crypto.getRandomValues(new Uint8Array(16));
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const key = await this.deriveKey(pin, salt);

    const ciphertext = await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv: iv },
      key,
      enc.encode(text)
    );

    return {
      ciphertext: Array.from(new Uint8Array(ciphertext)),
      iv: Array.from(iv),
      salt: Array.from(salt)
    };
  },

  async decrypt(encryptedObj, pin) {
    const dec = new TextDecoder();
    const salt = new Uint8Array(encryptedObj.salt);
    const iv = new Uint8Array(encryptedObj.iv);
    const data = new Uint8Array(encryptedObj.ciphertext);
    const key = await this.deriveKey(pin, salt);

    const decrypted = await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv: iv },
      key,
      data
    );

    return dec.decode(decrypted);
  }
};
