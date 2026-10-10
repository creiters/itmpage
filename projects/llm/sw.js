const CACHE_NAME = 'shared-llm-v4';
const OFFLINE_RESOURCES = [
  './',
  './index.html',
  './style.css',
  './manifest.json',
  './app.js',
  './storage.js',
  './chunker.js',
  './downloader.js',
  './mesh.js',
  './wasm-engine.js',
  './gguf-parser.js',
  './tokenizer.js',
  './llm-worker.js'
];

self.addEventListener('install', (event) => {
  // Pre-cache fresh application assets into the new cache bucket
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(OFFLINE_RESOURCES))
  );
  // Do NOT skipWaiting automatically: wait until user prompts to prevent memory corruption during active generation
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((response) => {
        if (!response || response.status !== 200 || response.type !== 'basic') {
          return response;
        }
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        return response;
      }).catch(() => caches.match('./index.html'));
    })
  );
});

// Listener for explicit update activation from the client
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
