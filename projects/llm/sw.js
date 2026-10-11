const CACHE_NAME = 'shared-llm-v10';
const ASSETS = [
  './',
  './index.html',
  './style.css',
  './manifest.json',
  './webmcp.json',
  './app.js',
  './storage.js',
  './chunk-indexer.js',
  './jsonllm-codec.js',
  './downloader.js',
  './mesh.js',
  './tokenizer.js',
  './gguf-parser.js',
  './gpu-kernel.js',
  './wasm-engine.js',
  './nano-engine.js',
  './webmcp-server.js',
  './chunk-worker.js',
  './llm-worker.js',
  './webmcp-worker.js',
  './engine.wasm'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
  );
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

self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
