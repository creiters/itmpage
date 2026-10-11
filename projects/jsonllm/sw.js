const CACHE_NAME = 'jsonllm-suite-v1';
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
  './rag-store.js',
  './domain-specs.js',
  './mesh.js',
  './nano-engine.js',
  './webmcp-server.js',
  './chunk-worker.js',
  './llm-worker.js',
  './webmcp-worker.js'
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS)));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  e.respondWith(
    caches.match(e.request).then((res) => {
      if (res) return res;
      return fetch(e.request).then((networkRes) => {
        if (!networkRes || networkRes.status !== 200 || networkRes.type !== 'basic') {
          return networkRes;
        }
        const clone = networkRes.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(e.request, clone));
        return networkRes;
      }).catch(() => caches.match('./index.html'));
    })
  );
});

self.addEventListener('message', (e) => {
  if (e.data && e.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
