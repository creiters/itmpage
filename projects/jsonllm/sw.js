const CACHE_NAME = 'vimtollm-app-v3';
const ASSETS = [
  './',
  './index.html',
  './style.css',
  './manifest.json',
  './webmcp.json',
  './storage.js',
  './jsonllm-codec.js',
  './rag-store.js',
  './nano-engine.js',
  './webmcp-server.js',
  './mesh.js',
  './llm-worker.js',
  './webmcp-worker.js',
  './app.js'
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS)));
  self.skipWaiting();
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
    caches.match(e.request).then((cached) => cached || fetch(e.request).catch(() => caches.match('./index.html')))
  );
});

self.addEventListener('message', (e) => {
  if (e.data && e.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
