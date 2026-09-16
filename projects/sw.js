const CACHE_NAME = 'meshi-core-v2026.1';
const STATIC_ASSETS = [
  './',
  './index.html',
  './manifest.webmanifest',
  './css/meshi.css',
  './data/meta.json',
  './data/initial_nodes.json',
  './js/db.js',
  './js/signaling.js',
  './js/mesh_socket.js',
  './js/webmcp.js',
  './js/app.js'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  event.respondWith(
    caches.match(event.request).then((cached) => {
      return cached || fetch(event.request).then((networkRes) => {
        if (!networkRes || networkRes.status !== 200 || networkRes.type !== 'basic') {
          return networkRes;
        }
        const clone = networkRes.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        return networkRes;
      }).catch(() => caches.match('./index.html'));
    })
  );
});
