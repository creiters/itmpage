const CACHE_NAME = 'meshi-core-v2026.3';
const STATIC_ASSETS = [
  './',
  './index.html',
  './manifest.webmanifest',
  './css/meshi.css',
  './data/meta.json',
  './js/crypto.js',
  './js/db.js',
  './js/signaling.js',
  './js/mesh_socket.js',
  './js/webmcp.js',
  './js/app.js'
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(STATIC_ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))))
  );
  self.clients.claim();
});

self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request).then(res => {
      if (!res || res.status !== 200 || res.type !== 'basic') return res;
      const copy = res.clone();
      caches.open(CACHE_NAME).then(c => c.put(e.request, copy));
      return res;
    }).catch(() => caches.match('./index.html')))
  );
});

// Background Sync API: triggered when device regains connectivity
self.addEventListener('sync', (event) => {
  if (event.tag === 'meshi-reconnect-sync') {
    event.waitUntil(notifyClientsOfNetworkRecovery());
  }
});

// Periodic Background Sync (if supported by modern browsers)
self.addEventListener('periodicsync', (event) => {
  if (event.tag === 'meshi-keepalive') {
    event.waitUntil(notifyClientsOfNetworkRecovery());
  }
});

async function notifyClientsOfNetworkRecovery() {
  const clients = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
  for (const client of clients) {
    client.postMessage({ type: 'MESHI_BACKGROUND_NET_STABLE', timestamp: Date.now() });
  }
}
