const CACHE_NAME = 'intra-personal-v1';

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) =>
      cache.addAll(['./', './index.html', './auth.js', './42.js', './app.js', './manifest.json'])
    )
  );
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (e) => {
  // Let 42 OAuth and API endpoints go straight to the network
  if (e.request.url.includes('api.intra.42.fr')) {
    e.respondWith(fetch(e.request));
    return;
  }
  // Serve app shell offline
  e.respondWith(
    caches.match(e.request).then((res) => res || fetch(e.request))
  );
});
