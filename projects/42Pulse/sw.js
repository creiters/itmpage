/**
 * sw.js - 100% Native JS Offline Cache
 */
const CACHE = 'peerfinder-v1';
const ASSETS = ['./index.html', './app.js', './manifest.json'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (e) => {
  if (e.request.url.includes('api.intra.42.fr') || e.request.url.includes('corsproxy.io')) {
    e.respondWith(fetch(e.request).catch(() => new Response(JSON.stringify([]))));
    return;
  }
  e.respondWith(caches.match(e.request).then((res) => res || fetch(e.request)));
});
