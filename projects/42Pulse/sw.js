/**
 * sw.js - Cache App Shell for Complete Offline Execution
 */
const CACHE_NAME = '42pulse-offline-v1';
const STATIC_FILES = [
  './',
  './index.html',
  './db.js',
  './42pulse.js',
  './manifest.json'
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_FILES)));
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (e) => {
  // Never cache 42 API network requests inside the Service Worker cache;
  // let 42pulse.js handle fallback through IndexedDB.
  if (e.request.url.includes('api.intra.42.fr')) {
    e.respondWith(fetch(e.request).catch(() => new Response(JSON.stringify({ offline: true }))));
    return;
  }

  // App Shell Cache First
  e.respondWith(
    caches.match(e.request).then((res) => res || fetch(e.request))
  );
});
