// Service Worker pour Maintso Vola PWA
const CACHE_NAME = 'maintso-vola-v1';
const urlsToCache = [
  '/',
  '/static/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png'
];

// Installation du service worker
self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(function(cache) {
        return cache.addAll(urlsToCache);
      })
  );
});

// Activation du service worker
self.addEventListener('activate', function(event) {
  event.waitUntil(
    caches.keys().then(function(cacheNames) {
      return Promise.all(
        cacheNames.map(function(cacheName) {
          if (cacheName !== CACHE_NAME) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

// Interception des requêtes
self.addEventListener('fetch', function(event) {
  event.respondWith(
    caches.match(event.request)
      .then(function(response) {
        // Retourner le cache si disponible, sinon fetch
        if (response) {
          return response;
        }
        return fetch(event.request);
      }
    )
  );
});

// Notification push
self.addEventListener('push', function(event) {
  const options = {
    body: event.data ? event.data.text() : 'Nouvelle notification Maintso Vola',
    icon: '/static/icon-192.png',
    badge: '/static/icon-192.png'
  };

  event.waitUntil(
    self.registration.showNotification('Maintso Vola', options)
  );
});