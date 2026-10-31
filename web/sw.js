/* Honest Orb service worker: offline shell for the installed app. */
var CACHE = "honest-orb-v1";
var SHELL = [
  "/", "/index.html", "/style.css", "/app.js", "/orb.js",
  "/manifest.webmanifest", "/icon.svg", "/icon-192.png", "/icon-512.png"
];

self.addEventListener("install", function (e) {
  e.waitUntil(
    caches.open(CACHE).then(function (c) { return c.addAll(SHELL); })
      .then(function () { return self.skipWaiting(); })
      .catch(function () {})
  );
});

self.addEventListener("activate", function (e) {
  e.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.map(function (k) {
        return k === CACHE ? null : caches.delete(k);
      }));
    }).then(function () { return self.clients.claim(); })
  );
});

self.addEventListener("fetch", function (e) {
  var url = new URL(e.request.url);
  // Never cache API answers or generated audio.
  if (url.pathname.indexOf("/api/") === 0 ||
      url.pathname.indexOf("/audio/") === 0) return;
  e.respondWith(
    caches.match(e.request).then(function (hit) {
      return hit || fetch(e.request).then(function (res) {
        if (e.request.method === "GET" && res.ok) {
          var copy = res.clone();
          caches.open(CACHE).then(function (c) { c.put(e.request, copy); });
        }
        return res;
      }).catch(function () {
        return caches.match("/index.html");
      });
    })
  );
});
