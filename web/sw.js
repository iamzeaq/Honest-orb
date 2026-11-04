/* Honest Orb service worker: offline shell for the installed app.
 * HTML/JS/CSS are network-first so returning visitors always get the
 * latest files; icons stay cache-first. Bump CACHE + ASSET_V together
 * with the ?v= strings in index.html on every frontend release. */
var CACHE = "honest-orb-v2";
var ASSET_V = "v=2";
var SHELL = [
  "/", "/index.html", "/style.css?" + ASSET_V, "/app.js?" + ASSET_V,
  "/orb.js?" + ASSET_V, "/manifest.webmanifest", "/icon.svg",
  "/icon-192.png", "/icon-512.png"
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
    var dest = e.request.destination;
  var networkFirst = (dest === "document" || dest === "script" ||
    dest === "style" || /\/$|\.html(\?|$)|\.js(\?|$)|\.css(\?|$)/.test(url.pathname + url.search));
  if (e.request.method !== "GET") return;
  if (!networkFirst) {
    // Icons and other static assets: cache-first, network fallback.
    e.respondWith(
      caches.match(e.request).then(function (hit) {
        return hit || fetch(e.request);
      })
    );
    return;
  }
  e.respondWith(
    fetch(e.request).then(function (res) {
      if (res.ok) {
        var copy = res.clone();
        caches.open(CACHE).then(function (c) { c.put(e.request, copy); });
      }
      return res;
    }).catch(function () {
      return caches.match(e.request).then(function (hit) {
        return hit || caches.match("/index.html");
      });
    })
  );
});
