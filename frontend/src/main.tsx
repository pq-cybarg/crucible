import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import App from "./App";

// DEV self-heal: the production PWA registers a service worker that precaches the app shell + JS.
// If one is left registered from a prior `vite build`/`preview`, it keeps serving STALE assets on
// the dev origin ("pages not refreshing"). In dev, unregister any lingering SW and drop its caches,
// then reload once so the live dev server serves fresh modules. (Prod keeps the PWA untouched.)
if (import.meta.env.DEV && "serviceWorker" in navigator) {
  navigator.serviceWorker.getRegistrations().then((regs) => {
    if (regs.length === 0) return;
    Promise.all(regs.map((r) => r.unregister()))
      .then(() => (window.caches ? caches.keys().then((ks) => Promise.all(ks.map((k) => caches.delete(k)))) : null))
      .then(() => window.location.reload());
  });
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
