import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";

// Crucible frontend. PWA so it installs on Android / iOS / desktop as a thin client
// pointing at a local or remote Crucible node. /api and /v1 are never cached.
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["icon.svg"],
      manifest: {
        name: "Crucible — model forge",
        short_name: "Crucible",
        description: "Local LLM censorship lab + agentic harness",
        theme_color: "#ff6a1a",
        background_color: "#08090b",
        display: "standalone",
        start_url: "/",
        icons: [
          { src: "icon.svg", sizes: "any", type: "image/svg+xml", purpose: "any" },
          { src: "icon.svg", sizes: "any", type: "image/svg+xml", purpose: "maskable" },
        ],
      },
      workbox: {
        navigateFallbackDenylist: [/^\/api/, /^\/v1/],
        runtimeCaching: [
          { urlPattern: /^https?:\/\/.*\/(api|v1)\//, handler: "NetworkOnly" },
        ],
      },
    }),
  ],
  server: {
    port: 5273,
    proxy: { "/api": { target: "http://127.0.0.1:8400", changeOrigin: true } },
  },
});
