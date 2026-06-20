import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy /api to the Crucible FastAPI backend (default port 8400).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5273,
    proxy: {
      "/api": { target: "http://127.0.0.1:8400", changeOrigin: true },
    },
  },
});
