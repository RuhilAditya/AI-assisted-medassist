import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the FastAPI backend runs on :8000; Vite proxies API calls to it.
// In production FastAPI serves the built app itself (see api/main.py), so no proxy is needed.
const API = "http://127.0.0.1:8000";
const apiPaths = ["/cases", "/health", "/status"];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(apiPaths.map((p) => [p, { target: API, changeOrigin: true }])),
  },
});
