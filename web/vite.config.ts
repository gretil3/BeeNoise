import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Cross-origin isolation lets ONNX Runtime use threads (SharedArrayBuffer).
// The same headers are set for production in vercel.json.
const headers = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };

export default defineConfig({
  plugins: [react()],
  publicDir: "../assets", // demo.mp4
  server: { headers },
  preview: { headers },
  worker: { format: "es" },
  // Pre-bundling moves onnxruntime-web away from its .wasm file.
  optimizeDeps: { exclude: ["onnxruntime-web", "@huggingface/transformers"] },
});
