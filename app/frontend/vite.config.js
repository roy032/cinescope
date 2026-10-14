import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the app on :5173 and forwards /api to the FastAPI back end on :8000.
// `npm run build` writes dist/, which the back end serves itself at /.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  build: { outDir: "dist", chunkSizeWarningLimit: 600 },
});
