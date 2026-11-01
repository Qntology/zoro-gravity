import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Tauri 는 고정 포트를 씁니다.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  base: "./",
  build: { outDir: "dist", emptyOutDir: true },
  server: { port: 1420, strictPort: true },
});
