import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { fileURLToPath } from 'node:url';

const repoRoot = fileURLToPath(new URL('..', import.meta.url));
const api = process.env.FOLO_API ?? 'http://127.0.0.1:7878';

export default defineConfig({
  plugins: [svelte()],
  resolve: {
    alias: { $lib: fileURLToPath(new URL('./src/lib', import.meta.url)) },
    // Tests run Svelte's client runtime so runes ($effect, $effect.root) behave as in the browser.
    ...(process.env.VITEST ? { conditions: ['browser'] } : {}),
  },
  server: {
    port: 5173,
    // design/ (fonts, tokens) lives next to web/.
    fs: { allow: [repoRoot] },
    proxy: { '/api': { target: api, changeOrigin: true } },
  },
  build: { target: 'es2022', sourcemap: true },
  test: { include: ['src/**/*.test.ts'], environment: 'node' },
});
