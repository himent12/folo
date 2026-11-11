import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { fileURLToPath } from 'node:url';
import { fw } from './framework/compiler/index.ts';

const repoRoot = fileURLToPath(new URL('..', import.meta.url));
const api = process.env.FOLO_API ?? 'http://127.0.0.1:7878';
const runtime = fileURLToPath(new URL('./framework/runtime', import.meta.url));

export default defineConfig({
  // fw compiles the new app's .tsx (JSX + scoped css``); svelte keeps compiling the Svelte app beside it.
  plugins: [fw(), svelte()],
  resolve: {
    alias: [
      { find: '$lib', replacement: fileURLToPath(new URL('./src/lib', import.meta.url)) },
      { find: /^fw$/, replacement: `${runtime}/index.ts` },
      { find: /^fw\/(.*)$/, replacement: `${runtime}/$1.ts` },
    ],
    // Tests run Svelte's client runtime so runes ($effect, $effect.root) behave as in the browser.
    ...(process.env.VITEST ? { conditions: ['browser'] } : {}),
  },
  server: {
    port: 5173,
    // design/ (fonts, tokens) lives next to web/.
    fs: { allow: [repoRoot] },
    proxy: { '/api': { target: api, changeOrigin: true } },
  },
  build: {
    target: 'es2022',
    sourcemap: true,
    // index.html: the Svelte app; next.html: the fw app it is being ported to.
    rollupOptions: { input: { index: fileURLToPath(new URL('./index.html', import.meta.url)), next: fileURLToPath(new URL('./next.html', import.meta.url)) } },
  },
  // An unhandled error (a deferred callback outliving its component) fails the run, never only a warning.
  test: {
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx', 'framework/**/*.test.ts', 'framework/**/*.test.tsx', 'scripts/**/*.test.mjs'],
    environment: 'node',
    dangerouslyIgnoreUnhandledErrors: false,
  },
});
