import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';
import { fw } from './framework/compiler/index.ts';

const repoRoot = fileURLToPath(new URL('..', import.meta.url));
const api = process.env.FOLO_API ?? 'http://127.0.0.1:7878';
const runtime = fileURLToPath(new URL('./framework/runtime', import.meta.url));

export default defineConfig({
  // fw compiles the app's .tsx (JSX + scoped css``) and serves the lucide icons (`$icons/<name>`).
  plugins: [fw()],
  resolve: {
    alias: [
      { find: '$lib', replacement: fileURLToPath(new URL('./src/lib', import.meta.url)) },
      { find: /^fw$/, replacement: `${runtime}/index.ts` },
      { find: /^fw\/(.*)$/, replacement: `${runtime}/$1.ts` },
    ],
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
  },
  // An unhandled error (a deferred callback outliving its component) fails the run, never only a warning.
  test: {
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx', 'framework/**/*.test.ts', 'framework/**/*.test.tsx', 'scripts/**/*.test.mjs'],
    environment: 'node',
    dangerouslyIgnoreUnhandledErrors: false,
  },
});
