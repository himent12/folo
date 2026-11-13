import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';
import { fw } from './framework/compiler/index.ts';
import { builtCssCheck } from './scripts/builtCss.mjs';

const repoRoot = fileURLToPath(new URL('..', import.meta.url));
const api = process.env.FOLO_API ?? 'http://127.0.0.1:7878';
const runtime = fileURLToPath(new URL('./framework/runtime', import.meta.url));

export default defineConfig({
  // fw compiles the app's .tsx (JSX + scoped css``) and serves the lucide icons (`$icons/<name>`).
  // builtCssCheck fails a build whose CSS lost an unprefixed declaration (scripts/builtCss.mjs).
  plugins: [fw(), builtCssCheck()],
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
  // In CI a test gets 20 s instead of vitest's 5 s: a runner shares its machine with other jobs, and a test of
  // routes/run/Logs.test.tsx (four tests, about 3 s together on a quiet runner) took over 5 s beside two Rust
  // builds. A test that hangs still fails; a slow one is a matter for review (docs/CI.md, Keeping CI fast), not
  // for the timeout.
  test: {
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx', 'framework/**/*.test.ts', 'framework/**/*.test.tsx', 'scripts/**/*.test.mjs'],
    environment: 'node',
    dangerouslyIgnoreUnhandledErrors: false,
    testTimeout: process.env.CI ? 20_000 : 5_000,
  },
});
