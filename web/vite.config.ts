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
    // Three chunks per screen, not sixty: the entry (the shell and what it imports), `shared` (every module two or
    // more chunks use: ui parts, charts, d3, the API's types and rules) and the screen's own. Left to itself the
    // bundler makes a chunk of every set of modules a distinct set of screens shares (115 files, many under 1 KB), and
    // a cold Overview fetched 41 JS and 19 CSS files: over HTTP/1.1 (folo serve) a browser opens 6 connections to a
    // host, so on a remote server those requests queued in waves of one round trip each. A first load fetches some
    // code its screen does not use (the Overview: 15 KB more JS, brotli), and every later screen only its own chunk
    // (README, Routing).
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            // What index.html's script imports statically stays in the entry, so the shell paints as soon as it has
            // loaded, never waiting on `shared`.
            { name: 'index', tags: ['$initial'], priority: 2 },
            { name: 'shared', minShareCount: 2, priority: 1 },
          ],
        },
      },
    },
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
