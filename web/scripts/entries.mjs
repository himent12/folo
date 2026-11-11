// The production build the scripts use (shoot --prod / --qa, parity, perf): index.html, the app, built into a folder
// of the script's own (never web/dist), whatever vite.config.ts says.

import { resolve } from 'node:path';

/**
 * Builds the app into `outDir` (no source maps).
 * @param {string} root the web/ folder
 * @param {string} outDir
 * @returns {Promise<{ input: Record<string, string> }>}
 */
export async function buildApps(root, outDir) {
  const { build } = await import('vite');
  const input = { index: resolve(root, 'index.html') };
  await build({ root, logLevel: 'silent', plugins: [entriesPlugin(input)], build: { outDir, emptyOutDir: true, sourcemap: false, reportCompressedSize: false } });
  return { input };
}

/**
 * A plugin that makes `input` the build's entries, whatever the config file says.
 * @param {Record<string, string>} input
 * @returns {import('vite').Plugin}
 */
export function entriesPlugin(input) {
  return {
    name: 'folo-script-entries',
    enforce: 'post',
    config(c) {
      c.build ??= {};
      c.build.rollupOptions ??= {};
      c.build.rollupOptions.input = input;
    },
  };
}
