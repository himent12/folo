// The HTML entries the scripts build (shoot --prod / --qa, parity, perf): index.html (the Svelte app) and next.html
// (its port) once that file exists. Set by a plugin, since an inline `build.rollupOptions.input` is merged with
// vite.config.ts's, which may name next.html before it is there.

import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

/**
 * @param {string} root the web/ folder
 * @param {{ next?: boolean }} [opts] next: false builds index.html alone (--qa checks only the Svelte app)
 * @returns {Record<string, string>}
 */
export function appEntries(root, { next = true } = {}) {
  /** @type {Record<string, string>} */
  const input = { index: resolve(root, 'index.html') };
  if (next && existsSync(resolve(root, 'next.html'))) input.next = resolve(root, 'next.html');
  return input;
}

/**
 * Builds the apps into `outDir` (no source maps). When next.html is there but does not build (a port in progress),
 * builds index.html alone and returns why next did not build, so the legacy app can still be measured or shot.
 * @param {string} root
 * @param {string} outDir
 * @param {{ next?: boolean }} [opts]
 * @returns {Promise<{ input: Record<string, string>, nextError: string | null }>}
 */
export async function buildApps(root, outDir, opts) {
  const { build } = await import('vite');
  const once = (/** @type {Record<string, string>} */ input) =>
    build({ root, logLevel: 'silent', plugins: [entriesPlugin(input)], build: { outDir, emptyOutDir: true, sourcemap: false, reportCompressedSize: false } });
  const input = appEntries(root, opts);
  if (!input.next) {
    await once(input);
    return { input, nextError: null };
  }
  try {
    await once(input);
    return { input, nextError: null };
  } catch (e) {
    const msg = (e instanceof Error ? e.message : String(e)).replace(/\u001b\[\d+m/g, '');
    const first = msg.split('\n').map((l) => l.trim()).filter(Boolean).slice(0, 4).join(' ');
    console.warn(`next.html does not build (${first.slice(0, 300)}); building index.html alone`);
    const legacy = { index: input.index };
    await once(legacy);
    return { input: legacy, nextError: `next.html does not build: ${first.slice(0, 300)}` };
  }
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
