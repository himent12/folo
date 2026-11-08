// Screenshots every screen and important state into web/shots/<round>/.
//   npm run shoot                 -> shots/<timestamp>/
//   npm run shoot -- r3           -> shots/r3/
//   npm run shoot -- r3 --only runs   (scene names containing "runs")
// Uses the installed Chrome (playwright-core, channel "chrome"); no browser download.
import { createServer } from 'vite';
import { chromium } from 'playwright-core';
import { mkdirSync, rmSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { scenes } from './scenes.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const onlyIdx = args.indexOf('--only');
const only = onlyIdx >= 0 ? args[onlyIdx + 1] : undefined;
const round = args.find((a, i) => !a.startsWith('--') && (onlyIdx < 0 || i !== onlyIdx + 1)) ?? new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
const outDir = resolve(root, 'shots', round);

const VIEWPORTS = {
  desktop: { width: 1440, height: 900 },
  laptop: { width: 1024, height: 768 },
  mobile: { width: 390, height: 844 },
};
// Fixed wall clock so relative times ("3 min ago") and the mock's live run are identical every round.
const FIXED_NOW = new Date('2026-11-08T14:30:00Z');

const server = await createServer({ root, logLevel: 'error', server: { host: '127.0.0.1', port: 5199, strictPort: false } });
await server.listen();
const base = (server.resolvedUrls?.local[0] ?? 'http://127.0.0.1:5199/').replace(/\/$/, '');

if (!only) rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

/** Waits (in real time; the page clock is frozen) until every image has decoded, then a beat for layout. */
async function settle(page) {
  for (let i = 0; i < 40; i++) {
    const pending = await page.evaluate(() => [...document.images].filter((im) => !im.complete).length);
    if (!pending) break;
    await page.waitForTimeout(100);
  }
  await page.clock.runFor(50);
  await page.waitForTimeout(150);
}

const browser = await chromium.launch({ channel: 'chrome', headless: true });
let count = 0;
const failures = [];
try {
  for (const scene of scenes) {
    for (const theme of scene.themes ?? ['dark', 'light']) {
      for (const vp of scene.viewports ?? ['desktop', 'mobile']) {
        const name = `${scene.name}--${theme}--${vp}`;
        if (only && !name.includes(only)) continue;
        const context = await browser.newContext({
          viewport: VIEWPORTS[vp],
          deviceScaleFactor: vp === 'mobile' ? 2 : 1,
          colorScheme: theme === 'light' ? 'light' : 'dark',
          reducedMotion: 'reduce',
        });
        const page = await context.newPage();
        await page.clock.install({ time: FIXED_NOW });
        await page.addInitScript((t) => {
          localStorage.setItem('folo.theme', t);
          localStorage.setItem('folo.shoot', '1');
        }, theme);
        page.on('pageerror', (e) => failures.push(`${name}: ${e.message}`));
        try {
          await page.goto(`${base}/?mock=1#${scene.path}`, { waitUntil: 'networkidle' });
          await page.evaluate(() => document.fonts.ready);
          // Let the mock's live run advance a little, deterministically.
          await page.clock.runFor(scene.advance ?? 2000);
          if (scene.setup) await scene.setup(page, { vp, theme });
          await settle(page);
          await page.screenshot({ path: resolve(outDir, `${name}.png`), fullPage: scene.fullPage ?? vp !== 'mobile' });
          count++;
        } catch (e) {
          failures.push(`${name}: ${e instanceof Error ? e.message : e}`);
        } finally {
          await context.close();
        }
      }
    }
  }
} finally {
  await browser.close();
  await server.close();
}

console.log(`${count} shots -> ${outDir}`);
if (failures.length) {
  console.error(`${failures.length} problem(s):\n  ${failures.join('\n  ')}`);
  process.exitCode = 1;
}
