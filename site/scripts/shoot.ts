// Screenshots of the built site (run `npm run build` first) into site/shots/: pages x viewports x themes.
//   npm run shoot                     the landing page and one reference page
//   npm run shoot -- cli/ api/ --full  other pages, full-page captures
// Uses the installed Chrome through playwright-core (no browser download), like web/scripts/shoot.mjs.
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium } from 'playwright-core';
import { preview } from 'vite';
import { DIST_DIR, SITE_DIR } from './config.ts';

const args = process.argv.slice(2);
const full = args.includes('--full');
const pages = args.filter((a) => !a.startsWith('--'));
const targets = pages.length ? pages : ['', 'cli/'];
const VIEWPORTS = { desktop: { width: 1440, height: 900 }, phone: { width: 390, height: 844 } };
const outDir = resolve(SITE_DIR, 'shots');
mkdirSync(outDir, { recursive: true });

const server = await preview({ root: DIST_DIR, build: { outDir: DIST_DIR }, preview: { port: 4179, strictPort: false }, configFile: false, logLevel: 'error' });
const base = (server.resolvedUrls?.local[0] ?? 'http://localhost:4179/').replace(/\/?$/, '/');
const browser = await chromium.launch({ channel: 'chrome' });
try {
  for (const [vp, size] of Object.entries(VIEWPORTS)) {
    for (const theme of ['dark', 'light'] as const) {
      const ctx = await browser.newContext({ viewport: size, deviceScaleFactor: vp === 'phone' ? 2 : 1, colorScheme: theme });
      const page = await ctx.newPage();
      for (const t of targets) {
        await page.goto(base + t, { waitUntil: 'networkidle' });
        await page.evaluate(() => document.fonts.ready);
        if (full) {
          // Lazy images load only near the viewport: scroll through the page first.
          await page.evaluate(async () => {
            for (let y = 0; y < document.documentElement.scrollHeight; y += innerHeight) {
              scrollTo(0, y);
              await new Promise((r) => setTimeout(r, 60));
            }
            scrollTo(0, 0);
          });
          await page.waitForLoadState('networkidle');
        }
        const name = `${(t.replace(/\/$/, '') || 'landing').replace(/\//g, '-')}-${vp}-${theme}.png`;
        await page.screenshot({ path: resolve(outDir, name), fullPage: full });
        console.log(name);
      }
      // Interactive states on the last page: the search dialog (opened with "/"), then the navigation drawer.
      if (!full) {
        await page.keyboard.press('/');
        await page.keyboard.type('mcp resources');
        await page.waitForSelector('.search-result');
        await page.keyboard.press('ArrowDown');
        await page.screenshot({ path: resolve(outDir, `search-${vp}-${theme}.png`) });
        await page.keyboard.press('Escape');
        if (vp === 'phone') {
          await page.click('.nav-toggle');
          await page.waitForTimeout(300);
          await page.screenshot({ path: resolve(outDir, `nav-${vp}-${theme}.png`) });
        }
      }
      await ctx.close();
    }
  }
} finally {
  await browser.close();
  await new Promise<void>((r) => server.httpServer.close(() => r()));
}
