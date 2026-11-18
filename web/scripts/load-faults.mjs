// Load faults on the production build: one request of the app's own files fails once (a dropped connection), and
// Studio must recover without a reload, as it recovers from a chunk that failed to load (src/ScreenError.tsx, Retry).
//   node scripts/load-faults.mjs [--dist <built dir>] [--base <url>] [--only <case,...>]
// Without --dist or --base it builds the app into node_modules/.faults-dist (never web/dist) and serves it with
// `vite preview`; --dist serves a build made elsewhere (another version's, to compare), --base uses a running server.
// Each case opens a cold `/?mock=1#/` (Overview, demo data) in a fresh context with one file failing:
// - stylesheet: the first stylesheet a screen needs beyond the entry's (the shared one: every screen's ui parts), once;
// - stylesheet-retry: the same, every time until Retry is pressed: the screen must say it could not load, and Retry
//   must fetch the stylesheet again;
// - screen-css: the first screen's own stylesheet, once;
// - mock: the in-browser mock's module (`?mock=1` loads it as a chunk of its own), once.
// What must follow: Overview (after Retry where it says it could not load) with every stylesheet a clean load of it
// applies, applied, and "Demo data" in the footer; then Settings the same way (a later screen, so a stylesheet left
// failed under it shows); no uncaught error, no reload. Exits 1 on any failure (`npm run qa`, so CI: its first shard).
// Uses the installed Chrome (playwright-core, channel "chrome"); without one, Playwright's Chromium.
import { preview } from 'vite';
import { chromium } from 'playwright-core';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildApps } from './entries.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const opt = (/** @type {string} */ flag) => {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
};

/** The screens checked, in order: the first one cold, the next one reached by a link (hash change). */
const SCREENS = [
  { hash: '#/', title: 'Overview' },
  { hash: '#/settings', title: 'Settings' },
];

/** The first stylesheet a screen needs beyond the entry's (index.html links that one: its failure is the browser's). */
const sharedCss = (/** @type {string} */ url) =>
  /\/assets\/[^/]+\.css$/.test(url) && !/\/assets\/index-[^/]+\.css$/.test(url) && !/\/Overview-[^/]+\.css$/.test(url);

/**
 * The cases: `pick(url)` says whether a request is of the file that fails (the first such file; the same one after);
 * `untilRetry`: it fails every time until Retry is pressed, else once.
 * @type {{ name: string, pick: (url: string) => boolean, untilRetry?: boolean }[]}
 */
const CASES = [
  { name: 'stylesheet', pick: sharedCss },
  { name: 'stylesheet-retry', pick: sharedCss, untilRetry: true },
  { name: 'screen-css', pick: (url) => /\/assets\/Overview-[^/]+\.css$/.test(url) },
  { name: 'mock', pick: (url) => /\/assets\/mock-[^/]+\.js$/.test(url) },
];

async function launch() {
  const want = process.env.FOLO_SHOOT_BROWSER;
  if (want !== 'chromium') {
    try {
      return await chromium.launch({ channel: 'chrome', headless: true });
    } catch (e) {
      if (want === 'chrome') throw e;
    }
  }
  return chromium.launch({ headless: true });
}

/** @param {import('playwright-core').Page} page */
const state = (page) =>
  page.evaluate(() => {
    const links = /** @type {HTMLLinkElement[]} */ ([...document.querySelectorAll('link[rel=stylesheet]')]);
    // A link whose load failed still has a sheet in Chrome, an empty one whose rules cannot be read.
    const applied = (/** @type {HTMLLinkElement} */ l) => {
      try {
        return !!l.sheet && l.sheet.cssRules.length > 0;
      } catch {
        return false;
      }
    };
    const path = (/** @type {HTMLLinkElement} */ l) => new URL(l.href).pathname;
    return {
      title: document.querySelector('.page h1')?.textContent?.trim() ?? '',
      failed: document.body.innerText.includes('Could not load this screen'),
      conn: document.querySelector('.conn .label')?.textContent?.trim() ?? '',
      // Every stylesheet the page applies, by file name with its hash; and every one it links to but does not apply.
      sheets: [...new Set(links.filter(applied).map(path))].sort(),
      dead: links.filter((l) => !applied(l)).map(path),
    };
  });

/**
 * Waits (real time) until the screen titled `title` shows, loaded, with demo data; the last state seen either way.
 * @param {import('playwright-core').Page} page
 */
async function shown(page, title, ms = 15_000) {
  const end = Date.now() + ms;
  let s = await state(page);
  while (Date.now() < end) {
    s = await state(page);
    if (s.title === title && !s.failed && s.conn === 'Demo data') return { ok: true, s };
    await page.waitForTimeout(100);
  }
  return { ok: false, s };
}

// `npm run qa` runs this after the shoot's checks, on the build they made; CI splits the QA over runners
// (FOLO_QA_SHARD=i/n), and the first one runs this.
const shard = process.env.FOLO_QA_SHARD;
if (shard && !shard.startsWith('1/')) {
  console.log(`Load faults: run by shard 1, not ${shard}.`);
  process.exit(0);
}

const given = opt('--base');
const dist = opt('--dist');
/** @type {{ close(): Promise<void> }} */
let server = { close: async () => {} };
let base = given?.replace(/\/$/, '');
if (!base) {
  const outDir = dist ? resolve(dist) : resolve(root, 'node_modules/.faults-dist');
  if (!dist) await buildApps(root, outDir);
  const p = await preview({ root, logLevel: 'error', build: { outDir }, preview: { host: '127.0.0.1', port: 5197, strictPort: false } });
  server = p;
  base = (p.resolvedUrls?.local[0] ?? 'http://127.0.0.1:5197/').replace(/\/$/, '');
}

const browser = await launch();
const only = opt('--only')?.split(',');
/** @type {string[]} */
const problems = [];
try {
  // What a clean load of each screen applies: the reference for each case.
  /** @type {Record<string, string[]>} */
  const clean = {};
  {
    const ctx = await browser.newContext();
    const page = await ctx.newPage();
    await page.goto(`${base}/?mock=1${SCREENS[0].hash}`);
    for (const [i, sc] of SCREENS.entries()) {
      if (i > 0) await page.evaluate((h) => (location.hash = h), sc.hash);
      const r = await shown(page, sc.title);
      if (!r.ok) throw new Error(`a clean load of ${sc.hash} never showed: ${JSON.stringify(r.s)}`);
      clean[sc.hash] = r.s.sheets;
    }
    await ctx.close();
  }

  for (const c of CASES) {
    if (only && !only.includes(c.name)) continue;
    const ctx = await browser.newContext();
    const page = await ctx.newPage();
    /** @type {string[]} */
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    if (args.includes('--verbose')) {
      page.on('requestfailed', (r) => console.log(`    x ${new URL(r.url()).pathname} ${r.failure()?.errorText} (${r.resourceType()})`));
      page.on('requestfinished', (r) => /\/assets\//.test(r.url()) && console.log(`    ok ${new URL(r.url()).pathname} (${r.resourceType()})`));
    }
    let failed = '';
    let failures = 0;
    let retried = false;
    await ctx.route(
      (u) => u.pathname.startsWith('/assets/'),
      async (route) => {
        const url = new URL(route.request().url()).pathname;
        if (!failed && c.pick(url)) failed = url;
        if (url === failed && (c.untilRetry ? !retried : failures === 0)) {
          failures++;
          return route.abort('connectionreset');
        }
        return route.continue();
      },
    );
    const log = [`${c.name}:`];
    const say = (/** @type {string} */ m) => log.push(`  ${m}`);
    const fail = (/** @type {string} */ m) => {
      problems.push(`${c.name}: ${m}`);
      say(`FAIL ${m}`);
    };
    await page.goto(`${base}/?mock=1${SCREENS[0].hash}`);
    // The first screen, or its error.
    const end = Date.now() + 10_000;
    while (Date.now() < end) {
      const s = await state(page);
      if (s.failed || (s.title === SCREENS[0].title && s.conn === 'Demo data')) break;
      await page.waitForTimeout(100);
    }
    if (!failed) fail('the request to fail was never made');
    for (const [i, sc] of SCREENS.entries()) {
      if (i > 0) await page.evaluate((h) => (location.hash = h), sc.hash);
      let s = await state(page);
      if (s.failed) {
        say(`${sc.hash}: "Could not load this screen" (${failed} failed ${failures} time(s)); Retry`);
        retried = true;
        await page.getByRole('button', { name: 'Retry' }).first().click();
      } else if (c.untilRetry && i === 0) fail(`${sc.hash} did not say it could not load while ${failed} failed`);
      const r = await shown(page, sc.title);
      s = r.s;
      if (!r.ok) fail(`${sc.hash} did not recover: title "${s.title}", load error ${s.failed}, footer "${s.conn}"`);
      const missing = clean[sc.hash].filter((x) => !s.sheets.includes(x));
      if (missing.length) fail(`${sc.hash} without ${missing.join(', ')}`);
      if (s.dead.length) fail(`${sc.hash} has a failed stylesheet link: ${s.dead.join(', ')}`);
      if (r.ok && !missing.length && !s.dead.length) say(`${sc.hash}: shown with every stylesheet, footer "${s.conn}" (${failed} failed ${failures} time(s))`);
    }
    if (errors.length) fail(`uncaught: ${errors.join(' | ')}`);
    console.log(log.join('\n'));
    await ctx.close();
  }
} finally {
  await browser.close();
  await server.close();
}
if (problems.length) {
  console.error(`LOAD FAULTS: ${problems.length} problem(s)\n${problems.join('\n')}`);
  process.exitCode = 1;
} else console.log('Load faults: Studio recovered from every case.');
