// Load faults on the production build: one of the app's own files fails to load (a dropped connection, or gone for
// good), and Studio must end where it ends on `main` or better, without a reload.
//   node scripts/load-faults.mjs [--dist <built dir>] [--base <url>] [--only <case,...>] [--verbose]
// Without --dist or --base it builds the app into node_modules/.faults-dist (never web/dist) and serves it with
// `vite preview`; --dist serves a build made elsewhere (another version's, to compare), --base uses a running server.
// Each case opens a cold `/?mock=1#/` (Overview, demo data) in a fresh context, then goes to Settings by a link (a hash
// change), with one file failing (the first request of it the case picks, and as many more as it says):
// - stylesheet: the first stylesheet a screen needs beyond the entry's (the shared one: every screen's ui parts), once.
//   When Overview says it could not load (the bundler's preload helper fails the import), Retry must show it.
// - screen-css: Overview's own stylesheet, once. Overview says it could not load, as on main; Retry must show it.
// - screen-css-gone: Overview's own stylesheet, every time. Overview says it could not load; Settings, opened from
//   there, must open as it does on main, with every stylesheet of its own; Overview's is asked for again (and fails).
// - mock: the in-browser mock's module (`?mock=1` loads it as a chunk of its own), twice: its modulepreload and its
//   import both fail; Studio must load it again and show the demo data.
// What must follow, on each screen: it shows with "Demo data" in the footer and every stylesheet a clean load of it
// applies (but the one gone for good), applied, in the order a clean load gives them (lib/failedStyles puts a stylesheet
// it asks for again where the failed link was); no failed stylesheet link but the one gone for good; the failing file
// asked for as often as the case says (the retry really went out); no uncaught error, no reload. Exits 1 on any failure
// (`npm run qa`, so CI: its first shard).
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
const overviewCss = (/** @type {string} */ url) => /\/assets\/Overview-[^/]+\.css$/.test(url);

/**
 * The cases. `pick(url)`: whether a request is of the file that fails (the first such file; the same one after);
 * `fails`: how many of its requests fail (Infinity: all); `asked`: how many requests of it must be made at least (the
 * retry went out); `first`: what Overview does, `shows`, `says` it could not load (then Retry, unless `gone`) or
 * either (`any`: the bundler's preload helper fails every import waiting on the stylesheet then); `gone`:
 * the file never loads, so it is missing at the end (its link failed) and Overview's error stays.
 * @type {{ name: string, pick: (url: string) => boolean, fails: number, asked: number, first: 'shows' | 'says' | 'any', gone?: boolean }[]}
 */
const CASES = [
  { name: 'stylesheet', pick: sharedCss, fails: 1, asked: 2, first: 'any' },
  { name: 'screen-css', pick: overviewCss, fails: 1, asked: 2, first: 'says' },
  { name: 'screen-css-gone', pick: overviewCss, fails: Infinity, asked: 2, first: 'says', gone: true },
  { name: 'mock', pick: (url) => /\/assets\/mock-[^/]+\.js$/.test(url), fails: 2, asked: 3, first: 'shows' },
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
      // Every stylesheet the page applies, by file name with its hash, in document order; and every one it links to but
      // does not apply.
      sheets: [...new Set(links.filter(applied).map(path))],
      dead: links.filter((l) => !applied(l)).map(path),
    };
  });

/**
 * Waits (real time) until `ok(state)`, or the screen titled `orError` says it could not load (a screen whose chunk has not
 * loaded keeps the one before on display, its error too); the last state seen.
 * @param {import('playwright-core').Page} page
 * @param {(s: Awaited<ReturnType<typeof state>>) => boolean} ok
 * @param {string} [orError]
 */
async function until(page, ok, orError, ms = 15_000) {
  const end = Date.now() + ms;
  let s = await state(page);
  while (Date.now() < end) {
    s = await state(page);
    if (ok(s) || (orError !== undefined && s.failed && s.title === orError)) return s;
    await page.waitForTimeout(100);
  }
  return s;
}
/** The screen titled `title` shows, loaded, with demo data. */
const showing = (/** @type {string} */ title) => (/** @type {Awaited<ReturnType<typeof state>>} */ s) => s.title === title && !s.failed && s.conn === 'Demo data';
/** The stylesheets whose order is checked: the entry's, the shared one and the two screens' own. */
const ordered = (/** @type {string[]} */ sheets) => sheets.filter((f) => /\/assets\/(index|shared|Overview|Settings)-[^/]+\.css$/.test(f));

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
      const s = await until(page, showing(sc.title));
      if (!showing(sc.title)(s)) throw new Error(`a clean load of ${sc.hash} never showed: ${JSON.stringify(s)}`);
      clean[sc.hash] = s.sheets;
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
      page.on('console', (m) => console.log(`    console.${m.type()}: ${m.text()}`));
    }
    let failed = '';
    let asked = 0;
    await ctx.route(
      (u) => u.pathname.startsWith('/assets/'),
      async (route) => {
        const url = new URL(route.request().url()).pathname;
        if (!failed && c.pick(url)) failed = url;
        if (url !== failed) return route.continue();
        asked++;
        return asked <= c.fails ? route.abort('connectionreset') : route.continue();
      },
    );
    const log = [`${c.name}:`];
    const say = (/** @type {string} */ m) => log.push(`  ${m}`);
    const fail = (/** @type {string} */ m) => {
      problems.push(`${c.name}: ${m}`);
      say(`FAIL ${m}`);
    };
    await page.goto(`${base}/?mock=1${SCREENS[0].hash}`);
    // Gone if the page reloads.
    await page.evaluate(() => Object.defineProperty(window, '__faults', { value: true }));
    for (const [i, sc] of SCREENS.entries()) {
      if (i > 0) await page.evaluate((h) => (location.hash = h), sc.hash);
      let s = await until(page, showing(sc.title), sc.title);
      if (i === 0 && c.first === 'says' && !s.failed) fail(`${sc.hash} did not say it could not load while ${failed} failed (as main does)`);
      if (s.failed && !(i === 0 && c.first !== 'shows'))
        fail(`${sc.hash} said it could not load: ${await page.evaluate(() => document.querySelector('main')?.textContent?.replace(/\s+/g, ' ').slice(0, 300))}`);
      if (s.failed && c.gone) {
        say(`${sc.hash}: "Could not load this screen" (${failed} gone, asked ${asked} time(s)), as on main`);
        continue;
      }
      if (s.failed) {
        say(`${sc.hash}: "Could not load this screen" (${failed} failed ${asked} time(s)); Retry`);
        await page.getByRole('button', { name: 'Retry' }).first().click();
      }
      // Every stylesheet a clean load applies (but the one gone for good): a stylesheet asked for again applies when it
      // arrives, so give it the time a load takes.
      const want = clean[sc.hash].filter((f) => !(c.gone && f === failed));
      s = await until(page, (x) => showing(sc.title)(x) && want.every((f) => x.sheets.includes(f)));
      if (!showing(sc.title)(s)) fail(`${sc.hash} did not show: title "${s.title}", load error ${s.failed}, footer "${s.conn}"`);
      const missing = want.filter((x) => !s.sheets.includes(x));
      if (missing.length) fail(`${sc.hash} without ${missing.join(', ')}`);
      const order = ordered(s.sheets);
      const cleanOrder = ordered(clean[sc.hash]).filter((f) => order.includes(f));
      if (order.filter((f) => cleanOrder.includes(f)).join() !== cleanOrder.join()) fail(`${sc.hash}: stylesheets in the order ${order.join(', ')}; a clean load: ${cleanOrder.join(', ')}`);
      const dead = s.dead.filter((x) => !(c.gone && x === failed));
      if (dead.length) fail(`${sc.hash} has a failed stylesheet link: ${dead.join(', ')}`);
      if (showing(sc.title)(s) && !missing.length && !dead.length) say(`${sc.hash}: shown with every stylesheet${c.gone ? ' but the one gone' : ''}, in order, footer "${s.conn}"`);
    }
    if (!failed) fail('the request to fail was never made');
    else if (asked < c.asked) fail(`${failed} asked for ${asked} time(s), expected at least ${c.asked} (the retry never went out)`);
    else say(`${failed} asked for ${asked} time(s), ${Math.min(asked, c.fails)} failed`);
    if (!(await page.evaluate(() => '__faults' in window))) fail('the page reloaded');
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
