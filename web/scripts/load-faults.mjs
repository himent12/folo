// Load faults on the production build: one of the app's own files fails to load (a dropped connection, or gone for
// good), and Studio must end where it ends on `main` or better, without a reload.
//   node scripts/load-faults.mjs [--dist <built dir>] [--base <url>] [--only <case,...>] [--delay <regex>=<ms>]
//     [--retry-wait <ms>] [--verbose]
// Without --dist or --base it builds the app into a new folder under the OS temp dir (never in the checkout), serves
// it with `vite preview` and removes it at the end; --dist serves a build made elsewhere (another version's, to
// compare), --base uses a running server. --delay: every request of an asset whose path matches is answered that much
// later in each case (a slow file; the script must not take a file still loading for a failed one). --retry-wait: a
// screen that says it could not load is given that long before Retry is clicked (a busy machine; Studio may retry it
// itself meanwhile, the idle preload of the other screens loading it too, and then there is no Retry to click).
// Each case opens a cold `/?mock=1#/` (Overview, demo data) in a fresh context, then goes to Settings by a link (a hash
// change), with one file failing (the first request of it the case picks, and as many more as it says):
// - stylesheet: the first stylesheet a screen needs beyond the entry's (the shared one: every screen's ui parts), once.
//   When Overview says it could not load (the bundler's preload helper fails the import), Retry must show it.
// - screen-css: Overview's own stylesheet, once. Overview says it could not load, as on main; Retry must show it.
// - screen-css-gone: Overview's own stylesheet, every time. Overview says it could not load; Settings, opened from
//   there, must open as it does on main, with every stylesheet of its own; Overview's is asked for again (and fails).
// - mock: the in-browser mock's module (`?mock=1` loads it as a chunk of its own), twice: its modulepreload and its
//   import both fail; Studio must load it again and show the demo data.
// What must follow, on each screen: it shows with "Demo data" in the footer and every stylesheet it needs (what a cold
// load of it alone had loaded when it first showed, before the idle preload of the other screens) loaded (but the one
// gone for good), in the order a clean load gives them (lib/failedStyles puts a stylesheet it asks for again where the
// failed link was); no failed stylesheet link but the one gone for good; the failing file asked for as often as the
// case says (the retry really went out), and in the end loaded (a stylesheet: a link of it loaded) or, gone for good,
// never; no uncaught error, no reload. A stylesheet's outcome is its link's `load` or `error` event: a link with neither
// yet (the idle preload of the other screens, still on its way) is still loading, neither loaded nor failed, and only
// the stylesheets a screen needs are waited for; a loaded link counts only while it applies (it has a sheet, is not
// disabled and its media matches). Exits 1 on any failure (`npm run qa`, so CI: its first shard).
// Uses the installed Chrome (playwright-core, channel "chrome"); without one, Playwright's Chromium.
import { preview } from 'vite';
import { chromium } from 'playwright-core';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
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

/**
 * Runs in every page before its own scripts: the outcome of each stylesheet link, from its `load` or `error` event
 * (neither bubbles: heard on the way down), in `window.__sheets` (a link with none yet is still loading); whether a
 * loaded link applies, `window.__applies(link)` (it has a sheet, is not disabled and its media matches); when a screen
 * first shows (its title in `.page h1`), the stylesheets loaded and applying at that moment, in
 * `window.__atShow[title]`; and each screen that said it could not load, even for a moment, in `window.__said[title]`.
 */
const watchSheets = () => {
  /** @type {WeakMap<Element, 'load' | 'error'>} */
  const sheets = new WeakMap();
  /** @type {Record<string, string[]>} */
  const atShow = {};
  /** @type {Record<string, true>} */
  const said = {};
  const applies = (/** @type {HTMLLinkElement} */ l) =>
    l.sheet !== null && !l.disabled && !l.sheet.disabled && (!l.media || matchMedia(l.media).matches);
  Object.defineProperty(window, '__sheets', { value: sheets });
  Object.defineProperty(window, '__applies', { value: applies });
  Object.defineProperty(window, '__atShow', { value: atShow });
  Object.defineProperty(window, '__said', { value: said });
  const on = (/** @type {'load' | 'error'} */ outcome) => (/** @type {Event} */ e) => {
    const t = e.target;
    if (t instanceof HTMLLinkElement && t.rel === 'stylesheet') sheets.set(t, outcome);
  };
  document.addEventListener('load', on('load'), true);
  document.addEventListener('error', on('error'), true);
  new MutationObserver(() => {
    const title = document.querySelector('.page h1')?.textContent?.trim();
    if (!title) return;
    if (!(title in said) && document.body?.innerText.includes('Could not load this screen')) said[title] = true;
    if (title in atShow) return;
    atShow[title] = /** @type {HTMLLinkElement[]} */ ([...document.querySelectorAll('link[rel=stylesheet]')])
      .filter((l) => sheets.get(l) === 'load' && applies(l))
      .map((l) => new URL(l.href).pathname);
  }).observe(document, { childList: true, subtree: true, characterData: true });
};

/** @param {import('playwright-core').Page} page */
const state = (page) =>
  page.evaluate(() => {
    const links = /** @type {HTMLLinkElement[]} */ ([...document.querySelectorAll('link[rel=stylesheet]')]);
    const w = /** @type {any} */ (window);
    const outcome = /** @type {WeakMap<Element, 'load' | 'error'>} */ (w.__sheets);
    const applies = /** @type {(l: HTMLLinkElement) => boolean} */ (w.__applies);
    const path = (/** @type {HTMLLinkElement} */ l) => new URL(l.href).pathname;
    const title = document.querySelector('.page h1')?.textContent?.trim() ?? '';
    return {
      title,
      failed: document.body.innerText.includes('Could not load this screen'),
      // The screen titled so said it could not load at some point (it may have been retried since).
      said: title in w.__said,
      conn: document.querySelector('.conn .label')?.textContent?.trim() ?? '',
      // Every stylesheet the page links to and has loaded and that applies, by file name with its hash, in document
      // order; every one whose load failed; every one still loading; every one loaded that does not apply.
      sheets: [...new Set(links.filter((l) => outcome.get(l) === 'load' && applies(l)).map(path))],
      dead: links.filter((l) => outcome.get(l) === 'error').map(path),
      loading: links.filter((l) => !outcome.has(l)).map(path),
      inert: links.filter((l) => outcome.get(l) === 'load' && !applies(l)).map(path),
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

/**
 * Clicks the screen's Retry if it is still there; whether it did. Studio may have retried the screen itself since its
 * error was seen (the idle preload of the other screens loads it too), and then the error and its button are gone: not
 * a failure, what follows checks the screen shows. Bounded: a button that detaches before a click lands is looked for
 * again, for `ms` in all.
 * @param {import('playwright-core').Page} page
 */
async function retryIfThere(page, ms = 5_000) {
  const button = page.getByRole('button', { name: 'Retry' }).first();
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if ((await button.count()) === 0) return false;
    try {
      await button.click({ timeout: Math.max(1, Math.min(1_000, end - Date.now())) });
      return true;
    } catch (e) {
      if (!(e instanceof Error && e.name === 'TimeoutError')) throw e;
    }
  }
  return false;
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
const delayArg = opt('--delay');
const delay = delayArg ? { re: new RegExp(delayArg.slice(0, delayArg.lastIndexOf('='))), ms: Number(delayArg.slice(delayArg.lastIndexOf('=') + 1)) } : null;
if (delay && !(delay.ms >= 0)) throw new Error(`--delay ${delayArg}: expected <regex>=<ms>`);
const retryWait = Number(opt('--retry-wait') ?? 0);
if (!(retryWait >= 0)) throw new Error(`--retry-wait ${opt('--retry-wait')}: expected <ms>`);
const only = opt('--only')?.split(',');
/** @type {string[]} */
const problems = [];
// What the set-up made, undone at the end whatever fails (the build, the server, the browser).
/** @type {{ close(): Promise<void> } | undefined} */
let server;
/** The build this run made (under the OS temp dir), removed at the end. */
let made = '';
/** @type {import('playwright-core').Browser | undefined} */
let browser;
try {
  let base = given?.replace(/\/$/, '');
  if (!base) {
    if (!dist) made = mkdtempSync(join(tmpdir(), 'folo-load-faults-'));
    const outDir = dist ? resolve(dist) : made;
    if (!dist) await buildApps(root, outDir);
    const p = await preview({ root, logLevel: 'error', build: { outDir }, preview: { host: '127.0.0.1', port: 5197, strictPort: false } });
    server = p;
    base = (p.resolvedUrls?.local[0] ?? 'http://127.0.0.1:5197/').replace(/\/$/, '');
  }
  browser = await launch();

  // What each screen needs: the stylesheets a cold load of it alone (a fresh context, no fault) had loaded when it
  // first showed, before the idle preload of the other screens. The reference for each case.
  /** @type {Record<string, string[]>} */
  const clean = {};
  for (const sc of SCREENS) {
    const ctx = await browser.newContext();
    await ctx.addInitScript(watchSheets);
    const page = await ctx.newPage();
    await page.goto(`${base}/?mock=1${sc.hash}`);
    const s = await until(page, showing(sc.title));
    if (!showing(sc.title)(s)) throw new Error(`a clean load of ${sc.hash} never showed: ${JSON.stringify(s)}`);
    const at = await page.evaluate((t) => /** @type {any} */ (window).__atShow[t], sc.title);
    if (!at?.length) throw new Error(`a clean load of ${sc.hash}: no stylesheet loaded when it showed`);
    clean[sc.hash] = at;
    await ctx.close();
  }
  // The order a clean load gives the stylesheets on each screen, reached as a case reaches it (the first cold, the next
  // by a link): what is loaded once it shows with what it needs.
  /** @type {Record<string, string[]>} */
  const cleanOrder = {};
  {
    const ctx = await browser.newContext();
    await ctx.addInitScript(watchSheets);
    const page = await ctx.newPage();
    await page.goto(`${base}/?mock=1${SCREENS[0].hash}`);
    for (const [i, sc] of SCREENS.entries()) {
      if (i > 0) await page.evaluate((h) => (location.hash = h), sc.hash);
      const s = await until(page, (x) => showing(sc.title)(x) && clean[sc.hash].every((f) => x.sheets.includes(f)));
      if (!showing(sc.title)(s)) throw new Error(`a clean load of ${sc.hash} never showed: ${JSON.stringify(s)}`);
      cleanOrder[sc.hash] = ordered(s.sheets);
    }
    await ctx.close();
  }

  for (const c of CASES) {
    if (only && !only.includes(c.name)) continue;
    const ctx = await browser.newContext();
    await ctx.addInitScript(watchSheets);
    const page = await ctx.newPage();
    /** @type {string[]} */
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    // The network outcome of each request of the failing file.
    const outcomes = { finished: 0, failed: 0 };
    const ofFailed = (/** @type {import('playwright-core').Request} */ r) => failed !== '' && new URL(r.url()).pathname === failed;
    page.on('requestfinished', (r) => ofFailed(r) && outcomes.finished++);
    page.on('requestfailed', (r) => ofFailed(r) && outcomes.failed++);
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
        if (url !== failed) {
          if (delay?.re.test(url)) await new Promise((later) => setTimeout(later, delay.ms));
          return route.continue();
        }
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
      // Whether it said so is what the page recorded (`said`), not only this snapshot: on a busy machine Studio may
      // have retried it before the first look.
      if (i === 0 && c.first === 'says' && !s.failed && !s.said) fail(`${sc.hash} did not say it could not load while ${failed} failed (as main does)`);
      if (s.failed && !(i === 0 && c.first !== 'shows'))
        fail(`${sc.hash} said it could not load: ${await page.evaluate(() => document.querySelector('main')?.textContent?.replace(/\s+/g, ' ').slice(0, 300))}`);
      if (s.failed && c.gone) {
        say(`${sc.hash}: "Could not load this screen" (${failed} gone, asked ${asked} time(s)), as on main`);
        continue;
      }
      if (s.failed) {
        const seen = `${sc.hash}: "Could not load this screen" (${failed} failed ${asked} time(s))`;
        if (retryWait) await page.waitForTimeout(retryWait);
        const clicked = await retryIfThere(page);
        say(`${seen}; ${clicked ? 'Retry' : 'no Retry left to click (the error went away: the screen was loaded again meanwhile)'}`);
      } else if (s.said) say(`${sc.hash}: "Could not load this screen" for a moment, gone before the first look (the screen was loaded again meanwhile)`);
      // Every stylesheet the screen needs (but the one gone for good): a stylesheet asked for again applies when it
      // arrives, so give it the time a load takes. Others still loading (the idle preload) are not waited for.
      const want = clean[sc.hash].filter((f) => !(c.gone && f === failed));
      s = await until(page, (x) => showing(sc.title)(x) && want.every((f) => x.sheets.includes(f)));
      if (!showing(sc.title)(s)) fail(`${sc.hash} did not show: title "${s.title}", load error ${s.failed}, footer "${s.conn}"`);
      const missing = want.filter((x) => !s.sheets.includes(x));
      if (missing.length) fail(`${sc.hash} without ${missing.map((x) => (s.inert.includes(x) ? `${x} (loaded, does not apply)` : x)).join(', ')}`);
      const order = ordered(s.sheets);
      const inOrder = cleanOrder[sc.hash].filter((f) => order.includes(f));
      if (order.filter((f) => inOrder.includes(f)).join() !== inOrder.join()) fail(`${sc.hash}: stylesheets in the order ${order.join(', ')}; a clean load: ${inOrder.join(', ')}`);
      const dead = s.dead.filter((x) => !(c.gone && x === failed));
      if (dead.length) fail(`${sc.hash} has a failed stylesheet link: ${dead.join(', ')}`);
      if (showing(sc.title)(s) && !missing.length && !dead.length)
        say(`${sc.hash}: shown with every stylesheet it needs${c.gone ? ' but the one gone' : ''}, in order, footer "${s.conn}" (${s.loading.length} other(s) still loading)`);
    }
    // The file failed on purpose: in the end it loaded (a stylesheet: a link of it), or, gone for good, never did.
    if (failed) {
      const end = await state(page);
      const css = failed.endsWith('.css');
      if (c.gone) {
        if (outcomes.finished > 0 || (css && end.sheets.includes(failed))) fail(`${failed} loaded, but it is gone for good`);
      } else if (outcomes.finished === 0) fail(`${failed} never loaded (${outcomes.failed} request(s) failed)`);
      else if (css && !end.sheets.includes(failed))
        fail(`${failed} loaded, but no link of it has and applies (${end.loading.includes(failed) ? 'still loading' : end.inert.includes(failed) ? 'loaded, does not apply' : 'not linked'})`);
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
  await browser?.close().catch((e) => console.error(`load-faults: closing the browser: ${e}`));
  await server?.close().catch((e) => console.error(`load-faults: closing the server: ${e}`));
  if (made) rmSync(made, { recursive: true, force: true });
}
if (problems.length) {
  console.error(`LOAD FAULTS: ${problems.length} problem(s)\n${problems.join('\n')}`);
  process.exitCode = 1;
} else console.log('Load faults: Studio recovered from every case.');
