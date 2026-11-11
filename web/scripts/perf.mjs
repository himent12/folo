// Performance of the Studio on the production build, alone or against a baseline (another version's build).
//   node scripts/perf.mjs [--runs 5] [--out <dir>] [--api <url>] [--only bundle,cold,nav,live,large]
//       [--long-runs N] [--cycles 20] [--live-secs 15]
//       [--baseline-dist <dir> | --ref <git ref>] [--app app|base|both]
// The app is built into <out>/.dist (a run's own build). The baseline, `base`, is a built web/dist of another version
// (--baseline-dist, e.g. D:/folo-build/ui-fast-runtime/legacy-dist: the Svelte app the fw app replaced, whose
// index.html is that app), or --ref: that git ref's web/ built in a temporary worktree (npm ci, vite build) into
// <out>/.base-dist. With a baseline, --app defaults to both. Each build is served with `vite preview`, and
// drives Chrome (playwright-core, channel "chrome"; Playwright's Chromium without it) with CDP for the numbers:
// - bundle: per entry, the JS and CSS its HTML loads at startup (raw, gzip, brotli) and what a cold Overview requests
//   before its content shows; all emitted JS and CSS; whether the in-browser mock is in the startup JS.
// - cold: each main screen loaded in a fresh context with the cache off (--runs times): first contentful paint, the
//   screen's content present (its selector, plus one frame), CDP Performance.getMetrics' ScriptDuration, TaskDuration,
//   LayoutDuration and RecalcStyleDuration at that point, long tasks, JS heap after a forced GC, DOM elements, nodes,
//   event listeners, requests.
// - nav: from a loaded Overview, `--cycles` hash navigations through every main screen (each timed in the page, from
//   the hash change to the content's selector plus one frame); heap after GC after the first cycle and at the end
//   (the leak indicator), nodes and listeners at the end.
// - live: --live-secs of real time on the Run page of the mock's live run (its stream: a 100 ms timer per subscriber,
//   src/lib/api/mock/trainingApi.ts runEvents) and on Overview: main-thread busy % (TaskDuration / wall), script ms/s,
//   layouts and style recalcs per second, app rAF callbacks per second, long tasks, heap growth after GC.
// - large: the biggest list the mock has, the dataset grid of ds_fasteners (2,400 images, a virtualised grid): time to
//   its first tiles, then scrolled to the end with the mouse wheel: time, wheel steps, long tasks, the most tiles
//   mounted at once.
// --long-runs (default: --runs, at most 2) is how many times nav, live and large run per app.
// --app both alternates the two (base first, then app first) so a shared machine's load falls on both; the
// report gives medians, min–max and the change. Data: the in-browser mock (?mock=1); --api <url> proxies /api to a
// running server instead (e.g. `folo serve --mock --port 7979`), without ?mock=1.
// Output: perf.json (every sample) and perf.md, in --out (default D:/folo-build/ui-fast-runtime/perf/<timestamp>).
import { preview } from 'vite';
import { chromium } from 'playwright-core';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { cpus, totalmem } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { brotliCompressSync, constants as zc, gzipSync } from 'node:zlib';
import { buildApps } from './entries.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const opt = (/** @type {string} */ flag) => {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
};
const baselineDist = opt('--baseline-dist');
const ref = opt('--ref');
const appArg = opt('--app') ?? (baselineDist || ref ? 'both' : 'app');
if (!['app', 'base', 'both'].includes(appArg)) throw new Error(`--app ${appArg}: expected app, base or both`);
if (appArg !== 'app' && !baselineDist && !ref) throw new Error(`--app ${appArg} needs a baseline: --baseline-dist <dir> or --ref <git ref>`);
const runs = Math.max(1, Number(opt('--runs') ?? 5));
const longRuns = Math.max(1, Number(opt('--long-runs') ?? Math.min(runs, 2)));
const cycles = Math.max(1, Number(opt('--cycles') ?? 20));
const liveSecs = Math.max(1, Number(opt('--live-secs') ?? 15));
const api = opt('--api');
const phases = new Set((opt('--only') ?? 'bundle,cold,nav,live,large').split(',').map((s) => s.trim()));
const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
const outDir = resolve(opt('--out') ?? (existsSync('D:/') ? `D:/folo-build/ui-fast-runtime/perf/${stamp}` : join(root, 'shots/perf', stamp)));
mkdirSync(outDir, { recursive: true });
const started = Date.now();
/** `base`: the baseline build; `app`: this checkout's. @typedef {'base' | 'app'} App */
const VIEW = { viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 };

// ---------------------------------------------------------------- the screens, and what says each one has loaded
/** @typedef {{ key: string, label: string, path: string, ready: string }} Screen */
/** The main screens, as the scenes open them (scripts/scenes/*), each with a selector its loaded content matches. */
async function screens() {
  let run = 'run_7d1e';
  let dataset = 'ds_fasteners';
  if (api) {
    // A real server: its running run (else its first) and its largest dataset.
    const rs = /** @type {{ runs: { id: string, status: string }[] }} */ (await (await fetch(`${api}/api/runs`)).json()).runs;
    run = (rs.find((r) => r.status === 'running') ?? rs[0])?.id ?? run;
    const ds = /** @type {{ datasets: { id: string, images: number }[] }} */ (await (await fetch(`${api}/api/datasets`)).json()).datasets;
    dataset = ds.sort((a, b) => b.images - a.images)[0]?.id ?? dataset;
  }
  /** @type {Screen[]} */
  const list = [
    { key: 'overview', label: 'Overview', path: '/', ready: 'table[aria-label="Recent runs"][aria-busy="false"] tbody tr' },
    { key: 'runs', label: 'Runs', path: '/runs', ready: 'table.runs[aria-busy="false"] tbody tr[data-row]' },
    { key: 'run', label: `Run (${run}, live)`, path: `/runs/${run}`, ready: 'main svg[aria-roledescription=chart] path' },
    { key: 'dataset', label: `Dataset (${dataset})`, path: `/datasets/${dataset}`, ready: 'main .cell img' },
    { key: 'models', label: 'Models', path: '/models', ready: 'table.models[aria-busy="false"] tbody tr' },
    { key: 'playground', label: 'Playground', path: '/playground', ready: 'main button.sample' },
    { key: 'benchmarks', label: 'Benchmarks', path: '/benchmarks', ready: 'table.benches[aria-busy="false"] tbody tr[data-row]' },
    { key: 'settings', label: 'Settings', path: '/settings', ready: 'main details.tech summary' },
  ];
  return list;
}

// ---------------------------------------------------------------- server and browser
// A build dir of this run's own (in --out), so concurrent runs never wipe each other's build.
const apps = /** @type {App[]} */ (appArg === 'both' ? ['base', 'app'] : [appArg]);

/** Builds `ref`'s web/ in a temporary worktree into <out>/.base-dist; returns that folder. */
function buildRef(/** @type {string} */ gitRef) {
  const tree = resolve(outDir, '.base-tree');
  const into = resolve(outDir, '.base-dist');
  const git = (/** @type {string[]} */ a) => execFileSync('git', a, { cwd: root, stdio: 'inherit' });
  rmSync(tree, { recursive: true, force: true });
  git(['worktree', 'add', '--detach', tree, gitRef]);
  try {
    const web = join(tree, 'web');
    // Vite's build of the app set NODE_ENV=production in this process: the ref's install needs its dev dependencies.
    const env = { ...process.env };
    delete env.NODE_ENV;
    const npm = (/** @type {string[]} */ a) => execFileSync('npm', a, { cwd: web, stdio: 'inherit', shell: process.platform === 'win32', env });
    npm(['ci', '--include=dev', '--no-audit', '--no-fund']);
    npm(['exec', '--', 'vite', 'build', '--outDir', into, '--emptyOutDir']);
  } finally {
    git(['worktree', 'remove', '--force', tree]);
  }
  return into;
}

/** Each app's build folder. @type {Record<App, string>} */
const dists = { app: resolve(outDir, '.dist'), base: baselineDist ? resolve(baselineDist) : '' };
if (apps.includes('app')) await buildApps(root, dists.app);
if (apps.includes('base') && ref) dists.base = buildRef(ref);
if (apps.includes('base') && !existsSync(join(dists.base, 'index.html'))) throw new Error(`no index.html in the baseline build ${dists.base}`);
/** What the baseline is, for the report. */
const baseLabel = ref ? `\`${ref}\` (built from its web/)` : baselineDist ? `\`${baselineDist}\`` : '';

/** @type {{ close(): Promise<void> }[]} */
const servers = [];
/** Each app's server. @type {Record<App, string>} */
const bases = /** @type {any} */ ({});
for (const [i, a] of apps.entries()) {
  const server = await preview({
    root,
    logLevel: 'error',
    build: { outDir: dists[a] },
    preview: { host: '127.0.0.1', port: 5197 + i * 10, strictPort: false, ...(api ? { proxy: { '/api': { target: api, changeOrigin: true } } } : {}) },
  });
  servers.push(server);
  bases[a] = (server.resolvedUrls?.local[0] ?? `http://127.0.0.1:${5197 + i * 10}/`).replace(/\/$/, '');
}
const url = (/** @type {App} */ app, /** @type {string} */ path) => `${bases[app]}/${api ? '' : '?mock=1'}#${path}`;

async function launch() {
  if (process.env.FOLO_SHOOT_BROWSER !== 'chromium') {
    try {
      return await chromium.launch({ channel: 'chrome', headless: true });
    } catch (e) {
      if (process.env.FOLO_SHOOT_BROWSER === 'chrome') throw e;
    }
  }
  return chromium.launch({ headless: true });
}
const browser = await launch();
const chrome = browser.version();
const SCREENS = await screens();
const byKey = Object.fromEntries(SCREENS.map((s) => [s.key, s]));

/**
 * What the page records (an init script): the screen's content showing (`readyDom`, then one frame later `ready`, in
 * performance.now ms), long tasks, and the app's requestAnimationFrame callbacks.
 */
function instrument(/** @type {string} */ sel) {
  /** @type {any} */
  const P = {
    sel,
    readyDom: null,
    ready: null,
    longtasks: /** @type {[number, number][]} */ ([]),
    raf: 0,
    onReady: /** @type {(() => void) | null} */ (null),
  };
  Object.defineProperty(window, '__perf', { value: P });
  try {
    new PerformanceObserver((l) => {
      for (const e of l.getEntries()) P.longtasks.push([e.startTime, e.duration]);
    }).observe({ type: 'longtask', buffered: true });
  } catch {}
  const raf = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = (cb) =>
    raf((t) => {
      P.raf++;
      cb(t);
    });
  P.check = () => {
    if (P.readyDom !== null || !P.sel || !document.querySelector(P.sel)) return;
    P.readyDom = performance.now();
    raf(() => {
      P.ready = performance.now();
      P.onReady?.();
    });
  };
  new MutationObserver(() => P.check()).observe(document, { subtree: true, childList: true, attributes: true });
}

/** CDP Performance.getMetrics as an object. */
async function metrics(/** @type {import('playwright-core').CDPSession} */ cdp) {
  const { metrics: list } = await cdp.send('Performance.getMetrics');
  return Object.fromEntries(list.map((/** @type {{ name: string, value: number }} */ m) => [m.name, m.value]));
}

/** JS heap in use after two forced GCs (MB), and the metrics then (Nodes, JSEventListeners without the garbage). */
async function afterGc(/** @type {import('playwright-core').CDPSession} */ cdp) {
  await cdp.send('HeapProfiler.collectGarbage');
  await cdp.send('HeapProfiler.collectGarbage');
  const { usedSize } = await cdp.send('Runtime.getHeapUsage');
  return { heapMB: usedSize / 2 ** 20, m: await metrics(cdp) };
}

/** A fresh context and page with CDP, the cache off, and the page instrumented for `sel`. */
async function open(/** @type {string} */ sel) {
  const context = await browser.newContext(VIEW);
  const page = await context.newPage();
  const cdp = await context.newCDPSession(page);
  await cdp.send('Network.enable');
  await cdp.send('Network.setCacheDisabled', { cacheDisabled: true });
  await cdp.send('Performance.enable');
  await cdp.send('HeapProfiler.enable');
  await page.addInitScript(instrument, sel);
  /** @type {string[]} */
  const requests = [];
  page.on('request', (r) => requests.push(r.url()));
  return { context, page, cdp, requests };
}

/** Waits for the screen's content; on a timeout, says what the page shows instead (a port missing a screen). */
async function waitReady(/** @type {import('playwright-core').Page} */ page, timeout = 30_000) {
  try {
    await page.waitForFunction(() => /** @type {any} */ (window).__perf.ready !== null, null, { timeout, polling: 16 });
  } catch (e) {
    const what = await page
      .evaluate(() => {
        const shows = (document.querySelector('main') ?? document.body).innerText.replace(/\s+/g, ' ').slice(0, 160);
        return `${location.hash} waits for ${/** @type {any} */ (window).__perf?.sel}; the page shows: ${shows}`;
      })
      .catch(() => '');
    throw new Error(`${e instanceof Error ? e.message.split('\n')[0] : e} (${what})`);
  }
}

/** Samples that failed, by phase and app: reported, left out of the numbers. @type {string[]} */
const failures = [];
/**
 * Runs one sample; a failure is recorded and gives undefined.
 * @template T
 * @param {string} what
 * @param {() => Promise<T>} f
 */
async function sample(what, f) {
  try {
    return await f();
  } catch (e) {
    const msg = `${what}: ${e instanceof Error ? e.message : e}`;
    failures.push(msg);
    console.warn(`! ${msg}`);
    return undefined;
  }
}

// ---------------------------------------------------------------- the phases
/** One cold load of `screen`. */
async function cold(/** @type {App} */ app, /** @type {Screen} */ screen) {
  const { context, page, cdp, requests } = await open(screen.ready);
  try {
    await page.goto(url(app, screen.path), { waitUntil: 'commit' });
    await waitReady(page);
    const reqs = requests.length;
    const m = await metrics(cdp);
    const t = await page.evaluate(async () => {
      const P = /** @type {any} */ (window).__perf;
      // Content can show before the first paint entry is recorded (a screen with nothing to fetch): wait for it.
      const fcp = await new Promise((done) => {
        new PerformanceObserver((l) => {
          const e = l.getEntries().find((x) => x.name === 'first-contentful-paint');
          if (e) done(e.startTime);
        }).observe({ type: 'paint', buffered: true });
        setTimeout(() => done(null), 2000);
      });
      return {
        fcp: /** @type {number | null} */ (fcp),
        ready: P.ready,
        readyDom: P.readyDom,
        longtasks: P.longtasks.filter((/** @type {[number, number]} */ l) => l[0] < P.ready),
        elements: document.querySelectorAll('*').length,
      };
    });
    const g = await afterGc(cdp);
    return {
      fcp: t.fcp,
      ready: t.ready,
      script: m.ScriptDuration * 1000,
      task: m.TaskDuration * 1000,
      layout: m.LayoutDuration * 1000,
      recalc: m.RecalcStyleDuration * 1000,
      longTasks: t.longtasks.length,
      longTaskMs: t.longtasks.reduce((/** @type {number} */ s, /** @type {[number, number]} */ l) => s + l[1], 0),
      heapMB: g.heapMB,
      elements: t.elements,
      nodes: g.m.Nodes,
      listeners: g.m.JSEventListeners,
      requests: reqs,
      startup: requests.slice(0, reqs),
    };
  } finally {
    await context.close();
  }
}

/** The navigation cycle, from a loaded Overview. */
async function nav(/** @type {App} */ app) {
  const order = ['runs', 'run', 'dataset', 'models', 'playground', 'benchmarks', 'settings', 'overview'].map((k) => byKey[k]);
  const { context, page, cdp } = await open(byKey.overview.ready);
  try {
    await page.goto(url(app, '/'), { waitUntil: 'commit' });
    await waitReady(page);
    await page.waitForTimeout(500);
    /** @type {Record<string, number[]>} */
    const times = Object.fromEntries(order.map((s) => [s.key, []]));
    let first = null;
    for (let c = 0; c < cycles; c++) {
      for (const s of order) {
        const ms = await page.evaluate(
          ({ sel, path }) =>
            new Promise((done, fail) => {
              const P = /** @type {any} */ (window).__perf;
              P.sel = sel;
              P.readyDom = null;
              P.ready = null;
              const t0 = performance.now();
              const timer = setTimeout(() => fail(new Error(`${path}: ${sel} never showed`)), 20_000);
              P.onReady = () => {
                clearTimeout(timer);
                P.onReady = null;
                done(P.ready - t0);
              };
              location.hash = `#${path}`;
              P.check();
            }),
          { sel: s.ready, path: s.path },
        );
        times[s.key].push(/** @type {number} */ (ms));
      }
      if (c === 0) first = await afterGc(cdp);
    }
    const end = await afterGc(cdp);
    return {
      times,
      heapFirstMB: first?.heapMB ?? 0,
      heapEndMB: end.heapMB,
      heapGrowthMB: end.heapMB - (first?.heapMB ?? 0),
      nodesFirst: first?.m.Nodes ?? 0,
      nodesEnd: end.m.Nodes,
      listenersFirst: first?.m.JSEventListeners ?? 0,
      listenersEnd: end.m.JSEventListeners,
    };
  } finally {
    await context.close();
  }
}

/** --live-secs on a screen that keeps updating, in real time. */
async function live(/** @type {App} */ app, /** @type {Screen} */ screen) {
  const { context, page, cdp } = await open(screen.ready);
  try {
    await page.goto(url(app, screen.path), { waitUntil: 'commit' });
    await waitReady(page);
    await page.waitForTimeout(2000);
    const g0 = await afterGc(cdp);
    const p0 = await page.evaluate(() => ({ raf: /** @type {any} */ (window).__perf.raf, lt: /** @type {any} */ (window).__perf.longtasks.length }));
    const m0 = await metrics(cdp);
    const t0 = Date.now();
    await page.waitForTimeout(liveSecs * 1000);
    const m1 = await metrics(cdp);
    const secs = (Date.now() - t0) / 1000;
    const p1 = await page.evaluate(() => ({ raf: /** @type {any} */ (window).__perf.raf, lt: /** @type {any} */ (window).__perf.longtasks.length }));
    const g1 = await afterGc(cdp);
    return {
      busyPct: ((m1.TaskDuration - m0.TaskDuration) / secs) * 100,
      scriptMsPerS: ((m1.ScriptDuration - m0.ScriptDuration) * 1000) / secs,
      layoutsPerS: (m1.LayoutCount - m0.LayoutCount) / secs,
      recalcsPerS: (m1.RecalcStyleCount - m0.RecalcStyleCount) / secs,
      rafPerS: (p1.raf - p0.raf) / secs,
      longTasks: p1.lt - p0.lt,
      heapGrowthMB: g1.heapMB - g0.heapMB,
      nodesGrowth: g1.m.Nodes - g0.m.Nodes,
      listenersGrowth: g1.m.JSEventListeners - g0.m.JSEventListeners,
    };
  } finally {
    await context.close();
  }
}

/** The dataset grid scrolled from top to bottom with the mouse wheel. */
async function large(/** @type {App} */ app) {
  const screen = byKey.dataset;
  const { context, page } = await open(screen.ready);
  try {
    await page.goto(url(app, screen.path), { waitUntil: 'commit' });
    await waitReady(page);
    const firstTiles = await page.evaluate(() => /** @type {any} */ (window).__perf.ready);
    await page.waitForTimeout(500);
    await page.mouse.move(VIEW.viewport.width / 2 + 100, VIEW.viewport.height / 2);
    const lt0 = await page.evaluate(() => /** @type {any} */ (window).__perf.longtasks.length);
    const t0 = Date.now();
    let steps = 0;
    let maxTiles = 0;
    for (; steps < 600; steps++) {
      await page.mouse.wheel(0, 800);
      const s = await page.evaluate(
        () =>
          new Promise((done) =>
            requestAnimationFrame(() => {
              // The scroller: the grid's nearest scrolling ancestor, else the page.
              let el = document.querySelector('main .cell')?.parentElement ?? null;
              while (el && !(el.scrollHeight > el.clientHeight + 1 && /auto|scroll/.test(getComputedStyle(el).overflowY))) el = el.parentElement;
              const sc = el ?? document.scrollingElement ?? document.documentElement;
              done({ end: sc.scrollTop + sc.clientHeight >= sc.scrollHeight - 2, tiles: document.querySelectorAll('main .cell').length });
            }),
          ),
      );
      maxTiles = Math.max(maxTiles, /** @type {any} */ (s).tiles);
      if (/** @type {any} */ (s).end) break;
    }
    const ms = Date.now() - t0;
    const lt1 = await page.evaluate(() => /** @type {any} */ (window).__perf.longtasks.length);
    return { firstTiles, scrollMs: ms, steps: steps + 1, msPerStep: ms / (steps + 1), longTasks: lt1 - lt0, maxTiles };
  } finally {
    await context.close();
  }
}

// ---------------------------------------------------------------- bundle
const gz = (/** @type {Buffer} */ b) => gzipSync(b, { level: 9 }).length;
const br = (/** @type {Buffer} */ b) => brotliCompressSync(b, { params: { [zc.BROTLI_PARAM_QUALITY]: 11 } }).length;
/** Sizes of the files of build `dist` at these paths (raw, gzip -9, brotli 11). */
function sizes(/** @type {string} */ dist, /** @type {string[]} */ paths) {
  let raw = 0;
  let g = 0;
  let b = 0;
  for (const p of paths) {
    const f = join(dist, p.replace(/^\//, ''));
    if (!existsSync(f)) continue;
    const buf = readFileSync(f);
    raw += buf.length;
    g += gz(buf);
    b += br(buf);
  }
  return { files: paths.length, raw, gzip: g, brotli: b };
}
/** What an entry's HTML loads before any code runs: its module script, modulepreloads and stylesheets. */
function entryFiles(/** @type {App} */ app) {
  const html = readFileSync(join(dists[app], 'index.html'), 'utf8');
  const js = [...html.matchAll(/<script[^>]+type="module"[^>]+src="([^"]+)"/g), ...html.matchAll(/<link[^>]+rel="modulepreload"[^>]+href="([^"]+)"/g)].map((m) => m[1]);
  const css = [...html.matchAll(/<link[^>]+rel="stylesheet"[^>]+href="([^"]+)"/g)].map((m) => m[1]);
  return { js, css };
}
function bundle(/** @type {App} */ app, /** @type {string[]} */ startupUrls) {
  const e = entryFiles(app);
  const local = startupUrls.map((u) => new URL(u).pathname);
  const sjs = local.filter((p) => p.endsWith('.js'));
  const scss = local.filter((p) => p.endsWith('.css'));
  const dist = dists[app];
  // The mock's seeded dataset id: in the startup JS when the mock ships with the app.
  const mockInStartup = sjs.some((p) => existsSync(join(dist, p)) && readFileSync(join(dist, p), 'utf8').includes('ds_fasteners_mini'));
  return { entryJs: sizes(dist, e.js), entryCss: sizes(dist, e.css), startupJs: sizes(dist, sjs), startupCss: sizes(dist, scss), startupRequests: startupUrls.length, mockInStartup };
}
/** Everything a build emitted (its assets). */
function emitted(/** @type {App} */ app) {
  const dist = dists[app];
  const files = readdirSync(join(dist, 'assets'));
  const pick = (/** @type {string} */ ext) => files.filter((f) => f.endsWith(ext)).map((f) => `/assets/${f}`);
  return { js: sizes(dist, pick('.js')), css: sizes(dist, pick('.css')), wasm: pick('.wasm').reduce((s, f) => s + statSync(join(dist, f)).size, 0) };
}

// ---------------------------------------------------------------- run
/** @type {Record<App, any>} */
const raw = /** @type {any} */ ({ base: { cold: {}, nav: [], live: { run: [], overview: [] }, large: [] }, app: { cold: {}, nav: [], live: { run: [], overview: [] }, large: [] } });
/** The order of the apps for round `i`: alternated, so neither always goes first. */
const orderFor = (/** @type {number} */ i) => (apps.length === 2 && i % 2 === 1 ? [...apps].reverse() : apps);
const log = (/** @type {string} */ s) => console.log(`[${Math.round((Date.now() - started) / 1000)}s] ${s}`);
/** @type {Record<App, any>} */
const bundles = /** @type {any} */ ({});
try {
  if (phases.has('cold') || phases.has('bundle')) {
    for (let r = 0; r < runs; r++) {
      for (const [si, s] of SCREENS.entries()) {
        if (!phases.has('cold') && s.key !== 'overview') continue;
        for (const app of orderFor(r + si)) {
          const v = await sample(`cold ${app} ${s.key}`, () => cold(app, s));
          if (!v) continue;
          if (s.key === 'overview' && !bundles[app]) bundles[app] = bundle(app, v.startup);
          delete (/** @type {any} */ (v)).startup;
          (raw[app].cold[s.key] ??= []).push(v);
        }
      }
      log(`cold run ${r + 1}/${runs}`);
      if (!phases.has('cold')) break;
    }
  }
  for (let r = 0; r < longRuns; r++) {
    for (const app of orderFor(r)) {
      if (phases.has('nav')) {
        const n = await sample(`nav ${app}`, () => nav(app));
        if (n) raw[app].nav.push(n);
        log(`nav ${app} ${r + 1}/${longRuns}`);
      }
      if (phases.has('live')) {
        const lr = await sample(`live ${app} run`, () => live(app, byKey.run));
        if (lr) raw[app].live.run.push(lr);
        const lo = await sample(`live ${app} overview`, () => live(app, byKey.overview));
        if (lo) raw[app].live.overview.push(lo);
        log(`live ${app} ${r + 1}/${longRuns}`);
      }
      if (phases.has('large')) {
        const g = await sample(`large ${app}`, () => large(app));
        if (g) raw[app].large.push(g);
        log(`large ${app} ${r + 1}/${longRuns}`);
      }
    }
  }
} finally {
  await browser.close();
  for (const s of servers) await s.close();
}

// ---------------------------------------------------------------- report
const median = (/** @type {number[]} */ xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : NaN;
};
/** @param {number[]} xs */
const stat = (xs) => ({ median: median(xs), min: Math.min(...xs), max: Math.max(...xs), n: xs.length });
/** @param {number} v @param {number} digits */
const fmt = (v, digits) => (Number.isFinite(v) ? v.toFixed(digits) : '–');
/** A cell: the median, with min–max when there are several samples. */
const cellOf = (/** @type {number[] | undefined} */ xs, /** @type {number} */ d) => {
  if (!xs || !xs.length) return '–';
  const s = stat(xs);
  return s.n > 1 && s.min !== s.max ? `${fmt(s.median, d)} (${fmt(s.min, d)}–${fmt(s.max, d)})` : fmt(s.median, d);
};
const change = (/** @type {number[] | undefined} */ a, /** @type {number[] | undefined} */ b) => {
  if (!a?.length || !b?.length) return '';
  const ma = median(a);
  const mb = median(b);
  if (!ma) return mb ? 'new' : '0%';
  const pct = ((mb - ma) / Math.abs(ma)) * 100;
  return `${pct > 0 ? '+' : ''}${pct.toFixed(1)}%`;
};
/** @type {string[]} */
const md = [];
const head = ['| metric | base | app | change |', '|---|---:|---:|---:|'];
const row = (/** @type {string} */ name, /** @type {(app: App) => number[] | undefined} */ get, /** @type {number} */ d) =>
  md.push(`| ${name} | ${apps.includes('base') ? cellOf(get('base'), d) : '–'} | ${apps.includes('app') ? cellOf(get('app'), d) : '–'} | ${change(get('base'), get('app'))} |`);

md.push(
  `# Studio perf: ${apps.join(' vs ')}`,
  '',
  `${new Date().toISOString()} · ${cpus()[0]?.model ?? '?'} ×${cpus().length}, ${Math.round(totalmem() / 2 ** 30)} GB · Chrome ${chrome} headless · ` +
    `production build (vite preview) · data: ${api ? `server ${api}` : 'in-browser mock (?mock=1)'} · ${runs} cold run(s) per screen, ` +
    `${longRuns} nav/live/large run(s), ${cycles} nav cycles, ${liveSecs} s live · ${Math.round((Date.now() - started) / 1000)} s in all` +
    `${apps.length === 2 ? ' · alternated (base first, then app first)' : ''}. Cells: median (min–max). Times in ms.`,
  '',
  ...(apps.includes('base') ? [`base: ${baseLabel}; app: this checkout.`, ''] : []),
);
if (failures.length) md.push(`**${failures.length} sample(s) failed** (left out of the numbers):`, '', ...failures.map((f) => `- ${f}`), '');

if (phases.has('bundle') || phases.has('cold')) {
  const em = Object.fromEntries(apps.map((a) => [a, emitted(a)]));
  md.push('## Bundle', '', ...head);
  const b = (/** @type {App} */ app, /** @type {(x: any) => number} */ f) => (bundles[app] ? [f(bundles[app])] : undefined);
  const kb = (/** @type {number} */ n) => n / 1024;
  row('entry JS (HTML script + modulepreload), KB raw', (a) => b(a, (x) => kb(x.entryJs.raw)), 1);
  row('entry JS, KB gzip', (a) => b(a, (x) => kb(x.entryJs.gzip)), 1);
  row('entry JS, KB brotli', (a) => b(a, (x) => kb(x.entryJs.brotli)), 1);
  row('entry CSS, KB raw', (a) => b(a, (x) => kb(x.entryCss.raw)), 1);
  row('entry CSS, KB gzip', (a) => b(a, (x) => kb(x.entryCss.gzip)), 1);
  row('JS requested before Overview shows, KB raw', (a) => b(a, (x) => kb(x.startupJs.raw)), 1);
  row('JS requested before Overview shows, KB brotli', (a) => b(a, (x) => kb(x.startupJs.brotli)), 1);
  row('JS files requested before Overview shows', (a) => b(a, (x) => x.startupJs.files), 0);
  row('requests before Overview shows (all)', (a) => b(a, (x) => x.startupRequests), 0);
  md.push(
    '',
    `In-browser mock in the startup JS: ${apps.map((a) => `${a} ${bundles[a] ? (bundles[a].mockInStartup ? '**yes**' : 'no') : '?'}`).join(', ')}. ` +
      `All emitted: ${apps
        .map((a) => {
          const e = em[a];
          return `${a} JS ${fmt(e.js.raw / 1024, 1)} KB raw / ${fmt(e.js.gzip / 1024, 1)} gzip / ${fmt(e.js.brotli / 1024, 1)} brotli in ${e.js.files} files, CSS ${fmt(e.css.raw / 1024, 1)} KB raw / ${fmt(e.css.gzip / 1024, 1)} gzip in ${e.css.files} files`;
        })
        .join('; ')}.`,
    '',
  );
}
if (phases.has('cold')) {
  md.push('## Cold load (fresh context, cache off)', '');
  for (const s of SCREENS) {
    md.push(`### ${s.label} \`#${s.path}\``, '', `Content: \`${s.ready}\``, '', ...head);
    const c = (/** @type {App} */ a, /** @type {string} */ k) => raw[a].cold[s.key]?.map((/** @type {any} */ v) => v[k]).filter((/** @type {any} */ x) => x !== null);
    row('first contentful paint', (a) => c(a, 'fcp'), 0);
    row('content shown (+1 frame)', (a) => c(a, 'ready'), 0);
    row('ScriptDuration', (a) => c(a, 'script'), 0);
    row('TaskDuration', (a) => c(a, 'task'), 0);
    row('LayoutDuration', (a) => c(a, 'layout'), 1);
    row('RecalcStyleDuration', (a) => c(a, 'recalc'), 1);
    row('long tasks', (a) => c(a, 'longTasks'), 0);
    row('long tasks, ms', (a) => c(a, 'longTaskMs'), 0);
    row('JS heap after GC, MB', (a) => c(a, 'heapMB'), 2);
    row('DOM elements', (a) => c(a, 'elements'), 0);
    row('DOM nodes (CDP)', (a) => c(a, 'nodes'), 0);
    row('JS event listeners', (a) => c(a, 'listeners'), 0);
    row('requests', (a) => c(a, 'requests'), 0);
    md.push('');
  }
}
if (phases.has('nav')) {
  md.push(`## Navigation (${cycles} cycles through the screens, from a loaded Overview)`, '', ...head);
  const all = (/** @type {App} */ a) => raw[a].nav.flatMap((/** @type {any} */ n) => Object.values(n.times).flat());
  row('any screen, ms', all, 1);
  for (const s of ['runs', 'run', 'dataset', 'models', 'playground', 'benchmarks', 'settings', 'overview'])
    row(`→ ${byKey[s].label}, ms`, (a) => raw[a].nav.flatMap((/** @type {any} */ n) => n.times[s]), 1);
  const n = (/** @type {App} */ a, /** @type {string} */ k) => raw[a].nav.map((/** @type {any} */ v) => v[k]);
  row('heap after GC after cycle 1, MB', (a) => n(a, 'heapFirstMB'), 2);
  row('heap after GC at the end, MB', (a) => n(a, 'heapEndMB'), 2);
  row(`heap growth over ${cycles - 1} cycles, MB (leak)`, (a) => n(a, 'heapGrowthMB'), 2);
  row('DOM nodes after cycle 1', (a) => n(a, 'nodesFirst'), 0);
  row('DOM nodes at the end', (a) => n(a, 'nodesEnd'), 0);
  row('listeners after cycle 1', (a) => n(a, 'listenersFirst'), 0);
  row('listeners at the end', (a) => n(a, 'listenersEnd'), 0);
  md.push('');
}
if (phases.has('live')) {
  for (const k of /** @type {const} */ (['run', 'overview'])) {
    md.push(`## Live: ${liveSecs} s on ${byKey[k].label}`, '', ...head);
    const l = (/** @type {App} */ a, /** @type {string} */ f) => raw[a].live[k].map((/** @type {any} */ v) => v[f]);
    row('main thread busy, %', (a) => l(a, 'busyPct'), 1);
    row('script, ms/s', (a) => l(a, 'scriptMsPerS'), 1);
    row('layouts/s', (a) => l(a, 'layoutsPerS'), 1);
    row('style recalcs/s', (a) => l(a, 'recalcsPerS'), 1);
    row('app rAF callbacks/s', (a) => l(a, 'rafPerS'), 1);
    row('long tasks', (a) => l(a, 'longTasks'), 0);
    row('heap growth after GC, MB', (a) => l(a, 'heapGrowthMB'), 2);
    row('DOM node growth', (a) => l(a, 'nodesGrowth'), 0);
    row('listener growth', (a) => l(a, 'listenersGrowth'), 0);
    md.push('');
  }
}
if (phases.has('large')) {
  md.push(`## Large list: ${byKey.dataset.label} image grid, wheel-scrolled to the end`, '', ...head);
  const g = (/** @type {App} */ a, /** @type {string} */ f) => raw[a].large.map((/** @type {any} */ v) => v[f]);
  row('first tiles shown, ms', (a) => g(a, 'firstTiles'), 0);
  row('scroll to the end, ms', (a) => g(a, 'scrollMs'), 0);
  row('wheel steps (800 px)', (a) => g(a, 'steps'), 0);
  row('ms per wheel step (+1 frame)', (a) => g(a, 'msPerStep'), 1);
  row('long tasks while scrolling', (a) => g(a, 'longTasks'), 0);
  row('most tiles mounted at once', (a) => g(a, 'maxTiles'), 0);
  md.push('');
}
writeFileSync(join(outDir, 'perf.json'), JSON.stringify({ apps, runs, longRuns, cycles, liveSecs, api: api ?? null, screens: SCREENS, bundles, emitted: Object.fromEntries(apps.map((a) => [a, emitted(a)])), raw }, null, 1));
writeFileSync(join(outDir, 'perf.md'), md.join('\n'));
console.log(md.join('\n'));
console.log(`\nperf: ${Math.round((Date.now() - started) / 1000)} s -> ${join(outDir, 'perf.md')}`);
