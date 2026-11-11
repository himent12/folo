// fw on the js-framework-benchmark operations, in Chrome (playwright-core), production build. (framework/README.md
// has the numbers against Svelte 5.57 measured before Studio left Svelte; these runs are for fw against itself.)
//
//   node framework/bench/run.mjs            # builds the app, then runs (under a minute)
//   node framework/bench/run.mjs --no-build --runs 15
//
// Each operation runs in a fresh page: its setup (unmeasured), then the operation, timed from the call to the DOM
// being updated and laid out (a forced layout after the framework's synchronous flush): script + DOM + style/layout.
// Memory: the JS heap after a forced GC (CDP HeapProfiler.collectGarbage + Runtime.getHeapUsage).

import { execFileSync } from 'node:child_process';
import { createReadStream, existsSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const here = fileURLToPath(new URL('.', import.meta.url));
const web = join(here, '..', '..');
const dist = join(here, 'dist');
const arg = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i >= 0 ? Number(process.argv[i + 1]) : fallback;
};
const RUNS = arg('--runs', 15);
const WARMUP = 3;

if (!process.argv.includes('--no-build')) {
  execFileSync(process.execPath, [join(web, 'node_modules', 'vite', 'bin', 'vite.js'), 'build', '-c', join(here, 'vite.config.ts'), '--logLevel', 'warn'], { cwd: web, stdio: 'inherit' });
}

const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };
const server = createServer((req, res) => {
  const path = normalize(join(dist, decodeURIComponent(new URL(req.url ?? '/', 'http://x').pathname)));
  if (!path.startsWith(dist) || !existsSync(path)) return void res.writeHead(404).end();
  res.writeHead(200, { 'content-type': types[extname(path)] ?? 'application/octet-stream' });
  createReadStream(path).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const origin = `http://127.0.0.1:${server.address().port}`;

let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--js-flags=--expose-gc'] });
} catch {
  browser = await chromium.launch({ headless: true });
}

/** [name, setup ops (unmeasured), measured op] */
const OPS = [
  ['create 1,000 rows', [], ['run']],
  ['replace all 1,000 rows', [['run']], ['run']],
  ['update every 10th row (1k)', [['run']], ['update']],
  ['select row', [['run']], ['select', 5]],
  ['swap rows', [['run']], ['swapRows']],
  ['remove row', [['run']], ['remove', 10]],
  ['create 10,000 rows', [], ['runLots']],
  ['append 1,000 rows (to 1k)', [['run']], ['add']],
  ['clear 1,000 rows', [['run']], ['clear']],
  ['clear 10,000 rows', [['runLots']], ['clear']],
];

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};

async function page(which) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const p = await context.newPage();
  await p.goto(`${origin}/framework/bench/${which}.html`);
  await p.waitForFunction(() => window.bench !== undefined);
  return { p, context };
}

async function measure(which, setup, op) {
  const { p, context } = await page(which);
  for (const [name, a] of setup) await p.evaluate(([n, x]) => (window.bench[n](x), document.body.offsetHeight), [name, a]);
  await p.evaluate(() => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0))));
  const ms = await p.evaluate(([n, x]) => {
    const t0 = performance.now();
    window.bench[n](x);
    const t1 = performance.now();
    void document.body.offsetHeight;
    return [t1 - t0, performance.now() - t0];
  }, op);
  await context.close();
  return ms;
}

async function heap(which, setup) {
  const { p, context } = await page(which);
  for (const [name, a] of setup) await p.evaluate(([n, x]) => (window.bench[n](x), document.body.offsetHeight), [name, a]);
  const cdp = await context.newCDPSession(p);
  await cdp.send('HeapProfiler.collectGarbage');
  await cdp.send('HeapProfiler.collectGarbage');
  const { usedSize } = await cdp.send('Runtime.getHeapUsage');
  await context.close();
  return usedSize / 1024 / 1024;
}

const rows = [];
for (const [name, setup, op] of OPS) {
  const times = [];
  for (let i = 0; i < WARMUP + RUNS; i++) {
    const ms = await measure('fw', setup, op);
    if (i >= WARMUP) times.push(ms);
  }
  const result = [median(times.map((t) => t[1])), median(times.map((t) => t[0]))];
  rows.push([name, result]);
  console.log(`${name.padEnd(28)} ${result[0].toFixed(2).padStart(8)} ms   (script ${result[1].toFixed(2)})`);
}

// Small operations take about a timer tick (Chrome rounds performance.now() to 0.1 ms in a page): they are also
// timed over REPEAT repetitions on 1,000 rows, scripting alone and with a forced layout after each.
const REPEAT = 400;
const repeated = [];
for (const [name, op] of [
  ['swap rows', ['swapRows']],
  ['select row', ['select', -1]],
  ['update every 10th row', ['update']],
  ['remove row', ['remove', 10]],
]) {
  const all = [];
  for (let k = 0; k < 6; k++) {
    const { p, context } = await page('fw');
    all.push(
      await p.evaluate(
        ([[n, x], times]) => {
          const arg = (i) => (x === -1 ? i % 1000 : x);
          window.bench.run();
          void document.body.offsetHeight;
          let t0 = performance.now();
          for (let i = 0; i < times; i++) window.bench[n](arg(i));
          const script = (performance.now() - t0) / times;
          window.bench.run();
          void document.body.offsetHeight;
          t0 = performance.now();
          for (let i = 0; i < times; i++) {
            window.bench[n](arg(i));
            void document.body.offsetHeight;
          }
          return [script, (performance.now() - t0) / times];
        },
        [op, REPEAT],
      ),
    );
    await context.close();
  }
  const r = [median(all.map((x) => x[0])), median(all.map((x) => x[1]))];
  repeated.push([name, r]);
  console.log(`${name.padEnd(28)} x${REPEAT}: script ${r[0].toFixed(3)} ms; +layout ${r[1].toFixed(3)} ms`);
}

const memory = [];
for (const [name, setup] of [
  ['ready (empty table)', []],
  ['1,000 rows', [['run']]],
  ['10,000 rows', [['runLots']]],
  ['1,000 rows created and cleared 5x', [['run'], ['clear'], ['run'], ['clear'], ['run'], ['clear'], ['run'], ['clear'], ['run'], ['clear']]],
]) {
  const f = median(await Promise.all([0, 1, 2].map(() => heap('fw', setup))));
  memory.push([name, f]);
  console.log(`heap ${name.padEnd(34)} ${f.toFixed(2).padStart(6)} MB`);
}

await browser.close();
server.close();

console.log(`
| operation (median of ${RUNS}, ms) | total | script |
|---|---:|---:|`);
for (const [n, f] of rows) console.log(`| ${n} | ${f[0].toFixed(2)} | ${f[1].toFixed(2)} |`);
console.log(`
| JS heap after GC (MB) | fw |
|---|---:|`);
for (const [n, f] of memory) console.log(`| ${n} | ${f.toFixed(2)} |`);
console.log(`
| small operation, mean of ${REPEAT} on 1k rows (ms) | script | + layout |
|---|---:|---:|`);
for (const [n, f] of repeated) console.log(`| ${n} | ${f[0].toFixed(3)} | ${f[1].toFixed(3)} |`);
