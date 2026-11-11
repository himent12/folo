// fw vs Svelte 5 on the js-framework-benchmark operations, in Chrome (playwright-core), production builds.
//
//   node framework/bench/run.mjs            # builds both apps, then runs (about a minute)
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
  // Interleaved, alternating which goes first, so drift (heat, background work) hits both alike.
  const times = { svelte: [], fw: [] };
  for (let i = 0; i < WARMUP + RUNS; i++) {
    for (const which of i % 2 ? ['fw', 'svelte'] : ['svelte', 'fw']) {
      const ms = await measure(which, setup, op);
      if (i >= WARMUP) times[which].push(ms);
    }
  }
  const result = {
    svelte: [median(times.svelte.map((t) => t[1])), median(times.svelte.map((t) => t[0]))],
    fw: [median(times.fw.map((t) => t[1])), median(times.fw.map((t) => t[0]))],
  };
  rows.push([name, result.svelte, result.fw]);
  console.log(`${name.padEnd(28)} svelte ${result.svelte[0].toFixed(2).padStart(8)} ms   fw ${result.fw[0].toFixed(2).padStart(8)} ms   (script ${result.svelte[1].toFixed(2)} / ${result.fw[1].toFixed(2)})`);
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
  const r = {};
  const all = { svelte: [], fw: [] };
  for (let k = 0; k < 6; k++) {
    for (const which of k % 2 ? ['fw', 'svelte'] : ['svelte', 'fw']) {
      const runs = all[which];
      const { p, context } = await page(which);
      runs.push(
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
  }
  for (const which of ['svelte', 'fw']) r[which] = [median(all[which].map((x) => x[0])), median(all[which].map((x) => x[1]))];
  repeated.push([name, r.svelte, r.fw]);
  console.log(`${name.padEnd(28)} x${REPEAT}: script svelte ${r.svelte[0].toFixed(3)} fw ${r.fw[0].toFixed(3)} ms; +layout svelte ${r.svelte[1].toFixed(3)} fw ${r.fw[1].toFixed(3)} ms`);
}

const memory = [];
for (const [name, setup] of [
  ['ready (empty table)', []],
  ['1,000 rows', [['run']]],
  ['10,000 rows', [['runLots']]],
  ['1,000 rows created and cleared 5x', [['run'], ['clear'], ['run'], ['clear'], ['run'], ['clear'], ['run'], ['clear'], ['run'], ['clear']]],
]) {
  const s = median(await Promise.all([0, 1, 2].map(() => heap('svelte', setup))));
  const f = median(await Promise.all([0, 1, 2].map(() => heap('fw', setup))));
  memory.push([name, s, f]);
  console.log(`heap ${name.padEnd(34)} svelte ${s.toFixed(2).padStart(6)} MB   fw ${f.toFixed(2).padStart(6)} MB`);
}

await browser.close();
server.close();

console.log(`\n| operation (median of ${RUNS}, ms) | Svelte 5 total | fw total | Svelte 5 script | fw script | script Svelte / fw |\n|---|---:|---:|---:|---:|---:|`);
for (const [n, s, f] of rows) console.log(`| ${n} | ${s[0].toFixed(2)} | ${f[0].toFixed(2)} | ${s[1].toFixed(2)} | ${f[1].toFixed(2)} | ${(s[1] / Math.max(f[1], 0.1)).toFixed(2)}x |`);
console.log(`\n| JS heap after GC (MB) | Svelte 5 | fw |\n|---|---:|---:|`);
for (const [n, s, f] of memory) console.log(`| ${n} | ${s.toFixed(2)} | ${f.toFixed(2)} |`);
console.log(`\n| small operation, mean of ${REPEAT} on 1k rows (ms) | Svelte 5 script | fw script | Svelte 5 + layout | fw + layout |\n|---|---:|---:|---:|---:|`);
for (const [n, s, f] of repeated) console.log(`| ${n} | ${s[0].toFixed(3)} | ${f[0].toFixed(3)} | ${s[1].toFixed(3)} | ${f[1].toFixed(3)} |`);
// Slower: script beyond 5% and a timer tick (0.1 ms) on the single runs, or beyond 5% on the precise repeated ones.
const slower = [
  // The script part decides (layout is the browser's, the same DOM for both, and the noisiest number).
  ...rows.filter(([, s, f]) => f[1] > s[1] * 1.05 + 0.1).map(([n]) => n),
  ...repeated.filter(([, s, f]) => f[0] > s[0] * 1.05 || f[1] > s[1] * 1.05).map(([n]) => `${n} (repeated)`),
];
if (slower.length) {
  console.log(`\nfw slower than Svelte on: ${slower.join(', ')}`);
  process.exitCode = 1;
}
