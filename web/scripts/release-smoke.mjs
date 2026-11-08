// Checks a release build's embedded Studio end to end (docs/RELEASE.md): starts `<folo> serve` on a throwaway data
// dir holding a synthetic dataset and the given model, opens the Playground in headless Chrome and detects a sample
// image on the server, then in the browser through folo-wasm on WebGPU and on the CPU. Reports page errors, console
// errors and failed requests; exits non-zero on any of them or when a backend does not finish.
//   node web/scripts/release-smoke.mjs --folo <folo binary> --weights <model.folo> [--backends server,webgpu,cpu]
//        [--shots <dir>] [--keep] [--verbose]
// Uses the installed Chrome (playwright-core from web/node_modules, channel "chrome"); no browser download.
import { spawn } from 'node:child_process';
import { copyFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { chromium } from 'playwright-core';

const args = process.argv.slice(2);
const opt = (name) => (args.includes(`--${name}`) ? args[args.indexOf(`--${name}`) + 1] : undefined);
const folo = opt('folo');
const weights = opt('weights');
if (!folo || !weights) {
  console.error('usage: node web/scripts/release-smoke.mjs --folo <folo binary> --weights <model.folo> [--backends server,webgpu,cpu] [--shots <dir>] [--keep]');
  process.exit(2);
}
/** @type {Record<string, string>} */
const LABEL = { server: 'Server', webgpu: 'WebGPU', cpu: 'CPU' };
const backends = (opt('backends') ?? 'server,webgpu,cpu').split(',');
const bad = backends.filter((b) => !LABEL[b]);
if (bad.length) {
  console.error(`unknown backends: ${bad.join(', ')} (server, webgpu, cpu)`);
  process.exit(2);
}
const shotsArg = opt('shots');
const shots = shotsArg ? resolve(shotsArg) : undefined;
if (shots) mkdirSync(shots, { recursive: true });

const dataDir = mkdtempSync(join(tmpdir(), 'folo-release-smoke-'));
mkdirSync(join(dataDir, 'models'));
copyFileSync(weights, join(dataDir, 'models', basename(weights)));

// The binary's own Studio: no --web-dir, so an embed-web build serves the copy compiled into it.
const server = spawn(resolve(folo), ['serve', '--data-dir', dataDir, '--port', '0', '--json'], {
  stdio: ['ignore', 'pipe', args.includes('--verbose') ? 'inherit' : 'ignore'],
});
const problems = [];
let browser;
try {
  const listening = await new Promise((ok, fail) => {
    server.on('exit', (code) => fail(new Error(`folo serve exited with ${code}`)));
    createInterface({ input: server.stdout }).on('line', (l) => {
      try {
        const v = JSON.parse(l);
        if (v.event === 'listening') ok(v);
      } catch {}
    });
  });
  const base = listening.url.replace(/\/$/, '');
  console.log(`folo serve ${base} (web_source: ${listening.web_source})`);
  if (listening.web_source !== 'embedded') problems.push(`web_source is ${listening.web_source}, not the embedded Studio`);

  const api = async (path, init) => {
    const r = await fetch(`${base}/api${path}`, init);
    if (!r.ok) throw new Error(`${init?.method ?? 'GET'} /api${path}: ${r.status} ${await r.text()}`);
    return r.json();
  };
  await api('/datasets/synthetic', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: 'release-smoke', images: 10, classes: ['circle', 'square', 'triangle'], imgsz: 320, seed: 1 }),
  });
  const model = (await api('/models')).models[0];
  if (!model) throw new Error(`folo serve lists no model for ${weights}`);
  console.log(`model ${model.id} (${model.name}, ${model.arch})`);

  // Headless Chrome exposes WebGPU only with this flag.
  browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
  const page = await (await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: 'dark' })).newPage();
  page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
  page.on('console', (m) => m.type() === 'error' && problems.push(`console: ${m.text()}`));
  // ERR_ABORTED is the Playground cancelling a request it no longer needs (switching backends supersedes it).
  page.on('requestfailed', (r) => {
    const why = r.failure()?.errorText ?? '';
    if (!why.includes('ERR_ABORTED')) problems.push(`failed: ${r.method()} ${r.url().replace(base, '')} (${why})`);
  });
  page.on('response', (r) => r.status() >= 400 && problems.push(`${r.status()} ${r.request().method()} ${r.url().replace(base, '')}`));
  const wasm = [];
  page.on('response', (r) => r.url().includes('/folo-wasm/') && wasm.push(`${r.status()} ${r.headers()['content-type']} ${r.url().replace(base, '')}`));

  await page.goto(`${base}/#/playground?model=${model.id}`, { waitUntil: 'networkidle' });
  const sample = page.locator('button.sample').first();
  await sample.waitFor({ timeout: 30_000 });
  await sample.click();
  for (const b of backends) {
    const t0 = Date.now();
    await page.getByRole('radio', { name: LABEL[b] }).click();
    const done = b === 'server' ? 'server' : `browser · ${LABEL[b]}`;
    try {
      await page.waitForFunction(
        // The Timing panel's meta says where the shown result ran ("server", "browser · WebGPU", ...).
        (done) => !document.querySelector('.busy') && document.querySelector('.timing')?.textContent?.includes(done),
        done,
        { timeout: 180_000 },
      );
    } catch {
      const shown = (await page.locator('main').first().innerText()).replace(/\s+/g, ' ').slice(0, 400);
      problems.push(`${b}: no result within 180 s; the page shows: ${shown}`);
      continue;
    }
    const timing = (await page.locator('.timing').first().innerText()).replace(/\s+/g, ' ');
    const found = (await page.getByText(/shown of/).first().innerText().catch(() => '?')).replace(/\s+/g, ' ');
    console.log(`  ${b}: ${Date.now() - t0} ms; ${found}; ${timing}`);
    if (shots) await page.screenshot({ path: join(shots, `playground-${b}.png`) });
  }
  for (const w of wasm) console.log(`  ${w}`);
  if (backends.some((b) => b !== 'server') && !wasm.some((w) => w.startsWith('200 application/wasm'))) problems.push('folo_wasm_bg.wasm was not served as application/wasm');
} finally {
  await browser?.close();
  server.kill();
  if (!args.includes('--keep')) rmSync(dataDir, { recursive: true, force: true });
  else console.log(`data dir kept: ${dataDir}`);
}
if (problems.length) {
  console.log(`${problems.length} problems:`);
  for (const p of problems) console.log(`  ${p}`);
  process.exit(1);
}
console.log('the embedded Playground ran on every backend; no page errors, console errors or failed requests');
