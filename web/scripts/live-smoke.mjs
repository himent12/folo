// Live smoke test against a real `folo serve`: starts a short training run through Studio's New run form, then
// watches the run page while it trains (screenshots every few seconds), and reports page errors, console errors
// and failed API calls. Needs a server with at least one dataset:
//   folo serve --data-dir D:/folo-data/studio-smoke --port 7979
//   curl -X POST http://127.0.0.1:7979/api/datasets/synthetic -H 'Content-Type: application/json' \
//        -d '{"name":"smoke-shapes","images":240,"classes":["circle","square","triangle"],"imgsz":320,"seed":1}'
//   FOLO_API=http://127.0.0.1:7979 npm run live-smoke -- <out dir> [--epochs 3] [--watch 120] [--weights <.folo>]
// With --weights (and web/public/folo-wasm built by scripts/build-wasm), the Playground also runs that model file in
// the browser on WebGPU and the CPU (the server has no route that sends model files yet).
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { createServer } from 'vite';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const api = process.env.FOLO_API ?? 'http://127.0.0.1:7878';
const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? Number(args[i + 1]) : dflt;
};
const out = resolve(root, args.find((a, i) => !a.startsWith('--') && !args[i - 1]?.startsWith('--')) ?? 'shots/live');
const epochs = opt('epochs', 3);
const watchS = opt('watch', 120);
const weights = args.includes('--weights') ? args[args.indexOf('--weights') + 1] : null;
mkdirSync(out, { recursive: true });

const ds = (await fetch(`${api}/api/datasets`).then((r) => r.json())).datasets[0];
if (!ds) throw new Error(`No dataset on ${api}; create one first (see the header of this script).`);

const server = await createServer({ root, logLevel: 'error', server: { host: '127.0.0.1', port: 5196, strictPort: false } });
await server.listen();
const base = (server.resolvedUrls?.local[0] ?? 'http://127.0.0.1:5196/').replace(/\/$/, '');
// Headless Chrome exposes WebGPU only with this flag.
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const problems = [];
let n = 0;
// Full-page captures resize the viewport mid-shot, which can catch the sidebar's width transition halfway: grow the
// viewport to the document instead (as scripts/shoot.mjs does), shoot it, and put it back.
async function shot(page, name) {
  const vp = page.viewportSize();
  const h = await page.evaluate(() => document.documentElement.scrollHeight);
  if (vp && h > vp.height) {
    await page.setViewportSize({ width: vp.width, height: Math.min(h, 8000) });
    await page.waitForTimeout(600);
  }
  await page.screenshot({ path: resolve(out, `${String(++n).padStart(2, '0')}-${name}.png`) });
  if (vp) await page.setViewportSize(vp);
}

/** A page that records page errors, console errors and failed API calls. */
async function open(viewport) {
  const page = await (await browser.newContext({ viewport, colorScheme: 'dark' })).newPage();
  page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
  // The browser logs every 404 as a console error without its URL; failed API calls are caught below by URL (where
  // the weights route's expected 404 is let through).
  page.on('console', (m) => m.type() === 'error' && !m.text().includes('404 (Not Found)') && problems.push(`console: ${m.text()}`));
  page.on('response', (r) => {
    if (r.url().includes('/api/') && r.status() >= 400 && !/\/weights$/.test(r.url())) problems.push(`${r.status()} ${r.request().method()} ${r.url().replace(base, '')}`);
  });
  return page;
}

try {
  // New run: the form against the real server (dataset preselected), short and small.
  const page = await open({ width: 1440, height: 900 });
  await page.goto(`${base}/#/runs/new?dataset=${ds.id}`, { waitUntil: 'networkidle' });
  await page.getByLabel('Epochs').first().fill(String(epochs));
  await page.getByRole('radio', { name: '320' }).first().click();
  await shot(page, 'desktop-new-run');
  await page.getByRole('button', { name: 'Start training' }).first().click();
  await page.waitForURL(/#\/runs\/run_/, { timeout: 30_000 });
  const runPath = page.url().replace(`${base}/#`, '');
  console.log(`run: ${runPath}`);

  // Watch it train.
  const t0 = Date.now();
  let last = '';
  while (Date.now() - t0 < watchS * 1000) {
    await page.waitForTimeout(8000);
    const status = (await page.locator('.badge').first().textContent())?.trim() ?? '';
    await shot(page, `desktop-run-${Math.round((Date.now() - t0) / 1000)}s-${status.toLowerCase()}`);
    if (status !== last) console.log(`  ${Math.round((Date.now() - t0) / 1000)}s: ${status}`);
    last = status;
    if (['Completed', 'Failed', 'Stopped'].includes(status)) break;
  }

  // Everything else, with the real run and its models present, on a desktop and a phone.
  const models = (await fetch(`${api}/api/models`).then((r) => r.json())).models;
  for (const [vp, viewport] of /** @type {const} */ ([['desktop', { width: 1440, height: 900 }], ['mobile', { width: 390, height: 844 }]])) {
    const p = vp === 'desktop' ? page : await open(viewport);
    const paths = ['/', '/runs', runPath, '/models', `/datasets/${ds.id}?tab=stats`, ...(models[0] ? [`/models/${models[0].id}`] : [])];
    for (const path of vp === 'desktop' ? paths.filter((x) => x !== runPath) : paths) {
      await p.goto(`${base}/#${path}`, { waitUntil: 'networkidle' });
      await p.waitForTimeout(2000);
      await shot(p, `${vp}-${path.replace(/[/?=#]+/g, '_').replace(/^_/, '') || 'overview'}`);
    }
    await p.goto(`${base}/#/playground${models[0] ? `?model=${models[0].id}` : ''}`, { waitUntil: 'networkidle' });
    const sample = p.locator('button.sample').first();
    try {
      await sample.waitFor({ timeout: 10_000 });
      await sample.click();
      await p.waitForTimeout(4000);
    } catch {
      problems.push(`${vp}: no playground sample to try`);
    }
    await shot(p, `${vp}-playground`);
  }

  // In the browser: the run's weights through folo-wasm, on each backend.
  if (weights) {
    const p = page;
    await p.goto(`${base}/#/models`, { waitUntil: 'networkidle' }); // leave the Playground, so it starts empty
    await p.goto(`${base}/#/playground${models[0] ? `?model=${models[0].id}` : ''}`, { waitUntil: 'networkidle' });
    await p.locator('button.sample').first().click();
    await p.waitForFunction(() => !document.querySelector('.busy'), null, { timeout: 60_000 });
    for (const backend of ['WebGPU', 'CPU']) {
      const t0 = Date.now();
      await p.getByRole('radio', { name: backend }).click();
      const need = p.getByRole('button', { name: 'Choose .folo file' });
      // Asked once, on the first backend (the file is kept for the model).
      if (backend === 'WebGPU' && (await need.waitFor({ timeout: 5000 }).then(() => true, () => false))) await p.locator('input[accept=".folo"]').setInputFiles(weights);
      await p.waitForFunction(
        (b) => !document.querySelector('.busy') && document.body.innerText.includes(`browser · ${b}`),
        backend,
        { timeout: 120_000 },
      );
      const timing = (await p.locator('.timing').first().innerText()).replace(/\s+/g, ' ');
      console.log(`  in the browser, ${backend}: ready in ${Date.now() - t0} ms; ${timing}`);
      await shot(p, `desktop-playground-${backend.toLowerCase()}`);
    }
  }
} finally {
  await browser.close();
  await server.close();
}
console.log(`${n} shots -> ${out}`);
if (problems.length) {
  console.log(`${problems.length} problems:`);
  for (const p of problems) console.log(`  ${p}`);
  process.exit(1);
}
console.log('no page errors, console errors or failed API calls');
