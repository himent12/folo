// Visits every Studio route against a running FOLO server and reports page errors, console errors and failed API
// calls. Use it to check the client against the real contract:
//   folo serve --mock --port 7979 &
//   FOLO_API=http://127.0.0.1:7979 npm run smoke
import { createServer } from 'vite';
import { chromium } from 'playwright-core';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const api = process.env.FOLO_API ?? 'http://127.0.0.1:7878';

const server = await createServer({ root, logLevel: 'error', server: { host: '127.0.0.1', port: 5197, strictPort: false } });
await server.listen();
const base = (server.resolvedUrls?.local[0] ?? 'http://127.0.0.1:5197/').replace(/\/$/, '');

const res = await fetch(`${api}/api/runs`).then((r) => r.json());
const ds = await fetch(`${api}/api/datasets`).then((r) => r.json());
const models = await fetch(`${api}/api/models`).then((r) => r.json());
const runIds = res.runs.map((/** @type {{id: string}} */ r) => r.id);
const dsId = ds.datasets[0]?.id;
const modelId = models.models[0]?.id;

/** Optional screenshots of each route (SMOKE_SHOTS=dir). */
const shots = process.env.SMOKE_SHOTS;
const paths = [
  '/',
  '/datasets',
  ...ds.datasets.map((/** @type {{id: string}} */ d) => `/datasets/${d.id}?tab=stats`),
  `/datasets/${dsId}?tab=images`,
  '/runs',
  ...runIds.map((/** @type {string} */ id) => `/runs/${id}`),
  '/runs/new',
  `/runs/compare?ids=${runIds.slice(0, 3).join(',')}`,
  '/models',
  ...models.models.map((/** @type {{id: string}} */ m) => `/models/${m.id}`),
  `/playground?model=${modelId}`,
  '/settings',
];

const browser = await chromium.launch({ channel: 'chrome', headless: true });
const problems = [];
try {
  for (const path of paths) {
    const page = await browser.newPage();
    page.on('pageerror', (e) => problems.push(`${path}: page error: ${e.message}`));
    page.on('console', (m) => {
      if (m.type() === 'error') problems.push(`${path}: console: ${m.text()}`);
    });
    page.on('response', (r) => {
      if (r.url().includes('/api/') && r.status() >= 400) problems.push(`${path}: ${r.status()} ${r.request().method()} ${r.url().replace(base, '')}`);
    });
    await page.goto(`${base}/#${path}`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1500);
    const mode = await page.evaluate(() => document.querySelector('.conn')?.textContent?.trim());
    if (!mode?.startsWith('Connected')) problems.push(`${path}: not connected to the server (status: ${mode})`);
    const errs = await page.locator('[role=alert]').allTextContents();
    // A failed run's banner is an alert by design.
    for (const e of errs.filter((t) => !/^\s*Failed at epoch/.test(t))) problems.push(`${path}: alert: ${e.replace(/\s+/g, ' ').trim().slice(0, 160)}`);
    if (shots) await page.screenshot({ path: resolve(shots, `${path.replace(/[^\w]+/g, '_') || 'root'}.png`), fullPage: true });
    await page.close();
  }
} finally {
  await browser.close();
  await server.close();
}

console.log(`${paths.length} routes checked against ${api}`);
if (problems.length) {
  console.log(`${problems.length} problem(s):\n  ${[...new Set(problems)].join('\n  ')}`);
  process.exitCode = 1;
} else console.log('no problems');
