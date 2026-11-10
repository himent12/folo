// Live smoke test of agent labeling against a real `folo serve` and a real agent CLI: creates a labeling job through
// Studio, uploads images, has the agent label them, edits one image by hand, asks the agent again about another,
// makes the dataset and opens it; reports page errors, console errors and failed API calls, and what the agent
// reported using. It spends tokens of the account the CLI is signed in with (a few thousand for a handful of images):
//   folo serve --data-dir D:/folo-data/label-smoke --port 7979
//   FOLO_API=http://127.0.0.1:7979 npm run label-smoke -- <out dir> --images <folder of images> \
//        [--agent claude|codex] [--classes person,car] [--count 6]
import { mkdirSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { createServer } from 'vite';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const api = process.env.FOLO_API ?? 'http://127.0.0.1:7878';
const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : dflt;
};
const out = resolve(root, args.find((a, i) => !a.startsWith('--') && !args[i - 1]?.startsWith('--')) ?? 'shots/label-live');
const agent = opt('agent', 'claude');
const agentName = { claude: 'Claude Code', codex: 'Codex' }[agent];
const classes = opt('classes', 'person,car').split(',');
const folder = opt('images', null);
if (!folder || !agentName) throw new Error('usage: npm run label-smoke -- <out dir> --images <folder> [--agent claude|codex] [--classes a,b] [--count 6]');
const files = readdirSync(folder)
  .filter((f) => /\.(jpe?g|png|bmp|webp)$/i.test(f))
  .slice(0, Number(opt('count', 6)))
  .map((f) => join(folder, f));
if (files.length < 2) throw new Error(`${folder} needs at least 2 images`);
mkdirSync(out, { recursive: true });

const found = (await fetch(`${api}/api/label/agents`).then((r) => r.json())).agents.find((a) => a.id === agent);
if (!found?.available) throw new Error(`${agentName} is not available on ${api}: ${found?.hint ?? 'the server has no agent labeling'}`);

const server = await createServer({ root, logLevel: 'error', server: { host: '127.0.0.1', port: 5197, strictPort: false } });
await server.listen();
const base = (server.resolvedUrls?.local[0] ?? 'http://127.0.0.1:5197/').replace(/\/$/, '');
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const problems = [];
let n = 0;
const shot = (page, name) => page.screenshot({ path: resolve(out, `${String(++n).padStart(2, '0')}-${name}.png`) });
const job = (id) => fetch(`${api}/api/label/jobs/${id}`).then((r) => r.json());
/** The `lb_...` or `ds_...` id in the page's URL. */
const idIn = (page, prefix) => new RegExp(`${prefix}_[0-9a-f]+`).exec(page.url())?.[0] ?? '';
/** Waits for the job's agent run to end and returns the job. */
async function idle(page, id, timeoutS = 300) {
  for (const t0 = Date.now(); Date.now() - t0 < timeoutS * 1000; await page.waitForTimeout(500)) {
    const j = await job(id);
    if (j.status === 'idle') return j;
  }
  throw new Error(`the run of ${id} did not end within ${timeoutS} s`);
}

try {
  const page = await (await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: 'dark' })).newPage();
  page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
  page.on('console', (m) => m.type() === 'error' && problems.push(`console: ${m.text()}`));
  page.on('response', (r) => {
    if (r.url().includes('/api/') && r.status() >= 400) problems.push(`${r.status()} ${r.request().method()} ${r.url().replace(base, '')}`);
  });

  // A new job, from the list.
  await page.goto(`${base}/#/datasets/label`, { waitUntil: 'networkidle' });
  await page.getByRole('button', { name: 'New labeling job' }).first().click();
  const dialog = page.locator('dialog[open]');
  await dialog.getByLabel('Name').fill(`label-smoke-${agent}`);
  for (const c of classes) {
    await dialog.getByLabel('Classes').fill(c);
    await dialog.getByLabel('Classes').press('Enter');
  }
  await dialog.getByLabel('Instructions for the agent').fill('Everyday photos. Label every instance, also partly hidden ones.');
  await shot(page, 'new-job');
  await dialog.getByRole('button', { name: 'Create job' }).click();
  await page.waitForURL(/#\/datasets\/label\/lb_/, { timeout: 15_000 });
  const id = idIn(page, 'lb');
  console.log(`job: ${id} (${files.length} images, classes ${classes.join(', ')})`);

  // Uploads through the page's file input.
  await page.locator('input[type=file][multiple]').setInputFiles(files);
  await page.waitForFunction((count) => document.querySelectorAll('[role=gridcell]').length >= count, files.length, { timeout: 60_000 });
  await shot(page, 'uploaded');

  // The agent: picked in the panel, started with the panel's one primary action.
  await page.getByRole('radio', { name: agentName }).click();
  await page.getByRole('button', { name: new RegExp(`^Label ${files.length} images$`) }).click();
  await page.getByRole('button', { name: 'Stop' }).waitFor({ timeout: 15_000 });
  await shot(page, 'running');
  const t0 = Date.now();
  let done = await idle(page, id);
  const u = done.run.usage;
  console.log(
    `${agentName} (${done.run.model ?? 'its default model'}, ${done.run.effort}): ${done.run.outcome} in ${Math.round((Date.now() - t0) / 1000)} s; ` +
      `${done.counts.labeled} labeled, ${done.counts.failed} failed, ${done.counts.boxes} boxes; ` +
      `${u.calls} calls, ${u.input_tokens} tokens in (${u.cached_input_tokens} cached), ${u.output_tokens} out` +
      (u.cost_usd === null ? '' : `, $${u.cost_usd.toFixed(4)}`),
  );
  if (done.run.outcome !== 'completed') problems.push(`the run ended ${done.run.outcome}: ${done.run.error}`);
  if (done.counts.labeled !== files.length) problems.push(`${done.counts.labeled} of ${files.length} images were labeled`);
  await page.getByRole('button', { name: 'Label all again' }).waitFor({ timeout: 15_000 });
  await shot(page, 'labeled');

  // The editor: remove one box of the first image by hand and save it.
  const images = (await fetch(`${api}/api/label/jobs/${id}/images`).then((r) => r.json())).images;
  const boxed = images.find((i) => i.boxes.length > 0);
  if (!boxed) problems.push('the agent drew no box at all');
  else {
    await page.goto(`${base}/#/datasets/label/${id}?image=${boxed.id}`, { waitUntil: 'networkidle' });
    const editor = page.locator('dialog[open]').first();
    await editor.locator('ul.boxes button').first().click();
    await shot(page, 'editor');
    await editor.getByRole('button', { name: 'Delete box' }).click();
    await editor.getByRole('button', { name: 'Save', exact: true }).click();
    await page.waitForFunction(
      async ({ url, want }) => (await fetch(url).then((r) => r.json())).images.some((i) => i.source === 'manual' && i.boxes.length === want),
      { url: `/api/label/jobs/${id}/images`, want: boxed.boxes.length - 1 },
      { timeout: 15_000 },
    );
    // Ask the agent again about the same image, with a note.
    await editor.getByLabel('Ask the agent again').fill('Keep the boxes tight around the visible part.');
    await editor.getByRole('button', { name: `Ask ${agentName}` }).click();
    await page.waitForTimeout(1500);
    await shot(page, 'asked-again');
    done = await idle(page, id);
    if (done.run.total !== 1 || done.run.outcome !== 'completed') problems.push(`asking again: ${JSON.stringify(done.run)}`);
    await page.keyboard.press('Escape');
  }

  // The dataset.
  await page.goto(`${base}/#/datasets/label/${id}`, { waitUntil: 'networkidle' });
  await page.getByRole('button', { name: 'Create dataset' }).first().click();
  const make = page.locator('dialog[open]');
  await shot(page, 'create-dataset');
  await make.getByRole('button', { name: 'Create dataset' }).click();
  await page.getByRole('button', { name: 'Open dataset' }).click({ timeout: 30_000 });
  await page.waitForURL(/#\/datasets\/ds_/, { timeout: 15_000 });
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(1500);
  await shot(page, 'dataset');
  const ds = await fetch(`${api}/api/datasets/${idIn(page, 'ds')}`).then((r) => r.json());
  console.log(`dataset: ${ds.name} at ${ds.path}: ${ds.splits.train} train, ${ds.splits.val} val, ${ds.stats.boxes} boxes`);
  if (ds.splits.train + ds.splits.val !== files.length) problems.push(`the dataset has ${ds.splits.train + ds.splits.val} of ${files.length} images`);

  // The job on a phone.
  const phone = await (await browser.newContext({ viewport: { width: 390, height: 844 }, colorScheme: 'light' })).newPage();
  phone.on('pageerror', (e) => problems.push(`phone pageerror: ${e.message}`));
  await phone.goto(`${base}/#/datasets/label/${id}`, { waitUntil: 'networkidle' });
  await phone.waitForTimeout(1500);
  await shot(phone, 'phone-job');
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
