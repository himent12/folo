// The heartbeat pause (src/lib/ui/heartbeat.ts) in a real browser: every frame shows what the animation shows unpaused,
// also when its timers run late.
//   node scripts/heartbeat-check.mjs [--cycles 4]
// A page (Vite's dev server, so heartbeat.ts as it is in the tree) holds two dots with the same CSS heartbeat, the
// keyframes of base.css's `live-pulse` on a 1.2 s period: one under heartbeat.ts, one a twin it does not track (another
// keyframes name), both started together. Every frame (requestAnimationFrame) the two dots' computed box-shadow are
// compared. Each case delays every timer of the page by a fixed amount (as a busy main thread or a throttled tab does):
// - 0 ms and 180 ms (a late timer within RESUME_EARLY, 250 ms): no frame may differ, and the tracked dot must have
//   been paused for some frames (else the check proves nothing);
// - 450 ms (later than RESUME_EARLY): frames must differ, so the comparison is shown to see a clipped beat.
// Exits 1 when a case does not hold (`npm run qa`, so CI: its first shard).
// Uses the installed Chrome (playwright-core, channel "chrome"); without one, Playwright's Chromium.
import { createServer } from 'vite';
import { chromium } from 'playwright-core';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const cycles = Math.max(2, Number(args[args.indexOf('--cycles') + 1]) || 4);

const shard = process.env.FOLO_QA_SHARD;
if (shard && !shard.startsWith('1/')) {
  console.log(`Heartbeat check: run by shard 1, not ${shard}.`);
  process.exit(0);
}

const PERIOD = 1200;
const FRAMES = `{ 0% { box-shadow: 0 0 0 0 rgba(190, 242, 100, 0.55); } 33%, 100% { box-shadow: 0 0 0 4px transparent; } }`;
const PAGE = `<!doctype html><meta charset="utf-8"><style>
@keyframes live-pulse ${FRAMES}
@keyframes twin-pulse ${FRAMES}
.dot { width: 8px; height: 8px; margin: 20px; border-radius: 50%; background: #bef264; }
#beat { animation: live-pulse ${PERIOD}ms cubic-bezier(0.16, 1, 0.3, 1) infinite; }
#twin { animation: twin-pulse ${PERIOD}ms cubic-bezier(0.16, 1, 0.3, 1) infinite; }
</style><body><script type="module">
const delay = Number(new URLSearchParams(location.search).get('delay'));
const later = window.setTimeout.bind(window);
window.setTimeout = (f, ms, ...rest) => later(f, (ms ?? 0) + delay, ...rest);
const { installHeartbeat } = await import('/src/lib/ui/heartbeat.ts');
installHeartbeat();
for (const id of ['beat', 'twin']) document.body.insertAdjacentHTML('beforeend', '<div class="dot" id="' + id + '"></div>');
const [beat, twin] = ['beat', 'twin'].map((id) => document.getElementById(id));
await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
// Started together (the same frame); made exact.
twin.getAnimations()[0].startTime = beat.getAnimations()[0].startTime;
window.sample = (ms) => new Promise((done) => {
  const out = { frames: 0, differ: 0, paused: 0, first: null };
  const end = performance.now() + ms;
  const frame = () => {
    const a = getComputedStyle(beat).boxShadow;
    const b = getComputedStyle(twin).boxShadow;
    out.frames++;
    if (beat.getAnimations()[0]?.playState === 'paused') out.paused++;
    if (a !== b) {
      out.differ++;
      out.first ??= { a, b, t: Math.round(document.timeline.currentTime - twin.getAnimations()[0].startTime) % ${PERIOD} };
    }
    if (performance.now() < end) requestAnimationFrame(frame);
    else done(out);
  };
  requestAnimationFrame(frame);
});
window.ready = true;
</script>`;

const server = await createServer({ root, logLevel: 'error', server: { host: '127.0.0.1', port: 5195, strictPort: false } });
await server.listen();
const base = (server.resolvedUrls?.local[0] ?? 'http://127.0.0.1:5195/').replace(/\/$/, '');

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

const browser = await launch();
/** @type {string[]} */
const problems = [];
try {
  for (const [delay, clipped] of /** @type {const} */ ([
    [0, false],
    [180, false],
    [450, true],
  ])) {
    const page = await browser.newPage();
    await page.route(`${base}/heartbeat-check.html*`, (r) => r.fulfill({ contentType: 'text/html', body: PAGE }));
    await page.goto(`${base}/heartbeat-check.html?delay=${delay}`);
    await page.waitForFunction(() => /** @type {any} */ (window).ready === true, null, { timeout: 30_000 });
    const s = await page.evaluate((ms) => /** @type {any} */ (window).sample(ms), cycles * PERIOD);
    const line = `timers ${delay} ms late: ${s.frames} frames, ${s.paused} with the heartbeat paused, ${s.differ} differing${s.first ? ` (first at ${s.first.t} ms into a beat: ${s.first.a} vs ${s.first.b})` : ''}`;
    console.log(line);
    if (!clipped && s.differ) problems.push(`${line}: no frame may differ`);
    if (!clipped && !s.paused) problems.push(`${line}: the heartbeat was never paused, so this proves nothing`);
    if (clipped && !s.differ) problems.push(`${line}: a resume this late must clip the beat; the comparison saw nothing`);
    await page.close();
  }
} finally {
  await browser.close();
  await server.close();
}
if (problems.length) {
  console.error(`HEARTBEAT: ${problems.length} problem(s)\n${problems.join('\n')}`);
  process.exitCode = 1;
} else console.log('Heartbeat check: every frame as unpaused while its timers ran up to 180 ms late.');
