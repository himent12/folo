// Parity check: the proof page rendered by fw (fw.html, Demo.tsx) and by Svelte (svelte.html, Demo.svelte) must be the
// same, pixel for pixel, at 1440x900 in dark and light, closed and with a menu open; and their DOM the same once scope
// classes, comments and whitespace-only text are set aside.
//
//   node framework/parity/check.mjs            # builds both, then checks (exit 1 on any difference)
//   node framework/parity/check.mjs --no-build # checks the last build
//
// Uses the installed Chrome (playwright-core, channel "chrome"). Diff images go to framework/parity/dist/diff-*.png.

import { execFileSync } from 'node:child_process';
import { createReadStream, existsSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { PNG } from 'pngjs';

const here = fileURLToPath(new URL('.', import.meta.url));
const web = join(here, '..', '..');
const dist = join(here, 'dist');

if (!process.argv.includes('--no-build')) {
  execFileSync(process.execPath, [join(web, 'node_modules', 'vite', 'bin', 'vite.js'), 'build', '-c', join(here, 'vite.config.ts'), '--logLevel', 'warn'], { cwd: web, stdio: 'inherit' });
}

const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.woff2': 'font/woff2', '.svg': 'image/svg+xml', '.map': 'application/json' };
const server = createServer((req, res) => {
  const path = normalize(join(dist, decodeURIComponent(new URL(req.url ?? '/', 'http://x').pathname)));
  if (!path.startsWith(dist) || !existsSync(path)) {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, { 'content-type': types[extname(path)] ?? 'application/octet-stream' });
  createReadStream(path).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const origin = `http://127.0.0.1:${server.address().port}`;

const args = ['--disable-gpu', '--disable-lcd-text', '--disable-partial-raster', '--force-color-profile=srgb', '--font-render-hinting=none'];
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true, args });
} catch {
  browser = await chromium.launch({ headless: true, args });
}

/** The DOM of #app, normalized: no comments, no whitespace-only text, no scope classes, attributes sorted. */
function normalizeDom() {
  const scope = /^(svelte|fw)-[a-z0-9]+$/;
  const out = [];
  const walk = (node, depth) => {
    for (const n of node.childNodes) {
      if (n.nodeType === 8) continue;
      if (n.nodeType === 3) {
        if (n.data.trim() === '') continue;
        out.push(`${'  '.repeat(depth)}"${n.data.replace(/\s+/g, ' ')}"`);
        continue;
      }
      if (n.nodeType !== 1) continue;
      const attrs = [];
      for (const a of n.attributes) {
        if (a.name === 'class') {
          const c = a.value.split(/\s+/).filter((x) => x && !scope.test(x)).sort().join(' ');
          if (c) attrs.push(`class="${c}"`);
        } else if (a.name === 'style') {
          const s = a.value.split(';').map((x) => x.trim()).filter(Boolean).sort().join('; ');
          if (s) attrs.push(`style="${s}"`);
        } else attrs.push(`${a.name}="${a.value}"`);
      }
      attrs.sort();
      out.push(`${'  '.repeat(depth)}<${n.localName}${attrs.length ? ' ' + attrs.join(' ') : ''}>`);
      walk(n, depth + 1);
    }
  };
  walk(document.getElementById('app'), 0);
  return out.join('\n');
}

async function capture(page, theme, which, step) {
  await page.goto(`${origin}/framework/parity/${which}.html`);
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(50);
  if (step === 'menu') {
    await page.click('button[aria-haspopup="menu"]');
    await page.waitForSelector('[role="menu"]');
    await page.waitForTimeout(250);
  }
  const png = await page.screenshot({ fullPage: true, animations: 'disabled', caret: 'hide' });
  const dom = await page.evaluate(normalizeDom);
  return { png: PNG.sync.read(png), dom };
}

let failed = false;
for (const theme of ['dark', 'light']) {
  for (const step of ['page', 'menu']) {
    const shots = {};
    for (const which of ['svelte', 'fw']) {
      const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1, reducedMotion: 'reduce' });
      await context.addInitScript((t) => localStorage.setItem('folo.theme', t), theme);
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', (e) => errors.push(String(e)));
      shots[which] = await capture(page, theme, which, step);
      if (errors.length) {
        console.error(`${which} ${theme} ${step}: page errors:\n  ${errors.join('\n  ')}`);
        failed = true;
      }
      await context.close();
    }
    const a = shots.svelte.png;
    const b = shots.fw.png;
    let diff = 0;
    if (a.width !== b.width || a.height !== b.height) {
      console.error(`${theme} ${step}: sizes differ (svelte ${a.width}x${a.height}, fw ${b.width}x${b.height})`);
      failed = true;
    } else {
      const out = new PNG({ width: a.width, height: a.height });
      for (let i = 0; i < a.data.length; i += 4) {
        const same = a.data[i] === b.data[i] && a.data[i + 1] === b.data[i + 1] && a.data[i + 2] === b.data[i + 2] && a.data[i + 3] === b.data[i + 3];
        if (!same) diff++;
        out.data[i] = same ? a.data[i] >> 2 : 255;
        out.data[i + 1] = same ? a.data[i + 1] >> 2 : 0;
        out.data[i + 2] = same ? a.data[i + 2] >> 2 : 0;
        out.data[i + 3] = 255;
      }
      if (diff) {
        writeFileSync(join(dist, `diff-${theme}-${step}.png`), PNG.sync.write(out));
        writeFileSync(join(dist, `svelte-${theme}-${step}.png`), PNG.sync.write(a));
        writeFileSync(join(dist, `fw-${theme}-${step}.png`), PNG.sync.write(b));
        failed = true;
      }
    }
    const domSame = shots.svelte.dom === shots.fw.dom;
    if (!domSame) {
      writeFileSync(join(dist, `dom-svelte-${theme}-${step}.txt`), shots.svelte.dom);
      writeFileSync(join(dist, `dom-fw-${theme}-${step}.txt`), shots.fw.dom);
      failed = true;
    }
    console.log(`${theme.padEnd(5)} ${step.padEnd(4)}  ${a.width}x${a.height}  differing pixels: ${diff}  DOM: ${domSame ? 'same' : 'DIFFERENT (dist/dom-*.txt)'}`);
  }
}

await browser.close();
server.close();
if (failed) {
  console.error('parity: FAILED');
  process.exit(1);
}
console.log('parity: OK (0 differing pixels, same DOM)');
