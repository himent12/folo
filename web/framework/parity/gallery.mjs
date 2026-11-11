// The ui gallery's parity check (check.mjs's method, per section): every `gallery/sections/<Name>` is rendered by
// Svelte (gallery-svelte.html, `<Name>.svelte`) and by fw (gallery-fw.html, `<Name>.tsx`), alone on the page
// (`?s=<Name>`), and the two must be the same, pixel for pixel (full page, 1440x900) and in normalized DOM (#app's, as
// check.mjs normalizes it), in dark and light, in every variant and step the section asks for:
//
//   - variants: `desktop` always; `coarse` (a touch screen, `pointer: coarse`) when the section's markup has an element
//     with `data-parity-coarse`; `phone` (390x844, touch) with `data-parity-phone`.
//   - steps: `page` (as loaded), then one per distinct `data-parity-step="<name>"` in the markup: the page is loaded
//     afresh and each element carrying that step, in document order, gets its action: `data-parity-do` = `click`
//     (default), `hover`, `focus`, `press:<Key>` (focus, then the key) or `type:<text>`; the action's target is the
//     element itself, or `data-parity-target` (a selector inside it, or `:document <selector>` for one anywhere).
//     `data-parity-wait="<ms>"` waits that long after the step's actions (default 250; a tooltip's delay is 450).
//
//   node framework/parity/gallery.mjs                     # builds both, checks every section (exit 1 on a difference)
//   node framework/parity/gallery.mjs --only Field,Input  # some sections (comma-separated)
//   node framework/parity/gallery.mjs --no-build          # checks the last build
//   node framework/parity/gallery.mjs --out <dir>         # build and report there (default framework/parity/dist-gallery)
//   node framework/parity/gallery.mjs --keep              # also keep the screenshots of shots that match
//
// Writes <out>/gallery.md (every shot, with its first DOM difference) and, for each difference, the two screenshots,
// the diff image and the two DOMs.

import { execFileSync } from 'node:child_process';
import { createReadStream, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { PNG } from 'pngjs';

const here = fileURLToPath(new URL('.', import.meta.url));
const web = join(here, '..', '..');
const arg = (name) => {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : undefined;
};
const dist = resolve(arg('--out') ?? join(here, 'dist-gallery'));
const keep = process.argv.includes('--keep');
const only = arg('--only')?.split(',').map((s) => s.trim()).filter(Boolean);

if (!process.argv.includes('--no-build')) {
  execFileSync(process.execPath, [join(web, 'node_modules', 'vite', 'bin', 'vite.js'), 'build', '-c', join(here, 'gallery.vite.config.ts'), '--logLevel', 'warn'], {
    cwd: web,
    stdio: 'inherit',
    env: { ...process.env, GALLERY_OUT: dist },
  });
}
const shots = join(dist, 'shots');
mkdirSync(shots, { recursive: true });

const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.woff2': 'font/woff2', '.svg': 'image/svg+xml', '.map': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg' };
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

/**
 * The DOM of #app, normalized as check.mjs does (no comments, no whitespace-only text, no scope classes, attributes
 * sorted), and with adjacent text nodes merged: `a{' '}{b}` is three nodes in fw and one in Svelte, the same outerHTML.
 */
function normalizeDom() {
  const scope = /^(svelte|fw)-[a-z0-9]+$/;
  const out = [];
  const walk = (node, depth) => {
    let text = null;
    const flush = () => {
      if (text !== null && text.trim() !== '') out.push(`${'  '.repeat(depth)}"${text.replace(/\s+/g, ' ')}"`);
      text = null;
    };
    for (const n of node.childNodes) {
      if (n.nodeType === 8) continue;
      if (n.nodeType === 3) {
        text = (text ?? '') + n.data;
        continue;
      }
      flush();
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
    flush();
  };
  walk(document.getElementById('app'), 0);
  return out.join('\n');
}

/** The section's variants and steps, read from its markup. */
function readPlan() {
  const steps = [];
  for (const el of document.querySelectorAll('[data-parity-step]')) {
    const s = el.getAttribute('data-parity-step');
    if (!steps.includes(s)) steps.push(s);
  }
  return { coarse: !!document.querySelector('[data-parity-coarse]'), phone: !!document.querySelector('[data-parity-phone]'), steps };
}

/** The actions of `step`, in document order: { do, target: [x, y] center or null, index }. */
function readActions(step) {
  const out = [];
  for (const el of document.querySelectorAll(`[data-parity-step="${CSS.escape(step)}"]`)) {
    const sel = el.getAttribute('data-parity-target');
    const target = !sel ? el : sel.startsWith(':document ') ? document.querySelector(sel.slice(10)) : el.querySelector(sel);
    out.push({ do: el.getAttribute('data-parity-do') ?? 'click', found: !!target, wait: Number(el.getAttribute('data-parity-wait') ?? 0) });
  }
  return out;
}

/** Runs action `i` of `step` inside the page for focus / press / type (needs the element); returns its center for pointer actions. */
function locate([step, i, focus]) {
  const el = document.querySelectorAll(`[data-parity-step="${CSS.escape(step)}"]`)[i];
  const sel = el.getAttribute('data-parity-target');
  const target = !sel ? el : sel.startsWith(':document ') ? document.querySelector(sel.slice(10)) : el.querySelector(sel);
  if (!target) return null;
  if (focus) {
    target.focus();
    return null;
  }
  target.scrollIntoView({ block: 'center' });
  const r = target.getBoundingClientRect();
  return [r.left + r.width / 2, r.top + r.height / 2];
}

const variants = {
  desktop: { viewport: { width: 1440, height: 900 } },
  coarse: { viewport: { width: 1440, height: 900 }, hasTouch: true },
  phone: { viewport: { width: 390, height: 844 }, hasTouch: true },
};

async function capture(context, which, section, step) {
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(`console: ${m.text()}`);
  });
  await page.goto(`${origin}/framework/parity/gallery-${which}.html?s=${encodeURIComponent(section)}`);
  await page.evaluate(() => document.fonts.ready);
  // Let observers (bind:clientWidth's ResizeObserver) and the effects they wake settle before anything is read.
  await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  await page.waitForTimeout(100);
  const plan = await page.evaluate(readPlan);
  if (step !== 'page') {
    const actions = await page.evaluate(readActions, step);
    for (let i = 0; i < actions.length; i++) {
      const a = actions[i];
      if (!a.found) {
        errors.push(`step ${step}: action ${i} has no target`);
        continue;
      }
      if (a.do === 'click' || a.do === 'hover') {
        const at = await page.evaluate(locate, [step, i, false]);
        if (a.do === 'click') await page.mouse.click(at[0], at[1]);
        else await page.mouse.move(at[0], at[1]);
      } else if (a.do === 'focus' || a.do.startsWith('press:') || a.do.startsWith('type:')) {
        // Keyboard modality first, so the focus ring shows as it does for a keyboard user.
        await page.keyboard.press('Shift');
        await page.evaluate(locate, [step, i, true]);
        if (a.do.startsWith('press:')) await page.keyboard.press(a.do.slice(6));
        if (a.do.startsWith('type:')) await page.keyboard.type(a.do.slice(5));
      } else errors.push(`step ${step}: unknown action ${a.do}`);
      await page.waitForTimeout(60);
    }
    await page.waitForTimeout(Math.max(250, ...actions.map((a) => a.wait || 0)));
  }
  const png = await page.screenshot({ fullPage: true, animations: 'disabled', caret: 'hide' });
  const dom = await page.evaluate(normalizeDom);
  await page.close();
  return { png: PNG.sync.read(png), dom, plan, errors };
}

function firstDiff(a, b) {
  const x = a.split('\n');
  const y = b.split('\n');
  for (let i = 0; i < Math.max(x.length, y.length); i++) {
    if (x[i] !== y[i]) return `line ${i + 1}: svelte \`${(x[i] ?? '(end)').trim().slice(0, 160)}\` / fw \`${(y[i] ?? '(end)').trim().slice(0, 160)}\``;
  }
  return '';
}

function compare(a, b) {
  if (a.width !== b.width || a.height !== b.height) return { diff: -1, image: null };
  const out = new PNG({ width: a.width, height: a.height });
  let diff = 0;
  for (let i = 0; i < a.data.length; i += 4) {
    const same = a.data[i] === b.data[i] && a.data[i + 1] === b.data[i + 1] && a.data[i + 2] === b.data[i + 2] && a.data[i + 3] === b.data[i + 3];
    if (!same) diff++;
    out.data[i] = same ? a.data[i] >> 2 : 255;
    out.data[i + 1] = same ? a.data[i + 1] >> 2 : 0;
    out.data[i + 2] = same ? a.data[i + 2] >> 2 : 0;
    out.data[i + 3] = 255;
  }
  return { diff, image: out };
}

// The section list, from the Svelte page (every section file has its twin, which the fw page's list must match).
const lists = {};
for (const which of ['svelte', 'fw']) {
  const page = await browser.newPage();
  await page.goto(`${origin}/framework/parity/gallery-${which}.html?s=-`);
  lists[which] = await page.evaluate(() => window.__gallery);
  await page.close();
}
let failed = false;
const report = ['# ui gallery parity', '', '| section | theme | variant | step | size | differing pixels | DOM |', '|---|---|---|---|---|---:|---|'];
const missing = [...lists.svelte.filter((s) => !lists.fw.includes(s)).map((s) => `${s}.tsx`), ...lists.fw.filter((s) => !lists.svelte.includes(s)).map((s) => `${s}.svelte`)];
if (missing.length) {
  console.error(`sections without a twin: ${missing.join(', ')}`);
  failed = true;
}
const sections = lists.svelte.filter((s) => lists.fw.includes(s) && (!only || only.includes(s)));
if (only) for (const s of only) if (!sections.includes(s)) console.error(`no section ${s}`), (failed = true);

const contexts = new Map();
async function contextFor(theme, variant) {
  const key = `${theme}/${variant}`;
  if (!contexts.has(key)) {
    const context = await browser.newContext({ ...variants[variant], deviceScaleFactor: 1, reducedMotion: 'reduce' });
    await context.addInitScript((t) => {
      localStorage.clear();
      localStorage.setItem('folo.theme', t);
    }, theme);
    contexts.set(key, context);
  }
  return contexts.get(key);
}

const notes = [];
for (const section of sections) {
  let plan = null;
  for (const theme of ['dark', 'light']) {
    for (const variant of ['desktop', 'coarse', 'phone']) {
      if (variant !== 'desktop' && plan && !plan[variant]) continue;
      const context = await contextFor(theme, variant);
      const steps = ['page', ...(plan?.steps ?? [])];
      for (let si = 0; si < steps.length; si++) {
        const step = steps[si];
        const got = {};
        for (const which of ['svelte', 'fw']) got[which] = await capture(context, which, section, step);
        if (!plan) {
          plan = got.svelte.plan;
          steps.push(...plan.steps);
          if (variant === 'desktop' && JSON.stringify(plan) !== JSON.stringify(got.fw.plan)) notes.push(`${section}: the fw twin's plan differs (${JSON.stringify(got.fw.plan)} vs ${JSON.stringify(plan)})`);
        }
        const tag = `${section}-${theme}-${variant}-${step}`;
        for (const which of ['svelte', 'fw']) {
          if (got[which].errors.length) {
            notes.push(`${tag} ${which}: ${got[which].errors.join('; ')}`);
            failed = true;
          }
        }
        const { diff, image } = compare(got.svelte.png, got.fw.png);
        const domSame = got.svelte.dom === got.fw.dom;
        const size = `${got.svelte.png.width}x${got.svelte.png.height}` + (diff < 0 ? ` / fw ${got.fw.png.width}x${got.fw.png.height}` : '');
        if (diff !== 0 || !domSame) failed = true;
        if (diff !== 0 || !domSame || keep) {
          writeFileSync(join(shots, `${tag}-svelte.png`), PNG.sync.write(got.svelte.png));
          writeFileSync(join(shots, `${tag}-fw.png`), PNG.sync.write(got.fw.png));
          if (image) writeFileSync(join(shots, `${tag}-diff.png`), PNG.sync.write(image));
          if (!domSame) {
            writeFileSync(join(shots, `${tag}-dom-svelte.txt`), got.svelte.dom);
            writeFileSync(join(shots, `${tag}-dom-fw.txt`), got.fw.dom);
          }
        }
        const domText = domSame ? 'same' : `DIFFERENT: ${firstDiff(got.svelte.dom, got.fw.dom)}`;
        report.push(`| ${section} | ${theme} | ${variant} | ${step} | ${size} | ${diff < 0 ? 'size differs' : diff} | ${domText.replace(/\|/g, '\\|')} |`);
        console.log(`${section.padEnd(16)} ${theme.padEnd(5)} ${variant.padEnd(7)} ${step.padEnd(10)} ${size.padEnd(11)} differing pixels: ${diff < 0 ? 'size differs' : diff}  DOM: ${domSame ? 'same' : 'DIFFERENT'}`);
      }
    }
  }
}
for (const c of contexts.values()) await c.close();
await browser.close();
server.close();

if (notes.length) report.push('', '## Notes', '', ...notes.map((n) => `- ${n}`));
report.push('', failed ? '**FAILED**' : `**OK**: ${sections.length} sections, 0 differing pixels, same DOM.`);
writeFileSync(join(dist, 'gallery.md'), report.join('\n') + '\n');
for (const n of notes) console.error(n);
console.log(`report: ${join(dist, 'gallery.md')}`);
if (failed) {
  console.error('gallery parity: FAILED');
  process.exit(1);
}
console.log(`gallery parity: OK (${sections.length} sections, 0 differing pixels, same DOM)`);
