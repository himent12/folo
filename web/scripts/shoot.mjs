// Screenshots every screen and important state into web/shots/<round>/.
//   npm run shoot                 -> shots/<timestamp>/
//   npm run shoot -- r3           -> shots/r3/
//   npm run shoot -- r3 --only runs   (scene names containing "runs")
//   npm run qa                    -> the checks below only, no screenshots; exits 1 if they find anything (CI runs it)
// Uses the installed Chrome (playwright-core, channel "chrome"); without one, the Chromium that
// `npx playwright-core install --only-shell chromium` downloads. FOLO_SHOOT_BROWSER=chromium forces that one.
// After each scene's setup, the QA lists every element whose content is wider than its box (and the page wider than the
// viewport), skipping deliberate scrollers, every fact an ellipsis cuts (a `ui/Meta` item or a `data-fact` span inside a
// truncated box, or clipped past its box without one: DESIGN_DIRECTION §2.3, never cut a fact), and every blank section: a selected tab or section picker whose content
// renders nothing, a page with almost nothing under its header, every control another element paints over, and every
// select too narrow for its current value. The list prints at the end and goes to qa.txt.
// Every state is the app at `/?mock=1&<search>#<path>`, at its viewport and clock, so two shoots (of two versions)
// compare state by state (scripts/parity.mjs). --exact is what
// `scripts/parity.mjs` shoots with: the clock paused at FIXED_NOW as for --qa (time moves only by the scenes' runFor),
// fonts loaded, animations frozen at the shot, scenes in parallel, a normalized DOM (`<shot>.dom.txt`) next to each
// shot and `shoot.json` (shots, failures). --out <dir> writes there instead of shots/<round>; --base <url> uses a
// running server; --prod shoots the production build (as --qa does); --jobs N states at once (--qa, --exact);
// --only a,b takes a comma list. --theme <id> shoots in that colour theme only (design/THEMES.md; a built-in id from
// tokens.json): each scene in the theme's mode, if the scene is shot in that mode, named `<scene>--<id>--<viewport>`.
// --qa runs only those checks, where phone layouts break: every scene at the tablet and mobile sizes it lists, in dark
// (or the one theme a scene is made for), and in light too where the page shows a phone section picker (Run, Model).
// Phone states run under a coarse pointer (touch: 44 px targets, 16 px field text), where controls grow and rows that
// fit with a mouse can collide. Each state gets its scene's setup; nothing is captured or saved.
import { createServer, preview } from 'vite';
import { chromium } from 'playwright-core';
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { availableParallelism, EOL } from 'node:os';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { scenes } from './scenes.mjs';
import { quiet, quietWaits } from './scenes/drive.mjs';
import { buildApps } from './entries.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
/** Flags that take a value: their values are not the round's name. */
const VALUED = ['--only', '--out', '--base', '--jobs', '--theme'];
const opt = (/** @type {string} */ flag) => {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
};
const only = opt('--only');
/** `--only a,b`: a state is shot when its name contains any of them. */
const onlyParts = only ? only.split(',').map((s) => s.trim()).filter(Boolean) : [];
const picked = (/** @type {string} */ name) => !onlyParts.length || onlyParts.some((p) => name.includes(p));
const qa = args.includes('--qa');
const exact = args.includes('--exact');
/** `--theme <id>`: the built-in theme to shoot in, with its mode; none shoots each scene's modes in their defaults. */
const themeId = opt('--theme');
const themeMode = (() => {
  if (!themeId) return undefined;
  const t = JSON.parse(readFileSync(resolve(root, '..', 'design', 'tokens.json'), 'utf8')).themes?.[themeId];
  if (!t || themeId.startsWith('$')) throw new Error(`--theme ${themeId}: not a built-in theme of design/tokens.json`);
  return /** @type {'dark' | 'light'} */ (t.mode);
})();
/** The app's page. */
const ENTRY = '/';
/**
 * CSS scope classes the DOM dumps leave out: fw's (FOLO_SCOPE_PREFIX overrides it), and the Svelte app's, so the stored
 * shoots of the Svelte app (D:/folo-build/ui-fast-runtime/baseline) stay comparable.
 */
const SCOPE_PREFIXES = ['svelte-', process.env.FOLO_SCOPE_PREFIX ?? 'fw-'];
const round =
  args.find((a, i) => !a.startsWith('--') && !VALUED.includes(args[i - 1])) ?? new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
const outDir = opt('--out') ? resolve(/** @type {string} */ (opt('--out'))) : resolve(root, 'shots', round);
/** The clock paused at FIXED_NOW (time moves only by the scenes' runFor), the same on every machine and every run. */
const paused = qa || exact;
const started = Date.now();

const VIEWPORTS = {
  desktop: { width: 1440, height: 900 },
  // The narrowest two-column Overview (Recent runs beside Best models): only the scenes that list it, for --qa's checks.
  wide: { width: 1280, height: 800 },
  laptop: { width: 1024, height: 768 },
  // Below the 960 px laptop tier: one column, the section picker, the New run dock.
  tablet: { width: 820, height: 1180 },
  mobile: { width: 390, height: 844 },
  // The narrowest phone a layout must hold (iPhone SE, 1st gen): only the scenes that list it, for --qa's checks.
  small: { width: 320, height: 640 },
};
// Fixed wall clock so relative times ("3 min ago") and the mock's live run are identical every round.
const FIXED_NOW = new Date('2026-11-08T14:30:00Z');

// The shoot loads pages from the dev server. --qa checks the production build instead (what ships): it is built here
// into node_modules/.qa-dist, never web/dist, and served as static files. Each state opens a fresh browser context
// with an empty cache, and the build is one script where the dev server answers a request per module, so the same
// states take about a quarter of the time.
const listen = { host: '127.0.0.1', port: 5199, strictPort: false };
/** @type {{ resolvedUrls: import('vite').ResolvedServerUrls | null, close(): Promise<void> }} */
let server;
const given = opt('--base');
if (given) {
  server = { resolvedUrls: { local: [given], network: [] }, close: async () => {} };
} else if (qa || args.includes('--prod')) {
  const outDir = resolve(root, qa ? 'node_modules/.qa-dist' : 'node_modules/.shoot-dist');
  await buildApps(root, outDir);
  server = await preview({ root, logLevel: 'error', build: { outDir }, preview: listen });
} else {
  const dev = await createServer({ root, logLevel: 'error', server: listen });
  await dev.listen();
  server = dev;
}
const base = (server.resolvedUrls?.local[0] ?? 'http://127.0.0.1:5199/').replace(/\/$/, '');

if (!qa) {
  if (!only) rmSync(outDir, { recursive: true, force: true });
  mkdirSync(outDir, { recursive: true });
}

/** Waits in real time until every image has decoded, then a beat for layout (and 50 ms of page clock). */
async function settle(page) {
  for (let i = 0; i < 40; i++) {
    const pending = await page.evaluate(() => [...document.images].filter((im) => !im.complete).length);
    if (!pending) break;
    await page.waitForTimeout(100);
  }
  await page.clock.runFor(50);
  await page.waitForTimeout(150);
  // A face a setup brought on screen (a mono figure in a dialog) loads after the first fonts.ready.
  if (paused) await quiet(page);
}

/**
 * --exact and --qa (a paused clock): makes `page` wait for its real-time work before and after every step of its
 * paused clock (and before each look of drive.mjs's waits). With the clock paused, a step fires the mock's timers; what follows them in real time
 * (a canvas encoded to a blob, an image loading or decoding, a lazy chunk, a font) would otherwise land before or after
 * the next step depending on the machine's load, and with it how many steps a wait takes and the live run's progress.
 * Counted in the page: pending blob encodes, bitmaps, image decodes and fetches, images not yet complete, fonts
 * loading, DOM mutations; outside it, requests in flight. Quiet: all zero and no mutation over two 25 ms polls, a frame's
 * ResizeObserver work included (3 s cap).
 * @param {import('playwright-core').Page} page
 */
async function makeQuiet(page) {
  let inflight = 0;
  page.on('request', () => inflight++);
  page.on('requestfinished', () => inflight--);
  page.on('requestfailed', () => inflight--);
  await page.addInitScript(() => {
    let pending = 0;
    let mutations = 0;
    const track = (/** @type {any} */ obj, /** @type {string} */ key) => {
      const f = obj?.[key];
      if (typeof f !== 'function') return;
      obj[key] = function (/** @type {unknown[]} */ ...a) {
        pending++;
        try {
          return Promise.resolve(f.apply(this, a)).finally(() => pending--);
        } catch (e) {
          pending--;
          throw e;
        }
      };
    };
    track(globalThis.OffscreenCanvas?.prototype, 'convertToBlob');
    track(globalThis, 'createImageBitmap');
    track(HTMLImageElement.prototype, 'decode');
    track(globalThis, 'fetch');
    const toBlob = HTMLCanvasElement.prototype.toBlob;
    HTMLCanvasElement.prototype.toBlob = function (cb, ...rest) {
      pending++;
      return toBlob.call(this, (b) => {
        pending--;
        cb(b);
      }, ...rest);
    };
    new MutationObserver(() => mutations++).observe(document, { subtree: true, childList: true, attributes: true, characterData: true });
    Object.defineProperty(globalThis, '__foloQuiet', {
      value: () => [pending + [...document.images].filter((im) => !im.complete).length + (document.fonts.status === 'loading' ? 1 : 0), mutations],
    });
  });
  const wait = async () => {
    let last = -1;
    let still = 0;
    for (const end = Date.now() + 3000; Date.now() < end; ) {
      /** @type {[number, number] | null} */
      const s = await page.evaluate(() => /** @type {any} */ (globalThis).__foloQuiet?.() ?? null).catch(() => null);
      still = s && s[0] === 0 && inflight <= 0 && s[1] === last ? still + 1 : 0;
      if (still >= 2) return;
      last = s ? s[1] : -1;
      await page.waitForTimeout(25);
    }
  };
  quietWaits.set(page, wait);
  const runFor = page.clock.runFor.bind(page.clock);
  // A long step is taken in parts (250 ms, or a 20th of it), quiet between them: the clock yields to the page between
  // timers, so real-time work a timer starts (a stream subscribed once a chunk has loaded) would otherwise land
  // wherever the machine's speed puts it inside the step.
  page.clock.runFor = async (/** @type {number | string} */ ms) => {
    await wait();
    if (typeof ms !== 'number') return runFor(ms).then(wait);
    const part = Math.max(250, Math.ceil(ms / 20));
    for (let left = ms; left > 0; left -= part) {
      await runFor(Math.min(part, left));
      await wait();
    }
  };
}

/**
 * The page's DOM, normalized for comparing two shoots (`<shot>.dom.txt`): one line per element (tag, attributes
 * sorted, classes sorted without the scope classes, style declarations sorted, the live value / checked state of form
 * fields) and per text run (adjacent text nodes joined, whitespace collapsed, blank runs dropped), indented by depth.
 * Comments (markers) are dropped; `blob:` URLs, which are random, read `blob:*`.
 * @param {import('playwright-core').Page} page
 */
async function domDump(page) {
  return page.evaluate((prefixes) => {
    /** @type {string[]} */
    const out = [];
    const q = (/** @type {string} */ v) => JSON.stringify(v.replace(/blob:[^\s"')]+/g, 'blob:*'));
    const walk = (/** @type {Node} */ node, /** @type {number} */ depth) => {
      const pad = '  '.repeat(depth);
      let text = '';
      const flush = () => {
        const t = text.replace(/\s+/g, ' ').trim();
        if (t) out.push(`${pad}${JSON.stringify(t)}`);
        text = '';
      };
      for (const n of node.childNodes) {
        if (n.nodeType === Node.TEXT_NODE) {
          text += n.textContent ?? '';
          continue;
        }
        if (n.nodeType !== Node.ELEMENT_NODE) continue;
        flush();
        const el = /** @type {Element} */ (n);
        const tag = el.tagName.toLowerCase();
        if (tag === 'script') continue;
        /** @type {[string, string][]} */
        const attrs = [];
        for (const a of el.attributes) {
          if (a.name.startsWith('data-svelte')) continue;
          if (a.name === 'class') {
            const cls = [...el.classList].filter((c) => !prefixes.some((p) => c.startsWith(p))).sort();
            if (cls.length) attrs.push(['class', cls.join(' ')]);
          } else if (a.name === 'style') {
            const decl = a.value.split(';').map((d) => d.trim().replace(/\s*:\s*/, ': ')).filter(Boolean).sort();
            if (decl.length) attrs.push(['style', decl.join('; ')]);
          } else attrs.push([a.name, a.value]);
        }
        if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement || el instanceof HTMLSelectElement) attrs.push([':value', el.value]);
        if (el instanceof HTMLInputElement && (el.type === 'checkbox' || el.type === 'radio')) attrs.push([':checked', String(el.checked)]);
        attrs.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
        out.push(`${pad}<${tag}${attrs.map(([k, v]) => ` ${k}=${q(v)}`).join('')}>`);
        walk(el, depth + 1);
      }
      flush();
    };
    walk(document.body, 0);
    return out.join('\n') + '\n';
  }, SCOPE_PREFIXES);
}

/**
 * Elements whose content overflows them horizontally (scrollWidth > clientWidth + 1), innermost only, plus the page
 * itself. Skipped: hidden and visually-hidden elements, ellipsis truncation (it says so with "…"), and deliberate
 * scrollers: `data-scrolls` on the element or an ancestor, or an `overflow-x: auto / scroll` box that shows a designed
 * affordance (a fade mask on it or its parent). A plain scrollbar is not one: every safety-net wrapper shows that, and
 * a table that needs it at a tier it was designed for is the bug this check exists to catch. Also skipped: spill from a
 * positioned descendant or a negative-margin hit area (the crumbs' links); each entry names the widest descendant.
 * @param {import('playwright-core').Page} page
 * @returns {Promise<{ selector: string, px: number, text: string }[]>}
 */
async function overflowCheck(page) {
  return page.evaluate((prefixes) => {
    const out = [];
    const doc = document.documentElement;
    const pageOver = doc.scrollWidth - doc.clientWidth;
    if (pageOver > 1) out.push({ selector: '(page)', px: pageOver, text: '' });
    /** A clip that hides the box (visually hidden text), not one that trims an edge (ui/Meta's leading dot). */
    const clipsAway = (/** @type {Element} */ el, /** @type {CSSStyleDeclaration} */ cs) => {
      const clipped = (cs.clipPath !== 'none' && cs.clipPath !== '') || (cs.clip !== 'auto' && cs.clip !== '');
      if (!clipped) return false;
      const r = el.getBoundingClientRect();
      return r.width <= 1 || r.height <= 1 || /inset\(\s*50%|rect\(\s*0(px)?[ ,]+0(px)?[ ,]+0(px)?[ ,]+0/.test(`${cs.clipPath} ${cs.clip}`);
    };
    const hiddenBySelf = (/** @type {Element} */ el, /** @type {CSSStyleDeclaration} */ cs) =>
      cs.display === 'none' ||
      cs.visibility === 'hidden' ||
      clipsAway(el, cs) ||
      (el.getAttribute('aria-hidden') === 'true' && cs.position === 'absolute');
    const masked = (/** @type {Element | null} */ el) => {
      if (!el) return false;
      const cs = getComputedStyle(el);
      return (cs.maskImage && cs.maskImage !== 'none') || (cs.webkitMaskImage && cs.webkitMaskImage !== 'none');
    };
    const describe = (/** @type {Element} */ el) => {
      const part = (/** @type {Element} */ e) => {
        const cls = [...e.classList].filter((c) => !prefixes.some((p) => c.startsWith(p))).slice(0, 3);
        return `${e.tagName.toLowerCase()}${e.id ? `#${e.id}` : ''}${cls.length ? `.${cls.join('.')}` : ''}`;
      };
      const parts = [];
      for (let e = /** @type {Element | null} */ (el); e && e !== document.body && parts.length < 3; e = e.parentElement) parts.unshift(part(e));
      return parts.join(' > ');
    };
    /** @type {{ el: Element, px: number }[]} */
    const hits = [];
    /** @type {{ el: Element, px: number }[]} */
    const clippedFacts = [];
    // A fact: a meta line's item (counts, units, statuses, identifiers) or a span marked `data-fact`. An ellipsis may
    // shorten a free-text name, never one of these (DESIGN_DIRECTION §2.3).
    const FACT = '.ui-meta, [data-fact]';
    for (const el of document.querySelectorAll('body *')) {
      if (!(el instanceof HTMLElement) || el.clientWidth === 0) continue;
      const px = el.scrollWidth - el.clientWidth;
      if (px <= 1) continue;
      const cs = getComputedStyle(el);
      if (cs.textOverflow === 'ellipsis') {
        if ((el.matches(FACT) || el.querySelector(FACT)) && el.checkVisibility({ checkVisibilityCSS: true })) {
          const text = el.innerText.replace(/\s+/g, ' ').trim().slice(0, 40);
          out.push({ selector: `cut fact: ${describe(el)}`, px, text });
        }
        continue;
      }
      if (el.closest('[data-scrolls]')) continue;
      if (/auto|scroll/.test(cs.overflowX) && (masked(el) || masked(el.parentElement))) continue;
      let hidden = false;
      for (let e = /** @type {Element | null} */ (el); e && e !== document.body; e = e.parentElement) {
        if (hiddenBySelf(e, getComputedStyle(e))) {
          hidden = true;
          break;
        }
      }
      if (hidden) continue;
      // A fact cut without an ellipsis: its text runs past its box and is clipped or covered (round 25's "Completed in
      // 53m 3" under the callout's button: a ui/Meta item that may shrink below its nowrap text).
      if (el.closest(FACT) && el.checkVisibility({ checkVisibilityCSS: true })) {
        clippedFacts.push({ el, px });
        continue;
      }
      hits.push({ el, px });
    }
    /** The descendant that sticks out furthest past the box's left or right edge (null: its own text is too wide). */
    const culprit = (/** @type {Element} */ el) => {
      const r = el.getBoundingClientRect();
      let worst = 1;
      /** @type {Element | null} */
      let found = null;
      for (const d of [...el.querySelectorAll('*')].slice(0, 3000)) {
        const dr = d.getBoundingClientRect();
        if (!dr.width && !dr.height) continue;
        const spill = Math.max(dr.right - r.right, r.left - dr.left);
        if (spill > worst + 0.5) {
          worst = spill;
          found = d;
        }
      }
      return found;
    };
    /** Not a layout bug: a hit area widened with a negative margin, or a positioned decoration (badge, popover). */
    const deliberate = (/** @type {Element} */ el, /** @type {Element} */ d) => {
      for (let e = /** @type {Element | null} */ (d); e && e !== el; e = e.parentElement) {
        const cs = getComputedStyle(e);
        if (cs.position === 'absolute' || cs.position === 'fixed') return true;
        if (parseFloat(cs.marginLeft) < 0 || parseFloat(cs.marginRight) < 0) return true;
      }
      return false;
    };
    for (const h of clippedFacts) {
      if (clippedFacts.some((o) => o !== h && h.el.contains(o.el) && o.px >= h.px - 1)) continue;
      const text = /** @type {HTMLElement} */ (h.el).innerText.replace(/\s+/g, ' ').trim().slice(0, 40);
      out.push({ selector: `cut fact: ${describe(h.el)}`, px: h.px, text });
    }
    // Overflow propagates up through every overflow-visible ancestor: keep the innermost box (an ancestor whose overflow
    // is no larger than a descendant's is the same spill seen again).
    for (const h of hits) {
      if (hits.some((o) => o !== h && h.el.contains(o.el) && o.px >= h.px - 1)) continue;
      const d = culprit(h.el);
      if (d && deliberate(h.el, d)) continue;
      // Nothing inside spills: a positioned ::before / ::after (a 44 px hit area around a small control, a touch target
      // around a small box on a thumbnail) on the box or inside it is the cause.
      const pseudo = (/** @type {Element} */ e) =>
        ['::before', '::after'].some((p) => {
          const ps = getComputedStyle(e, p);
          return ps.position === 'absolute' && ps.content !== 'none';
        });
      if (!d && (pseudo(h.el) || [...h.el.querySelectorAll('*')].slice(0, 3000).some(pseudo))) continue;
      const text = ((d ?? h.el) instanceof HTMLElement ? /** @type {HTMLElement} */ (d ?? h.el).innerText : (d ?? h.el).textContent) ?? '';
      out.push({
        selector: describe(h.el) + (d ? `  [widest: ${describe(d).split(' > ').pop()}]` : ''),
        px: h.px,
        text: text.replace(/\s+/g, ' ').trim().slice(0, 40),
      });
    }
    return out;
  }, SCOPE_PREFIXES);
}

/**
 * Blank sections: a selected tab or section-picker radio whose content renders nothing, and a page with almost nothing
 * under its header. The round-19 phone rule that hid every `.rail` left three sections blank behind a selected picker
 * and nothing caught it; this does.
 * - `[role=tab][aria-selected=true]` and a checked radio with `aria-controls`: the controlled element must be visible
 *   (not `display: none` itself or through an ancestor) and taller than 40 px.
 * - A checked radio in a section picker without `aria-controls` (a Segmented labelled "Show", `.phone-pick`): the
 *   content under the picker in `main` must reach 40 px, and start 12 px under it (every section, one geometry).
 * - `main`: the visible content below the page header (or main's top) must reach 80 px.
 * "Content" is what a reader can see: text, images, canvases, SVG, form controls, and boxes with a fill or an edge
 * (panel frames, skeleton blocks); fixed bars (the tab bar, docks) don't count.
 * @param {import('playwright-core').Page} page
 * @returns {Promise<{ selector: string, text: string }[]>}
 */
async function blankCheck(page) {
  return page.evaluate(() => {
    /** @type {{ selector: string, text: string }[]} */
    const out = [];
    const shown = (/** @type {Element} */ el) => el.checkVisibility({ checkVisibilityCSS: true, checkOpacity: true });
    const replaced = new Set(['IMG', 'SVG', 'CANVAS', 'VIDEO', 'INPUT', 'SELECT', 'TEXTAREA', 'BUTTON']);
    /** Visible content boxes inside `scope`: things a reader sees, not empty wrappers. */
    const contentRects = (/** @type {Element} */ scope) => contentBoxes(scope).map((b) => b.r);
    const contentBoxes = (/** @type {Element} */ scope) => {
      /** @type {{ el: Element, r: DOMRect }[]} */
      const rects = [];
      for (const el of scope.querySelectorAll('*')) {
        const r = el.getBoundingClientRect();
        if (r.width < 1 || r.height < 1) continue;
        const tag = el.tagName.toUpperCase();
        let seen = replaced.has(tag) || [...el.childNodes].some((n) => n.nodeType === Node.TEXT_NODE && n.textContent?.trim());
        if (!seen) {
          const cs = getComputedStyle(el);
          const fill = cs.backgroundColor !== 'rgba(0, 0, 0, 0)' && cs.backgroundColor !== 'transparent';
          const edge = ['Top', 'Right', 'Bottom', 'Left'].some((s) => parseFloat(cs.getPropertyValue(`border-${s.toLowerCase()}-width`)) > 0);
          seen = fill || edge || cs.backgroundImage !== 'none';
        }
        if (!seen || !shown(el)) continue;
        let fixed = false;
        for (let e = /** @type {Element | null} */ (el); e && e !== scope; e = e.parentElement) {
          if (getComputedStyle(e).position === 'fixed') {
            fixed = true;
            break;
          }
        }
        if (!fixed) rects.push({ el, r });
      }
      return rects;
    };
    /** How far visible content inside `scope` reaches below `y` (px). */
    const reachBelow = (/** @type {Element} */ scope, /** @type {number} */ y) => {
      let top = Infinity;
      let bottom = -Infinity;
      for (const r of contentRects(scope)) {
        if (r.top < y - 1) continue;
        top = Math.min(top, r.top);
        bottom = Math.max(bottom, r.bottom);
      }
      return bottom > top ? bottom - Math.max(y, top) : 0;
    };
    const name = (/** @type {Element} */ el) =>
      (el.getAttribute('aria-label') ?? (el instanceof HTMLElement ? el.innerText : el.textContent) ?? '').replace(/\s+/g, ' ').trim().slice(0, 40);
    const main = document.querySelector('main');

    const selected = [
      ...document.querySelectorAll('[role=tab][aria-selected="true"], [role=radiogroup] [role=radio][aria-checked="true"]'),
    ].filter(shown);
    for (const ctl of selected) {
      const group = ctl.closest('[role=tablist], [role=radiogroup]');
      const groupName = group ? (group.getAttribute('aria-label') ?? '') : '';
      const ids = (ctl.getAttribute('aria-controls') ?? '').split(/\s+/).filter(Boolean);
      if (ids.length) {
        for (const id of ids) {
          const target = document.getElementById(id);
          const h = target ? target.getBoundingClientRect().height : 0;
          if (!target || !shown(target) || h <= 40) {
            out.push({ selector: `${groupName || 'tabs'} › ${name(ctl)}: #${id} ${!target ? 'missing' : !shown(target) ? 'hidden' : `${Math.round(h)}px tall`}`, text: '' });
          }
        }
        continue;
      }
      const picker = group?.getAttribute('aria-label') === 'Show' || !!ctl.closest('.phone-pick');
      if (!picker || !main || !group) continue;
      const pickBottom = group.getBoundingClientRect().bottom;
      const reach = reachBelow(main, pickBottom);
      if (reach < 40) out.push({ selector: `section picker "${groupName}" › ${name(ctl)}: ${Math.round(reach)}px of content under it`, text: '' });
      // Every section starts 12 px under its picker: a hidden section must not leave a track or a gap behind. Only at
      // the page's top: once scrolled, the sticky picker sits over the section's middle.
      // Rows a scroller has moved out of its box above (a log scrolled to its end) are not where the section starts.
      const clippedAbove = (/** @type {Element} */ el, /** @type {DOMRect} */ r) => {
        for (let e = el.parentElement; e && e !== main; e = e.parentElement) {
          if (getComputedStyle(e).overflowY !== 'visible' && r.top < e.getBoundingClientRect().top - 0.5) return true;
        }
        return false;
      };
      const first =
        window.scrollY > 0
          ? Infinity
          : Math.min(...contentBoxes(main).filter(({ el, r }) => r.top >= pickBottom - 1 && !clippedAbove(el, r)).map(({ r }) => r.top));
      if (Number.isFinite(first) && Math.abs(first - pickBottom - 12) > 1) {
        out.push({ selector: `section picker "${groupName}" › ${name(ctl)}: the section starts ${Math.round(first - pickBottom)}px under it, not 12`, text: '' });
      }
    }

    if (main && shown(main)) {
      const header = main.querySelector('.page > header');
      const y = header && shown(header) ? header.getBoundingClientRect().bottom : main.getBoundingClientRect().top;
      const reach = reachBelow(main, y);
      if (reach < 80) out.push({ selector: `main: ${Math.round(reach)}px of content under the page header`, text: '' });
    }

    // A page header's actions are one row at every tier (shell/Page, DESIGN_DIRECTION §10): a second row leaves the
    // title floating between them. Only the visible leaves count (Gate and Menu wrappers are `display: contents`).
    for (const acts of document.querySelectorAll('.page > header .actions')) {
      if (!shown(acts)) continue;
      // An open menu's sheet or popover is not the header's row.
      const floating = (/** @type {Element} */ el) => {
        for (let e = el.parentElement; e && e !== acts; e = e.parentElement) {
          if (e.matches('[role=menu], [role=dialog], dialog') || ['fixed', 'absolute'].includes(getComputedStyle(e).position)) return true;
        }
        return false;
      };
      const tops = [...acts.querySelectorAll('.btn, [data-gate-why]')]
        .filter((el) => shown(el) && el.getBoundingClientRect().width > 0 && !floating(el))
        .map((el) => el.getBoundingClientRect());
      const rows = tops.filter((r, i) => tops.findIndex((q) => Math.abs(q.top + q.height / 2 - (r.top + r.height / 2)) < 8) === i).length;
      if (rows > 1) out.push({ selector: `page header: its actions wrap onto ${rows} rows`, text: '' });
    }
    return out;
  });
}

/**
 * Covered controls: a control whose own box another element paints over (DESIGN_DIRECTION §12, a control is never
 * covered by its neighbour; round 27: Model's Band / All curves lost its bottom 6 px to the Trace bar's negative
 * margin). Every visible button, select, input, tab, radio and segmented group is hit-tested 3 px inside its edges' midpoints and at its
 * centre; a hit that is neither the control, inside it nor one of its ancestors (a scroller clipping it) covers it.
 * Not a fault: fixed chrome and overlays (the tab bar, toasts, sheets, popovers), and a sticky bar that is stuck (the
 * content scrolls under it by design). A neighbour's enlarged hit area (a `::after` over the control) counts, since a
 * tap there goes to the neighbour.
 * @param {import('playwright-core').Page} page
 * @returns {Promise<{ selector: string, text: string }[]>}
 */
async function coverCheck(page) {
  return page.evaluate((prefixes) => {
    /** @type {{ selector: string, text: string }[]} */
    const out = [];
    const describe = (/** @type {Element} */ e) => {
      const cls = [...e.classList].filter((c) => !prefixes.some((p) => c.startsWith(p))).slice(0, 2);
      return `${e.tagName.toLowerCase()}${cls.length ? `.${cls.join('.')}` : ''}`;
    };
    const name = (/** @type {Element} */ el) =>
      (el.getAttribute('aria-label') ?? (el instanceof HTMLElement ? el.innerText : el.textContent) ?? '').replace(/\s+/g, ' ').trim().slice(0, 32);
    /** Overlays and fixed chrome, and sticky bars stuck at their offset: allowed to lie over content. */
    const allowed = (/** @type {Element} */ cover) => {
      for (let e = /** @type {Element | null} */ (cover); e && e !== document.body; e = e.parentElement) {
        if (e.matches('dialog, [popover], [role=menu], [role=dialog], [role=tooltip]')) return true;
        const cs = getComputedStyle(e);
        if (cs.position === 'fixed') return true;
        if (cs.position === 'sticky') {
          const top = parseFloat(cs.top);
          if (Number.isFinite(top) && Math.abs(e.getBoundingClientRect().top - top) < 1.5) return true;
        }
      }
      return false;
    };
    // A segmented group's frame (its radiogroup) is part of the control: round 27 cut that, not the radios inside.
    const controls = document.querySelectorAll(
      'main button, main select, main input:not([type=hidden]), main [role=tab], main [role=radio], main [role=radiogroup], main a.btn',
    );
    for (const el of controls) {
      if (!el.checkVisibility({ checkVisibilityCSS: true, checkOpacity: true })) continue;
      const cs = getComputedStyle(el);
      if (cs.pointerEvents === 'none') continue;
      const r = el.getBoundingClientRect();
      if (r.width < 8 || r.height < 8) continue;
      const inset = 3;
      const points = [
        [r.left + r.width / 2, r.top + r.height / 2],
        [r.left + r.width / 2, r.top + inset],
        [r.left + r.width / 2, r.bottom - inset],
        [r.left + inset, r.top + r.height / 2],
        [r.right - inset, r.top + r.height / 2],
      ];
      for (const [x, y] of points) {
        if (x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) continue;
        const hit = document.elementFromPoint(x, y);
        if (!hit || hit === el || el.contains(hit) || hit.contains(el) || allowed(hit)) continue;
        out.push({ selector: `covered: ${describe(el)} "${name(el)}" under ${describe(hit)} "${name(hit)}"`, text: '' });
        break;
      }
    }
    return out;
  }, SCOPE_PREFIXES);
}

/**
 * Cut selects: a closed select whose current value does not fit its field (DESIGN_DIRECTION §12, a select shows its
 * whole current value at 16 px touch text; round 27: Compare's "mAP50-…" beside the axis toggle). A native select
 * draws its own text, which no scrollWidth reports, so the selected option's text is measured in the select's font
 * against the field's content box; a select showing a `short` label (ui/Select) is checked on that label's box.
 * @param {import('playwright-core').Page} page
 * @returns {Promise<{ selector: string, text: string }[]>}
 */
async function selectCheck(page) {
  return page.evaluate(() => {
    /** @type {{ selector: string, text: string }[]} */
    const out = [];
    const ctx = document.createElement('canvas').getContext('2d');
    if (!ctx) return out;
    for (const el of document.querySelectorAll('main select')) {
      if (!(el instanceof HTMLSelectElement) || !el.checkVisibility({ checkVisibilityCSS: true, checkOpacity: true })) continue;
      const name = el.getAttribute('aria-label') ?? '';
      const shown = el.parentElement?.querySelector('.shown');
      if (shown instanceof HTMLElement) {
        if (shown.scrollWidth > shown.clientWidth + 1) out.push({ selector: `cut select: "${name}" shows "${shown.textContent?.trim()}"`, text: '' });
        continue;
      }
      const text = el.selectedOptions[0]?.textContent?.trim() ?? '';
      if (!text) continue;
      const cs = getComputedStyle(el);
      ctx.font = `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
      const room = el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
      if (ctx.measureText(text).width > room + 1) out.push({ selector: `cut select: "${name}" shows "${text}"`, text: '' });
    }
    return out;
  });
}

/**
 * --exact's Chrome: the same pixels for the same page every time. Rastering on the GPU (headless Chrome uses the real
 * one on Windows) leaves a few pixels of an edge or an image a level or two apart between shoots, so everything is
 * drawn in software. Chrome also picks, by timing, between paths that draw an image differently: a 2D canvas on the
 * GPU or the CPU (the mock paints its images on one), a JPEG decoded to YUV planes or to RGB, an image decoded during
 * raster or later (checker-imaging), a tile rastered whole or in part, text antialiased for LCD or in grey (by whether its
 * layer is opaque, which an animation or a backdrop changes).
 */
const EXACT_ARGS = exact
  ? ['--disable-gpu', '--disable-lcd-text', '--disable-partial-raster', '--disable-accelerated-2d-canvas', '--disable-checker-imaging', '--disable-features=DecodeJpeg420ImagesToYUV,DecodeLossyWebPImagesToYUV', '--force-color-profile=srgb']
  : [];

/**
 * The installed Chrome, else the Chromium `npx playwright-core install chromium` downloads (CI's Linux runners have no
 * Chrome). FOLO_SHOOT_BROWSER=chrome or chromium picks one and fails if it is missing.
 */
async function launch() {
  const want = process.env.FOLO_SHOOT_BROWSER;
  if (want !== 'chromium') {
    try {
      return await chromium.launch({ channel: 'chrome', headless: true, args: EXACT_ARGS });
    } catch (e) {
      if (want === 'chrome') throw e;
    }
  }
  try {
    const b = await chromium.launch({ headless: true, args: EXACT_ARGS });
    console.log(`Using Playwright's Chromium ${b.version()} (no Chrome, or FOLO_SHOOT_BROWSER=chromium).`);
    return b;
  } catch (e) {
    throw new Error(
      'No browser: install Chrome, or Playwright\'s Chromium with `npx playwright-core install chromium` ' +
        `(on Linux its system libraries too: \`npx playwright-core install-deps chromium\`).\n${e instanceof Error ? e.message : e}`,
    );
  }
}

const browser = await launch();
/** Bold yellow in a terminal (the overflow QA must be impossible to miss). */
const yellow = (/** @type {string} */ t) => `\u001b[1;33m${t}\u001b[0m`;
let count = 0;
const failures = [];
/** Each failed state's first problem, for shoot.json. @type {Record<string, string>} */
const failed = {};
/** The shots taken. @type {string[]} */
const shots = [];
const fail = (/** @type {string} */ name, /** @type {string} */ why) => {
  failures.push(`${name}: ${why}`);
  failed[name] ??= why;
};
/** Why no state can be shot, or null (kept in shoot.json for scripts/parity.mjs). @type {string | null} */
const missing = null;
if (missing) console.warn(`${yellow('! missing')} ${missing}: every state fails`);
/** Overflow (`px` > 0) and blank sections (`px` 0), one list. @type {{ name: string, selector: string, px: number, text: string }[]} */
const overflows = [];

/**
 * Opens one scene in one theme and viewport, sets it up, runs both checks and (without --qa) takes the shot.
 * @param {import('./scenes.mjs').Scene} scene
 * @param {'dark' | 'light'} theme
 * @param {'desktop' | 'wide' | 'laptop' | 'tablet' | 'mobile' | 'small'} vp
 * @param {boolean} [touch] a touch screen (`pointer: coarse`, `hover: none`): --qa's phone states
 * @returns {Promise<boolean>} whether the page showed a phone section picker
 */
async function shoot(scene, theme, vp, touch = false) {
  const name = `${scene.name}--${themeId ?? theme}--${vp}${touch ? '+touch' : ''}`;
  if (missing) {
    failed[name] = missing;
    return false;
  }
  const context = await browser.newContext({
    viewport: VIEWPORTS[vp],
    hasTouch: touch,
    deviceScaleFactor: vp === 'mobile' || vp === 'small' ? 2 : 1,
    colorScheme: theme === 'light' ? 'light' : 'dark',
    reducedMotion: 'reduce',
  });
  const page = await context.newPage();
  page.on('pageerror', (e) => fail(name, e.message));
  let picker = false;
  try {
    // An installed clock keeps running in real time, so where a page's timers stand at the check depends on load. --qa
    // pauses it at FIXED_NOW before the page loads (installed an hour early: pauseAt only moves forward), so time then
    // moves only by the scenes' runFor, the same on every machine. --exact too: a scene's shots then match.
    await page.clock.install({ time: paused ? FIXED_NOW.getTime() - 3_600_000 : FIXED_NOW });
    if (paused) await page.clock.pauseAt(FIXED_NOW);
    await page.addInitScript(
      ([t, id]) => {
        localStorage.setItem('folo.theme', t);
        localStorage.setItem('folo.shoot', '1');
        // --theme: that mode's theme (prefs.ts darkTheme / lightTheme), read by the first-paint script and the app.
        if (id) localStorage.setItem('folo.prefs', JSON.stringify({ theme: t, [`${t}Theme`]: id }));
      },
      [theme, themeId ?? ''],
    );
    // --exact: the mock's latency jitter (Math.random, src/lib/api/mock/server.ts) decides how many of the scenes'
    // 250 ms clock steps a wait takes, so the live run's progress drifts between shoots. A constant draw gives every
    // call its nominal latency, whatever order an app makes its calls in.
    if (exact) await page.addInitScript(() => (Math.random = () => 0.5));
    // A paused clock (--qa too) steps only by the scenes' runFor: each step waits for the real-time work the last one
    // started, or a loaded machine lets the steps run out before a lazy chunk or an image is in, and a wait that
    // needs the mock's next timer never sees it (the qa states that timed out on CI only).
    if (paused) await makeQuiet(page);
    const res = await page.goto(`${base}${ENTRY}?mock=1${scene.search ? `&${scene.search}` : ''}#${scene.path}`, { waitUntil: 'networkidle' });
    if (res && !res.ok()) throw new Error(`${ENTRY} answered ${res.status()}`);
    await page.evaluate(() => document.fonts.ready);
    // Let the mock's live run advance a little, deterministically.
    await page.clock.runFor(scene.advance ?? 2000);
    if (scene.setup) await scene.setup(page, { vp, theme });
    await settle(page);
    // --exact (parity) skips the layout checks: they take time and change nothing the shot shows.
    if (exact) {
      count++;
      await shot(page, scene, vp, name);
      return false;
    }
    // Content cut off or spilling sideways (the page included: wide content must scroll inside its own box).
    for (const o of await overflowCheck(page)) {
      overflows.push({ name, ...o });
      console.warn(`${yellow('! overflow')} ${name}  ${o.selector}  +${o.px}px`);
    }
    // A selected section that renders nothing, or a page that is empty under its header.
    for (const b of await blankCheck(page)) {
      overflows.push({ name, px: 0, ...b });
      console.warn(`${yellow('! blank')} ${name}  ${b.selector}`);
    }
    // A control another element paints over (a sticky bar pulled up over a header, a neighbour's hit area), and a
    // select too narrow for its own value.
    for (const c of [...(await coverCheck(page)), ...(await selectCheck(page))]) {
      overflows.push({ name, px: 0, ...c });
      console.warn(`${yellow('! covered')} ${name}  ${c.selector}`);
    }
    picker = await page.evaluate(() =>
      [...document.querySelectorAll('.phone-pick, [role=radiogroup][aria-label="Show"]')].some((el) => el.checkVisibility({ checkVisibilityCSS: true })),
    );
    count++;
    if (qa) return picker;
    await shot(page, scene, vp, name);
  } catch (e) {
    fail(name, e instanceof Error ? e.message : String(e));
  } finally {
    await context.close();
  }
  return picker;
}

/**
 * Takes a state's shot (with --exact, its DOM dump too): from the top unless the scene keeps its scroll.
 * @param {import('playwright-core').Page} page
 * @param {import('./scenes.mjs').Scene} scene
 * @param {string} vp
 * @param {string} name
 */
async function shot(page, scene, vp, name) {
  {
    const full = scene.fullPage ?? (vp !== 'mobile' && vp !== 'small');
    // Capture what a user sees: from the top, with no stray focus ring on the skip link. For full-page shots the
    // viewport grows to the document's height, so sticky and fixed bars (top bar, tab bar, docks) sit where they
    // belong instead of wherever the viewport happened to be when the image was stitched.
    // A scene that scrolled on purpose (`keepScroll`, a viewport shot of a sticky rail) keeps its scroll.
    await page.evaluate((keep) => {
      if (!keep) window.scrollTo(0, 0);
      const a = document.activeElement;
      if (a instanceof HTMLElement && a.classList.contains('skip')) a.blur();
    }, !!scene.keepScroll && !full);
    // Only pages with a visible fixed bar need it; others keep their real viewport (some size to 100dvh).
    const hasFixedBar = await page.evaluate(() =>
      [...document.querySelectorAll('body *')].some((el) => {
        const cs = getComputedStyle(el);
        const r = el.getBoundingClientRect();
        return cs.position === 'fixed' && cs.display !== 'none' && cs.visibility !== 'hidden' && r.height > 24 && r.width > 200;
      }),
    );
    if (full && hasFixedBar) {
      for (let k = 0; k < 2; k++) {
        const h = await page.evaluate(() => document.documentElement.scrollHeight);
        const cur = page.viewportSize();
        if (!cur || cur.height >= h) break;
        await page.setViewportSize({ width: cur.width, height: Math.min(h, 8000) });
        await settle(page);
      }
    }
    if (exact) writeFileSync(resolve(outDir, `${name}.dom.txt`), await domDump(page));
    // --exact: finite animations and transitions jump to their end and infinite ones (spinners, the skeletons'
    // shimmer, the live pulse) back to their start, since the compositor runs them in real time whatever the clock.
    await page.screenshot({ path: resolve(outDir, `${name}.png`), fullPage: full, ...(exact ? { animations: /** @type {const} */ ('disabled'), caret: /** @type {const} */ ('hide') } : {}) });
    shots.push(name);
  }
}

/** --qa's sizes: wide, tablet, mobile and small (the scenes listing them); FOLO_QA_SIZES=laptop,tablet (comma-separated) checks others on demand. */
const qaSizes = (process.env.FOLO_QA_SIZES ?? 'wide,tablet,mobile,small').split(',').map((v) => v.trim());
/** --qa's states, run by a small pool below. @type {(() => Promise<void>)[]} */
const qaJobs = [];
/** The shots, in order: one at a time, or with --exact by the same pool. @type {(() => Promise<void>)[]} */
const shotJobs = [];
try {
  for (const scene of scenes) {
    const themes = (scene.themes ?? ['dark', 'light']).filter((m) => !themeMode || m === themeMode);
    const viewports = scene.viewports ?? ['desktop', 'mobile'];
    if (!qa) {
      for (const theme of themes) {
        for (const vp of viewports) {
          if (picked(`${scene.name}--${themeId ?? theme}--${vp}`)) shotJobs.push(() => shoot(scene, theme, vp).then(() => {}));
        }
      }
      continue;
    }
    // --qa: the tablet and phone sizes in dark (or the one theme a scene is made for), and light too behind a picker.
    const first = themes.includes('dark') ? 'dark' : themes[0];
    for (const vp of viewports.filter((v) => qaSizes.includes(v))) {
      if (!picked(`${scene.name}--${themeId ?? first}--${vp}`)) continue;
      // A phone is a touch screen, and touch is the stricter case, so a phone state is checked once, under a coarse
      // pointer. A scene about a fine or a coarse pointer in particular (keyboard shortcuts, the touch hint) sets
      // `touch: false` and keeps the mouse.
      const touch = (vp === 'mobile' || vp === 'small') && scene.touch !== false;
      qaJobs.push(async () => {
        if ((await shoot(scene, first, vp, touch)) && first === 'dark' && themes.includes('light')) await shoot(scene, 'light', vp, touch);
      });
    }
  }
  // FOLO_QA_SHARD=2/3 checks every third state, starting with the second: CI splits --qa over several runners, since
  // a state costs a few CPU seconds of browser work and one runner has only so many cores.
  const [shard, shards] = (process.env.FOLO_QA_SHARD ?? '1/1').split('/').map(Number);
  if (!(Number.isInteger(shard) && Number.isInteger(shards) && shard >= 1 && shard <= shards)) throw new Error(`FOLO_QA_SHARD=${process.env.FOLO_QA_SHARD}: expected i/n with 1 <= i <= n`);
  const mine = qaJobs.filter((_, i) => i % shards === shard - 1);
  qaJobs.length = 0;
  qaJobs.push(...mine, ...shotJobs);
  // Each state has its own context and paused clock, so --qa (and --exact) runs several at once: --jobs or
  // FOLO_QA_JOBS, default up to 8. A plain shoot's clock runs in real time: one state at a time, as it always has.
  const jobs = qa || exact ? Math.max(1, Number(opt('--jobs')) || Number(process.env.FOLO_QA_JOBS) || Math.min(8, availableParallelism())) : 1;
  await Promise.all(
    Array.from({ length: jobs }, async () => {
      for (let job = qaJobs.shift(); job; job = qaJobs.shift()) await job();
    }),
  );
} finally {
  await browser.close();
  await server.close();
}

const secs = Math.round((Date.now() - started) / 1000);
const part = process.env.FOLO_QA_SHARD ? ` (shard ${process.env.FOLO_QA_SHARD})` : '';
console.log(qa ? `QA: ${count} scene states checked in ${secs} s${part}` : `${count} shots -> ${outDir} (${secs} s)`);
// What parity.mjs reads: which states were shot and why the others were not.
if (exact) writeFileSync(resolve(outDir, 'shoot.json'), JSON.stringify({ entry: ENTRY, base, only: only ?? null, missing, secs, shots: shots.sort(), failed }, null, 1));
// The QA list (overflow, blank sections and covered controls): printed last, boxed in yellow. The shoot saves it next to the shots and
// never fails on it; --qa fails on it.
const report = resolve(outDir, only ? `qa-${only.replace(/[^\w-]+/g, '_')}.txt` : 'qa.txt');
// The pool finishes states out of order: list them in scene order.
const order = (/** @type {string} */ n) => scenes.findIndex((s) => s.name === n.split('--')[0]);
overflows.sort((a, b) => order(a.name) - order(b.name) || a.name.localeCompare(b.name));
if (overflows.length) {
  const shots = new Set(overflows.map((o) => o.name)).size;
  const wide = overflows.filter((o) => o.px > 0).length;
  const head = `SHOOT QA: ${wide} element(s) overflow horizontally, ${overflows.length - wide} blank section(s) or covered control(s), in ${shots} ${qa ? 'state' : 'shot'}(s)`;
  const lines = overflows.map((o) => {
    const [scene, theme, vp] = o.name.split('--');
    const what = o.px > 0 ? `+${String(o.px).padStart(4)}px` : o.selector.startsWith('covered') ? 'covered' : o.selector.startsWith('cut select') ? '    cut' : ' blank';
    return `${scene.padEnd(28)} ${theme.padEnd(5)} ${vp.padEnd(7)} ${what}  ${o.selector}${o.text ? `  "${o.text}"` : ''}`;
  });
  if (!qa) writeFileSync(report, [head, ...lines, ''].join(EOL));
  const bar = '='.repeat(Math.max(head.length, 72));
  console.warn([yellow(bar), yellow(head), yellow(bar), ...lines, yellow(bar), ...(qa ? [] : [`(saved to ${report})`])].join(EOL));
  if (qa) process.exitCode = 1;
} else {
  if (!qa) rmSync(report, { force: true });
  console.log('Shoot QA: nothing overflows horizontally, no blank sections, no covered controls.');
}
if (failures.length) {
  console.error(`${failures.length} problem(s):\n  ${failures.join('\n  ')}`);
  process.exitCode = 1;
}
if (missing) {
  console.error(`${Object.keys(failed).length} state(s) not shot: ${missing}`);
  process.exitCode = 1;
}
