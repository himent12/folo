// Screenshots every screen and important state into web/shots/<round>/.
//   npm run shoot                 -> shots/<timestamp>/
//   npm run shoot -- r3           -> shots/r3/
//   npm run shoot -- r3 --only runs   (scene names containing "runs")
// Uses the installed Chrome (playwright-core, channel "chrome"); no browser download.
// After each scene's setup, the QA lists every element whose content is wider than its box (and the page wider than the
// viewport), skipping deliberate scrollers, and every blank section: a selected tab or section picker whose content
// renders nothing, a page with almost nothing under its header. The list prints at the end and goes to qa.txt.
import { createServer } from 'vite';
import { chromium } from 'playwright-core';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { EOL } from 'node:os';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { scenes } from './scenes.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const onlyIdx = args.indexOf('--only');
const only = onlyIdx >= 0 ? args[onlyIdx + 1] : undefined;
const round = args.find((a, i) => !a.startsWith('--') && (onlyIdx < 0 || i !== onlyIdx + 1)) ?? new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
const outDir = resolve(root, 'shots', round);

const VIEWPORTS = {
  desktop: { width: 1440, height: 900 },
  laptop: { width: 1024, height: 768 },
  // Below the 960 px laptop tier: one column, the section picker, the New run dock.
  tablet: { width: 820, height: 1180 },
  mobile: { width: 390, height: 844 },
};
// Fixed wall clock so relative times ("3 min ago") and the mock's live run are identical every round.
const FIXED_NOW = new Date('2026-11-08T14:30:00Z');

const server = await createServer({ root, logLevel: 'error', server: { host: '127.0.0.1', port: 5199, strictPort: false } });
await server.listen();
const base = (server.resolvedUrls?.local[0] ?? 'http://127.0.0.1:5199/').replace(/\/$/, '');

if (!only) rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

/** Waits (in real time; the page clock is frozen) until every image has decoded, then a beat for layout. */
async function settle(page) {
  for (let i = 0; i < 40; i++) {
    const pending = await page.evaluate(() => [...document.images].filter((im) => !im.complete).length);
    if (!pending) break;
    await page.waitForTimeout(100);
  }
  await page.clock.runFor(50);
  await page.waitForTimeout(150);
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
  return page.evaluate(() => {
    const out = [];
    const doc = document.documentElement;
    const pageOver = doc.scrollWidth - doc.clientWidth;
    if (pageOver > 1) out.push({ selector: '(page)', px: pageOver, text: '' });
    const hiddenBySelf = (/** @type {Element} */ el, /** @type {CSSStyleDeclaration} */ cs) =>
      cs.display === 'none' ||
      cs.visibility === 'hidden' ||
      (cs.clipPath !== 'none' && cs.clipPath !== '') ||
      (cs.clip !== 'auto' && cs.clip !== '') ||
      (el.getAttribute('aria-hidden') === 'true' && cs.position === 'absolute');
    const masked = (/** @type {Element | null} */ el) => {
      if (!el) return false;
      const cs = getComputedStyle(el);
      return (cs.maskImage && cs.maskImage !== 'none') || (cs.webkitMaskImage && cs.webkitMaskImage !== 'none');
    };
    const describe = (/** @type {Element} */ el) => {
      const part = (/** @type {Element} */ e) => {
        const cls = [...e.classList].filter((c) => !c.startsWith('svelte-')).slice(0, 3);
        return `${e.tagName.toLowerCase()}${e.id ? `#${e.id}` : ''}${cls.length ? `.${cls.join('.')}` : ''}`;
      };
      const parts = [];
      for (let e = /** @type {Element | null} */ (el); e && e !== document.body && parts.length < 3; e = e.parentElement) parts.unshift(part(e));
      return parts.join(' > ');
    };
    /** @type {{ el: Element, px: number }[]} */
    const hits = [];
    for (const el of document.querySelectorAll('body *')) {
      if (!(el instanceof HTMLElement) || el.clientWidth === 0) continue;
      const px = el.scrollWidth - el.clientWidth;
      if (px <= 1) continue;
      const cs = getComputedStyle(el);
      if (cs.textOverflow === 'ellipsis') continue;
      if (el.closest('[data-scrolls]')) continue;
      if (/auto|scroll/.test(cs.overflowX) && (masked(el) || masked(el.parentElement))) continue;
      let hidden = false;
      for (let e = /** @type {Element | null} */ (el); e && e !== document.body; e = e.parentElement) {
        if (hiddenBySelf(e, getComputedStyle(e))) {
          hidden = true;
          break;
        }
      }
      if (!hidden) hits.push({ el, px });
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
    // Overflow propagates up through every overflow-visible ancestor: keep the innermost box (an ancestor whose overflow
    // is no larger than a descendant's is the same spill seen again).
    for (const h of hits) {
      if (hits.some((o) => o !== h && h.el.contains(o.el) && o.px >= h.px - 1)) continue;
      const d = culprit(h.el);
      if (d && deliberate(h.el, d)) continue;
      const text = ((d ?? h.el) instanceof HTMLElement ? /** @type {HTMLElement} */ (d ?? h.el).innerText : (d ?? h.el).textContent) ?? '';
      out.push({
        selector: describe(h.el) + (d ? `  [widest: ${describe(d).split(' > ').pop()}]` : ''),
        px: h.px,
        text: text.replace(/\s+/g, ' ').trim().slice(0, 40),
      });
    }
    return out;
  });
}

/**
 * Blank sections: a selected tab or section-picker radio whose content renders nothing, and a page with almost nothing
 * under its header. The round-19 phone rule that hid every `.rail` left three sections blank behind a selected picker
 * and nothing caught it; this does.
 * - `[role=tab][aria-selected=true]` and a checked radio with `aria-controls`: the controlled element must be visible
 *   (not `display: none` itself or through an ancestor) and taller than 40 px.
 * - A checked radio in a section picker without `aria-controls` (a Segmented labelled "Show", `.phone-pick`): the
 *   content under the picker in `main` must reach 40 px.
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
    const contentRects = (/** @type {Element} */ scope) => {
      /** @type {DOMRect[]} */
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
        if (!fixed) rects.push(r);
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
      const reach = reachBelow(main, group.getBoundingClientRect().bottom);
      if (reach < 40) out.push({ selector: `section picker "${groupName}" › ${name(ctl)}: ${Math.round(reach)}px of content under it`, text: '' });
    }

    if (main && shown(main)) {
      const header = main.querySelector('.page > header');
      const y = header && shown(header) ? header.getBoundingClientRect().bottom : main.getBoundingClientRect().top;
      const reach = reachBelow(main, y);
      if (reach < 80) out.push({ selector: `main: ${Math.round(reach)}px of content under the page header`, text: '' });
    }
    return out;
  });
}

const browser = await chromium.launch({ channel: 'chrome', headless: true });
/** Bold yellow in a terminal (the overflow QA must be impossible to miss). */
const yellow = (/** @type {string} */ t) => `\u001b[1;33m${t}\u001b[0m`;
let count = 0;
const failures = [];
/** Overflow (`px` > 0) and blank sections (`px` 0), one list. @type {{ name: string, selector: string, px: number, text: string }[]} */
const overflows = [];
try {
  for (const scene of scenes) {
    for (const theme of scene.themes ?? ['dark', 'light']) {
      for (const vp of scene.viewports ?? ['desktop', 'mobile']) {
        const name = `${scene.name}--${theme}--${vp}`;
        if (only && !name.includes(only)) continue;
        const context = await browser.newContext({
          viewport: VIEWPORTS[vp],
          deviceScaleFactor: vp === 'mobile' ? 2 : 1,
          colorScheme: theme === 'light' ? 'light' : 'dark',
          reducedMotion: 'reduce',
        });
        const page = await context.newPage();
        await page.clock.install({ time: FIXED_NOW });
        await page.addInitScript((t) => {
          localStorage.setItem('folo.theme', t);
          localStorage.setItem('folo.shoot', '1');
        }, theme);
        page.on('pageerror', (e) => failures.push(`${name}: ${e.message}`));
        try {
          await page.goto(`${base}/?mock=1${scene.search ? `&${scene.search}` : ''}#${scene.path}`, { waitUntil: 'networkidle' });
          await page.evaluate(() => document.fonts.ready);
          // Let the mock's live run advance a little, deterministically.
          await page.clock.runFor(scene.advance ?? 2000);
          if (scene.setup) await scene.setup(page, { vp, theme });
          await settle(page);
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
          const full = scene.fullPage ?? vp !== 'mobile';
          // Capture what a user sees: from the top, with no stray focus ring on the skip link. For full-page shots the
          // viewport grows to the document's height, so sticky and fixed bars (top bar, tab bar, docks) sit where they
          // belong instead of wherever the viewport happened to be when the image was stitched.
          await page.evaluate(() => {
            window.scrollTo(0, 0);
            const a = document.activeElement;
            if (a instanceof HTMLElement && a.classList.contains('skip')) a.blur();
          });
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
          await page.screenshot({ path: resolve(outDir, `${name}.png`), fullPage: full });
          count++;
        } catch (e) {
          failures.push(`${name}: ${e instanceof Error ? e.message : e}`);
        } finally {
          await context.close();
        }
      }
    }
  }
} finally {
  await browser.close();
  await server.close();
}

console.log(`${count} shots -> ${outDir}`);
// The QA list (overflow and blank sections): never fails the shoot, but printed last, boxed in yellow, and saved next
// to the shots.
const report = resolve(outDir, only ? `qa-${only.replace(/[^\w-]+/g, '_')}.txt` : 'qa.txt');
if (overflows.length) {
  const shots = new Set(overflows.map((o) => o.name)).size;
  const wide = overflows.filter((o) => o.px > 0).length;
  const head = `SHOOT QA: ${wide} element(s) overflow horizontally, ${overflows.length - wide} blank section(s), in ${shots} shot(s)`;
  const lines = overflows.map((o) => {
    const [scene, theme, vp] = o.name.split('--');
    const what = o.px > 0 ? `+${String(o.px).padStart(4)}px` : ' blank';
    return `${scene.padEnd(28)} ${theme.padEnd(5)} ${vp.padEnd(7)} ${what}  ${o.selector}${o.text ? `  "${o.text}"` : ''}`;
  });
  writeFileSync(report, [head, ...lines, ''].join(EOL));
  const bar = '='.repeat(Math.max(head.length, 72));
  console.warn([yellow(bar), yellow(head), yellow(bar), ...lines, yellow(bar), `(saved to ${report})`].join(EOL));
} else {
  rmSync(report, { force: true });
  console.log('Shoot QA: nothing overflows horizontally, no blank sections.');
}
if (failures.length) {
  console.error(`${failures.length} problem(s):\n  ${failures.join('\n  ')}`);
  process.exitCode = 1;
}
