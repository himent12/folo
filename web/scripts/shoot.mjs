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
// renders nothing, a page with almost nothing under its header. The list prints at the end and goes to qa.txt.
// --qa runs only those checks, where phone layouts break: every scene at the tablet and mobile sizes it lists, in dark
// (or the one theme a scene is made for), and in light too where the page shows a phone section picker (Run, Model).
// Each state gets its scene's setup; nothing is captured or saved.
import { createServer } from 'vite';
import { chromium } from 'playwright-core';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { availableParallelism, EOL } from 'node:os';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { scenes } from './scenes.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const onlyIdx = args.indexOf('--only');
const only = onlyIdx >= 0 ? args[onlyIdx + 1] : undefined;
const qa = args.includes('--qa');
const round = args.find((a, i) => !a.startsWith('--') && (onlyIdx < 0 || i !== onlyIdx + 1)) ?? new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
const outDir = resolve(root, 'shots', round);
const started = Date.now();

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
        const cls = [...e.classList].filter((c) => !c.startsWith('svelte-')).slice(0, 3);
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
      // Nothing inside spills: a positioned ::before / ::after (a 44 px hit area around a small control) is the cause.
      if (!d && ['::before', '::after'].some((p) => getComputedStyle(h.el, p).position === 'absolute')) continue;
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
    return out;
  });
}

/**
 * The installed Chrome, else the Chromium `npx playwright-core install chromium` downloads (CI's Linux runners have no
 * Chrome). FOLO_SHOOT_BROWSER=chrome or chromium picks one and fails if it is missing.
 */
async function launch() {
  const want = process.env.FOLO_SHOOT_BROWSER;
  if (want !== 'chromium') {
    try {
      return await chromium.launch({ channel: 'chrome', headless: true });
    } catch (e) {
      if (want === 'chrome') throw e;
    }
  }
  try {
    const b = await chromium.launch({ headless: true });
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
/** Overflow (`px` > 0) and blank sections (`px` 0), one list. @type {{ name: string, selector: string, px: number, text: string }[]} */
const overflows = [];

/**
 * Opens one scene in one theme and viewport, sets it up, runs both checks and (without --qa) takes the shot.
 * @param {import('./scenes.mjs').Scene} scene
 * @param {'dark' | 'light'} theme
 * @param {'desktop' | 'laptop' | 'tablet' | 'mobile'} vp
 * @returns {Promise<boolean>} whether the page showed a phone section picker
 */
async function shoot(scene, theme, vp) {
  const name = `${scene.name}--${theme}--${vp}`;
  const context = await browser.newContext({
    viewport: VIEWPORTS[vp],
    deviceScaleFactor: vp === 'mobile' ? 2 : 1,
    colorScheme: theme === 'light' ? 'light' : 'dark',
    reducedMotion: 'reduce',
  });
  const page = await context.newPage();
  page.on('pageerror', (e) => failures.push(`${name}: ${e.message}`));
  let picker = false;
  try {
    // An installed clock keeps running in real time, so where a page's timers stand at the check depends on load. --qa
    // pauses it at FIXED_NOW before the page loads (installed an hour early: pauseAt only moves forward), so time then
    // moves only by the scenes' runFor, the same on every machine.
    await page.clock.install({ time: qa ? FIXED_NOW.getTime() - 3_600_000 : FIXED_NOW });
    if (qa) await page.clock.pauseAt(FIXED_NOW);
    await page.addInitScript((t) => {
      localStorage.setItem('folo.theme', t);
      localStorage.setItem('folo.shoot', '1');
    }, theme);
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
    picker = await page.evaluate(() =>
      [...document.querySelectorAll('.phone-pick, [role=radiogroup][aria-label="Show"]')].some((el) => el.checkVisibility({ checkVisibilityCSS: true })),
    );
    count++;
    if (qa) return picker;
    const full = scene.fullPage ?? vp !== 'mobile';
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
    await page.screenshot({ path: resolve(outDir, `${name}.png`), fullPage: full });
  } catch (e) {
    failures.push(`${name}: ${e instanceof Error ? e.message : e}`);
  } finally {
    await context.close();
  }
  return picker;
}

/** --qa's states, run by a small pool below. @type {(() => Promise<void>)[]} */
const qaJobs = [];
try {
  for (const scene of scenes) {
    const themes = scene.themes ?? ['dark', 'light'];
    const viewports = scene.viewports ?? ['desktop', 'mobile'];
    if (!qa) {
      for (const theme of themes) {
        for (const vp of viewports) {
          if (!only || `${scene.name}--${theme}--${vp}`.includes(only)) await shoot(scene, theme, vp);
        }
      }
      continue;
    }
    // --qa: the tablet and phone sizes in dark (or the one theme a scene is made for), and light too behind a picker.
    const first = themes.includes('dark') ? 'dark' : themes[0];
    for (const vp of viewports.filter((v) => v === 'tablet' || v === 'mobile')) {
      if (only && !`${scene.name}--${first}--${vp}`.includes(only)) continue;
      qaJobs.push(async () => {
        if ((await shoot(scene, first, vp)) && first === 'dark' && themes.includes('light')) await shoot(scene, 'light', vp);
      });
    }
  }
  // Each state has its own context and paused clock, so --qa runs several at once: FOLO_QA_JOBS, default up to 8 (most
  // of a state's time is waiting on the page, not the CPU).
  const jobs = Math.max(1, Number(process.env.FOLO_QA_JOBS) || Math.min(8, availableParallelism()));
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
console.log(qa ? `QA: ${count} scene states checked in ${secs} s` : `${count} shots -> ${outDir} (${secs} s)`);
// The QA list (overflow and blank sections): printed last, boxed in yellow. The shoot saves it next to the shots and
// never fails on it; --qa fails on it.
const report = resolve(outDir, only ? `qa-${only.replace(/[^\w-]+/g, '_')}.txt` : 'qa.txt');
// The pool finishes states out of order: list them in scene order.
const order = (/** @type {string} */ n) => scenes.findIndex((s) => s.name === n.split('--')[0]);
overflows.sort((a, b) => order(a.name) - order(b.name) || a.name.localeCompare(b.name));
if (overflows.length) {
  const shots = new Set(overflows.map((o) => o.name)).size;
  const wide = overflows.filter((o) => o.px > 0).length;
  const head = `SHOOT QA: ${wide} element(s) overflow horizontally, ${overflows.length - wide} blank section(s), in ${shots} ${qa ? 'state' : 'shot'}(s)`;
  const lines = overflows.map((o) => {
    const [scene, theme, vp] = o.name.split('--');
    const what = o.px > 0 ? `+${String(o.px).padStart(4)}px` : ' blank';
    return `${scene.padEnd(28)} ${theme.padEnd(5)} ${vp.padEnd(7)} ${what}  ${o.selector}${o.text ? `  "${o.text}"` : ''}`;
  });
  if (!qa) writeFileSync(report, [head, ...lines, ''].join(EOL));
  const bar = '='.repeat(Math.max(head.length, 72));
  console.warn([yellow(bar), yellow(head), yellow(bar), ...lines, yellow(bar), ...(qa ? [] : [`(saved to ${report})`])].join(EOL));
  if (qa) process.exitCode = 1;
} else {
  if (!qa) rmSync(report, { force: true });
  console.log('Shoot QA: nothing overflows horizontally, no blank sections.');
}
if (failures.length) {
  console.error(`${failures.length} problem(s):\n  ${failures.join('\n  ')}`);
  process.exitCode = 1;
}
