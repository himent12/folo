// The docs site's client: theme switch, mobile navigation, search, table-of-contents tracking, copy buttons and
// scrollable tables. Every page works without it; this only adds behaviour.
import './styles/base.css';
import './styles/layout.css';
import './styles/prose.css';
import './styles/landing.css';
import './styles/search.css';
import { escapeHtml, highlight, loadIndex, search, type Hit } from './search.ts';

const root = document.body.dataset.root ?? './';
const $ = <T extends Element>(sel: string, from: ParentNode = document) => from.querySelector<T>(sel);
const $$ = <T extends Element>(sel: string, from: ParentNode = document) => [...from.querySelectorAll<T>(sel)];

// A polite live region for "Copied" and similar announcements.
const live = document.createElement('p');
live.className = 'visually-hidden';
live.setAttribute('role', 'status');
live.setAttribute('aria-live', 'polite');
live.setAttribute('aria-atomic', 'true');
document.body.append(live);
let announceRaf = 0;
function announce(msg: string) {
  cancelAnimationFrame(announceRaf);
  live.textContent = '';
  announceRaf = requestAnimationFrame(() => {
    live.textContent = msg;
  });
}

// Theme: a saved choice wins; otherwise the CSS follows the system.
function currentTheme(): 'light' | 'dark' {
  const set = document.documentElement.dataset.theme;
  if (set === 'light' || set === 'dark') return set;
  return matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}
const themeBtn = $<HTMLButtonElement>('.theme-toggle');
const themeColorMeta = $<HTMLMetaElement>('meta[name="theme-color"]');
function syncTheme() {
  themeBtn?.setAttribute('aria-label', currentTheme() === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
  if (themeColorMeta) {
    const bg = getComputedStyle(document.documentElement).getPropertyValue('--bg-panel').trim();
    if (bg) themeColorMeta.content = bg;
  }
}
themeBtn?.addEventListener('click', () => {
  const next = currentTheme() === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem('folo.docs.theme', next);
  } catch {
    // Storage can be unavailable (private windows); the choice then lasts for this page.
  }
  syncTheme();
});
matchMedia('(prefers-color-scheme: light)').addEventListener('change', syncTheme);
syncTheme();

// Mobile navigation drawer.
const navBtn = $<HTMLButtonElement>('.nav-toggle');
const scrim = $<HTMLElement>('.scrim');
const sidebar = $<HTMLElement>('#sidebar');
const desktopNav = matchMedia('(min-width: 960px)');
const navInertTargets = $$<HTMLElement>('.skip, #main, .topbar > :not(.nav-toggle)');
const isNavOpen = () => document.documentElement.classList.contains('nav-open');

function setNav(open: boolean, restoreFocus = true) {
  if (open && desktopNav.matches) return;
  if (open === isNavOpen()) return;
  document.documentElement.classList.toggle('nav-open', open);
  for (const el of navInertTargets) el.inert = open;
  navBtn?.setAttribute('aria-expanded', String(open));
  navBtn?.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
  if (scrim) scrim.hidden = !open;
  if (open && sidebar) ($<HTMLElement>('a[aria-current="page"]', sidebar) ?? $<HTMLElement>('a', sidebar))?.focus();
  else if (!open && restoreFocus) navBtn?.focus();
}
navBtn?.addEventListener('click', () => setNav(!isNavOpen()));
scrim?.addEventListener('click', () => setNav(false));
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && isNavOpen() && !dialog?.open) {
    e.preventDefault();
    setNav(false);
  }
});
// Close the drawer if focus somehow leaves both the drawer and its toggle button.
sidebar?.addEventListener('focusout', (e) => {
  if (!isNavOpen()) return;
  const to = e.relatedTarget as Node | null;
  if (to && !sidebar.contains(to) && to !== navBtn) setNav(false, false);
});
desktopNav.addEventListener('change', (e) => {
  if (e.matches) setNav(false, false);
});
$<HTMLElement>('a[aria-current="page"]', sidebar ?? document)?.scrollIntoView({ block: 'center' });

// Search.
const dialog = $<HTMLDialogElement>('#search');
const input = $<HTMLInputElement>('#search-input');
const results = $<HTMLUListElement>('#search-results');
const status = $<HTMLElement>('.search-status', dialog ?? document);
let hits: Hit[] = [];
let active = -1;
let lastPointerX = Number.NaN;
let lastPointerY = Number.NaN;

document.addEventListener(
  'pointermove',
  (e) => {
    if (e.pointerType === 'touch' || dialog?.open) return;
    lastPointerX = e.clientX;
    lastPointerY = e.clientY;
  },
  { passive: true },
);

function renderHits(query: string) {
  if (!results || !status || !input) return;
  const terms = [...new Set(query.toLowerCase().split(/\s+/).filter(Boolean))];
  results.innerHTML = hits
    .map(
      (h, i) =>
        `<li class="search-result" id="sr-${i}" data-index="${i}" role="option" aria-selected="${i === active}"><a href="${h.href}" tabindex="-1"><span class="sr-path">${escapeHtml(h.path)}</span><span class="sr-title">${highlight(h.title, terms)}</span><span class="sr-snippet">${h.snippet}</span></a></li>`,
    )
    .join('');
  input.setAttribute('aria-expanded', String(hits.length > 0));
  if (active >= 0) input.setAttribute('aria-activedescendant', `sr-${active}`);
  else input.removeAttribute('aria-activedescendant');
  status.textContent = !query.trim()
    ? 'Type to search every page. ↑ ↓ to move, Enter to open.'
    : hits.length
      ? `${hits.length === 30 ? 'Top 30' : hits.length} result${hits.length === 1 ? '' : 's'}. ↑ ↓ to move, Enter to open.`
      : `No results for “${query.trim()}”.`;
}

async function runSearch() {
  if (!input || !status) return;
  const q = input.value;
  try {
    const all = await loadIndex(root);
    if (q !== input.value) return;
    hits = search(all, q, root);
    active = hits.length ? 0 : -1;
    renderHits(q);
  } catch {
    status.textContent = 'Search is unavailable: the index did not load.';
  }
}

function openSearch() {
  if (!dialog || !input || dialog.open) return;
  setNav(false, true);
  dialog.showModal();
  input.select();
  void runSearch();
}

function setActive(next: number, scroll: boolean) {
  if (!hits.length || !input || !results) return;
  const clamped = Math.max(0, Math.min(hits.length - 1, next));
  if (clamped !== active) {
    if (active >= 0) $(`#sr-${active}`, results)?.setAttribute('aria-selected', 'false');
    active = clamped;
    $(`#sr-${active}`, results)?.setAttribute('aria-selected', 'true');
    input.setAttribute('aria-activedescendant', `sr-${active}`);
  }
  if (scroll) $(`#sr-${active}`, results)?.scrollIntoView({ block: 'nearest' });
}

function moveActive(by: number) {
  if (!hits.length) return;
  const next = active < 0 ? (by > 0 ? 0 : hits.length - 1) : (active + by + hits.length) % hits.length;
  setActive(next, true);
}

$('.search-trigger')?.addEventListener('click', openSearch);
$('.search-close')?.addEventListener('click', () => dialog?.close());
input?.addEventListener('input', () => void runSearch());
input?.addEventListener('keydown', (e) => {
  if (e.isComposing) return;
  const modified = e.altKey || e.ctrlKey || e.metaKey;
  if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && !modified && !e.shiftKey && hits.length) {
    e.preventDefault();
    moveActive(e.key === 'ArrowDown' ? 1 : -1);
  } else if ((e.key === 'Home' || e.key === 'End') && !modified && !e.shiftKey && hits.length) {
    e.preventDefault();
    setActive(e.key === 'Home' ? 0 : hits.length - 1, true);
  } else if (e.key === 'Enter' && hits[active]) {
    e.preventDefault();
    dialog?.close();
    location.href = hits[active].href;
  } else if (e.key === 'Escape') {
    // A search input's own Escape only clears it; the shortcut hint says Esc closes.
    e.preventDefault();
    dialog?.close();
  }
});
dialog?.addEventListener('pointermove', (e) => {
  if (e.pointerType === 'touch') return;
  const initial = Number.isNaN(lastPointerX);
  const stationary = e.clientX === lastPointerX && e.clientY === lastPointerY;
  lastPointerX = e.clientX;
  lastPointerY = e.clientY;
  if (initial || stationary) return;
  const item = (e.target as Element).closest<HTMLElement>('.search-result');
  if (!item) return;
  const idx = Number(item.dataset.index);
  if (Number.isInteger(idx) && idx >= 0 && idx < hits.length) setActive(idx, false);
});
// Back to the button that opened it (the native dialog restores focus to whatever had it, which may be <body>).
dialog?.addEventListener('close', () => {
  if (document.activeElement === document.body || !document.activeElement) $<HTMLElement>('.search-trigger')?.focus();
});
dialog?.addEventListener('click', (e) => {
  if (e.target === dialog) dialog.close();
  // A result on this page only scrolls, so close the dialog for it too.
  if ((e.target as Element).closest('.search-result a')) dialog.close();
});
document.addEventListener('keydown', (e) => {
  const t = e.target as HTMLElement;
  const typing = t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName);
  if ((e.key === 'k' && (e.metaKey || e.ctrlKey)) || (e.key === '/' && !typing && !e.metaKey && !e.ctrlKey && !e.altKey)) {
    e.preventDefault();
    openSearch();
  }
});
if (/Mac|iPhone|iPad/.test(navigator.platform)) {
  const kbd = $('.search-trigger .kbd');
  if (kbd) kbd.textContent = '⌘K';
}

// Table of contents: mark the section being read and keep the active TOC link visible without scrolling the page.
const tocBox = $<HTMLElement>('.toc');
const tocLinks = $$<HTMLAnchorElement>('.toc a');
if (tocLinks.length) {
  const targets = tocLinks
    .map((a) => [a, document.getElementById(decodeURIComponent(a.hash.slice(1)))] as const)
    .filter((x): x is readonly [HTMLAnchorElement, HTMLElement] => !!x[1]);
  let activeTocLink: HTMLAnchorElement | null = null;
  let queued = false;
  const revealActiveToc = () => {
    if (!activeTocLink || !tocBox || tocBox.scrollHeight <= tocBox.clientHeight || tocBox.offsetParent === null) return;
    const boxRect = tocBox.getBoundingClientRect();
    const linkRect = activeTocLink.getBoundingClientRect();
    if (linkRect.top < boxRect.top) tocBox.scrollTop += linkRect.top - boxRect.top;
    else if (linkRect.bottom > boxRect.bottom) tocBox.scrollTop += linkRect.bottom - boxRect.bottom;
  };
  const update = () => {
    queued = false;
    if (!targets.length) return;
    const line = 120;
    let cur = targets[0];
    for (const t of targets) if (t[1].getBoundingClientRect().top <= line) cur = t;
    // At the very bottom, the last heading is current even when it cannot reach the line.
    if (innerHeight + scrollY >= document.documentElement.scrollHeight - 2) cur = targets[targets.length - 1];
    if (cur[0] !== activeTocLink) {
      activeTocLink?.removeAttribute('aria-current');
      activeTocLink = cur[0];
      activeTocLink.setAttribute('aria-current', 'true');
      revealActiveToc();
    }
  };
  const scheduleUpdate = () => {
    if (!queued) requestAnimationFrame(update);
    queued = true;
  };
  addEventListener('scroll', scheduleUpdate, { passive: true });
  addEventListener(
    'resize',
    () => {
      update();
      revealActiveToc();
    },
    { passive: true },
  );
  update();
}
const tocInline = $<HTMLDetailsElement>('.toc-inline');
tocInline?.addEventListener('click', (e) => {
  if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
  if ((e.target as Element).closest('a')) tocInline.open = false;
});

// Copy buttons on code blocks.
const copyTimers = new WeakMap<HTMLButtonElement, number>();
const copySeq = new WeakMap<HTMLButtonElement, number>();
let copyOp = 0;

async function copyText(text: string, isCurrent: () => boolean): Promise<boolean> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Fall through to execCommand fallback (e.g. non-secure HTTP context or denied permission).
    }
  }
  if (!isCurrent()) return false;
  const prevFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.setAttribute('readonly', '');
  ta.style.position = 'fixed';
  ta.style.top = '0';
  ta.style.left = '-9999px';
  ta.style.opacity = '0';
  document.body.append(ta);
  try {
    ta.focus({ preventScroll: true });
    ta.select();
    ta.setSelectionRange(0, text.length);
    return document.execCommand('copy');
  } catch {
    return false;
  } finally {
    ta.remove();
    prevFocus?.focus({ preventScroll: true });
  }
}

document.addEventListener('click', async (e) => {
  const btn = (e.target as Element).closest<HTMLButtonElement>('.copy');
  const code = btn?.closest('.code')?.querySelector('pre');
  if (!btn || !code) return;
  const seq = (copySeq.get(btn) ?? 0) + 1;
  copySeq.set(btn, seq);
  const op = ++copyOp;
  const ok = await copyText(code.textContent ?? '', () => copyOp === op);
  if (copySeq.get(btn) !== seq) return;
  const prevTimer = copyTimers.get(btn);
  if (prevTimer !== undefined) clearTimeout(prevTimer);
  btn.dataset.copyState = ok ? 'copied' : 'failed';
  const label = btn.querySelector('.copy-label');
  if (label) label.textContent = ok ? 'Copied' : 'Failed';
  announce(ok ? 'Copied to the clipboard' : 'Could not copy to the clipboard');
  copyTimers.set(
    btn,
    window.setTimeout(() => {
      copyTimers.delete(btn);
      delete btn.dataset.copyState;
      if (label) label.textContent = 'Copy';
    }, ok ? 1600 : 2400),
  );
});

// Wide tables and code blocks scroll sideways: make those reachable by keyboard and name them for screen readers.
function markScrollers() {
  for (const el of $$<HTMLElement>('.table-wrap, .shiki')) {
    const scrolls = el.scrollWidth > el.clientWidth + 1;
    if (scrolls) {
      el.tabIndex = 0;
      if (el.classList.contains('table-wrap')) {
        el.setAttribute('role', 'region');
        el.setAttribute('aria-label', 'Table (scrolls sideways)');
      }
    } else {
      el.removeAttribute('tabindex');
      el.removeAttribute('role');
      el.removeAttribute('aria-label');
    }
  }
}
markScrollers();
let resizeTimer = 0;
addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(markScrollers, 150);
});
document.fonts?.ready.then(markScrollers);
