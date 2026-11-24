// The HTML shell shared by every page: top bar, sidebar, table of contents, search dialog.
import { NAV, REPO_URL, slugFor } from './config.ts';
import { relUrl } from './links.ts';
import { escapeHtml, type Heading } from './markdown.ts';
import { themeColors } from './tokens.ts';

export const SITE_NAME = 'FOLO docs';

const icon = (body: string, size = 16) =>
  `<svg viewBox="0 0 16 16" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;

export const ICONS = {
  menu: icon('<path d="M2.5 4.5h11M2.5 8h11M2.5 11.5h11"/>'),
  close: icon('<path d="m4 4 8 8M12 4l-8 8"/>'),
  search: icon('<circle cx="7" cy="7" r="4.25"/><path d="m10.25 10.25 3.25 3.25"/>'),
  sun: icon('<circle cx="8" cy="8" r="2.75"/><path d="M8 1.5v1.25M8 13.25v1.25M1.5 8h1.25M13.25 8h1.25M3.4 3.4l.9.9M11.7 11.7l.9.9M3.4 12.6l.9-.9M11.7 4.3l.9-.9"/>'),
  moon: icon('<path d="M13.25 9.6A5.5 5.5 0 0 1 6.4 2.75a5.5 5.5 0 1 0 6.85 6.85Z"/>'),
  monitor: icon('<rect x="2" y="2.75" width="12" height="8.5" rx="1.5"/><path d="M5.5 13.75h5M8 11.25v2.5"/>'),
  github: `<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M8 .2a8 8 0 0 0-2.53 15.59c.4.07.55-.17.55-.38v-1.33c-2.23.48-2.7-1.07-2.7-1.07-.36-.92-.89-1.17-.89-1.17-.73-.5.05-.49.05-.49.8.06 1.23.83 1.23.83.71 1.22 1.87.87 2.33.66.07-.52.28-.87.5-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.03 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0 0 8 .2Z"/></svg>`,
  arrowLeft: icon('<path d="M13 8H3M7 4 3 8l4 4"/>'),
  arrowRight: icon('<path d="M3 8h10M9 4l4 4-4 4"/>'),
  external: icon('<path d="M6.5 3.5h-3v9h9v-3M9.5 2.5h4v4M13.5 2.5 7.5 8.5"/>', 12),
};

// The wordmark (design/logo-wordmark.svg): "folo" with its last o as the accent dot.
export const LOGO = `<svg class="logo" viewBox="0 0 207 93" width="35.6" height="16" fill="none" aria-hidden="true"><path d="M16 91V29A20 20 0 0 1 36 9h8" stroke="currentColor" stroke-width="16"/><path d="M0 38h38" stroke="currentColor" stroke-width="14"/><circle cx="74" cy="61" r="23" stroke="currentColor" stroke-width="16"/><path d="M125 1v90" stroke="currentColor" stroke-width="16"/><circle cx="176" cy="61" r="31" fill="var(--accent-solid)"/></svg>`;

// Before first paint: the saved theme, else the system's (the CSS already follows the system without a script).
function themeInitScript(darkBg: string, lightBg: string): string {
  return `<script>try{let t;try{t=localStorage.getItem('folo.docs.theme')}catch{}const r=document.documentElement;if(t==='light'||t==='dark')r.dataset.theme=t;const m=document.querySelector('meta[name="theme-color"]');if(m)m.content=t==='light'||(t!=='dark'&&matchMedia('(prefers-color-scheme: light)').matches)?'${lightBg}':'${darkBg}'}catch{}</script>`;
}

export interface ShellPage {
  /** Site path of this page ('' is the landing page). */
  slug: string;
  title: string;
  description: string;
  /** The <main> content. */
  main: string;
  /** Extra class on <body>. */
  kind: 'doc' | 'landing';
}

function sidebar(slug: string): string {
  const groups = NAV.map((g) => {
    const items = g.pages
      .map((p) => {
        const to = slugFor(p.src);
        const cur = to === slug ? ' aria-current="page"' : '';
        return `<li><a href="${relUrl(slug, to)}"${cur}>${escapeHtml(p.label)}</a></li>`;
      })
      .join('');
    return `<div class="nav-group"><h2 class="nav-title">${escapeHtml(g.title)}</h2><ul>${items}</ul></div>`;
  }).join('');
  return `<aside class="sidebar" id="sidebar" aria-label="Documentation"><nav class="sidebar-nav">${groups}</nav></aside>`;
}

/** "On this page": H2s and H3s. Empty when there are fewer than two. */
export function toc(headings: Heading[]): string {
  const items = headings.filter((h) => h.level === 2 || h.level === 3);
  if (items.length < 2) return '';
  return items
    .map((h) => `<li class="toc-l${h.level}"><a href="#${h.id}">${escapeHtml(h.text)}</a></li>`)
    .join('');
}

export function shell(p: ShellPage): string {
  const { dark: darkBg, light: lightBg } = themeColors();
  const root = relUrl(p.slug, '');
  const title = p.slug === '' ? `${SITE_NAME}: FOLO Only Looks Once` : `${p.title} · ${SITE_NAME}`;
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light">
<meta name="theme-color" content="${darkBg}">
<title>${escapeHtml(title)}</title>
<meta name="description" content="${escapeHtml(p.description)}">
<link rel="icon" type="image/svg+xml" href="${root}favicon.svg">
<link rel="alternate" type="text/plain" title="llms.txt" href="${root}llms.txt">
${themeInitScript(darkBg, lightBg)}
<script type="module" src="/entry.ts"></script>
</head>
<body class="kind-${p.kind}" data-root="${root}">
<a class="skip" href="#main">Skip to content</a>
<header class="topbar">
  <button class="icon-btn nav-toggle" type="button" aria-controls="sidebar" aria-expanded="false" aria-label="Open navigation"><span class="i-menu">${ICONS.menu}</span><span class="i-close">${ICONS.close}</span></button>
  <a class="brand" href="${root}" aria-label="FOLO docs home"><span class="brand-lockup">${LOGO}<span class="brand-product">Docs</span></span></a>
  <button class="search-trigger" type="button" aria-haspopup="dialog" aria-controls="search" aria-label="Search docs" aria-keyshortcuts="/ Control+K Meta+K">${ICONS.search}<span class="search-trigger-label">Search docs</span><kbd class="kbd">/</kbd></button>
  <div class="topbar-end">
    <a class="icon-btn" href="${REPO_URL}" aria-label="FOLO on GitHub">${ICONS.github}</a>
    <button class="icon-btn theme-toggle" type="button" aria-label="Switch theme"><span class="i-system">${ICONS.monitor}</span><span class="i-sun">${ICONS.sun}</span><span class="i-moon">${ICONS.moon}</span></button>
  </div>
</header>
<div class="scrim" hidden></div>
<div class="shell">
${sidebar(p.slug)}
<main class="main" id="main" tabindex="-1">
${p.main}
</main>
</div>
<dialog class="search" id="search" aria-label="Search the docs">
  <div class="search-box">
    <div class="search-field">${ICONS.search}<input id="search-input" type="search" placeholder="Search the docs" autocomplete="off" spellcheck="false" role="combobox" aria-expanded="false" aria-controls="search-results" aria-autocomplete="list"><button class="search-close" type="button" aria-label="Close search"><kbd class="kbd">Esc</kbd><span class="search-close-icon">${ICONS.close}</span></button></div>
    <ul class="search-results" id="search-results" role="listbox" aria-label="Results"></ul>
    <div class="search-foot">
      <p class="search-status" role="status" aria-live="polite"></p>
      <button class="search-retry" type="button" hidden>Retry</button>
    </div>
  </div>
</dialog>
</body>
</html>
`;
}
