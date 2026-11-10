# FOLO docs site

A static documentation site generated from the repository's Markdown: `docs/*.md`, `README.md` (the landing page)
and `examples/*/README.md`. The Markdown stays the single source of truth; nothing here is copied by hand, and the
site is rebuilt from it on every build. It is not deployed anywhere yet: build it and serve it locally.

```sh
cd site
npm ci
npm run dev        # http://localhost:5173, regenerates and reloads when a source Markdown file changes
npm run build      # the static site in site/dist
npm run preview    # serves site/dist on http://localhost:4173
npm run check      # TypeScript, plus every link, anchor and image in the Markdown (generates in memory)
npm run shoot      # screenshots of the built site into site/shots (installed Chrome via playwright-core)
```

`site/dist` is plain static files with relative URLs, so any static file server works too (for example
`python -m http.server -d dist`), from any folder or subpath.

## What it does
- **Pages.** Every `docs/*.md` and `examples/*/README.md` becomes a page at `<name>/` (`docs/AGENT_COOKBOOK.md` →
  `agent-cookbook/`, `examples/shapes/README.md` → `examples/shapes/`). The sidebar groups and labels are in
  `scripts/config.ts` (`NAV`); a doc missing from `NAV` fails the build.
- **Landing page.** Built from `README.md` by heading: the pitch and highlights, "Why FOLO: the numbers" (the headline
  comparison, whose every number comes from docs/BENCHMARKS.md), the quickstart, Studio and Desktop with their
  screenshots, agent setup, the docs index (with the README's docs table as blurbs) and project status. Renaming one
  of those README headings fails the build rather than silently dropping a section.
- **Links.** Relative links are written for GitHub (`../docs/CLI.md#onnx-export`). The generator resolves each one
  against the repository: a page becomes its site URL and its `#anchor` is checked against that page's headings
  (GitHub-style ids, github-slugger); any other existing file or folder links to GitHub; anything else is an error.
  Plain mentions of a page in prose or inline code (`docs/MODEL.md`) become links too. Path case is checked exactly,
  so a link that only works on Windows fails here as well.
- **Search.** A prebuilt index (`search.json`, one entry per H2/H3 section) is fetched on first use and searched in
  the browser; no service. Open it with `/` or Ctrl+K (⌘K), move with ↑ ↓, open with Enter.
- **Reading aids.** A per-page table of contents that tracks the section being read (inline and collapsible below
  1200 px), copy buttons on code blocks, build-time highlighting (shiki), previous / next links, a link to the
  source file, and roadmap status markers for `- [x]` / `- [~]` / `- [ ]`.
- **For agents.** `llms.txt` (the repository's llms.txt with its links pointed at the site), `llms-full.txt` (every
  page's Markdown in one file) and the raw Markdown of each page at `md/<repository path>`.
- **Theme.** Colours, type, spacing, radii and motion come from `design/tokens.json`, the tokens Studio and Desktop
  use, generated into CSS variables at build time (`scripts/tokens.ts`; the same variable names as Studio). With no
  saved choice the site follows the system's light or dark setting; the top-bar button switches and remembers it.
  Fonts are the self-hosted Geist and Geist Mono from `design/fonts`. Highlighting colours are the tokens' status
  `text` colours, which meet AA on every surface.
- **Layout.** Three columns from 1200 px, no side table of contents below that, and below 960 px the sidebar becomes
  a drawer. Every control has a visible focus ring, there is a skip link, and wide tables and code blocks become
  keyboard-scrollable regions when they overflow.

## Layout of this folder
```
scripts/config.ts     sources, sidebar groups, URL scheme
scripts/generate.ts   Markdown -> .gen/ (pages, search.json, llms*.txt, media); throws on any broken link
scripts/markdown.ts   markdown-it setup: heading ids, highlighting, copy buttons, tables, status markers
scripts/links.ts      link and image resolution and checking
scripts/landing.ts    the landing page from README.md
scripts/layout.ts     the HTML shell (top bar, sidebar, table of contents, search dialog)
scripts/tokens.ts     design/tokens.json -> CSS variables
scripts/check.ts      `npm run check`
scripts/shoot.ts      `npm run shoot`
src/                  the client script and styles
vite.config.ts        runs the generator; .gen/ is Vite's root, dist/ the output
```
`.gen/`, `dist/` and `shots/` are generated and ignored by git.

CI's `site` workflow (`.github/workflows/site.yml`, docs/CI.md) runs `npm ci && npm run check && npm run build`
when the site, a Markdown file or an example changes.

## Adding a page
Add the Markdown file under `docs/` (or an `examples/<name>/README.md`), then add it to `NAV` in `scripts/config.ts`
under the right group with a short sidebar label. Its H1 is the page title and its first paragraph the page
description.
