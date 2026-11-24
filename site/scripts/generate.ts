// Generates the site's pages from the repository's Markdown into site/.gen (Vite's root), with the search index,
// llms.txt / llms-full.txt and the images they use. Broken links, missing images and docs missing from the navigation
// are errors: `generate` throws, so `npm run build` and `npm run check` fail.
import { mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, posix, resolve } from 'node:path';
import type { MarkdownIt } from 'markdown-it';
import { BLOB_URL, GEN_DIR, LANDING_SRC, LLMS_SRC, NAV, REPO_DIR, slugFor } from './config.ts';
import { renderLanding } from './landing.ts';
import { ICONS, shell, toc } from './layout.ts';
import { clearListings, relUrl, resolveHref, rewriteLinks, type LinkContext, type PageRef } from './links.ts';
import { createMarkdown, escapeHtml, parsePage, sections, withoutTitle, type ParsedPage } from './markdown.ts';
import { tokensCss } from './tokens.ts';

export interface Generated {
  /** Vite inputs: name -> generated HTML file. */
  inputs: Record<string, string>;
  /** Repository-relative sources, for the dev server's watcher. */
  sources: string[];
}

export class SiteError extends Error {
  readonly problems: string[];
  constructor(problems: string[]) {
    super(`the docs site has ${problems.length} problem${problems.length === 1 ? '' : 's'}:\n  ${problems.join('\n  ')}`);
    this.problems = problems;
  }
}

interface Page extends PageRef {
  group: string;
  label: string;
  parsed: ParsedPage;
  markdown: string;
}

const read = (src: string) => readFileSync(resolve(REPO_DIR, src), 'utf8').replace(/\r\n/g, '\n');

/** Every Markdown file that must be a page: docs/*.md and examples/<name>/README.md. */
function expectedSources(): string[] {
  const docs = readdirSync(resolve(REPO_DIR, 'docs')).filter((f) => f.endsWith('.md')).map((f) => `docs/${f}`);
  const examples = readdirSync(resolve(REPO_DIR, 'examples'), { withFileTypes: true })
    .filter((d) => d.isDirectory() && readdirSync(resolve(REPO_DIR, 'examples', d.name)).includes('README.md'))
    .map((d) => `examples/${d.name}/README.md`);
  return [...docs, ...examples];
}

function writeFile(path: string, data: string | Buffer) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, data);
}

export async function generate({ write }: { write: boolean }): Promise<Generated> {
  const md = await createMarkdown();
  const problems: string[] = [];
  clearListings();

  const navSources = NAV.flatMap((g) => g.pages.map((p) => p.src));
  for (const src of expectedSources()) {
    if (!navSources.includes(src)) problems.push(`${src} is not in the site navigation (site/scripts/config.ts, NAV)`);
  }

  const pages = new Map<string, Page>();
  const parse = (src: string, group: string, label: string) => {
    let markdown: string;
    try {
      markdown = read(src);
    } catch {
      problems.push(`${src}: listed in the navigation but missing`);
      return;
    }
    try {
      const parsed = parsePage(md, markdown);
      const anchors = new Set(parsed.headings.map((h) => h.id));
      pages.set(src, { src, slug: slugFor(src), anchors, group, label, parsed, markdown });
    } catch (e) {
      problems.push(`${src}: ${(e as Error).message}`);
    }
  };
  parse(LANDING_SRC, '', 'Home');
  for (const g of NAV) for (const p of g.pages) parse(p.src, g.title, p.label);
  const landing = pages.get(LANDING_SRC);
  if (!landing) throw new SiteError(problems);

  // The landing page shows only some README sections: links into it may only use those anchors.
  landing.anchors = renderLanding(md, landing.parsed, '').anchors;

  const media = new Map<string, string>();
  for (const page of pages.values()) {
    const ctx: LinkContext = { src: page.src, slug: page.slug, pages: pages as Map<string, PageRef>, media, errors: problems };
    rewriteLinks(page.parsed.tokens, ctx);
  }

  const outputs: [string, string | Buffer][] = [];
  const inputs: Record<string, string> = {};
  const html = (slug: string, content: string) => {
    const file = resolve(GEN_DIR, slug, 'index.html');
    outputs.push([file, content]);
    inputs[slug === '' ? 'index' : slug.replace(/\/$/, '')] = file;
  };

  const lp = renderLanding(md, landing.parsed, '');
  html('', shell({ slug: '', title: 'FOLO', description: lp.description, main: lp.main, kind: 'landing' }));

  const order = NAV.flatMap((g) => g.pages.map((p) => pages.get(p.src))).filter((p): p is Page => !!p);
  order.forEach((page, i) => {
    html(page.slug, shell({ slug: page.slug, title: page.parsed.title, description: page.parsed.description, main: docMain(md, page, order[i - 1], order[i + 1]), kind: 'doc' }));
  });

  // Search: one entry per H2/H3 section (including the landing page's rendered excerpts), loaded on first use.
  const indexed = [landing, ...order];
  const index = {
    pages: indexed.map((p) => ({ t: p.parsed.title, l: p.label, u: p.slug, g: p.group })),
    sections: indexed.flatMap((p, i) =>
      (p === landing ? lp.sections : sections(withoutTitle(p.parsed.tokens), p.parsed.title)).map((s) => [i, s.title, s.id, s.text]),
    ),
  };
  outputs.push([resolve(GEN_DIR, 'public', 'search.json'), JSON.stringify(index)]);

  // For agents: llms.txt (the repository's index, pointed at the Markdown copies served next to the pages) and
  // llms-full.txt (every page's Markdown in one file), plus the copies themselves at md/<repository path>.
  const all = [landing, ...order];
  for (const p of all) outputs.push([resolve(GEN_DIR, 'public', 'md', p.src), p.markdown]);
  outputs.push([resolve(GEN_DIR, 'public', 'llms.txt'), llmsTxt(pages, problems)]);
  const full = all.map((p) => `<!-- source: ${p.src} (${BLOB_URL}/${p.src}) -->\n\n${p.markdown.trim()}\n`).join('\n---\n\n');
  outputs.push([resolve(GEN_DIR, 'public', 'llms-full.txt'), `${full}`]);

  for (const [repoPath, out] of media) outputs.push([resolve(GEN_DIR, 'public', out), readFileSync(resolve(REPO_DIR, repoPath))]);
  outputs.push([resolve(GEN_DIR, 'public', 'favicon.svg'), read('design/logo-icon.svg')]);
  outputs.push([resolve(GEN_DIR, 'tokens.css'), tokensCss(GEN_DIR)]);
  outputs.push([resolve(GEN_DIR, 'entry.ts'), "// Generated by site/scripts/generate.ts.\nimport './tokens.css';\nimport '../src/main.ts';\n"]);

  if (problems.length) throw new SiteError(problems);
  if (write) {
    rmSync(GEN_DIR, { recursive: true, force: true });
    for (const [path, data] of outputs) writeFile(path, data);
  }
  return { inputs, sources: [LANDING_SRC, LLMS_SRC, 'design/tokens.json', ...navSources] };
}

function docMain(m: MarkdownIt, page: Page, prev?: Page, next?: Page): string {
  const tokens = page.parsed.tokens;
  const h1 = tokens.findIndex((t) => t.type === 'heading_open' && t.tag === 'h1');
  const titleHtml = m.renderer.renderInline(tokens[h1 + 1].children ?? [], m.options, {});
  const body = m.renderer.render(withoutTitle(tokens), m.options, {});
  const tocItems = toc(page.parsed.headings);
  const pager = (p: Page | undefined, dir: 'prev' | 'next') =>
    p
      ? `<a class="pager-${dir}" href="${relUrl(page.slug, p.slug)}" rel="${dir}"><span class="pager-dir">${dir === 'prev' ? `${ICONS.arrowLeft} Previous` : `Next ${ICONS.arrowRight}`}</span><span class="pager-label">${escapeHtml(p.label)}</span></a>`
      : '<span></span>';
  return `<div class="doc-layout${tocItems ? '' : ' no-toc'}">
<article class="doc">
  <header class="doc-head">
    <p class="eyebrow">${escapeHtml(page.group)}</p>
    <h1 id="${page.parsed.headings[0]?.id ?? ''}">${titleHtml}</h1>
  </header>
  ${tocItems ? `<details class="toc-inline"><summary>On this page</summary><ul>${tocItems}</ul></details>` : ''}
  <div class="prose">
${body}
  </div>
  <footer class="doc-foot">
    <nav class="pager" aria-label="Previous and next page">${pager(prev, 'prev')}${pager(next, 'next')}</nav>
    <p class="doc-source"><a href="${BLOB_URL}/${page.src}">View <span class="mono">${escapeHtml(page.src)}</span> on GitHub ${ICONS.external}</a></p>
  </footer>
</article>
${tocItems ? `<aside class="toc" aria-labelledby="toc-h"><h2 class="toc-title" id="toc-h">On this page</h2><ul>${tocItems}</ul></aside>` : ''}
</div>`;
}

/** The repository's llms.txt with every link pointed at the site's Markdown copies (or GitHub for other files). */
function llmsTxt(pages: Map<string, Page>, problems: string[]): string {
  const text = read(LLMS_SRC);
  const ctx: LinkContext = { src: LLMS_SRC, slug: '', pages: new Map(), media: new Map(), errors: problems };
  return text.replace(/\]\(([^)\s]+)\)/g, (whole, href: string) => {
    const [path, hash] = href.split('#');
    const repoPath = posix.normalize(path);
    if (pages.has(repoPath)) return `](md/${repoPath}${hash ? `#${hash}` : ''})`;
    const url = resolveHref(href, ctx, '');
    return url === null ? whole : `](${url})`;
  });
}
