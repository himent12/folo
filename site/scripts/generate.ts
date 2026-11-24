// Generates the site's pages from the repository's Markdown into site/.gen (Vite's root), with the search index,
// llms.txt / llms-full.txt and the images they use. Broken links, missing images and docs missing from the navigation
// are errors: `generate` throws, so `npm run build` and `npm run check` fail.
import { mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, posix, resolve } from 'node:path';
import type { MarkdownIt } from 'markdown-it';
import { BLOB_URL, GEN_DIR, LANDING_SRC, LLMS_SRC, NAV, REPO_DIR, slugFor } from './config.ts';
import { renderLanding } from './landing.ts';
import { ICONS, shell, toc } from './layout.ts';
import { clearListings, encodePath, extractHtmlAnchorHrefs, relUrl, resolveHref, rewriteLinks, rewriteMarkdown, type LinkContext, type PageRef } from './links.ts';
import { attr, createMarkdown, escapeHtml, parsePage, sections, withoutTitle, type Heading, type ParsedPage } from './markdown.ts';
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

  const navEntries = NAV.flatMap((g) => g.pages.map((p) => ({ group: g.title, label: p.label, src: p.src })));
  const validNav: { group: string; label: string; src: string }[] = [];
  const seenNav = new Set<string>();
  const seenSlugs = new Set<string>([slugFor(LANDING_SRC)]);
  const seenInputKeys = new Set<string>(['index']);
  let badNavSlug = false;
  for (const entry of navEntries) {
    const { src } = entry;
    if (src === LANDING_SRC) {
      problems.push(`${src} is already the landing page and cannot be listed in site navigation (site/scripts/config.ts, NAV)`);
      continue;
    }
    if (seenNav.has(src)) {
      problems.push(`${src} is listed more than once in site navigation (site/scripts/config.ts, NAV)`);
      continue;
    }
    seenNav.add(src);
    let slug: string;
    try {
      slug = slugFor(src);
    } catch (e) {
      problems.push(`${src}: ${(e as Error).message}`);
      badNavSlug = true;
      continue;
    }
    const inputKey = slug === '' ? 'index' : slug.replace(/\/$/, '');
    if (seenSlugs.has(slug) || seenInputKeys.has(inputKey)) {
      problems.push(`${src} collides on site output "${slug || 'index'}" with another page`);
    }
    seenSlugs.add(slug);
    seenInputKeys.add(inputKey);
    validNav.push(entry);
  }
  for (const src of expectedSources()) {
    if (!seenNav.has(src)) problems.push(`${src} is not in the site navigation (site/scripts/config.ts, NAV)`);
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
  for (const entry of validNav) parse(entry.src, entry.group, entry.label);
  const landing = pages.get(LANDING_SRC);
  if (!landing || badNavSlug) throw new SiteError(problems);

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

  const order = [...new Set(NAV.flatMap((g) => g.pages.map((p) => pages.get(p.src))).filter((p): p is Page => !!p && p !== landing))];
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
  for (const p of all) {
    outputs.push([resolve(GEN_DIR, 'public', 'md', p.src), rewriteMarkdown(p.markdown, p.parsed.tokens, p.src, `md/${p.src}`, pages, media)]);
  }
  outputs.push([resolve(GEN_DIR, 'public', 'llms.txt'), llmsTxt(md, pages, media, problems)]);
  checkDocsIndex(pages, order, problems);
  const full = all
    .map((p) => `<!-- source: ${p.src} (${BLOB_URL}/${p.src}) -->\n\n${rewriteMarkdown(p.markdown, p.parsed.tokens, p.src, 'llms-full.txt', pages, media).trim()}\n`)
    .join('\n---\n\n');
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
  return { inputs, sources: [LANDING_SRC, LLMS_SRC, 'design/tokens.json', ...navEntries.map((e) => e.src)] };
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
    <div class="doc-source">
      <a href="${BLOB_URL}/${encodePath(page.src)}">View <span class="mono">${escapeHtml(page.src)}</span> on GitHub ${ICONS.external}</a>
      <a class="doc-raw" href="${relUrl(page.slug, `md/${encodePath(page.src)}`)}" type="text/markdown">${ICONS.fileText} View raw Markdown</a>
    </div>
  </footer>
</article>
${tocItems ? `<aside class="toc" aria-labelledby="toc-h"><h2 class="toc-title" id="toc-h">On this page</h2><ul>${tocItems}</ul></aside>` : ''}
</div>`;
}

/** Checks that every NAV page (other than docs/README.md itself) is linked from docs/README.md so no doc is orphaned. */
function checkDocsIndex(pages: Map<string, Page>, order: Page[], problems: string[]): void {
  const docsIndex = pages.get('docs/README.md');
  if (!docsIndex) return;
  const linked = new Set<string>();
  const record = (href: string) => {
    const target = href.split('#')[0].split('?')[0];
    if (target) linked.add(target);
  };
  for (const tok of docsIndex.parsed.tokens) {
    if (tok.type === 'html_block') {
      for (const href of extractHtmlAnchorHrefs(tok.content)) record(href);
    }
    if (tok.type !== 'inline' || !tok.children) continue;
    for (const child of tok.children) {
      if (child.type === 'link_open') record(attr(child, 'href'));
      else if (child.type === 'html_inline' && !child.meta?.generated) {
        for (const href of extractHtmlAnchorHrefs(child.content)) record(href);
      }
    }
  }
  for (const page of order) {
    if (page.src === 'docs/README.md') continue;
    const expected = relUrl(docsIndex.slug, page.slug);
    if (!linked.has(expected)) problems.push(`${page.src} is orphaned: not linked from docs/README.md`);
  }
}

/**
 * The repository's llms.txt with every link validated via markdown-it tokens (including #anchors on llms.txt and on
 * site pages, exact-case paths, reference links and HTML anchors), rewritten to the site's Markdown copies (or GitHub
 * for other files), plus an orphan check so every site page appears.
 */
function llmsTxt(md: MarkdownIt, pages: Map<string, Page>, media: Map<string, string>, problems: string[]): string {
  let text: string;
  try {
    text = read(LLMS_SRC);
  } catch {
    problems.push(`${LLMS_SRC}: missing file`);
    return '';
  }
  const env: { headings?: Heading[] } = {};
  const tokens = md.parse(text, env);
  const llmsPages = new Map<string, PageRef>();
  llmsPages.set(LLMS_SRC, { src: LLMS_SRC, slug: '', anchors: new Set((env.headings ?? []).map((h) => h.id)) });
  for (const [src, p] of pages) {
    llmsPages.set(src, { src, slug: p.slug, anchors: new Set(p.parsed.headings.map((h) => h.id)) });
  }
  const ctx: LinkContext = { src: LLMS_SRC, slug: '', pages: llmsPages, media, errors: problems };
  const linkedPages = new Set<string>();

  const checkHref = (href: string, where: string) => {
    if (!href || /^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith('//')) return;
    const rawPath = href.split('#')[0].split('?')[0];
    if (rawPath && !rawPath.startsWith('/')) {
      try {
        const decoded = decodeURIComponent(rawPath);
        const norm = posix.normalize(decoded).replace(/\/$/, '');
        if (!norm.startsWith('..') && pages.has(norm)) {
          linkedPages.add(norm);
          if (decoded.endsWith('/')) {
            problems.push(`${LLMS_SRC}${where}: broken link "${href}": ${norm} is a file, not a directory`);
            return;
          }
        }
      } catch {
        // Malformed percent-encoding is reported below by resolveHref.
      }
    }
    resolveHref(href, ctx, where);
  };

  for (const tok of tokens) {
    const where = tok.map ? `:${tok.map[0] + 1}` : '';
    if (tok.type === 'html_block') {
      for (const href of extractHtmlAnchorHrefs(tok.content)) checkHref(href, where);
    }
    if (tok.type !== 'inline' || !tok.children) continue;
    for (const child of tok.children) {
      if (child.type === 'link_open') checkHref(attr(child, 'href'), where);
      else if (child.type === 'html_inline' && !child.meta?.generated) {
        for (const href of extractHtmlAnchorHrefs(child.content)) checkHref(href, where);
      }
    }
  }
  for (const src of pages.keys()) {
    if (!linkedPages.has(src)) problems.push(`${src} is orphaned: not linked from ${LLMS_SRC}`);
  }
  return rewriteMarkdown(text, tokens, LLMS_SRC, LLMS_SRC, pages, media);
}
