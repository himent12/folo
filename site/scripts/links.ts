// Link rewriting and checking. Relative links in the Markdown are written for GitHub (`../docs/CLI.md#onnx-export`);
// on the site they must point at pages. Every link is resolved against the repository: a page gets its site URL (and
// its anchor is checked against that page's headings), any other existing file or folder gets its GitHub URL, and
// anything else is an error that fails the build. Plain-text mentions of a page (`docs/MODEL.md`) become links too.
import { readdirSync, statSync } from 'node:fs';
import { posix, resolve } from 'node:path';
import type { Token } from 'markdown-it';
import { BLOB_URL, REPO_DIR, TREE_URL } from './config.ts';
import { attr } from './markdown.ts';
import { imageSize } from './media.ts';

export interface PageRef {
  src: string;
  slug: string;
  anchors: Set<string>;
}

export interface LinkContext {
  src: string;
  slug: string;
  pages: Map<string, PageRef>;
  /** Repository-relative image path -> site path, filled in as images are found. */
  media: Map<string, string>;
  errors: string[];
}

/** URL of `to` relative to the page at `from` (both site paths; pages are folders with an index.html). */
export function relUrl(from: string, to: string): string {
  const fromDir = `/${from}`.replace(/\/$/, '') || '/';
  const isDir = to === '' || to.endsWith('/');
  const target = `/${to}`.replace(/\/$/, '') || '/';
  let rel = posix.relative(fromDir, target);
  if (isDir) rel = rel ? `${rel}/` : './';
  return rel;
}

const listings = new Map<string, Set<string>>();

/** Forgets cached directory listings; each generation starts fresh, so the dev server sees added or removed files. */
export function clearListings(): void {
  listings.clear();
}

/** Whether a repository-relative path exists with exactly this spelling (Windows and macOS ignore case; Linux CI does not). */
export function repoPathKind(path: string): 'file' | 'dir' | null {
  if (path === '' || path === '.') return 'dir';
  let dir = REPO_DIR;
  for (const part of path.split('/')) {
    let names = listings.get(dir);
    if (!names) {
      try {
        names = new Set(readdirSync(dir));
      } catch {
        return null;
      }
      listings.set(dir, names);
    }
    if (!names.has(part)) return null;
    dir = resolve(dir, part);
  }
  try {
    return statSync(dir).isDirectory() ? 'dir' : 'file';
  } catch {
    return null;
  }
}

/** A repository path as a URL path: each segment percent-encoded (`plot#1.png` must not become a fragment). */
export function encodePath(path: string): string {
  return path.split('/').map(encodeURIComponent).join('/');
}

/** Splits a URL reference into path, query (with its "?") and fragment (without its "#"). */
function splitRef(ref: string): { path: string; query: string; hash: string } {
  const hashAt = ref.indexOf('#');
  const hash = hashAt < 0 ? '' : ref.slice(hashAt + 1);
  const beforeHash = hashAt < 0 ? ref : ref.slice(0, hashAt);
  const queryAt = beforeHash.indexOf('?');
  return {
    path: queryAt < 0 ? beforeHash : beforeHash.slice(0, queryAt),
    query: queryAt < 0 ? '' : beforeHash.slice(queryAt),
    hash,
  };
}

function decode(s: string): string | null {
  try {
    return decodeURIComponent(s);
  } catch {
    return null;
  }
}

function lineOf(tok: Token | undefined): string {
  return tok?.map ? `:${tok.map[0] + 1}` : '';
}

/** Repository path of a relative (URL-encoded) reference from `src`; null if it leaves the repository or is malformed. */
function resolveRepoPath(src: string, path: string): string | null {
  const decoded = decode(path);
  if (decoded === null) return null;
  const joined = posix.normalize(posix.join(posix.dirname(src), decoded));
  if (joined.startsWith('..') || posix.isAbsolute(joined)) return null;
  return joined.replace(/\/$/, '');
}

/** The site URL for `href` as written in `ctx.src`, or null (with an error recorded) when it is broken. */
export function resolveHref(href: string, ctx: LinkContext, where: string): string | null {
  if (/^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith('//')) return href;
  const fail = (why: string) => {
    ctx.errors.push(`${ctx.src}${where}: broken link "${href}": ${why}`);
    return null;
  };
  const { path, query, hash } = splitRef(href);
  const anchor = hash ? decode(hash) : '';
  if (anchor === null) return fail('malformed #fragment');
  const tail = `${query}${hash ? `#${hash}` : ''}`;
  if (path === '') {
    const self = ctx.pages.get(ctx.src);
    if (anchor && self && !self.anchors.has(anchor)) return fail(`no heading #${hash} on this page`);
    return href;
  }
  if (path.startsWith('/')) return fail('absolute paths do not work on GitHub or on the site; use a relative path');
  const repoPath = resolveRepoPath(ctx.src, path);
  if (repoPath === null) return fail('it points outside the repository or is not a valid URL path');
  const page = ctx.pages.get(repoPath);
  if (page) {
    if (anchor && !page.anchors.has(anchor)) return fail(`${repoPath} has no heading #${hash}`);
    const url = page.src === ctx.src ? '' : relUrl(ctx.slug, page.slug);
    return `${url || (tail.startsWith('#') ? '' : './')}${tail}`;
  }
  const kind = repoPathKind(repoPath);
  if (!kind) return fail(`${repoPath} does not exist`);
  return `${kind === 'dir' ? TREE_URL : BLOB_URL}/${encodePath(repoPath)}${tail}`;
}

/** Site URL and repository path of a local image; null for remote images and (with an error) for missing ones. */
function resolveImage(src: string, ctx: LinkContext, where: string): { url: string; repoPath: string } | null {
  if (/^[a-z][a-z0-9+.-]*:/i.test(src) || src.startsWith('//')) return null;
  const { path, query, hash } = splitRef(src);
  const repoPath = resolveRepoPath(ctx.src, path);
  if (repoPath === null || repoPathKind(repoPath) !== 'file') {
    ctx.errors.push(`${ctx.src}${where}: missing image "${src}"`);
    return null;
  }
  ctx.media.set(repoPath, `media/${repoPath}`);
  // The query and fragment (an SVG view, a media fragment) stay on the URL; only the path names the file.
  return { url: `${relUrl(ctx.slug, `media/${encodePath(repoPath)}`)}${query}${hash ? `#${hash}` : ''}`, repoPath };
}

/** Comments, and start tags (attributes may be double-quoted, single-quoted or unquoted). */
const HTML_PART = /<!--[\s\S]*?(?:-->|$)|<[a-zA-Z][^\s/>]*(?:\s+[^\s"'>\/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'=<>`]+))?)*\s*\/?>/g;
const HTML_ATTR = /(\s)(href|src)(\s*=\s*)(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))/gi;

/**
 * Links and images in raw HTML (`<a href>`, `<img src>`, ...) get the same checks and rewriting as Markdown ones, so
 * HTML can't slip a broken or GitHub-only `.md` link past the build. Only attributes inside start tags count: a link
 * in an HTML comment renders nothing and is left alone.
 */
function rewriteHtml(html: string, ctx: LinkContext, where: string): string {
  return html.replace(HTML_PART, (part) => {
    if (part.startsWith('<!--')) return part;
    return part.replace(HTML_ATTR, (whole, sp: string, name: string, eq: string, dq?: string, sq?: string, bare?: string) => {
      const value = dq ?? sq ?? bare ?? '';
      const quote = dq !== undefined ? '"' : sq !== undefined ? "'" : '"';
      const url = name.toLowerCase() === 'href' ? resolveHref(value, ctx, where) : (resolveImage(value, ctx, where)?.url ?? null);
      return url === null ? whole : `${sp}${name}${eq}${quote}${url}${quote}`;
    });
  });
}

const MENTION = /(?<![\w/.-])((?:docs\/[A-Z][A-Z0-9_-]*\.md)|(?:examples\/[a-z0-9_-]+\/README\.md))(#[a-z0-9_-]+)?(?![\w/-])/g;

function linkTokens(TokenCtor: new (type: string, tag: string, nesting: -1 | 0 | 1) => Token, href: string, inner: Token[]): Token[] {
  const open = new TokenCtor('link_open', 'a', 1);
  open.attrs = [['href', href]];
  const close = new TokenCtor('link_close', 'a', -1);
  return [open, ...inner, close];
}

/** Rewrites links and images in place, and links plain mentions of other pages. */
export function rewriteLinks(tokens: Token[], ctx: LinkContext): void {
  for (const block of tokens) {
    if (block.type === 'html_block') block.content = rewriteHtml(block.content, ctx, lineOf(block));
    if (block.type !== 'inline' || !block.children) continue;
    const where = lineOf(block);
    const TokenCtor = block.constructor as new (type: string, tag: string, nesting: -1 | 0 | 1) => Token;
    const out: Token[] = [];
    let depth = 0;
    for (const t of block.children) {
      if (t.type === 'link_open') {
        depth++;
        const href = attr(t, 'href');
        const url = resolveHref(href, ctx, where);
        if (url !== null) t.attrSet('href', url);
        out.push(t);
        continue;
      }
      if (t.type === 'link_close') depth--;
      // Raw HTML written in the Markdown; the heading anchors and status markers added by markdown.ts are already right.
      if (t.type === 'html_inline' && !t.meta?.generated) t.content = rewriteHtml(t.content, ctx, where);
      if (t.type === 'image') {
        const src = attr(t, 'src');
        const img = resolveImage(src, ctx, where);
        if (img) {
          t.attrSet('src', img.url);
          const size = imageSize(resolve(REPO_DIR, img.repoPath));
          if (size) {
            t.attrSet('width', String(size.width));
            t.attrSet('height', String(size.height));
          }
        }
      }
      if (depth === 0 && t.type === 'code_inline') {
        const m = /^((?:docs\/[A-Z][A-Z0-9_-]*\.md)|(?:examples\/[a-z0-9_-]+\/README\.md))(#[a-z0-9_-]+)?$/.exec(t.content);
        const page = m && ctx.pages.get(m[1]);
        if (m && page && page.src !== ctx.src) {
          const url = resolveHref(relFrom(ctx.src, m[1]) + (m[2] ?? ''), ctx, where);
          if (url !== null) {
            out.push(...linkTokens(TokenCtor, url, [t]));
            continue;
          }
        }
      }
      const mentions = depth === 0 && t.type === 'text' ? [...t.content.matchAll(MENTION)] : [];
      if (mentions.length) {
        let last = 0;
        for (const m of mentions) {
          const page = ctx.pages.get(m[1]);
          if (!page || page.src === ctx.src) continue;
          const url = resolveHref(relFrom(ctx.src, m[1]) + (m[2] ?? ''), ctx, where);
          if (url === null) continue;
          const before = new TokenCtor('text', '', 0);
          before.content = t.content.slice(last, m.index);
          const label = new TokenCtor('text', '', 0);
          label.content = m[0];
          out.push(before, ...linkTokens(TokenCtor, url, [label]));
          last = m.index + m[0].length;
        }
        const rest = new TokenCtor('text', '', 0);
        rest.content = t.content.slice(last);
        out.push(rest);
        continue;
      }
      out.push(t);
    }
    block.children = out;
  }
}

/** A repository path written relative to the file `src` (mentions are repository-relative). */
function relFrom(src: string, repoPath: string): string {
  return posix.relative(posix.dirname(src), repoPath) || posix.basename(repoPath);
}
