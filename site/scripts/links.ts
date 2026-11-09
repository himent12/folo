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
  return statSync(dir).isDirectory() ? 'dir' : 'file';
}

function lineOf(tok: Token | undefined): string {
  return tok?.map ? `:${tok.map[0] + 1}` : '';
}

/** Repository path of a relative reference from `src`, or null if it leaves the repository. */
function resolveRepoPath(src: string, path: string): string | null {
  const joined = posix.normalize(posix.join(posix.dirname(src), decodeURIComponent(path)));
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
  const hashAt = href.indexOf('#');
  const path = hashAt < 0 ? href : href.slice(0, hashAt);
  const hash = hashAt < 0 ? '' : href.slice(hashAt + 1);
  if (path === '') {
    const self = ctx.pages.get(ctx.src);
    if (hash && self && !self.anchors.has(decodeURIComponent(hash))) return fail(`no heading #${hash} on this page`);
    return href;
  }
  if (path.startsWith('/')) return fail('absolute paths do not work on GitHub or on the site; use a relative path');
  const repoPath = resolveRepoPath(ctx.src, path);
  if (repoPath === null) return fail('it points outside the repository');
  const page = ctx.pages.get(repoPath);
  if (page) {
    if (hash && !page.anchors.has(decodeURIComponent(hash))) return fail(`${repoPath} has no heading #${hash}`);
    const url = page.src === ctx.src ? '' : relUrl(ctx.slug, page.slug);
    return `${url || (hash ? '' : './')}${hash ? `#${hash}` : ''}`;
  }
  const kind = repoPathKind(repoPath);
  if (!kind) return fail(`${repoPath} does not exist`);
  return `${kind === 'dir' ? TREE_URL : BLOB_URL}/${repoPath}${hash ? `#${hash}` : ''}`;
}

/** Site URL and repository path of a local image; null for remote images and (with an error) for missing ones. */
function resolveImage(src: string, ctx: LinkContext, where: string): { url: string; repoPath: string } | null {
  if (/^[a-z][a-z0-9+.-]*:/i.test(src)) return null;
  const repoPath = resolveRepoPath(ctx.src, src);
  if (repoPath === null || repoPathKind(repoPath) !== 'file') {
    ctx.errors.push(`${ctx.src}${where}: missing image "${src}"`);
    return null;
  }
  const out = `media/${repoPath}`;
  ctx.media.set(repoPath, out);
  return { url: relUrl(ctx.slug, out), repoPath };
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
