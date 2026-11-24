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
const HTML_TAG_NAME = /^<[a-zA-Z][^\s/>]*/;
/** One attribute, its whole value consumed, so text inside another attribute's value is never read as an attribute. */
const HTML_ATTR = /(\s+)([^\s"'>\/=]+)(?:(\s*=\s*)(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/y;

/** Extracts `href` values from HTML `<a>` start tags using the quote-aware attribute scanner (skipping comments). */
export function extractHtmlAnchorHrefs(html: string): string[] {
  const out: string[] = [];
  for (const part of html.matchAll(HTML_PART)) {
    const tag = part[0];
    if (tag.startsWith('<!--')) continue;
    const head = HTML_TAG_NAME.exec(tag)![0];
    if (head.slice(1).toLowerCase() !== 'a') continue;
    let at = head.length;
    for (;;) {
      HTML_ATTR.lastIndex = at;
      const m = HTML_ATTR.exec(tag);
      if (!m) break;
      at = HTML_ATTR.lastIndex;
      const [, , name, eq, dq, sq, bare] = m;
      if (eq && name.toLowerCase() === 'href') out.push(dq ?? sq ?? bare ?? '');
    }
  }
  return out;
}

/**
 * Links and images in raw HTML (`<a href>`, `<img src>`, ...) get the same checks and rewriting as Markdown ones, so
 * HTML can't slip a broken or GitHub-only `.md` link past the build. Only attributes inside start tags count: a link
 * in an HTML comment renders nothing and is left alone.
 */
function rewriteHtml(html: string, ctx: LinkContext, where: string): string {
  return html.replace(HTML_PART, (part) => {
    if (part.startsWith('<!--')) return part;
    // Walk the attributes one after another from the end of the tag name; HTML_PART guarantees they all parse.
    const head = HTML_TAG_NAME.exec(part)![0];
    let out = head;
    let at = head.length;
    for (;;) {
      HTML_ATTR.lastIndex = at;
      const m = HTML_ATTR.exec(part);
      if (!m) break;
      at = HTML_ATTR.lastIndex;
      const [whole, sp, name, eq, dq, sq, bare] = m;
      const lower = name.toLowerCase();
      const value = dq ?? sq ?? bare ?? '';
      // A `srcset` naming one file (a <picture>'s <source>) is checked like `src`; one with several candidates or
      // descriptors is left alone.
      const single = lower === 'srcset' && !/[\s,]/.test(value.trim());
      if (!eq || (lower !== 'href' && lower !== 'src' && !single)) {
        out += whole;
        continue;
      }
      const quote = sq !== undefined ? "'" : '"';
      const url = lower === 'href' ? resolveHref(value, ctx, where) : (resolveImage(value, ctx, where)?.url ?? null);
      out += url === null ? whole : `${sp}${name}${eq}${quote}${url}${quote}`;
    }
    return out + part.slice(at);
  });
}

/**
 * `<picture>` with a dark `<source>` and a light `<img>` is how a Markdown file follows GitHub's theme. The site has
 * its own theme switch, which a media query cannot see, so each such picture becomes two images, one per theme, and
 * the stylesheet shows the one for the theme in use (`img.theme-light`, `img.theme-dark` in prose.css).
 */
const THEMED_PICTURE =
  /<picture>\s*<source\s+media="\(prefers-color-scheme:\s*dark\)"\s+srcset="([^"\s]+)"\s*\/?>\s*<img\s+src="([^"]+)"([^>]*?)\s*\/?>\s*<\/picture>/g;
const themedPictures = (html: string) =>
  html.replace(THEMED_PICTURE, (_m, dark: string, light: string, rest: string) => `<img class="theme-light" src="${light}"${rest}><img class="theme-dark" src="${dark}"${rest}>`);

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
    if (block.type === 'html_block') block.content = rewriteHtml(themedPictures(block.content), ctx, lineOf(block));
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

/** Percent-encodes a relative POSIX path while keeping `.` and `..` segments intact. */
function encodeRelPath(rel: string): string {
  return rel
    .split('/')
    .map((s) => (s === '.' || s === '..' ? s : encodeURIComponent(s)))
    .join('/');
}

/** Unescapes CommonMark ASCII punctuation backslash escapes so file lookups match the parsed destination. */
function unescapeMdDest(s: string): string {
  return s.replace(/\\([!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~])/g, '$1');
}

function resolveMdRef(
  ref: string,
  src: string,
  fromFile: string,
  isImage: boolean,
  pages: Map<string, PageRef>,
  media: Map<string, string>,
): string {
  const clean = unescapeMdDest(ref);
  if (!clean || /^[a-z][a-z0-9+.-]*:/i.test(clean) || clean.startsWith('//')) return ref;
  const { path, query, hash } = splitRef(clean);
  if (path.startsWith('/')) return ref;
  const tail = `${query}${hash ? `#${hash}` : ''}`;
  const fromDir = posix.dirname(fromFile);
  if (path === '') {
    // In a bundled document like llms-full.txt, bare #fragments can collide across sections: qualify them to md/<src>.
    const ownTarget = `md/${src}`;
    if (!isImage && fromFile !== ownTarget && tail && pages.has(src)) {
      const rel = posix.relative(fromDir, ownTarget) || posix.basename(ownTarget);
      return `${encodeRelPath(rel)}${tail}`;
    }
    return ref;
  }
  const repoPath = resolveRepoPath(src, path);
  if (repoPath === null) return ref;
  if (!isImage && pages.has(repoPath)) {
    const target = `md/${repoPath}`;
    if (fromFile === target && tail.startsWith('#')) return tail === clean ? ref : tail;
    const rel = posix.relative(fromDir, target) || posix.basename(target);
    const next = `${encodeRelPath(rel)}${tail}`;
    return next === clean ? ref : next;
  }
  if (isImage || media.has(repoPath)) {
    if (repoPathKind(repoPath) === 'file') {
      media.set(repoPath, `media/${repoPath}`);
      const next = `${encodeRelPath(posix.relative(fromDir, `media/${repoPath}`))}${tail}`;
      return next === clean ? ref : next;
    }
    return ref;
  }
  const kind = repoPathKind(repoPath);
  if (!kind) return ref;
  return `${kind === 'dir' ? TREE_URL : BLOB_URL}/${encodePath(repoPath)}${tail}`;
}

function rewriteSrcset(
  value: string,
  src: string,
  fromFile: string,
  pages: Map<string, PageRef>,
  media: Map<string, string>,
): string {
  return value
    .split(',')
    .map((part) => {
      const m = /^(\s*)(\S+)([\s\S]*)$/.exec(part);
      if (!m) return part;
      const next = resolveMdRef(m[2], src, fromFile, true, pages, media);
      return next === m[2] ? part : `${m[1]}${next}${m[3]}`;
    })
    .join(',');
}

function rewriteMdHtmlTag(
  tag: string,
  src: string,
  fromFile: string,
  pages: Map<string, PageRef>,
  media: Map<string, string>,
): string {
  const head = HTML_TAG_NAME.exec(tag)![0];
  let out = head;
  let at = head.length;
  for (;;) {
    HTML_ATTR.lastIndex = at;
    const m = HTML_ATTR.exec(tag);
    if (!m) break;
    at = HTML_ATTR.lastIndex;
    const [whole, sp, name, eq, dq, sq, bare] = m;
    const lower = name.toLowerCase();
    const value = dq ?? sq ?? bare ?? '';
    if (!eq || (lower !== 'href' && lower !== 'src' && lower !== 'srcset')) {
      out += whole;
      continue;
    }
    const quote = sq !== undefined ? "'" : '"';
    const next =
      lower === 'srcset'
        ? rewriteSrcset(value, src, fromFile, pages, media)
        : resolveMdRef(value, src, fromFile, lower !== 'href', pages, media);
    out += next === value ? whole : `${sp}${name}${eq}${quote}${next}${quote}`;
  }
  return out + tag.slice(at);
}

const MD_HTML_TAG = /<[a-zA-Z][^\s/>]*(?:\s+[^\s"'>\/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'=<>`]+))?)*\s*\/?>/y;
const MD_REF_DEF =
  /( {0,3}\[[^\]\n]+\]:[ \t]*(?:\n[ \t]*)?)(?:<([^>\n]+)>|([^\s\n]+))([ \t]*(?:\n?[ \t]*(?:"(?:\\.|[^"\n])*"|'(?:\\.|[^'\n])*'|\((?:\\.|[^()\n])*\)))?[ \t]*(?=\n|$))/y;

function findClosingBackticks(s: string, from: number, count: number): number {
  let i = from;
  while (i < s.length) {
    if (s[i] === '\n' && s[i + 1] === '\n') return -1;
    if (s[i] === '`') {
      let run = 0;
      while (i + run < s.length && s[i + run] === '`') run++;
      if (run === count) return i;
      i += run;
    } else {
      i++;
    }
  }
  return -1;
}

/** Finds the matching `]` for a `[` at `open`, respecting escapes, code spans, nested brackets and paragraph breaks. */
function findLinkLabelEnd(s: string, open: number): number {
  let depth = 1;
  let i = open + 1;
  while (i < s.length && depth > 0) {
    const ch = s[i];
    if (ch === '\n' && s[i + 1] === '\n') return -1;
    if (ch === '\\' && i + 1 < s.length) {
      i += 2;
      continue;
    }
    if (ch === '`') {
      let run = 0;
      while (i + run < s.length && s[i + run] === '`') run++;
      const close = findClosingBackticks(s, i + run, run);
      i = close >= 0 ? close + run : i + run;
      continue;
    }
    if (ch === '[') depth++;
    else if (ch === ']') depth--;
    if (depth === 0) return i;
    i++;
  }
  return -1;
}

/** Parses `(...)` starting at `open` (`(`); returns destination bounds and closing `)` index, or null. */
function parseInlineDest(s: string, open: number): { destStart: number; destEnd: number; wrapped: boolean; close: number } | null {
  let i = open + 1;
  while (i < s.length && (s[i] === ' ' || s[i] === '\t' || s[i] === '\n')) {
    if (s[i] === '\n' && s[i + 1] === '\n') return null;
    i++;
  }
  if (i >= s.length) return null;
  let destStart: number;
  let destEnd: number;
  let wrapped = false;
  if (s[i] === '<') {
    wrapped = true;
    destStart = i + 1;
    i++;
    while (i < s.length && s[i] !== '>' && s[i] !== '\n') {
      if (s[i] === '\\' && i + 1 < s.length) i += 2;
      else i++;
    }
    if (s[i] !== '>') return null;
    destEnd = i;
    i++;
  } else {
    destStart = i;
    let parens = 0;
    while (i < s.length) {
      const ch = s[i];
      if (ch === '\\' && i + 1 < s.length) {
        i += 2;
        continue;
      }
      if (ch === ' ' || ch === '\t' || ch === '\n') break;
      if (ch === '(') parens++;
      else if (ch === ')') {
        if (parens === 0) break;
        parens--;
      }
      i++;
    }
    destEnd = i;
    if (destEnd === destStart) return null;
  }
  while (i < s.length && (s[i] === ' ' || s[i] === '\t' || s[i] === '\n')) {
    if (s[i] === '\n' && s[i + 1] === '\n') return null;
    i++;
  }
  if (i < s.length && (s[i] === '"' || s[i] === "'" || s[i] === '(')) {
    const endQuote = s[i] === '(' ? ')' : s[i];
    i++;
    while (i < s.length && s[i] !== endQuote) {
      if (s[i] === '\n' && s[i + 1] === '\n') return null;
      if (s[i] === '\\' && i + 1 < s.length) i += 2;
      else i++;
    }
    if (s[i] !== endQuote) return null;
    i++;
    while (i < s.length && (s[i] === ' ' || s[i] === '\t' || s[i] === '\n')) {
      if (s[i] === '\n' && s[i + 1] === '\n') return null;
      i++;
    }
  }
  if (s[i] !== ')') return null;
  return { destStart, destEnd, wrapped, close: i };
}

function rewriteProseChunk(
  s: string,
  src: string,
  fromFile: string,
  pages: Map<string, PageRef>,
  media: Map<string, string>,
  allowRefDef = true,
): string {
  let out = '';
  let i = 0;
  while (i < s.length) {
    const ch = s[i];
    if (ch === '\\' && i + 1 < s.length) {
      out += s.slice(i, i + 2);
      i += 2;
      continue;
    }
    if (ch === '`') {
      let run = 0;
      while (i + run < s.length && s[i + run] === '`') run++;
      const close = findClosingBackticks(s, i + run, run);
      const end = close >= 0 ? close + run : i + run;
      out += s.slice(i, end);
      i = end;
      continue;
    }
    if (allowRefDef && (ch === '[' || ch === ' ') && (i === 0 || s[i - 1] === '\n')) {
      MD_REF_DEF.lastIndex = i;
      const ref = MD_REF_DEF.exec(s);
      if (ref) {
        const prefix = ref[1];
        const raw = ref[2] ?? ref[3];
        const suffix = ref[4];
        const next = resolveMdRef(raw, src, fromFile, false, pages, media);
        const formatted = ref[2] !== undefined || (next !== raw && /[\s()]/.test(next)) ? `<${next}>` : next;
        out += `${prefix}${formatted}${suffix}`;
        i += ref[0].length;
        continue;
      }
    }
    if (ch === '<') {
      if (s.startsWith('<!--', i)) {
        const end = s.indexOf('-->', i + 4);
        const to = end >= 0 ? end + 3 : s.length;
        out += s.slice(i, to);
        i = to;
        continue;
      }
      MD_HTML_TAG.lastIndex = i;
      const tag = MD_HTML_TAG.exec(s);
      if (tag) {
        out += rewriteMdHtmlTag(tag[0], src, fromFile, pages, media);
        i += tag[0].length;
        continue;
      }
    }
    if (ch === '[' || (ch === '!' && s[i + 1] === '[')) {
      const isImage = ch === '!';
      const openBracket = isImage ? i + 1 : i;
      const closeBracket = findLinkLabelEnd(s, openBracket);
      if (closeBracket >= 0 && s[closeBracket + 1] === '(') {
        const dest = parseInlineDest(s, closeBracket + 1);
        if (dest) {
          const inner = rewriteProseChunk(s.slice(openBracket + 1, closeBracket), src, fromFile, pages, media, false);
          const rawDest = s.slice(dest.destStart, dest.destEnd);
          const nextDest = resolveMdRef(rawDest, src, fromFile, isImage, pages, media);
          const needsAngles = !dest.wrapped && nextDest !== rawDest && /[\s()]/.test(nextDest);
          const beforeDest = s.slice(closeBracket + 1, dest.destStart);
          const afterDest = s.slice(dest.destEnd, dest.close + 1);
          out += `${isImage ? '!' : ''}[${inner}]${beforeDest}${needsAngles ? `<${nextDest}>` : nextDest}${afterDest}`;
          i = dest.close + 1;
          continue;
        }
      }
    }
    out += ch;
    i++;
  }
  return out;
}

/**
 * Rewrites relative Markdown/HTML links and images in an LF-normalized Markdown source file for serving at `fromFile`
 * (e.g. `md/docs/CLI.md` or `llms-full.txt`). Uses parsed `tokens` to skip all CommonMark fenced and indented code
 * blocks (including inside lists/blockquotes) and skips inline code spans; expects `media` to be pre-populated by
 * `rewriteLinks()`.
 */
export function rewriteMarkdown(
  markdown: string,
  tokens: Token[],
  src: string,
  fromFile: string,
  pages: Map<string, PageRef>,
  media: Map<string, string>,
): string {
  const codeLines = new Set<number>();
  for (const t of tokens) {
    if ((t.type === 'fence' || t.type === 'code_block') && t.map) {
      for (let ln = t.map[0]; ln < t.map[1]; ln++) codeLines.add(ln);
    }
  }
  const lines = markdown.split('\n');
  let out = '';
  let chunk: string[] = [];

  const flushChunk = () => {
    if (!chunk.length) return;
    out += rewriteProseChunk(chunk.join(''), src, fromFile, pages, media);
    chunk = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const sep = i < lines.length - 1 ? '\n' : '';
    if (codeLines.has(i)) {
      flushChunk();
      out += line + sep;
    } else {
      chunk.push(line + sep);
    }
  }
  flushChunk();
  return out;
}
