// Markdown -> HTML: markdown-it with GitHub-compatible heading ids, build-time highlighting (shiki, coloured from the
// design tokens through CSS variables), copy buttons, scrollable tables and roadmap status markers.
import markdownIt from 'markdown-it';
import type { MarkdownIt, Token } from 'markdown-it';
import GithubSlugger from 'github-slugger';
import { createCssVariablesTheme, createHighlighter, type Highlighter } from 'shiki';

export interface Heading {
  level: number;
  text: string;
  id: string;
}

export interface ParsedPage {
  tokens: Token[];
  /** Every heading, including the H1, with the id GitHub would give it. */
  headings: Heading[];
  title: string;
  /** Plain text of the first paragraph after the title. */
  description: string;
}

const LANGS = [
  'shellscript',
  'powershell',
  'json',
  'jsonc',
  'javascript',
  'typescript',
  'rust',
  'python',
  'toml',
  'yaml',
  'wgsl',
  'http',
  'diff',
  'html',
  'css',
  'markdown',
];
const PLAIN_LANGS = new Set(['text', 'txt', 'plain', 'plaintext']);
const THEME = 'folo';

let highlighter: Promise<Highlighter> | undefined;

function getHighlighter(): Promise<Highlighter> {
  highlighter ??= createHighlighter({
    themes: [createCssVariablesTheme({ name: THEME, variablePrefix: '--shiki-', fontStyle: true })],
    langs: LANGS,
  });
  return highlighter;
}

export function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/** A token attribute as a string ('' when absent). */
export function attr(tok: Token, name: string): string {
  const v = tok.attrGet(name);
  return v === null ? '' : String(v);
}

/** The text a reader sees in an inline token (what GitHub slugs a heading from). */
export function inlineText(tok: Token): string {
  if (!tok.children) return tok.content;
  return tok.children
    .map((c) => {
      if (c.type === 'text' || c.type === 'code_inline') return c.content;
      if (c.type === 'softbreak' || c.type === 'hardbreak') return ' ';
      if (c.type === 'image') return '';
      return '';
    })
    .join('');
}

const COPY_ICON =
  '<svg class="i-copy" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><rect x="5.25" y="5.25" width="8" height="8" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M10.75 3.5V3.25a1.5 1.5 0 0 0-1.5-1.5h-5a1.5 1.5 0 0 0-1.5 1.5v5a1.5 1.5 0 0 0 1.5 1.5h.25" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>' +
  '<svg class="i-check" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="m3.5 8.5 3 3 6-7" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/></svg>';

export function copyButton(): string {
  return `<button class="copy" type="button" aria-label="Copy code">${COPY_ICON}<span class="copy-label">Copy</span></button>`;
}

const TASKS: Record<string, [string, string]> = {
  x: ['done', 'Done'],
  '~': ['doing', 'In progress'],
  ' ': ['todo', 'To do'],
};

export async function createMarkdown(): Promise<MarkdownIt> {
  const hl = await getHighlighter();
  const md = new markdownIt({ html: true, linkify: true });
  // Only URLs with a scheme become links: `main.rs` or `axum.rs` in prose must stay text.
  md.linkify.set({ fuzzyLink: false, fuzzyEmail: false });

  // Heading ids, GitHub style (github-slugger, deduplicated per page), so `FILE.md#anchor` links written for GitHub
  // keep working. The headings are collected into env for the table of contents and the link check.
  md.core.ruler.push('folo_headings', (state) => {
    const slugger = new GithubSlugger();
    const headings: Heading[] = [];
    const tokens = state.tokens;
    for (let i = 0; i < tokens.length; i++) {
      const open = tokens[i];
      if (open.type !== 'heading_open') continue;
      const inline = tokens[i + 1];
      const text = inlineText(inline).trim();
      const id = slugger.slug(text);
      const level = Number(open.tag.slice(1));
      open.attrSet('id', id);
      headings.push({ level, text, id });
      if (level >= 2 && inline.children) {
        const a = new state.Token('html_inline', '', 0);
        a.meta = { generated: true };
        a.content = `<a class="h-anchor" href="#${id}" aria-label="Link to ${escapeHtml(text)}">#</a>`;
        inline.children.push(a);
      }
    }
    state.env.headings = headings;
  });

  // Roadmap-style status markers: "- [x] ...", "- [~] ...", "- [ ] ...".
  md.core.ruler.push('folo_tasks', (state) => {
    const tokens = state.tokens;
    for (let i = 2; i < tokens.length; i++) {
      const inline = tokens[i];
      if (inline.type !== 'inline' || tokens[i - 1].type !== 'paragraph_open' || tokens[i - 2].type !== 'list_item_open') continue;
      const first = inline.children?.[0];
      const m = first?.type === 'text' ? /^\[([ x~])\]\s+/.exec(first.content) : null;
      if (!first || !m) continue;
      const [cls, label] = TASKS[m[1]];
      first.content = first.content.slice(m[0].length);
      const mark = new state.Token('html_inline', '', 0);
      mark.meta = { generated: true };
      mark.content = `<span class="task task-${cls}" role="img" aria-label="${label}" title="${label}"></span>`;
      inline.children!.unshift(mark);
      tokens[i - 2].attrJoin('class', 'task-item');
    }
  });

  md.renderer.rules.fence = (tokens, idx) => {
    const tok = tokens[idx];
    const lang = tok.info.trim().split(/\s+/)[0]?.toLowerCase() ?? '';
    const code = tok.content.replace(/\n$/, '');
    let pre: string;
    if (lang && !PLAIN_LANGS.has(lang) && hl.getLoadedLanguages().includes(lang)) {
      pre = hl.codeToHtml(code, { lang, theme: THEME });
    } else {
      pre = `<pre class="shiki plain"><code>${escapeHtml(code)}</code></pre>`;
    }
    const label = lang && !PLAIN_LANGS.has(lang) ? escapeHtml(lang) : '';
    const langAttr = label ? ` data-lang="${label}"` : '';
    const langBadge = label ? `<span class="code-lang" aria-hidden="true">${label}</span>` : '';
    return `<div class="code"${langAttr}>${pre}<div class="code-tools">${copyButton()}${langBadge}</div></div>\n`;
  };
  md.renderer.rules.code_block = (tokens, idx) =>
    `<div class="code"><pre class="shiki plain"><code>${escapeHtml(tokens[idx].content.replace(/\n$/, ''))}</code></pre><div class="code-tools">${copyButton()}</div></div>\n`;

  md.renderer.rules.table_open = () => '<div class="table-wrap"><table>\n';
  md.renderer.rules.table_close = () => '</table></div>\n';

  const image = md.renderer.rules.image!;
  md.renderer.rules.image = (tokens, idx, options, env, self) => {
    tokens[idx].attrSet('loading', 'lazy');
    tokens[idx].attrSet('decoding', 'async');
    return image(tokens, idx, options, env, self);
  };

  const linkOpen = md.renderer.rules.link_open ?? ((t, i, o, _e, s) => s.renderToken(t, i, o));
  md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
    const href = attr(tokens[idx], 'href');
    if (/^https?:\/\//.test(href)) tokens[idx].attrJoin('class', 'ext');
    return linkOpen(tokens, idx, options, env, self);
  };
  return md;
}

export function parsePage(md: MarkdownIt, text: string): ParsedPage {
  const env: { headings?: Heading[] } = {};
  const tokens = md.parse(text, env);
  const headings = env.headings ?? [];
  const h1 = headings.find((h) => h.level === 1);
  if (!h1) throw new Error('no H1 title');
  let description = '';
  const h1Index = tokens.findIndex((t: Token) => t.type === 'heading_open' && t.tag === 'h1');
  for (let i = h1Index + 3; i < tokens.length; i++) {
    if (tokens[i].type === 'heading_open') break;
    if (tokens[i].type === 'paragraph_open' && tokens[i].level === 0) {
      description = inlineText(tokens[i + 1]).replace(/\s+/g, ' ').trim();
      break;
    }
  }
  return { tokens, headings, title: h1.text, description };
}

/** Removes the H1 (rendered by the page template instead) and returns the remaining tokens. */
export function withoutTitle(tokens: Token[]): Token[] {
  const i = tokens.findIndex((t) => t.type === 'heading_open' && t.tag === 'h1');
  return i < 0 ? tokens : [...tokens.slice(0, i), ...tokens.slice(i + 3)];
}

export interface Section {
  id: string;
  title: string;
  text: string;
}

/** Splits a page into H2/H3 sections of plain text for the search index. */
export function sections(tokens: Token[], title: string): Section[] {
  const out: Section[] = [{ id: '', title, text: '' }];
  let cur = out[0];
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (t.type === 'heading_open') {
      const text = inlineText(tokens[i + 1]).trim();
      if (t.tag === 'h2' || t.tag === 'h3') {
        cur = { id: attr(t, 'id'), title: text, text: '' };
        out.push(cur);
      } else if (t.tag !== 'h1') {
        cur.text += ` ${text}.`;
      }
      i += 2;
    } else if (t.type === 'inline') {
      cur.text += ` ${inlineText(t)}`;
    } else if (t.type === 'fence' || t.type === 'code_block') {
      cur.text += ` ${t.content}`;
    }
  }
  for (const s of out) s.text = s.text.replace(/\s+/g, ' ').trim();
  return out.filter((s) => s.id === '' || s.text || s.title);
}
