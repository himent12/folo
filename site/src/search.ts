// Client-side search over the prebuilt index (search.json, written by scripts/generate.ts), loaded on first use.
// Every query term must appear in a section's title, its page's title or its text; title hits rank first.

interface Index {
  pages: { t: string; l: string; u: string; g: string }[];
  sections: [page: number, title: string, id: string, text: string][];
}

interface Entry {
  page: Index['pages'][number];
  title: string;
  id: string;
  text: string;
  lTitle: string;
  lPage: string;
  lText: string;
}

export interface Hit {
  href: string;
  path: string;
  title: string;
  snippet: string;
}

let entries: Promise<Entry[]> | undefined;

export function loadIndex(root: string): Promise<Entry[]> {
  entries ??= fetch(`${root}search.json`)
    .then((r) => {
      if (!r.ok) throw new Error(`search.json: HTTP ${r.status}`);
      return r.json() as Promise<Index>;
    })
    .then((ix) =>
      ix.sections.map(([p, title, id, text]) => {
        const page = ix.pages[p];
        return { page, title, id, text, lTitle: title.toLowerCase(), lPage: `${page.t} ${page.l}`.toLowerCase(), lText: text.toLowerCase() };
      }),
    )
    .catch((e: unknown) => {
      entries = undefined;
      throw e;
    });
  return entries;
}

const escapeHtml = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

function isWordStart(hay: string, at: number): boolean {
  return at === 0 || !/[\p{L}\p{N}]/u.test(hay[at - 1]);
}

function count(hay: string, needle: string, max: number): number {
  let n = 0;
  for (let i = hay.indexOf(needle); i >= 0 && n < max; i = hay.indexOf(needle, i + needle.length)) n++;
  return n;
}

function highlight(text: string, terms: string[]): string {
  const re = new RegExp(terms.map(escapeRe).sort((a, b) => b.length - a.length).join('|'), 'gi');
  let out = '';
  let last = 0;
  for (const m of text.matchAll(re)) {
    out += `${escapeHtml(text.slice(last, m.index))}<mark>${escapeHtml(m[0])}</mark>`;
    last = m.index + m[0].length;
  }
  return out + escapeHtml(text.slice(last));
}

function snippet(e: Entry, terms: string[]): string {
  // Where the first term occurs in the text; 0 when the terms only matched the titles.
  const hits = terms.map((t) => e.lText.indexOf(t)).filter((i) => i >= 0);
  const at = hits.length ? Math.min(...hits) : 0;
  const start = Math.max(0, at - 50);
  const head = start > 0 ? e.text.slice(start).replace(/^\S*\s/, '') : e.text;
  const body = head.slice(0, 200);
  return `${start > 0 ? '… ' : ''}${highlight(body, terms)}${head.length > 200 ? ' …' : ''}`;
}

export function search(all: Entry[], query: string, root: string, limit = 30): Hit[] {
  const terms = [...new Set(query.toLowerCase().split(/\s+/).filter(Boolean))];
  if (!terms.length) return [];
  const scored: [number, Entry][] = [];
  for (const e of all) {
    let score = 0;
    let ok = true;
    for (const t of terms) {
      const inTitle = e.lTitle.indexOf(t);
      const inPage = e.lPage.indexOf(t);
      const inText = count(e.lText, t, 8);
      if (inTitle < 0 && inPage < 0 && !inText) {
        ok = false;
        break;
      }
      if (inTitle >= 0) score += isWordStart(e.lTitle, inTitle) ? 12 : 6;
      if (inPage >= 0) score += 3;
      score += inText;
    }
    if (!ok) continue;
    if (e.id === '') score += 2;
    scored.push([score, e]);
  }
  scored.sort((a, b) => b[0] - a[0]);
  return scored.slice(0, limit).map(([, e]) => ({
    href: `${root}${e.page.u}${e.id ? `#${e.id}` : ''}`,
    path: e.id ? `${e.page.g} › ${e.page.l}` : e.page.g,
    title: e.id ? e.title : e.page.l,
    snippet: snippet(e, terms),
  }));
}

export { escapeHtml, highlight };
