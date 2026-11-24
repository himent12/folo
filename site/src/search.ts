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
  lPageTitle: string;
  lPageLabel: string;
  lText: string;
}

export interface Hit {
  href: string;
  path: string;
  title: string;
  snippet: string;
}

let entries: Promise<Entry[]> | undefined;
let loaded = false;

export function isIndexLoaded(): boolean {
  return loaded;
}

export function loadIndex(root: string, reload = false): Promise<Entry[]> {
  entries ??= fetch(`${root}search.json`, reload ? { cache: 'reload' } : undefined)
    .then((r) => {
      if (!r.ok) throw new Error(`search.json: HTTP ${r.status}`);
      return r.json() as Promise<Index>;
    })
    .then((ix) => {
      loaded = true;
      return ix.sections.map(([p, title, id, text]) => {
        const page = ix.pages[p];
        return {
          page,
          title,
          id,
          text,
          lTitle: title.toLowerCase().replace(/\s+/g, ' ').trim(),
          lPageTitle: page.t.toLowerCase().replace(/\s+/g, ' ').trim(),
          lPageLabel: page.l.toLowerCase().replace(/\s+/g, ' ').trim(),
          lText: text.toLowerCase(),
        };
      });
    })
    .catch((e: unknown) => {
      loaded = false;
      entries = undefined;
      throw e;
    });
  return entries;
}

const escapeHtml = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const WORD_CHAR = /[\p{L}\p{N}]/u;

function isWordStart(hay: string, at: number): boolean {
  return at === 0 || !WORD_CHAR.test(hay[at - 1]);
}

function isWordEnd(hay: string, end: number): boolean {
  return end >= hay.length || !WORD_CHAR.test(hay[end]);
}

/** Best match quality for `needle` in `hay`: 0 absent, 1 substring, 2 word prefix, 3 whole word. */
function matchQuality(hay: string, needle: string): 0 | 1 | 2 | 3 {
  if (!needle) return 0;
  let best: 0 | 1 | 2 | 3 = 0;
  for (let i = hay.indexOf(needle); i >= 0; i = hay.indexOf(needle, i + 1)) {
    const start = isWordStart(hay, i);
    const end = isWordEnd(hay, i + needle.length);
    if (start && end) return 3;
    if (start && best < 2) best = 2;
    else if (best < 1) best = 1;
  }
  return best;
}

function findBestHit(hay: string, needle: string): number {
  if (!needle) return -1;
  let fallback = -1;
  for (let i = hay.indexOf(needle); i >= 0; i = hay.indexOf(needle, i + 1)) {
    if (isWordStart(hay, i)) return i;
    if (fallback < 0) fallback = i;
  }
  return fallback;
}

function count(hay: string, needle: string, max: number): number {
  let n = 0;
  for (let i = hay.indexOf(needle); i >= 0 && n < max; i = hay.indexOf(needle, i + needle.length)) n++;
  return n;
}

function highlight(text: string, terms: string[]): string {
  const clean = terms.filter(Boolean);
  if (!clean.length) return escapeHtml(text);
  const re = new RegExp(clean.map(escapeRe).sort((a, b) => b.length - a.length).join('|'), 'gi');
  let out = '';
  let last = 0;
  for (const m of text.matchAll(re)) {
    out += `${escapeHtml(text.slice(last, m.index))}<mark>${escapeHtml(m[0])}</mark>`;
    last = m.index + m[0].length;
  }
  return out + escapeHtml(text.slice(last));
}

function snippet(e: Entry, terms: string[], phrase: string): string {
  // Prefer an exact-phrase match in body text, then the earliest word-boundary term hit, then any match.
  let at = terms.length > 1 ? findBestHit(e.lText, phrase) : -1;
  if (at < 0) {
    const hits = terms.map((t) => findBestHit(e.lText, t)).filter((i) => i >= 0);
    at = hits.length ? Math.min(...hits) : 0;
  }
  const start = Math.max(0, at - 50);
  const head = start > 0 ? e.text.slice(start).replace(/^\S*\s/, '') : e.text;
  const body = head.slice(0, 200);
  return `${start > 0 ? '… ' : ''}${highlight(body, terms)}${head.length > 200 ? ' …' : ''}`;
}

export function search(all: Entry[], query: string, root: string, limit = 30): Hit[] {
  const phrase = query.trim().toLowerCase().replace(/\s+/g, ' ');
  const terms = [...new Set(phrase.split(' ').filter(Boolean))];
  if (!terms.length) return [];
  const scored: [number, Entry][] = [];
  for (const e of all) {
    let score = 0;
    let ok = true;
    for (const t of terms) {
      const titleQ = matchQuality(e.lTitle, t);
      const pageQ = Math.max(matchQuality(e.lPageTitle, t), matchQuality(e.lPageLabel, t)) as 0 | 1 | 2 | 3;
      const textCount = count(e.lText, t, 8);
      if (!titleQ && !pageQ && !textCount) {
        ok = false;
        break;
      }
      if (titleQ) score += titleQ === 3 ? 18 : titleQ === 2 ? 12 : 6;
      if (pageQ) score += pageQ === 3 ? 6 : pageQ === 2 ? 4 : 2;
      if (textCount) {
        const textQ = matchQuality(e.lText, t);
        score += textCount + (textQ === 3 ? 3 : textQ === 2 ? 1.5 : 0);
      }
    }
    if (!ok) continue;
    if (terms.length > 1) {
      const phraseTitleQ = matchQuality(e.lTitle, phrase);
      const phrasePageQ = Math.max(matchQuality(e.lPageTitle, phrase), matchQuality(e.lPageLabel, phrase)) as 0 | 1 | 2 | 3;
      const phraseTextQ = matchQuality(e.lText, phrase);
      if (phraseTitleQ) score += phraseTitleQ === 3 ? 28 : phraseTitleQ === 2 ? 20 : 12;
      if (phrasePageQ) score += phrasePageQ === 3 ? 12 : 8;
      if (phraseTextQ) score += phraseTextQ === 3 ? 12 : phraseTextQ === 2 ? 8 : 5;
    }
    if (e.lTitle === phrase || (e.id === '' && e.lPageLabel === phrase)) score += 25;
    if (e.id === '') score += 2;
    scored.push([score, e]);
  }
  scored.sort((a, b) => b[0] - a[0]);
  return scored.slice(0, limit).map(([, e]) => ({
    href: `${root}${e.page.u}${e.id ? `#${e.id}` : ''}`,
    path: e.id ? (e.page.g ? `${e.page.g} › ${e.page.l}` : e.page.l) : e.page.g || e.page.l,
    title: e.id ? e.title : e.page.g ? e.page.l : e.page.t,
    snippet: snippet(e, terms, phrase),
  }));
}

export { escapeHtml, highlight };
