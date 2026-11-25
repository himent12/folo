// Query matching, cached keystroke scoring, highlight slicing, and O(N) grouping for the command palette.
import type { Item, PaletteGroup } from './types';

export interface WordToken {
  at: number;
  w: string;
}

export interface PreparedText {
  lower: string;
  words: WordToken[];
}

export interface PreparedQuery {
  /** Trimmed, lowercased, single-space collapsed query string (used for word/text matching). */
  readonly norm: string;
  /** Trimmed, lowercased query string without collapsing internal spaces (used for exact/prefix/suffix id matching). */
  readonly idNorm: string;
  readonly words: readonly string[];
  readonly hasSpace: boolean;
}

/** What the palette ranks: a label, and the other phrasings it answers to ("toggle theme", a dataset's own name). */
export interface Rankable {
  group: string;
  label: string;
  /** More texts to match, e.g. the entity an action is about; the best of these and the label wins. */
  match?: string[];
  /** Only offered for a query: an action about one entity ("New run on shapes") would flood the empty palette. */
  scoped?: boolean;
  /** The entity's id (`run_8f5b`), found by the whole id, or by its start or end from four characters (shell F3). */
  entityId?: string;
}

interface PreparedItem {
  label: string;
  group: string;
  entityId: string | undefined;
  match: readonly string[] | undefined;
  labelText: PreparedText;
  groupLabelText: PreparedText;
  matchTexts: PreparedText[];
  entityIdLower: string | undefined;
}

const WORD_RE = /[^\s\-_/.()]+/g;
const itemCache = new WeakMap<Rankable, PreparedItem>();
let lastQueryRaw: string | undefined;
let lastQueryPrepared: PreparedQuery | undefined;

/** Normalizes a query once per keystroke; single-entry cache avoids repeating regex work across highlight() calls. */
export function prepareQuery(query: string): PreparedQuery {
  if (query === lastQueryRaw && lastQueryPrepared) return lastQueryPrepared;
  const idNorm = query.trim().toLowerCase();
  const norm = idNorm ? idNorm.replace(/\s+/g, ' ') : '';
  const hasSpace = norm.includes(' ');
  const words = norm ? (hasSpace ? norm.split(' ') : [norm]) : [];
  const prepared: PreparedQuery = { norm, idNorm, words, hasSpace };
  lastQueryRaw = query;
  lastQueryPrepared = prepared;
  return prepared;
}

/** Lowercases and tokenizes `text` into word spans between whitespace and name separators (`-`, `_`, `/`, `.`, `(` `)`). */
export function prepareText(text: string): PreparedText {
  const lower = text.toLowerCase();
  const words: WordToken[] = [];
  WORD_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = WORD_RE.exec(lower)) !== null) {
    words.push({ at: m.index, w: m[0] });
  }
  return { lower, words };
}

function sameMatch(a: readonly string[] | undefined, b: readonly string[] | undefined): boolean {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function prepareItem(item: Rankable): PreparedItem {
  const cached = itemCache.get(item);
  if (
    cached &&
    cached.label === item.label &&
    cached.group === item.group &&
    cached.entityId === item.entityId &&
    sameMatch(cached.match, item.match)
  ) {
    return cached;
  }
  const prepared: PreparedItem = {
    label: item.label,
    group: item.group,
    entityId: item.entityId,
    match: item.match ? [...item.match] : undefined,
    labelText: prepareText(item.label),
    groupLabelText: prepareText(`${item.group} ${item.label}`),
    matchTexts: item.match ? item.match.map(prepareText) : [],
    entityIdLower: item.entityId?.toLowerCase(),
  };
  itemCache.set(item, prepared);
  return prepared;
}

/**
 * Hot-path scorer (zero allocations): returns the match score for `pq` against `pt`, or -1 for no match.
 * Precedence: exact text, prefix, word start, ordered word prefixes ("shapes s" in "shapes-small"), substring,
 * word initials ("nr" in "New run"). Characters scattered anywhere in order do not match (shell F14).
 */
function scorePrepared(pq: PreparedQuery, pt: PreparedText): number {
  const q = pq.norm;
  if (!q) return 0;
  const t = pt.lower;
  const extra = (t.length - q.length) * 0.1;
  if (t === q) return 2000;
  const tw = pt.words;
  let atWord: number | undefined;
  for (let i = 0; i < tw.length; i++) {
    const at = tw[i]!.at;
    if (t.startsWith(q, at)) {
      atWord = at;
      break;
    }
  }
  if (atWord === 0) return 1600 - extra;
  if (atWord !== undefined) return 1400 - atWord * 2 - extra;
  if (pq.words.length > 1) {
    let j = 0;
    let matched = 0;
    for (let wi = 0; wi < pq.words.length; wi++) {
      const w = pq.words[wi]!;
      while (j < tw.length && !tw[j]!.w.startsWith(w)) j++;
      if (j === tw.length) break;
      matched++;
      j++;
    }
    if (matched === pq.words.length) return 1100 - extra;
  }
  const mid = t.indexOf(q);
  if (mid >= 0) return 800 - mid * 2 - extra;
  if (!pq.hasSpace && q.length >= 2 && q.length <= tw.length) {
    let j = 0;
    for (const ch of q) {
      while (j < tw.length && tw[j]!.w[0] !== ch) j++;
      if (j === tw.length) return -1;
      j++;
    }
    return 500 - extra;
  }
  return -1;
}

/** Computes highlight spans for `pq` against `pt` using the same precedence as `scorePrepared`. */
function matchSpans(pq: PreparedQuery, pt: PreparedText): [number, number][] {
  const q = pq.norm;
  if (!q) return [];
  const t = pt.lower;
  if (t === q) return [[0, q.length]];
  const tw = pt.words;
  let atWord: number | undefined;
  for (let i = 0; i < tw.length; i++) {
    const at = tw[i]!.at;
    if (t.startsWith(q, at)) {
      atWord = at;
      break;
    }
  }
  if (atWord === 0) return [[0, q.length]];
  if (atWord !== undefined) return [[atWord, atWord + q.length]];
  if (pq.words.length > 1) {
    const spans: [number, number][] = [];
    let j = 0;
    for (let wi = 0; wi < pq.words.length; wi++) {
      const w = pq.words[wi]!;
      while (j < tw.length && !tw[j]!.w.startsWith(w)) j++;
      if (j === tw.length) break;
      spans.push([tw[j]!.at, tw[j]!.at + w.length]);
      j++;
    }
    if (spans.length === pq.words.length) return spans;
  }
  const mid = t.indexOf(q);
  if (mid >= 0) return [[mid, mid + q.length]];
  if (!pq.hasSpace && q.length >= 2 && q.length <= tw.length) {
    const spans: [number, number][] = [];
    let j = 0;
    for (const ch of q) {
      while (j < tw.length && tw[j]!.w[0] !== ch) j++;
      if (j === tw.length) return [];
      spans.push([tw[j]!.at, tw[j]!.at + 1]);
      j++;
    }
    return spans;
  }
  return [];
}

/** Scores how well `query` matches `text` (higher is better; see `scorePrepared`), or -1 for no match. */
export function fuzzy(query: string, text: string): number {
  return scorePrepared(prepareQuery(query), prepareText(text));
}

function sliceHighlight(text: string, spans: [number, number][]): { s: string; m: boolean }[] {
  if (!spans.length) return [{ s: text, m: false }];
  const out: { s: string; m: boolean }[] = [];
  let pos = 0;
  for (const [a, b] of spans) {
    const start = Math.max(pos, Math.min(a, text.length));
    const end = Math.min(b, text.length);
    if (end <= start) continue;
    if (start > pos) out.push({ s: text.slice(pos, start), m: false });
    const seg = text.slice(start, end);
    const prev = out[out.length - 1];
    if (prev && prev.m) prev.s += seg;
    else out.push({ s: seg, m: true });
    pos = end;
  }
  if (pos < text.length) out.push({ s: text.slice(pos), m: false });
  return out;
}

/** Splits `text` into [plain, matched] segments for highlighting what `query` matched. */
export function highlight(query: string, text: string): { s: string; m: boolean }[] {
  if (!text) return [];
  const pq = prepareQuery(query);
  if (!pq.norm) return [{ s: text, m: false }];
  return sliceHighlight(text, matchSpans(pq, prepareText(text)));
}

/** Highlights a rankable item's label using its cached PreparedText (zero per-keystroke tokenization). */
export function highlightItem(query: string, item: Rankable): { s: string; m: boolean }[] {
  if (!item.label) return [];
  const pq = prepareQuery(query);
  if (!pq.norm) return [{ s: item.label, m: false }];
  return sliceHighlight(item.label, matchSpans(pq, prepareItem(item).labelText));
}

function idScorePrepared(idNorm: string, idLower: string): number {
  // Above an exact name (2000): an id names one thing, a name may be another thing's id.
  if (idNorm === idLower) return 2100;
  return idNorm.length >= 4 && (idLower.startsWith(idNorm) || idLower.endsWith(idNorm)) ? 1500 : -1;
}

/** How well `query` names the id `id` (see Rankable.entityId), or -1: never by a word of it, so "run" is not every run. */
export function idScore(query: string, id: string): number {
  return idScorePrepared(prepareQuery(query).idNorm, id.toLowerCase());
}

/**
 * Orders palette items for `query`: with none, everything not `scoped`; with one, every item that matches, best first,
 * actions about an entity ranked with the entities themselves (an action a hair below its entity, so "shapes" lists
 * the dataset, then "New run on shapes").
 */
export function rank<T extends Rankable>(items: T[], query: string, limit = 40): T[] {
  const pq = prepareQuery(query);
  if (!pq.norm) return items.filter((i) => !i.scoped);
  const scored: { i: T; s: number }[] = [];
  for (let idx = 0; idx < items.length; idx++) {
    const item = items[idx]!;
    const p = prepareItem(item);
    let best = scorePrepared(pq, p.labelText);
    const grpScore = scorePrepared(pq, p.groupLabelText);
    if (grpScore >= 0 && grpScore - 20 > best) best = grpScore - 20;
    const penalty = item.scoped ? 10 : 0;
    for (let mi = 0; mi < p.matchTexts.length; mi++) {
      const mScore = scorePrepared(pq, p.matchTexts[mi]!);
      if (mScore >= 0 && mScore - penalty > best) best = mScore - penalty;
    }
    if (p.entityIdLower !== undefined) {
      const ids = idScorePrepared(pq.idNorm, p.entityIdLower);
      if (ids > best) best = ids;
    }
    if (best >= 0) scored.push({ i: item, s: best });
  }
  scored.sort((a, b) => b.s - a.s);
  return scored.slice(0, limit).map((x) => x.i);
}

/**
 * Groups ranked results in O(N) while preserving visual order: groups appear in order of their best non-descriptive
 * item, so entity-scoped actions ("New run on shapes-five") never pull Actions above the entity's own group.
 */
export function groupResults(all: readonly Item[]): PaletteGroup[] {
  const byGroup = new Map<string, PaletteGroup>();
  const out: PaletteGroup[] = [];
  const ensure = (name: string): PaletteGroup => {
    let g = byGroup.get(name);
    if (!g) {
      g = { group: name, items: [] };
      byGroup.set(name, g);
      out.push(g);
    }
    return g;
  };
  for (let i = 0; i < all.length; i++) if (!all[i]!.describes) ensure(all[i]!.group);
  for (let i = 0; i < all.length; i++) if (all[i]!.describes) ensure(all[i]!.group);
  for (let i = 0; i < all.length; i++) {
    const item = all[i]!;
    byGroup.get(item.group)!.items.push({ item, n: i });
  }
  let n = 0;
  for (const g of out) for (const entry of g.items) entry.n = n++;
  return out;
}
