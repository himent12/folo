// Matching and ranking for the command palette.

/** The words of a text: runs between spaces and the separators of names (`shapes-small`, `a/b`, `x_1`, `(now dark)`). */
function words(t: string): { at: number; w: string }[] {
  return [...t.matchAll(/[^\s\-_/.()]+/g)].map((m) => ({ at: m.index, w: m[0] }));
}

/**
 * Where `query` matches `text`, best way first, as the spans to mark, and how good that way is: the whole text,
 * its start, a word's start, every query word starting a word in order ("shapes s" in "shapes-small"), the middle of a
 * word, the words' initials ("nr" in "New run"). Characters scattered anywhere in order do not match: "cloud" must
 * not find every "Open … in Playground" (shell F14). Null for no match.
 */
function match(query: string, text: string): { score: number; spans: [number, number][] } | null {
  const q = query.trim().toLowerCase().replace(/\s+/g, ' ');
  const t = text.toLowerCase();
  if (!q) return { score: 0, spans: [] };
  const extra = (t.length - q.length) * 0.1;
  if (t === q) return { score: 2000, spans: [[0, q.length]] };
  const tw = words(t);
  const atWord = tw.find((x) => t.startsWith(q, x.at))?.at;
  if (atWord === 0) return { score: 1600 - extra, spans: [[0, q.length]] };
  if (atWord !== undefined) return { score: 1400 - atWord * 2 - extra, spans: [[atWord, atWord + q.length]] };
  const qw = q.split(' ').filter(Boolean);
  if (qw.length > 1) {
    const spans: [number, number][] = [];
    let j = 0;
    for (const w of qw) {
      while (j < tw.length && !tw[j]!.w.startsWith(w)) j++;
      if (j === tw.length) break;
      spans.push([tw[j]!.at, tw[j]!.at + w.length]);
      j++;
    }
    if (spans.length === qw.length) return { score: 1100 - extra, spans };
  }
  const mid = t.indexOf(q);
  if (mid >= 0) return { score: 800 - mid * 2 - extra, spans: [[mid, mid + q.length]] };
  if (!q.includes(' ') && q.length >= 2 && q.length <= tw.length) {
    const spans: [number, number][] = [];
    let j = 0;
    for (const ch of q) {
      while (j < tw.length && tw[j]!.w[0] !== ch) j++;
      if (j === tw.length) return null;
      spans.push([tw[j]!.at, tw[j]!.at + 1]);
      j++;
    }
    return { score: 500 - extra, spans };
  }
  return null;
}

/** Scores how well `query` matches `text` (higher is better; see `match`), or -1 for no match. */
export function fuzzy(query: string, text: string): number {
  return match(query, text)?.score ?? -1;
}

/** Splits `text` into [plain, matched] segments for highlighting what `query` matched. */
export function highlight(query: string, text: string): { s: string; m: boolean }[] {
  const spans = query.trim() ? (match(query, text)?.spans ?? []) : [];
  const marks = new Array<boolean>(text.length).fill(false);
  for (const [a, b] of spans) for (let i = a; i < b; i++) marks[i] = true;
  const out: { s: string; m: boolean }[] = [];
  for (let i = 0; i < text.length; i++) {
    const last = out[out.length - 1];
    if (last && last.m === marks[i]) last.s += text[i];
    else out.push({ s: text[i] as string, m: marks[i] as boolean });
  }
  return out;
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

/** How well `query` names the id `id` (see Rankable.entityId), or -1: never by a word of it, so "run" is not every run. */
export function idScore(query: string, id: string): number {
  const q = query.trim().toLowerCase();
  const t = id.toLowerCase();
  if (q === t) return 1900;
  return q.length >= 4 && (t.startsWith(q) || t.endsWith(q)) ? 1500 : -1;
}

/**
 * Orders palette items for `query`: with none, everything not `scoped`; with one, every item that matches, best first,
 * actions about an entity ranked with the entities themselves (an action a hair below its entity, so "shapes" lists
 * the dataset, then "New run on shapes").
 */
export function rank<T extends Rankable>(items: T[], query: string, limit = 40): T[] {
  if (!query.trim()) return items.filter((i) => !i.scoped);
  return items
    .map((i) => ({
      i,
      s: Math.max(
        fuzzy(query, i.label),
        fuzzy(query, `${i.group} ${i.label}`) - 20,
        ...(i.match ?? []).map((m) => fuzzy(query, m) - (i.scoped ? 10 : 0)),
        i.entityId === undefined ? -1 : idScore(query, i.entityId),
      ),
    }))
    .filter((x) => x.s >= 0)
    .sort((a, b) => b.s - a.s)
    .slice(0, limit)
    .map((x) => x.i);
}
