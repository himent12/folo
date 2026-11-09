// Fuzzy matching and ranking for the command palette.

/**
 * Scores how well `query` matches `text` (higher is better), or -1 for no match. Characters must appear in order;
 * consecutive runs, word starts and an early first match score higher.
 */
export function fuzzy(query: string, text: string): number {
  const q = query.trim().toLowerCase();
  if (!q) return 0;
  const t = text.toLowerCase();
  const direct = t.indexOf(q);
  if (direct >= 0) return 1000 - direct * 2 - (t.length - q.length) * 0.1 + (direct === 0 || /[\s\-_/]/.test(t[direct - 1] ?? '') ? 50 : 0);
  let score = 0;
  let ti = 0;
  let run = 0;
  for (const ch of q) {
    if (ch === ' ') continue;
    const found = t.indexOf(ch, ti);
    if (found < 0) return -1;
    run = found === ti ? run + 1 : 0;
    score += 10 + run * 5 + (found === 0 || /[\s\-_/]/.test(t[found - 1] ?? '') ? 15 : 0) - Math.min(9, found - ti);
    ti = found + 1;
  }
  return score;
}

/** Splits `text` into [plain, matched] segments for highlighting the characters `query` matched. */
export function highlight(query: string, text: string): { s: string; m: boolean }[] {
  const q = query.trim().toLowerCase();
  if (!q) return [{ s: text, m: false }];
  const t = text.toLowerCase();
  const marks = new Array<boolean>(text.length).fill(false);
  const direct = t.indexOf(q);
  if (direct >= 0) for (let i = 0; i < q.length; i++) marks[direct + i] = true;
  else {
    let ti = 0;
    for (const ch of q) {
      if (ch === ' ') continue;
      const f = t.indexOf(ch, ti);
      if (f < 0) break;
      marks[f] = true;
      ti = f + 1;
    }
  }
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
      s: Math.max(fuzzy(query, i.label), fuzzy(query, `${i.group} ${i.label}`) - 20, ...(i.match ?? []).map((m) => fuzzy(query, m) - (i.scoped ? 10 : 0))),
    }))
    .filter((x) => x.s >= 0)
    .sort((a, b) => b.s - a.s)
    .slice(0, limit)
    .map((x) => x.i);
}
