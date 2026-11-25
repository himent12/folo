// The leaderboard's logic, apart from its markup: ranks, which entries compare, what the Compare action needs, and the
// shared scale every row's 95% interval is drawn on, so overlapping intervals are seen to overlap.

import type { BenchEntry } from '$lib/api/types';
import { rankEntries } from '$lib/bench/rules';

/** A finished entry has a score to rank and a result to open. */
export const finished = (e: BenchEntry) => e.status === 'done' && !!e.score;

/**
 * Each finished entry's rank in the leaderboard's order ($lib/bench/rules rankEntries: complete entries first, then
 * partial ones, each by score): 1 for the first; entries of the same kind with the same score as printed share a rank,
 * the next one counting them (1, 2, 2, 4). The others have none.
 */
export function ranks(entries: readonly BenchEntry[]): Map<string, number> {
  const done = rankEntries(entries).filter(finished);
  const same = (a: BenchEntry, b: BenchEntry) => a.score!.partial === b.score!.partial && Math.round(a.score!.total * 10) === Math.round(b.score!.total * 10);
  const out = new Map<string, number>();
  done.forEach((e, i) => {
    const prev = done[i - 1];
    out.set(e.id, prev && same(prev, e) ? out.get(prev.id)! : i + 1);
  });
  return out;
}

/** The entry whose interval is the board's band: the leading complete entry, else (none complete) the leading one. */
export const bandEntry = (entries: readonly BenchEntry[]) => {
  const done = rankEntries(entries).filter(finished);
  return done.find((e) => !e.score!.partial) ?? done[0];
};

/** The fewest and most entries Compare takes. */
export const COMPARE_MIN = 2;
export const COMPARE_MAX = 4;

/** Tooltip on disabled row checkboxes once COMPARE_MAX entries are already selected. */
export const COMPARE_CAP_TIP = `Compare up to ${COMPARE_MAX} entries`;

/** The line beside the Compare action, for `n` entries selected. */
export function compareHint(n: number): string {
  if (n === 0) return `Select ${COMPARE_MIN}–${COMPARE_MAX} finished entries to compare`;
  if (n < COMPARE_MIN) return `Select one more entry to compare · ${COMPARE_MIN}–${COMPARE_MAX} entries`;
  if (n < COMPARE_MAX) return `${n} selected · up to ${COMPARE_MAX - n} more`;
  return `${n} selected (the maximum)`;
}

/** Why Compare is disabled, or null once it can run: the same words as the visible line. */
export const compareReason = (n: number): string | null => (n < COMPARE_MIN ? compareHint(n) : null);

/** The selection after toggling `id`: finished entries only, at most four; the rest stay as they were. */
export function toggleSelection(selected: readonly string[], id: string, entries: readonly BenchEntry[]): string[] {
  if (selected.includes(id)) return selected.filter((x) => x !== id);
  const e = entries.find((x) => x.id === id);
  if (!e || !finished(e) || selected.length >= COMPARE_MAX) return [...selected];
  return [...selected, id];
}

/** The selection that still holds once the leaderboard changed (an entry deleted, or no longer finished). */
export const keepSelection = (selected: readonly string[], entries: readonly BenchEntry[]) => selected.filter((id) => entries.some((e) => e.id === id && finished(e)));

/** The compare page's path for a selection, in leaderboard order. */
export function comparePath(benchId: string, selected: readonly string[], entries: readonly BenchEntry[]): string {
  const ids = entries.filter((e) => selected.includes(e.id)).map((e) => e.id);
  return `/benchmarks/${encodeURIComponent(benchId)}/compare?entries=${ids.map(encodeURIComponent).join(',')}`;
}

/** The entry ids a compare path names (`?entries=a,b`), in that order, each once (a link may name one twice). */
export const compareIds = (q: URLSearchParams) => [
  ...new Set(
    (q.get('entries') ?? '')
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean),
  ),
];

/**
 * The one scale of a leaderboard's intervals: from just under the lowest interval to just over the highest, on a
 * 0.05 grid (at least 0.1 wide, inside 0–1), so every row's dot and whiskers sit on the same axis and a whisker never
 * touches the plot's edge. Null with no interval to draw.
 */
export function intervalDomain(intervals: readonly (readonly [number, number])[]): [number, number] | null {
  if (!intervals.length) return null;
  const lo = Math.min(...intervals.map((c) => c[0]));
  const hi = Math.max(...intervals.map((c) => c[1]));
  let a = Math.max(0, Math.floor((lo - 0.01) * 20) / 20);
  let b = Math.min(1, Math.ceil((hi + 0.01) * 20) / 20);
  if (b - a < 0.1) {
    const mid = (a + b) / 2;
    a = Math.max(0, Math.round((mid - 0.05) * 20) / 20);
    b = Math.min(1, a + 0.1);
    // Against the top of the scale, the 0.1 is taken below.
    a = Math.max(0, b - 0.1);
  }
  return [Number(a.toFixed(2)), Number(b.toFixed(2))];
}

/** How far under the leader's interval an entry's interval may end and still be on the scale. */
export const FOCUS = 0.1;

/**
 * The leaderboard's interval scale, focused on the leaders: an entry far behind (its interval ending more than FOCUS
 * under the leader's) would stretch the scale and crowd the leaders' whiskers against its edge, so it is left off the
 * scale and drawn as an off-scale mark at the left edge with its value (`below`). `rows` are the finished entries in
 * leaderboard order, the leader first. Null with no interval.
 */
export function intervalScale(rows: readonly { id: string; ci: readonly [number, number] | null }[]): { domain: [number, number]; below: Set<string> } | null {
  const known = rows.filter((r): r is { id: string; ci: readonly [number, number] } => !!r.ci);
  const lead = known[0];
  if (!lead) return null;
  const floor = lead.ci[0] - FOCUS;
  const below = new Set(known.filter((r) => r.ci[1] < floor).map((r) => r.id));
  const domain = intervalDomain(known.filter((r) => !below.has(r.id)).map((r) => r.ci));
  return domain ? { domain, below } : null;
}

/**
 * The least distance between two tick labels' centres, in px: a two-decimal label at the chart's 11 px tick size is
 * about 24 px wide, and 12 px between two keeps them apart as two numbers ("0.45 0.50" touching read as one).
 */
export const TICK_GAP = 36;

/** The steps a scale's ticks may take, the finest first: every one a round number. */
const STEPS = [0.05, 0.1, 0.2, 0.25, 0.5, 1];

/**
 * Ticks of a leaderboard scale: the multiples of one round step inside it (0.05, else 0.1, 0.2, ...), the finest
 * step whose labels sit at least `gap` px apart on a plot `width` px wide (at most five ticks when the width is not
 * known yet). Evenly spaced, never an end squeezed in beside its neighbour: an end that is not on the step is left
 * unlabelled. At least two ticks while two fit; a plot too narrow for two labels gets its lower end only.
 */
export function scaleTicks(domain: readonly [number, number], width = 0, gap = TICK_GAP): number[] {
  const [a, b] = domain;
  if (!(b > a)) return [a];
  const known = width > 0;
  const on = (step: number) => {
    const out: number[] = [];
    for (let k = Math.ceil(a / step - 1e-9); k * step <= b + 1e-9; k++) out.push(+(k * step).toFixed(2));
    return out;
  };
  for (const step of STEPS) {
    if (known && (step / (b - a)) * width < gap) continue;
    const ticks = on(step);
    if (!known && ticks.length > 5) continue;
    if (ticks.length >= 2) return ticks;
  }
  return known && width >= gap ? [a, b] : [a];
}

/** Where a value falls on a domain, 0..1 (clamped). */
export function along(domain: readonly [number, number], v: number): number {
  const [a, b] = domain;
  return b > a ? Math.max(0, Math.min(1, (v - a) / (b - a))) : 0.5;
}

/** Whether two intervals overlap: the entries are not clearly different on this benchmark. */
export const overlaps = (x: readonly [number, number], y: readonly [number, number]) => {
  // As printed (three decimals): what the reader checks against the numbers shown.
  const r = (v: number) => Math.round(v * 1000);
  return r(x[0]) <= r(y[1]) && r(y[0]) <= r(x[1]);
};

/**
 * Whether an entry was graded on an earlier version of its benchmark: its fingerprint is not the benchmark's (the
 * engine's version is part of it, so an engine upgrade changes it while old entries keep theirs). Such an entry does
 * not compare with the current ones (docs/BENCH.md: two entries compare when their fingerprints agree).
 */
export const stale = (e: Pick<BenchEntry, 'fingerprint'>, fingerprint: string) => !!e.fingerprint && e.fingerprint !== fingerprint;

/**
 * The leaderboard as Studio shows it: the server's order (rankEntries) kept within each group, the finished entries
 * graded on the benchmark as it is first, then the finished ones graded on an earlier version of it, then the rest.
 * Only the first group is ranked, banded and highlighted.
 */
export function boardOrder(entries: readonly BenchEntry[], fingerprint: string): BenchEntry[] {
  const order = rankEntries(entries);
  const current = order.filter((e) => finished(e) && !stale(e, fingerprint));
  // The server ranks these by score alone (complete or not), ties to the older one.
  const earlier = order
    .filter((e) => finished(e) && stale(e, fingerprint))
    .sort((a, b) => (b.score?.total ?? 0) - (a.score?.total ?? 0) || a.created.localeCompare(b.created) || a.id.localeCompare(b.id));
  return [...current, ...earlier, ...order.filter((e) => !finished(e))];
}
