// The leaderboard's logic, apart from its markup: ranks, which entries compare, what the Compare action needs, and the
// shared scale every row's 95% interval is drawn on, so overlapping intervals are seen to overlap.

import type { BenchEntry } from '$lib/api/types';

/** A finished entry has a score to rank and a result to open. */
export const finished = (e: BenchEntry) => e.status === 'done' && !!e.score;

/**
 * Each finished entry's rank (1 for the best; entries with the same score share it, the next one counting them: 1, 2,
 * 2, 4). The others have none.
 */
export function ranks(entries: readonly BenchEntry[]): Map<string, number> {
  const done = entries.filter(finished).sort((a, b) => b.score!.total - a.score!.total);
  const out = new Map<string, number>();
  done.forEach((e, i) => {
    const prev = done[i - 1];
    out.set(e.id, prev && prev.score!.total === e.score!.total ? out.get(prev.id)! : i + 1);
  });
  return out;
}

/** The fewest and most entries Compare takes. */
export const COMPARE_MIN = 2;
export const COMPARE_MAX = 4;

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

/** The entry ids a compare path names (`?entries=a,b`), in that order. */
export const compareIds = (q: URLSearchParams) =>
  (q.get('entries') ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);

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

/** Ticks of a leaderboard scale: its ends and the 0.05 steps between them, at most five (every other one past that). */
export function scaleTicks(domain: readonly [number, number]): number[] {
  const n = Math.round((domain[1] - domain[0]) / 0.05);
  const step = n > 4 ? Math.ceil(n / 4) : 1;
  const out: number[] = [];
  for (let k = 0; k <= n; k += step) out.push(+(domain[0] + k * 0.05).toFixed(2));
  if (out.at(-1) !== domain[1]) out.push(domain[1]);
  return out;
}

/** Where a value falls on a domain, 0..1 (clamped). */
export function along(domain: readonly [number, number], v: number): number {
  const [a, b] = domain;
  return b > a ? Math.max(0, Math.min(1, (v - a) / (b - a))) : 0.5;
}

/** Whether two intervals overlap: the entries are not clearly different on this benchmark. */
export const overlaps = (x: readonly [number, number], y: readonly [number, number]) => x[0] <= y[1] && y[0] <= x[1];
