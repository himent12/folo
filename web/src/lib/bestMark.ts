// Which values in a comparable group are its best (DESIGN_DIRECTION §2.6: a tertiary "best" word beside an ink value),
// for the Models list and Compare's side by side alike. Exact ties are a joint best, every one of them marked "joint
// best" (round 25 marked one of fasteners-n-best and fasteners-n-f16, whose scores are identical, by a strict `>`). A
// best that differs from another value only past the display's decimals is shown with as many more as it takes, so a
// "best" never sits beside a value that reads the same.

export interface BestMark {
  /** Indexes of the best values: one, or every one of an exact tie. Empty with fewer than two values to compare. */
  best: Set<number>;
  /** Whether the best is shared (an exact tie). */
  joint: boolean;
  /** Decimals that tell the best apart from every other value (at least the display's own, at most `max`). */
  decimals: number;
}

/** The best of `values` (higher is better); missing values never win and never count towards the two needed. */
export function bestMark(values: readonly (number | null | undefined)[], decimals = 3, max = 6): BestMark {
  const finite = values.flatMap((v, i) => (v != null && Number.isFinite(v) ? [{ v, i }] : []));
  if (finite.length < 2) return { best: new Set(), joint: false, decimals };
  const top = Math.max(...finite.map((f) => f.v));
  const best = new Set(finite.filter((f) => f.v === top).map((f) => f.i));
  const rest = finite.filter((f) => f.v !== top).map((f) => f.v);
  let d = decimals;
  while (d < max && rest.some((v) => v.toFixed(d) === top.toFixed(d))) d++;
  return { best, joint: best.size > 1, decimals: d };
}

/** The word beside a best value: "best", or "joint best" for a tie. */
export const bestWord = (m: Pick<BestMark, 'joint'>) => (m.joint ? 'joint best' : 'best');
