// Confusion-matrix helpers for the Model view.

/**
 * Row/column of model class `cls` in a confusion matrix whose entries are the model classes `ids` (a reduced matrix
 * of the most frequent classes plus "Other"), or every class in order when `ids` is undefined. Null when the class
 * is folded into "Other", where no cell is its own.
 */
export function cmIndex(cls: number, ids: readonly number[] | undefined): number | null {
  if (!ids) return cls >= 0 ? cls : null;
  const i = ids.indexOf(cls);
  return cls >= 0 && i >= 0 ? i : null;
}

/** One line of the phone confusion list: a true class taken for another (`c` its column), or missed (`c` = background). */
export interface ConfusionEntry {
  kind: 'confused' | 'missed';
  /** True class (matrix row). */
  r: number;
  /** Predicted class (matrix column); the background column for a miss. */
  c: number;
  count: number;
  /** The true class's objects (its row's sum). */
  total: number;
  /** count / total. */
  v: number;
}

/**
 * The strongest confusions and misses of an (n+1)×(n+1) matrix (last row/column background), largest share of the
 * true class first (ties: more objects first). Rows with fewer than `minRow` objects are skipped, as in the summary,
 * so one stray box can't top the list; the background row (false detections) is not a true class and is left out.
 */
export function rankConfusions(conf: readonly (readonly number[])[], limit = 8, minRow = 20): ConfusionEntry[] {
  const k = conf.length - 1;
  const out: ConfusionEntry[] = [];
  for (let r = 0; r < k; r++) {
    const row = conf[r] ?? [];
    const total = row.reduce((a, b) => a + b, 0);
    if (total < minRow) continue;
    for (let c = 0; c <= k; c++) {
      const count = row[c] ?? 0;
      if (c === r || count <= 0) continue;
      out.push({ kind: c === k ? 'missed' : 'confused', r, c, count, total, v: count / total });
    }
  }
  return out.sort((a, b) => b.v - a.v || b.count - a.count).slice(0, limit);
}
