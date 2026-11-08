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
