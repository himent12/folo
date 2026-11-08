// Where a model's weights started: a run's `weights` (docs/API.md) is null (from scratch), a model id (fine-tuned
// from that model), or a path to a .folo file (fine-tuned from an imported file).

export type Lineage = { kind: 'scratch' } | { kind: 'model'; id: string } | { kind: 'file'; path: string; name: string };

/** Reads a run config's `weights`. Anything with a path separator or a .folo extension is a file, else a model id. */
export function lineageOf(weights: string | null | undefined): Lineage {
  const w = weights?.trim();
  if (!w) return { kind: 'scratch' };
  if (/[\\/]/.test(w) || /\.folo$/i.test(w)) return { kind: 'file', path: w, name: w.split(/[\\/]/).pop() || w };
  return { kind: 'model', id: w };
}
