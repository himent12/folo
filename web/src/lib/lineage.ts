// Where a run (and the models it saved) started: the model whose weights it was fine-tuned from, a weights file, or
// scratch. A run's `config.weights` is a model id, a path to a .folo file, or null (docs/API.md). One reading for
// every screen that says it: Runs, Models, Model detail, Compare.

import type { Model } from '$lib/api/types';

export type Origin =
  | { kind: 'scratch'; label: 'scratch' }
  | { kind: 'model'; id: string; label: string }
  | { kind: 'file'; path: string; label: string };

/**
 * Resolves `weights` through the models list (by name); an id that is not listed and has no path separators stays an
 * id. A file is named by its file name, extension included, so it reads as a file.
 */
export function origin(weights: string | null | undefined, models: Map<string, Model> | readonly Pick<Model, 'id' | 'name'>[] = []): Origin {
  const w = weights?.trim();
  if (!w) return { kind: 'scratch', label: 'scratch' };
  const byId: Map<string, Pick<Model, 'id' | 'name'>> = models instanceof Map ? models : new Map(models.map((m) => [m.id, m]));
  const m = byId.get(w);
  if (m) return { kind: 'model', id: m.id, label: m.name };
  if (/[/\\]/.test(w) || /\.folo$/i.test(w)) {
    const base = w.split(/[/\\]/).filter(Boolean).pop() ?? w;
    return { kind: 'file', path: w, label: base };
  }
  return { kind: 'model', id: w, label: w };
}
