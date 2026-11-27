// The models list, kept current for every screen that shows it (Models, the Playground's picker, a model page,
// Overview's best models). The server has no stream of model changes, but a model only comes or goes with a run: its
// checkpoints become models when it finishes or stops, and go with it when it is deleted. So the list is fetched
// again when the shell's run list (shell/runList, polled every 2 s for the whole app) shows a run reach an end or
// disappear, when the tab is shown again (files dropped into the models folder meanwhile), and when a screen says it
// changed something (`modelsChanged`). Each screen keeps its own Resource (its lifetime, its loading state); identical
// reads issued in the same turn are one request (./share).

import { createRoot, effect, signal, untrack } from 'fw';
import { watched } from '$lib/shell/runList';
import { hub } from './index';
import { Resource } from './resource';
import type { Model, Run } from './types';

const ACTIVE = new Set(['queued', 'running', 'stopping']);
const changes = signal(0);

/** Says the models may have changed: every models list on screen fetches again, quietly. */
export function modelsChanged(): void {
  changes.set(changes.peek() + 1);
}

/**
 * What of a run list decides the models: the runs that are not training, with their status and model. Equal keys mean
 * the same models (a run starting, or progressing, changes none). Exported for tests.
 */
export function modelsKey(runs: readonly Run[]): string {
  return runs
    .filter((r) => !ACTIVE.has(r.status))
    .map((r) => `${r.id}:${r.status}:${r.model_id ?? ''}`)
    .sort()
    .join('|');
}

let watching = false;
/** Starts following the run list and the tab's visibility, once for the app. */
function watch() {
  if (watching) return;
  watching = true;
  createRoot(() => {
    let key: string | undefined;
    effect(() => {
      const runs = watched.runs;
      if (!runs) return;
      const next = modelsKey(runs);
      // The first list is where things stand, not a change.
      if (key !== undefined && next !== key) untrack(modelsChanged);
      key = next;
    });
    // A different server (reconnect, mock): its runs are not a change from the last one's.
    effect(() => {
      void hub.epoch;
      key = undefined;
    });
  });
  if (typeof document !== 'undefined') {
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible' && hub.mode !== 'connecting') modelsChanged();
    });
  }
}

/** Refreshes `r` (quietly, keeping what it shows) whenever the models may have changed. Call in a component's body. */
export function refreshOnModelsChange(r: { refresh(): Promise<void> }): void {
  watch();
  const seen = untrack(() => changes.get());
  effect(() => {
    if (changes.get() !== seen) untrack(() => void r.refresh());
  });
}

/** The models list as a Resource that stays current (see above). Call in a component's body. */
export function modelsResource(): Resource<Model[]> {
  const r = new Resource((s) => hub.api.models(s));
  refreshOnModelsChange(r);
  return r;
}
