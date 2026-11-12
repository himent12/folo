// Datasets added or removed in this tab, so a list on screen (or the next one shown) reads them again.

import { signal } from 'fw';

/** Bumped by every dataset action that changes the list (register, generate, remove). */
export const datasetChanges = signal(0);

/** Says the list of datasets changed. */
export function datasetsChanged() {
  datasetChanges.set(datasetChanges.peek() + 1);
}

/**
 * A synthetic dataset this tab asked for that it has not seen land: its name, when, and the datasets there were then.
 * In sessionStorage, so a reload while it generates keeps it (datasets F10): the list then reads itself again until the
 * dataset is there (or the generation is long overdue). No server API says a generation is running.
 */
export interface Generation {
  name: string;
  since: number;
  ids: string[];
}

const KEY = 'folo.datasets.generating';
/** Past this, a generation that never landed (it failed while no tab listened) is no longer waited for. */
const GIVE_UP_MS = 30 * 60_000;

export function generationStarted(g: Generation): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(g));
  } catch {
    // Private mode or storage off: a reload then forgets it, as before.
  }
}

export function generationEnded(): void {
  try {
    sessionStorage.removeItem(KEY);
  } catch {
    // As above.
  }
}

/** The generation this tab waits for, if any and not overdue at `now`. */
export function pendingGeneration(now = Date.now()): Generation | null {
  try {
    const g = JSON.parse(sessionStorage.getItem(KEY) ?? 'null') as Generation | null;
    if (g && typeof g.name === 'string' && typeof g.since === 'number' && Array.isArray(g.ids) && now - g.since < GIVE_UP_MS) return g;
  } catch {
    // Unreadable: nothing to wait for.
  }
  return null;
}

/** Whether `list` has the dataset `g` makes: one that was not there, named as asked (or numbered, `name-2`). */
export function landed(g: Generation, list: readonly { id: string; name: string }[]): boolean {
  const numbered = (n: string) => n.startsWith(`${g.name}-`) && /^\d+$/.test(n.slice(g.name.length + 1));
  return list.some((d) => !g.ids.includes(d.id) && (d.name === g.name || numbered(d.name)));
}
