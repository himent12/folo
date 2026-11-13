// Datasets added or removed in this tab, so a list on screen (or the next one shown) reads them again.

import { signal } from 'fw';

/** Bumped by every dataset action that changes the list (register, generate, remove). */
export const datasetChanges = signal(0);

/** Says the list of datasets changed. */
export function datasetsChanged() {
  datasetChanges.set(datasetChanges.peek() + 1);
}

/**
 * A dataset this tab asked for that it has not seen land (a sample set generating, or a folder being read: COCO takes
 * minutes): the server it was asked of (`hub.server`), its name, when, and the datasets there were then; for a folder,
 * its path; and what the list says meanwhile ("Reading coco"). In sessionStorage, so a reload while it works keeps it
 * (datasets F10, studio-data F1): the list of that server then says so and reads itself again until the dataset is
 * there (or it is long overdue). No server API says a generation or a registration is running.
 */
export interface Generation {
  server: string;
  name: string;
  since: number;
  ids: string[];
  /** A folder registration's path, as asked: it has landed when a dataset of that path is listed. */
  path?: string;
  /** What is going on, for the list to say ("Generating 5,000 images", "Reading coco"). */
  what?: string;
}

const KEY = 'folo.datasets.generating';

/**
 * Whether this page is going away (a reload, a link out). Its requests fail then, while the server goes on with them:
 * a dataset on its way is not forgotten for that, so the page that follows still waits for it.
 */
let leaving = false;
/** What waits for the page to turn out to stay (`whenStaying`). */
let held: (() => void)[] = [];
const stayed = () => {
  leaving = false;
  const run = held;
  held = [];
  for (const f of run) f();
};
if (typeof window !== 'undefined') {
  // From `beforeunload`: Chrome drops the page's requests (and runs their failure handlers) before `pagehide`. A page
  // still here a while later was not left after all (a leave cancelled at another screen's unsaved-work prompt).
  let stay: ReturnType<typeof setTimeout> | undefined;
  window.addEventListener('beforeunload', () => {
    leaving = true;
    clearTimeout(stay);
    stay = setTimeout(stayed, 3000);
  });
  window.addEventListener('pagehide', () => {
    clearTimeout(stay);
    leaving = true;
  });
  // Back from the back-forward cache: it is a live page again.
  window.addEventListener('pageshow', stayed);
}
export const pageLeaving = () => leaving;

/** Runs `f` if the page that looked like going away stays after all (a leave cancelled at a prompt). */
export function whenStaying(f: () => void): void {
  held.push(f);
}

/** Past this, a generation that never landed (it failed while no tab listened) is no longer waited for. */
const GIVE_UP_MS = 30 * 60_000;

export function generationStarted(g: Generation): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(g));
  } catch {
    // Private mode or storage off: a reload then forgets it, as before.
  }
}

/** The generation `g` is over (landed, done, failed): forgotten, unless a newer one has been asked for since. */
export function generationEnded(g: Generation): void {
  try {
    const kept = JSON.parse(sessionStorage.getItem(KEY) ?? 'null') as Partial<Generation> | null;
    if (kept && (kept.server !== g.server || kept.name !== g.name || kept.since !== g.since)) return;
    sessionStorage.removeItem(KEY);
  } catch {
    // As above.
  }
}

/** The generation this tab waits for on `server` (`hub.server`), if any and not overdue at `now`. */
export function pendingGeneration(server: string, now = Date.now()): Generation | null {
  try {
    const g = JSON.parse(sessionStorage.getItem(KEY) ?? 'null') as Generation | null;
    const valid =
      g &&
      g.server === server &&
      typeof g.name === 'string' &&
      typeof g.since === 'number' &&
      Array.isArray(g.ids) &&
      (g.path === undefined || typeof g.path === 'string') &&
      (g.what === undefined || typeof g.what === 'string');
    if (valid && now - g.since < GIVE_UP_MS) return g;
  } catch {
    // Unreadable: nothing to wait for.
  }
  return null;
}

/**
 * A path as the server lists it and as it was typed, compared: separators, a trailing one, `.` and `..` aside, and
 * case on a Windows path (a drive letter or a UNC share) only: a Linux server's /data/coco and /data/COCO differ.
 */
const samePath = (a: string, b: string) => {
  const norm = (p: string) => {
    const s = p.replace(/\\/g, '/');
    const parts: string[] = [];
    for (const x of s.split('/')) {
      if (x === '..' && parts.length > 1) parts.pop();
      else if (x !== '.' && (x !== '' || parts.length === 0)) parts.push(x);
    }
    const out = parts.join('/');
    return /^([A-Za-z]:|\/\/)/.test(s) ? out.toLowerCase() : out;
  };
  return norm(a) === norm(b);
};

/**
 * Whether `list` has the dataset `g` makes: a folder's, by its path (a folder registered before answers with that
 * dataset, so it may be an old one); a sample set, one that was not there, named as asked (or numbered, `name-2`).
 */
export function landed(g: Generation, list: readonly { id: string; name: string; path?: string }[]): boolean {
  if (g.path !== undefined) return list.some((d) => d.path !== undefined && samePath(d.path, g.path!));
  const numbered = (n: string) => n.startsWith(`${g.name}-`) && /^\d+$/.test(n.slice(g.name.length + 1));
  return list.some((d) => !g.ids.includes(d.id) && (d.name === g.name || numbered(d.name)));
}
