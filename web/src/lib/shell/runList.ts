// The run list as the shell polls it (RunWatch, every 2 s): the sidebar's live row and the Runs badge read it, so
// the shell asks once. It belongs to the server it came from (`hub.epoch` when it was published): after a reconnect, which may be to another
// server or the mock, it reads as no list until the server in use has sent its own.

import { batch, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { streamedStatus } from '$lib/api/live';
import { currentResourceSeq, type Resource } from '$lib/api/resource';
import type { Run } from '$lib/api/types';

const ACTIVE = new Set(['queued', 'running', 'stopping']);

interface EndedOverride {
  run: Run;
  epoch: number;
  /** Minimum global Resource fetch seq that must commit to retire this override. */
  minSeq: number;
}

const runs = signal<{ v: Run[] | undefined; epoch: number }>({ v: undefined, epoch: -1 });
const since = signal<{ v: number | null; epoch: number }>({ v: null, epoch: -1 });
/** Latest `watched.source.committedSeq` that `RunWatch` has atomically published into `runs` and `since`. */
const publishedSeq = signal<{ v: number; epoch: number }>({ v: 0, epoch: -1 });
const ended = signal<ReadonlyMap<string, EndedOverride>>(new Map());

const isRetired = (entry: EndedOverride, epoch: number, source?: Resource<Run[]>): boolean => {
  if (entry.epoch !== epoch) return true;
  if (source) return source.committedSeq >= entry.minSeq;
  const pub = publishedSeq.get();
  return pub.epoch === epoch && pub.v >= entry.minSeq;
};

function pruneEnded(list: readonly Run[] | undefined, epoch: number, pubSeq?: number) {
  const cur = ended.peek();
  if (cur.size === 0) return;
  if (!list) {
    ended.set(new Map());
    return;
  }
  const byId = new Map(list.map((r) => [r.id, r]));
  const drop = (id: string, entry: EndedOverride) => {
    const listed = byId.get(id);
    return entry.epoch !== epoch || (pubSeq !== undefined && pubSeq >= entry.minSeq) || !listed || !ACTIVE.has(listed.status);
  };
  let dirty = false;
  for (const [id, entry] of cur) {
    if (drop(id, entry)) {
      dirty = true;
      break;
    }
  }
  if (!dirty) return;
  const next = new Map<string, EndedOverride>();
  for (const [id, entry] of cur) {
    if (!drop(id, entry)) next.set(id, entry);
  }
  ended.set(next);
}

/** `since`: when the list last arrived, while its polls fail (the list is then a snapshot, not the current state). */
export const watched = {
  /** RunWatch's Resource while it is mounted: a page that reads the same list starts from it (`startFrom`). */
  source: undefined as Resource<Run[]> | undefined,
  get runs(): Run[] | undefined {
    const r = runs.get();
    return r.epoch === hub.epoch ? r.v : undefined;
  },
  set runs(v: Run[] | undefined) {
    const epoch = untrack(() => hub.epoch);
    runs.set({ v, epoch });
    pruneEnded(v, epoch);
  },
  get since(): number | null {
    const s = since.get();
    return s.epoch === hub.epoch ? s.v : null;
  },
  set since(v: number | null) {
    since.set({ v, epoch: untrack(() => hub.epoch) });
  },
  /** Atomically publishes `RunWatch`'s list, staleness timestamp, and committed sequence so readers never see a torn state. */
  publish(v: Run[] | undefined, staleSince: number | null, committedSeq: number) {
    const epoch = untrack(() => hub.epoch);
    batch(() => {
      runs.set({ v, epoch });
      since.set({ v: staleSince, epoch });
      publishedSeq.set({ v: committedSeq, epoch });
      pruneEnded(v, epoch, committedSeq);
    });
  },
};

/**
 * Records a run whose live SSE stream already reported a terminal state (`completed` / `stopped` / `failed`), so
 * Overview (its header summary, live slot, and Recent runs), the sidebar live row, and the Runs badge reflect the
 * terminal state in the same frame instead of disagreeing with the live card until the next list poll. Immediately
 * triggers an authoritative run-list refresh and retires the override as soon as the Resource supplying the displayed
 * list commits a fetch started at or after this call — never on an aborted/superseded refresh or a failed offline poll.
 */
export function noteRunEnded(run: Run, fallback?: Resource<Run[]>): void {
  if (ACTIVE.has(run.status)) return;
  const epoch = untrack(() => hub.epoch);
  const src = watched.source;
  const target = src ?? fallback;
  let minSeq = currentResourceSeq() + 1;
  if (target) {
    const before = target.requestSeq;
    void (src ? src.invalidate() : target.refresh());
    const after = target.requestSeq;
    if (after > before) minSeq = after;
  }
  const pub = publishedSeq.peek();
  const next = new Map<string, EndedOverride>();
  for (const [id, entry] of ended.peek()) {
    if (entry.epoch === epoch && !(pub.epoch === epoch && pub.v >= entry.minSeq)) next.set(id, entry);
  }
  next.set(run.id, { run, epoch, minSeq });
  ended.set(next);
}

/**
 * Reconciles a polled run list with live stream status transitions (`streamedStatus`) and recently ended runs
 * (`noteRunEnded`) without recomputing on 10 Hz step ticks. Pass `source` when `list` came from a page-local
 * `Resource` rather than `watched.runs`, so terminal overrides retire against the Resource that actually supplied
 * `list`. Only overrides runs that the polled list still reports as active (so an older stream never drags a fresh
 * terminal poll back to `running`), and prefers an up, live stream over any stale terminal override. Returns `list`
 * unchanged when no entry differs.
 */
export function effectiveRuns(list: readonly Run[] | undefined, source?: Resource<Run[]>): Run[] | undefined {
  if (!list) return undefined;
  const epoch = hub.epoch;
  const notes = ended.get();
  let changed = false;
  const out = list.map((r) => {
    if (!ACTIVE.has(r.status)) return r;
    const st = streamedStatus(r.id);
    if (st) {
      if (st !== r.status) {
        changed = true;
        return { ...r, status: st, ...(st !== 'queued' && r.queue_position != null ? { queue_position: null } : {}) };
      }
      return r;
    }
    const note = notes.get(r.id);
    if (note && !isRetired(note, epoch, source)) {
      changed = true;
      return note.run;
    }
    return r;
  });
  return changed ? out : (list as Run[]);
}

/** A run that holds this machine's trainer: training, or stopping after it. */
export const training = (r: { status: string }) => r.status === 'running' || r.status === 'stopping';

/** A run a page streams (the run page, Overview's hero) and the stream says trains: current whatever the list says. */
const streaming = (r: Run) => {
  const st = streamedStatus(r.id);
  return st !== undefined && training({ status: st });
};

/**
 * The Runs badge (sidebar, icon rail, tab bar): how many runs train and how many wait, from the list the shell polls.
 * `live`: one of them is training now (the icon rail's and the tab bar's lime dot). `held`: the list stopped refreshing,
 * so the counts are its last known state (lib/api/freshness); a run a page still streams stays live, as the sidebar's
 * live row does.
 */
export function activeRuns(): { training: number; queued: number; live: boolean; held: boolean } {
  const list = effectiveRuns(watched.runs) ?? [];
  const now = list.filter(training);
  const held = watched.since !== null;
  return {
    training: now.length,
    queued: list.filter((r) => r.status === 'queued').length,
    live: held ? now.some(streaming) : now.length > 0,
    held,
  };
}

/** The Runs row's name with its badge said in words ("Runs, 1 training, 2 queued"), for its tooltip and screen readers. */
export function runsBadgeText(): string {
  const a = activeRuns();
  const parts = [a.training ? `${a.training} training` : '', a.queued ? `${a.queued} queued` : ''].filter(Boolean);
  return parts.length && a.held ? `${parts.join(', ')} (last known)` : parts.join(', ');
}
