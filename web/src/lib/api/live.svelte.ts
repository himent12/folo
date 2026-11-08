// A run kept live: follows its SSE stream and loads its snapshot and metric history. Steps arrive up to ~10/s;
// they are appended to plain arrays and published once per animation frame through `version`, so charts redraw at
// most once a frame no matter how fast events come.
//
// Ordering: the stream replays nothing, so we subscribe *before* fetching history and merge the two by step /
// epoch number. A history response never removes newer events, and a snapshot never overrides a status that an
// event reported after the request was sent.
//
// Resume: a stopped / failed run that becomes active again continues from its last finished epoch and repeats the
// steps after it. We follow the server's own rule for its history (docs/API.md, folo-server read_metrics): a step or
// epoch at or below the last one seen replaces everything from it on. The view rewinds to the last finished epoch
// right away; history requested before a rewind is dropped (a generation counter), and history requested after it
// is the server's word up to its last step.

import { untrack } from 'svelte';
import { hub } from './index.svelte';
import type { Epoch, LogLine, Run, Step } from './types';

const MAX_LOGS = 2000;
const ACTIVE = ['queued', 'running', 'stopping'];

/** Union of two lists sorted by `key`, preferring `b` (newer) on equal keys. Exported for tests. */
export function mergeBy<T>(a: T[], b: T[], key: (x: T) => number): T[] {
  if (!b.length) return a;
  if (!a.length) return b;
  const out: T[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length || j < b.length) {
    const x = a[i];
    const y = b[j];
    if (y === undefined || (x !== undefined && key(x) < key(y))) {
      out.push(x as T);
      i++;
    } else if (x === undefined || key(y) < key(x)) {
      out.push(y);
      j++;
    } else {
      out.push(y);
      i++;
      j++;
    }
  }
  return out;
}

function mergeLogs(a: LogLine[], b: LogLine[]): LogLine[] {
  const seen = new Set(a.map((l) => `${l.t}|${l.message}`));
  const out = [...a, ...b.filter((l) => !seen.has(`${l.t}|${l.message}`))];
  out.sort((x, y) => x.t.localeCompare(y.t));
  return out.slice(-MAX_LOGS);
}

export class LiveRun {
  run = $state.raw<Run | undefined>(undefined);
  error = $state.raw<unknown>(undefined);
  loading = $state(true);
  connection = $state<'idle' | 'open' | 'reconnecting'>('idle');
  /** Bumps when steps/epochs/logs change (at most once per frame). */
  version = $state(0);
  steps: Step[] = [];
  epochs: Epoch[] = [];
  logs: LogLine[] = [];
  #raf = 0;
  #retry = $state(0);
  /** Increments with every status event, so a snapshot can tell whether one arrived while it was in flight. */
  #statusSeq = 0;
  #lastStatus: Pick<Run, 'status' | 'error'> | null = null;
  /** Bumps on every rewind (resume): history requested before it describes an earlier attempt and is dropped. */
  #gen = 0;

  constructor(id: () => string | null) {
    $effect(() => {
      const runId = id();
      if (hub.mode === 'connecting' || !runId) return;
      void hub.epoch;
      void this.#retry;
      const api = hub.api;
      let closed = false;
      this.#lastStatus = null;
      untrack(() => {
        this.loading = true;
        this.error = undefined;
        this.run = undefined;
        this.steps = [];
        this.epochs = [];
        this.logs = [];
        this.connection = 'idle';
        this.#bump();
      });

      // Snapshots can resolve out of order (history loads, terminal refreshes): a snapshot applies only if no newer
      // request has been applied by the time it is used.
      let snapIssued = 0;
      let snapApplied = 0;
      const snapshot = (): Promise<{ run: Run; claim: () => boolean }> => {
        const my = ++snapIssued;
        return api.run(runId).then((run) => ({
          run,
          claim: () => {
            if (my < snapApplied) return false;
            snapApplied = my;
            return true;
          },
        }));
      };

      const load = async () => {
        const seq = this.#statusSeq;
        const gen = this.#gen;
        const [snap, metrics, logs] = await Promise.all([
          snapshot(),
          api.runMetrics(runId),
          // Older servers have no /logs; the live stream still fills the view.
          api.runLogs(runId, 500).catch(() => ({ lines: [] })),
        ]);
        if (closed) return;
        // Requested before a resume rewound the view: it may hold the interrupted attempt. Ask again.
        if (gen !== this.#gen) return sync();
        this.#absorb(snap.claim() ? snap.run : (this.run ?? snap.run), metrics.steps, metrics.epochs, seq);
        this.logs = mergeLogs(logs.lines, this.logs);
        this.loading = false;
        this.#bump();
      };

      // Re-synchronising history (on open / reconnect) must not silently give up: a gap would stay forever.
      let syncTimer: ReturnType<typeof setTimeout> | undefined;
      const sync = (attempt = 0) => {
        clearTimeout(syncTimer);
        load().catch(() => {
          if (closed) return;
          syncTimer = setTimeout(() => sync(attempt + 1), Math.min(15_000, 1000 * 2 ** attempt));
        });
      };

      // Subscribe first: events that arrive while history loads are kept and merged, not lost.
      const unsub = api.runEvents(runId, {
        connection: (s) => {
          if (closed) return;
          // EventSource construction does not mean the subscription is live: events emitted between our first
          // history load and the stream opening would be lost, so load (and merge) again once it opens.
          if (s === 'open' && this.connection === 'idle') sync();
          this.connection = s;
        },
        status: ({ status, error }) => {
          if (closed) return;
          this.#statusSeq++;
          this.#lastStatus = { status, error };
          // Finished → active again: the run was resumed (here or elsewhere).
          if (this.run && !ACTIVE.includes(this.run.status) && ACTIVE.includes(status)) this.#rewind();
          if (this.run) this.run = { ...this.run, status, error };
          if (!['queued', 'running', 'stopping'].includes(status)) {
            // Terminal: refresh the snapshot for finished time, best and model id (even if history is still loading).
            const seq = this.#statusSeq;
            snapshot().then((snap) => {
              if (closed || seq !== this.#statusSeq || !snap.claim()) return;
              this.run = snap.run;
            }, () => {});
          }
        },
        step: (s) => {
          if (closed) return;
          // The server's rule: a step at or below the last one replaces everything from it on (a resumed epoch).
          while (this.steps.length && this.steps[this.steps.length - 1]!.step >= s.step) this.steps.pop();
          if (this.epochs.length && this.epochs[this.epochs.length - 1]!.epoch >= s.epoch) this.epochs = this.epochs.filter((e) => e.epoch < s.epoch);
          this.steps.push(s);
          if (this.run) this.run = { ...this.run, step: s.step, epoch: s.epoch, eta_s: s.eta_s };
          this.#bump();
        },
        epoch: (e) => {
          if (closed) return;
          if (this.epochs.length && this.epochs[this.epochs.length - 1]!.epoch >= e.epoch) this.epochs = this.epochs.filter((x) => x.epoch < e.epoch);
          this.epochs.push(e);
          const best = this.run?.best;
          if (this.run && (!best || e.val.map50_95 > best.map50_95))
            this.run = { ...this.run, best: { epoch: e.epoch, map50_95: e.val.map50_95 } };
          this.#bump();
        },
        log: (l) => {
          if (closed) return;
          this.logs.push(l);
          if (this.logs.length > MAX_LOGS) this.logs.splice(0, this.logs.length - MAX_LOGS);
          this.#bump();
        },
        reconnect: () => {
          // The stream replays nothing: refetch history and merge it with what arrived meanwhile.
          sync();
        },
      });

      load().catch((e) => {
        if (closed) return;
        this.error = e;
        this.loading = false;
      });

      return () => {
        closed = true;
        clearTimeout(syncTimer);
        unsub();
        cancelAnimationFrame(this.#raf);
        this.#raf = 0;
      };
    });
  }

  /** Reloads from scratch (after a load error). */
  retry() {
    this.#retry++;
  }

  /**
   * Resumes the run through `call` (POST /api/runs with resume_run_id). The worker can start before the response
   * arrives: a status event received meanwhile is newer than the response and wins, and the history is rewound once
   * whichever of the two comes first.
   */
  async resume(call: () => Promise<Run>): Promise<Run> {
    const seq = this.#statusSeq;
    const r = await call();
    if (this.#statusSeq === seq && this.run && !ACTIVE.includes(this.run.status)) this.#rewind();
    // Progress comes from the rewind and the stream (the response may still carry the interrupted step count); the
    // status from the response unless an event already reported a newer one.
    const cur = this.run;
    const status = this.#statusSeq === seq ? { status: r.status, error: r.error } : (this.#lastStatus ?? { status: r.status, error: r.error });
    this.run = { ...r, ...(cur ? { step: cur.step, epoch: cur.epoch, best: cur.best } : {}), ...status };
    return r;
  }

  /** Drops the steps after the last finished epoch (1-based; a resumed run repeats them) and moves the run there. */
  #rewind() {
    this.#gen++;
    const done = this.epochs[this.epochs.length - 1]?.epoch ?? 0;
    this.steps = this.steps.filter((s) => s.epoch <= done);
    const kept = this.steps[this.steps.length - 1]?.step ?? 0;
    if (this.run) this.run = { ...this.run, step: kept, epoch: done };
    this.#bump();
  }

  /** Merges a snapshot + history into what the stream has delivered so far, never moving backwards. */
  #absorb(run: Run, steps: Step[], epochs: Epoch[], statusSeqAtStart: number) {
    // The server's history is its word up to its last step / epoch; streamed ones beyond it are newer and kept.
    const hs = steps[steps.length - 1]?.step ?? 0;
    const he = epochs[epochs.length - 1]?.epoch ?? 0;
    this.steps = [...steps, ...this.steps.filter((s) => s.step > hs)];
    this.epochs = [...epochs, ...this.epochs.filter((e) => e.epoch > he)];
    let next = run;
    const last = this.steps[this.steps.length - 1];
    if (last && last.step > next.step && ['running', 'stopping'].includes(next.status))
      next = { ...next, step: last.step, epoch: last.epoch, eta_s: last.eta_s };
    for (const e of this.epochs)
      if (!next.best || e.val.map50_95 > next.best.map50_95) next = { ...next, best: { epoch: e.epoch, map50_95: e.val.map50_95 } };
    // A status event that arrived after this request was sent is newer than the snapshot.
    if (this.#statusSeq !== statusSeqAtStart && this.#lastStatus) next = { ...next, ...this.#lastStatus };
    this.run = next;
  }

  #bump() {
    if (typeof requestAnimationFrame === 'undefined') {
      this.version++;
      return;
    }
    if (this.#raf) return;
    this.#raf = requestAnimationFrame(() => {
      this.#raf = 0;
      this.version++;
    });
  }
}
