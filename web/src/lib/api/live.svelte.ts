// A run kept live: follows its SSE stream and loads its snapshot and metric history. Steps arrive up to ~10/s;
// they are appended to plain arrays and published once per animation frame through `version`, so charts redraw at
// most once a frame no matter how fast events come.
//
// Ordering: the stream replays nothing, so we subscribe *before* fetching history and merge the two by step /
// epoch number. A history response never removes newer events, and a snapshot never overrides a status that an
// event reported after the request was sent.

import { untrack } from 'svelte';
import { hub } from './index.svelte';
import { MockApi } from './mock';
import type { Epoch, LogLine, Run, Step } from './types';

const MAX_LOGS = 2000;

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

      const load = async () => {
        const seq = this.#statusSeq;
        const [run, metrics] = await Promise.all([api.run(runId), api.runMetrics(runId)]);
        if (closed) return;
        this.#absorb(run, metrics.steps, metrics.epochs, seq);
        if (api instanceof MockApi) this.logs = mergeLogs(api.logHistory(runId), this.logs);
        this.loading = false;
        this.#bump();
      };

      // Subscribe first: events that arrive while history loads are kept and merged, not lost.
      const unsub = api.runEvents(runId, {
        connection: (s) => {
          if (!closed) this.connection = s;
        },
        status: ({ status, error }) => {
          if (closed) return;
          this.#statusSeq++;
          this.#lastStatus = { status, error };
          if (this.run) this.run = { ...this.run, status, error };
          if (this.run && !['queued', 'running', 'stopping'].includes(status)) {
            // Terminal: refresh the snapshot for finished time, best and model id.
            const seq = this.#statusSeq;
            api.run(runId).then((r) => {
              if (!closed && seq === this.#statusSeq) this.run = r;
            }, () => {});
          }
        },
        step: (s) => {
          if (closed) return;
          const last = this.steps[this.steps.length - 1];
          if (last && s.step <= last.step) return;
          this.steps.push(s);
          if (this.run && s.step > this.run.step) this.run = { ...this.run, step: s.step, epoch: s.epoch + 1, eta_s: s.eta_s };
          this.#bump();
        },
        epoch: (e) => {
          if (closed) return;
          const last = this.epochs[this.epochs.length - 1];
          if (last && e.epoch <= last.epoch) return;
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
          load().catch(() => {});
        },
      });

      load().catch((e) => {
        if (closed) return;
        this.error = e;
        this.loading = false;
      });

      return () => {
        closed = true;
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

  /** Merges a snapshot + history into what the stream has delivered so far, never moving backwards. */
  #absorb(run: Run, steps: Step[], epochs: Epoch[], statusSeqAtStart: number) {
    this.steps = mergeBy(steps, this.steps, (s) => s.step);
    this.epochs = mergeBy(epochs, this.epochs, (e) => e.epoch);
    let next = run;
    const last = this.steps[this.steps.length - 1];
    if (last && last.step > next.step && ['running', 'stopping'].includes(next.status))
      next = { ...next, step: last.step, epoch: last.epoch + 1, eta_s: last.eta_s };
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
