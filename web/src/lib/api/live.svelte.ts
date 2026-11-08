// A run kept live: loads its snapshot and metric history, then follows its SSE stream. Steps arrive up to ~10/s;
// they are appended to plain arrays and published once per animation frame through `version`, so charts redraw
// at most once a frame no matter how fast events come.

import { untrack } from 'svelte';
import { hub } from './index.svelte';
import { MockApi } from './mock';
import type { Epoch, LogLine, Run, RunStatus, Step } from './types';

const MAX_LOGS = 2000;
const ACTIVE: RunStatus[] = ['queued', 'running', 'stopping'];

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
  #seq = 0;

  constructor(id: () => string | null) {
    $effect(() => {
      const runId = id();
      if (hub.mode === 'connecting' || !runId) return;
      void hub.epoch;
      const seq = ++this.#seq;
      let unsub: (() => void) | null = null;
      let closed = false;
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
      const api = hub.api;
      const load = async () => {
        const [run, metrics] = await Promise.all([api.run(runId), api.runMetrics(runId)]);
        if (closed || seq !== this.#seq) return;
        this.run = run;
        this.steps = metrics.steps;
        this.epochs = metrics.epochs;
        if (api instanceof MockApi) this.logs = api.logHistory(runId).slice(-MAX_LOGS);
        this.loading = false;
        this.#bump();
        if (ACTIVE.includes(run.status)) unsub = this.#follow(runId, () => closed || seq !== this.#seq);
      };
      load().catch((e) => {
        if (closed || seq !== this.#seq) return;
        this.error = e;
        this.loading = false;
      });
      return () => {
        closed = true;
        unsub?.();
        cancelAnimationFrame(this.#raf);
        this.#raf = 0;
      };
    });
  }

  #follow(runId: string, stale: () => boolean): () => void {
    const api = hub.api;
    return api.runEvents(runId, {
      connection: (s) => {
        if (!stale()) this.connection = s;
      },
      status: ({ status, error }) => {
        if (stale() || !this.run) return;
        this.run = { ...this.run, status, error };
        if (!ACTIVE.includes(status)) {
          // Terminal: refresh the snapshot for finished time, best and model id.
          api.run(runId).then((r) => {
            if (!stale()) this.run = r;
          }, () => {});
        }
      },
      step: (s) => {
        if (stale() || !this.run) return;
        const last = this.steps[this.steps.length - 1];
        if (last && s.step <= last.step) return;
        this.steps.push(s);
        this.run = { ...this.run, step: s.step, epoch: s.epoch + 1, eta_s: s.eta_s };
        this.#bump();
      },
      epoch: (e) => {
        if (stale() || !this.run) return;
        const last = this.epochs[this.epochs.length - 1];
        if (last && e.epoch <= last.epoch) return;
        this.epochs.push(e);
        const best = this.run.best;
        if (!best || e.val.map50_95 > best.map50_95) this.run = { ...this.run, best: { epoch: e.epoch, map50_95: e.val.map50_95 } };
        this.#bump();
      },
      log: (l) => {
        if (stale()) return;
        this.logs.push(l);
        if (this.logs.length > MAX_LOGS) this.logs.splice(0, this.logs.length - MAX_LOGS);
        this.#bump();
      },
      reconnect: () => {
        // The stream replays nothing: refetch history to fill the gap.
        Promise.all([api.run(runId), api.runMetrics(runId)]).then(([run, m]) => {
          if (stale()) return;
          this.run = run;
          this.steps = m.steps;
          this.epochs = m.epochs;
          if (api instanceof MockApi) this.logs = api.logHistory(runId).slice(-MAX_LOGS);
          this.#bump();
        }, () => {});
      },
    });
  }

  #bump() {
    if (this.#raf || typeof requestAnimationFrame === 'undefined') {
      if (typeof requestAnimationFrame === 'undefined') this.version++;
      return;
    }
    this.#raf = requestAnimationFrame(() => {
      this.#raf = 0;
      this.version++;
    });
  }
}
