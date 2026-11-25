// Reactive entity fetcher for the command palette: abortable per-open requests, server-epoch isolation, and deferred loop loads.
import { computed, effect, onCleanup, signal, untrack, type Signal } from 'fw';
import { ApiError, hub } from '$lib/api/index';
import type { Bench, BenchLoop, Dataset, LabelJob, Model, Run } from '$lib/api/types';
import { hasCloud } from '$lib/cloud';
import { LABEL_FEATURE } from '$lib/label';

const isUnsupported = (e: unknown) => e instanceof ApiError && (e.status === 404 || e.status === 501);

export interface PaletteDataOptions {
  open: () => boolean;
  isDialogOpen: () => boolean;
  query: Signal<string>;
  recent: Signal<string[]>;
}

/** Manages all entity list fetches for an open CommandPalette without leaking stale answers across closes or server switches. */
export function createPaletteData(opts: PaletteDataOptions) {
  const runs = signal<Run[]>([]);
  const datasets = signal<Dataset[]>([]);
  const models = signal<Model[]>([]);
  /** Benchmarks, on servers that have them (a server without answers 501 or 404: none listed). */
  const benches = signal<Bench[]>([]);
  const benchesOk = signal(false);
  /** Labeling jobs, and the benchmarks' self-improvement loops, on servers that have them (none listed otherwise). */
  const labelJobs = signal<LabelJob[]>([]);
  const loops = signal<BenchLoop[]>([]);
  /** The server has cloud compute (`features`: `cloud`): only then is Settings' Cloud section there to open. */
  const cloud = signal(false);
  /** In-flight entity lookup keys for the active opening (so empty results say "Searching…" instead of "No matches"). */
  const pendingSources = signal<Set<string>>(new Set());
  /** Entity lookups that failed transiently (non-404/501) during the active opening. */
  const failedSources = signal<Set<string>>(new Set());
  const loadingEntities = computed(() => pendingSources.get().size > 0);
  const failedEntityList = computed(() => [...failedSources.get()]);

  /** The hub epoch the lists above were fetched under. */
  let listed: number | null = null;
  /** Per-opening generation token + AbortControllers so closed/superseded fetches are aborted and discarded. */
  let openSeq = 0;
  let activeEpoch = -1;
  let activeApi: typeof hub.api | null = null;
  let fetchCtrl: AbortController | null = null;
  let loopsCtrl: AbortController | null = null;
  let loopsFetchedSeq = -1;
  /** Reactive generation token of the latest completed `benches()` response for the current opening. */
  const benchesSeq = signal(-1);

  const cancelFetches = () => {
    openSeq++;
    fetchCtrl?.abort();
    loopsCtrl?.abort();
    fetchCtrl = null;
    loopsCtrl = null;
    loopsFetchedSeq = -1;
    benchesSeq.set(-1);
  };
  onCleanup(cancelFetches);

  function markPending(key: string) {
    const nextPending = new Set(pendingSources.peek());
    nextPending.add(key);
    pendingSources.set(nextPending);
    const curFailed = failedSources.peek();
    if (curFailed.has(key)) {
      const nextFailed = new Set(curFailed);
      nextFailed.delete(key);
      failedSources.set(nextFailed);
    }
  }

  function settlePending(key: string, failed: boolean) {
    const curPending = pendingSources.peek();
    if (curPending.has(key)) {
      const nextPending = new Set(curPending);
      nextPending.delete(key);
      pendingSources.set(nextPending);
    }
    if (failed) {
      const nextFailed = new Set(failedSources.peek());
      nextFailed.add(key);
      failedSources.set(nextFailed);
    }
  }

  function clearPending() {
    pendingSources.set(new Set());
  }

  function startFetch(epoch: number) {
    cancelFetches();
    const seq = ++openSeq;
    const ctrl = new AbortController();
    fetchCtrl = ctrl;
    pendingSources.set(new Set());
    failedSources.set(new Set());
    const api = hub.api;
    activeEpoch = epoch;
    activeApi = api;
    // What the lists hold came from the server in use at the last opening: another one's (a reconnect since) goes,
    // and so does an answer that arrives after the next reconnect or a subsequent open/close cycle.
    if (epoch !== listed) {
      listed = epoch;
      runs.set([]);
      datasets.set([]);
      models.set([]);
      benches.set([]);
      benchesOk.set(false);
      labelJobs.set([]);
      loops.set([]);
      cloud.set(false);
    }
    const isCurrent = () => opts.isDialogOpen() && hub.epoch === epoch && openSeq === seq && !ctrl.signal.aborted;
    const mine =
      <A extends unknown[]>(f: (...a: A) => void) =>
      (...a: A) => {
        if (isCurrent()) f(...a);
      };
    const load = <T,>(key: string, req: () => Promise<T>, onOk: (v: T) => void, onErr?: (err: unknown) => boolean) => {
      markPending(key);
      req().then(
        mine((v) => {
          onOk(v);
          settlePending(key, false);
        }),
        mine((err) => {
          const failed = onErr ? onErr(err) : true;
          settlePending(key, failed);
        }),
      );
    };
    load('runs', () => api.runs(ctrl.signal), (r) => runs.set(r));
    load('datasets', () => api.datasets(ctrl.signal), (d) => datasets.set(d));
    load('models', () => api.models(ctrl.signal), (m) => models.set(m));
    load(
      'benchmarks',
      () => api.benches(ctrl.signal),
      (b) => {
        benches.set(b);
        benchesOk.set(true);
        if (!b.length) loops.set([]);
        benchesSeq.set(seq);
      },
      (err) => {
        benches.set([]);
        benchesOk.set(false);
        loops.set([]);
        return !isUnsupported(err);
      },
    );
    if (untrack(() => hub.lacks(LABEL_FEATURE))) {
      labelJobs.set([]);
    } else {
      load(
        'labeling jobs',
        () => api.labelJobs(ctrl.signal),
        (j) => labelJobs.set(j),
        (err) => {
          labelJobs.set([]);
          return !isUnsupported(err);
        },
      );
    }
    api.system(ctrl.signal).then(mine((s) => cloud.set(hasCloud(s))), mine(() => cloud.set(false)));
  }

  // Defer per-benchmark loop fetches until a non-empty query (or a recent loop entry) needs them AND the current
  // opening's benchmark list has arrived, avoiding N+1 benchLoops calls on open and stale benchmark races on reopen.
  effect(() => {
    if (!opts.open()) return;
    const ok = benchesOk.get();
    const bSeq = benchesSeq.get();
    const list = benches.get();
    const q = opts.query.get().trim();
    const wantsLoops = q.length > 0 || opts.recent.get().some((id) => id.startsWith('loop-'));
    if (!opts.isDialogOpen() || !ok || !list.length || !wantsLoops) return;
    if (hub.epoch !== activeEpoch || bSeq !== openSeq || !activeApi) return;
    const seq = openSeq;
    if (loopsFetchedSeq === seq) return;
    loopsFetchedSeq = seq;
    const epoch = activeEpoch;
    const api = activeApi;
    loopsCtrl?.abort();
    const ctrl = new AbortController();
    loopsCtrl = ctrl;
    untrack(() => {
      markPending('loops');
      const perBench: BenchLoop[][] = Array.from({ length: list.length }, () => []);
      let failed = false;
      let idx = 0;
      const workers = Math.min(6, list.length);
      Promise.all(
        Array.from({ length: workers }, async () => {
          while (idx < list.length && !ctrl.signal.aborted && opts.isDialogOpen()) {
            const myIdx = idx++;
            const b = list[myIdx]!;
            try {
              const res = await api.benchLoops(b.id, ctrl.signal);
              if (ctrl.signal.aborted) return;
              perBench[myIdx] = res;
            } catch (err) {
              if (ctrl.signal.aborted) return;
              if (!isUnsupported(err)) failed = true;
            }
          }
        }),
      ).then(() => {
        if (!opts.isDialogOpen() || hub.epoch !== epoch || openSeq !== seq || ctrl.signal.aborted) return;
        loops.set(perBench.flat());
        settlePending('loops', failed);
      });
    });
  });

  return {
    runs,
    datasets,
    models,
    benches,
    benchesOk,
    labelJobs,
    loops,
    cloud,
    loadingEntities,
    failedEntityList,
    isStaleEpoch: (epoch: number) => epoch !== listed,
    startFetch,
    cancelFetches,
    clearPending,
  };
}
