// Asking the server or browser wasm for every candidate of a busy image once the confidence (or any per-class
// override) drops to where the first answer was cut, or below the fetched candidate floor.
import { computed, effect, signal, untrack, type Signal } from 'fw';
import { hub } from '$lib/api/index';
import type { Prediction } from '$lib/api/types';
import { effectiveMinConf, thresholdForClass } from '$lib/detect/nms';
import {
  DEFAULT_INFERENCE_CONFIG,
  inferenceConfigKey,
  inferenceMetaKey,
  type BrowserEngine,
  type BrowserPrecision,
  type InferenceConfig,
} from '$lib/inference/browser';
import { errorDetail, toast } from '$lib/ui/toast';
import { ALL, ask, candidateFloor, FETCH, noServerWeights, serverFetchConf, type RunOn } from './predict';
import { perClassConfKey, type PerClassConfMap } from './perClassConf';
import type { Source } from './source';

const KEY = 'all-candidates';

/** Answers that hold every candidate: module-wide, as the session keeps an answer when the page is left. */
const complete = new WeakSet<Prediction>();
/** Capped/under-floored answers whose completion already failed once (suppresses automatic replay until Retry). */
const completionFailed = new WeakSet<Prediction>();
/** Confidence floors (base + optional per-class overrides) that produced a prediction (server or browser). */
const fetchedFloors = new WeakMap<Prediction, { base: number; inclusive: boolean; perClass: PerClassConfMap }>();

/**
 * Extracts only per-class overrides strictly below `FETCH.conf` (`0.01`). Forwarding only sub-`0.01` floors during
 * loose candidate fetches preserves full `[0.01, 1.0]` candidates for instant client-side re-filtering and lossless
 * batch caching while still reaching down to `[0, 0.01)` calibrated overrides.
 */
export function loosePerClassFloor(
  perClassConf?: Readonly<Record<number, number>> | null,
): PerClassConfMap | undefined {
  if (!perClassConf) return undefined;
  const out: PerClassConfMap = {};
  let count = 0;
  for (const [k, v] of Object.entries(perClassConf)) {
    const id = Number(k);
    if (Number.isInteger(id) && id >= 0 && Number.isFinite(v) && v >= 0 && v < FETCH.conf) {
      out[id] = v;
      count++;
    }
  }
  return count > 0 ? out : undefined;
}

/** Records the confidence floors that produced a server or browser prediction so lower thresholds refresh cleanly. */
export function recordFetchedFloors(
  p: Prediction,
  base: number,
  perClass?: Readonly<Record<number, number>> | null,
  inclusive = false,
): void {
  const map: PerClassConfMap = {};
  if (perClass) {
    for (const [k, v] of Object.entries(perClass)) {
      const id = Number(k);
      if (Number.isInteger(id) && id >= 0 && Number.isFinite(v)) map[id] = v;
    }
  }
  fetchedFloors.set(p, { base, inclusive, perClass: map });
}

/**
 * True when a server or browser prediction was fetched with a higher confidence floor than the currently active
 * global `conf` or `perClassConf` overrides require, so candidates in `[wanted, fetchedFloor)` may be missing.
 */
export function needsPerClassFloorRefresh(
  p: Prediction | null,
  conf: number,
  perClassConf?: Readonly<Record<number, number>> | null,
): boolean {
  if (!p) return false;
  const info = fetchedFloors.get(p);
  const base = info?.base ?? FETCH.conf;
  const inclusive = info?.inclusive ?? false;
  const prevMap = info?.perClass ?? {};

  const missingFor = (wanted: number, fetched: number): boolean => {
    if (!Number.isFinite(wanted)) return false;
    if (wanted < fetched - 1e-6) return true;
    // Server / strict `score > fetched` drops exact boundary scores when `fetched === wanted` and `fetched > 0`
    // (only relevant below `FETCH.conf` where we need exact inclusive calibration boundaries).
    if (!inclusive && wanted < FETCH.conf && fetched > 0 && Math.abs(wanted - fetched) <= 1e-6) return true;
    return false;
  };

  if (missingFor(conf, base)) return true;
  for (const [k, prevFloor] of Object.entries(prevMap)) {
    const id = Number(k);
    const wanted = thresholdForClass(id, conf, perClassConf);
    if (missingFor(wanted, prevFloor)) return true;
  }
  if (perClassConf) {
    for (const [k, wanted] of Object.entries(perClassConf)) {
      const id = Number(k);
      const prevFloor = prevMap[id] ?? base;
      if (missingFor(wanted, prevFloor)) return true;
    }
  }
  return false;
}

/** Marks a prediction as holding every candidate (fetched with `ALL`). */
export function markPredictionComplete(p: Prediction): void {
  completionFailed.delete(p);
  complete.add(p);
}

/** Marks a capped prediction's completion as failed so the stage does not automatically replay it. */
export function markCompletionFailed(p: Prediction): void {
  completionFailed.add(p);
}

/** True when `p`'s automatic cap completion already failed once. */
export function hasCompletionFailed(p: Prediction): boolean {
  return completionFailed.has(p);
}

/**
 * True when `p` hit the candidate cap of the request that produced it (`FETCH.max_det` for standard runs,
 * `ALL.max_det` for enhanced SAHI/TTA or already-completed runs) at or above the lowest active confidence threshold.
 */
export function isPredictionCutAt(
  p: Prediction | null,
  conf: number,
  perClassConf?: Readonly<Record<number, number>> | null,
): boolean {
  if (!p) return false;
  const cap = p.inference || complete.has(p) ? ALL.max_det : FETCH.max_det;
  const f = candidateFloor(p, cap);
  return f !== null && effectiveMinConf(conf, perClassConf) <= f;
}

/**
 * True when `p` can still be upgraded by a wider `ALL` candidate fetch at `conf` / `perClassConf`. Enhanced (`sahi`/`tta`)
 * predictions already start with `ALL.max_det`, so replaying them cannot increase the budget.
 */
export function needsCompletionAt(
  p: Prediction | null,
  conf: number,
  perClassConf?: Readonly<Record<number, number>> | null,
): boolean {
  if (!p || p.inference || complete.has(p)) return false;
  const f = candidateFloor(p, FETCH.max_det);
  return f !== null && effectiveMinConf(conf, perClassConf) <= f;
}

export interface RefreshPlan {
  useAll: boolean;
  /** Loose per-class floor map for browser WASM (`< FETCH.conf` overrides only), else `undefined`. */
  lowMap: PerClassConfMap | undefined;
  /** Effective base `conf` sent to `FETCH` / `ALL` (honoring sub-`0.01` global/per-class thresholds). */
  fetchConf: number;
  /** Stable key describing this refresh plan (used for equality checks so slider drags never abort in-flight work). */
  key: string;
}

/**
 * Unified coverage check for Single, Compare, and Batch QA / exports: returns a `RefreshPlan` whenever `p` is
 * missing candidates due to either the `FETCH.max_det` cap or a threshold lowered below `p`'s fetched floor.
 */
export function needsCandidateRefresh(
  p: Prediction | null,
  conf: number,
  perClassConf?: Readonly<Record<number, number>> | null,
  on?: RunOn,
): RefreshPlan | null {
  if (!p) return null;
  const capNeeded = needsCompletionAt(p, conf, perClassConf);
  const floorNeeded = needsPerClassFloorRefresh(p, conf, perClassConf);
  if (!capNeeded && !floorNeeded) return null;
  const useAll = capNeeded || complete.has(p);
  const minConf = effectiveMinConf(conf, perClassConf);
  const isBrowser = on ? on !== 'server' : !!p.backend;
  const lowMap = isBrowser ? loosePerClassFloor(perClassConf) : undefined;
  const fetchConf = isBrowser ? Math.min(FETCH.conf, conf) : serverFetchConf(minConf);
  const key = `${useAll ? 'all' : 'fetch'}|${fetchConf}|${perClassConfKey(lowMap)}`;
  return { useAll, lowMap, fetchConf, key };
}

/**
 * A busy image's answer stops at FETCH.max_det candidates and is exact only above the lowest score among them. When the
 * confidence (or a per-class override) goes to or under it — or below the prediction's fetched candidate floor —
 * candidates are upgraded once in a single pass, keeping the original timing. Returns whether the answer on screen is
 * cut at its candidate cap, so its candidate count reads "1,000+".
 */
export function askAllWhenCut(o: {
  pred: Signal<Prediction | null>;
  conf: () => number;
  perClassConf?: () => PerClassConfMap;
  model: () => string | null;
  source: () => Source | null;
  runOn: () => RunOn;
  /** Active browser WASM precision override (ignored on server). */
  precision?: () => BrowserPrecision;
  /** A .folo the user picked for the model. */
  file: (model: string) => File | null;
  /** A prediction of the image is on its way: its answer, not the one on screen, is the one to complete. */
  busy: () => boolean;
  engine: BrowserEngine;
  /** Active inference enhancement (SAHI / TTA / standard) for this prediction. */
  inference?: () => InferenceConfig;
  /** In the browser, the server cannot send the model's file (`noServerWeights`) and none was picked. */
  weightsMissing: () => void;
  /** Optional hook called when full-candidate completion upgrades `pred`, so batch caches stay in sync. */
  onCompleted?: (source: Source, model: string, runOn: RunOn, next: Prediction) => void;
}): () => boolean {
  const floor = computed(() => {
    const p = o.pred.get();
    if (!p) return null;
    const cap = p.inference || complete.has(p) ? ALL.max_det : FETCH.max_det;
    return candidateFloor(p, cap);
  });
  // Stable value-compared refresh target: moving a slider further while an identical completion is already in flight
  // does not abort or restart it.
  const short = computed(
    () => {
      const p = o.pred.get();
      if (!p) return null;
      const plan = needsCandidateRefresh(p, o.conf(), o.perClassConf ? o.perClassConf() : undefined, o.runOn());
      return plan ? { pred: p, plan } : null;
    },
    { equals: (a, b) => a === b || (a !== null && b !== null && a.pred === b.pred && a.plan.key === b.plan.key) },
  );
  const retry = signal(0);
  // The answer a failure toast is about: the toast goes with it (another image, model or backend) and with the page.
  let failed: Prediction | null = null;
  const takeBack = () => {
    if (failed) toast.clear(KEY);
    failed = null;
  };
  effect(() => {
    if (o.pred.get() !== failed) takeBack();
  });
  effect(() => takeBack);
  // The model, image, backend and file are tracked too: a kept answer restored before the model list has loaded asks
  // once the model is known. Never while another prediction runs (a new backend or file): its answer comes next.
  let lastRetry = 0;
  effect(() => {
    const job = short.get();
    const p = job?.pred ?? null;
    const plan = job?.plan ?? null;
    const id = o.model();
    const s = o.source();
    const on = o.runOn();
    const retryTick = retry.get();
    const explicitRetry = retryTick !== lastRetry;
    lastRetry = retryTick;
    if (!p || !plan || !id || !s || p.model_id !== id || o.busy()) return;
    if (!explicitRetry && completionFailed.has(p)) return;
    completionFailed.delete(p);
    const file = o.file(id);
    const inference = o.inference ? untrack(() => o.inference!()) : DEFAULT_INFERENCE_CONFIG;
    if (inferenceMetaKey(p.inference) !== inferenceConfigKey(inference)) return;
    const baseOpts = plan.useAll ? ALL : FETCH;
    const reqOpts = { ...baseOpts, conf: plan.fetchConf };
    const precision = on !== 'server' && o.precision ? untrack(() => o.precision!()) : undefined;
    const ctrl = new AbortController();
    ask({
      api: hub.api,
      engine: o.engine,
      model: id,
      source: s,
      on,
      file,
      options: reqOpts,
      ...(on !== 'server' && plan.lowMap ? { perClassConf: plan.lowMap, confInclusive: true } : {}),
      ...(on !== 'server' && plan.fetchConf < FETCH.conf ? { confInclusive: true } : {}),
      precision,
      inference,
      signal: ctrl.signal,
    }).then(
      (all) => {
        if (ctrl.signal.aborted || o.pred.peek() !== p) return;
        const next = { ...all, timing: p.timing, cold: p.cold, ...(p.inference ? { inference: p.inference } : {}) };
        o.engine.copyPredictionInfo(all, next);
        if (plan.useAll) complete.add(next);
        recordFetchedFloors(next, reqOpts.conf, on !== 'server' ? plan.lowMap : undefined, on !== 'server' && (!!plan.lowMap || reqOpts.conf < FETCH.conf));
        takeBack();
        o.pred.set(next);
        o.onCompleted?.(s, id, on, next);
      },
      // The first answer stays: it is still right above its lowest score, and the count still says "1,000+".
      (e: unknown) => {
        if (ctrl.signal.aborted || o.pred.peek() !== p) return;
        if (on !== 'server' && !file && noServerWeights(e)) return o.weightsMissing();
        completionFailed.add(p);
        const below = floor.peek()?.toFixed(2);
        failed = p;
        toast.show({
          tone: 'error',
          title: 'Could not load every candidate',
          detail: below
            ? `Some boxes scored below ${below} may be missing. ${errorDetail(e)}`
            : errorDetail(e),
          action: { label: 'Retry', run: () => retry.set(retry.peek() + 1) },
          key: KEY,
        });
      },
    );
    return () => ctrl.abort();
  });
  return () => floor.get() !== null;
}
