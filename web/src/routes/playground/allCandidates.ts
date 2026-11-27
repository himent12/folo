// Asking the server or browser wasm for every candidate of a busy image once the confidence (or any per-class
// override) drops to where the first answer was cut, or below the per-class floor sent to folo-wasm.
import { computed, effect, signal, type Signal } from 'fw';
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
import { ALL, ask, candidateFloor, FETCH, noServerWeights, type RunOn } from './predict';
import type { PerClassConfMap } from './perClassConf';
import type { Source } from './source';

const KEY = 'all-candidates';

/** Answers that hold every candidate: module-wide, as the session keeps an answer when the page is left. */
const complete = new WeakSet<Prediction>();
/** Capped answers whose full-candidate completion already failed once (suppresses automatic replay until Retry). */
const completionFailed = new WeakSet<Prediction>();
/** Per-class confidence floors used when a browser WASM prediction was fetched. */
const fetchedPerClassFloors = new WeakMap<Prediction, { base: number; perClass: PerClassConfMap }>();

/** Records the confidence floors that produced a browser WASM prediction so lowering an override refreshes cleanly. */
export function recordFetchedFloors(
  p: Prediction,
  base: number,
  perClass?: Readonly<Record<number, number>> | null,
): void {
  const map: PerClassConfMap = {};
  if (perClass) {
    for (const [k, v] of Object.entries(perClass)) {
      const id = Number(k);
      if (Number.isInteger(id) && id >= 0 && Number.isFinite(v)) map[id] = v;
    }
  }
  fetchedPerClassFloors.set(p, { base, perClass: map });
}

/**
 * True when a browser prediction was fetched with higher per-class (or base) thresholds than the currently
 * active thresholds require, so candidates between the new threshold and the old floor may be missing.
 */
export function needsPerClassFloorRefresh(
  p: Prediction | null,
  conf: number,
  perClassConf?: Readonly<Record<number, number>> | null,
): boolean {
  if (!p || !p.backend) return false;
  const info = fetchedPerClassFloors.get(p);
  const base = info?.base ?? FETCH.conf;
  const prevMap = info?.perClass ?? {};
  // Any previously fetched per-class floor now exceeds the active threshold for that class.
  for (const [k, prevFloor] of Object.entries(prevMap)) {
    const id = Number(k);
    const wanted = thresholdForClass(id, conf, perClassConf);
    if (wanted < prevFloor - 1e-6) return true;
  }
  // Any newly lowered per-class override drops below the base candidate floor (`FETCH.conf`).
  if (perClassConf) {
    for (const [k, wanted] of Object.entries(perClassConf)) {
      const id = Number(k);
      const prevFloor = prevMap[id] ?? base;
      if (Number.isFinite(wanted) && wanted < prevFloor - 1e-6) return true;
    }
  }
  return false;
}

/** Computes a loose browser WASM per-class map covering every active override down to `min(FETCH.conf, override)`. */
function refreshPerClassMap(
  perClassConf?: Readonly<Record<number, number>> | null,
): PerClassConfMap | undefined {
  if (!perClassConf) return undefined;
  const out: PerClassConfMap = {};
  let count = 0;
  for (const [k, v] of Object.entries(perClassConf)) {
    const id = Number(k);
    if (Number.isInteger(id) && id >= 0 && Number.isFinite(v) && v < FETCH.conf) {
      out[id] = v;
      count++;
    }
  }
  return count > 0 ? out : undefined;
}

/** Marks a prediction as holding every candidate (fetched with `ALL`). */
export function markPredictionComplete(p: Prediction): void {
  completionFailed.delete(p);
  complete.add(p);
}

/** Marks a capped prediction's `ALL` completion as failed so the stage does not automatically replay it. */
export function markCompletionFailed(p: Prediction): void {
  completionFailed.add(p);
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

/**
 * A busy image's answer stops at FETCH.max_det candidates and is exact only above the lowest score among them. When the
 * confidence (or a per-class override) goes to or under it — or below a browser WASM per-class floor — every candidate
 * is asked for once, and they replace the answer (its timing, the image's run, stays). Returns whether the answer on
 * screen is cut that way, so its candidate count is "1,000+".
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
  // The answer to complete: a computed, so moving the slider further down does not ask (or abort) again.
  // Enhanced SAHI/TTA runs already use `ALL.max_det`, so they report the cap via `floor` without replaying.
  const short = computed(() => {
    const p = o.pred.get();
    if (!p) return null;
    const c = o.conf();
    const pcc = o.perClassConf ? o.perClassConf() : undefined;
    if (needsPerClassFloorRefresh(p, c, pcc)) return { pred: p, reason: 'floor' as const };
    if (p.inference || complete.has(p)) return null;
    const f = candidateFloor(p, FETCH.max_det);
    return f !== null && effectiveMinConf(c, pcc) <= f ? { pred: p, reason: 'cap' as const } : null;
  });
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
    const id = o.model();
    const s = o.source();
    const on = o.runOn();
    const retryTick = retry.get();
    const explicitRetry = retryTick !== lastRetry;
    lastRetry = retryTick;
    if (!p || !id || !s || p.model_id !== id || o.busy()) return;
    if (!explicitRetry && completionFailed.has(p)) return;
    completionFailed.delete(p);
    const file = o.file(id);
    const inference = o.inference ? o.inference() : DEFAULT_INFERENCE_CONFIG;
    if (inferenceMetaKey(p.inference) !== inferenceConfigKey(inference)) return;
    const pcc = o.perClassConf ? o.perClassConf() : undefined;
    const lowMap = on !== 'server' ? refreshPerClassMap(pcc) : undefined;
    const opts = job?.reason === 'cap' ? ALL : complete.has(p) ? ALL : FETCH;
    const precision = on !== 'server' && o.precision ? o.precision() : undefined;
    const ctrl = new AbortController();
    ask({
      api: hub.api,
      engine: o.engine,
      model: id,
      source: s,
      on,
      file,
      options: opts,
      ...(lowMap ? { perClassConf: lowMap, confInclusive: true } : {}),
      precision,
      inference,
      signal: ctrl.signal,
    }).then(
      (all) => {
        if (ctrl.signal.aborted || o.pred.peek() !== p) return;
        const next = { ...all, timing: p.timing, cold: p.cold, ...(p.inference ? { inference: p.inference } : {}) };
        if (opts === ALL) complete.add(next);
        if (on !== 'server') recordFetchedFloors(next, opts.conf, lowMap);
        takeBack();
        o.pred.set(next);
        o.onCompleted?.(s, id, on, next);
      },
      // The first answer stays: it is still right above its lowest score, and the count still says "1,000+".
      (e: unknown) => {
        if (ctrl.signal.aborted || o.pred.peek() !== p) return;
        if (on !== 'server' && !file && noServerWeights(e)) return o.weightsMissing();
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
