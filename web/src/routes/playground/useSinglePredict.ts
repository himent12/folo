// Model A prediction effect and busy-image full-candidate completion for Playground.
import { computed, effect, untrack, type Signal } from 'fw';
import { hub } from '$lib/api/index';
import type { Prediction } from '$lib/api/types';
import {
  DEFAULT_INFERENCE_CONFIG,
  inferenceConfigKey,
  inferenceMetaKey,
  isDetectorLoadError,
  type BrowserEngine,
  type InferenceConfig,
} from '$lib/inference/browser';
import { askAllWhenCut } from './allCandidates';
import { ask, FETCH, noServerWeights, type RunOn } from './predict';
import { sourceImageKey, type Source } from './source';
import type { Det } from './viewState';

export interface SinglePredictInput {
  modelId: () => string | null;
  source: Signal<Source | null>;
  runOn: Signal<RunOn>;
  localWeights: Signal<Map<string, File>>;
  retryTick: Signal<number>;
  restored: (model: string, s: Source, runOn: string) => boolean;
  /** Optional one-shot reuse check (e.g. after swapping Model A and Model B in Compare mode). */
  consumeReuse?: (model: string, s: Source, runOn: RunOn) => boolean;
  pred: Signal<Prediction | null>;
  /** Tracks which `sourceImageKey(source)` produced `pred` so exports never pair a frame with stale boxes. */
  predSourceKey: Signal<string | null>;
  predicting: Signal<boolean>;
  error: Signal<unknown>;
  weightsMissing: Signal<boolean>;
  hidden: Signal<Set<number>>;
  hover: Signal<Det | null>;
  picked: Signal<Det | null>;
  conf: Signal<number>;
  engine: BrowserEngine;
  /** Active effective inference config (standard / SAHI / TTA) for Model A's backend. */
  inference?: () => InferenceConfig;
  dropLocal: (id: string, f: File, e: unknown) => void;
  /** True while streaming live video/webcam frames (skips secondary full-candidate re-fetches). */
  isLiveStream?: () => boolean;
  /** True while batch Run All is driving sequential inference (suppresses duplicate single-image requests). */
  isBatchRunningAll?: () => boolean;
  /** Returns a cached batch prediction for `(model, source, runOn, inferenceKey)` if already computed. */
  getCachedBatchPred?: (model: string, s: Source, runOn: RunOn, inferenceKey?: string) => Prediction | null;
  /** True when the user stopped Run All on this unrun batch item (prevents instant auto-restart). */
  wasBatchStopped?: (s: Source) => boolean;
  /** Mirrors single-image completion/error into the active batch item so filmstrip badges update live. */
  onBatchResult?: (s: Source, model: string, runOn: RunOn, pred: Prediction | null, error: unknown, inferenceKey?: string) => void;
  /** Mirrors single-image running state into the active batch item. */
  onBatchBusy?: (s: Source, busy: boolean) => void;
  /** Receives a synchronous abort callback so `Run all` cancels any in-flight single request before dispatching batch work. */
  onRegisterAbort?: (abort: () => void) => void;
}

export function useSinglePredict(o: SinglePredictInput): () => boolean {
  // Subscribe only to Model A's effective local weights file (and ignore weights map changes while running on Server).
  const activeFile = computed(() => {
    const on = o.runOn.get();
    const id = o.modelId();
    if (on === 'server' || !id) return null;
    return o.localWeights.get().get(id) ?? null;
  });

  let lastRetry = 0;
  let activeCtrl: AbortController | null = null;
  o.onRegisterAbort?.(() => {
    activeCtrl?.abort();
    activeCtrl = null;
  });
  effect(() => {
    const id = o.modelId();
    const s = o.source.get();
    const on = o.runOn.get();
    const file = activeFile.get();
    const inf = o.inference ? o.inference() : DEFAULT_INFERENCE_CONFIG;
    const infKey = inferenceConfigKey(inf);
    const retry = o.retryTick.get();
    const batchRunning = !!o.isBatchRunningAll?.();
    const explicitRetry = retry !== lastRetry;
    lastRetry = retry;
    if (batchRunning) return;
    if (!id || !s || (infKey === 'standard' && untrack(() => o.restored(id, s, on))) || untrack(() => !!o.consumeReuse?.(id, s, on))) {
      untrack(() => {
        const curPred = o.pred.peek();
        if (s && curPred) {
          o.predSourceKey.set(sourceImageKey(s));
          if (id) o.onBatchResult?.(s, id, on, curPred, null, infKey);
        }
        o.predicting.set(false);
        o.error.set(null);
        o.weightsMissing.set(false);
      });
      return;
    }
    const cached = !explicitRetry ? untrack(() => o.getCachedBatchPred?.(id, s, on, infKey) ?? null) : null;
    if (cached) {
      untrack(() => {
        const sameDisplayed = o.pred.peek() === cached;
        o.pred.set(cached);
        o.predSourceKey.set(sourceImageKey(s));
        o.predicting.set(false);
        // Preserve batch-halting error / missing-weights notices if re-entering solely because Run All ended.
        if (!sameDisplayed) {
          o.error.set(null);
          o.weightsMissing.set(false);
        }
      });
      return;
    }
    if (!explicitRetry && untrack(() => !!o.wasBatchStopped?.(s))) {
      untrack(() => o.predicting.set(false));
      return;
    }
    const frameKey = sourceImageKey(s);
    const ctrl = (activeCtrl = new AbortController());
    untrack(() => {
      o.predicting.set(true);
      o.error.set(null);
      o.weightsMissing.set(false);
      o.onBatchBusy?.(s, true);
      // Results belong to a model, backend, and inference mode: never show stale boxes while (or if) this run fails.
      const p = o.pred.get();
      if (p && (p.model_id !== id || (p.backend ?? 'server') !== on || inferenceMetaKey(p.inference) !== infKey)) {
        o.pred.set(null);
        o.predSourceKey.set(null);
        if (p.model_id !== id) o.hidden.set(new Set());
        o.hover.set(null);
        o.picked.set(null);
      }
    });
    const run = ask({ api: hub.api, engine: o.engine, model: id, source: s, on, file, options: FETCH, inference: inf, signal: ctrl.signal });
    run.then(
      (p) => {
        if (ctrl.signal.aborted) return;
        if (activeCtrl === ctrl) activeCtrl = null;
        o.pred.set(p);
        o.predSourceKey.set(frameKey);
        o.hover.set(null);
        o.picked.set(null);
        o.predicting.set(false);
        o.onBatchResult?.(s, id, on, p, null, infKey);
      },
      (e: unknown) => {
        if (ctrl.signal.aborted) return;
        if (activeCtrl === ctrl) activeCtrl = null;
        o.predicting.set(false);
        if (file && on !== 'server' && isDetectorLoadError(e)) {
          o.pred.set(null);
          o.predSourceKey.set(null);
          o.hover.set(null);
          o.picked.set(null);
          o.onBatchBusy?.(s, false);
          o.dropLocal(id, file, e);
          return;
        }
        if (on !== 'server' && !file && noServerWeights(e)) {
          o.pred.set(null);
          o.predSourceKey.set(null);
          o.hover.set(null);
          o.picked.set(null);
          o.weightsMissing.set(true);
          o.onBatchBusy?.(s, false);
          return;
        }
        o.pred.set(null);
        o.predSourceKey.set(null);
        o.hover.set(null);
        o.picked.set(null);
        o.error.set(e);
        o.onBatchResult?.(s, id, on, null, e, infKey);
      },
    );
    return () => {
      ctrl.abort();
      if (activeCtrl === ctrl) activeCtrl = null;
      untrack(() => o.onBatchBusy?.(s, false));
    };
  });

  return askAllWhenCut({
    pred: o.pred,
    conf: () => o.conf.get(),
    model: o.modelId,
    source: () => o.source.get(),
    runOn: () => o.runOn.get(),
    file: () => activeFile.peek(),
    busy: () => o.predicting.get() || !!o.isLiveStream?.() || !!o.isBatchRunningAll?.(),
    engine: o.engine,
    inference: () => (o.inference ? untrack(() => o.inference!()) : DEFAULT_INFERENCE_CONFIG),
    weightsMissing: () => o.weightsMissing.set(true),
    onCompleted: (s, id, on, next) => o.onBatchResult?.(s, id, on, next, null, inferenceMetaKey(next.inference)),
  });
}
