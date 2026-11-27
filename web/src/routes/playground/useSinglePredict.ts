// Model A prediction effect and busy-image full-candidate completion for Playground.
import { effect, untrack, type Signal } from 'fw';
import { hub } from '$lib/api/index';
import type { Prediction } from '$lib/api/types';
import type { BrowserEngine } from '$lib/inference/browser';
import { askAllWhenCut } from './allCandidates';
import { ask, FETCH, noServerWeights, type RunOn } from './predict';
import type { Source } from './source';
import type { Det } from './viewState';

export interface SinglePredictInput {
  modelId: () => string | null;
  source: Signal<Source | null>;
  runOn: Signal<RunOn>;
  localWeights: Signal<Map<string, File>>;
  retryTick: Signal<number>;
  restored: (model: string, s: Source, runOn: string) => boolean;
  pred: Signal<Prediction | null>;
  predicting: Signal<boolean>;
  error: Signal<unknown>;
  weightsMissing: Signal<boolean>;
  hidden: Signal<Set<number>>;
  hover: Signal<Det | null>;
  picked: Signal<Det | null>;
  conf: Signal<number>;
  engine: BrowserEngine;
  dropLocal: (id: string, f: File, e: unknown) => void;
}

export function useSinglePredict(o: SinglePredictInput): () => boolean {
  effect(() => {
    const id = o.modelId();
    const s = o.source.get();
    const on = o.runOn.get();
    const file = o.localWeights.get().get(id ?? '');
    void o.retryTick.get();
    if (!id || !s || untrack(() => o.restored(id, s, on))) {
      untrack(() => {
        o.predicting.set(false);
        o.error.set(null);
        o.weightsMissing.set(false);
      });
      return;
    }
    const ctrl = new AbortController();
    untrack(() => {
      o.predicting.set(true);
      o.error.set(null);
      o.weightsMissing.set(false);
      // Results belong to a model and backend: never show another model's or backend's boxes while (or if) this one fails.
      const p = o.pred.get();
      if (p && (p.model_id !== id || (p.backend ?? 'server') !== on)) {
        o.pred.set(null);
        if (p.model_id !== id) o.hidden.set(new Set());
        o.hover.set(null);
        o.picked.set(null);
      }
    });
    const run = ask({ api: hub.api, engine: o.engine, model: id, source: s, on, file: file ?? null, options: FETCH, signal: ctrl.signal });
    run.then(
      (p) => {
        if (ctrl.signal.aborted) return;
        o.pred.set(p);
        o.hover.set(null);
        o.picked.set(null);
        o.predicting.set(false);
      },
      (e: unknown) => {
        if (ctrl.signal.aborted) return;
        o.predicting.set(false);
        if (file && on !== 'server' && o.engine.phase !== 'ready') {
          o.pred.set(null);
          o.hover.set(null);
          o.picked.set(null);
          o.dropLocal(id, file, e);
          return;
        }
        if (on !== 'server' && !file && noServerWeights(e)) {
          o.pred.set(null);
          o.hover.set(null);
          o.picked.set(null);
          o.weightsMissing.set(true);
          return;
        }
        o.pred.set(null);
        o.hover.set(null);
        o.picked.set(null);
        o.error.set(e);
      },
    );
    return () => ctrl.abort();
  });

  return askAllWhenCut({
    pred: o.pred,
    conf: () => o.conf.get(),
    model: o.modelId,
    source: () => o.source.get(),
    runOn: () => o.runOn.get(),
    file: (id) => o.localWeights.get().get(id) ?? null,
    busy: () => o.predicting.get(),
    engine: o.engine,
    weightsMissing: () => o.weightsMissing.set(true),
  });
}
