import { computed, effect, type Signal } from 'fw';
import { hub } from '$lib/api/index';
import type { Prediction } from '$lib/api/types';
import type { BrowserEngine } from '$lib/inference/browser';
import { ALL, ask, candidateFloor, noServerWeights, type RunOn } from './predict';
import type { Source } from './source';

/** Answers that hold every candidate: module-wide, as the session keeps an answer when the page is left. */
const complete = new WeakSet<Prediction>();

/**
 * A busy image's answer stops at FETCH.max_det candidates and is exact only above the lowest score among them. When the
 * confidence goes to or under it, every candidate is asked for once, and they replace the answer (its timing, the
 * image's run, stays). Returns whether the answer on screen is cut that way, so its candidate count is "1,000+".
 */
export function askAllWhenCut(o: {
  pred: Signal<Prediction | null>;
  conf: () => number;
  model: () => string | null;
  source: () => Source | null;
  runOn: () => RunOn;
  /** A .folo the user picked for the model. */
  file: (model: string) => File | null;
  engine: BrowserEngine;
  /** In the browser, the server cannot send the model's file (`noServerWeights`) and none was picked. */
  weightsMissing: () => void;
}): () => boolean {
  const floor = computed(() => {
    const p = o.pred.get();
    return p && !complete.has(p) ? candidateFloor(p) : null;
  });
  // The answer to complete: a computed, so moving the slider further down does not ask (or abort) again.
  const short = computed(() => {
    const f = floor.get();
    return f !== null && o.conf() <= f ? o.pred.get() : null;
  });
  // The model, image, backend and file are tracked too: a kept answer restored before the model list has loaded asks
  // once the model is known.
  effect(() => {
    const p = short.get();
    const id = o.model();
    const s = o.source();
    const on = o.runOn();
    if (!p || !id || !s || p.model_id !== id) return;
    const file = o.file(id);
    const ctrl = new AbortController();
    ask({ api: hub.api, engine: o.engine, model: id, source: s, on, file, options: ALL, signal: ctrl.signal }).then(
      (all) => {
        if (ctrl.signal.aborted || o.pred.peek() !== p) return;
        const next = { ...all, timing: p.timing, cold: p.cold };
        complete.add(next);
        o.pred.set(next);
      },
      // The first answer stays: it is still right above its lowest score.
      (e: unknown) => {
        if (!ctrl.signal.aborted && on !== 'server' && !file && noServerWeights(e)) o.weightsMissing();
      },
    );
    return () => ctrl.abort();
  });
  return () => floor.get() !== null;
}
