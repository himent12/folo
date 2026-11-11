// Wires the Playground's state to its session (./session): restored as the page opens, kept as it changes, and
// written into the URL (replacing the entry, a moment after the last change, so a slider drag is one write).

import { effect, untrack, type Signal } from 'fw';
import type { Prediction } from '$lib/api/types';
import { router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import { DEFAULTS } from './predict';
import { findImage, fromUrl, keep, playgroundPath, restore, session } from './session';
import type { Source } from './source';

/** How long the URL waits after the last change before it is written. */
export const URL_DELAY_MS = 300;

export interface Kept {
  source: Signal<Source | null>;
  pred: Signal<Prediction | null>;
  runOn: Signal<string>;
  conf: Signal<number>;
  iou: Signal<number>;
  maxDet: Signal<number>;
  labels: Signal<boolean>;
  scores: Signal<boolean>;
  truth: Signal<boolean>;
  hidden: Signal<Set<number>>;
  /** Sets a new image (clearing the result). */
  setSource: (s: Source) => void;
  /** The model shown, else the one the URL names (while the list loads). */
  model: () => string | null;
}

const sourceKey = (s: Source) => (s.kind === 'file' ? `file:${s.name}:${s.file.size}` : `ds:${s.ds.id}:${s.img.id}`);

/**
 * Restores the session (call where every signal of `k` exists, before the prediction effect) and keeps it from then
 * on. Returns `restored(model, source, runOn)`: true once, for the first prediction the page would make, when the
 * kept result is that prediction already (no second request for the same answer).
 */
export function keepSession(k: Kept): (model: string, s: Source, runOn: string) => boolean {
  const saved = session.get();
  const url = fromUrl(router.route.query);
  if (saved) {
    k.labels.set(saved.labels);
    k.scores.set(saved.scores);
    k.truth.set(saved.truth);
    k.runOn.set(saved.runOn);
  }
  k.conf.set(url.conf ?? saved?.conf ?? DEFAULTS.conf);
  k.iou.set(url.iou ?? saved?.iou ?? DEFAULTS.iou);
  k.maxDet.set(url.maxDet ?? saved?.maxDet ?? DEFAULTS.maxDet);

  let reuse: string | null = null;
  const kept = saved?.source ?? null;
  const same = url.image && kept?.kind === 'dataset' && kept.ds.id === url.image.dataset && kept.img.id === url.image.image;
  if (kept && (!url.image || same)) {
    const s = restore(kept);
    k.setSource(s);
    if (saved?.pred) {
      k.pred.set(saved.pred);
      k.hidden.set(new Set(saved.hidden));
      reuse = `${saved.pred.model_id}|${sourceKey(s)}|${saved.runOn}`;
    }
  } else if (url.image) {
    const ref = url.image;
    const ctrl = new AbortController();
    findImage(ref, ctrl.signal).then(
      (found) => {
        if (ctrl.signal.aborted) return;
        if (found) k.setSource({ kind: 'dataset', ...found });
        else toast.info('The linked image was not found', 'It is no longer in its dataset. Pick another image.');
      },
      () => {
        if (!ctrl.signal.aborted) toast.info('The linked image could not be loaded', 'Its dataset may have been removed. Pick another image.');
      },
    );
    effect(() => () => ctrl.abort());
  }

  // Kept as it changes.
  effect(() => {
    const next = {
      source: keep(k.source.get()),
      pred: k.pred.get(),
      runOn: k.runOn.get(),
      conf: k.conf.get(),
      iou: k.iou.get(),
      maxDet: k.maxDet.get(),
      labels: k.labels.get(),
      scores: k.scores.get(),
      truth: k.truth.get(),
      hidden: [...k.hidden.get()],
    };
    session.set(next);
  });

  // Into the URL, once the changes stop for a moment; never after the page is left.
  effect(() => {
    const path = playgroundPath({ model: k.model(), source: k.source.get(), conf: k.conf.get(), iou: k.iou.get(), maxDet: k.maxDet.get() });
    const timer = setTimeout(
      () =>
        untrack(() => {
          if (router.route.name === 'playground' && location.hash !== `#${path}`) router.go(path, { replace: true });
        }),
      URL_DELAY_MS,
    );
    return () => clearTimeout(timer);
  });

  return (model, s, runOn) => {
    const r = reuse;
    reuse = null;
    return r !== null && r === `${model}|${sourceKey(s)}|${runOn}`;
  };
}
