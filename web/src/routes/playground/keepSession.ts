// Wires the Playground's state to its session (./session): restored as the page opens, kept as it changes, and
// written into the URL (replacing the entry, a moment after the last change, so a slider drag is one write). The URL
// is followed while the page is open, too: an edited URL, Back or Forward, a link from the Playground to itself.

import { effect, signal, untrack, type Signal } from 'fw';
import type { Prediction } from '$lib/api/types';
import { router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import { DEFAULTS } from './predict';
import { findImage, fromUrl, keep, playgroundPath, restore, session, type FromUrl, type ImageRef } from './session';
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
  /**
   * Starts a choice of image that arrives later, as a file's decode does: the returned function is true while no other
   * image has been chosen (or Clear pressed) since and the page is open.
   */
  choose: () => () => boolean;
  /** The model shown, else the one the URL names (while the list loads). */
  model: () => string | null;
  /** Shows this model, as a pick in the list does (without writing the URL). */
  pickModel: (id: string) => void;
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

  // A linked image being looked up is a choice like any other: it shows only if nothing was chosen after it. Until
  // then the URL goes on naming it, not the image still on screen. A newer link or leaving the page cancels it.
  const linking = signal<{ ref: ImageRef; live: () => boolean } | null>(null);
  let lookup: AbortController | null = null;
  function follow(ref: ImageRef) {
    lookup?.abort();
    const ctrl = (lookup = new AbortController());
    const link = { ref, live: k.choose() };
    linking.set(link);
    const settle = (show: () => void) => {
      if (ctrl.signal.aborted) return;
      lookup = null;
      linking.set(null);
      if (link.live()) show();
    };
    findImage(ref, ctrl.signal).then(
      (found) =>
        settle(() => {
          if (found) k.setSource({ kind: 'dataset', ...found });
          else toast.info('The linked image was not found', 'It is no longer in its dataset. Pick another image.');
        }),
      () => settle(() => toast.info('The linked image could not be loaded', 'Its dataset may have been removed. Pick another image.')),
    );
  }
  effect(() => () => lookup?.abort());

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
  } else if (url.image) follow(url.image);

  /** A URL changed while the page is open: what it names replaces what is shown; what it leaves out stays. */
  function apply(u: FromUrl) {
    if (u.model) k.pickModel(u.model);
    if (u.conf !== null) k.conf.set(u.conf);
    if (u.iou !== null) k.iou.set(u.iou);
    if (u.maxDet !== null) k.maxDet.set(u.maxDet);
    const ref = u.image;
    if (!ref) return;
    const s = k.source.peek();
    const l = linking.peek();
    const shown = s?.kind === 'dataset' && s.ds.id === ref.dataset && s.img.id === ref.image;
    const coming = l?.live() && l.ref.dataset === ref.dataset && l.ref.image === ref.image;
    if (!shown && !coming) follow(ref);
  }

  // Followed from the outside (an edited URL, Back or Forward, a link). The page's own writes (below) are told apart
  // by the path they wrote, and name what is shown already besides.
  let wrote: string | null = null;
  let seen = String(router.route.query);
  effect(() => {
    const r = router.route;
    const qs = String(r.query);
    if (r.name !== 'playground' || qs === seen) return;
    seen = qs;
    if (`${r.path}${qs ? `?${qs}` : ''}` !== wrote) untrack(() => apply(fromUrl(r.query)));
  });

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
    if (linking.get()?.live()) return;
    const timer = setTimeout(
      () =>
        untrack(() => {
          if (router.route.name !== 'playground' || location.hash === `#${path}`) return;
          wrote = path;
          router.go(path, { replace: true });
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
