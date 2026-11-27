// Wires the Playground's state to its session (./session): restored as the page opens, kept as it changes, and
// written into the URL (replacing the entry, a moment after the last change, so a slider drag is one write). The URL
// is followed while the page is open, too: an edited URL, Back or Forward, a link from the Playground to itself.

import { effect, onCleanup, signal, untrack, type Signal } from 'fw';
import { ApiError } from '$lib/api/client';
import type { Prediction } from '$lib/api/types';
import { inferenceMetaKey, type InferenceConfig } from '$lib/inference/browser';
import { router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import { DEFAULTS } from './predict';
import { formatConfThreshold, normalizePerClassConf, type PerClassConfMap } from './perClassConf';
import type { CompareViewMode } from './compare';
import {
  compareFromUrl,
  consumeThresholdHandoff,
  findImage,
  fromUrl,
  keep,
  perClassFromUrl,
  playgroundPath,
  restore,
  session,
  type FromUrl,
  type ImageRef,
} from './session';
import type { Source } from './source';

/** How long the URL waits after the last change before it is written. */
export const URL_DELAY_MS = 300;

export interface Kept {
  source: Signal<Source | null>;
  pred: Signal<Prediction | null>;
  /** Optional guard confirming `pred` was computed for the current `source` (avoids saving stale live-frame results). */
  predMatchesSource?: () => boolean;
  runOn: Signal<string>;
  /** Optional inference enhancement state (SAHI / TTA) persisted across tab navigation. */
  inference?: { config: () => InferenceConfig; restore: (cfg: InferenceConfig) => void };
  conf: Signal<number>;
  iou: Signal<number>;
  maxDet: Signal<number>;
  perClassConf?: Signal<PerClassConfMap>;
  /** Current model's class list, persisted with `perClassConf` so cross-model links remap by name cleanly. */
  classes?: () => string[];
  /** Called during restore when `perClassConf` was loaded from a saved model, providing its origin model & classes. */
  onRestorePerClassOrigin?: (origin: { modelId: string | null; classes: string[] }) => void;
  labels: Signal<boolean>;
  scores: Signal<boolean>;
  truth: Signal<boolean>;
  hidden: Signal<Set<number>>;
  /** Optional Compare A/B state synced into the URL only when Compare is active. */
  compare?: Signal<boolean>;
  modelB?: () => string | null;
  pickModelB?: (id: string) => void;
  compareView?: Signal<CompareViewMode>;
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
  /** Receives a writer for immediate model-selection URL updates that preserves any in-flight linked image. */
  onModelUrl?: (write: (id: string) => void) => void;
}

const sourceKey = (s: Source) => (s.kind === 'file' ? `file:${s.name}:${s.file.size}` : `ds:${s.ds.id}:${s.img.id}`);

/**
 * Restores the session (call where every signal of `k` exists, before the prediction effect) and keeps it from then
 * on. Returns `restored(model, source, runOn)`: true once, for the first prediction the page would make, when the
 * kept result is that prediction already (no second request for the same answer).
 */
export function keepSession(k: Kept): (model: string, s: Source, runOn: string, infKey?: string) => boolean {
  const saved = session.get();
  const url = fromUrl(router.route.query);
  const urlPcc = perClassFromUrl(router.route.query);
  const cmpUrl = compareFromUrl(router.route.query);
  const handoff = consumeThresholdHandoff(url.model);
  if (k.compare) {
    if (cmpUrl.compare) k.compare.set(true);
    if (cmpUrl.modelB && k.pickModelB) k.pickModelB(cmpUrl.modelB);
    if (cmpUrl.view && k.compareView) k.compareView.set(cmpUrl.view);
  }
  if (saved) {
    k.labels.set(saved.labels);
    k.scores.set(saved.scores);
    k.truth.set(saved.truth);
    k.runOn.set(saved.runOn);
    if (saved.inference && k.inference) k.inference.restore(saved.inference);
  }
  // Track the active per-class origin (so handoff / URL overrides are never misattributed to a prior saved model
  // while the models list is still loading) and preserve explicit default threshold links across debounced URL writes.
  let perClassOrigin: { modelId: string | null; classes: string[] } | null = null;
  let keepExplicitConf = handoff !== null || url.conf === DEFAULTS.conf;
  let keepExplicitEmptyPcc =
    (handoff !== null && Object.keys(handoff.perClassConf).length === 0) ||
    (urlPcc !== null && Object.keys(urlPcc).length === 0);

  if (handoff && k.perClassConf) {
    const norm = normalizePerClassConf(handoff.perClassConf);
    perClassOrigin = { modelId: handoff.modelId, classes: [...handoff.classes] };
    k.perClassConf.set(norm);
    k.onRestorePerClassOrigin?.(perClassOrigin);
    const n = Object.keys(norm).length;
    const detail =
      n > 0
        ? `Global ${formatConfThreshold(handoff.globalConf)} and ${n} ${n === 1 ? 'class threshold' : 'class thresholds'}${handoff.modelName ? ` from ${handoff.modelName}` : ''}.`
        : `Global confidence ${formatConfThreshold(handoff.globalConf)}${handoff.modelName ? ` from ${handoff.modelName}` : ''}.`;
    toast.success('Calibrated thresholds applied', detail, undefined, { key: 'playground-thresholds' });
  } else if (urlPcc !== null && k.perClassConf) {
    perClassOrigin = url.model ? { modelId: url.model, classes: [] } : null;
    k.perClassConf.set(normalizePerClassConf(urlPcc));
    if (perClassOrigin) k.onRestorePerClassOrigin?.(perClassOrigin);
  } else if (saved?.perClassConf && k.perClassConf) {
    perClassOrigin = {
      modelId: saved.perClassModelId ?? saved.pred?.model_id ?? null,
      classes: [...(saved.perClassClasses ?? [])],
    };
    k.perClassConf.set(normalizePerClassConf(saved.perClassConf));
    k.onRestorePerClassOrigin?.(perClassOrigin);
  }
  k.conf.set(handoff?.globalConf ?? url.conf ?? saved?.conf ?? DEFAULTS.conf);
  k.iou.set(url.iou ?? saved?.iou ?? DEFAULTS.iou);
  k.maxDet.set(url.maxDet ?? saved?.maxDet ?? DEFAULTS.maxDet);

  // A linked image being looked up is a choice like any other: it shows only if nothing was chosen after it. Until
  // then the URL goes on naming it, not the image still on screen. A newer link or leaving the page cancels it.
  // A transient lookup error (not 404) keeps the image ref in the URL until the user picks another image, so a reload
  // can retry after the outage.
  const linking = signal<{ ref: ImageRef; live: () => boolean } | null>(null);
  const failedLink = signal<{ ref: ImageRef; live: () => boolean } | null>(null);
  const activeImageRef = (peek: boolean): ImageRef | null => {
    const l = peek ? linking.peek() : linking.get();
    if (l?.live()) return l.ref;
    const f = peek ? failedLink.peek() : failedLink.get();
    return f?.live() ? f.ref : null;
  };
  let lookup: AbortController | null = null;
  function follow(ref: ImageRef) {
    lookup?.abort();
    failedLink.set(null);
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
      (e: unknown) =>
        settle(() => {
          const gone = e instanceof ApiError && e.status === 404;
          if (!gone) failedLink.set(link);
          toast.info(
            'The linked image could not be loaded',
            gone ? 'Its dataset may have been removed. Pick another image.' : 'Reload to try again, or pick another image.',
          );
        }),
    );
  }
  onCleanup(() => lookup?.abort());

  let reuse: string | null = null;
  const kept = saved?.source ?? null;
  const same = url.image && kept?.kind === 'dataset' && kept.ds.id === url.image.dataset && kept.img.id === url.image.image;
  if (kept && (!url.image || same)) {
    const s = restore(kept);
    k.setSource(s);
    if (saved?.pred) {
      k.pred.set(saved.pred);
      k.hidden.set(new Set(saved.hidden));
      reuse = `${saved.pred.model_id}|${sourceKey(s)}|${saved.runOn}|${inferenceMetaKey(saved.pred.inference)}`;
    }
  } else if (url.image) follow(url.image);

  /**
   * A URL changed while the page is open: what it names replaces what is shown. A threshold it leaves out is the
   * default (playgroundPath writes only the others); a model or image it leaves out stays.
   */
  function apply(u: FromUrl, q: URLSearchParams, skipModel = false) {
    const extHandoff = consumeThresholdHandoff(u.model);
    const extPcc = perClassFromUrl(q);
    if (!skipModel && u.model) k.pickModel(u.model);
    if (k.compare) {
      const c = compareFromUrl(q);
      k.compare.set(c.compare);
      if (c.modelB && k.pickModelB) k.pickModelB(c.modelB);
      if (k.compareView) k.compareView.set(c.view ?? 'side-by-side');
    }
    if (extHandoff && k.perClassConf) {
      const norm = normalizePerClassConf(extHandoff.perClassConf);
      perClassOrigin = { modelId: extHandoff.modelId, classes: [...extHandoff.classes] };
      keepExplicitConf = true;
      keepExplicitEmptyPcc = Object.keys(norm).length === 0;
      k.onRestorePerClassOrigin?.(perClassOrigin);
      k.perClassConf.set(norm);
      k.conf.set(extHandoff.globalConf);
    } else {
      keepExplicitConf = u.conf === DEFAULTS.conf;
      keepExplicitEmptyPcc = extPcc !== null && Object.keys(extPcc).length === 0;
      if (k.perClassConf) {
        if (extPcc !== null) {
          perClassOrigin = u.model ? { modelId: u.model, classes: [] } : null;
          if (perClassOrigin) k.onRestorePerClassOrigin?.(perClassOrigin);
          k.perClassConf.set(normalizePerClassConf(extPcc));
        } else {
          // Omitting `pcc` on a live history/URL navigation clears overrides, matching omitted `conf`/`iou`/`max`.
          perClassOrigin = u.model ? { modelId: u.model, classes: [] } : null;
          k.onRestorePerClassOrigin?.(perClassOrigin ?? { modelId: null, classes: [] });
          k.perClassConf.set({});
        }
      }
      k.conf.set(u.conf ?? DEFAULTS.conf);
    }
    k.iou.set(u.iou ?? DEFAULTS.iou);
    k.maxDet.set(u.maxDet ?? DEFAULTS.maxDet);
    const ref = u.image;
    const cancelLink = () => {
      lookup?.abort();
      lookup = null;
      linking.set(null);
      failedLink.set(null);
    };
    if (!ref) {
      cancelLink();
      return;
    }
    const s = k.source.peek();
    const l = linking.peek();
    const shown = s?.kind === 'dataset' && s.ds.id === ref.dataset && s.img.id === ref.image;
    const coming = l?.live() && l.ref.dataset === ref.dataset && l.ref.image === ref.image;
    if (shown) cancelLink();
    if (!shown && !coming) follow(ref);
  }

  // Followed from the outside (an edited URL, Back or Forward, a link). The page's own write (below) is told apart by
  // the path it wrote, once: the same path reached later by Back or Forward is followed (it may name other values).
  let wrote: string | null = null;
  let seen = String(router.route.query);
  function syncFromRoute(skipModel = false) {
    const r = router.route;
    const qs = String(r.query);
    if (r.name !== 'playground' || qs === seen) return;
    seen = qs;
    const own = `${r.path}${qs ? `?${qs}` : ''}` === wrote;
    wrote = null;
    if (!own) untrack(() => apply(fromUrl(r.query), r.query, skipModel));
  }
  k.onModelUrl?.((id) => {
    // Apply any pending external route change first (followVanished can run before our route effect).
    syncFromRoute(true);
    // Check router.location (not router.route) so a lazy destination chunk holding the displayed route is never cancelled.
    const loc = router.location;
    if (loc.name !== 'playground') return;
    const path = playgroundPath({
      model: id,
      source: k.source.peek(),
      imageRef: activeImageRef(true),
      conf: k.conf.peek(),
      iou: k.iou.peek(),
      maxDet: k.maxDet.peek(),
      perClassConf: k.perClassConf?.peek(),
      explicitConf: keepExplicitConf,
      explicitPerClass: keepExplicitEmptyPcc,
      compare: k.compare?.peek(),
      modelB: k.modelB ? untrack(() => k.modelB!()) : null,
      view: k.compareView?.peek(),
    });
    const qs = String(loc.query);
    if (`${loc.path}${qs ? `?${qs}` : ''}` === path) return;
    wrote = path;
    router.go(path, { replace: true });
  });
  effect(() => {
    syncFromRoute();
  });

  // Kept as it changes.
  effect(() => {
    const matches = k.predMatchesSource ? k.predMatchesSource() : true;
    const next = {
      source: keep(k.source.get()),
      pred: matches ? k.pred.get() : null,
      runOn: k.runOn.get(),
      inference: k.inference ? k.inference.config() : undefined,
      conf: k.conf.get(),
      iou: k.iou.get(),
      maxDet: k.maxDet.get(),
      ...(k.perClassConf
        ? (() => {
            const cls = k.classes ? [...k.classes()] : [];
            const curModel = k.model();
            if (cls.length > 0) {
              perClassOrigin = { modelId: curModel, classes: cls };
            }
            const keepOrigin =
              cls.length === 0 && perClassOrigin !== null && (curModel === null || perClassOrigin.modelId === curModel);
            return {
              perClassConf: normalizePerClassConf(k.perClassConf.get()),
              perClassModelId: keepOrigin ? perClassOrigin!.modelId : curModel,
              perClassClasses: keepOrigin ? [...perClassOrigin!.classes] : cls,
            };
          })()
        : {}),
      labels: k.labels.get(),
      scores: k.scores.get(),
      truth: k.truth.get(),
      hidden: [...k.hidden.get()],
    };
    session.set(next);
  });

  // Into the URL, once the changes stop for a moment; never after the page is left (including while a destination chunk loads).
  effect(() => {
    const liveLink = linking.get();
    const cmpOn = k.compare ? k.compare.get() : false;
    const path = playgroundPath({
      model: k.model(),
      source: k.source.get(),
      imageRef: activeImageRef(false),
      conf: k.conf.get(),
      iou: k.iou.get(),
      maxDet: k.maxDet.get(),
      perClassConf: k.perClassConf?.get(),
      explicitConf: keepExplicitConf,
      explicitPerClass: keepExplicitEmptyPcc,
      compare: cmpOn,
      modelB: cmpOn && k.modelB ? k.modelB() : null,
      view: cmpOn && k.compareView ? k.compareView.get() : undefined,
    });
    if (liveLink?.live()) return;
    const timer = setTimeout(
      () =>
        untrack(() => {
          const loc = router.location;
          const qs = String(loc.query);
          if (loc.name !== 'playground' || `${loc.path}${qs ? `?${qs}` : ''}` === path) return;
          wrote = path;
          router.go(path, { replace: true });
        }),
      URL_DELAY_MS,
    );
    return () => clearTimeout(timer);
  });

  return (model, s, runOn, infKey = 'standard') => {
    const r = reuse;
    reuse = null;
    return r !== null && r === `${model}|${sourceKey(s)}|${runOn}|${infKey}`;
  };
}
