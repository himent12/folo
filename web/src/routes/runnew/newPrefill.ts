// Query-parameter and `?from=` run prefill for New run, including `?weights=<id>` fine-tuning links.
import { deep, effect, signal, untrack, type ReadSignal, type Signal } from 'fw';
import { ApiError, hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { Arch, Augment, Model, Optimizer, Run } from '$lib/api/types';
import { IMGSZ_STEPS } from '$lib/oom';
import { errorDetail, toast } from '$lib/ui/toast';
import type { CloudPick } from './cloudPick';
import { canonicalDevice } from './device';
import type { Draft } from './draft';
import { arches } from './estimate';
import type { copiedSeed } from './jump';
import { runName, uniqueName } from './name';

/** Toast keys scoped to the active New run link. */
export const FROM_TOAST = 'runnew-from';
export const WEIGHTS_TOAST = 'runnew-weights';

export const DEFAULT_AUGMENT: Augment = {
  mosaic: 1,
  mixup: 0,
  hsv: [0.015, 0.7, 0.4],
  fliplr: 0.5,
  scale: 0.5,
  translate: 0.1,
  close_mosaic: 10,
};

export const IMGSZ_OPTIONS = IMGSZ_STEPS.map(String);

/** Parses an integer query override within `[lo, hi]` aligned to `step`, else null. */
export function parseQueryInt(q: URLSearchParams, key: string, lo: number, hi: number, step = 1): string | null {
  const raw = q.get(key);
  if (raw === null || raw.trim() === '') return null;
  const v = Number(raw);
  return Number.isInteger(v) && v >= lo && v <= hi && v % step === 0 ? String(v) : null;
}

/** Derives the form's initial query-driven fields before async resources load. */
export function parseInitialQuery(q: URLSearchParams, own: Draft | null, draft: Draft | null) {
  const fromId = q.get('from');
  const datasetId = (own ? own.datasetId : q.get('dataset')) ?? draft?.datasetId ?? '';
  const queryImgsz = !own && IMGSZ_OPTIONS.includes(q.get('imgsz') ?? '');
  const imgsz = queryImgsz ? q.get('imgsz')! : (draft?.imgsz ?? '640');
  const imgszChosen = queryImgsz || !!draft?.imgszChosen;
  const rawWeights = !own ? q.get('weights')?.trim() || null : null;
  const queryWeights = rawWeights && rawWeights !== 'none' ? rawWeights : null;
  // Keep `weights` at 'none' while `queryWeights` awaits the model catalog (or when `?weights=none` is explicit)
  // so validation does not flash an arch-mismatch error before the target model's architecture is known.
  const weights = rawWeights === 'none' || queryWeights ? 'none' : (draft?.weights ?? 'none');
  return { fromId, datasetId, queryImgsz, imgsz, imgszChosen, queryWeights, weights };
}

export interface RunPrefillOptions {
  q: URLSearchParams;
  fromId: string | null;
  own: Draft | null;
  queryWeights: string | null;
  runs: Resource<Run[]>;
  models: Resource<Model[]>;
  startable: ReadSignal<Model[]>;
  taken: ReadSignal<Set<string>>;
  datasetId: Signal<string>;
  arch: Signal<Arch>;
  epochs: Signal<number>;
  batch: Signal<string>;
  imgsz: Signal<string>;
  imgszChosen: Signal<boolean>;
  device: Signal<string>;
  optimizer: Signal<Optimizer>;
  amp: Signal<boolean>;
  e2e: Signal<boolean>;
  lr0: Signal<number>;
  seedCopy: ReturnType<typeof copiedSeed>;
  weights: Signal<string>;
  cloudPick: CloudPick;
  augment: Signal<Augment>;
  closeEdited: Signal<boolean>;
  name: Signal<string>;
  nameEdited: Signal<boolean>;
}

/**
 * Loads a copied run (`?from=`) and resolves a linked starting checkpoint (`?weights=`), switching `arch` to match
 * the linked weights' architecture when found. Cleans up link-scoped toasts on unmount.
 */
export function useRunPrefill(o: RunPrefillOptions) {
  const source = new Resource((s) => (o.fromId ? hub.api.run(o.fromId, s) : Promise.resolve(null)));
  const prefilled = signal(!o.fromId || !!o.own);
  const pendingWeights = signal<string | null>(o.queryWeights);

  // Copy an earlier run (?from=run_x, e.g. "Retry with batch 32").
  effect(() => {
    const r = source.data;
    if (!r || (!o.runs.data && !o.runs.error) || untrack(() => prefilled.get())) return;
    const c = r.config;
    o.datasetId.set(c.dataset_id);
    o.arch.set(arches.find((a) => a.arch === o.q.get('arch'))?.arch ?? c.arch);
    o.epochs.set(Number(parseQueryInt(o.q, 'epochs', 1, 10000) ?? c.epochs));
    o.batch.set(parseQueryInt(o.q, 'batch', 1, 1024) ?? String(c.batch));
    o.imgsz.set(parseQueryInt(o.q, 'imgsz', 64, 4096, 32) ?? String(c.imgsz));
    o.imgszChosen.set(true);
    o.device.set(o.q.get('device') === 'cpu' ? 'cpu' : canonicalDevice(c.device));
    o.optimizer.set(c.optimizer);
    o.amp.set(c.amp ?? false);
    o.e2e.set(c.head === 'o2o');
    o.lr0.set(c.lr0);
    o.seedCopy.copy(c.seed);
    // When `?weights=` is also present (including explicit `?weights=none`), keep weights at 'none'
    // instead of restoring the source run's checkpoint.
    o.weights.set(o.queryWeights || o.q.get('weights')?.trim() === 'none' ? 'none' : (c.weights ?? 'none'));
    o.cloudPick.prefill(c.compute, r.cloud?.offer.gpu ?? 'GPU');
    o.augment.set(deep(structuredClone(c.augment)));
    o.closeEdited.set(true);
    o.name.set(uniqueName(`${runName(c.name)}-retry`, o.taken.get()));
    o.nameEdited.set(true);
    prefilled.set(true);
  });

  effect(() => {
    if (source.error && !untrack(() => prefilled.get())) {
      const e = source.error;
      untrack(() => {
        if (e instanceof ApiError && e.code === 'run_not_found') {
          toast.show({
            tone: 'error',
            title: 'Run to copy not found',
            detail: 'It was deleted or the link is wrong. The form starts from the defaults.',
            key: FROM_TOAST,
          });
        } else {
          toast.error(e, 'Could not load the run to copy', { key: FROM_TOAST });
        }
      });
      prefilled.set(true);
    }
  });

  // Resolve `?weights=<id>` once the model catalog arrives (and after any `?from=` copy finishes).
  effect(() => {
    const id = pendingWeights.get();
    if (!id || !prefilled.get()) return;
    const err = o.models.error;
    const all = o.models.data;
    if (err && !all) {
      untrack(() => {
        toast.show({
          tone: 'error',
          title: 'Could not load linked weights',
          detail: `${errorDetail(err)} Retry to load the linked checkpoint.`,
          timeout: 0,
          action: {
            label: 'Retry',
            run: () => void o.models.refresh(),
          },
          key: WEIGHTS_TOAST,
        });
      });
      return;
    }
    if (!all) return;
    const match = o.startable.get().find((m) => m.id === id);
    if (match) {
      untrack(() => toast.clear(WEIGHTS_TOAST));
      if (untrack(() => o.arch.get()) !== match.arch) o.arch.set(match.arch);
      o.weights.set(match.id);
    } else {
      const raw = all.find((m) => m.id === id);
      o.weights.set('none');
      untrack(() => {
        if (raw?.fused === true) {
          toast.show({
            tone: 'info',
            title: 'Cannot fine-tune from fused weights',
            detail: `${raw.name} is an exported inference model. Starting from scratch instead.`,
            key: WEIGHTS_TOAST,
          });
        } else {
          toast.show({
            tone: 'info',
            title: 'Linked weights not found',
            detail: 'That model was deleted or the link is wrong. Starting from scratch instead.',
            key: WEIGHTS_TOAST,
          });
        }
      });
    }
    pendingWeights.set(null);
  });

  effect(() => () => {
    toast.clear(FROM_TOAST);
    toast.clear(WEIGHTS_TOAST);
  });

  return { source, prefilled, pendingWeights };
}
