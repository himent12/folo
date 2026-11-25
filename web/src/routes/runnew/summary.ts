// What New run's Summary and its dock show: the server's estimate when there is one, else the reference figures.
import type { Augment, Capabilities, Dataset, Model, Optimizer, RunEstimate } from '$lib/api/types';
import { compact, deviceName, num, sentence } from '$lib/format';
import { FROM_SCRATCH } from '$lib/lineage';
import { referenceSource } from '$lib/reference';
import { basisLine, confidenceLine, referenceBasis, updatedWords } from '$lib/throughput';
import { HEAD_LABEL, optimizerLabel } from '../run/resolved';
import type { CloudPick } from './cloudPick';
import { closeMosaicEpochs, type Estimate } from './estimate';
import type { FormErrors, Jump } from './jump';
import type { EstimatePhase } from '$lib/api/types';
import { bytes } from '$lib/format';
import { prefs } from '$lib/prefs';

// Estimates are ranges (lib/throughput `timeRange`); memory is one figure, in the app's one size format.
export const roughBytes = (b: number) => `~${bytes(b, prefs.units)}`;

/** What the Summary, its phone sheet and the dock show: the server's estimate when there is one, else the reference's. */
export interface Shown {
  lo: number;
  hi: number;
  vram: number;
  vramMeasured: boolean;
  basis: string;
  rough: boolean;
  how: string;
  phases: Pick<EstimatePhase, 'phase' | 'epochs' | 'train_imgs_per_s' | 'epoch_s'>[];
  val: { images: number; rate: number; reference: boolean };
  startup: number;
  updated: string | null;
}

/** What the Summary and the dock show: the server's estimate when there is one, else the reference's. */
export function shownEstimate(
  served: { key: string; est: RunEstimate } | null,
  est: Estimate,
  onCloud: boolean,
  refMachine: boolean,
  onCpu: boolean,
  canEstimate: boolean,
  estimating: boolean,
): Shown {
  const r = served?.est;
  if (r)
    return {
      lo: r.est_total_s_low,
      hi: r.est_total_s_high,
      vram: r.est_peak_vram_bytes ?? 0,
      vramMeasured: r.vram_source === 'measured',
      basis: onCloud ? 'The reference machine’s speed' : basisLine(r),
      rough: r.confidence === 'low',
      how: onCloud ? sentence(r.basis.note) : confidenceLine(r),
      phases: r.phases,
      // A measured basis whose validation rate is still the reference's says so on that line.
      val: { images: r.val_images, rate: r.val_imgs_per_s, reference: r.basis.source === 'measured' && r.val_source === 'reference' },
      startup: r.startup_s,
      updated: r.basis.source === 'measured' && r.basis.updated ? updatedWords(r.basis) : null,
    };
  return {
    lo: est.lowSeconds,
    hi: est.highSeconds,
    vram: est.vramBytes,
    vramMeasured: false,
    basis: referenceBasis(refMachine, onCpu),
    rough: true,
    how: !canEstimate
      ? `Estimated from ${referenceSource(onCpu)}, scaled to these settings: this server does not estimate runs.`
      : estimating
        ? "Loading this server's estimate."
        : `Estimated from ${referenceSource(onCpu)}, scaled to these settings: this server could not estimate them.`,
    phases: est.phases,
    val: { images: est.valImages, rate: est.valImgsPerS, reference: false },
    startup: est.startupS,
    updated: null,
  };
}

/**
 * Close mosaic as the trainer applies it: the server's count for these settings (`RunEstimate.close_mosaic`), else
 * the trainer's cap of the request (at most half the run), which the server's agrees with.
 */
export function closeMosaicShown(errors: FormErrors, served: { key: string; est: RunEstimate } | null, estKey: string, augment: Augment, epochs: number): number | null {
  if (errors.close || errors.epochs) return null;
  const n = served?.key === estKey ? served.est.close_mosaic : undefined;
  return typeof n === 'number' && Number.isFinite(n) ? n : closeMosaicEpochs(augment.close_mosaic, epochs);
}

/**
 * What a cloud run costs: the server's range for these settings (`RunEstimate.cost`), else the offer's price over the
 * time shown; the most it can cost, at its time limit. Null without a price (compute that is not billed).
 */
export function cloudCostOf(onCloud: boolean, cloudPick: CloudPick, served: { key: string; est: RunEstimate } | null, shown: Shown): { lo: number; hi: number } | null {
  const price = onCloud ? (cloudPick.offer?.price_per_hour ?? null) : null;
  if (price === null) return null;
  const c = served?.est.cost;
  return c ? { lo: c.est_usd_low, hi: c.est_usd_high } : { lo: (price * shown.lo) / 3600, hi: (price * shown.hi) / 3600 };
}

/** The settings that shape the run, each a way back to its control. Optimizer and head only once not the default. */
export function reviewRows(
  dataset: Dataset | undefined,
  errors: FormErrors,
  arch: string,
  epochs: number,
  batchN: number,
  imgszN: number,
  onCloud: boolean,
  cloudPick: CloudPick,
  deviceLabel: string,
  deviceMsg: string | null,
  caps: Capabilities | undefined,
  ampOn: boolean,
  weights: string,
  picked: Model | undefined,
  optimizer: Optimizer,
  e2e: boolean,
): { key: Jump; label: string; value: string; mono?: boolean; warn?: boolean }[] {
  return [
    { key: 'data', label: 'Data', value: dataset ? `${dataset.name} · ${compact(dataset.splits.train)} ${dataset.splits.train === 1 ? 'image' : 'images'}` : '—', warn: !!errors.dataset },
    // An architecture is a name, in sans like a run's (design/UNIFIED.md §2); a weights path is an identifier, in mono.
    { key: 'model', label: 'Model', value: arch },
    { key: 'epochs', label: 'Epochs', value: errors.epochs ? '—' : num(epochs), warn: !!errors.epochs },
    { key: 'batch', label: 'Batch', value: String(batchN) },
    { key: 'imgsz', label: 'Image size', value: `${imgszN} px` },
    // A cloud run says where it runs ("RunPod · RTX 4090") in the device's place.
    onCloud
      ? { key: 'runon' as const, label: 'Runs on', value: cloudPick.offer ? `${cloudPick.provider} · ${deviceName(cloudPick.offer.gpu)}` : cloudPick.provider, warn: !!errors.runon || !!errors.offer }
      : { key: 'device' as const, label: 'Device', value: deviceLabel, warn: !!deviceMsg },
    ...(onCloud && errors.hours ? [{ key: 'hours' as const, label: 'Stop after', value: 'Invalid', warn: true }] : []),
    ...(caps || onCloud ? [{ key: 'amp' as const, label: 'Mixed precision', value: ampOn ? 'On' : 'Off', warn: !!errors.amp }] : []),
    {
      key: 'weights',
      label: 'Start from',
      value: weights === 'none' ? FROM_SCRATCH : (picked?.name ?? weights),
      mono: weights !== 'none' && !picked,
      warn: !!errors.weights,
    },
    ...(optimizer !== 'auto' ? [{ key: 'opt' as const, label: 'Optimizer', value: optimizerLabel(optimizer) }] : []),
    ...(errors.lr0 ? [{ key: 'lr0' as const, label: 'Learning rate', value: 'Invalid', warn: true }] : []),
    ...(errors.seed ? [{ key: 'seed' as const, label: 'Seed', value: 'Invalid', warn: true }] : []),
    ...(errors.close ? [{ key: 'close' as const, label: 'Close mosaic', value: 'Invalid', warn: true }] : []),
    ...(e2e ? [{ key: 'head' as const, label: 'Head', value: HEAD_LABEL.o2o }] : []),
  ];
}
