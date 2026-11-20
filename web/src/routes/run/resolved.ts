// A run's settings as it trains with them (docs/API.md, `Run.resolved`): `config` is what was asked for ("Optimizer
// auto", "Initial LR 0.01"), `resolved` what the trainer made of it (MuSGD at 0.01). With `resolved`, the resolved
// value is the one shown, the request a note beside it when it differs; without it (queued, older servers) the
// request as today.

import { closeMosaicEpochs } from '$lib/api/progress';
import type { Optimizer, ResolvedTrain, TrainConfig } from '$lib/api/types';
import { pct } from '$lib/format';

/**
 * One setting: its value, whether that value is an identifier (mono, `ui/Ident`), a sans note after it, and what the
 * value means. A value cell holds one short value ("Auto"); an explanation of it ("MuSGD for every run") is
 * `about`, set as a `.sub` line under the setting's label, never a long value that wraps to a second line of its own.
 */
export interface Setting {
  value: string;
  ident?: boolean;
  note?: string;
  about?: string;
}

const OPTIMIZER: Record<ResolvedTrain['optimizer'], string> = { sgd: 'SGD', adamw: 'AdamW', musgd: 'MuSGD' };
/** What "Auto" picks, in the words of the picker's option (`optimizerLabel`). */
export const AUTO_OPTIMIZER_ABOUT = 'MuSGD for every run';

/**
 * An optimizer as every screen names it (New run's picker and Summary, a run's Config, Compare): "Auto (MuSGD for every
 * run)", "SGD", "AdamW". Named optimizers are names, set in sans like any other word: in Geist Mono the
 * struck W of "AdamW" reads as "₩" (DESIGN_DIRECTION §2.3, identifiers vs names).
 */
export function optimizerLabel(o: Optimizer | ResolvedTrain['optimizer']): string {
  return o === 'auto' ? `Auto (${AUTO_OPTIMIZER_ABOUT})` : OPTIMIZER[o];
}

/**
 * "MuSGD (auto)", "SGD" (asked for by name); unresolved, the request: "Auto", what it picks under the label (`about`).
 * Never an identifier.
 */
export function optimizerSetting(config: TrainConfig, resolved: ResolvedTrain | null | undefined): Setting {
  if (!resolved) return config.optimizer === 'auto' ? { value: 'Auto', about: AUTO_OPTIMIZER_ABOUT } : { value: optimizerLabel(config.optimizer) };
  return { value: optimizerLabel(resolved.optimizer), note: config.optimizer === 'auto' ? '(auto)' : undefined };
}

/** "0.001429 (requested 0.01)" when the trainer's LR differs from the request, else the one value. */
export function lrSetting(config: TrainConfig, resolved: ResolvedTrain | null | undefined): Setting {
  if (!resolved || resolved.lr0 === config.lr0) return { value: String(resolved?.lr0 ?? config.lr0) };
  return { value: String(resolved.lr0), note: `(requested ${config.lr0})` };
}

/**
 * A batch and an optimizer step are two things (DESIGN_DIRECTION §2.23): the trainer sums the gradients of
 * `accumulate` batches into one optimizer step. The batch is the images per batch ("16"); the step is its own row,
 * "Effective batch 64 (4 × 16)", in the one wording New run's Summary uses beside its schedule (`accumulationText`).
 */
export function batchSetting(config: TrainConfig, resolved: ResolvedTrain | null | undefined): Setting {
  return { value: String(resolved?.batch ?? config.batch) };
}

/** "4 batches per optimizer step · effective batch 64": the one wording for accumulation, wherever it is shown. */
export function accumulationText(accumulate: number, batch: number): string {
  return `${accumulate} ${accumulate === 1 ? 'batch' : 'batches'} per optimizer step · effective batch ${accumulate * batch}`;
}

/**
 * The Effective batch row: the images behind one optimizer step, "64 (4 × 16)", four batches of 16 (the note short
 * enough for a phone's Compare column; New run's schedule says it in words). No note when each batch is its own step.
 */
export function effectiveBatchSetting(resolved: ResolvedTrain): Setting {
  const a = resolved.accumulate;
  return a > 1 ? { value: String(a * resolved.batch), note: `(${a} × ${resolved.batch})` } : { value: String(resolved.batch) };
}

/** The warmup in epochs (one decimal under 10): 375 optimizer steps of 125 an epoch are 3. */
export function warmupEpochs(resolved: ResolvedTrain): number {
  const epochs = resolved.warmup_iters / Math.max(1, resolved.optimizer_steps_per_epoch);
  return Number(epochs.toFixed(epochs < 10 ? 1 : 0));
}

/** The warmup in optimizer steps, and what that is in epochs: "100 optimizer steps (7.7 epochs)"; "None" without one. */
export function warmupSetting(resolved: ResolvedTrain): Setting {
  if (resolved.warmup_iters <= 0) return { value: 'None' };
  const e = warmupEpochs(resolved);
  return { value: `${resolved.warmup_iters} optimizer steps`, note: `(${e} ${e === 1 ? 'epoch' : 'epochs'})` };
}

/** "Linear to 1%": the decay over the run, to `lrf` of the initial LR at the last step. */
export function decaySetting(resolved: ResolvedTrain): Setting {
  return { value: `${resolved.cos_lr ? 'Cosine' : 'Linear'} to ${pct(resolved.lrf, resolved.lrf < 0.01 ? 1 : 0)}` };
}

/** AdamW's betas ("0.9 · 0.999"), or SGD's momentum, after warmup. */
export function momentumSetting(resolved: ResolvedTrain): [label: string, Setting] {
  return resolved.betas ? ['Betas', { value: resolved.betas.join(' · ') }] : ['Momentum', { value: String(resolved.momentum) }];
}

/**
 * The EMA's nominal window `1 / (1 − decay)` in optimizer steps, as the trainer's configuration log line says it
 * ("EMA window 100 steps"; Desktop's Config row has the same words): "window 100 steps". Null for a run without an
 * EMA and from servers older than `resolved.ema_decay`.
 */
export function emaSetting(resolved: ResolvedTrain): Setting | null {
  const d = resolved.ema_decay;
  if (typeof d !== 'number' || !(d < 1)) return null;
  const n = Math.round(1 / (1 - d));
  return { value: `window ${n} ${n === 1 ? 'step' : 'steps'}` };
}

/** The resolved rows a run's Config adds after its request: decay, warmup, momentum, the scaled weight decay, the EMA. */
export function resolvedRows(resolved: ResolvedTrain): [label: string, Setting][] {
  const ema = emaSetting(resolved);
  return [
    ['LR decay', decaySetting(resolved)],
    ['Warmup', warmupSetting(resolved)],
    momentumSetting(resolved),
    ['Weight decay', { value: String(resolved.weight_decay) }],
    ...(ema ? [['EMA', ema] as [string, Setting]] : []),
  ];
}

/** A setting as one string, for telling cells apart ("6 identical settings") and plain-text uses. */
export const settingText = (s: Setting) => (s.note ? `${s.value} ${s.note}` : s.value);

/** The detection head in words, as Model › Details says it (docs/MODEL.md §6.6). */
export const HEAD_LABEL = { o2m: 'NMS (one-to-many)', o2o: 'End-to-end (no NMS)' } as const;
/** Training precision in words, as Model › Details says it ("Trained in"). */
export const PRECISION_LABEL = { amp: 'Mixed precision (f16)', f32: 'f32' } as const;

/** The run's head (`config.head`; missing means o2m). */
export const headSetting = (config: TrainConfig): Setting => ({ value: HEAD_LABEL[config.head ?? 'o2m'] });
/** The run's precision (`config.amp`; missing means f32). */
export const precisionSetting = (config: TrainConfig): Setting => ({ value: config.amp ? PRECISION_LABEL.amp : PRECISION_LABEL.f32 });

/**
 * The epochs at the end of the run that train without mosaic, as the trainer runs them: the server's figure where it
 * sends one (`resolved.close_mosaic`, the one place it is read), else the trainer's cap of the request,
 * `min(close_mosaic, floor(epochs / 2))` (folo-data `close_mosaic_epochs`): a 15-epoch run asked for 10 closes for 7.
 */
export function effectiveCloseMosaic(run: { config: TrainConfig; resolved?: ResolvedTrain | null }): number {
  const served = run.resolved?.close_mosaic;
  if (typeof served === 'number' && Number.isFinite(served)) return served;
  return closeMosaicEpochs(run.config.augment.close_mosaic, run.config.epochs);
}

/** "last 7 epochs (requested 10)" where the trainer capped the request, "last 10 epochs", "never". */
export function closeMosaicSetting(run: { config: TrainConfig; resolved?: ResolvedTrain | null }): Setting {
  const n = effectiveCloseMosaic(run);
  const asked = run.config.augment.close_mosaic;
  const value = n > 0 ? `last ${n} ${n === 1 ? 'epoch' : 'epochs'}` : 'never';
  return n === asked ? { value } : { value, note: `(requested ${asked})` };
}
