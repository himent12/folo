// A run's settings as it trains with them (docs/API.md, `Run.resolved`): `config` is what was asked for ("Optimizer
// auto", "Initial LR 0.01"), `resolved` what the trainer made of it (AdamW at 0.001429). With `resolved`, the resolved
// value is the one shown, the request a note beside it when it differs; without it (queued, older servers) the
// request as today.

import { closeMosaicEpochs } from '$lib/api/progress';
import type { ResolvedTrain, TrainConfig } from '$lib/api/types';
import { pct } from '$lib/format';

/** One setting: its value, whether that value is an identifier (mono, `ui/Ident`), and a sans note after it. */
export interface Setting {
  value: string;
  ident?: boolean;
  note?: string;
}

const OPTIMIZER: Record<ResolvedTrain['optimizer'], string> = { sgd: 'SGD', adamw: 'AdamW' };

/** "AdamW (auto)", "SGD" (asked for by name); unresolved, the request: "auto" (prose) or a named one. */
export function optimizerSetting(config: TrainConfig, resolved: ResolvedTrain | null | undefined): Setting {
  if (!resolved) return { value: config.optimizer, ident: config.optimizer !== 'auto' };
  return { value: OPTIMIZER[resolved.optimizer], ident: true, note: config.optimizer === 'auto' ? '(auto)' : undefined };
}

/** "0.001429 (requested 0.01)" when the trainer's LR differs from the request, else the one value. */
export function lrSetting(config: TrainConfig, resolved: ResolvedTrain | null | undefined): Setting {
  if (!resolved || resolved.lr0 === config.lr0) return { value: String(resolved?.lr0 ?? config.lr0) };
  return { value: String(resolved.lr0), note: `(requested ${config.lr0})` };
}

/** Gradients of `accumulate` batches per optimizer step: "16 × 4 = 64" (the batch alone when it does not accumulate). */
export function batchSetting(config: TrainConfig, resolved: ResolvedTrain | null | undefined): Setting {
  if (!resolved || resolved.accumulate <= 1) return { value: String(resolved?.batch ?? config.batch) };
  return { value: `${resolved.batch} × ${resolved.accumulate} = ${resolved.batch * resolved.accumulate}` };
}

/** The warmup in optimizer steps, and what that is in epochs: "100 steps (7.7 epochs)"; "None" without one. */
export function warmupSetting(resolved: ResolvedTrain): Setting {
  if (resolved.warmup_iters <= 0) return { value: 'None' };
  const epochs = resolved.warmup_iters / Math.max(1, resolved.optimizer_steps_per_epoch);
  const e = Number(epochs.toFixed(epochs < 10 ? 1 : 0));
  return { value: `${resolved.warmup_iters} steps`, note: `(${e} ${e === 1 ? 'epoch' : 'epochs'})` };
}

/** "Linear to 1%": the decay over the run, to `lrf` of the initial LR at the last step. */
export function decaySetting(resolved: ResolvedTrain): Setting {
  return { value: `${resolved.cos_lr ? 'Cosine' : 'Linear'} to ${pct(resolved.lrf, resolved.lrf < 0.01 ? 1 : 0)}` };
}

/** AdamW's betas ("0.9 · 0.999"), or SGD's momentum, after warmup. */
export function momentumSetting(resolved: ResolvedTrain): [label: string, Setting] {
  return resolved.betas ? ['Betas', { value: resolved.betas.join(' · ') }] : ['Momentum', { value: String(resolved.momentum) }];
}

/** The resolved rows a run's Config adds after its request: decay, warmup, momentum and the scaled weight decay. */
export function resolvedRows(resolved: ResolvedTrain): [label: string, Setting][] {
  return [
    ['LR decay', decaySetting(resolved)],
    ['Warmup', warmupSetting(resolved)],
    momentumSetting(resolved),
    ['Weight decay', { value: String(resolved.weight_decay) }],
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
