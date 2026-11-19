// What an evaluation was run on, in one wording for every screen that shows evaluation results (docs/API.md `/eval`,
// Provenance and Split fallback): the dataset's name, the split actually evaluated, and, when the dataset has no
// validation images and the server evaluated its training split, one plain sentence saying so. Those numbers are never
// called "validation". The confusion matrix's own thresholds are worded here too, apart from precision and recall's.

import type { Dataset, ModelEval, Split, SplitFallback } from '$lib/api/types';
import { conf, num } from '$lib/format';

export interface EvalSource {
  datasetId: string | null;
  datasetName: string | null;
  /** The split actually evaluated; null when the server does not say (older servers: unknown, not `val`). */
  split: Split | null;
  fallback: SplitFallback | null;
}

/** The confusion matrix's thresholds: a prediction counts above `conf`, matched to a truth above IoU `iou`. */
export interface MatrixAt {
  conf: number;
  iou: number;
}

export function evalSource(e: ModelEval): EvalSource {
  const fallback = e.split_fallback ?? null;
  return {
    datasetId: e.dataset_id ?? null,
    datasetName: e.dataset_name ?? null,
    // A fallback is a training-split evaluation (the only one there is) even when `split` is missing.
    split: e.split ?? (fallback ? 'train' : null),
    fallback,
  };
}

/**
 * What a training run's per-epoch scores are on (docs/API.md, Split fallback: training falls back as `/eval` does): the
 * validation split, or the training split when its dataset has no validation images. Null until the dataset is known.
 */
export function runEvalSource(d: Pick<Dataset, 'id' | 'name' | 'splits'> | null | undefined): EvalSource | null {
  if (!d) return null;
  const empty = d.splits.val === 0;
  return {
    datasetId: d.id,
    datasetName: d.name,
    split: empty ? 'train' : 'val',
    fallback: empty ? { requested: 'val', reason: 'empty_split', message: `${d.name} has no validation images` } : null,
  };
}

/** The trainer's sample images an epoch: at most the first 16 of the split it validates on (docs/API.md `/samples`). */
const SAMPLE_LIMIT = 16;

/**
 * What a run that has not trained yet will show in Samples: "Predictions on 2 validation images after every epoch",
 * the smaller of the trainer's limit and the split's size (the training split's when there are no validation images).
 * No count while the dataset is not known (removed, not loaded): never a number the page cannot know.
 */
export function samplesAhead(s: EvalSource | null, d: Pick<Dataset, 'splits'> | null | undefined): string {
  const kind = scoresTitle(s).toLowerCase();
  const n = d ? Math.min(SAMPLE_LIMIT, d.splits.val || d.splits.train) : 0;
  return n ? `Predictions on ${num(n)} ${kind} ${n === 1 ? 'image' : 'images'} after every epoch` : `Predictions on ${kind} images after every epoch`;
}

/**
 * The scores' name where a title names them: "Validation" ("Validation mAP50-95", "Validation samples") on a known
 * validation split, else the split they are really on ("Training-split mAP50-95"). "Evaluation" while the split is
 * unknown (a run whose dataset is not registered may have trained without validation images): never "Validation"
 * without knowing.
 */
export function scoresTitle(s: EvalSource | null | undefined): string {
  if (!s || s.split === null) return 'Evaluation';
  if (s.split === 'val') return 'Validation';
  const w = splitWord(s.split);
  return `${w.charAt(0).toUpperCase()}${w.slice(1)}-split`;
}

export function matrixAt(e: ModelEval): MatrixAt | null {
  const c = e.confusion_conf;
  const i = e.confusion_iou;
  return c != null && i != null && Number.isFinite(c) && Number.isFinite(i) ? { conf: c, iou: i } : null;
}

/** A split as people say it: "validation", "training", "test". */
export function splitWord(s: Split): string {
  return s === 'val' ? 'validation' : s === 'train' ? 'training' : 'test';
}

/**
 * Why training-split numbers are not validation numbers, in one wording (the Model page's notice and Dataset
 * Health's "No validation split" finding).
 */
export const TRAINED_ON = 'on images the model trained on, not new ones';

/** Whether the numbers are not on the validation split (a fallback, or a split the server names as another). */
export function offValidation(s: EvalSource): boolean {
  return !!s.fallback || (s.split !== null && s.split !== 'val');
}

export interface Notice {
  /** "Evaluated on the training split" */
  title: string;
  /** "fasteners-mini has no validation images, so these scores are on images the model trained on, not new ones." */
  why: string;
  /** Both, as one sentence. */
  text: string;
}

/**
 * What a view says, once, when the evaluation is not on the validation split: "Evaluated on the training split:
 * fasteners-mini has no validation images, so these scores are on images the model trained on, not new ones." Null on
 * the validation split, and when the server does not say which split.
 */
export function fallbackNotice(s: EvalSource): Notice | null {
  if (!offValidation(s)) return null;
  const used = s.split ?? 'train';
  const title = `Evaluated on the ${splitWord(used)} split`;
  const f = s.fallback;
  // Lower case after the title's colon; the dataset's name keeps its own case.
  const named = !!f && f.reason === 'empty_split' && !!s.datasetName;
  let cause: string;
  if (!f) cause = 'not the validation split';
  else if (f.reason === 'empty_split') cause = `${s.datasetName ?? 'this dataset'} has no ${splitWord(f.requested)} images`;
  // A reason this client has no words for is still a fallback (docs/API.md): the server's own sentence says why.
  else cause = `not the ${splitWord(f.requested)} split (${f.message})`;
  const rest = `${cause}${used === 'train' ? `, so these scores are ${TRAINED_ON}` : ''}.`;
  const why = named ? rest : rest.charAt(0).toUpperCase() + rest.slice(1);
  return { title, why, text: `${title}: ${rest}` };
}

/** A panel's short qualifier: "fasteners · validation split"; null when the server names neither. */
export function sourceShort(s: EvalSource): string | null {
  const parts = [s.datasetName, s.split ? `${splitWord(s.split)} split` : null].filter(Boolean);
  return parts.length ? parts.join(' · ') : null;
}

/** The images counted, in a sentence: "360 validation images of fasteners", "500 images" when the server does not say. */
export function imagesOf(s: EvalSource, images: number): string {
  const what = s.split ? `${splitWord(s.split)} ${images === 1 ? 'image' : 'images'}` : images === 1 ? 'image' : 'images';
  return `${num(images)} ${what}${s.datasetName ? ` of ${s.datasetName}` : ''}`;
}

/**
 * What was evaluated, as the start of a sentence: "The validation split of fasteners (360 images)"; with no split
 * named, "fasteners, split not reported (500 images)".
 */
export function evaluatedOn(s: EvalSource, images: number | null): string {
  const count = images === null ? '' : ` (${num(images)} ${images === 1 ? 'image' : 'images'})`;
  if (s.split) return `The ${splitWord(s.split)} split of ${s.datasetName ?? "the model's dataset"}${count}`;
  return `${s.datasetName ?? "The model's dataset"}, split not reported${count}`;
}

/** The matrix's thresholds as its labels say them: "at conf 0.25, IoU 0.45"; null when the server does not say. */
export function matrixAtText(m: MatrixAt | null): string | null {
  return m ? `at conf ${conf(m.conf)}, IoU ${conf(m.iou)}` : null;
}

/** The headline metrics' accessible name: "Validation metrics", or the split they are on. */
export function metricsLabel(s: EvalSource): string {
  if (!s.split) return 'Evaluation metrics';
  return s.split === 'val' ? 'Validation metrics' : `Metrics on the ${splitWord(s.split)} split`;
}

/**
 * What the confusion matrix counted, in one sentence: "Counted at conf 0.25, IoU 0.45 on 360 validation images of
 * fasteners." Without the thresholds it says the server does not report them, never borrowing precision and recall's.
 */
export function matrixCounted(s: EvalSource, images: number, m: MatrixAt | null): string {
  const at = matrixAtText(m);
  return at ? `Counted ${at} on ${imagesOf(s, images)}.` : `Counted on ${imagesOf(s, images)}, at thresholds this server does not report.`;
}

/**
 * How the matrix's thresholds differ from precision and recall's (`prAt`: "at conf 0.31 (best F1)"), for the
 * matrix's "How the shares are computed".
 */
export function matrixVsPr(m: MatrixAt | null, prAt: string): string {
  const own = m
    ? `A predicted box counts when its confidence is above ${conf(m.conf)}, and finds a true box when they overlap by IoU above ${conf(m.iou)}: the matrix's own thresholds.`
    : "The matrix counts at its own confidence and IoU thresholds, which this server does not report.";
  return `${own} Precision and recall are read ${prAt}, IoU 0.50, so the diagonal can differ from the per-class recall.`;
}
