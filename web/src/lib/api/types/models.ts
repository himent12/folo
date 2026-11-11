// Models: weights, their evaluation and exports (docs/API.md, Models).

import type { Split } from './datasets';
import type { ClassMetrics } from './runs';
import type { Time } from './time';

// ---------- Models ----------

export type Arch = 'folo-n' | 'folo-s' | 'folo-m' | 'folo-l' | 'folo-x';

export interface ModelMetrics {
  map50: number;
  map50_95: number;
  /** Precision and recall at `best_f1_conf`. */
  precision: number;
  recall: number;
  /**
   * The confidence threshold precision and recall are taken at (best class-mean F1 at IoU 0.5, 0.001 grid);
   * `null` when unknown, missing from older servers.
   */
  best_f1_conf?: number | null;
}

export interface Model {
  id: string;
  name: string;
  arch: Arch;
  imgsz: number;
  classes: string[];
  params: number;
  flops_g: number;
  bytes: number;
  created: Time;
  run_id: string | null;
  metrics: ModelMetrics | null;
  /** Detection head, from the file's `folo.config` (docs/MODEL.md §6.6); missing (older servers) = `o2m`. */
  head?: 'o2m' | 'o2o';
  /** Training settings from the file's `folo.train`; `null` for exports and imported weights (missing: older server). */
  trained_with?: TrainedWith | null;
  /**
   * Stored weight dtype, from the file's `folo.dtype`; missing = unknown (older server, a file without the key, or
   * a dtype the server has no name for). Treat any other string as unknown too.
   */
  precision?: 'f32' | 'f16';
  /** Conv+BN fused (`folo.fused`: an inference-only export); missing = unknown. */
  fused?: boolean;
  /**
   * The model's file on the server (requested of the API: `folo train --weights` takes it); missing = unknown. Read
   * only through `$lib/trainCli` `modelPath`.
   */
  path?: string;
  /** `run`, a `file` under the models folder, or `published` weights from `folo weights pull` (id = the name). */
  source?: 'run' | 'file' | 'published';
  /** What published weights are; absent for other models. */
  about?: string;
}

/** `Model.trained_with`. */
export interface TrainedWith {
  amp: boolean;
  imgsz: number;
  epochs: number;
}

export type ExportFormat = 'folo-f16' | 'onnx';

/**
 * `folo export`'s ONNX switches (docs/API.md, export; docs/CLI.md, ONNX export), ONNX only. Servers that implement
 * them list `export_onnx_switches` in `SystemInfo.features`.
 */
export interface OnnxSwitches {
  /** Decode boxes and class scores in the graph instead of the raw head outputs. */
  decode?: boolean;
  /** Symbolic batch, height and width (any batch, sides in multiples of 32). */
  dynamic?: boolean;
  /** f16 weights, input, activations and outputs. */
  half?: boolean;
}

/** Precision–recall curve of one class at IoU 0.5, sampled at 101 recall points (0.00..1.00). */
export interface PrCurve {
  class: number;
  recall: number[];
  /** COCO-interpolated envelope: never rises with recall; its mean is `ap50`. */
  precision: number[];
  ap50: number;
}

/** GET /api/models/{id}/eval: validation results of the model on its dataset's val split. */
export interface ModelEval {
  pr: PrCurve[];
  /** (classes + 1)² counts: rows true class, columns predicted, last row/column background; [n][n] is 0. */
  confusion: number[][];
  /** As in Epoch.val.per_class; map50 equals pr[class].ap50. */
  per_class: ClassMetrics[];
  images: number;
  /** The confidence `per_class` precision / recall are taken at; `null` when unknown, missing from older servers. */
  best_f1_conf?: number | null;
  // Provenance (docs/API.md `/eval`); each is `null` when unknown and missing from older servers.
  /** The evaluated dataset's id and name. */
  dataset_id?: string | null;
  dataset_name?: string | null;
  /** The split actually evaluated: `val`, or `train` when the dataset has no val images (`split_fallback` set). */
  split?: Split | null;
  /** Set when `split` is not `val`: say so wherever the numbers are labelled "validation". */
  split_fallback?: SplitFallback | null;
  /** The confusion matrix counts predictions with `score > confusion_conf`, matched at IoU `> confusion_iou`. */
  confusion_conf?: number | null;
  confusion_iou?: number | null;
}

/** `ModelEval.split_fallback`: the split that was wanted, why it was not used, one sentence for people. */
export interface SplitFallback {
  requested: Split;
  /** `empty_split` (it has no images); treat an unknown code the same way. */
  reason: string;
  message: string;
}

export interface ExportResult {
  path: string;
  bytes: number;
}
