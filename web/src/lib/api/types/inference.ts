// Inference: predictions and validation (docs/API.md, Inference).

import type { InferenceMeta, ServerTilePayload, ServerTtaPayload } from '$lib/inference/options';
import type { ScoredBox, Split } from './datasets';
import type { Epoch } from './runs';

// ---------- Inference ----------

/** Prediction boxes always carry the class name (and optional ByteTrack `track_id` in live tracking mode). */
export interface PredictBox extends ScoredBox {
  track_id?: number;
}

export interface Timing {
  preprocess_ms: number;
  inference_ms: number;
  postprocess_ms: number;
  total_ms: number;
}

export interface Prediction {
  model_id: string;
  image: { width: number; height: number };
  boxes: PredictBox[];
  timing: Timing;
  device: string;
  /** Set when the prediction ran in this browser (folo-wasm, docs/WASM.md) instead of on the server. */
  backend?: 'webgpu' | 'cpu';
  /** The model's first prediction since the server loaded it: `timing` includes its warm-up (absent: false). */
  cold?: boolean;
  /** Client-attached enhancement metadata (SAHI or TTA) when non-standard inference produced this result. */
  inference?: InferenceMeta;
}

export interface PredictOptions {
  model_id: string;
  conf?: number;
  iou?: number;
  max_det?: number;
  /** SAHI tiled inference config (only sent when the server advertises `predict_options`). */
  tile?: ServerTilePayload;
  /** Test-Time Augmentation config (only sent when the server advertises `predict_options`). */
  tta?: ServerTtaPayload;
}

export type PredictInput =
  | { kind: 'file'; file: Blob }
  | { kind: 'dataset'; dataset_id: string; image_id: string };

/** POST /api/val body. Only model_id and dataset_id are required. */
export interface ValRequest {
  model_id: string;
  dataset_id: string;
  split?: Split;
  imgsz?: number | null;
  conf?: number;
  iou?: number;
}

/** Global F1-optimal confidence operating point from `folo_vision::OperatingPoint`. */
export interface ThresholdOperatingPoint {
  best_conf: number;
  best_f1: number;
  precision: number;
  recall: number;
}

/** Per-class F1-optimal confidence threshold from `folo_vision::ClassThreshold`. */
export interface ClassThreshold {
  class: number;
  name?: string;
  best_conf: number;
  best_f1: number;
  precision: number;
  recall: number;
  truths: number;
  preds: number;
}

/** Calibrated global and per-class F1 confidence thresholds from `POST /api/val` (`folo_vision::ThresholdCalibration`). */
export interface ThresholdCalibration {
  global: ThresholdOperatingPoint;
  macro_f1: number;
  per_class: ClassThreshold[];
}

/** POST /api/val response. */
export interface ValReport {
  model_id: string;
  dataset_id: string;
  /** Missing from older servers. */
  dataset_name?: string | null;
  /** As requested: `POST /api/val` never falls back (an empty split is `invalid_dataset`). */
  split: Split;
  imgsz: number;
  images: number;
  val: Epoch['val'];
  timing: { total_s: number; per_image_ms: number };
  /** Present when the server advertises `threshold_calibration`. */
  thresholds?: ThresholdCalibration;
}
