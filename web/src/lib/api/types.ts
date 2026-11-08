// Types for the FOLO Studio HTTP API. Mirrors docs/API.md exactly; change both together.

/** RFC 3339 timestamp. */
export type Time = string;

// ---------- Errors ----------

export interface ApiErrorBody {
  error: { code: string; message: string; hint?: string };
}

// ---------- System ----------

export interface Health {
  status: 'ok';
  version: string;
  mock: boolean;
}

export type GpuBackend = 'vulkan' | 'dx12' | 'metal';

export interface Gpu {
  ordinal: number;
  name: string;
  backend: GpuBackend;
  memory_bytes: number;
  /** 0..1, or null when the driver does not report it. */
  util: number | null;
}

export interface SystemInfo {
  cpu: { name: string; threads: number; util: number };
  memory: { used_bytes: number; total_bytes: number };
  gpus: Gpu[];
}

// ---------- Datasets ----------

export type DatasetFormat = 'yolo' | 'coco';
export type Split = 'train' | 'val' | 'test';

export interface Histogram {
  /** Bin edges, length counts.length + 1. */
  edges: number[];
  counts: number[];
}

export interface DatasetStats {
  boxes: number;
  empty_images: number;
  broken_labels: number;
  /** Indexed by class id. */
  class_counts: number[];
  /** Box area as a fraction of the image area. */
  box_size_hist: Histogram;
  boxes_per_image_hist: Histogram;
}

export interface Dataset {
  id: string;
  name: string;
  path: string;
  format: DatasetFormat;
  classes: string[];
  created: Time;
  splits: Record<Split, number>;
  /** Present on GET /api/datasets/{id}. */
  stats?: DatasetStats;
}

export interface CreateDataset {
  name: string;
  path: string;
  format: 'auto' | DatasetFormat;
}

export interface CreateSyntheticDataset {
  name: string;
  images: number;
  classes: string[];
  imgsz: number;
  seed: number;
}

/** Pixel xyxy box in the original image. */
export interface Box {
  class: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface ScoredBox extends Box {
  score: number;
}

export interface DatasetImage {
  id: string;
  file: string;
  split: Split;
  width: number;
  height: number;
  boxes: Box[];
}

export interface DatasetImagesQuery {
  split?: Split;
  offset?: number;
  limit?: number;
  class?: number;
}

export interface DatasetImages {
  total: number;
  images: DatasetImage[];
}

// ---------- Models ----------

export type Arch = 'folo-n' | 'folo-s' | 'folo-m' | 'folo-l' | 'folo-x';

export interface ModelMetrics {
  map50: number;
  map50_95: number;
  precision: number;
  recall: number;
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
}

export type ExportFormat = 'folo-f16' | 'onnx';

export interface ExportResult {
  path: string;
  bytes: number;
}

// ---------- Training runs ----------

export type RunStatus = 'queued' | 'running' | 'stopping' | 'stopped' | 'completed' | 'failed';
export type Device = 'auto' | 'cpu' | `gpu:${number}`;
export type Optimizer = 'auto' | 'sgd' | 'adamw';

export interface Augment {
  mosaic: number;
  mixup: number;
  /** HSV jitter gains [h, s, v]. */
  hsv: [number, number, number];
  fliplr: number;
  scale: number;
  translate: number;
  /** Disable mosaic for the last N epochs. */
  close_mosaic: number;
}

export interface TrainConfig {
  name: string;
  dataset_id: string;
  arch: Arch;
  epochs: number;
  batch: number;
  imgsz: number;
  device: Device;
  optimizer: Optimizer;
  lr0: number;
  weights: string | null;
  augment: Augment;
  seed: number;
}

export interface Run {
  id: string;
  name: string;
  status: RunStatus;
  config: TrainConfig;
  created: Time;
  started: Time | null;
  finished: Time | null;
  epoch: number;
  epochs: number;
  step: number;
  steps_per_epoch: number;
  eta_s: number | null;
  best: { epoch: number; map50_95: number } | null;
  device: string;
  model_id: string | null;
  error: string | null;
}

export interface Loss {
  box: number;
  cls: number;
  dfl: number;
  total: number;
}

export interface Step {
  epoch: number;
  step: number;
  lr: number;
  loss: Loss;
  imgs_per_s: number;
  eta_s: number;
}

export interface ClassMetrics {
  class: number;
  map50: number;
  map50_95: number;
  precision: number;
  recall: number;
}

export interface Epoch {
  epoch: number;
  train_loss: Loss;
  val: ModelMetrics & { per_class: ClassMetrics[] };
  duration_s: number;
}

export interface RunMetrics {
  steps: Step[];
  epochs: Epoch[];
}

export interface Sample {
  image_id: string;
  dataset_id: string;
  width: number;
  height: number;
  truth: Box[];
  pred: ScoredBox[];
}

export interface RunSamples {
  epoch: number;
  samples: Sample[];
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogLine {
  t: Time;
  level: LogLevel;
  message: string;
}

/** SSE events on GET /api/runs/{id}/events, keyed by the SSE `event:` name. */
export interface RunEventMap {
  status: { status: RunStatus; error: string | null };
  step: Step;
  epoch: Epoch;
  log: LogLine;
}

// ---------- Inference ----------

export interface PredictBox extends ScoredBox {
  name: string;
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
}

export interface PredictOptions {
  model_id: string;
  conf?: number;
  iou?: number;
  max_det?: number;
}

export type PredictInput =
  | { kind: 'file'; file: Blob }
  | { kind: 'dataset'; dataset_id: string; image_id: string };
