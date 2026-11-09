// Types for the FOLO Studio HTTP API. Mirrors docs/API.md exactly; change both together.

/** RFC 3339 timestamp. */
export type Time = string;

// ---------- Errors ----------

export interface ApiErrorBody {
  /** `hint` is always present on the server, `null` when there is nothing to add. */
  error: { code: string; message: string; hint: string | null };
}

// ---------- System ----------

export interface Health {
  status: 'ok';
  version: string;
  /** API generation, `v1`. */
  api: string;
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
  /** The real server's data dir (`/` separators); absent in mock mode. */
  data_dir?: string;
  /** Optional API features the server implements, e.g. `export_onnx_switches`; empty or absent from older ones. */
  features?: string[];
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
  /** Relative box size, `sqrt(box area / image area)`; counts[i] covers [edges[i], edges[i+1]), the last bin is open. */
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

/** A prediction: a box with its score and class name. */
export interface ScoredBox extends Box {
  name: string;
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

/** `GET /api/datasets/{id}/issues` (docs/API.md): the `folo dataset check` findings. */
export type IssueKind =
  | 'broken_label'
  | 'class_overflow'
  | 'out_of_range_box'
  | 'clipped_box'
  | 'zero_size_box'
  | 'duplicate_label'
  | 'corrupt_image'
  | 'missing_image'
  | 'duplicate_image'
  | 'empty_image'
  | 'other';

export type IssueSeverity = 'error' | 'warning' | 'info';

export interface DatasetIssue {
  kind: IssueKind;
  severity: IssueSeverity;
  /** Relative to the dataset root (absolute outside it). */
  file: string;
  /** 1-based label line, or null. */
  line: number | null;
  message: string;
  /** What a cleanup would do, e.g. `delete line 3` (absent for infos). */
  fix?: string;
}

export interface DatasetIssuesQuery {
  kind?: IssueKind;
  severity?: IssueSeverity;
  offset?: number;
  limit?: number;
}

/** Every issue of the dataset by severity and kind, whatever the filter. */
export interface IssueCounts {
  error: number;
  warning: number;
  info: number;
  by_kind: Partial<Record<IssueKind, number>>;
}

export interface DatasetIssues {
  total: number;
  issues: DatasetIssue[];
  /** Absent from older servers. */
  counts?: IssueCounts;
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
}

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
  /** Mixed-precision training (docs/MODEL.md §8.8); missing from older servers (= false). */
  amp?: boolean;
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
  /** The run's length (Run.steps_per_epoch / Run.epochs); absent from older servers. */
  steps_per_epoch?: number;
  epochs?: number;
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

/** GET /api/runs/{id}/logs?limit=N: the most recent lines, oldest first. */
export interface RunLogs {
  lines: LogLine[];
}

/** SSE events on GET /api/runs/{id}/events, keyed by the SSE `event:` name. */
export interface RunEventMap {
  /** steps_per_epoch is 0 until the run is built; both are absent from older servers. */
  status: { status: RunStatus; error: string | null; steps_per_epoch?: number; epochs?: number };
  step: Step;
  epoch: Epoch;
  log: LogLine;
}

// ---------- Inference ----------

/** Prediction boxes always carry the class name. */
export type PredictBox = ScoredBox;

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

/** POST /api/val body. Only model_id and dataset_id are required. */
export interface ValRequest {
  model_id: string;
  dataset_id: string;
  split?: Split;
  imgsz?: number | null;
  conf?: number;
  iou?: number;
}

/** POST /api/val response. */
export interface ValReport {
  model_id: string;
  dataset_id: string;
  split: Split;
  imgsz: number;
  images: number;
  val: Epoch['val'];
  timing: { total_s: number; per_image_ms: number };
}
