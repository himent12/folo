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
  /** What `device: auto` runs can use (f16, tensor cores, amp); absent from older servers. */
  capabilities?: Capabilities;
  /** FOLO's GPU buffer pool, once the server has used the GPU; absent otherwise and from older servers. */
  gpu_pool?: GpuPool;
}

/** GPU memory held by FOLO's buffer pool (released after a run and when inference is idle). */
export interface GpuPool {
  device: string;
  live_bytes: number;
  cached_bytes: number;
  peak_live_bytes: number;
}

/** `/api/system`'s `capabilities` (docs/API.md). A `reason` is null exactly when the feature is usable. */
export interface Capabilities {
  /** What `device: auto` resolves to: `gpu:0` or `cpu`. */
  device: string;
  /** f16 inference runs in half precision (else it falls back to f32). */
  shader_f16: boolean;
  tensor_cores: { available: boolean; enabled: boolean; reason: string | null };
  /** Mixed-precision training (`TrainConfig.amp`). */
  amp: { available: boolean; reason: string | null };
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
  /** Detection head (docs/MODEL.md §6.6): `o2o` adds the NMS-free one-to-one branch; missing = `o2m`. */
  head?: 'o2m' | 'o2o';
  /** Train on a cloud connection (servers with the `cloud` feature); left out (missing) means this machine. */
  compute?: Compute | null;
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
  /** Stable error code when failed (e.g. `not_supported`); missing otherwise and from older servers. */
  error_code?: string;
  /** What the trainer resolved `config` to (auto optimizer, effective LR, ...); `null` until built, missing: older server. */
  resolved?: ResolvedTrain | null;
  /** 1-based place in the training queue while waiting there (1 starts next); `null` otherwise, missing: older server. */
  queue_position?: number | null;
  /** Where a cloud run (`config.compute`) executes and how far it is; missing for every other run. */
  cloud?: RunCloud;
}

/** `Run.resolved` (docs/API.md): the settings a run actually trains with. */
export interface ResolvedTrain {
  optimizer: 'sgd' | 'adamw';
  lr0: number;
  lrf: number;
  cos_lr: boolean;
  /** SGD momentum or AdamW β1 after warmup. */
  momentum: number;
  /** AdamW `[β1, β2]`; `null` for SGD. */
  betas: [number, number] | null;
  /** After the batch scaling `weight_decay · batch · accumulate / nbs`. */
  weight_decay: number;
  batch: number;
  accumulate: number;
  nbs: number;
  optimizer_steps_per_epoch: number;
  warmup_iters: number;
  warmup_epochs: number;
  warmup_momentum: number;
  warmup_bias_lr: number;
  /**
   * Epochs at the end of the run without mosaic, after the trainer's cap (requested of the API); missing = computed
   * from the request. Read only through `effectiveCloseMosaic` (routes/run/resolved).
   */
  close_mosaic?: number;
}

/** GET /api/runs/{id}/cli (servers list `run_cli`): the run's `folo train` command on the server's machine. */
export interface CliCommand {
  cli: string;
  argv: string[];
}

/** POST /api/runs/{id}/queue: exactly one of `action` and `position` (1-based). Servers list `queue_control`. */
export type QueueRequest = { action: 'top' | 'up' | 'down' | 'bottom' | 'start' } | { position: number };

/** SSE `queue`: a waiting run's place in the queue and how many runs wait, it included. */
export interface QueueEvent {
  position: number;
  waiting: number;
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
  status: { status: RunStatus; error: string | null; steps_per_epoch?: number; epochs?: number; error_code?: string };
  step: Step;
  epoch: Epoch;
  log: LogLine;
  /** Servers with `queue_control` only. */
  queue: QueueEvent;
  /** A cloud run's `Run.cloud` after it changed (state, not a delta); servers with `cloud` only. */
  cloud: RunCloud;
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
  /** Missing from older servers. */
  dataset_name?: string | null;
  /** As requested: `POST /api/val` never falls back (an empty split is `invalid_dataset`). */
  split: Split;
  imgsz: number;
  images: number;
  val: Epoch['val'];
  timing: { total_s: number; per_image_ms: number };
}

// ---- Throughput and estimates (docs/API.md); servers that have them list `throughput`, `run_estimate` and
// `benchmark` in `SystemInfo.features`.

/** Where a number comes from: measured on this server, or docs/BENCHMARKS.md's reference (see `note`). */
export type ThroughputSource = 'measured' | 'reference';
export type Confidence = 'high' | 'medium' | 'low';

/** Median of the recent samples with their 10th / 90th percentiles (min / max below 10 samples). */
export interface Rate {
  imgs_per_s: number;
  low: number;
  high: number;
  samples: number;
}

export interface TrainThroughput {
  /** The device of this server the row is for (`gpu:0`, `cpu`). */
  device: string;
  /** The adapter or CPU the number was measured on (a reference's: the dev machine's). */
  measured_on: string;
  arch: Arch;
  imgsz: number;
  batch: number;
  amp: boolean;
  /** Headline training rate: the mosaic epochs' median, else the close-mosaic epochs'. */
  imgs_per_s: number;
  mosaic: Rate | null;
  close_mosaic: Rate | null;
  val: Rate | null;
  startup_s: number | null;
  peak_vram_bytes: number | null;
  source: ThroughputSource;
  samples: number;
  probes: number;
  updated: Time | null;
  note: string | null;
}

export interface InferThroughput {
  device: string;
  measured_on: string;
  arch: Arch;
  imgsz: number;
  /** Always 1. */
  batch: number;
  precision: 'f32' | 'f16';
  tensor_cores: boolean;
  latency_ms: number;
  latency_ms_low: number;
  latency_ms_high: number;
  inference_ms: number | null;
  imgs_per_s: number;
  source: ThroughputSource;
  samples: number;
  updated: Time | null;
  note: string | null;
}

/** GET /api/system/throughput. */
export interface ThroughputTable {
  device: string;
  device_name: string;
  training: TrainThroughput[];
  inference: InferThroughput[];
}

export interface EstimatePhase {
  phase: 'mosaic' | 'close_mosaic';
  epochs: number;
  train_imgs_per_s: number;
  train_s: number;
  val_s: number;
  epoch_s: number;
}

export interface EstimateBasis {
  source: ThroughputSource;
  arch: Arch;
  imgsz: number;
  batch: number;
  amp: boolean;
  measured_on: string;
  samples: number;
  updated: Time | null;
  exact: boolean;
  note: string;
}

/** POST /api/runs/estimate (body: TrainConfig). */
export interface RunEstimate {
  device: string;
  device_name: string;
  train_images: number;
  val_images: number;
  batches_per_epoch: number;
  epochs: number;
  phases: EstimatePhase[];
  val_imgs_per_s: number;
  startup_s: number;
  est_epoch_s: number;
  est_total_s: number;
  est_total_s_low: number;
  est_total_s_high: number;
  source: ThroughputSource;
  confidence: Confidence;
  basis: EstimateBasis;
  val_source: ThroughputSource;
  est_peak_vram_bytes: number | null;
  vram_source: ThroughputSource | null;
  /** Close-mosaic epochs after the cap at half the run (0 from older servers). */
  close_mosaic?: number;
  /** The `folo train` command for these settings, built by the server (requested of the API); missing = none. */
  cli?: string | null;
  /** What a cloud run would cost (a config with `compute` on an offer with a price); missing otherwise. */
  cost?: CostEstimate;
}

/** POST /api/system/benchmark body; every field is optional. */
export interface BenchmarkRequest {
  arch?: Arch;
  imgsz?: number;
  batch?: number;
  amp?: boolean;
  budget_s?: number;
}

export interface BenchmarkReport {
  device: string;
  device_name: string;
  train_imgs_per_s: number;
  train_batches: number;
  latency_ms: number;
  predictions: number;
  elapsed_s: number;
  training: TrainThroughput;
  inference: InferThroughput;
}

// ---------- Cloud compute (docs/API.md, Cloud compute; docs/CLOUD.md); servers list `cloud` in `features` ----------

/** One field of a connection form (`CloudProvider.fields`, `ConnectCloud.values`). */
export interface CloudField {
  /** Key in `ConnectCloud.values`, e.g. `api_key`. */
  key: string;
  label: string;
  /** A credential: entered masked, never returned (connections show only a hint of it). */
  secret: boolean;
  required: boolean;
  placeholder: string | null;
  help: string | null;
}

/** `GET /api/cloud/providers`: a kind of compute, with the fields a connection to it takes. */
export interface CloudProvider {
  /** `runpod`, `remote`. */
  id: string;
  name: string;
  /** One sentence on what it is. */
  about: string;
  /** Where its credentials are made, if anywhere. */
  docs_url: string | null;
  /** In form order. */
  fields: CloudField[];
  /** It launches an instance per run and terminates it afterwards (`false`: a machine that already runs). */
  provisions: boolean;
  /** Its instances terminate themselves after a run's deadline, also while this server is off. */
  self_terminates: boolean;
}

/** `POST /api/cloud/connections` body. `values` holds credentials: never log, store or show them. */
export interface ConnectCloud {
  provider: string;
  /** Empty means the provider's name (with ` 2`, ` 3`, ... when taken). */
  name?: string;
  /** A value per `CloudField.key`; empty values count as missing. */
  values: Record<string, string>;
}

export type CloudConnectionStatus = 'ok' | 'error';

/** A provider with credentials (`GET /api/cloud/connections`). */
export interface CloudConnection {
  /** e.g. `cc_1f2e`. */
  id: string;
  provider: string;
  name: string;
  created: Time;
  /** Result of the last check. */
  status: CloudConnectionStatus;
  /** Who the credentials belong to (an email, a host); `null` when unknown. */
  account: string | null;
  /** In US dollars, where the provider reports one. */
  balance_usd: number | null;
  /** Why the last check failed (`status: error`). */
  error: string | null;
  /** Something to know about a working connection (an unencrypted `http://` URL). */
  note: string | null;
  /** Time of the last check. */
  checked: Time | null;
  /** Fields that are not secret as entered; secrets as a hint (`••••` and their last 4 characters). */
  values: Record<string, string>;
}

/** How likely an offer is to start now (`none`: launching would be refused). */
export type CloudAvailability = 'high' | 'medium' | 'low' | 'none' | 'unknown';

/** Something a connection can run on: a GPU type with its price (`GET /api/cloud/connections/{id}/offers`). */
export interface CloudOffer {
  /** Opaque, unique within the connection (`Compute.offer_id`). */
  id: string;
  /** e.g. `RTX 4090`. */
  gpu: string;
  /** GPUs of the instance (a run trains on one). */
  gpu_count: number;
  /** Memory of one GPU in bytes (0 when unknown). */
  vram_bytes: number;
  /** US dollars per hour; `null` for compute that is not billed (your own machine). */
  price_per_hour: number | null;
  region: string | null;
  /** A qualifier of the offer, e.g. `Secure Cloud`. */
  detail: string | null;
  availability: CloudAvailability;
}

export interface CloudOfferList {
  /** Cheapest first (offers without a price first). */
  offers: CloudOffer[];
  /** When the provider was asked (answers are cached for a minute). */
  fetched: Time;
}

/** Where an instance is in its life; alive until `terminated` or `lost`. */
export type CloudInstanceStatus = 'launching' | 'starting' | 'running' | 'terminating' | 'terminated' | 'lost';

/** A machine FOLO launched at a provider (`GET /api/cloud/instances`). */
export interface CloudInstance {
  /** e.g. `ci_8a2f`. */
  id: string;
  connection_id: string;
  provider: string;
  /** The provider's own id (to find it in its console); `null` while launching. */
  provider_ref: string | null;
  /** Its name at the provider, `folo-<run id>-<nonce>`. */
  name: string;
  run_id: string | null;
  offer: CloudOffer;
  status: CloudInstanceStatus;
  /** Time of the launch request. */
  created: Time;
  /** When it was confirmed gone. */
  ended: Time | null;
  /** When it is terminated at the latest (the run's `max_hours`). */
  deadline: Time | null;
  /** Estimated cost so far in US dollars; `null` without a price. */
  cost_usd: number | null;
  /** The last failed provider request about it (a termination being retried). */
  error: string | null;
}

/** `TrainConfig.compute`: train on a cloud connection instead of this machine. */
export interface Compute {
  connection_id: string;
  offer_id: string;
  /** The run is stopped and its instance terminated after this many hours at the latest (0.25 to 720; default 12). */
  max_hours?: number;
  /** Disk of the instance in GB; missing sizes it for the dataset (twice its size plus 20 GB). */
  disk_gb?: number;
}

/**
 * Where a cloud run is in its lifecycle (docs/CLOUD.md), in order. A newer server may send a phase this client does not
 * know: read the field as a string and show an unknown one as its raw word.
 */
export type CloudPhase = 'provisioning' | 'booting' | 'syncing' | 'training' | 'collecting' | 'releasing' | 'released';

/** Bytes moved of a transfer (`RunCloud.progress`). */
export interface TransferProgress {
  done_bytes: number;
  /** 0 when unknown. */
  total_bytes: number;
}

/** `Run.cloud` and the SSE `cloud` event: where a cloud run executes and how far it is. */
export interface RunCloud {
  connection_id: string;
  provider: string;
  /** The offer as it was when the run started. */
  offer: CloudOffer;
  /** `null` before the launch and for providers that launch none. */
  instance_id: string | null;
  /** A `CloudPhase`, or a newer server's phase this client does not know. */
  phase: CloudPhase | (string & {});
  /** Progress of the phase's transfer (`syncing`, `collecting`), else `null`. */
  progress: TransferProgress | null;
  max_hours: number;
  /** When the time limit ends; `null` until the instance exists. */
  deadline: Time | null;
  /** Estimated cost so far in US dollars (final once `released`); `null` without a price. */
  cost_usd: number | null;
  /** When `weights/last.folo` was last copied to this machine. */
  synced: Time | null;
}

/** `RunEstimate.cost`: what a cloud run would cost. */
export interface CostEstimate {
  price_per_hour: number;
  /** `price_per_hour` × `est_total_s` (training only: booting and the upload come on top). */
  est_usd: number;
  est_usd_low: number;
  est_usd_high: number;
  /** The most the run can cost: `price_per_hour` × `compute.max_hours`. */
  max_usd: number;
}

// ---------- Agent labeling (docs/API.md); servers that have it list `agent_labeling` in `SystemInfo.features` ----------

/** A coding-agent CLI on the server's machine that FOLO can ask for labels. */
export type LabelAgentId = 'claude' | 'codex';

/** A model of a `LabelAgent`: `id` is what the CLI takes, `label` the display name. */
export interface LabelModel {
  id: string;
  label: string;
}

/** `GET /api/label/agents`: one agent CLI and whether the server found it. */
export interface LabelAgent {
  id: LabelAgentId;
  /** Product name: "Claude Code", "Codex". */
  name: string;
  /** Its CLI is installed and answered `--version`; whether it is signed in is only known once a run calls it. */
  available: boolean;
  version: string | null;
  /** Suggestions: a run takes any model name the CLI accepts. */
  models: LabelModel[];
  /** The model a run uses when it names none; null: the CLI's own default. */
  default_model: string | null;
  /** The effort a run uses when it names none: what labels best for its cost with this agent. */
  default_effort: LabelEffort;
  /** How to make the agent available (install, sign in); null when it is. */
  hint: string | null;
}

export type LabelJobStatus = 'idle' | 'running' | 'stopping';

/** `pending + labeled + failed == images`. */
export interface LabelCounts {
  images: number;
  pending: number;
  labeled: number;
  failed: number;
  boxes: number;
}

export type LabelEffort = 'low' | 'medium' | 'high';
export type LabelOutcome = 'completed' | 'stopped' | 'failed';

/** Tokens the agent CLIs reported. `cost_usd` is null when no call reported one (Codex reports none). */
export interface LabelUsage {
  calls: number;
  /** Cached ones included. */
  input_tokens: number;
  cached_input_tokens: number;
  /** Reasoning included. */
  output_tokens: number;
  cost_usd: number | null;
}

/** One agent run over some images of a `LabelJob`. */
export interface LabelRun {
  agent: LabelAgentId;
  /** null: the CLI's default. */
  model: string | null;
  effort: LabelEffort;
  started: Time;
  /** null while the run is active. */
  finished: Time | null;
  total: number;
  done: number;
  failed: number;
  /** null while the run is active. */
  outcome: LabelOutcome | null;
  /** Why it failed (`outcome: failed`), else null. */
  error: string | null;
  /** How to fix `error`, when there is something useful to say. */
  hint?: string | null;
  usage: LabelUsage;
}

/** A labeling job: a set of images, the classes to find in them, and the labels so far. */
export interface LabelJob {
  id: string;
  name: string;
  /** A box's `class` indexes this list. */
  classes: string[];
  /** What the agent is told, in the user's words (may be empty). */
  instructions: string;
  created: Time;
  status: LabelJobStatus;
  counts: LabelCounts;
  /** The active run, else the most recent one; null before the first. */
  run: LabelRun | null;
  /** Every run of the job. */
  usage: LabelUsage;
  /** The dataset most recently made from the job. */
  dataset_id: string | null;
}

export type LabelImageStatus = 'pending' | 'labeling' | 'labeled' | 'failed';
/** Who drew an image's current boxes. */
export type LabelSource = 'agent' | 'manual';

export interface LabelImage {
  id: string;
  /** Unique within the job; the dataset keeps it. */
  file: string;
  width: number;
  height: number;
  status: LabelImageStatus;
  /** Pixel xyxy; kept while the image is labeled again. */
  boxes: Box[];
  /** null until labeled. */
  source: LabelSource | null;
  /** Why the last agent call for it failed (`status: failed`). */
  error: string | null;
}

/** `POST /api/label/jobs`. */
export interface CreateLabelJob {
  name: string;
  classes: string[];
  instructions?: string;
  /** A folder of images on the server's machine to start with. */
  path?: string;
}

/** `PUT /api/label/jobs/{id}`: absent fields stay. Once the job has boxes, classes may only be renamed or appended. */
export interface UpdateLabelJob {
  name?: string;
  classes?: string[];
  instructions?: string;
}

/** A file `POST /api/label/jobs/{id}/images` did not add, and why. */
export interface SkippedImage {
  file: string;
  reason: string;
}

/** `POST /api/label/jobs/{id}/images`. */
export interface LabelImagesAdded {
  added: LabelImage[];
  skipped: SkippedImage[];
  job: LabelJob;
}

export interface LabelImagesQuery {
  status?: LabelImageStatus;
  offset?: number;
  /** Default 100, at most 1000. */
  limit?: number;
}

/** `GET /api/label/jobs/{id}/images`: `total` counts every match before offset / limit; in the order added. */
export interface LabelImagePage {
  total: number;
  images: LabelImage[];
}

/** `unlabeled`: pending and failed images (the default); `all`: every image, replacing its labels. */
export type LabelScope = 'unlabeled' | 'all';

/** `POST /api/label/jobs/{id}/start`: only `agent` is required. */
export interface StartLabeling {
  agent: LabelAgentId;
  model?: string | null;
  /** Absent or null: the agent's `default_effort`. */
  effort?: LabelEffort | null;
  scope?: LabelScope;
  /** Exactly these images, whatever their status, instead of `scope`. */
  image_ids?: string[] | null;
  /** An extra instruction for this run only (at most 2000 characters). */
  note?: string;
  batch?: number | null;
  parallel?: number | null;
}

/** `POST /api/label/jobs/{id}/dataset`. */
export interface LabelDatasetRequest {
  /** Defaults to the job's name. */
  name?: string | null;
  /** In 0..=0.5; default 0.2. */
  val_fraction?: number;
  seed?: number;
}

/** SSE events on GET /api/label/jobs/{id}/events, keyed by the SSE `event:` name. */
export interface LabelEventMap {
  job: LabelJob;
  image: LabelImage;
  removed: { id: string };
  /** More changed than the stream carries (a bulk add, a client that fell behind): load the images again. */
  reload: Record<string, never>;
  log: LogLine;
}

// ---------- Vision benchmarks (docs/API.md); servers that have them list `vision_bench` in `SystemInfo.features` ----------
// `folo_api::bench` field for field (docs/BENCH.md defines every number).

/** What kind of trouble a `BenchCondition` is. */
export type BenchConditionGroup = 'light' | 'blur' | 'noise' | 'medium' | 'digital' | 'geometry';

/** One stress condition: a deterministic change to an image that keeps its size, at severities 1 to 5. */
export interface BenchCondition {
  /** Stable id, e.g. `haze`; what a `BenchPlanCondition` names. */
  id: string;
  /** Display name, e.g. `Haze`. */
  name: string;
  group: BenchConditionGroup;
  /** What it does to an image and what it stands for, in a sentence. */
  description: string;
}

/** An axis of the score. */
export type BenchAxis = 'accuracy' | 'robustness' | 'self_labeling' | 'calibration' | 'localization';

/** An axis as the catalog lists it. */
export interface BenchAxisInfo {
  axis: BenchAxis;
  /** Display name, e.g. `Self-labeling`. */
  name: string;
  /** Its share of the score when every axis is measured; the five sum to 1. */
  weight: number;
  /** What it measures, in a sentence. */
  description: string;
}

/** `GET /api/bench/catalog`: what a benchmark can be made of and how it is scored. */
export interface BenchCatalog {
  /** Every stress condition, in display order. */
  conditions: BenchCondition[];
  /** The highest severity (5). */
  severities: number;
  /** The plan a benchmark gets when it names none (every condition at 1, 3 and 5, weight 1). */
  default_plan: BenchPlan;
  /** The axes of the score, in display order, with their weights. */
  axes: BenchAxisInfo[];
}

/** One condition of a `BenchPlan`. */
export interface BenchPlanCondition {
  /** A `BenchCondition.id`. */
  condition: string;
  /** The severities to run: 1 to 5 distinct values in `1..=5`, ascending. */
  severities: number[];
  /** How much the condition counts in the robustness axis, relative to the others (`0 < weight <= 10`; default 1). */
  weight: number;
  /** Why it is in the plan, when someone said (an agent's design does). */
  why?: string | null;
}

/** The agent behind a designed `BenchPlan`. */
export interface BenchDesigner {
  agent: LabelAgentId;
  /** The model it was asked for; null means the CLI's default. */
  model: string | null;
}

/** Which conditions a benchmark runs, how hard and how much each counts. */
export interface BenchPlan {
  /** Each condition at most once. Empty: the unchanged images only (no robustness axis). */
  conditions: BenchPlanCondition[];
  /** Why this plan fits the benchmark, when someone said (an agent's design does). */
  rationale?: string | null;
  /** The agent that proposed the plan; null for the default plan and one written by hand. */
  designed_by?: BenchDesigner | null;
}

/**
 * Where a set of images comes from: a benchmark's images with their truth, or a self-improvement loop's pool (whose
 * labels are never read). A benchmark's split defaults to `val`, a pool's to `train`.
 */
export type BenchSource = { kind: 'dataset'; dataset_id: string; split?: Split | null } | { kind: 'label_job'; job_id: string };

/** The leading entry of a `Bench`. */
export interface BenchBest {
  entry_id: string;
  name: string;
  /** `0..=100`. */
  score: number;
  /** The leader's `score.partial`: its number compares only loosely with complete entries. */
  partial: boolean;
}

/**
 * A benchmark: images with their true boxes, copied in and frozen, and the plan of conditions to run them under. Its
 * images and truth never change; its plan can change only while it has no entries.
 */
export interface Bench {
  /** Opaque id, e.g. `bn_3f9a`. */
  id: string;
  name: string;
  /** Where the camera is and what makes the job hard, in the user's words (may be empty). An agent designing the plan reads it. */
  description: string;
  /** A box's `class` indexes this list. */
  classes: string[];
  created: Time;
  source: BenchSource;
  /** The source's display name when the benchmark was made (the dataset's or the labeling job's). */
  source_name: string;
  images: number;
  /** True boxes over all images. */
  boxes: number;
  /** True boxes per class, aligned with `classes`. */
  class_boxes: number[];
  plan: BenchPlan;
  /** Seed of everything random in the benchmark (which images were sampled, where noise falls). */
  seed: number;
  /** A hash of the images, truth, plan and seed, as hex: two entries compare when their fingerprints agree. */
  fingerprint: string;
  /** Entries, whatever their status. */
  entries: number;
  /** The finished entry with the highest score; null before the first. */
  best: BenchBest | null;
  /** An entry or a loop of it is queued, running or stopping (exactly when DELETE answers `409 bench_active`). */
  active: boolean;
}

/** `POST /api/bench/benchmarks`. */
export interface CreateBench {
  name: string;
  /** At most 4000 characters. */
  description?: string;
  source: BenchSource;
  /** At most this many images (1 to 5000, default 500), sampled with `seed` when the source has more. */
  max_images?: number | null;
  /** Default 0. */
  seed?: number;
  /** Absent: the catalog's `default_plan`. */
  plan?: BenchPlan | null;
}

/** `PUT /api/bench/benchmarks/{id}`: absent fields stay. `plan` only while the benchmark has no entries (`409 bench_frozen`). */
export interface UpdateBench {
  name?: string;
  description?: string;
  plan?: BenchPlan;
}

/** `POST /api/bench/benchmarks/{id}/design`: ask a coding agent to propose the plan. */
export interface DesignBench {
  agent: LabelAgentId;
  /** Absent or null: the agent's default. */
  model?: string | null;
  /** Absent or null: the agent's default. */
  effort?: LabelEffort | null;
  /** Anything to add to the benchmark's description for this request. */
  note?: string;
}

/** The answer of `POST …/design`: a proposal (valid as it is, `designed_by` and `rationale` set), not saved until it is `PUT`. */
export interface BenchDesign {
  plan: BenchPlan;
  /** What the call used. */
  usage: LabelUsage;
}

/** One image of a `Bench` with its truth. */
export interface BenchImage {
  /** Opaque, unique within the benchmark. */
  id: string;
  /** File name, unique within the benchmark. */
  file: string;
  /** Pixels (after EXIF orientation); the same under every condition. */
  width: number;
  height: number;
  /** The true boxes, pixel xyxy, under the condition asked for (a geometric condition moves them). */
  boxes: Box[];
}

/** `GET /api/bench/benchmarks/{id}/images`: `limit` at most 1000; `condition` and `severity` go together. */
export interface BenchImagesQuery {
  offset?: number;
  limit?: number;
  condition?: string;
  severity?: number;
}

/** `GET /api/bench/benchmarks/{id}/images`. */
export interface BenchImagePage {
  /** All images of the benchmark (before offset / limit). */
  total: number;
  /** This page, in the benchmark's order. */
  images: BenchImage[];
}

/** What `benchImageUrl` asks for: the image under a condition (both or neither), and a thumbnail size. */
export interface BenchImageOptions {
  condition?: string;
  severity?: number;
  thumb?: number;
}

/** A detection: a box, its class and how sure the contestant is. */
export interface BenchDetection extends Box {
  /** Confidence in `0..=1`; 1 when absent, and for a contestant that gives none. */
  score: number;
}

/** The detections of one image under one condition, as an external contestant submits them. */
export interface BenchPrediction {
  /** A `BenchImage.id`. */
  image: string;
  /** The condition the image was fetched under; absent or null for the unchanged image. */
  condition?: string | null;
  /** Its severity (required with `condition`). */
  severity?: number | null;
  /** What the model found, in pixels of the image as it was served; `score` may be left out (1). */
  boxes: (Box & { score?: number })[];
}

/** Who a new entry is (`StartBenchEntry.contestant`). */
export type BenchContestantSpec =
  /** A model of this server, run by FOLO. */
  | { kind: 'model'; model_id: string }
  /** A coding agent on the server's machine, asked for boxes as agent labeling asks. */
  | { kind: 'agent'; agent: LabelAgentId; model?: string | null; effort?: LabelEffort | null }
  /**
   * Any other vision model: its predictions, made wherever it runs. `scored: false` when it gives no confidences
   * (default true). Every unchanged image must be answered for; a condition of the plan is answered for every image at
   * every planned severity, or left out. One record carries at most 1000 boxes.
   * `open: false` (the default): `predictions` is the whole entry, scored in the request (up to 64 MiB). `open: true`:
   * the entry is created `running`, to be filled in parts (`POST …/entries/{entry_id}/predictions`, `BenchPredictions`)
   * and scored by `POST …/entries/{entry_id}/finish`; `predictions` here, if any, is its first part. An open entry is
   * active; stopping it discards it.
   */
  | { kind: 'external'; scored?: boolean; open?: boolean; predictions?: BenchPrediction[] };

/**
 * `POST /api/bench/benchmarks/{id}/entries/{entry_id}/predictions`: a part of an open external entry (up to 64 MiB),
 * all or nothing: `400 invalid_body` names the first bad record (a record for an image, condition and severity already
 * received is one) and adds nothing. `409 bench_entry_not_open` on any entry that is not an open external one.
 */
export interface BenchPredictions {
  predictions: BenchPrediction[];
}

/** `POST /api/bench/benchmarks/{id}/entries`: run a contestant on the benchmark. */
export interface StartBenchEntry {
  contestant: BenchContestantSpec;
  /**
   * The name on the leaderboard (one line, at most 100 characters); absent uses the model's name, or the agent's with
   * its model in parentheses, `Claude Code (haiku)`, `Codex (<model>)` (the agent's name alone when it has no default
   * model). Required for an external contestant.
   */
  name?: string | null;
  /** Only the first this many images (a seeded shuffle, so a prefix is a fair sample). Absent: all for a model, 48 for an agent. Not for external. */
  max_images?: number | null;
  /** Whether to run the plan's conditions. Absent: true for a model, false for an agent. Not for external. */
  conditions?: boolean | null;
}

/** Who an entry is. */
export type BenchContestant =
  /** A model of this server (it may have been deleted since). */
  | { kind: 'model'; model_id: string; arch: Arch; params: number; imgsz: number }
  /** A coding agent; `model` null means the CLI's default. */
  | { kind: 'agent'; agent: LabelAgentId; model: string | null; effort: LabelEffort }
  /** Predictions made elsewhere. */
  | { kind: 'external' };

/** Where a `BenchEntry` stands; `queued`, `running` and `stopping` are active. */
export type BenchEntryStatus = 'queued' | 'running' | 'stopping' | 'done' | 'failed' | 'stopped';

/** How far a running entry is. */
export interface BenchProgress {
  /** Images answered so far, over all conditions; for an open external entry, the records received so far. */
  done: number;
  /** The entry's images times one plus its conditions' severities (for an open external entry, the plan's). */
  total: number;
  /** The condition being run; null for the unchanged images. */
  condition: string | null;
  severity: number | null;
}

/** How fast a model contestant ran on the server (once `done`). Not part of the score: it depends on the machine. */
export interface BenchSpeed {
  /** e.g. `gpu:0`. */
  device: string;
  /**
   * Wall time per image on the unchanged images, decode to boxes, in milliseconds: run in batches (images prepared on
   * several threads while the previous batch runs) and counted from the end of the first batch, so loading the model
   * and warming up the device are left out. A throughput, not the latency of a single image.
   */
  ms_per_image: number;
}

/** One axis of a `BenchScore`. */
export interface BenchAxisScore {
  axis: BenchAxis;
  /** `0..=1`; null when it was not measured. */
  value: number | null;
  /** Its share of the score. */
  weight: number;
}

/** The score of an entry: one number and what it is made of. */
export interface BenchScore {
  /** `0..=100`: the weighted mean of the measured axes, times 100. */
  total: number;
  /** An axis was not measured, or the entry ran a part of the images: it compares loosely with complete entries. */
  partial: boolean;
  /** Every axis, in display order (accuracy, robustness, self-labeling, calibration, localization). */
  axes: BenchAxisScore[];
}

/** A contestant's run of a benchmark, on its leaderboard. */
export interface BenchEntry {
  /** Opaque id, e.g. `be_91c2`. */
  id: string;
  bench_id: string;
  /** Name on the leaderboard. */
  name: string;
  contestant: BenchContestant;
  status: BenchEntryStatus;
  created: Time;
  /** When it finished, failed or was stopped; null until then. */
  finished: Time | null;
  progress: BenchProgress;
  /** Images it runs: all of the benchmark's, or a prefix. */
  images: number;
  /** Whether it runs the plan's conditions. */
  conditions: boolean;
  /** Whether the contestant gives confidences. */
  scored: boolean;
  /** The benchmark's fingerprint when the entry ran. */
  fingerprint: string;
  /** Classes of the benchmark the contestant does not know (a model matched by class name); its AP on them is 0. */
  missing_classes: string[];
  /** Null until `done`. */
  score: BenchScore | null;
  /** mAP@0.5:0.95 on the unchanged images, repeated here for lists; null until `done`. */
  map50_95: number | null;
  /** The 95% interval of `map50_95` (`BenchAccuracy.ci95`), for lists; null until `done`. */
  ci95: [number, number] | null;
  /** How many of the plan's conditions it ran completely; 0 when none ran or it has not finished. */
  conditions_run: number;
  /** A model contestant, once `done`. */
  speed?: BenchSpeed | null;
  /** What an agent contestant's calls used. */
  usage?: LabelUsage | null;
  /** Why it failed (`status: failed`), else null. */
  error: string | null;
  /** How to fix `error`, when there is something useful to say. */
  hint?: string | null;
}

/** `GET …/entries/{id}/result`: accuracy on one class. */
export interface BenchClassAccuracy {
  /** Class index. */
  class: number;
  /** True boxes of the class. */
  truths: number;
  /** AP@0.5. */
  map50: number;
  /** AP@0.5:0.95. */
  map50_95: number;
}

/** Accuracy on the unchanged images (docs/METRICS.md for AP). */
export interface BenchAccuracy {
  /** mAP@0.5:0.95: the accuracy axis. */
  map50_95: number;
  /** Its 95% bootstrap interval over images, `[low, high]`. */
  ci95: [number, number];
  map50: number;
  map75: number;
  /** Class-mean precision, recall and F1 at `best_conf`. */
  precision: number;
  recall: number;
  f1: number;
  /** The confidence with the best class-mean F1. */
  best_conf: number;
  /** mAP@0.5:0.95 of objects under 32² px, 32²–96² px, over 96² px; null when the benchmark has none. */
  map_small: number | null;
  map_medium: number | null;
  map_large: number | null;
  /** Per class with at least one true box, in class order. */
  per_class: BenchClassAccuracy[];
}

/** Accuracy at one severity of a condition. */
export interface BenchSeverityPoint {
  /** `1..=5`. */
  severity: number;
  map50_95: number;
  map50: number;
  /** `min(1, map50_95 / the unchanged images' map50_95)`; 0 when that is 0. */
  retention: number;
}

/** One condition's result. */
export interface BenchConditionResult {
  /** The condition's id. */
  condition: string;
  /** Its weight in the plan. */
  weight: number;
  /** The mean of its severities' `retention`. */
  retention: number;
  /** Per severity run, ascending. */
  points: BenchSeverityPoint[];
}

/** Accuracy under the plan's conditions. */
export interface BenchRobustness {
  /** The robustness axis: the weighted mean of the conditions' `retention`. */
  retention: number;
  /** Per condition, in plan order (only those that ran). */
  conditions: BenchConditionResult[];
}

/** One confidence bin of the reliability diagram. */
export interface BenchCalibrationBin {
  lo: number;
  hi: number;
  /** Detections in the bin. */
  count: number;
  /** Their mean confidence; null for an empty bin. */
  confidence: number | null;
  /** The share of them that are right; null for an empty bin. */
  precision: number | null;
}

/** Reliability of the confidences on the unchanged images. */
export interface BenchCalibration {
  /** Expected calibration error, `0..=1` (0 is perfect); the axis is `1 - ece`. */
  ece: number;
  /** Detections that entered it (confidence at least 0.1). */
  detections: number;
  /** Ten bins of confidence, `[0, 0.1)` to `[0.9, 1]`. */
  bins: BenchCalibrationBin[];
}

/** Box quality of the correct detections on the unchanged images. */
export interface BenchLocalization {
  /** Mean IoU of the correct detections with their true boxes, `0.5..=1`; null when nothing was correct. The axis is `(mean_iou - 0.5) / 0.5`. */
  mean_iou: number | null;
  /** Correct detections it is the mean of. */
  matches: number;
  /** How many fall in each tenth of the IoU range 0.5..1: `[0.5, 0.55)`, …, `[0.95, 1]` (ten counts). */
  histogram: number[];
}

/** Whether a contestant's own labels are good enough to train on. */
export type BenchReadiness = 'ready' | 'partial' | 'not_ready';

/** One confidence threshold of `BenchSelfLabeling.curve`. */
export interface BenchLabelPoint {
  /** Detections with at least this confidence are kept. */
  confidence: number;
  /** The share of them that are right; null when none is kept. */
  precision: number | null;
  /** The share of all true boxes they find. */
  recall: number;
  /** How many are kept. */
  labels: number;
}

/** What the contestant's own labels would be worth, answered on images whose truth is known. */
export interface BenchSelfLabeling {
  /** The precision labels must reach (0.9). */
  target_precision: number;
  /** The lowest confidence at which the kept detections are that precise; null when none is (and without confidences). */
  threshold: number | null;
  /** Precision of the labels at `threshold` (of all detections without confidences). */
  precision: number;
  /** The share of all true boxes those labels find: the self-labeling axis, 0 when the target is not reached. */
  recall: number;
  /** Mean IoU of the right ones with their true boxes; null when there are none. */
  mean_iou: number | null;
  /** Labels per image at `threshold`. */
  labels_per_image: number;
  readiness: BenchReadiness;
  /** Precision and recall at each confidence 0.00, 0.01, …, 1.00 (one point without confidences). */
  curve: BenchLabelPoint[];
}

/** `GET …/entries/{id}/result`: everything measured, for a finished entry (`409 bench_entry_not_done` before). */
export interface BenchResult {
  entry: BenchEntry;
  accuracy: BenchAccuracy;
  /** Null when the conditions were not run. */
  robustness: BenchRobustness | null;
  /** Null without confidences. */
  calibration: BenchCalibration | null;
  localization: BenchLocalization;
  self_labeling: BenchSelfLabeling;
}

/** One image of a finished entry: the truth and what the contestant found, unchanged image. */
export interface BenchEntryImage {
  /** The image with its true boxes. */
  image: BenchImage;
  /** The detections at or above the entry's `best_conf`, highest confidence first. */
  detections: BenchDetection[];
  /** True boxes none of them matches (IoU at least 0.5, same class). */
  missed: number;
  /** Detections that match no true box. */
  wrong: number;
}

/** `GET …/entries/{id}/images`: `order` is the benchmark's order, `worst` the most `missed + wrong` first. */
export interface BenchEntryImagesQuery {
  sort?: 'order' | 'worst';
  offset?: number;
  limit?: number;
}

/** `GET …/entries/{id}/images`. */
export interface BenchEntryImagePage {
  /** Images the entry ran (before offset / limit). */
  total: number;
  images: BenchEntryImage[];
}

/** How the students of a loop train: the fields of `TrainConfig` a loop lets you choose. */
export interface BenchStudent {
  /** Absent: the first teacher's. A larger one is "the next bigger model". */
  arch?: Arch | null;
  /** Epochs per generation (default 30). */
  epochs?: number | null;
  /** Default: the first teacher's. */
  imgsz?: number | null;
  batch?: number | null;
  /** Default `auto`. */
  device?: string | null;
  /** Starting weights as `TrainConfig.weights` takes them; absent trains from scratch. */
  weights?: string | null;
  /** Default 0. */
  seed?: number;
}

/**
 * `POST /api/bench/benchmarks/{id}/loops`: each generation, the current model labels the pool by itself, a student
 * trains on those labels alone, and the benchmark grades the student, the next generation's teacher.
 */
export interface StartBenchLoop {
  /**
   * Absent: `<teacher's name> loop`. Its generations' datasets and training runs are named `<name> gen <n>`, so each
   * student model is `<name> gen <n>-best` (a run's best checkpoint), and so is its benchmark entry.
   */
  name?: string | null;
  /** The first teacher: a model of this server whose classes are the benchmark's (same names, same order). */
  teacher: string;
  /** The images the teachers label; its own labels are never read, images also in the benchmark are left out. */
  pool: BenchSource;
  /** 1 to 5 (default 1). */
  generations?: number | null;
  /**
   * Keep a teacher's detections from this confidence up; absent: the threshold its benchmark entry found (0.5 without
   * one). A teacher that keeps a box on fewer than 10 pool images at it fails the loop at labeling.
   */
  min_confidence?: number | null;
  student?: BenchStudent;
}

/** Where a `BenchLoop` stands; `running` and `stopping` are active. */
export type BenchLoopStatus = 'running' | 'stopping' | 'done' | 'failed' | 'stopped';

/** What a running loop is doing. */
export type BenchLoopPhase = 'benchmarking' | 'labeling' | 'training';

/** The last generation against the first teacher on mAP@0.5:0.95: above its 95% interval, inside it, below it. */
export type BenchLoopVerdict = 'improved' | 'held' | 'regressed';

/**
 * The labels a teacher made of a loop's pool. A teacher's silence is not a label: a pool image it kept no box on is left
 * out of the student's dataset (never written as a background image), so the dataset is `labeled_images` images. With
 * fewer than 10 the loop fails at labeling instead (its `error` says how many boxes on how many images, at which
 * confidence; its `hint` points at a lower `min_confidence`), and the generation has no labels.
 */
export interface BenchLabels {
  /** The confidence from which detections were kept. */
  threshold: number;
  /** Images of the pool the teacher looked at (the pool less the benchmark's own images). */
  images: number;
  /** Of those, the images it kept at least one box on: the student's dataset (train and val). */
  labeled_images: number;
  /** Boxes kept, on those images. */
  boxes: number;
  /** Pool images left out because the benchmark has them. */
  excluded: number;
  /** The dataset written from them (train and val both from the pool, both with the teacher's labels). */
  dataset_id: string;
}

/** One generation of a `BenchLoop`: 0 is the first teacher, then one per student. */
export interface BenchGeneration {
  generation: number;
  /** The labels its teacher made for it; null for generation 0 and until they are made. */
  labels: BenchLabels | null;
  /** The training run of this generation's model; null for generation 0 and until it starts. */
  run_id: string | null;
  /** The generation's model; null until its training has finished. */
  model_id: string | null;
  model_name: string | null;
  /** Its benchmark entry; null until it is started. */
  entry_id: string | null;
  /** Its score; null until the entry is done. */
  score: number | null;
  /** Its mAP@0.5:0.95 on the unchanged images, and that number's 95% interval. */
  map50_95: number | null;
  ci95: [number, number] | null;
}

/** A self-improvement loop of a benchmark. */
export interface BenchLoop {
  /** Opaque id, e.g. `bl_07d1`. */
  id: string;
  bench_id: string;
  name: string;
  status: BenchLoopStatus;
  /** Null once it has ended. */
  phase: BenchLoopPhase | null;
  created: Time;
  /** Null while it runs. */
  finished: Time | null;
  /** The request, defaults filled in. */
  config: StartBenchLoop;
  /** Generation 0 (the first teacher), then one per student, as far as the loop has come. */
  generations: BenchGeneration[];
  /** Null until `done`. */
  verdict: BenchLoopVerdict | null;
  /** Why it failed (`status: failed`), else null. */
  error: string | null;
  /** How to fix `error`, when there is something useful to say. */
  hint?: string | null;
}

/** SSE events on `GET /api/bench/benchmarks/{id}/events`, keyed by the SSE `event:` name. */
export interface BenchEventMap {
  /** The benchmark after a change (name, plan, entry count, best entry); also first on connect, and to a subscriber that fell behind (load the lists again). */
  bench: Bench;
  /** An entry after a change (created, progress at most twice a second, done, failed). */
  entry: BenchEntry;
  loop: BenchLoop;
  /** An entry or a loop was deleted. */
  removed: { id: string };
}
