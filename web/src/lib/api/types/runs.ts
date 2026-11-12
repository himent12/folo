// Training runs: their config, metrics, samples, logs, queue and events (docs/API.md, Training runs).

import type { Compute, RunCloud } from './cloud';
import type { Box, ScoredBox } from './datasets';
import type { Arch, ModelMetrics } from './models';
import type { Time } from './time';

// ---------- Training runs ----------

export type RunStatus = 'queued' | 'running' | 'stopping' | 'stopped' | 'completed' | 'failed';
export type Device = 'auto' | 'cpu' | `gpu:${number}`;
export type Optimizer = 'auto' | 'sgd' | 'adamw' | 'musgd';

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
  /**
   * A resumed run's seconds not training since `started` (from each end until it trained again): its training time is
   * `finished` (or now) − `started` − this. Missing for a run never resumed and from older servers.
   */
  paused_s?: number;
  epoch: number;
  epochs: number;
  step: number;
  steps_per_epoch: number;
  eta_s: number | null;
  best: { epoch: number; map50_95: number } | null;
  device: string;
  model_id: string | null;
  error: string | null;
  /**
   * Stable error code when failed (e.g. `not_supported`); `interrupted` on a `stopped` run the server's stop or restart
   * ended; missing otherwise and from older servers.
   */
  error_code?: string;
  /** What the trainer resolved `config` to (auto optimizer, effective LR, ...); `null` until built, missing: older server. */
  resolved?: ResolvedTrain | null;
  /** 1-based place in the training queue while waiting there (1 starts next); `null` otherwise, missing: older server. */
  queue_position?: number | null;
  /** Where a cloud run (`config.compute`) executes and how far it is; missing for every other run. */
  cloud?: RunCloud;
  /**
   * The dataset as it was when the run was created, so the run still names it (and its classes) once the dataset is
   * removed; `classes` is empty for runs from before they were recorded. Missing from older servers.
   */
  dataset?: { name: string; classes: string[] };
}

/** `Run.resolved` (docs/API.md): the settings a run actually trains with. */
export interface ResolvedTrain {
  optimizer: 'sgd' | 'adamw' | 'musgd';
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
