// Throughput, run estimates and the benchmark probe (docs/API.md, Throughput and estimates).

import type { CostEstimate } from './cloud';
import type { Arch } from './models';
import type { Time } from './time';

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
