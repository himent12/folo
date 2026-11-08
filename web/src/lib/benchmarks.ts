// Measured speed, copied from docs/BENCHMARKS.md (2026-11-08, one RTX 4080 SUPER and an i7-14700KF, 640 × 640;
// FOLO with untrained weights, YOLO26 with its COCO-pretrained checkpoints). Nothing here is estimated: a model size or metric the document does not list is absent, and the
// UI says "not measured". Keep in step with the document (its row numbers are noted).

import type { Arch } from '$lib/api/types';

export const BENCH_SOURCE = 'docs/BENCHMARKS.md, 2026-11-08 · RTX 4080 SUPER · 640 px';

/** Per FOLO size: GPU end-to-end latency at batch 1 (decoded image in, boxes out, NMS included; row 5), f32. */
export const e2eMs: Partial<Record<Arch, number>> = { 'folo-n': 5.18, 'folo-s': 7.87 };

/** Model size as the document lists it (params, GFLOPs at 640). */
export const modelSize: Partial<Record<Arch, { params: number; gflops: number }>> = {
  'folo-n': { params: 2_624_064, gflops: 6.54 },
  'folo-s': { params: 9_458_736, gflops: 21.59 },
};

export interface Versus {
  /** What is measured, in plain words. */
  metric: string;
  unit: 'ms' | 'img/s' | 'MiB';
  /** Lower is better for times and memory, higher for throughput. */
  better: 'lower' | 'higher';
  folo: number;
  /** The incumbent's best configuration in the document (its bar), not a weaker one. */
  yolo: number;
  /** How each side was run, when it matters for the comparison. */
  note?: string;
}

/**
 * folo-n against yolo26n, the same size class, on the same machine: every row of the document's summary table where
 * both have a number, including where FOLO is behind.
 */
export const versusYolo26n: Versus[] = [
  { metric: 'GPU forward, batch 1', unit: 'ms', better: 'lower', folo: 2.99, yolo: 4.95, note: 'f32' }, // row 1
  { metric: 'GPU forward throughput, batch 16', unit: 'img/s', better: 'higher', folo: 1385, yolo: 1454, note: 'FOLO f32 vs YOLO26n TF32 convs (strict fp32: 1334); FOLO 1461 in the previous run' }, // row 3
  { metric: 'GPU end to end, batch 1', unit: 'ms', better: 'lower', folo: 5.18, yolo: 6.92, note: 'f32, NMS included' }, // row 5
  { metric: 'GPU end-to-end throughput, batch 16', unit: 'img/s', better: 'higher', folo: 722, yolo: 615, note: 'FOLO f32 vs YOLO26n fp16, its best' }, // row 6
  { metric: 'CPU forward, batch 1', unit: 'ms', better: 'lower', folo: 21.4, yolo: 22.8, note: 'f32, best thread count; FOLO under load from other jobs' }, // row 7
  { metric: 'CPU end to end, batch 1', unit: 'ms', better: 'lower', folo: 26.9, yolo: 24.8, note: 'f32, NMS included' }, // row 8
  { metric: 'Training step, batch 16', unit: 'img/s', better: 'higher', folo: 219, yolo: 202, note: 'f32 vs f32; YOLO26n with AMP: 252' }, // row 9
  { metric: 'Training peak GPU memory', unit: 'MiB', better: 'lower', folo: 3474, yolo: 4353, note: 'f32 vs f32, FOLO train step vs YOLO26n trainer (its train step alone: 4517); with AMP: 2435' }, // row 11
];

/** FOLO's factor over the incumbent (> 1: FOLO is ahead), in the metric's own direction. */
export function advantage(v: Versus): number {
  return v.better === 'lower' ? v.yolo / v.folo : v.folo / v.yolo;
}
