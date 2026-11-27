// The reference training figures and how they carry to other settings: docs/BENCHMARKS.md, "Trainer throughput"
// (the fallback of GET /api/system/throughput and POST /api/runs/estimate on a server that has not measured a size).
// New run uses them where the server has no estimate; the mock serves them as its reference rows. Keep in step with
// the document.

import type { Arch } from '$lib/api/types';
import { modelSize } from '$lib/benchmarks';

/** Params and GFLOPs at 640 px: docs/BENCHMARKS.md for v1 n/s, docs/MODEL.md's scaling tables for v1 m/l/x and v2 n..x. */
export const archSpec: Record<Arch, { params: number; gflops: number }> = {
  'folo-n': modelSize['folo-n']!,
  'folo-s': modelSize['folo-s']!,
  'folo-m': { params: 20_050_000, gflops: 68.0 },
  'folo-l': { params: 25_310_000, gflops: 86.9 },
  'folo-x': { params: 56_880_000, gflops: 195.3 },
  'folo2-n': { params: 2_850_000, gflops: 6.9 },
  'folo2-s': { params: 9_850_000, gflops: 22.4 },
  'folo2-m': { params: 20_800_000, gflops: 70.5 },
  'folo2-l': { params: 26_200_000, gflops: 90.0 },
  'folo2-x': { params: 58_800_000, gflops: 201.5 },
};

/** Images per second of a training phase: with mosaic, without (close mosaic), and validation. */
export interface PhaseRates {
  mosaic: number;
  close: number;
  val: number;
}

interface RefRow extends PhaseRates {
  arch: Arch;
  imgsz: number;
  amp: boolean;
  cpu: boolean;
}

/**
 * The document's whole-run rows (batch 16), plus amp at 640 as the f32 rate times the training step's ratio (1.24 for
 * folo-n, 1.61 for folo-s). Validation of the 1080p run is overhead-bound (~97 img/s).
 */
const ROWS: RefRow[] = [
  { arch: 'folo-n', imgsz: 640, amp: false, cpu: false, mosaic: 208, close: 215, val: 275 },
  { arch: 'folo-s', imgsz: 640, amp: false, cpu: false, mosaic: 105, close: 110, val: 97 },
  { arch: 'folo-n', imgsz: 320, amp: false, cpu: false, mosaic: 440, close: 471, val: 365 },
  { arch: 'folo-n', imgsz: 320, amp: true, cpu: false, mosaic: 406, close: 430, val: 395 },
  { arch: 'folo-n', imgsz: 640, amp: true, cpu: false, mosaic: 208 * 1.24, close: 215 * 1.24, val: 275 },
  { arch: 'folo-s', imgsz: 640, amp: true, cpu: false, mosaic: 105 * 1.61, close: 110 * 1.61, val: 97 },
  { arch: 'folo-n', imgsz: 320, amp: false, cpu: true, mosaic: 14, close: 20, val: 60 },
];

/** The reference machine: where the rows were measured. */
export const REF_GPU = 'NVIDIA GeForce RTX 4080 SUPER';
export const REF_CPU = 'Intel Core i7-14700KF';
/**
 * The reference figures as a person reads them (rule 23: one wording): "FOLO's reference benchmarks (RTX 4080
 * SUPER)", never a document path.
 */
export const referenceSource = (cpu: boolean) => `FOLO's reference benchmarks (${cpu ? 'Core i7-14700KF' : 'RTX 4080 SUPER'})`;
/** The first epoch over a steady one (pipeline compilation, buffer pool growth). */
export const REF_STARTUP_S = 6;

/** Compute of a setting: GFLOPs at the image size. */
export const compute = (arch: Arch, imgsz: number) => archSpec[arch].gflops * (imgsz / 640) ** 2;

/**
 * Rates scale with the compute ratio to the 0.55 within 4x; beyond 4x larger to the 0.8, beyond 4x smaller to the 0.25
 * (applied to the part of the ratio past 4x).
 */
export function computeFactor(from: number, to: number): number {
  const r = to / from;
  if (r > 4) return 4 ** -0.55 * (r / 4) ** -0.8;
  if (r < 1 / 4) return 4 ** 0.55 * (r * 4) ** -0.25;
  return r ** -0.55;
}

/**
 * The reference rates for a setting: its own row, else the nearest row by compute with the same precision and device
 * (mixed precision is GPU only), scaled by `computeFactor`.
 */
export function referenceRates(arch: Arch, imgsz: number, amp: boolean, cpu: boolean): PhaseRates {
  const half = amp && !cpu;
  const c = compute(arch, imgsz);
  const rows = ROWS.filter((r) => r.amp === half && r.cpu === cpu);
  const near = rows.sort((a, b) => Math.abs(Math.log(compute(a.arch, a.imgsz) / c)) - Math.abs(Math.log(compute(b.arch, b.imgsz) / c)))[0]!;
  const k = computeFactor(compute(near.arch, near.imgsz), c);
  return { mosaic: near.mosaic * k, close: near.close * k, val: near.val * k };
}

const MiB = 2 ** 20;
/** Activations per 640 px image in f32, from the training-step peaks (n, s), Studio's ratios for m / l / x and v2. */
const ACT_MIB: Record<Arch, number> = {
  'folo-n': 214.6,
  'folo-s': 399.8,
  'folo-m': 757.5,
  'folo-l': 989.0,
  'folo-x': 1683.4,
  'folo2-n': 228.0,
  'folo2-s': 420.0,
  'folo2-m': 790.0,
  'folo2-l': 1030.0,
  'folo2-x': 1750.0,
};
const AMP_ACT: Record<Arch, number> = {
  'folo-n': 0.58,
  'folo-s': 0.56,
  'folo-m': 0.56,
  'folo-l': 0.56,
  'folo-x': 0.56,
  'folo2-n': 0.58,
  'folo2-s': 0.56,
  'folo2-m': 0.56,
  'folo2-l': 0.56,
  'folo2-x': 0.56,
};

/** Peak GPU memory of a training run: 16 bytes per parameter plus the activations of the batch. */
export function referenceVram(arch: Arch, imgsz: number, batch: number, amp: boolean): number {
  return archSpec[arch].params * 16 + batch * (imgsz / 640) ** 2 * ACT_MIB[arch] * MiB * (amp ? AMP_ACT[arch] : 1);
}

/**
 * The range around a reference estimate (docs/API.md): -30% / +60% on the reference machine's GPU or CPU, -60% /
 * +200% elsewhere.
 */
export const referenceSpread = (sameMachine: boolean): [number, number] => (sameMachine ? [-0.3, 0.6] : [-0.6, 2]);
