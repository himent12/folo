// The mock's throughput table, run estimates and benchmark probe (docs/API.md, Throughput and estimates), as
// `folo serve --mock` has them: measured rows on its GPU for folo-n and folo-s at 640 (f32, from its runs), a folo-n
// 320 px amp probe and folo-n 640 predictions, plus one older row from another adapter under the same ordinal;
// everything else is the reference. Deterministic: values derive from the keys, times from the mock's start.

import { ApiError } from '../client';
import type {
  Arch,
  BenchmarkRequest,
  Confidence,
  EstimateBasis,
  EstimatePhase,
  InferThroughput,
  Rate,
  RunEstimate,
  ThroughputSource,
  ThroughputTable,
  TrainConfig,
  TrainThroughput,
} from '../types';
import { compute, REF_CPU, REF_STARTUP_S, referenceRates, referenceSpread, referenceVram } from '$lib/reference';
import { GPU_NAME } from './runs';

export const CPU_NAME = 'AMD Ryzen 9 7950X 16-Core Processor';
const ARCHES: Arch[] = ['folo-n', 'folo-s', 'folo-m', 'folo-l', 'folo-x'];

const REF_TRAIN_NOTE =
  'docs/BENCHMARKS.md: FOLO trainer, COCO train2017 (50 epochs, batch 16) on an RTX 4080 SUPER, scaled by compute to other sizes';
const REF_TRAIN_NOTE_CPU = 'docs/BENCHMARKS.md: FOLO trainer on an i7-14700KF, scaled by compute; not measured on this device';
const REF_INFER_NOTE = 'docs/BENCHMARKS.md: end to end, batch 1, on an RTX 4080 SUPER, scaled by compute to other sizes';
const REF_INFER_NOTE_CPU = 'docs/BENCHMARKS.md: end to end, batch 1, on an i7-14700KF, scaled by compute; not measured on this device';

/** Reference end-to-end latency at 640 px, f32, batch 1 (row 5 for n and s; the others by compute to the 0.35). */
const REF_E2E_640: Record<Arch, number> = { 'folo-n': 5.18, 'folo-s': 7.87, 'folo-m': 11.4, 'folo-l': 12.4, 'folo-x': 16.0 };

const area = (imgsz: number) => (imgsz / 640) ** 2;
const cost = compute;

/** The reference's headline (mosaic) rate; `referenceRates` has the phases. */
const refTrainRate = (arch: Arch, imgsz: number, amp: boolean, cpu: boolean) => referenceRates(arch, imgsz, amp, cpu).mosaic;

/** Latency grows with compute to the 0.35 on the GPU (rows 1 and 5); the CPU row 8 (26.9 ms for folo-n) near linearly. */
function refLatency(arch: Arch, imgsz: number, f16: boolean, cpu: boolean): number {
  if (cpu) return 26.9 * (compute(arch, imgsz) / compute('folo-n', 640)) ** 0.9;
  const r = REF_E2E_640[arch] * area(imgsz) ** 0.35;
  return f16 ? r * 0.8 : r;
}

const refVram = referenceVram;

const r1 = (x: number) => Math.round(x * 10) / 10;
const r2 = (x: number) => Math.round(x * 100) / 100;
const rate = (imgs: number, lo: number, hi: number, samples: number): Rate => ({ imgs_per_s: r1(imgs), low: r1(imgs * lo), high: r1(imgs * hi), samples });

function refTrainRow(device: string, arch: Arch, imgsz: number, amp: boolean): TrainThroughput {
  const cpu = device === 'cpu';
  const v = referenceRates(arch, imgsz, amp, cpu);
  // A reference carries its expected range (docs/API.md): the rate's side of -30% / +60% in time.
  return {
    device,
    measured_on: cpu ? REF_CPU : GPU_NAME,
    arch,
    imgsz,
    batch: 16,
    amp,
    imgs_per_s: r1(v.mosaic),
    mosaic: rate(v.mosaic, 1 / 1.6, 1 / 0.7, 0),
    close_mosaic: rate(v.close, 1 / 1.6, 1 / 0.7, 0),
    val: rate(v.val, 1 / 1.6, 1 / 0.7, 0),
    startup_s: REF_STARTUP_S,
    peak_vram_bytes: cpu ? null : Math.round(refVram(arch, imgsz, 16, amp)),
    source: 'reference',
    samples: 0,
    probes: 0,
    updated: null,
    note: cpu ? REF_TRAIN_NOTE_CPU : REF_TRAIN_NOTE,
  };
}

function refInferRow(device: string, arch: Arch, imgsz: number, f16: boolean): InferThroughput {
  const cpu = device === 'cpu';
  const ms = refLatency(arch, imgsz, f16, cpu);
  return {
    device,
    measured_on: cpu ? REF_CPU : GPU_NAME,
    arch,
    imgsz,
    batch: 1,
    precision: f16 ? 'f16' : 'f32',
    tensor_cores: f16,
    latency_ms: r2(ms),
    latency_ms_low: r2(ms * 0.85),
    latency_ms_high: r2(ms * 1.3),
    inference_ms: r2(ms * 0.55),
    imgs_per_s: r1(1000 / ms),
    source: 'reference',
    samples: 0,
    updated: null,
    note: cpu ? REF_INFER_NOTE_CPU : REF_INFER_NOTE,
  };
}

const iso = (t: number) => new Date(t).toISOString().replace(/\.\d{3}Z$/, 'Z');

/** The measured rows the mock starts with, newest first. */
function seedMeasured(t0: number): { training: TrainThroughput[]; inference: InferThroughput[] } {
  const gpu = (o: Partial<TrainThroughput> & Pick<TrainThroughput, 'arch' | 'imgsz' | 'amp' | 'imgs_per_s'>): TrainThroughput => ({
    device: 'gpu:0',
    measured_on: GPU_NAME,
    batch: 16,
    mosaic: null,
    close_mosaic: null,
    val: null,
    startup_s: 5.4,
    peak_vram_bytes: null,
    source: 'measured',
    samples: 0,
    probes: 0,
    updated: null,
    note: null,
    ...o,
  });
  const training = [
    gpu({
      arch: 'folo-n',
      imgsz: 640,
      amp: false,
      imgs_per_s: 212.4,
      mosaic: { imgs_per_s: 212.4, low: 205.1, high: 219.8, samples: 81 },
      close_mosaic: { imgs_per_s: 221.2, low: 214.6, high: 226.0, samples: 18 },
      val: { imgs_per_s: 281.9, low: 270.3, high: 290.4, samples: 99 },
      peak_vram_bytes: 3_652_000_000,
      samples: 99,
      updated: iso(t0 - 40 * 60_000),
    }),
    gpu({
      arch: 'folo-s',
      imgsz: 640,
      amp: false,
      imgs_per_s: 124.6,
      mosaic: { imgs_per_s: 124.6, low: 119.8, high: 129.1, samples: 196 },
      close_mosaic: { imgs_per_s: 129.3, low: 125.0, high: 133.2, samples: 30 },
      val: { imgs_per_s: 171.5, low: 165.2, high: 176.9, samples: 226 },
      peak_vram_bytes: 5_918_000_000,
      samples: 226,
      updated: iso(t0 - 5 * 3_600_000),
    }),
    gpu({
      arch: 'folo-n',
      imgsz: 320,
      amp: true,
      imgs_per_s: 905.3,
      mosaic: { imgs_per_s: 905.3, low: 871.0, high: 931.8, samples: 3 },
      close_mosaic: null,
      val: null,
      startup_s: null,
      peak_vram_bytes: 1_206_000_000,
      samples: 3,
      probes: 3,
      updated: iso(t0 - 2 * 86_400_000),
    }),
  ];
  const inference: InferThroughput[] = [
    {
      device: 'gpu:0',
      measured_on: GPU_NAME,
      arch: 'folo-n',
      imgsz: 640,
      batch: 1,
      precision: 'f32',
      tensor_cores: false,
      latency_ms: 5.9,
      latency_ms_low: 5.5,
      latency_ms_high: 6.4,
      inference_ms: 3.1,
      imgs_per_s: 169.5,
      source: 'measured',
      samples: 214,
      updated: iso(t0 - 12 * 60_000),
      note: null,
    },
    // Measured before the data dir moved to this machine: listed, never used for estimates or "measured here".
    {
      device: 'gpu:0',
      measured_on: 'NVIDIA GeForce RTX 3060',
      arch: 'folo-s',
      imgsz: 640,
      batch: 1,
      precision: 'f32',
      tensor_cores: false,
      latency_ms: 14.8,
      latency_ms_low: 13.9,
      latency_ms_high: 16.2,
      inference_ms: 9.6,
      imgs_per_s: 67.6,
      source: 'measured',
      samples: 57,
      updated: iso(t0 - 19 * 86_400_000),
      note: null,
    },
  ];
  return { training, inference };
}

const sameTrain = (a: TrainThroughput, b: Pick<TrainThroughput, 'device' | 'measured_on' | 'arch' | 'imgsz' | 'batch' | 'amp'>) =>
  a.device === b.device && a.measured_on === b.measured_on && a.arch === b.arch && a.imgsz === b.imgsz && a.batch === b.batch && a.amp === b.amp;
const sameInfer = (a: InferThroughput, b: Pick<InferThroughput, 'device' | 'measured_on' | 'arch' | 'imgsz' | 'precision'>) =>
  a.device === b.device && a.measured_on === b.measured_on && a.arch === b.arch && a.imgsz === b.imgsz && a.precision === b.precision;

/** The mock server's throughput store: measured rows (mutable through probes) over the reference. */
export class MockThroughput {
  #training: TrainThroughput[];
  #inference: InferThroughput[];

  constructor(t0: number) {
    const m = seedMeasured(t0);
    this.#training = m.training;
    this.#inference = m.inference;
  }

  /** `GET /api/system/throughput`: measured rows first (newest first), then the reference where nothing replaces it. */
  table(): ThroughputTable {
    const byNew = <T extends { updated: string | null }>(a: T, b: T) => (b.updated ?? '').localeCompare(a.updated ?? '');
    const training = [...this.#training].sort(byNew);
    const inference = [...this.#inference].sort(byNew);
    for (const device of ['gpu:0', 'cpu']) {
      const here = device === 'cpu' ? CPU_NAME : GPU_NAME;
      for (const arch of ARCHES)
        for (const imgsz of [640, 320])
          for (const half of device === 'cpu' ? [false] : [false, true]) {
            const t = refTrainRow(device, arch, imgsz, half);
            if (!this.#training.some((m) => sameTrain(m, { ...t, measured_on: here }))) training.push(t);
            const i = refInferRow(device, arch, imgsz, half);
            if (!this.#inference.some((m) => sameInfer(m, { ...i, measured_on: here }))) inference.push(i);
          }
    }
    return { device: 'gpu:0', device_name: GPU_NAME, training, inference };
  }

  /** `POST /api/runs/estimate` for a config the caller has already checked (dataset, device). */
  estimate(c: TrainConfig, trainImages: number, valImages: number): RunEstimate {
    if (!Number.isInteger(c.epochs) || c.epochs < 1) throw new ApiError(400, 'invalid_config', '`epochs` must be at least 1.');
    if (c.imgsz % 32 !== 0 || c.imgsz < 64 || c.imgsz > 1280) throw new ApiError(400, 'invalid_config', '`imgsz` must be a multiple of 32 in 64..=1280.');
    if (c.batch < 1) throw new ApiError(400, 'invalid_config', '`batch` must be at least 1.');
    const device = c.device === 'auto' ? 'gpu:0' : c.device;
    const cpu = device === 'cpu';
    const here = cpu ? CPU_NAME : GPU_NAME;
    const amp = !!c.amp && !cpu;
    const rows = this.#training.filter((r) => r.source === 'measured' && r.device === device && r.measured_on === here && r.mosaic);
    const exact = rows.find((r) => r.arch === c.arch && r.imgsz === c.imgsz && r.batch === c.batch && r.amp === amp);

    let row: TrainThroughput;
    let factor = 1;
    let confidence: Confidence;
    let spread: [number, number];
    let source: ThroughputSource = 'measured';
    let note: string;
    if (exact) {
      row = exact;
      const m = exact.mosaic!;
      if (exact.samples >= 3) {
        confidence = 'high';
        // Slower rate → longer time: the 90th percentile of time is the 10th of the rate.
        spread = [Math.min(-0.1, m.imgs_per_s / m.high - 1), Math.max(0.1, m.imgs_per_s / m.low - 1)];
      } else {
        confidence = 'medium';
        spread = [-0.2, 0.3];
      }
      note = `measured on this device with these settings: ${exact.samples} ${exact.probes === exact.samples ? 'probes' : 'epochs of runs'}`;
    } else {
      const target = cost(c.arch, c.imgsz);
      const dist = (r: TrainThroughput) => Math.abs(Math.log(cost(r.arch, r.imgsz) / target)) + (r.amp === amp ? 0 : 10) + Math.abs(Math.log(r.batch / c.batch)) * 0.01;
      const near = [...rows].sort((a, b) => dist(a) - dist(b))[0];
      if (near) {
        row = near;
        factor = refTrainRate(c.arch, c.imgsz, amp, cpu) / refTrainRate(near.arch, near.imgsz, near.amp, cpu);
        const ratio = Math.max(target / cost(near.arch, near.imgsz), cost(near.arch, near.imgsz) / target);
        if (ratio <= 4 && near.amp === amp) {
          confidence = 'medium';
          spread = [-0.25, 0.4];
        } else {
          confidence = 'low';
          spread = [-0.4, 0.8];
        }
        note = `carried over from ${near.arch} at ${near.imgsz} px, batch ${near.batch}${near.amp ? ', amp' : ''}, measured on this device, by the reference's ratio`;
      } else {
        row = refTrainRow(device, c.arch, c.imgsz, amp);
        source = 'reference';
        confidence = 'low';
        // The mock's GPU is the reference machine's; its CPU is not.
        spread = referenceSpread(!cpu);
        note = cpu ? 'reference figures, not measured on this device' : 'reference figures (docs/BENCHMARKS.md) for this GPU model';
      }
    }

    const mosaicRate = row.mosaic!.imgs_per_s * factor;
    // A phase the row has not measured comes from the reference, as the server does.
    const ref = referenceRates(c.arch, c.imgsz, amp, cpu);
    const closeRate = row.close_mosaic ? row.close_mosaic.imgs_per_s * factor : ref.close;
    const valRow = row.val;
    const valRate = valRow ? valRow.imgs_per_s * factor : ref.val;
    const startup = row.startup_s ?? REF_STARTUP_S;
    const closeN = c.augment.mosaic <= 0 ? c.epochs : Math.max(0, Math.min(c.augment.close_mosaic, Math.floor(c.epochs / 2)));
    const val = valImages > 0 ? valImages : trainImages;
    const phase = (p: EstimatePhase['phase'], epochs: number, r: number): EstimatePhase => {
      const train_s = trainImages / r;
      const val_s = val / valRate;
      return { phase: p, epochs, train_imgs_per_s: r1(r), train_s: r2(train_s), val_s: r2(val_s), epoch_s: r2(train_s + val_s) };
    };
    const phases = [
      ...(c.epochs - closeN > 0 ? [phase('mosaic', c.epochs - closeN, mosaicRate)] : []),
      ...(closeN > 0 ? [phase('close_mosaic', closeN, closeRate)] : []),
    ];
    const total = startup + phases.reduce((s, p) => s + p.epochs * p.epoch_s, 0);

    // Peak memory: a measured peak with the run's device, arch, imgsz and amp at the nearest batch, scaled to the
    // run's batch by the reference model; else the reference model.
    let vram: number | null = null;
    let vramSource: ThroughputSource | null = null;
    if (!cpu) {
      const peaks = rows.filter((r) => r.arch === c.arch && r.imgsz === c.imgsz && r.amp === amp && r.peak_vram_bytes);
      const p = peaks.sort((a, b) => Math.abs(a.batch - c.batch) - Math.abs(b.batch - c.batch))[0];
      if (p) {
        vram = p.peak_vram_bytes! * (refVram(c.arch, c.imgsz, c.batch, amp) / refVram(c.arch, c.imgsz, p.batch, amp));
        vramSource = 'measured';
      } else {
        vram = refVram(c.arch, c.imgsz, c.batch, amp);
        vramSource = 'reference';
      }
    }

    const basis: EstimateBasis = {
      source,
      arch: row.arch,
      imgsz: row.imgsz,
      batch: row.batch,
      amp: row.amp,
      measured_on: row.measured_on,
      samples: row.samples,
      updated: row.updated,
      exact: !!exact,
      note,
    };
    return {
      device,
      device_name: here,
      train_images: trainImages,
      val_images: val,
      batches_per_epoch: Math.ceil(trainImages / c.batch),
      epochs: c.epochs,
      phases,
      val_imgs_per_s: r1(valRate),
      startup_s: startup,
      est_epoch_s: r2((total - startup) / c.epochs),
      est_total_s: r1(total),
      est_total_s_low: r1(total * (1 + spread[0])),
      est_total_s_high: r1(total * (1 + spread[1])),
      source,
      confidence,
      basis,
      val_source: valRow && valRow.samples > 0 ? 'measured' : 'reference',
      est_peak_vram_bytes: vram === null ? null : Math.round(vram),
      vram_source: vramSource,
      // The trainer's cap at half the run, whatever the mosaic probability (as `Run.resolved.close_mosaic`).
      close_mosaic: Math.max(0, Math.min(c.augment.close_mosaic, Math.floor(c.epochs / 2))),
    };
  }

  /** Adds a probe's two samples to the table and returns the rows after it (`BenchmarkReport`). */
  probe(req: BenchmarkRequest, now: number, elapsedS: number) {
    const arch = req.arch ?? 'folo-n';
    const imgsz = req.imgsz ?? 320;
    const batch = req.batch ?? 16;
    const amp = req.amp ?? false;
    const budget = req.budget_s ?? 4;
    if (imgsz % 32 !== 0 || imgsz < 64 || imgsz > 1280) throw new ApiError(400, 'invalid_body', '`imgsz` must be a multiple of 32 in 64..=1280.');
    if (batch < 1 || batch > 64) throw new ApiError(400, 'invalid_body', '`batch` must be in 1..=64.');
    if (budget < 1 || budget > 10) throw new ApiError(400, 'invalid_body', '`budget_s` must be in 1..=10.');
    const key = { device: 'gpu:0', measured_on: GPU_NAME, arch, imgsz, batch, amp };
    let t = this.#training.find((r) => sameTrain(r, key));
    // Near the reference, a little different each probe (deterministic from the sample count).
    const n = (t?.samples ?? 0) + 1;
    const jitter = 1 + 0.03 * Math.sin(n * 2.1);
    const trainRate = refTrainRate(arch, imgsz, amp, false) * jitter;
    const updated = iso(now);
    if (!t) {
      t = { ...refTrainRow('gpu:0', arch, imgsz, amp), batch, source: 'measured', samples: 0, probes: 0, note: null, startup_s: null, close_mosaic: null, val: null };
      this.#training.unshift(t);
    }
    const prev = t.mosaic && t.samples > 0 ? t.mosaic : null;
    const median = prev ? (prev.imgs_per_s * prev.samples + trainRate) / (prev.samples + 1) : trainRate;
    t.mosaic = {
      imgs_per_s: r1(median),
      low: r1(Math.min(prev?.low ?? trainRate, trainRate)),
      high: r1(Math.max(prev?.high ?? trainRate, trainRate)),
      samples: (prev?.samples ?? 0) + 1,
    };
    t.imgs_per_s = t.mosaic.imgs_per_s;
    t.samples += 1;
    t.probes += 1;
    t.updated = updated;

    const precision = amp ? 'f16' : 'f32';
    const ikey = { device: 'gpu:0', measured_on: GPU_NAME, arch, imgsz, precision } as const;
    let inf = this.#inference.find((r) => sameInfer(r, ikey));
    const lat = refLatency(arch, imgsz, amp, false) * (2 - jitter);
    if (!inf) {
      inf = { ...refInferRow('gpu:0', arch, imgsz, amp), source: 'measured', samples: 0, note: null, latency_ms_low: r2(lat), latency_ms_high: r2(lat) };
      this.#inference.unshift(inf);
    }
    const predictions = Math.round(((budget * 0.4) / lat) * 1000);
    const prevLat = inf.samples > 0 ? inf.latency_ms : lat;
    inf.latency_ms = r2((prevLat * inf.samples + lat * predictions) / (inf.samples + predictions));
    inf.latency_ms_low = r2(Math.min(inf.latency_ms_low, lat * 0.94));
    inf.latency_ms_high = r2(Math.max(inf.latency_ms_high, lat * 1.08));
    inf.imgs_per_s = r1(1000 / inf.latency_ms);
    inf.samples += predictions;
    inf.updated = updated;

    return {
      device: 'gpu:0',
      device_name: GPU_NAME,
      train_imgs_per_s: r1(trainRate),
      train_batches: Math.round((budget * 0.6 * 0.7 * trainRate) / batch),
      latency_ms: r2(lat),
      predictions,
      elapsed_s: r1(elapsedS),
      training: { ...t },
      inference: { ...inf },
    };
  }
}
