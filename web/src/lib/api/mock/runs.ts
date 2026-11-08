// Mock training runs. A run's whole history is a pure function of its profile and the clock, so the live run can
// be reconstructed at any instant (page reloads, screenshots with a frozen clock) and its SSE stream simply reports
// the state at "now" as time passes.

import type { ClassMetrics, Epoch, LogLine, Loss, Run, RunStatus, Step, TrainConfig } from '../types';
import { gauss, hash, rng, smoothNoise } from './rng';

export interface RunProfile {
  id: string;
  name: string;
  config: TrainConfig;
  classes: number;
  /** Class names, snapshotted at creation so models keep them if the dataset is unregistered. */
  classNames: string[];
  /** Per-class difficulty multipliers (1 = average). */
  classEase: number[];
  stepsPerEpoch: number;
  /** Wall seconds per epoch (training + validation). */
  epochSeconds: number;
  /** Final mAP50-95 the curve approaches. */
  mapMax: number;
  /** Epochs to reach ~63 % of the gain. */
  tau: number;
  imgsPerS: number;
  createdAt: number;
  startedAt: number | null;
  /** Fixed terminal state for history runs; live runs derive it from the clock. */
  terminal?: {
    status: 'completed' | 'stopped' | 'failed';
    /** Fractional epochs done when it ended. */
    atEpoch: number;
    error?: string;
    /** When it ended (defaults to start + atEpoch epochs). */
    at?: number;
    /** Report `stopping` until this time (a stop was just requested). */
    stoppingUntil?: number;
  };
  /** Loop forever (the demo live run). */
  loop?: boolean;
  modelId?: string;
  device: string;
}

export const GPU_NAME = 'NVIDIA GeForce RTX 4080 SUPER';

export function totalSeconds(p: RunProfile) {
  return p.config.epochs * p.epochSeconds;
}

/** Progress of a run at `now`: fractional epochs done (0..epochs) and the status. */
export function progressAt(p: RunProfile, now: number): { done: number; status: RunStatus; finishedAt: number | null } {
  if (p.terminal) {
    const t = p.terminal;
    if (t.stoppingUntil !== undefined && now < t.stoppingUntil) return { done: t.atEpoch, status: 'stopping', finishedAt: null };
    const finishedAt = t.at ?? (p.startedAt ?? p.createdAt) + t.atEpoch * p.epochSeconds * 1000;
    return { done: t.atEpoch, status: t.status, finishedAt };
  }
  if (p.startedAt === null || now < p.startedAt) return { done: 0, status: 'queued', finishedAt: null };
  let elapsed = (now - p.startedAt) / 1000;
  const total = totalSeconds(p);
  if (p.loop) elapsed %= total;
  if (elapsed >= total) return { done: p.config.epochs, status: 'completed', finishedAt: p.startedAt + total * 1000 };
  return { done: elapsed / p.epochSeconds, status: 'running', finishedAt: null };
}

/** Start time of the current loop iteration (live demo runs restart when they finish). */
export function loopStart(p: RunProfile, now: number): number {
  if (!p.loop || p.startedAt === null) return p.startedAt ?? now;
  const total = totalSeconds(p) * 1000;
  return p.startedAt + Math.floor(Math.max(0, now - p.startedAt) / total) * total;
}

// ---------- curves ----------

const seedOf = (p: RunProfile) => hash(p.id);

export function lrAt(p: RunProfile, epochF: number): number {
  const warm = 3;
  const lr0 = p.config.lr0;
  if (epochF < warm) return lr0 * (0.1 + 0.9 * (epochF / warm));
  const t = (epochF - warm) / Math.max(1, p.config.epochs - warm);
  return lr0 * (0.01 + 0.99 * 0.5 * (1 + Math.cos(Math.PI * Math.min(1, t))));
}

/** Smooth training loss at fractional epoch, plus step noise. */
export function lossAt(p: RunProfile, epochF: number, noisy: boolean): Loss {
  const s = seedOf(p);
  const closing = Math.max(0, epochF - (p.config.epochs - p.config.augment.close_mosaic));
  const close = closing > 0 ? 1 - 0.12 * Math.min(1, closing / 2) : 1;
  const n = (k: number, amp: number) => (noisy ? 1 + amp * smoothNoise(s + k, epochF * 9) + amp * 0.6 * (rng(s + k + Math.floor(epochF * 997))() - 0.5) : 1);
  // Harder runs (lower ceiling) plateau higher; slower runs (larger tau) decay slower.
  const k = 0.7 + (0.8 - p.mapMax) * 1.6;
  const t = p.tau / 14;
  const box = (0.92 * k + 1.55 * Math.exp(-epochF / (16 * t))) * close * n(1, 0.05);
  const cls = (0.42 * k + 2.6 * Math.exp(-epochF / (9 * t))) * close * n(2, 0.07);
  const dfl = (0.9 * k + 0.62 * Math.exp(-epochF / (22 * t))) * close * n(3, 0.035);
  return { box: r4(box), cls: r4(cls), dfl: r4(dfl), total: r4(box + cls + dfl) };
}

function mapCurve(p: RunProfile, e: number): number {
  const x = 1 - Math.exp(-(e + 1) / p.tau);
  const closingBoost = Math.max(0, e - (p.config.epochs - p.config.augment.close_mosaic)) * 0.0022;
  const wobble = 0.012 * smoothNoise(seedOf(p) + 11, e / 2.5) + 0.006 * (rng(seedOf(p) + e * 31)() - 0.5);
  return Math.max(0.001, Math.min(0.97, p.mapMax * Math.pow(x, 1.35) + closingBoost + wobble * x));
}

export function epochAt(p: RunProfile, e: number): Epoch {
  const m = mapCurve(p, e);
  const r = rng(seedOf(p) + e * 101);
  const map50 = Math.min(0.995, m * 1.3 + 0.035 * (1 - m) + (r() - 0.5) * 0.008);
  const precision = Math.min(0.995, 0.25 + m * 1.02 + (r() - 0.5) * 0.02);
  const recall = Math.min(0.99, 0.18 + m * 1.0 + (r() - 0.5) * 0.02);
  const per_class: ClassMetrics[] = Array.from({ length: p.classes }, (_, c) => {
    const ease = p.classEase[c] ?? 1;
    const cr = rng(seedOf(p) + e * 101 + c * 7);
    const cm = Math.max(0.001, Math.min(0.98, m * ease + (cr() - 0.5) * 0.02));
    return {
      class: c,
      map50: r4(Math.min(0.995, cm * 1.28 + 0.03)),
      map50_95: r4(cm),
      precision: r4(Math.min(0.995, 0.24 + cm * 1.02 + (cr() - 0.5) * 0.03)),
      recall: r4(Math.min(0.99, 0.16 + cm * 1.0 + (cr() - 0.5) * 0.03)),
    };
  });
  return {
    epoch: e + 1,
    train_loss: lossAt(p, e + 1, false),
    val: { map50: r4(map50), map50_95: r4(m), precision: r4(precision), recall: r4(recall), per_class },
    duration_s: Math.round((p.epochSeconds + gauss(r) * 0.25) * 10) / 10,
  };
}

export function stepAt(p: RunProfile, globalStep: number): Step {
  const epochF = globalStep / p.stepsPerEpoch;
  const r = rng(seedOf(p) + globalStep * 13);
  const total = p.config.epochs * p.stepsPerEpoch;
  const ips = p.imgsPerS * (1 + 0.04 * smoothNoise(seedOf(p) + 5, epochF * 3) + (r() - 0.5) * 0.03);
  return {
    // 1-based; step N (1-based, global) belongs to epoch ceil(N / steps per epoch).
    epoch: Math.max(1, Math.min(p.config.epochs, Math.ceil(globalStep / p.stepsPerEpoch))),
    step: globalStep,
    lr: r6(lrAt(p, epochF)),
    loss: lossAt(p, epochF, true),
    imgs_per_s: Math.round(ips * 10) / 10,
    eta_s: Math.max(0, Math.round(((total - globalStep) / p.stepsPerEpoch) * p.epochSeconds)),
  };
}

const r4 = (x: number) => Math.round(x * 10000) / 10000;
const r6 = (x: number) => Math.round(x * 1e6) / 1e6;

// ---------- run snapshots ----------

export function toRun(p: RunProfile, now: number): Run {
  const { done, status, finishedAt } = progressAt(p, now);
  const started = p.startedAt === null ? null : loopStart(p, now);
  const epochsDone = Math.floor(done);
  const best = bestOf(p, epochsDone);
  const step = Math.floor(done * p.stepsPerEpoch);
  const running = status === 'running';
  return {
    id: p.id,
    name: p.name,
    status,
    config: p.config,
    created: new Date(p.createdAt).toISOString(),
    started: started === null ? null : new Date(started).toISOString(),
    finished: finishedAt === null ? null : new Date(finishedAt).toISOString(),
    epoch: status === 'queued' ? 0 : Math.min(p.config.epochs, running ? epochsDone + 1 : epochsDone),
    epochs: p.config.epochs,
    step,
    steps_per_epoch: p.stepsPerEpoch,
    eta_s: running ? Math.round((p.config.epochs - done) * p.epochSeconds) : null,
    best,
    device: p.device,
    model_id: status === 'completed' || status === 'stopped' ? (p.modelId ?? null) : null,
    error: p.terminal?.error ?? null,
  };
}

export function bestOf(p: RunProfile, epochsDone: number): Run['best'] {
  let best: Run['best'] = null;
  for (let e = 0; e < epochsDone; e++) {
    const m = mapCurve(p, e);
    if (!best || m > best.map50_95) best = { epoch: e + 1, map50_95: r4(m) };
  }
  return best;
}

/** Full metric history up to `now`: every `stride`-th step plus the latest one, as `/metrics?every=N` does. */
export function history(p: RunProfile, now: number, stride = 4): { steps: Step[]; epochs: Epoch[] } {
  const { done } = progressAt(p, now);
  const lastStep = Math.floor(done * p.stepsPerEpoch);
  const every = Math.max(1, Math.floor(stride));
  const steps: Step[] = [];
  if (done <= 0) return { steps, epochs: [] };
  for (let s = 0; s <= lastStep; s += every) steps.push(stepAt(p, s));
  if (lastStep % every !== 0) steps.push(stepAt(p, lastStep));
  const epochs: Epoch[] = [];
  for (let e = 0; e < Math.floor(done); e++) epochs.push(epochAt(p, e));
  return { steps, epochs };
}

// ---------- logs ----------

export function epochLog(p: RunProfile, ep: Epoch, startedAt: number): LogLine[] {
  const t = new Date(startedAt + ep.epoch * p.epochSeconds * 1000).toISOString();
  const lines: LogLine[] = [
    {
      t,
      level: 'info',
      message: `epoch ${ep.epoch}/${p.config.epochs}  loss ${ep.train_loss.total.toFixed(3)}  mAP50 ${ep.val.map50.toFixed(3)}  mAP50-95 ${ep.val.map50_95.toFixed(3)}  ${ep.duration_s.toFixed(1)}s`,
    },
  ];
  const best = bestOf(p, ep.epoch);
  if (best && best.epoch === ep.epoch) lines.push({ t, level: 'info', message: `new best mAP50-95 ${best.map50_95.toFixed(4)}, saved best.folo` });
  if (ep.epoch === p.config.epochs - p.config.augment.close_mosaic)
    lines.push({ t, level: 'info', message: `closing mosaic: augmentation off for the last ${p.config.augment.close_mosaic} epochs` });
  if ((ep.epoch - 1) % 17 === 9) lines.push({ t, level: 'warn', message: `data loader stalled 140 ms waiting for decode workers (8 threads); consider --workers 12` });
  return lines;
}

export function startLogs(p: RunProfile, startedAt: number, datasetName: string, images: number): LogLine[] {
  const t = (ms: number) => new Date(startedAt + ms).toISOString();
  const c = p.config;
  return [
    { t: t(0), level: 'info', message: `FOLO 0.1.0  ${c.arch}  device ${p.device}` },
    { t: t(40), level: 'info', message: `dataset ${datasetName}: ${images} train images, ${p.classes} classes, imgsz ${c.imgsz}` },
    { t: t(90), level: 'debug', message: `model ${c.arch}: params ${archInfo[c.arch].params.toLocaleString('en-US')}, ${archInfo[c.arch].flops} GFLOPs` },
    { t: t(120), level: 'info', message: `optimizer ${c.optimizer === 'auto' ? 'AdamW' : c.optimizer} lr0 ${c.lr0}  batch ${c.batch}  warmup 3 epochs, cosine to ${c.lr0 * 0.01}` },
    { t: t(200), level: 'info', message: `augment mosaic ${c.augment.mosaic} mixup ${c.augment.mixup} fliplr ${c.augment.fliplr} scale ${c.augment.scale}` },
    { t: t(260), level: 'info', message: `training ${c.epochs} epochs, ${p.stepsPerEpoch} steps/epoch` },
  ];
}

/** Published size/speed hints per architecture (also used by the New run form). */
// Sizes: n and s as measured in docs/BENCHMARKS.md. `ms` only paces the demo's simulated inference timings.
export const archInfo = {
  'folo-n': { params: 2_624_064, flops: 6.54, ms: 1.4 },
  'folo-s': { params: 9_458_736, flops: 21.59, ms: 2.3 },
  'folo-m': { params: 20_050_000, flops: 68.0, ms: 4.6 },
  'folo-l': { params: 25_310_000, flops: 86.9, ms: 6.1 },
  'folo-x': { params: 56_970_000, flops: 194.9, ms: 11.3 },
} as const;
