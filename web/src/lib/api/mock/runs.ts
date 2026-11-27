// Mock training runs. A run's whole history is a pure function of its profile and the clock, so the live run can
// be reconstructed at any instant (page reloads, screenshots with a frozen clock) and its SSE stream simply reports
// the state at "now" as time passes.

import type { ClassMetrics, Epoch, LogLine, Loss, ResolvedTrain, Run, RunStatus, Step, TrainConfig } from '../types';
import { closeMosaicEpochs } from '../progress';
import { COLLECT_MS, RELEASE_MS, releasedAt, runCloudAt, trainAt, type CloudTimeline } from './cloud';
import { gauss, hash, rng, smoothNoise } from './rng';

export interface RunProfile {
  id: string;
  name: string;
  config: TrainConfig;
  classes: number;
  /** Class names, snapshotted at creation so models keep them if the dataset is unregistered. */
  classNames: string[];
  /** The dataset's name, recorded the first time the run is read (`Run.dataset`). */
  datasetName?: string;
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
    /** `Run.error_code` of a failed run, where it has one (`cloud_instance_lost`). */
    errorCode?: string;
    /** When it ended (defaults to start + atEpoch epochs). */
    at?: number;
    /** Report `stopping` until this time (a stop was just requested). */
    stoppingUntil?: number;
  };
  /** Loop forever (the demo live run). */
  loop?: boolean;
  modelId?: string;
  device: string;
  /** What the trainer resolves `config` to ([`resolveTrain`]); `Run.resolved` once the run has started. */
  resolved?: ResolvedTrain;
  /** Stopped while it waited: its trainer was never built, so it has no `resolved`. */
  neverStarted?: boolean;
  /**
   * A cloud run (`config.compute`): its hand-over around the training (./cloud). `startedAt` is when the worker starts
   * training; the run's status waits after its last epoch while the weights are collected and the instance released.
   */
  cloud?: CloudTimeline;
  /** Per-run spot preemption retry budget (`0..=20`, default 3) for cloud runs. */
  maxPreemptRetries?: number;
}

export const GPU_NAME = 'NVIDIA GeForce RTX 4080 SUPER';

export function totalSeconds(p: Pick<RunProfile, 'config' | 'epochSeconds'>) {
  return p.config.epochs * p.epochSeconds;
}

/** What a run's timing is made of. */
type Timing = Pick<RunProfile, 'createdAt' | 'startedAt' | 'terminal' | 'epochSeconds' | 'config' | 'cloud'>;

/** When a run's training ends if nothing stops it: its last epoch (a finished run's recorded end). */
export function trainEnd(p: Timing): number {
  // A cloud run's training starts when its hand-over is done, also when it was stopped before (and never started).
  const start = p.cloud ? trainAt(p.cloud) : (p.startedAt ?? p.createdAt);
  const t = p.terminal;
  if (!t) return start + totalSeconds(p) * 1000;
  // A finished cloud run's recorded end includes the release; its training ended at its last epoch.
  return start + t.atEpoch * p.epochSeconds * 1000;
}

/** A cloud run's `Run.cloud` at `now`. */
export const cloudOf = (p: Timing & { cloud: CloudTimeline }, now: number) => runCloudAt(p.cloud, trainEnd(p), p.epochSeconds * 1000, now);

/** When a cloud run is done at the provider: a finished one's recorded end (`terminal.at`). */
export const cloudFinish = (p: Timing & { cloud: CloudTimeline }) => releasedAt(p.cloud, trainEnd(p));

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
  if (elapsed >= total) {
    // A cloud run's final status comes once its weights are home and its instance released (docs/CLOUD.md).
    const end = p.startedAt + total * 1000 + (p.cloud ? COLLECT_MS + RELEASE_MS : 0);
    if (now < end) return { done: p.config.epochs, status: 'running', finishedAt: null };
    return { done: p.config.epochs, status: 'completed', finishedAt: end };
  }
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

/** The LR the resolved schedule gives at fractional epoch `epochF`: warmup, then linear (or cosine) decay to `lrf`. */
export function lrAt(p: RunProfile, epochF: number): number {
  const r = p.resolved;
  const warm = r ? r.warmup_iters / Math.max(1, r.optimizer_steps_per_epoch) : 3;
  const lr0 = r?.lr0 ?? p.config.lr0;
  const lrf = r?.lrf ?? 0.01;
  if (epochF < warm) return lr0 * (0.1 + 0.9 * (epochF / warm));
  const t = Math.min(1, epochF / Math.max(1, p.config.epochs));
  const decay = r?.cos_lr ? 0.5 * (1 + Math.cos(Math.PI * t)) : 1 - t;
  return lr0 * (lrf + (1 - lrf) * decay);
}

/** The epochs the trainer closes mosaic for: the request, capped at half the run. */
const closeEpochs = (p: RunProfile) => closeMosaicEpochs(p.config.augment.close_mosaic, p.config.epochs);

/** Smooth training loss at fractional epoch, plus step noise. */
export function lossAt(p: RunProfile, epochF: number, noisy: boolean): Loss {
  const s = seedOf(p);
  const closing = Math.max(0, epochF - (p.config.epochs - closeEpochs(p)));
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
  const closingBoost = Math.max(0, e - (p.config.epochs - closeEpochs(p))) * 0.0022;
  const wobble = 0.012 * smoothNoise(seedOf(p) + 11, e / 2.5) + 0.006 * (rng(seedOf(p) + e * 31)() - 0.5);
  return Math.max(0.001, Math.min(0.97, p.mapMax * Math.pow(x, 1.35) + closingBoost + wobble * x));
}

/**
 * A run's validation after epoch `e + 1`. As folo-server's mock (and the engine, docs/METRICS.md) reports them, the
 * overall numbers are class means: each class's mAP50-95 deviates from the epoch's by a zero-mean factor (its
 * `classEase` re-centred on 1, plus a little noise), scaled down where a class would pass 0.98 or drop under 0.001 so
 * no clamp moves the mean; mAP50, precision and recall are the means of the classes' own (`mock.test.ts` checks it).
 */
export function epochAt(p: RunProfile, e: number): Epoch {
  const m = mapCurve(p, e);
  const r = rng(seedOf(p) + e * 101);
  const n = p.classes;
  const easeMean = Array.from({ length: n }, (_, c) => p.classEase[c] ?? 1).reduce((a, b) => a + b, 0) / Math.max(1, n);
  const noise = Array.from({ length: n }, (_, c) => rng(seedOf(p) + e * 101 + c * 7));
  const raw = noise.map((cr, c) => (p.classEase[c] ?? 1) / easeMean - 1 + (cr() - 0.5) * 0.03);
  const rawMean = raw.reduce((a, b) => a + b, 0) / Math.max(1, n);
  const dev = raw.map((d) => d - rawMean);
  const room = dev.reduce((k, d) => (d > 0 ? Math.min(k, (0.98 - m) / (m * d)) : d < 0 ? Math.min(k, (1 - 0.001 / m) / -d) : k), 1);
  const per_class: ClassMetrics[] = dev.map((d, c) => {
    const cr = noise[c]!;
    const cm = m * (1 + room * d);
    return {
      class: c,
      map50: r4(Math.min(0.995, cm * 1.28 + 0.03)),
      map50_95: r4(cm),
      precision: r4(Math.min(0.995, 0.24 + cm * 1.02 + (cr() - 0.5) * 0.03)),
      recall: r4(Math.min(0.99, 0.16 + cm * 1.0 + (cr() - 0.5) * 0.03)),
    };
  });
  // Without classes, the old overall curves (their draws also keep the epoch durations where they were).
  const flat = { map50: Math.min(0.995, m * 1.3 + 0.035 * (1 - m) + (r() - 0.5) * 0.008), precision: Math.min(0.995, 0.25 + m * 1.02 + (r() - 0.5) * 0.02), recall: Math.min(0.99, 0.18 + m * 1.0 + (r() - 0.5) * 0.02) };
  const mean = (k: 'map50' | 'precision' | 'recall') => (n ? per_class.reduce((a, c) => a + c[k], 0) / n : flat[k]);
  const map50 = mean('map50');
  const precision = mean('precision');
  const recall = mean('recall');
  // The best-F1 confidence on the 0.001 grid, as folo-server's mock derives it: low while the model is unsure,
  // settling near 0.3-0.45.
  const best_f1_conf = r3(Math.max(0.001, Math.min(0.999, 0.12 + 0.35 * map50 + 0.02 * smoothNoise(seedOf(p) + 22, e))));
  return {
    epoch: e + 1,
    train_loss: lossAt(p, e + 1, false),
    val: { map50: r4(map50), map50_95: r4(m), precision: r4(precision), recall: r4(recall), per_class, best_f1_conf },
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

const r3 = (x: number) => Math.round(x * 1000) / 1000;
const r4 = (x: number) => Math.round(x * 10000) / 10000;
const r6 = (x: number) => Math.round(x * 1e6) / 1e6;

// ---------- run snapshots ----------

export function toRun(p: RunProfile, now: number): Run {
  const { done, status, finishedAt } = progressAt(p, now);
  // A cloud run starts when its worker trains; until then its timeline's start is still to come.
  const started = p.startedAt === null || (p.cloud && now < p.startedAt) ? null : loopStart(p, now);
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
    // A cloud run's device is the worker's GPU once it trains (docs/API.md, RunCloud).
    device: p.cloud && (p.startedAt === null || now < p.startedAt) ? '' : p.device,
    model_id: status === 'completed' || status === 'stopped' ? (p.modelId ?? null) : null,
    error: p.terminal?.error ?? null,
    ...(p.terminal?.errorCode ? { error_code: p.terminal.errorCode } : {}),
    ...(p.cloud ? { cloud: cloudOf({ ...p, cloud: p.cloud }, now) } : {}),
    // Set once the trainer is built (by the time the run is `running`); never for a run stopped while it waited.
    resolved: p.resolved && started !== null && now >= started && !p.neverStarted ? p.resolved : null,
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
  if (ep.epoch === p.config.epochs - closeEpochs(p))
    lines.push({ t, level: 'info', message: `closing mosaic: augmentation off for the last ${closeEpochs(p)} epochs` });
  if ((ep.epoch - 1) % 17 === 9) lines.push({ t, level: 'warn', message: `data loader stalled 140 ms waiting for decode workers (8 threads); consider --workers 12` });
  return lines;
}

/** Steps a run from scratch needs before the trainer stops warning (folo_vision::train::SHORT_RUN_STEPS). */
const SHORT_RUN_STEPS = 200;

/** Batches accumulated into one optimizer step: `round(64 / batch)`, at least 1 (docs/MODEL.md §8.3). */
export const accumulate = (batch: number) => Math.max(1, Math.round(64 / Math.max(1, batch)));

/** The trainer's optimizer steps an epoch: `ceil(ceil(images / batch) / accumulate)`. */
export const optimizerSteps = (images: number, batch: number) =>
  Math.max(1, Math.ceil(Math.ceil(images / Math.max(1, batch)) / accumulate(batch)));

/**
 * Warmup in optimizer steps (folo_nn::warmup_iters_floor with a new run's `warmup_floor` 0, 3 epochs): `warmup_epochs`
 * epochs of steps, at most every epoch but the last.
 */
export function warmupIters(epochs: number, stepsPerEpoch: number, warmupEpochs = 3): number {
  return Math.min(Math.round(warmupEpochs * stepsPerEpoch), Math.max(epochs - 1, 0) * stepsPerEpoch);
}

/**
 * `Run.resolved` as the real trainer resolves `config` on `images` training images of `classes` classes (folo-vision
 * `TrainConfig::resolved_train`, docs/MODEL.md §8.2): `auto` (rule 5) is MuSGD at 0.01 for every run, with momentum
 * 0.937 and warmup bias LR 0.1; a named optimizer keeps `lr0` and the same momentum and bias LR. `stepsPerEpoch` is
 * the run's own count, so the two never disagree.
 */
export function resolveTrain(c: TrainConfig, images: number, classes: number, stepsPerEpoch: number): ResolvedTrain {
  const nbs = 64;
  const acc = accumulate(c.batch);
  // docs/MODEL.md §8.2, `auto_rule` 5: MuSGD at 0.01 for every run, at step weights 0.5 / 0.5 from scratch.
  const kind: ResolvedTrain['optimizer'] = c.optimizer === 'auto' ? 'musgd' : c.optimizer;
  const lr0 = c.optimizer === 'auto' ? 0.01 : c.lr0;
  const momentum = 0.937;
  const warmupBias = 0.1;
  // The EMA (docs/MODEL.md §8.5, `folo_nn::ema_for_run`): the default 0.9999 / 2000 from scratch; a fine-tune averages
  // over a quarter of its optimizer steps (`ema_run_fraction` 0.25) until that reaches the default's 10,000.
  const window = 0.25 * c.epochs * stepsPerEpoch;
  const [emaDecay, emaTau] = !c.weights || window >= 10_000 ? [0.9999, 2000] : [1 - 1 / Math.max(window, 1), Math.max(Math.max(window, 1) / 5, 1)];
  return {
    optimizer: kind,
    lr0,
    lrf: 0.01,
    cos_lr: false,
    momentum,
    betas: kind === 'adamw' ? [momentum, 0.999] : null,
    weight_decay: r6((0.0005 * c.batch * acc) / nbs),
    batch: c.batch,
    accumulate: acc,
    nbs,
    optimizer_steps_per_epoch: stepsPerEpoch,
    warmup_iters: warmupIters(c.epochs, stepsPerEpoch),
    warmup_epochs: 3,
    warmup_momentum: 0.8,
    warmup_bias_lr: warmupBias,
    // After the trainer's cap at half the run (docs/API.md, `Run.resolved.close_mosaic`).
    close_mosaic: closeMosaicEpochs(c.augment.close_mosaic, c.epochs),
    ema_decay: emaDecay,
    ema_tau: emaTau,
    ...(kind === 'musgd' ? { musgd: { ...(c.optimizer === 'auto' && !c.weights ? { muon: 0.5, sgd: 0.5 } : { muon: 0.2, sgd: 1 }), cls_lr: 3 } } : {}),
  };
}

/**
 * The trainer's schedule warnings for a run, worded as folo_vision::train::schedule_warnings words them (docs/MODEL.md
 * §8.4): a run from scratch shorter than 200 optimizer steps, and a warmup longer than half the run.
 */
export function scheduleWarnings(p: RunProfile, images: number): string[] {
  const c = p.config;
  const spe = p.stepsPerEpoch;
  const total = c.epochs * spe;
  const warmup = warmupIters(c.epochs, spe);
  const out: string[] = [];
  const acc = accumulate(c.batch);
  if (!c.weights && total < SHORT_RUN_STEPS)
    out.push(
      `this run takes only ${total} optimizer steps (${c.epochs} epochs × ${spe}: ${images} training images, ${acc > 1 ? `batch ${c.batch} × accumulate ${acc}` : `batch ${c.batch}`}); ` +
        'a detector trained from scratch usually needs a few hundred steps to learn anything, so expect an mAP near 0. ' +
        `Train for at least ${Math.ceil(SHORT_RUN_STEPS / Math.max(1, spe))} epochs, add images, or start from pretrained weights`,
    );
  if (warmup * 2 > total)
    out.push(
      `warmup takes ${warmup} of the run's ${total} optimizer steps, so the learning rate reaches lr0 only in the second half of the run; ` +
        'lower warmup_epochs (now 3) or train for more epochs',
    );
  return out;
}

export function startLogs(p: RunProfile, startedAt: number, datasetName: string, images: number): LogLine[] {
  const t = (ms: number) => new Date(startedAt + ms).toISOString();
  const c = p.config;
  return [
    { t: t(0), level: 'info', message: `FOLO 0.1.0  ${c.arch}  device ${p.device}` },
    // Straight after the first line, as the trainer writes them.
    ...scheduleWarnings(p, images).map((message): LogLine => ({ t: t(0), level: 'warn', message })),
    // folo_vision::train: a close_mosaic over half of the run is capped (folo-data `close_mosaic_epochs`).
    ...(c.augment.mosaic > 0 && c.augment.close_mosaic > closeEpochs(p)
      ? [
          {
            t: t(0),
            level: 'info',
            message: `close_mosaic ${c.augment.close_mosaic} is more than half of the ${c.epochs} epochs: mosaic closes for the last ${closeEpochs(p)} instead`,
          } satisfies LogLine,
        ]
      : []),
    { t: t(40), level: 'info', message: `dataset ${datasetName}: ${images} train images, ${p.classes} classes, imgsz ${c.imgsz}` },
    { t: t(90), level: 'debug', message: `model ${c.arch}: params ${archInfo[c.arch].params.toLocaleString('en-US')}, ${archInfo[c.arch].flops} GFLOPs` },
    ...(p.resolved
      ? [
          {
            t: t(120),
            level: 'info',
            message: `optimizer ${({ sgd: 'SGD', adamw: 'AdamW', musgd: 'MuSGD' } as const)[p.resolved.optimizer]}${c.optimizer === 'auto' ? ' (auto)' : ''} lr0 ${p.resolved.lr0}  batch ${c.batch} × accumulate ${p.resolved.accumulate}  warmup ${p.resolved.warmup_iters} steps, ${p.resolved.cos_lr ? 'cosine' : 'linear'} to ${r6(p.resolved.lr0 * p.resolved.lrf)}`,
          } satisfies LogLine,
        ]
      : []),
    { t: t(200), level: 'info', message: `augment mosaic ${c.augment.mosaic} mixup ${c.augment.mixup} fliplr ${c.augment.fliplr} scale ${c.augment.scale}` },
    { t: t(260), level: 'info', message: `training ${c.epochs} epochs, ${p.stepsPerEpoch} steps/epoch` },
    // folo_vision::train, for a `head: "o2o"` run (docs/MODEL.md §6.6).
    ...(c.head === 'o2o'
      ? [
          {
            t: t(260),
            level: 'info',
            message:
              'head o2o: dual assignment (one-to-many + one-to-one losses, weights 0.8 / 0.2 → 0.1 / 0.9 over the run); validation uses the one-to-one head without NMS',
          } satisfies LogLine,
        ]
      : []),
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
  'folo2-n': { params: 2_850_000, flops: 6.9, ms: 1.5 },
  'folo2-s': { params: 9_850_000, flops: 22.4, ms: 2.5 },
  'folo2-m': { params: 20_800_000, flops: 70.5, ms: 4.9 },
  'folo2-l': { params: 26_200_000, flops: 90.0, ms: 6.5 },
  'folo2-x': { params: 58_800_000, flops: 201.5, ms: 11.9 },
} as const;
