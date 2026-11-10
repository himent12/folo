// The demo's runs as the mock starts with them (docs/API.md, Mock mode): their configs and timelines, the cloud
// runs among them, and what each resolves its config to.

import type { Arch, TrainConfig } from '../types';
import { specs } from './datasets';
import { GPU_NAME, cloudFinish, optimizerSteps, resolveTrain, type RunProfile } from './runs';
import { RTX_4090, trainAt, type CloudTimeline } from './cloud';
import type { Scenario } from './scenario';

const MIN = 60_000;
const HOUR = 60 * MIN;
export const DAY = 24 * HOUR;

const defaultAugment = { mosaic: 1, mixup: 0, hsv: [0.015, 0.7, 0.4] as [number, number, number], fliplr: 0.5, scale: 0.5, translate: 0.1, close_mosaic: 10 };

function config(name: string, dataset_id: string, arch: Arch, epochs: number, extra: Partial<TrainConfig> = {}): TrainConfig {
  return {
    name,
    dataset_id,
    arch,
    epochs,
    batch: 16,
    imgsz: 640,
    device: 'gpu:0',
    optimizer: 'auto',
    lr0: 0.01,
    weights: null,
    augment: { ...defaultAugment },
    seed: 0,
    ...extra,
  };
}

export const DEVICE = `${GPU_NAME} (vulkan)`;

const FASTENER_EASE = [1.05, 1.1, 1.12, 0.92, 0.78, 1.0, 0.85, 0.7];

export function profiles(t0: number, scenario: Scenario = 'normal'): RunProfile[] {
  const live = t0 - 222_000;
  const names = (id: string) => [...(specs.find((s) => s.id === id)?.classes ?? [])];
  const list: Omit<RunProfile, 'classNames'>[] = [
    {
      id: 'run_q4n2',
      name: 'shapes-hard-l',
      config: config('shapes-hard-l', 'ds_shapes_hard', 'folo-l', 120, { batch: 32 }),
      classes: 6,
      classEase: [1.1, 1.05, 1, 0.82, 0.95, 0.75],
      stepsPerEpoch: 125,
      epochSeconds: 30,
      mapMax: 0.7,
      tau: 16,
      imgsPerS: 140,
      createdAt: t0 - 40_000,
      startedAt: null,
      device: DEVICE,
    },
    {
      // Second in the training queue, behind shapes-hard-l: moving it means something.
      id: 'run_q7c5',
      name: 'fasteners-m',
      config: config('fasteners-m', 'ds_fasteners', 'folo-m', 80),
      classes: 8,
      classEase: FASTENER_EASE,
      stepsPerEpoch: 120,
      epochSeconds: 14,
      mapMax: 0.69,
      tau: 15,
      imgsPerS: 170,
      createdAt: t0 - 25_000,
      startedAt: null,
      device: DEVICE,
    },
    {
      id: 'run_7d1e',
      name: 'fasteners-s',
      config: config('fasteners-s', 'ds_fasteners', 'folo-s', 100, { weights: 'm_coco_s' }),
      classes: 8,
      classEase: FASTENER_EASE,
      stepsPerEpoch: 120,
      epochSeconds: 6,
      mapMax: 0.66,
      tau: 14,
      imgsPerS: 318,
      createdAt: live - 5_000,
      startedAt: live,
      loop: true,
      modelId: 'm_fast_s',
      device: DEVICE,
    },
    {
      id: 'run_c19a',
      name: 'fasteners-n',
      // Fine-tuned from shapes-n (the Model page's lineage line), in mixed precision (as the server mock's m_53d1).
      config: config('fasteners-n', 'ds_fasteners', 'folo-n', 150, { weights: 'm_shapes_n', amp: true }),
      classes: 8,
      classEase: FASTENER_EASE,
      stepsPerEpoch: 120,
      epochSeconds: 21.4,
      mapMax: 0.58,
      tau: 18,
      imgsPerS: 512,
      createdAt: t0 - DAY - 2 * HOUR,
      startedAt: t0 - DAY - 2 * HOUR + 4_000,
      terminal: { status: 'completed', atEpoch: 150 },
      modelId: 'm_fast_n',
      device: DEVICE,
    },
    {
      // A short fine-tune (DOGFOOD-3 S2, S4): 15 epochs ask for the default close mosaic 10, which the trainer caps at
      // 7 (half the run); in mixed precision with the end-to-end head, so the run's header says both.
      id: 'run_f15c',
      name: 'fasteners-n-ft',
      config: config('fasteners-n-ft', 'ds_fasteners', 'folo-n', 15, { weights: 'm_fast_n', amp: true, head: 'o2o' }),
      classes: 8,
      classEase: FASTENER_EASE,
      stepsPerEpoch: 120,
      epochSeconds: 21.4,
      mapMax: 0.6,
      tau: 4,
      imgsPerS: 90,
      createdAt: t0 - DAY - 4 * HOUR,
      startedAt: t0 - DAY - 4 * HOUR + 3_000,
      terminal: { status: 'completed', atEpoch: 15 },
      device: DEVICE,
    },
    {
      id: 'run_a2f0',
      name: 'shapes-hard-m',
      config: config('shapes-hard-m', 'ds_shapes_hard', 'folo-m', 100, { batch: 32, imgsz: 960, augment: { ...defaultAugment, close_mosaic: 15 } }),
      classes: 6,
      classEase: [1.1, 1.05, 1, 0.82, 0.95, 0.75],
      stepsPerEpoch: 125,
      epochSeconds: 41,
      mapMax: 0.68,
      tau: 15,
      imgsPerS: 98,
      createdAt: t0 - DAY - 6 * HOUR,
      startedAt: t0 - DAY - 6 * HOUR + 3_000,
      terminal: {
        status: 'failed',
        atEpoch: 23.4,
        // As folo-vision's `Error::summary` words it: the first line of the driver's report in parentheses.
        error: 'the GPU ran out of memory (wgpu backend: Out of Memory)',
      },
      device: DEVICE,
    },
    {
      id: 'run_5be3',
      name: 'shapes-hard-s',
      // Fine-tuned from shapes-s: Compare's Start from row and legend show it next to the runs from scratch.
      config: config('shapes-hard-s', 'ds_shapes_hard', 'folo-s', 100, { weights: 'm_shapes_s' }),
      classes: 6,
      classEase: [1.1, 1.05, 1, 0.82, 0.95, 0.75],
      stepsPerEpoch: 250,
      epochSeconds: 19,
      mapMax: 0.61,
      tau: 13,
      imgsPerS: 330,
      createdAt: t0 - 2 * DAY - 3 * HOUR,
      startedAt: t0 - 2 * DAY - 3 * HOUR + 2_000,
      terminal: { status: 'stopped', atEpoch: 61 },
      modelId: 'm_hard_s',
      device: DEVICE,
    },
    {
      id: 'run_0e77',
      name: 'shapes-s',
      // The NMS-free one-to-one head (as the server mock's m_31ab, also shapes-s-best).
      config: config('shapes-s', 'ds_shapes', 'folo-s', 100, { head: 'o2o' }),
      classes: 3,
      classEase: [1.04, 1.0, 0.95],
      stepsPerEpoch: 50,
      epochSeconds: 8.2,
      mapMax: 0.83,
      tau: 9,
      imgsPerS: 335,
      createdAt: t0 - 11 * DAY,
      startedAt: t0 - 11 * DAY + 2_000,
      terminal: { status: 'completed', atEpoch: 100 },
      modelId: 'm_shapes_s',
      device: DEVICE,
    },
    {
      id: 'run_31aa',
      name: 'shapes-n',
      config: config('shapes-n', 'ds_shapes', 'folo-n', 100),
      classes: 3,
      classEase: [1.04, 1.0, 0.95],
      stepsPerEpoch: 50,
      epochSeconds: 5.1,
      mapMax: 0.77,
      tau: 10,
      imgsPerS: 540,
      createdAt: t0 - 12 * DAY,
      startedAt: t0 - 12 * DAY + 2_000,
      terminal: { status: 'completed', atEpoch: 100 },
      modelId: 'm_shapes_n',
      device: 'cpu (AMD Ryzen 9 7950X)',
    },
    // `short-run`: 800 images at batch 32 are 25 batches an epoch, 2 accumulated per optimizer step (docs/MODEL.md
    // §8.3), so 13 steps an epoch and 65 in 5 epochs: from scratch, the trainer warns it is too short (and its warmup
    // too long). New run's `run-new-short` scene asks for the same run and suggests the same 16 epochs.
    ...(scenario === 'short-run'
      ? [
          {
            id: 'run_s5q1',
            name: 'shapes-quick',
            config: config('shapes-quick', 'ds_shapes', 'folo-n', 5, { batch: 32 }),
            classes: 3,
            classEase: [1.04, 1.0, 0.95],
            stepsPerEpoch: optimizerSteps(800, 32),
            epochSeconds: 6,
            mapMax: 0.3,
            tau: 40,
            imgsPerS: 540,
            createdAt: t0 - 2 * HOUR,
            startedAt: t0 - 2 * HOUR + 2_000,
            terminal: { status: 'completed' as const, atEpoch: 5 },
            device: DEVICE,
          },
        ]
      : []),
  ];
  // `finishing`: the run in the training slot completes 3 s from now. It is a run of its own, shapes-hard-n, not the
  // demo's live fasteners-s (epoch 38 of 100 on every other screen), so the toast that says it finished, with its real
  // best epoch, contradicts nothing another shot shows; fasteners-s is not in this scenario.
  const finishing: Omit<RunProfile, 'classNames'> = {
    id: 'run_e6b9',
    name: 'shapes-hard-n',
    config: config('shapes-hard-n', 'ds_shapes_hard', 'folo-n', 60),
    classes: 6,
    classEase: [1.1, 1.05, 1, 0.82, 0.95, 0.75],
    stepsPerEpoch: 250,
    epochSeconds: 6,
    mapMax: 0.52,
    tau: 12,
    imgsPerS: 520,
    createdAt: t0 - 60 * 6000 - 5000,
    startedAt: t0 - 60 * 6000 + 3000,
    device: DEVICE,
  };
  // `train-split`: a fine-tune on fasteners-mini (24 training images, none for validation), so every epoch's scores are
  // on its training split; the run page says so and names that split (the `run-train-split` scene).
  const trainSplit: Omit<RunProfile, 'classNames'> = {
    id: 'run_m1n1',
    name: 'fasteners-mini-n',
    config: config('fasteners-mini-n', 'ds_fasteners_mini', 'folo-n', 60, { weights: 'm_fast_n' }),
    classes: 2,
    classEase: [1.02, 0.97],
    stepsPerEpoch: optimizerSteps(24, 16),
    epochSeconds: 2.5,
    mapMax: 0.74,
    tau: 9,
    imgsPerS: 480,
    createdAt: t0 - 3 * HOUR,
    startedAt: t0 - 3 * HOUR + 2_000,
    terminal: { status: 'completed', atEpoch: 60 },
    device: DEVICE,
    // Its best weights, scored on the training split: Overview and Models tag the score and rank it apart.
    modelId: 'm_mini_run',
  };
  const all = [
    ...(scenario === 'finishing' ? [...list.filter((p) => p.id !== 'run_7d1e'), finishing] : scenario === 'train-split' ? [...list, trainSplit] : list),
    ...cloudRuns(t0, scenario),
  ];
  return all.map((p) => withResolved({ ...p, classNames: names(p.config.dataset_id) }));
}

/** A cloud run's timeline as the seeded runs have it: a secure-cloud RTX 4090 on the demo's connection. */
function rented(id: string, nonce: string, ref: string, launchedAt: number, ms: { provision: number; boot: number; sync: number }, upload: number, weights: number, maxHours: number): CloudTimeline {
  return {
    connectionId: 'cc_4d2a',
    provider: 'runpod',
    offer: RTX_4090,
    maxHours,
    instanceId: `ci_${nonce}`,
    instanceName: `folo-${id}-${nonce}`,
    providerRef: ref,
    launchedAt,
    provisionMs: ms.provision,
    bootMs: ms.boot,
    syncMs: ms.sync,
    uploadBytes: upload,
    weightsBytes: weights,
  };
}

/**
 * The seeded cloud runs (docs/API.md, Mock mode → Cloud): shapes-m, finished on a rented RTX 4090 two days ago (in
 * `cloud-trouble` two hours ago, its instance's termination never confirmed); `cloud-training` adds fasteners-l
 * training there, `cloud-upload` shapes-hard-x uploading its dataset.
 */
function cloudRuns(t0: number, scenario: Scenario): Omit<RunProfile, 'classNames'>[] {
  // Without the feature, or before any connection was made (`cloud-none`: a cloud never used), there are none.
  if (scenario === 'cloud-off' || scenario === 'cloud-none' || scenario === 'no-throughput' || scenario === 'empty') return [];
  const compute = (hours: number) => ({ device: 'auto' as const, compute: { connection_id: 'cc_4d2a', offer_id: RTX_4090.id, max_hours: hours } });
  const GPU_4090 = 'NVIDIA GeForce RTX 4090 (vulkan)';
  const MiB = 2 ** 20;
  const finishedAt = scenario === 'cloud-trouble' ? t0 - 2 * HOUR : t0 - 2 * DAY - 5 * HOUR;
  const done = rented('run_c3d5', '4c1a', 'k2x9wq7m3fz1', finishedAt, { provision: 40_000, boot: 3 * MIN, sync: 70_000 }, 64 * MiB, 330 * MiB, 2);
  if (scenario === 'cloud-trouble') done.stuck = 'RunPod rejected this API key';
  const finished: Omit<RunProfile, 'classNames'> = {
    id: 'run_c3d5',
    name: 'shapes-m',
    config: config('shapes-m', 'ds_shapes', 'folo-m', 100, compute(2)),
    classes: 3,
    classEase: [1.04, 1.0, 0.95],
    stepsPerEpoch: optimizerSteps(800, 16),
    epochSeconds: 9.4,
    mapMax: 0.8,
    tau: 9,
    imgsPerS: 410,
    createdAt: finishedAt,
    startedAt: trainAt(done),
    modelId: 'm_shapes_m',
    device: GPU_4090,
    cloud: done,
  };
  finished.terminal = { status: 'completed', atEpoch: 100, at: cloudFinish({ ...finished, cloud: done }) };
  const out = [finished];
  if (scenario === 'cloud-training') {
    const c = rented('run_c7b1', '9d03', 'p7m2v8x4qk1c', t0 - 40 * MIN, { provision: 40_000, boot: 3 * MIN, sync: 140_000 }, 515 * MiB, 410 * MiB, 3);
    out.push({
      id: 'run_c7b1',
      name: 'fasteners-l',
      config: config('fasteners-l', 'ds_fasteners', 'folo-l', 100, compute(3)),
      classes: 8,
      classEase: FASTENER_EASE,
      stepsPerEpoch: optimizerSteps(1920, 16),
      epochSeconds: 60,
      mapMax: 0.7,
      tau: 16,
      imgsPerS: 160,
      createdAt: c.launchedAt,
      startedAt: trainAt(c),
      device: GPU_4090,
      cloud: c,
    });
  }
  if (scenario === 'cloud-upload') {
    // A quarter of the upload is through at load: 90 s of six minutes.
    const c = rented('run_c9e4', '2b7e', 'q4w8e2r6t1y5', t0 - 3 * MIN, { provision: 30_000, boot: 60_000, sync: 6 * MIN }, 460 * MiB, 1100 * MiB, 8);
    out.push({
      id: 'run_c9e4',
      name: 'shapes-hard-x',
      config: config('shapes-hard-x', 'ds_shapes_hard', 'folo-x', 100, compute(8)),
      classes: 6,
      classEase: [1.1, 1.05, 1, 0.82, 0.95, 0.75],
      stepsPerEpoch: optimizerSteps(4000, 16),
      epochSeconds: 45,
      mapMax: 0.72,
      tau: 14,
      imgsPerS: 120,
      createdAt: c.launchedAt,
      startedAt: trainAt(c),
      device: GPU_4090,
      cloud: c,
    });
  }
  return out;
}

/** The profile with `resolved`: what the real trainer resolves its config to on its dataset's training split. */
export function withResolved(p: RunProfile, train?: number): RunProfile {
  const images = train ?? specs.find((s) => s.id === p.config.dataset_id)?.splits.train ?? 0;
  return { ...p, resolved: resolveTrain(p.config, images, p.classes, p.stepsPerEpoch) };
}
