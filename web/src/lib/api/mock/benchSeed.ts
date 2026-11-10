// The demo's benchmarks (./bench), four that tell the feature's story:
// - "Sorting tray, night shift": just made from shapes-hard's validation split, no entries yet, its plan still open.
// - "Fasteners val": fasteners' validation split with the default plan and an entry of every kind and state: two FOLO
//   models with full results, an external yolo26n, Claude Code and Codex on 48 unchanged images (partial, with what
//   their calls used), a model that knows two of the eight classes, an agent that failed (not signed in), and the f16
//   export running now; and a self-improvement loop of fasteners-n-best, its first student training.
// - "Pond fish": made from a labeling job that has been deleted since (the benchmark keeps its own copy), with a plan
//   Claude Code designed for murky water, and four entries (one whose confidences never reach 90% precision).
// - "Shapes val": two self-improvement loops, one that improved over three generations and one that held.

import type { BenchContestant, Model } from '../types';
import { AGENT_NAME } from '../../label';
import { agentSkill, agentUsage, fingerprintOf, missingClasses, sampleImages, specImages, type MBench, type MEntry, type MLoop, type MockBench } from './bench';
import { catalog } from './benchCatalog';
import { designPlan } from './benchDesign';
import type { Skill } from './benchResults';
import { image, specs, type MockDatasetSpec } from './datasets';

const SEC = 1000;
const MIN = 60 * SEC;
const HOUR = 60 * MIN;
const DAY = 24 * HOUR;

/** The pond camera's images: koi, goldfish and the odd frog, from above, in murky green water (./render's `pond`). */
export const POND: MockDatasetSpec = {
  id: 'bench_pond',
  name: 'pond-cam',
  path: '',
  format: 'yolo',
  classes: ['koi', 'goldfish', 'frog'],
  classWeights: [7, 9, 2],
  classShape: ['long', 'long', 'round'],
  splits: { train: 240, val: 0, test: 0 },
  width: 960,
  height: 720,
  meanBoxes: 4.2,
  emptyRate: 0.03,
  sizeRange: [0.1, 0.3],
  brokenLabels: 0,
  createdAgoS: 0,
  style: 'pond',
  seed: 4242,
};

export const POND_DESCRIPTION =
  'A camera over a garden pond, looking down through the water at koi, goldfish and the odd frog. The water is murky and green, and at dawn and dusk the light is low.';

/** A split of a mock dataset as benchmark images. */
function splitImages(spec: MockDatasetSpec, split: 'train' | 'val' | 'test') {
  const lo = split === 'train' ? 0 : split === 'val' ? spec.splits.train : spec.splits.train + spec.splits.val;
  return Array.from({ length: spec.splits[split] }, (_, k) => {
    const d = image(spec, lo + k);
    return { file: d.file.split('/').pop() ?? d.file, width: d.width, height: d.height, boxes: d.boxes, from: { spec, index: lo + k } };
  });
}

export function seedBenches(mb: MockBench, t0: number) {
  const spec = (id: string) => specs.find((s) => s.id === id) as MockDatasetSpec;
  const models = new Map(mb.host.models().map((m) => [m.id, m]));
  const cat = catalog();

  const bench = (b: Omit<MBench, 'plan'> & { plan?: MBench['plan'] }): MBench => {
    const out: MBench = { ...b, plan: b.plan ?? cat.default_plan };
    mb.addBench(out);
    return out;
  };

  /** A finished, running or failed entry, its timeline set so `advance` brings it to its state at `t0`. */
  const entry = (
    b: MBench,
    o: {
      id: string;
      name: string;
      contestant: BenchContestant;
      at: number;
      skill: Skill;
      images?: number;
      conditions?: boolean;
      scored?: boolean;
      missing?: number[];
      conditionsRun?: string[] | null;
      /** How long it ran; it is done once that is over (by `t0`, unless it runs now). */
      runs: number;
      fail?: { ms: number; error: string; hint: string };
      usage?: MEntry['usage'];
      speed?: MEntry['speed'];
    },
  ): MEntry => {
    const e: MEntry = {
      id: o.id,
      benchId: b.id,
      name: o.name,
      contestant: o.contestant,
      createdAt: o.at,
      images: o.images ?? b.images.length,
      conditions: o.conditions ?? true,
      scored: o.scored ?? true,
      fingerprint: fingerprintOf(b),
      missing: o.missing ?? [],
      skill: o.skill,
      conditionsRun: o.conditionsRun ?? null,
      slot: o.contestant.kind === 'model',
      startedAt: o.at + 400,
      duration: o.runs,
      stopAt: null,
      failAfter: o.fail ? { ms: o.fail.ms, error: o.fail.error, hint: o.fail.hint } : null,
      end: null,
      result: null,
      usage: o.usage ?? null,
      speed: o.speed ?? null,
      lastEmit: 0,
    };
    mb.addEntry(e);
    return e;
  };
  const model = (id: string, fallback: Partial<Model> & { arch: Model['arch'] }): BenchContestant => {
    const m = models.get(id);
    return { kind: 'model', model_id: id, arch: m?.arch ?? fallback.arch, params: m?.params ?? fallback.params ?? 2_630_000, imgsz: m?.imgsz ?? 640 };
  };
  const metric = (id: string, fallback: number) => models.get(id)?.metrics?.map50_95 ?? fallback;

  // ---------- Shapes val: two loops ----------
  const shapes = spec('ds_shapes');
  const shapesVal = bench({
    id: 'bn_5e12',
    name: 'Shapes val',
    description: 'Coloured shapes on soft backgrounds, as the shapes dataset has them: a check that a model still finds them when the picture gets worse.',
    classes: [...shapes.classes],
    createdAt: t0 - 9 * DAY,
    source: { kind: 'dataset', dataset_id: shapes.id, split: 'val' },
    sourceName: shapes.name,
    images: sampleImages(splitImages(shapes, 'val'), 0, 'Shapes val', 500),
    seed: 0,
  });
  const nBest = metric('m_shapes_n', 0.71);
  const sBest = metric('m_shapes_s', 0.78);
  const loopGen = (b: MBench, id: string, name: string, arch: Model['arch'], at: number, skill: Skill) =>
    entry(b, { id, name, contestant: { kind: 'model', model_id: `m_${id.slice(3)}`, arch, params: arch === 'folo-s' ? 9_100_000 : 2_630_000, imgsz: 640 }, at, skill, runs: 2 * MIN, speed: { device: 'gpu:0', ms_per_image: arch === 'folo-s' ? 4.1 : 2.35 } });
  const base = { fragility: 0.95, ece: 0.04, iou: 0.8 };
  const n0 = entry(shapesVal, { id: 'be_5a10', name: 'shapes-n-best', contestant: model('m_shapes_n', { arch: 'folo-n' }), at: t0 - 8 * DAY, skill: { ...base, map: nBest, selfRecall: 0.68 }, runs: 2 * MIN, speed: { device: 'gpu:0', ms_per_image: 2.35 } });
  const n1 = loopGen(shapesVal, 'be_5a11', 'shapes-n-g1', 'folo-s', t0 - 8 * DAY + 40 * MIN, { ...base, map: nBest + 0.022, selfRecall: 0.7 });
  const n2 = loopGen(shapesVal, 'be_5a12', 'shapes-n-g2', 'folo-s', t0 - 8 * DAY + 80 * MIN, { ...base, map: nBest + 0.041, selfRecall: 0.72 });
  const n3 = loopGen(shapesVal, 'be_5a13', 'shapes-n-g3', 'folo-s', t0 - 8 * DAY + 120 * MIN, { ...base, map: nBest + 0.063, selfRecall: 0.74 });
  const s0 = entry(shapesVal, { id: 'be_5a20', name: 'shapes-s-best', contestant: model('m_shapes_s', { arch: 'folo-s' }), at: t0 - 6 * DAY, skill: { ...base, fragility: 0.85, map: sBest, selfRecall: 0.76 }, runs: 2 * MIN, speed: { device: 'gpu:0', ms_per_image: 4.1 } });
  const s1 = loopGen(shapesVal, 'be_5a21', 'shapes-s-g1', 'folo-s', t0 - 6 * DAY + 45 * MIN, { ...base, fragility: 0.85, map: sBest + 0.006, selfRecall: 0.75 });

  // ---------- Fasteners val: every kind and state ----------
  const fast = spec('ds_fasteners');
  const fastVal = bench({
    id: 'bn_3c90',
    name: 'Fasteners val',
    description: 'The inspection camera over the fastener line: steady light from above, parts that tumble, overlap and lie at any angle.',
    classes: [...fast.classes],
    createdAt: t0 - 5 * HOUR,
    source: { kind: 'dataset', dataset_id: fast.id, split: 'val' },
    sourceName: fast.name,
    images: sampleImages(splitImages(fast, 'val'), 0, 'Fasteners val', 500),
    seed: 0,
  });
  const at = (h: number) => t0 - h * HOUR;
  entry(fastVal, {
    id: 'be_f101',
    name: 'fasteners-s-v1',
    contestant: { kind: 'model', model_id: 'm_f3s1', arch: 'folo-s', params: 9_100_000, imgsz: 640 },
    at: at(4.8),
    skill: { map: 0.641, fragility: 0.82, fragile: { distance: 0.9 }, ece: 0.041, iou: 0.81, selfRecall: 0.66 },
    runs: 70 * SEC,
    speed: { device: 'gpu:0', ms_per_image: 4.1 },
  });
  entry(fastVal, {
    id: 'be_f102',
    name: 'yolo26n',
    contestant: { kind: 'external' },
    at: at(4.1),
    skill: { map: 0.606, fragility: 0.9, fragile: { low_light: 1.15, sensor_noise: 1.2 }, ece: 0.054, iou: 0.8, selfRecall: 0.6 },
    runs: 1500,
  });
  entry(fastVal, {
    id: 'be_f103',
    name: models.get('m_fast_n')?.name ?? 'fasteners-n-best',
    contestant: model('m_fast_n', { arch: 'folo-n' }),
    at: at(4.6),
    skill: { map: metric('m_fast_n', 0.6) - 0.008, fragility: 1, ece: 0.047, iou: 0.785, selfRecall: 0.55 },
    runs: 50 * SEC,
    speed: { device: 'gpu:0', ms_per_image: 2.35 },
  });
  const claude = agentSkill('claude', 'sonnet', 'medium', 'bn_3c90:f104');
  entry(fastVal, {
    id: 'be_f104',
    name: `${AGENT_NAME.claude} · Sonnet`,
    contestant: { kind: 'agent', agent: 'claude', model: 'sonnet', effort: 'medium' },
    at: at(3.2),
    images: 48,
    conditions: false,
    scored: false,
    skill: claude,
    runs: 18 * SEC,
    usage: agentUsage('claude', 'sonnet', 'medium', 48, fastVal.description, 'f104'),
  });
  entry(fastVal, {
    id: 'be_f105',
    name: `${AGENT_NAME.codex} · GPT-6.1-Sol`,
    contestant: { kind: 'agent', agent: 'codex', model: 'gpt-6.1-sol', effort: 'low' },
    at: at(2.9),
    images: 48,
    conditions: false,
    scored: false,
    skill: agentSkill('codex', 'gpt-6.1-sol', 'low', 'bn_3c90:f105'),
    runs: 9 * SEC,
    usage: agentUsage('codex', 'gpt-6.1-sol', 'low', 48, fastVal.description, 'f105'),
  });
  const mini = models.get('m_mini_n');
  entry(fastVal, {
    id: 'be_f106',
    name: mini?.name ?? 'fasteners-mini-n',
    contestant: model('m_mini_n', { arch: 'folo-n' }),
    at: at(2.1),
    missing: missingClasses(fast.classes, mini?.classes ?? ['bolt', 'nut']),
    skill: { map: metric('m_mini_n', 0.7) * 0.9, fragility: 1.05, ece: 0.06, iou: 0.77, selfRecall: 0.5 },
    runs: 48 * SEC,
    speed: { device: 'gpu:0', ms_per_image: 2.35 },
  });
  entry(fastVal, {
    id: 'be_f107',
    name: `${AGENT_NAME.claude} · Opus`,
    contestant: { kind: 'agent', agent: 'claude', model: 'opus', effort: 'medium' },
    at: at(1.2),
    images: 48,
    conditions: false,
    scored: false,
    skill: agentSkill('claude', 'opus', 'medium', 'bn_3c90:f107'),
    runs: 18 * SEC,
    fail: { ms: 300, error: 'Not logged in · Please run /login', hint: 'Sign in to Claude Code on the server’s machine (run `claude` and follow /login), then add the entry again.' },
    usage: agentUsage('claude', 'opus', 'medium', 48, fastVal.description, 'f107'),
  });
  // Running now: started 81 s before the page loaded, at haze by then, done about a minute after.
  const f16 = models.get('m_fast_n_f16');
  entry(fastVal, {
    id: 'be_f108',
    name: f16?.name ?? 'fasteners-n-f16',
    contestant: model('m_fast_n_f16', { arch: 'folo-n' }),
    at: t0 - 81 * SEC - 400,
    skill: { map: metric('m_fast_n', 0.6) - 0.011, fragility: 1.02, ece: 0.05, iou: 0.78, selfRecall: 0.54 },
    runs: 150 * SEC,
    speed: { device: 'gpu:0', ms_per_image: 1.62 },
  });

  // ---------- Pond fish: a designed plan, made from a labeling job deleted since ----------
  const pond = bench({
    id: 'bn_7a41',
    name: 'Pond fish',
    description: POND_DESCRIPTION,
    classes: [...POND.classes],
    createdAt: t0 - 2 * DAY,
    source: { kind: 'label_job', job_id: 'lb_8d02' },
    sourceName: 'pond-cam',
    images: sampleImages(specImages(POND), 0, 'Pond fish', 500),
    seed: 0,
    plan: designPlan(POND_DESCRIPTION, 'claude', 'sonnet'),
  });
  entry(pond, {
    id: 'be_p201',
    name: 'yolo26n-pond',
    contestant: { kind: 'external' },
    at: t0 - 2 * DAY + 3 * HOUR,
    skill: { map: 0.472, fragility: 1.2, fragile: { haze: 1.15, color_cast: 1.7, low_contrast: 1.2 }, ece: 0.071, iou: 0.77, selfRecall: 0.39 },
    runs: 1500,
  });
  entry(pond, {
    id: 'be_p202',
    name: `${AGENT_NAME.claude} · Sonnet`,
    contestant: { kind: 'agent', agent: 'claude', model: 'sonnet', effort: 'medium' },
    at: t0 - 2 * DAY + 4 * HOUR,
    images: 48,
    conditions: false,
    scored: false,
    skill: agentSkill('claude', 'sonnet', 'medium', 'bn_7a41:p202'),
    runs: 18 * SEC,
    usage: agentUsage('claude', 'sonnet', 'medium', 48, pond.description, 'p202'),
  });
  entry(pond, {
    id: 'be_p203',
    name: 'grounding-dino-t',
    contestant: { kind: 'external' },
    at: t0 - DAY - 2 * HOUR,
    scored: false,
    conditionsRun: ['low_light', 'low_contrast', 'haze', 'color_cast'],
    skill: { map: 0.33, fragility: 0.7, ece: 0, iou: 0.74, selfRecall: 0, point: { precision: 0.84, recall: 0.66 } },
    runs: 1500,
  });

  // A quick int8 export of the same detector: its confidences never reach 9 in 10 right, so it cannot label for itself.
  entry(pond, {
    id: 'be_p204',
    name: 'yolo26n-pond-int8',
    contestant: { kind: 'external' },
    at: t0 - DAY + 3 * HOUR,
    skill: { map: 0.418, fragility: 1.3, fragile: { haze: 1.2, color_cast: 1.6 }, ece: 0.112, iou: 0.74, selfRecall: 0 },
    runs: 1500,
  });

  // ---------- Sorting tray, night shift: fresh ----------
  const hard = spec('ds_shapes_hard');
  bench({
    id: 'bn_9f03',
    name: 'Sorting tray, night shift',
    description: 'Overhead camera above the sorting tray on the night shift: dim light, and motion blur from the conveyor.',
    classes: [...hard.classes],
    createdAt: t0 - 25 * MIN,
    source: { kind: 'dataset', dataset_id: hard.id, split: 'val' },
    sourceName: hard.name,
    images: sampleImages(splitImages(hard, 'val'), 7, 'Sorting tray, night shift', 200),
    seed: 7,
  });

  // Every seeded entry ends (or runs on) as its timeline says; the loops are made of the finished ones.
  mb.advance(t0);
  const loop = (o: { id: string; name: string; teacher: MEntry; teacherModel: string; student: Model['arch']; pool: { dataset: MockDatasetSpec; split: 'train' }; gens: MEntry[]; at: number }): MLoop => {
    const r = (x: MEntry) => x.result!;
    const poolImages = o.pool.dataset.splits[o.pool.split];
    const l: MLoop = {
      id: o.id,
      benchId: shapesVal.id,
      name: o.name,
      createdAt: o.at,
      config: {
        name: o.name,
        teacher: o.teacherModel,
        pool: { kind: 'dataset', dataset_id: o.pool.dataset.id, split: o.pool.split },
        generations: o.gens.length - 1,
        min_confidence: null,
        student: { arch: o.student, epochs: 30, imgsz: 640, batch: null, device: 'auto', weights: null, seed: 0 },
      },
      teacher: { id: o.teacherModel, name: o.teacher.name, arch: (o.teacher.contestant as { arch: string }).arch, imgsz: 640, skill: o.teacher.skill },
      student: o.student,
      pool: { images: poolImages, excluded: 0 },
      generations: o.gens.map((e, g) => {
        const prev = g > 0 ? r(o.gens[g - 1]!) : null;
        const labeled = Math.round(poolImages * (0.93 + g * 0.01));
        return {
          generation: g,
          labels: prev ? { threshold: prev.self_labeling.threshold ?? 0.5, images: poolImages, labeled_images: labeled, boxes: Math.round(labeled * 3.9), excluded: 0, dataset_id: `ds_${o.id.slice(3)}${g}` } : null,
          run_id: g ? `run_${o.id.slice(3)}${g}` : null,
          model_id: g ? (e.contestant as { model_id: string }).model_id : o.teacherModel,
          model_name: e.name,
          entry_id: e.id,
          score: r(e).score.total,
          map50_95: r(e).accuracy.map50_95,
          ci95: [r(e).accuracy.ci95[0], r(e).accuracy.ci95[1]],
        };
      }),
      step: null,
      gains: [],
      stopAt: null,
      end: { status: 'done', at: (o.gens.at(-1)!.end?.at ?? o.at) + 1000, error: null, hint: null },
      lastEmit: 0,
    };
    mb.addLoop(l);
    return l;
  };
  // Running as the demo opens: fasteners-n-best's own labels of the fasteners training split, the student training
  // now (its teacher's complete entry is reused), then benchmarked once the GPU is free.
  if (models.has('m_fast_n')) {
    mb.startLoop(fastVal.id, { teacher: 'm_fast_n', pool: { kind: 'dataset', dataset_id: fast.id, split: 'train' }, generations: 2, student: { arch: 'folo-s' } }, t0 - 8 * SEC, 'bl_f3a0');
    mb.advance(t0);
  }
  loop({ id: 'bl_5h02', name: 'shapes-s-best', teacher: s0, teacherModel: 'm_shapes_s', student: 'folo-s', pool: { dataset: spec('ds_shapes_hard'), split: 'train' }, gens: [s0, s1], at: t0 - 6 * DAY + 10 * MIN });
  loop({ id: 'bl_07d1', name: 'shapes-n-best', teacher: n0, teacherModel: 'm_shapes_n', student: 'folo-s', pool: { dataset: shapes, split: 'train' }, gens: [n0, n1, n2, n3], at: t0 - 8 * DAY + 5 * MIN });
}
