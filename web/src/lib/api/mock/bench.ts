// Mock vision benchmarks (docs/API.md, Vision benchmarks; docs/BENCH.md defines every number): benchmarks over images
// of the mock's datasets, labeling jobs and a pond camera of its own (./benchSeed), entries of every kind and state,
// and self-improvement loops. As the labeling mock does, everything that runs is a function of when it started:
// whatever is due is applied lazily (`advance`) before every answer and on every tick of a subscribed stream, so the
// demo moves in real time and a paused clock (the screenshots) holds it still. Results come from ./benchResults,
// generated from a seed and derived with the definitions, so they agree with each other as a server's do.

import { ApiError, type BenchEventHandlers } from '../client';
import type {
  Bench,
  BenchCatalog,
  BenchContestant,
  BenchDesign,
  BenchEntry,
  BenchEntryImagePage,
  BenchEntryImagesQuery,
  BenchGeneration,
  BenchImage,
  BenchImageOptions,
  BenchImagePage,
  BenchImagesQuery,
  BenchLoop,
  BenchLoopPhase,
  BenchPlan,
  BenchPrediction,
  BenchResult,
  BenchSource,
  Box,
  CreateBench,
  DesignBench,
  LabelAgent,
  LabelUsage,
  Model,
  StartBenchEntry,
  StartBenchLoop,
  UpdateBench,
} from '../types';
import { AGENT_NAME } from '../../label';
import { boxesKey, checkPredictions, predictionKey } from '../../bench/external';
import { clonePlan, planError } from '../../bench/plan';
import {
  AGENT_IMAGES,
  BENCH_NAME_MAX,
  DESCRIPTION_MAX,
  DESIGN_NOTE_MAX,
  GENERATIONS_MAX,
  IMAGES_DEFAULT,
  IMAGES_MAX,
  IMAGES_PAGE_MAX,
  isEntryActive,
  isLoopActive,
  MAX_SEVERITY,
  planFreezes,
  rankEntries,
} from '../../bench/rules';
import { f1 } from '../../bench/score';
import { catalog, CONDITIONS } from './benchCatalog';
import { conditionBoxes } from './benchConditions';
import { designPlan, designUsage } from './benchDesign';
import { renderCondition } from './benchRender';
import { generateResult, imageDetections, type GeneratedResult, type Skill } from './benchResults';
import { iou, image, total, type MockDatasetSpec } from './datasets';
import type { MockLabeling } from './label';
import { renderImage } from './render';
import { hash, rng } from './rng';

const SEC = 1000;

/** A model entry's time per image pass, by architecture (the demo's pace, not the GPU's). */
const MS_PER_PASS: Record<string, number> = { 'folo-n': 3, 'folo-s': 4.2, 'folo-m': 6.5, 'folo-l': 9, 'folo-x': 13 };
/** The speed a model entry reports: milliseconds per image on the server's GPU, by architecture. */
const MS_PER_IMAGE: Record<string, number> = { 'folo-n': 2.35, 'folo-s': 4.1, 'folo-m': 7.8, 'folo-l': 11.9, 'folo-x': 18.6 };
/** Parameters by architecture, as a model reports them. */
const PARAMS: Record<string, number> = { 'folo-n': 2_630_000, 'folo-s': 9_100_000, 'folo-m': 20_100_000, 'folo-l': 25_400_000, 'folo-x': 57_000_000 };
/** An agent answers this many images per call (as agent labeling batches them), a call every 1.5 s. */
const PER_CALL: Record<string, number> = { claude: 4, codex: 8 };
const CALL_MS = 1500;
/** A stop ends what is in flight within this long. */
const STOP_MS = 1000;
/** A loop's labeling and its student's training, in the demo's time. */
const LABEL_MS = 4 * SEC;
const TRAIN_MS = 12 * SEC;

/** A benchmark image: where its pixels come from and its truth. */
export interface BenchImg {
  id: string;
  file: string;
  width: number;
  height: number;
  boxes: Box[];
  /** A mock dataset's image (its own objects drawn), or a file uploaded to a labeling job. */
  from: { spec: MockDatasetSpec; index: number } | { blob: Blob; key: string };
}

export interface MBench {
  id: string;
  name: string;
  description: string;
  classes: string[];
  createdAt: number;
  source: BenchSource;
  sourceName: string;
  images: BenchImg[];
  plan: BenchPlan;
  seed: number;
}

export interface MEntry {
  id: string;
  benchId: string;
  name: string;
  contestant: BenchContestant;
  createdAt: number;
  images: number;
  conditions: boolean;
  scored: boolean;
  fingerprint: string;
  missing: number[];
  skill: Skill;
  /** External: the plan's conditions it answered (null: all). */
  conditionsRun: string[] | null;
  /** Runs one at a time with the other model entries (they share the GPU). */
  slot: boolean;
  /** When it started running; null while it waits. */
  startedAt: number | null;
  /** How long it runs once started. */
  duration: number;
  stopAt: number | null;
  /** A failure that ends it this long after it starts (the seeded agent that is not signed in). */
  failAfter: { ms: number; error: string; hint: string | null } | null;
  /** Set once it ended. */
  end: { status: 'done' | 'failed' | 'stopped'; at: number; error: string | null; hint: string | null } | null;
  result: GeneratedResult | null;
  usage: LabelUsage | null;
  speed: { device: string; ms_per_image: number } | null;
  lastEmit: number;
  /**
   * An open external entry: the records received so far, their (image, condition, severity) keys and boxes (a retried
   * part is recognised by them), its boxes, and when its last part arrived.
   */
  open?: { records: BenchPrediction[]; keys: Set<string>; byKey: Map<string, string>; boxes: number; lastPart: number } | null;
}

export interface MLoop {
  id: string;
  benchId: string;
  name: string;
  createdAt: number;
  config: StartBenchLoop;
  teacher: { id: string; name: string; arch: string; imgsz: number; skill: Skill };
  student: string;
  /** Pool images, and how many of them the benchmark has. */
  pool: { images: number; excluded: number };
  generations: BenchGeneration[];
  /** What it does now and since when. */
  step: { gen: number; phase: BenchLoopPhase; at: number; entryId: string | null } | null;
  /** Each generation's gain over the last, in mAP (seeded loops say theirs; a new one draws its own). */
  gains: number[];
  stopAt: number | null;
  end: { status: 'done' | 'failed' | 'stopped'; at: number; error: string | null; hint: string | null } | null;
  lastEmit: number;
}

/** What the benchmark mock needs of the rest of the mock server. */
export interface BenchHost {
  models: () => Model[];
  datasets: () => MockDatasetSpec[];
  labeling: MockLabeling;
  agents: () => LabelAgent[];
}

const iso = (ms: number) => new Date(ms).toISOString();
const hex = (n: number, len = 4) => (n >>> 0).toString(16).padStart(len, '0').slice(-len);
const r4 = (x: number) => Math.round(x * 1e4) / 1e4;

/** A benchmark's fingerprint: its images, truth, plan and seed, hashed (16 hex digits). */
export function fingerprintOf(b: Pick<MBench, 'images' | 'plan' | 'seed'>): string {
  const plan = b.plan.conditions.map((c) => `${c.condition}:${c.severities.join('.')}:${c.weight}`).join(',');
  const imgs = b.images.map((i) => `${i.file}:${i.boxes.map((x) => `${x.class},${x.x1},${x.y1},${x.x2},${x.y2}`).join(';')}`).join('|');
  return hex(hash(`${imgs}#${plan}#${b.seed}`), 8) + hex(hash(`${b.seed}#${plan}#${imgs.length}`), 8);
}

/**
 * An agent entry's default name, as the server makes it (crates/folo-server/src/bench/mod.rs): the agent's name with its
 * model in parentheses, `Claude Code (haiku)`, `Codex (gpt-6.1-sol)`; the agent's name alone without a model.
 */
export const agentEntryName = (agent: 'claude' | 'codex', model: string | null) => (model ? `${AGENT_NAME[agent]} (${model})` : AGENT_NAME[agent]);

/** A loop's default name, as the server makes it: `<teacher's name> loop`. */
export const loopName = (teacher: string) => `${teacher} loop`;

/** A loop's student of generation `g`: its run is `<loop> gen <g>`, so its model (the run's best) `<loop> gen <g>-best`. */
export const studentName = (loop: string, g: number) => `${loop} gen ${g}-best`;

/** The fewest pool images a teacher must keep a box on (the server's MIN_LABELED_IMAGES). */
export const MIN_LABELED_IMAGES = 10;

/**
 * What a teacher keeps of a pool of `images` at a confidence where it gives `perImage` labels an image on the
 * benchmark (its curve's labels there over its entry's images): about that many boxes an image, each image's count a
 * Poisson draw, so `labeled` images get at least one and the rest are left out (a teacher's silence is not a label).
 */
export function poolLabels(images: number, perImage: number, seed: string): { boxes: number; labeled: number } {
  const r = rng(hash(`pool:${seed}`));
  const boxes = Math.round(images * perImage * (0.96 + r() * 0.08));
  // A few pool images hold nothing to label at all (backgrounds): at most 98% get a box, however many it keeps.
  const labeled = Math.round(images * (1 - Math.exp(-perImage)) * (0.95 + r() * 0.03));
  return { boxes, labeled: Math.min(images, boxes, labeled) };
}

/** The server's message when a teacher keeps boxes on too few pool images (crates/folo-server/src/bench/loops.rs). */
export function starvedError(teacher: string, boxes: number, labeled: number, images: number, threshold: number): { error: string; hint: string } {
  return {
    error: `the teacher '${teacher}' kept ${boxes} ${boxes === 1 ? 'box' : 'boxes'} on ${labeled} of the pool's ${images} images at confidence ${Math.round(threshold * 100) / 100}: too few to train a student on (at least ${MIN_LABELED_IMAGES} images)`,
    hint: 'a lower min_confidence keeps more of its detections (the self-labeling chart of its benchmark entry shows what each confidence costs in precision); a better teacher finds more',
  };
}

/**
 * The server's message when a loop has no confidence to keep a teacher's labels from: none set, and the teacher's
 * entry found no self-labeling threshold (a loop never guesses one).
 */
export function noThresholdError(teacher: string): { error: string; hint: string } {
  return {
    error: `'${teacher}' never reaches 90% precision on at least 10 labels on this benchmark, so there is no confidence to label the pool from`,
    hint: "set min_confidence: its entry's self-labeling chart shows what each confidence costs in precision and coverage",
  };
}

/** An open external entry fails after this long without a part. */
export const OPEN_IDLE_MS = 30 * 60 * 1000;
/** The most boxes an open external entry holds (the server's). */
export const OPEN_BOXES_MAX = 20_000_000;
let openBoxesMax = OPEN_BOXES_MAX;
/** Tests: a smaller cap, so its refusal can be seen without 20 million boxes. */
export const setOpenBoxesMax = (n: number) => (openBoxesMax = n);
/** The largest seed a request may carry: JSON's exact integers. */
export const SEED_MAX = Number.MAX_SAFE_INTEGER;

/** A teacher's labels per benchmark image at a confidence, read off its result's self-labeling curve. */
export function labelsPerImage(r: Pick<GeneratedResult, 'self_labeling'>, images: number, threshold: number): number {
  const curve = r.self_labeling.curve;
  if (!curve.length || images <= 0) return 0;
  const p = curve.reduce((a, x) => (Math.abs(x.confidence - threshold) < Math.abs(a.confidence - threshold) ? x : a));
  return p.labels / images;
}

/** The classes of `model` matched to the benchmark's by name, without case: the benchmark's indices it lacks. */
export function missingClasses(benchClasses: readonly string[], modelClasses: readonly string[]): number[] {
  const have = new Set(modelClasses.map((c) => c.toLowerCase()));
  return benchClasses.map((c, i) => (have.has(c.toLowerCase()) ? -1 : i)).filter((i) => i >= 0);
}

export class MockBench {
  #benches: MBench[] = [];
  #entries: MEntry[] = [];
  #loops: MLoop[] = [];
  #subs = new Map<string, Set<BenchEventHandlers>>();
  #urls = new Map<string, Promise<string>>();
  #next = 1;
  /** Scenario `bench-agent-fails`: a design request's answer is not a plan. */
  designFails = false;

  constructor(
    readonly t0: number,
    readonly host: BenchHost,
  ) {}

  // ---------- state the seed (./benchSeed) and the API both build with ----------

  addBench(b: MBench) {
    this.#benches.unshift(b);
    this.#benches.sort((x, y) => y.createdAt - x.createdAt);
  }

  addEntry(e: MEntry) {
    this.#entries.push(e);
  }

  addLoop(l: MLoop) {
    this.#loops.unshift(l);
  }

  /** An id that is not taken: `prefix_` and `digits` hex digits (a new benchmark's are 12, as the server's are now). */
  id(prefix: string, salt: string, digits = 4): string {
    for (let k = 0; ; k++) {
      const n = this.#next++;
      const id =
        digits <= 4
          ? `${prefix}_${hex(hash(`${salt}:${n}:${k}`))}`
          : `${prefix}_${Array.from({ length: Math.ceil(digits / 8) }, (_, i) => hex(hash(`${salt}:${n}:${k}:${i}`), 8)).join('').slice(0, digits)}`;
      if (!this.#benches.some((b) => b.id === id) && !this.#entries.some((e) => e.id === id) && !this.#loops.some((l) => l.id === id)) return id;
    }
  }

  // ---------- reading ----------

  #bench(id: string): MBench {
    const b = this.#benches.find((x) => x.id === id);
    if (!b) throw new ApiError(404, 'bench_not_found', `No benchmark with id ${id}.`, 'List them with GET /api/bench/benchmarks.');
    return b;
  }

  #entry(b: MBench, id: string): MEntry {
    const e = this.#entries.find((x) => x.id === id && x.benchId === b.id);
    if (!e) throw new ApiError(404, 'bench_entry_not_found', `${b.name} has no entry ${id}.`);
    return e;
  }

  #loop(b: MBench, id: string): MLoop {
    const l = this.#loops.find((x) => x.id === id && x.benchId === b.id);
    if (!l) throw new ApiError(404, 'bench_loop_not_found', `${b.name} has no loop ${id}.`);
    return l;
  }

  #entriesOf(b: MBench) {
    return this.#entries.filter((e) => e.benchId === b.id);
  }

  toBench(b: MBench): Bench {
    const entries = rankEntries(this.#entriesOf(b).map((e) => this.toEntry(e)));
    // The leader as the server names it: done, on the benchmark's own fingerprint (one graded before the engine
    // changed ranks below every current entry, so it never leads).
    const fp = fingerprintOf(b);
    const top = entries.find((e) => e.status === 'done' && e.score && e.fingerprint === fp);
    const class_boxes = b.classes.map(() => 0);
    for (const im of b.images) for (const x of im.boxes) class_boxes[x.class] = (class_boxes[x.class] ?? 0) + 1;
    return {
      id: b.id,
      name: b.name,
      description: b.description,
      classes: [...b.classes],
      created: iso(b.createdAt),
      source: { ...b.source },
      source_name: b.sourceName,
      images: b.images.length,
      boxes: class_boxes.reduce((a, n) => a + n, 0),
      class_boxes,
      plan: clonePlan(b.plan),
      seed: b.seed,
      fingerprint: fp,
      entries: entries.length,
      best: top?.score ? { entry_id: top.id, name: top.name, score: top.score.total, partial: top.score.partial } : null,
      // Exactly when DELETE answers 409 bench_active: an entry queued, running or stopping, or a loop running.
      active: entries.some((e) => isEntryActive(e)) || this.#loops.some((l) => l.benchId === b.id && !l.end),
    };
  }

  #status(e: MEntry, now: number): BenchEntry['status'] {
    if (e.end) return e.end.status;
    if (e.startedAt === null || now < e.startedAt) return 'queued';
    return e.stopAt !== null ? 'stopping' : 'running';
  }

  /** The image passes of an entry: its images, then once per condition and severity of the plan. */
  #passes(e: MEntry, b: MBench): { condition: string | null; severity: number | null }[] {
    const out: { condition: string | null; severity: number | null }[] = [{ condition: null, severity: null }];
    if (!e.conditions) return out;
    for (const c of b.plan.conditions) if (!e.conditionsRun || e.conditionsRun.includes(c.condition)) for (const s of c.severities) out.push({ condition: c.condition, severity: s });
    return out;
  }

  toEntry(e: MEntry, now = Date.now()): BenchEntry {
    const b = this.#bench(e.benchId);
    const passes = this.#passes(e, b);
    // An open external entry: the plan's every pass is what it may receive, the records received so far its progress.
    const total = e.open ? e.images * (1 + b.plan.conditions.reduce((n, c) => n + c.severities.length, 0)) : e.images * passes.length;
    const status = this.#status(e, now);
    let done = 0;
    if (e.open) done = e.open.records.length;
    else if (e.end?.status === 'done') done = total;
    else if (e.startedAt !== null && now > e.startedAt) {
      const until = Math.min(now, e.end?.at ?? now, e.stopAt ?? now);
      done = Math.min(total, Math.floor(((until - e.startedAt) / e.duration) * total));
    }
    const pass = status === 'done' || e.open ? null : passes[Math.min(passes.length - 1, Math.floor(done / Math.max(1, e.images)))];
    const r = e.end?.status === 'done' ? e.result : null;
    return {
      id: e.id,
      bench_id: e.benchId,
      name: e.name,
      contestant: { ...e.contestant },
      status,
      created: iso(e.createdAt),
      finished: e.end ? iso(e.end.at) : null,
      progress: { done, total, condition: pass?.condition ?? null, severity: pass?.severity ?? null },
      images: e.images,
      conditions: e.conditions,
      scored: e.scored,
      fingerprint: e.fingerprint,
      missing_classes: e.missing.map((i) => b.classes[i] ?? String(i)),
      score: r ? r.score : null,
      map50_95: r ? r.accuracy.map50_95 : null,
      ci95: r ? [r.accuracy.ci95[0], r.accuracy.ci95[1]] : null,
      // The plan's conditions it ran at every planned severity: what its robustness is over.
      conditions_run: r ? (r.robustness?.conditions.length ?? 0) : 0,
      speed: r && e.speed ? { ...e.speed } : null,
      usage: e.usage ? this.#usageAt(e, done, total) : null,
      error: e.end?.error ?? null,
      ...(e.end?.hint ? { hint: e.end.hint } : {}),
    };
  }

  /** An agent's usage so far: the calls it has made, in proportion to the images answered. */
  #usageAt(e: MEntry, done: number, total: number): LabelUsage {
    const u = e.usage as LabelUsage;
    if (e.end?.status === 'done' || total === 0) return { ...u };
    const f = done / total;
    const calls = e.end?.status === 'failed' ? 1 : Math.round(u.calls * f);
    return {
      calls,
      input_tokens: Math.round(u.input_tokens * f),
      cached_input_tokens: Math.round(u.cached_input_tokens * f),
      output_tokens: Math.round(u.output_tokens * f),
      cost_usd: u.cost_usd === null ? null : Math.round(u.cost_usd * f * 1e6) / 1e6,
    };
  }

  toLoop(l: MLoop): BenchLoop {
    return {
      id: l.id,
      bench_id: l.benchId,
      name: l.name,
      status: l.end ? l.end.status : l.stopAt !== null ? 'stopping' : 'running',
      phase: l.end ? null : (l.step?.phase ?? 'benchmarking'),
      created: iso(l.createdAt),
      finished: l.end ? iso(l.end.at) : null,
      config: structuredClone(l.config),
      generations: l.generations.map((g) => ({ ...g, labels: g.labels ? { ...g.labels } : null, ci95: g.ci95 ? [g.ci95[0], g.ci95[1]] : null })),
      verdict: l.end?.status === 'done' ? this.#verdict(l) : null,
      error: l.end?.error ?? null,
      ...(l.end?.hint ? { hint: l.end.hint } : {}),
    };
  }

  #verdict(l: MLoop): BenchLoop['verdict'] {
    const first = l.generations[0];
    const last = l.generations.at(-1);
    if (!first?.ci95 || last?.map50_95 == null || l.generations.length < 2) return null;
    return last.map50_95 > first.ci95[1] ? 'improved' : last.map50_95 < first.ci95[0] ? 'regressed' : 'held';
  }

  catalog(): BenchCatalog {
    return catalog();
  }

  list(now: number): Bench[] {
    this.advance(now);
    return this.#benches.map((b) => this.toBench(b));
  }

  get(id: string, now: number): Bench {
    const b = this.#bench(id);
    this.advance(now);
    return this.toBench(b);
  }

  // ---------- benchmarks ----------

  #checkFields(body: { name?: string; description?: string }) {
    if (body.name !== undefined && (!body.name.trim() || body.name.trim().length > BENCH_NAME_MAX))
      throw new ApiError(400, 'invalid_body', `field \`name\`: must not be blank, at most ${BENCH_NAME_MAX} characters`);
    if (body.description !== undefined && body.description.length > DESCRIPTION_MAX)
      throw new ApiError(400, 'invalid_body', `field \`description\`: at most ${DESCRIPTION_MAX} characters`);
  }

  #checkPlan(plan: BenchPlan) {
    const e = planError(plan, CONDITIONS);
    if (e) throw new ApiError(400, 'invalid_body', `field \`plan\`: ${e}`);
  }

  /** The images of a source, with their truth, in the source's order. */
  sourceImages(source: BenchSource, now: number): { name: string; classes: string[]; images: Omit<BenchImg, 'id'>[] } {
    if (source.kind === 'dataset') {
      const spec = this.host.datasets().find((d) => d.id === source.dataset_id);
      if (!spec) throw new ApiError(404, 'dataset_not_found', `No dataset with id ${source.dataset_id}.`, 'List datasets with GET /api/datasets.');
      const split = source.split ?? 'val';
      const lo = split === 'train' ? 0 : split === 'val' ? spec.splits.train : spec.splits.train + spec.splits.val;
      const images: Omit<BenchImg, 'id'>[] = [];
      for (let i = lo; i < lo + spec.splits[split]; i++) {
        const d = image(spec, i);
        images.push({ file: d.file.split('/').pop() ?? d.file, width: d.width, height: d.height, boxes: d.boxes, from: { spec, index: i } });
      }
      return { name: spec.name, classes: [...spec.classes], images };
    }
    let job;
    try {
      job = this.host.labeling.labeled(source.job_id, now);
    } catch {
      throw new ApiError(404, 'label_job_not_found', `No labeling job with id ${source.job_id}.`, 'List them with GET /api/label/jobs.');
    }
    return { name: job.name, classes: job.classes, images: job.images.map((i) => ({ file: i.file, width: i.width, height: i.height, boxes: i.boxes, from: i.from })) };
  }

  create(body: CreateBench, now: number): Bench {
    this.#checkFields({ name: body.name, description: body.description });
    if (!body.name?.trim()) throw new ApiError(400, 'invalid_body', 'field `name`: must not be blank');
    const max = body.max_images ?? IMAGES_DEFAULT;
    if (!Number.isInteger(max) || max < 1 || max > IMAGES_MAX) throw new ApiError(400, 'invalid_body', `field \`max_images\`: 1 to ${IMAGES_MAX}`);
    if (body.plan) this.#checkPlan(body.plan);
    const src = this.sourceImages(body.source, now);
    const where = body.source.kind === 'dataset' ? `the ${body.source.split ?? 'val'} split of ${src.name}` : src.name;
    if (!src.images.length) throw new ApiError(400, 'invalid_dataset', `${where[0]!.toUpperCase()}${where.slice(1)} has no labeled image.`, body.source.kind === 'label_job' ? 'Label some of its images first.' : 'Pick a split that has images.');
    const seed = body.seed ?? 0;
    if (!Number.isSafeInteger(seed) || seed < 0) throw new ApiError(400, 'invalid_body', `field \`seed\`: a whole number from 0 to ${SEED_MAX}`);
    // A seeded shuffle, then the first `max_images`: the sample when the source has more, the order either way.
    const r = rng(hash(`bench:${seed}:${body.name}`));
    const order = src.images.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1));
      [order[i], order[j]] = [order[j]!, order[i]!];
    }
    const images = order.slice(0, max).map((k, n) => ({ ...src.images[k]!, id: String(n + 1) }));
    const b: MBench = {
      id: this.id('bn', body.name + now, 12),
      name: body.name.trim(),
      description: body.description ?? '',
      classes: src.classes,
      createdAt: now,
      source: body.source.kind === 'dataset' ? { kind: 'dataset', dataset_id: body.source.dataset_id, split: body.source.split ?? 'val' } : { ...body.source },
      sourceName: src.name,
      images,
      plan: clonePlan(body.plan ?? catalog().default_plan),
      seed,
    };
    this.addBench(b);
    return this.toBench(b);
  }

  update(id: string, body: UpdateBench, now: number): Bench {
    const b = this.#bench(id);
    this.advance(now);
    this.#checkFields(body);
    if (body.plan) {
      // Frozen only by an entry that finished or is under way: a failed or stopped one graded nothing to keep comparable.
      const n = this.#entriesOf(b).filter((e) => planFreezes(this.#status(e, now))).length;
      if (n) throw new ApiError(409, 'bench_frozen', `${b.name} has ${n} finished or running ${n === 1 ? 'entry' : 'entries'}: its plan is fixed so they stay comparable.`, 'Delete those entries, or make a new benchmark with the plan you want.');
      this.#checkPlan(body.plan);
    }
    if (body.name !== undefined) b.name = body.name.trim();
    if (body.description !== undefined) b.description = body.description;
    if (body.plan) b.plan = clonePlan(body.plan);
    const out = this.toBench(b);
    this.#emit(b.id, (h) => h.bench?.(out));
    return out;
  }

  delete(id: string, now: number) {
    const b = this.#bench(id);
    this.advance(now);
    if (this.#entriesOf(b).some((e) => isEntryActive(this.toEntry(e, now))) || this.#loops.some((l) => l.benchId === b.id && !l.end))
      throw new ApiError(409, 'bench_active', `An entry or a loop of ${b.name} is running.`, 'Stop it first, or wait for it to finish.');
    this.#benches = this.#benches.filter((x) => x !== b);
    this.#entries = this.#entries.filter((e) => e.benchId !== b.id);
    this.#loops = this.#loops.filter((l) => l.benchId !== b.id);
  }

  /** The agent's proposal: one call, answered `delay` ms later by the caller. */
  design(id: string, body: DesignBench, now: number): BenchDesign {
    const b = this.#bench(id);
    this.advance(now);
    if (body.agent !== 'claude' && body.agent !== 'codex')
      throw new ApiError(422, 'invalid_body', `field \`agent\`: unknown variant \`${String(body.agent)}\`, expected \`claude\` or \`codex\``);
    if ((body.note ?? '').length > DESIGN_NOTE_MAX) throw new ApiError(400, 'invalid_body', `note must be at most ${DESIGN_NOTE_MAX} characters`);
    const agent = this.host.agents().find((a) => a.id === body.agent);
    if (!agent?.available) throw new ApiError(501, 'not_supported', `${AGENT_NAME[body.agent]} is not installed on the server's machine.`, agent?.hint);
    const model = body.model?.trim() || agent.default_model;
    if (this.designFails)
      throw new ApiError(
        502,
        'agent_failed',
        `${AGENT_NAME[body.agent]} answered, but not with a plan: its reply had no \`conditions\` array.`,
        'Ask again; a larger model (Sonnet) follows the format more reliably.',
      );
    const text = `${b.description}\n${body.note ?? ''}`;
    return { plan: designPlan(text, body.agent, model), usage: designUsage(text, body.agent, model) };
  }

  // ---------- images ----------

  images(id: string, q: BenchImagesQuery, now: number): BenchImagePage {
    const b = this.#bench(id);
    this.advance(now);
    this.#checkCondition(q.condition, q.severity);
    const offset = Math.max(0, q.offset ?? 0);
    const limit = Math.max(0, Math.min(IMAGES_PAGE_MAX, q.limit ?? 100));
    return { total: b.images.length, images: b.images.slice(offset, offset + limit).map((im) => this.#toImage(im, q.condition, q.severity)) };
  }

  #checkCondition(condition: string | undefined, severity: number | undefined) {
    if (condition === undefined && severity === undefined) return;
    if (condition === undefined || severity === undefined) throw new ApiError(400, 'invalid_query', '`condition` and `severity` go together');
    if (!CONDITIONS.some((c) => c.id === condition)) throw new ApiError(400, 'invalid_query', `unknown condition \`${condition}\``);
    if (!Number.isInteger(severity) || severity < 1 || severity > MAX_SEVERITY) throw new ApiError(400, 'invalid_query', `severity must be in 1..=${MAX_SEVERITY}`);
  }

  #toImage(im: BenchImg, condition?: string, severity?: number): BenchImage {
    return { id: im.id, file: im.file, width: im.width, height: im.height, boxes: conditionBoxes(im.boxes, im.width, im.height, condition, severity) };
  }

  /** The image's URL: rendered, then changed by the condition, as a blob. */
  imageUrl(benchId: string, imageId: string, o: BenchImageOptions): Promise<string> {
    const b = this.#bench(benchId);
    this.#checkCondition(o.condition, o.severity);
    const im = b.images.find((x) => x.id === imageId);
    if (!im) return Promise.reject(new ApiError(404, 'bench_image_not_found', `${b.name} has no image ${imageId}.`));
    const key = `${b.id}/${im.id}/${o.condition ?? ''}/${o.severity ?? ''}/${o.thumb ?? 0}`;
    let url = this.#urls.get(key);
    if (!url) {
      const from = im.from;
      url = (async () => {
        const base = 'blob' in from ? from.blob : await renderImage(from.spec, image(from.spec, from.index), from.index, o.thumb);
        const out = o.condition ? await renderCondition(base, { condition: o.condition, severity: o.severity ?? 1, boxes: im.boxes, fullWidth: im.width, seed: `${b.seed}:${im.id}` }) : base;
        return URL.createObjectURL(out);
      })();
      this.#urls.set(key, url);
      if (this.#urls.size > 600) {
        const [oldKey, old] = this.#urls.entries().next().value as [string, Promise<string>];
        this.#urls.delete(oldKey);
        void old.then((u) => setTimeout(() => URL.revokeObjectURL(u), 30_000), () => {});
      }
    }
    return url;
  }

  // ---------- entries ----------

  entries(id: string, now: number): BenchEntry[] {
    const b = this.#bench(id);
    this.advance(now);
    return rankEntries(this.#entriesOf(b).map((e) => this.toEntry(e, now)));
  }

  entry(id: string, entryId: string, now: number): BenchEntry {
    const b = this.#bench(id);
    this.advance(now);
    return this.toEntry(this.#entry(b, entryId), now);
  }

  /** A model entry's skill on a benchmark: its own validation score where it was trained on the same data. */
  modelSkill(m: Model, salt: string): Skill {
    const r = rng(hash(`skill:${m.id}:${salt}`));
    const map = m.metrics?.map50_95 ?? 0.45;
    const big = m.arch === 'folo-n' ? 1 : m.arch === 'folo-s' ? 0.9 : 0.82;
    return { map: map * (0.97 + r() * 0.05), fragility: big * (0.95 + r() * 0.1), ece: 0.035 + r() * 0.04, iou: 0.76 + r() * 0.05, selfRecall: Math.min(0.9, 0.2 + map * 0.8) };
  }

  #startError(b: MBench, body: StartBenchEntry) {
    const k = body.contestant?.kind;
    if (k !== 'model' && k !== 'agent' && k !== 'external') throw new ApiError(422, 'invalid_body', `field \`contestant\`: unknown variant \`${String(k)}\`, expected \`model\`, \`agent\` or \`external\``);
    if (k === 'external') {
      if (!body.name?.trim()) throw new ApiError(400, 'invalid_body', 'field `name`: required for an external contestant');
      if (body.max_images != null || body.conditions != null) throw new ApiError(400, 'invalid_body', '`max_images` and `conditions` are not for an external contestant');
    }
    if (body.name != null && body.name.trim().length > BENCH_NAME_MAX) throw new ApiError(400, 'invalid_body', `field \`name\`: at most ${BENCH_NAME_MAX} characters`);
    if (body.max_images != null && (!Number.isInteger(body.max_images) || body.max_images < 1 || body.max_images > b.images.length))
      throw new ApiError(400, 'invalid_body', `field \`max_images\`: 1 to ${b.images.length}, the benchmark's images`);
  }

  start(id: string, body: StartBenchEntry, now: number): BenchEntry {
    const b = this.#bench(id);
    this.advance(now);
    this.#startError(b, body);
    const c = body.contestant;
    const fp = fingerprintOf(b);
    const base = { benchId: b.id, createdAt: now, fingerprint: fp, stopAt: null, failAfter: null, end: null, result: null, lastEmit: 0, conditionsRun: null };
    let e: MEntry;
    if (c.kind === 'model') {
      const m = this.host.models().find((x) => x.id === c.model_id);
      if (!m) throw new ApiError(404, 'model_not_found', `No model with id ${c.model_id}.`);
      const missing = missingClasses(b.classes, m.classes);
      if (missing.length === b.classes.length)
        throw new ApiError(400, 'bench_class_mismatch', `None of ${m.name}'s ${m.classes.length} classes is a class of ${b.name}.`, `The benchmark's classes are ${b.classes.join(', ')}; a model is matched to them by name.`);
      const images = body.max_images ?? b.images.length;
      const conditions = body.conditions ?? true;
      e = {
        ...base,
        id: this.id('be', m.id + now),
        name: body.name?.trim() || m.name,
        contestant: { kind: 'model', model_id: m.id, arch: m.arch, params: m.params, imgsz: m.imgsz },
        images,
        conditions,
        scored: true,
        missing,
        skill: this.modelSkill(m, b.id),
        slot: true,
        startedAt: null,
        duration: 0,
        usage: null,
        speed: { device: 'gpu:0', ms_per_image: MS_PER_IMAGE[m.arch] ?? 5 },
      };
      e.duration = images * this.#passes(e, b).length * (MS_PER_PASS[m.arch] ?? 5);
    } else if (c.kind === 'agent') {
      if (c.agent !== 'claude' && c.agent !== 'codex') throw new ApiError(422, 'invalid_body', `field \`agent\`: unknown variant \`${String(c.agent)}\``);
      const agent = this.host.agents().find((a) => a.id === c.agent);
      if (!agent?.available) throw new ApiError(501, 'not_supported', `${AGENT_NAME[c.agent]} is not installed on the server's machine.`, agent?.hint);
      const model = c.model?.trim() || agent.default_model;
      const effort = c.effort ?? agent.default_effort;
      const images = body.max_images ?? Math.min(AGENT_IMAGES, b.images.length);
      const conditions = body.conditions ?? false;
      e = {
        ...base,
        id: this.id('be', c.agent + now),
        name: body.name?.trim() || agentEntryName(c.agent, model),
        contestant: { kind: 'agent', agent: c.agent, model, effort },
        images,
        conditions,
        scored: false,
        missing: [],
        skill: agentSkill(c.agent, model, effort, `${b.id}:${now}`),
        slot: false,
        startedAt: now + 600,
        duration: 0,
        usage: null,
        speed: null,
      };
      const passes = this.#passes(e, b).length;
      e.duration = Math.ceil((images * passes) / PER_CALL[c.agent]!) * CALL_MS;
      e.usage = agentUsage(c.agent, model, effort, images * passes, b.description, `${b.id}:${e.id}`);
    } else if (c.open) {
      // Open: created running, filled in parts, scored by its finish; the predictions here are its first part.
      const keys = new Set<string>();
      const first = c.predictions ?? [];
      const check = checkPredictions(first, this.#ctx(b, c.scored), { part: true, seen: keys });
      if (!check.ok) throw new ApiError(400, 'invalid_body', check.error);
      e = {
        ...base,
        id: this.id('be', (body.name ?? '') + now),
        name: (body.name ?? '').trim(),
        contestant: { kind: 'external' },
        images: b.images.length,
        conditions: false,
        scored: c.scored ?? true,
        missing: [],
        skill: { map: 0, fragility: 1, ece: 0, iou: 0.6, selfRecall: 0 },
        slot: false,
        startedAt: now,
        duration: Infinity,
        usage: null,
        speed: null,
        open: { records: [...first], keys, byKey: new Map(first.map((r) => [predictionKey(r.image, r.condition, r.severity), boxesKey(r.boxes, c.scored ?? true)])), lastPart: now, boxes: first.reduce((n, r) => n + r.boxes.length, 0) },
      };
    } else {
      const check = checkPredictions(c.predictions ?? [], this.#ctx(b, c.scored));
      if (!check.ok) throw new ApiError(400, 'invalid_body', check.error);
      const scored = c.scored ?? true;
      e = {
        ...base,
        id: this.id('be', (body.name ?? '') + now),
        name: (body.name ?? '').trim(),
        contestant: { kind: 'external' },
        images: b.images.length,
        conditions: check.summary.conditions.length > 0,
        conditionsRun: check.summary.conditions.length < b.plan.conditions.length ? check.summary.conditions : null,
        scored,
        missing: [],
        skill: externalSkill(b, check.predictions, scored),
        slot: false,
        startedAt: now,
        duration: 0,
        usage: null,
        speed: null,
      };
      this.addEntry(e);
      this.#finish(e, now);
      const out = this.toEntry(e, now);
      this.#emit(b.id, (h) => h.entry?.(out));
      const bench = this.toBench(b);
      this.#emit(b.id, (h) => h.bench?.(bench));
      return out;
    }
    this.addEntry(e);
    this.advance(now);
    const out = this.toEntry(e, now);
    this.#emit(b.id, (h) => h.entry?.(out));
    const bench = this.toBench(b);
    this.#emit(b.id, (h) => h.bench?.(bench));
    return out;
  }

  /** What an external contestant's records are checked against. */
  #ctx(b: MBench, scored?: boolean) {
    return { imageIds: b.images.map((i) => i.id), classes: b.classes.length, plan: b.plan, catalog: CONDITIONS.map((x) => x.id), names: Object.fromEntries(CONDITIONS.map((x) => [x.id, x.name])), scored };
  }

  #open(b: MBench, entryId: string, now: number): MEntry & { open: NonNullable<MEntry['open']> } {
    const e = this.#entry(b, entryId);
    if (!e.open) throw new ApiError(409, 'bench_entry_not_open', `${e.name} is not an open external entry (it is ${this.#status(e, now)}).`, 'Create one with `contestant.open: true`.');
    return e as MEntry & { open: NonNullable<MEntry['open']> };
  }

  /**
   * A part of an open external entry, all of it or nothing: a record already received with the same boxes is taken
   * again and adds nothing (a part retried after a lost answer), one with other boxes is refused; the entry holds at
   * most OPEN_BOXES_MAX boxes.
   */
  sendPredictions(id: string, entryId: string, predictions: BenchPrediction[], now: number): BenchEntry {
    const b = this.#bench(id);
    this.advance(now);
    const e = this.#open(b, entryId, now);
    const fresh: BenchPrediction[] = [];
    for (const [i, r] of (predictions ?? []).entries()) {
      const k = r && typeof r.image === 'string' ? predictionKey(r.image, r.condition, r.severity) : null;
      const had = k !== null ? e.open.byKey.get(k) : undefined;
      if (had === undefined) fresh.push(r);
      else if (had !== boxesKey(r.boxes, e.scored))
        throw new ApiError(400, 'invalid_body', `Record ${i + 1}: image "${r.image}"${r.condition ? ` under ${r.condition} ${r.severity}` : ''} was received before with other boxes.`);
    }
    const keys = new Set(e.open.keys);
    const check = checkPredictions(fresh, this.#ctx(b, e.scored), { part: true, seen: keys });
    if (!check.ok) throw new ApiError(400, 'invalid_body', check.error);
    const boxes = e.open.boxes + fresh.reduce((n, r) => n + r.boxes.length, 0);
    if (boxes > openBoxesMax)
      throw new ApiError(
        400,
        'invalid_body',
        `this part would make the entry hold ${boxes} boxes in memory, more than its ${openBoxesMax}: send it pass by pass (the unchanged images, then each condition and severity), so each condition's severity is graded and let go of as soon as all its images are in`,
      );
    e.open.records.push(...fresh);
    for (const r of fresh) e.open.byKey.set(predictionKey(r.image, r.condition, r.severity), boxesKey(r.boxes, e.scored));
    e.open.keys = keys;
    e.open.boxes = boxes;
    e.open.lastPart = now;
    const out = this.toEntry(e, now);
    this.#emit(b.id, (h) => h.entry?.(out));
    return out;
  }

  /** Scores an open external entry; what is missing is named and it stays open. */
  finishEntry(id: string, entryId: string, now: number): BenchEntry {
    const b = this.#bench(id);
    this.advance(now);
    const e = this.#open(b, entryId, now);
    const check = checkPredictions(e.open.records, this.#ctx(b, e.scored));
    if (!check.ok) throw new ApiError(400, 'invalid_body', check.error);
    e.conditions = check.summary.conditions.length > 0;
    e.conditionsRun = check.summary.conditions.length < b.plan.conditions.length ? check.summary.conditions : null;
    e.skill = externalSkill(b, e.open.records, e.scored);
    (e as MEntry).open = null;
    this.#finish(e, now);
    const out = this.toEntry(e, now);
    this.#emit(b.id, (h) => h.entry?.(out));
    const bench = this.toBench(b);
    this.#emit(b.id, (h) => h.bench?.(bench));
    return out;
  }

  result(id: string, entryId: string, now: number): BenchResult {
    const b = this.#bench(id);
    this.advance(now);
    const e = this.#entry(b, entryId);
    if (e.end?.status !== 'done' || !e.result) throw this.#notDone(e, now);
    const r = e.result;
    return {
      entry: this.toEntry(e, now),
      accuracy: structuredClone(r.accuracy),
      robustness: r.robustness ? structuredClone(r.robustness) : null,
      calibration: r.calibration ? structuredClone(r.calibration) : null,
      localization: structuredClone(r.localization),
      self_labeling: structuredClone(r.self_labeling),
    };
  }

  #notDone(e: MEntry, now: number) {
    return new ApiError(409, 'bench_entry_not_done', `${e.name} is ${this.#status(e, now)}, not done.`, 'Results are there once the entry is done.');
  }

  entryImages(id: string, entryId: string, q: BenchEntryImagesQuery, now: number): BenchEntryImagePage {
    const b = this.#bench(id);
    this.advance(now);
    const e = this.#entry(b, entryId);
    if (e.end?.status !== 'done' || !e.result) throw this.#notDone(e, now);
    if (q.sort && q.sort !== 'order' && q.sort !== 'worst') throw new ApiError(400, 'invalid_query', `unknown sort \`${q.sort}\`: order or worst`);
    const missing = new Set(e.missing);
    const all = b.images.slice(0, e.images).map((im, k) => {
      const d = imageDetections(im, {
        quality: Math.min(0.97, e.skill.map + 0.24),
        seed: hash(`${e.id}:${im.id}`),
        classes: b.classes.length,
        missing,
        bestConf: e.result!.accuracy.best_conf,
        scored: e.scored,
      });
      return { k, page: { image: this.#toImage(im), ...d } };
    });
    if (q.sort === 'worst') all.sort((x, y) => y.page.missed + y.page.wrong - (x.page.missed + x.page.wrong) || x.k - y.k);
    const offset = Math.max(0, q.offset ?? 0);
    const limit = Math.max(0, Math.min(IMAGES_PAGE_MAX, q.limit ?? 100));
    return { total: all.length, images: all.slice(offset, offset + limit).map((x) => x.page) };
  }

  stopEntry(id: string, entryId: string, now: number): BenchEntry {
    const b = this.#bench(id);
    this.advance(now);
    const e = this.#entry(b, entryId);
    const status = this.#status(e, now);
    if (!isEntryActive({ status })) throw new ApiError(409, 'bench_entry_not_active', `${e.name} is ${status}: nothing to stop.`);
    if (e.open) {
      // An open external entry: what it received is discarded and it ends stopped, on the leaderboard until deleted.
      e.end = { status: 'stopped', at: now, error: null, hint: null };
      e.open = null;
      const out = this.toEntry(e, now);
      this.#emit(b.id, (h) => h.entry?.(out));
      const bench = this.toBench(b);
      this.#emit(b.id, (h) => h.bench?.(bench));
      return out;
    }
    if (status === 'queued') {
      // Never started: it ends at once.
      e.end = { status: 'stopped', at: now, error: null, hint: null };
    } else if (e.stopAt === null) e.stopAt = now + STOP_MS;
    const out = this.toEntry(e, now);
    this.#emit(b.id, (h) => h.entry?.(out));
    return out;
  }

  deleteEntry(id: string, entryId: string, now: number) {
    const b = this.#bench(id);
    this.advance(now);
    const e = this.#entry(b, entryId);
    if (isEntryActive({ status: this.#status(e, now) })) throw new ApiError(409, 'bench_entry_active', `${e.name} is running.`, 'Stop it first.');
    const loop = this.#loops.find((l) => !l.end && l.generations.some((g) => g.entry_id === e.id));
    if (loop) throw new ApiError(409, 'bench_entry_in_loop', `The loop ${loop.name} uses ${e.name}.`, 'Wait for the loop to end, or stop it.');
    this.#entries = this.#entries.filter((x) => x !== e);
    this.#emit(b.id, (h) => h.removed?.({ id: e.id }));
    const bench = this.toBench(b);
    this.#emit(b.id, (h) => h.bench?.(bench));
  }

  // ---------- loops ----------

  loops(id: string, now: number): BenchLoop[] {
    const b = this.#bench(id);
    this.advance(now);
    return this.#loops.filter((l) => l.benchId === b.id).sort((x, y) => y.createdAt - x.createdAt).map((l) => this.toLoop(l));
  }

  loop(id: string, loopId: string, now: number): BenchLoop {
    const b = this.#bench(id);
    this.advance(now);
    return this.toLoop(this.#loop(b, loopId));
  }

  /** `fixedId`: the seed's loop that runs as the demo opens has a stable id, so it can be linked (./benchSeed). */
  startLoop(id: string, body: StartBenchLoop, now: number, fixedId?: string): BenchLoop {
    const b = this.#bench(id);
    this.advance(now);
    const m = this.host.models().find((x) => x.id === body.teacher);
    if (!m) throw new ApiError(404, 'model_not_found', `No model with id ${body.teacher}.`);
    if (m.classes.length !== b.classes.length || m.classes.some((c, i) => c !== b.classes[i]))
      throw new ApiError(400, 'invalid_body', `${m.name}'s classes are not ${b.name}'s (the same names in the same order).`, `The benchmark's classes: ${b.classes.join(', ')}.`);
    const gens = body.generations ?? 1;
    if (!Number.isInteger(gens) || gens < 1 || gens > GENERATIONS_MAX) throw new ApiError(400, 'invalid_body', `field \`generations\`: 1 to ${GENERATIONS_MAX}`);
    if (body.min_confidence != null && !(body.min_confidence >= 0 && body.min_confidence <= 1)) throw new ApiError(400, 'invalid_body', 'field `min_confidence`: 0 to 1');
    // The students' settings, checked as POST /api/runs checks a run's, before anything is made.
    const w = body.student?.weights;
    if (w && !this.host.models().some((x) => x.id === w)) throw new ApiError(404, 'model_not_found', `No model with id ${w} (student.weights).`, 'List models with GET /api/models.');
    const dev = body.student?.device;
    if (dev && dev !== 'auto' && dev !== 'cpu' && dev !== 'gpu:0') throw new ApiError(404, 'device_not_found', `No device ${dev} on this machine.`, 'Devices: cpu, gpu:0.');
    if (body.student?.imgsz != null && body.student.imgsz < 64) throw new ApiError(400, 'invalid_config', `imgsz must be at least 64 for training, got ${body.student.imgsz}`);
    const seed = body.student?.seed ?? 0;
    if (!Number.isSafeInteger(seed) || seed < 0) throw new ApiError(400, 'invalid_body', `field \`student.seed\`: a whole number from 0 to ${SEED_MAX}`);
    const pool = this.sourceImages(body.pool.kind === 'dataset' ? { ...body.pool, split: body.pool.split ?? 'train' } : body.pool, now);
    const inBench = new Set(b.images.map((i) => ('spec' in i.from ? `${i.from.spec.id}/${i.from.index}` : i.from.key)));
    const excluded = pool.images.filter((i) => inBench.has('spec' in i.from ? `${i.from.spec.id}/${i.from.index}` : i.from.key)).length;
    if (pool.images.length - excluded <= 0) throw new ApiError(400, 'invalid_dataset', `Every image of the pool is in ${b.name}.`, 'Pick a pool the benchmark was not made from (another split, another job).');
    const config: StartBenchLoop = {
      name: body.name?.trim() || loopName(m.name),
      teacher: m.id,
      pool: body.pool.kind === 'dataset' ? { kind: 'dataset', dataset_id: body.pool.dataset_id, split: body.pool.split ?? 'train' } : { ...body.pool },
      generations: gens,
      min_confidence: body.min_confidence ?? null,
      student: { arch: body.student?.arch ?? m.arch, epochs: body.student?.epochs ?? 30, imgsz: body.student?.imgsz ?? m.imgsz, batch: body.student?.batch ?? null, device: body.student?.device ?? 'auto', weights: body.student?.weights ?? null, seed: body.student?.seed ?? 0 },
    };
    const r = rng(hash(`loop:${b.id}:${m.id}:${now}`));
    const l: MLoop = {
      id: fixedId ?? this.id('bl', m.id + now),
      benchId: b.id,
      name: config.name as string,
      createdAt: now,
      config,
      teacher: { id: m.id, name: m.name, arch: m.arch, imgsz: m.imgsz, skill: this.modelSkill(m, b.id) },
      student: config.student?.arch ?? m.arch,
      pool: { images: pool.images.length, excluded },
      generations: [],
      step: null,
      gains: Array.from({ length: gens }, (_, g) => 0.004 + r() * 0.02 - g * 0.002),
      stopAt: null,
      end: null,
      lastEmit: 0,
    };
    this.addLoop(l);
    this.#advanceLoop(l, b, now);
    const out = this.toLoop(l);
    this.#emit(b.id, (h) => h.loop?.(out));
    return out;
  }

  stopLoop(id: string, loopId: string, now: number): BenchLoop {
    const b = this.#bench(id);
    this.advance(now);
    const l = this.#loop(b, loopId);
    if (l.end) throw new ApiError(409, 'bench_loop_not_active', `${l.name} has ended.`);
    if (l.stopAt === null) {
      l.stopAt = now + STOP_MS;
      const e = l.step?.entryId ? this.#entries.find((x) => x.id === l.step!.entryId) : undefined;
      if (e && !e.end) this.stopEntry(b.id, e.id, now);
    }
    const out = this.toLoop(l);
    this.#emit(b.id, (h) => h.loop?.(out));
    return out;
  }

  deleteLoop(id: string, loopId: string, now: number) {
    const b = this.#bench(id);
    this.advance(now);
    const l = this.#loop(b, loopId);
    if (!l.end) throw new ApiError(409, 'bench_loop_active', `${l.name} is running.`, 'Stop it first.');
    this.#loops = this.#loops.filter((x) => x !== l);
    this.#emit(b.id, (h) => h.removed?.({ id: l.id }));
  }

  // ---------- time ----------

  /**
   * Applies what is due by `now`, in time order across every benchmark: model entries take the GPU one at a time (the
   * oldest waiting first), entries end, loops move through their phases; each change is told on its benchmark's
   * stream, and a running entry's progress at most twice a second.
   */
  advance(now: number) {
    // An open external entry that has had no part for OPEN_IDLE_MS fails (its sender is gone).
    for (const e of this.#entries) {
      if (!e.open || now - e.open.lastPart < OPEN_IDLE_MS) continue;
      e.open = null;
      e.end = { status: 'failed', at: now, error: 'no predictions arrived for 30 minutes', hint: 'start it again (POST /api/bench/benchmarks/{id}/entries with contestant.open: true) and send its parts without long pauses' };
      this.#tell(e, now);
    }
    for (let guard = 0; guard < 200; guard++) {
      if (!this.#step(now)) break;
    }
    for (const e of this.#entries) {
      if (e.end || e.startedAt === null || now < e.startedAt || now - e.lastEmit < 500) continue;
      e.lastEmit = now;
      const out = this.toEntry(e, now);
      this.#emit(e.benchId, (h) => h.entry?.(out));
    }
  }

  /** One change that is due (the earliest), applied; false when nothing is. */
  #step(now: number): boolean {
    // The earliest end of an entry that runs.
    let first: { e: MEntry; at: number; kind: 'done' | 'failed' | 'stopped' } | null = null;
    for (const e of this.#entries) {
      if (e.end || e.startedAt === null) continue;
      const candidates: { at: number; kind: 'done' | 'failed' | 'stopped' }[] = [{ at: e.startedAt + e.duration, kind: 'done' }];
      if (e.failAfter) candidates.push({ at: e.startedAt + e.failAfter.ms, kind: 'failed' });
      if (e.stopAt !== null) candidates.push({ at: e.stopAt, kind: 'stopped' });
      const c = candidates.sort((a, b) => a.at - b.at)[0]!;
      if (c.at <= now && (!first || c.at < first.at)) first = { e, ...c };
    }
    // The slot: a waiting model entry starts once no model entry runs.
    const busy = this.#entries.some((e) => e.slot && !e.end && e.startedAt !== null);
    const waiting = this.#entries.filter((e) => e.slot && !e.end && e.startedAt === null).sort((a, b) => a.createdAt - b.createdAt)[0];
    if (!busy && waiting && waiting.createdAt + 400 <= now) {
      const lastEnd = Math.max(0, ...this.#entries.filter((e) => e.slot && e.end).map((e) => e.end!.at));
      waiting.startedAt = Math.max(waiting.createdAt + 400, lastEnd);
      this.#tell(waiting, now);
      return true;
    }
    // A loop's phase that is due before the next entry ends.
    const loopAt = this.#loops.filter((l) => !l.end).map((l) => ({ l, at: this.#loopDue(l) })).filter((x) => x.at !== null && x.at <= now).sort((a, b) => a.at! - b.at!)[0];
    if (loopAt && (!first || loopAt.at! <= first.at)) {
      this.#advanceLoop(loopAt.l, this.#bench(loopAt.l.benchId), now);
      return true;
    }
    if (!first) return false;
    const { e, at, kind } = first;
    if (kind === 'done') this.#finish(e, at);
    else if (kind === 'failed') e.end = { status: 'failed', at, error: e.failAfter!.error, hint: e.failAfter!.hint };
    else e.end = { status: 'stopped', at, error: null, hint: null };
    this.#tell(e, now);
    const b = this.#benches.find((x) => x.id === e.benchId);
    if (b) {
      const bench = this.toBench(b);
      this.#emit(b.id, (h) => h.bench?.(bench));
    }
    return true;
  }

  #tell(e: MEntry, now: number) {
    e.lastEmit = now;
    const out = this.toEntry(e, now);
    this.#emit(e.benchId, (h) => h.entry?.(out));
  }

  /** Scores an entry that ran to its end at `at`. */
  #finish(e: MEntry, at: number) {
    const b = this.#bench(e.benchId);
    e.result = generateResult({
      benchId: b.id,
      entryId: e.id,
      classes: b.classes.length,
      plan: b.plan,
      benchImages: b.images.length,
      images: b.images.slice(0, e.images),
      conditions: e.conditions,
      conditionsRun: e.conditionsRun,
      scored: e.scored,
      missing: new Set(e.missing),
      skill: e.skill,
    });
    e.end = { status: 'done', at, error: null, hint: null };
  }

  /** When the loop's next phase is due, or null while it waits on an entry. */
  #loopDue(l: MLoop): number | null {
    if (l.stopAt !== null) return l.stopAt;
    const s = l.step;
    if (!s) return l.createdAt;
    if (s.phase === 'labeling') return s.at + LABEL_MS;
    if (s.phase === 'training') return s.at + TRAIN_MS;
    const e = s.entryId ? this.#entries.find((x) => x.id === s.entryId) : undefined;
    return e?.end ? e.end.at : null;
  }

  /** Moves a loop on by one phase, if one is due. */
  #advanceLoop(l: MLoop, b: MBench, now: number) {
    if (l.end) return;
    const tell = () => {
      const out = this.toLoop(l);
      this.#emit(b.id, (h) => h.loop?.(out));
    };
    if (l.stopAt !== null && l.stopAt <= now) {
      l.end = { status: 'stopped', at: l.stopAt, error: null, hint: null };
      l.step = null;
      return tell();
    }
    const s = l.step;
    if (!s) {
      // Generation 0: the teacher's entry, an existing finished complete one on this fingerprint when there is one.
      const fp = fingerprintOf(b);
      const had = this.#entries.find(
        (e) => e.benchId === b.id && e.contestant.kind === 'model' && e.contestant.model_id === l.teacher.id && e.end?.status === 'done' && e.fingerprint === fp && !e.result?.score.partial,
      );
      const entry = had ?? this.#loopEntry(l, b, l.teacher.name, l.teacher.id, l.teacher.arch, l.teacher.skill, l.createdAt);
      l.generations = [{ generation: 0, labels: null, run_id: null, model_id: l.teacher.id, model_name: l.teacher.name, entry_id: entry.id, score: null, map50_95: null, ci95: null }];
      l.step = { gen: 0, phase: 'benchmarking', at: l.createdAt, entryId: entry.id };
      this.#fillGeneration(l, 0, entry);
      if (had) this.#afterBenchmark(l, l.createdAt + 300);
      return tell();
    }
    const at = this.#loopDue(l);
    if (at === null || at > now) return;
    if (s.phase === 'benchmarking') {
      const e = this.#entries.find((x) => x.id === s.entryId);
      if (e?.end?.status !== 'done') {
        l.end = { status: 'failed', at, error: `The benchmark entry of ${l.generations[s.gen]?.model_name ?? 'the model'} did not finish (${e?.end?.status ?? 'gone'}).`, hint: 'Start the loop again.' };
        l.step = null;
        return tell();
      }
      this.#fillGeneration(l, s.gen, e);
      this.#afterBenchmark(l, at);
    } else if (s.phase === 'labeling') {
      // The teacher's labels of the pool, from its entry's curve at the confidence (as the server keeps them): the
      // images it kept nothing on are left out, and with fewer than MIN_LABELED_IMAGES the loop fails here.
      const teacher = l.generations[s.gen - 1]!;
      const te = this.#entries.find((x) => x.id === teacher.entry_id);
      const threshold = l.config.min_confidence ?? te?.result?.self_labeling.threshold ?? null;
      if (threshold === null) {
        // Checked when the generation starts (#afterBenchmark); here only if the entry changed under it.
        const why = noThresholdError(teacher.model_name ?? l.teacher.name);
        l.end = { status: 'failed', at, error: why.error, hint: why.hint };
        l.step = null;
        return tell();
      }
      const imgs = l.pool.images - l.pool.excluded;
      const per = te?.result ? labelsPerImage(te.result, te.images, threshold) : 0;
      const kept = poolLabels(imgs, per, `${l.id}:${s.gen}`);
      if (kept.labeled < MIN_LABELED_IMAGES) {
        const why = starvedError(teacher.model_name ?? l.teacher.name, kept.boxes, kept.labeled, imgs, threshold);
        l.end = { status: 'failed', at, error: why.error, hint: why.hint };
        l.step = null;
        return tell();
      }
      l.generations[s.gen]!.labels = { threshold, images: imgs, labeled_images: kept.labeled, boxes: kept.boxes, excluded: l.pool.excluded, dataset_id: `ds_${hex(hash(`${l.id}:${s.gen}:ds`))}` };
      l.generations[s.gen]!.run_id = `run_${hex(hash(`${l.id}:${s.gen}:run`))}`;
      l.step = { gen: s.gen, phase: 'training', at, entryId: null };
    } else {
      const g = l.generations[s.gen]!;
      g.model_id = `m_${hex(hash(`${l.id}:${s.gen}:model`))}`;
      g.model_name = studentName(l.name, s.gen);
      const prev = l.generations[s.gen - 1]!;
      const prevEntry = this.#entries.find((x) => x.id === prev.entry_id);
      const skill: Skill = { ...(prevEntry?.skill ?? l.teacher.skill), map: (prev.map50_95 ?? l.teacher.skill.map) + (l.gains[s.gen - 1] ?? 0.01) };
      const e = this.#loopEntry(l, b, g.model_name, g.model_id, l.student, skill, at);
      g.entry_id = e.id;
      l.step = { gen: s.gen, phase: 'benchmarking', at, entryId: e.id };
    }
    tell();
  }

  /** Generation `gen` has its score: the next one's labeling, or the loop's end. */
  #afterBenchmark(l: MLoop, at: number) {
    const gens = l.config.generations ?? 1;
    const s = l.step!;
    if (s.gen >= gens) {
      l.end = { status: 'done', at, error: null, hint: null };
      l.step = null;
      return;
    }
    l.generations.push({ generation: s.gen + 1, labels: null, run_id: null, model_id: null, model_name: null, entry_id: null, score: null, map50_95: null, ci95: null });
    l.step = { gen: s.gen + 1, phase: 'labeling', at, entryId: null };
    // No confidence to keep labels from (none set, none found by the teacher's entry): it fails before labeling.
    const te = this.#entries.find((x) => x.id === l.generations[s.gen]?.entry_id);
    if (l.config.min_confidence == null && te?.result && te.result.self_labeling.threshold === null) {
      const why = noThresholdError(l.generations[s.gen]?.model_name ?? l.teacher.name);
      l.end = { status: 'failed', at, error: why.error, hint: why.hint };
      l.step = null;
    }
  }

  #fillGeneration(l: MLoop, gen: number, e: MEntry) {
    const g = l.generations[gen]!;
    if (e.end?.status !== 'done' || !e.result) return;
    g.score = e.result.score.total;
    g.map50_95 = e.result.accuracy.map50_95;
    g.ci95 = [e.result.accuracy.ci95[0], e.result.accuracy.ci95[1]];
  }

  /** A loop's model entry: complete (every image, every condition), queued with the other model entries. */
  #loopEntry(l: MLoop, b: MBench, name: string, modelId: string, arch: string, skill: Skill, at: number): MEntry {
    const e: MEntry = {
      id: this.id('be', `${l.id}:${name}`),
      benchId: b.id,
      name,
      contestant: { kind: 'model', model_id: modelId, arch: arch as 'folo-n', params: PARAMS[arch] ?? 9_000_000, imgsz: l.teacher.imgsz },
      createdAt: at,
      images: b.images.length,
      conditions: true,
      scored: true,
      fingerprint: fingerprintOf(b),
      missing: [],
      skill,
      conditionsRun: null,
      slot: true,
      startedAt: null,
      duration: 0,
      stopAt: null,
      failAfter: null,
      end: null,
      result: null,
      usage: null,
      speed: { device: 'gpu:0', ms_per_image: MS_PER_IMAGE[arch] ?? 5 },
      lastEmit: 0,
    };
    e.duration = e.images * this.#passes(e, b).length * (MS_PER_PASS[arch] ?? 5) * 0.4;
    this.addEntry(e);
    const out = this.toEntry(e, at);
    this.#emit(b.id, (h) => h.entry?.(out));
    return e;
  }

  // ---------- events ----------

  subscribe(id: string, h: BenchEventHandlers): () => void {
    const subs = this.#subs.get(id) ?? new Set<BenchEventHandlers>();
    this.#subs.set(id, subs);
    subs.add(h);
    return () => subs.delete(h);
  }

  tick(id: string, now: number) {
    const b = this.#benches.find((x) => x.id === id);
    if (b) this.advance(now);
  }

  #emit(id: string, f: (h: BenchEventHandlers) => void) {
    for (const h of this.#subs.get(id) ?? []) f(h);
  }
}

/** What an agent finds, by agent and model: no confidences, high precision, recall by how capable the model is. */
export function agentSkill(agent: 'claude' | 'codex', model: string | null, effort: string, salt: string): Skill {
  const r = rng(hash(`agent:${salt}`));
  const q = agent === 'claude' ? ({ haiku: 0.62, sonnet: 0.72, opus: 0.78 } as Record<string, number>)[model ?? 'haiku'] ?? 0.66 : 0.6;
  const lift = effort === 'high' ? 0.04 : effort === 'medium' ? 0.02 : 0;
  return {
    map: 0.22 + (q + lift) * 0.22 + r() * 0.02,
    fragility: 0.55,
    fragile: { color_cast: 0.4, low_contrast: 0.6 },
    ece: 0,
    iou: 0.7 + r() * 0.03,
    selfRecall: 0,
    point: { precision: 0.88 + (q + lift) * 0.08, recall: q + lift - 0.04 + r() * 0.04 },
  };
}

/** What an agent's calls use for `passes` image passes, as agent labeling's calls do (./label). */
export function agentUsage(agent: 'claude' | 'codex', model: string | null, effort: string, passes: number, description: string, salt: string): LabelUsage {
  const per = PER_CALL[agent]!;
  const calls = Math.ceil(passes / per);
  const fixed = agent === 'claude' ? 420 + Math.round(description.length / 4) : 10_200;
  const input = calls * fixed + passes * 1012 + (hash(salt) % 400);
  const cached = Math.round((calls - 1) * fixed * 0.86);
  const reasoning = ({ low: 0, medium: 260, high: 820 } as Record<string, number>)[effort] ?? 0;
  const output = calls * 48 + passes * (5 * 23 + reasoning);
  const price = agent === 'claude' ? ({ sonnet: [3, 15], haiku: [1, 5], opus: [15, 75] } as Record<string, number[]>)[model ?? 'haiku'] : undefined;
  const cost = price ? Math.round((((input - cached) * price[0]! + cached * price[0]! * 0.1 + output * price[1]!) / 1e6) * 1e6) / 1e6 : null;
  return { calls, input_tokens: input, cached_input_tokens: cached, output_tokens: output, cost_usd: cost };
}

/** An external contestant's skill, from how its unchanged-image predictions match the truth (IoU 0.5, same class). */
function externalSkill(b: MBench, predictions: readonly BenchPrediction[], scored: boolean): Skill {
  let right = 0;
  let found = 0;
  let truths = 0;
  let ious = 0;
  const byImage = new Map(b.images.map((i) => [i.id, i]));
  for (const p of predictions) {
    if (p.condition) continue;
    const im = byImage.get(p.image);
    if (!im) continue;
    truths += im.boxes.length;
    const used = new Set<Box>();
    for (const d of [...p.boxes].sort((x, y) => (y.score ?? 1) - (x.score ?? 1))) {
      found++;
      let hit: Box | null = null;
      let best = 0.5;
      for (const t of im.boxes) {
        if (t.class !== d.class || used.has(t)) continue;
        const v = iou(t, d);
        if (v >= best) {
          best = v;
          hit = t;
        }
      }
      if (hit) {
        used.add(hit);
        right++;
        ious += best;
      }
    }
  }
  const precision = found ? right / found : 0;
  const recall = truths ? right / truths : 0;
  const f = f1(precision, recall);
  return {
    map: r4(Math.max(0.01, f * (scored ? 0.68 : 0.5))),
    fragility: 0.9,
    ece: 0.07,
    iou: right ? ious / right : 0.6,
    selfRecall: scored ? Math.max(0, recall - 0.15) : 0,
    point: { precision, recall },
  };
}

/** For ./benchSeed: an image list made from a source, as `create` makes it, sampled with a seed. */
export function sampleImages(images: Omit<BenchImg, 'id'>[], seed: number, salt: string, max: number): BenchImg[] {
  const r = rng(hash(`bench:${seed}:${salt}`));
  const order = images.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [order[i], order[j]] = [order[j]!, order[i]!];
  }
  return order.slice(0, max).map((k, n) => ({ ...images[k]!, id: String(n + 1) }));
}

/** A mock dataset's images as benchmark images (all of them), for a spec the benchmark owns (the pond camera). */
export function specImages(spec: MockDatasetSpec): Omit<BenchImg, 'id'>[] {
  return Array.from({ length: total(spec) }, (_, i) => {
    const d = image(spec, i);
    return { file: `${spec.name}-${String(i + 1).padStart(4, '0')}.jpg`, width: d.width, height: d.height, boxes: d.boxes, from: { spec, index: i } };
  });
}


