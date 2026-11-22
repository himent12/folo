// The benchmark mock's records (./benchState, ./bench) and the server's rules it keeps: the demo's pace, ids and
// fingerprints, the names it gives entries and loops, its limits, and the messages of a loop that cannot label.

import type {
  BenchContestant,
  BenchGeneration,
  BenchLoopPhase,
  BenchPlan,
  BenchPrediction,
  BenchSource,
  Box,
  LabelAgent,
  LabelAgentId,
  LabelUsage,
  Model,
  StartBenchLoop,
} from '../types';
import { AGENT_NAME } from '../../label';
import type { GeneratedResult, Skill } from './benchResults';
import type { MockDatasetSpec } from './datasets';
import type { MockLabeling } from './label';
import { hash, rng } from './rng';

const SEC = 1000;

/** A model entry's time per image pass, by architecture (the demo's pace, not the GPU's). */
export const MS_PER_PASS: Record<string, number> = { 'folo-n': 3, 'folo-s': 4.2, 'folo-m': 6.5, 'folo-l': 9, 'folo-x': 13 };
/** The speed a model entry reports: milliseconds per image on the server's GPU, by architecture. */
export const MS_PER_IMAGE: Record<string, number> = { 'folo-n': 2.35, 'folo-s': 4.1, 'folo-m': 7.8, 'folo-l': 11.9, 'folo-x': 18.6 };
/** Parameters by architecture, as a model reports them. */
export const PARAMS: Record<string, number> = { 'folo-n': 2_630_000, 'folo-s': 9_100_000, 'folo-m': 20_100_000, 'folo-l': 25_400_000, 'folo-x': 57_000_000 };
/** An agent answers this many images per call (as agent labeling batches them), a call every 1.5 s. */
export const PER_CALL: Record<LabelAgentId, number> = { claude: 4, codex: 8, antigravity: 8 };
export const CALL_MS = 1500;
/** A stop ends what is in flight within this long. */
export const STOP_MS = 1000;
/** A loop's labeling and its student's training, in the demo's time. */
export const LABEL_MS = 4 * SEC;
export const TRAIN_MS = 12 * SEC;

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

export const iso = (ms: number) => new Date(ms).toISOString();
export const hex = (n: number, len = 4) => (n >>> 0).toString(16).padStart(len, '0').slice(-len);
export const r4 = (x: number) => Math.round(x * 1e4) / 1e4;

/** A benchmark's fingerprint: its images, truth, plan and seed, hashed (16 hex digits). */
export function fingerprintOf(b: Pick<MBench, 'images' | 'plan' | 'seed'>): string {
  const plan = b.plan.conditions.map((c) => `${c.condition}:${c.severities.join('.')}:${c.weight}`).join(',');
  const imgs = b.images.map((i) => `${i.file}:${i.boxes.map((x) => `${x.class},${x.x1},${x.y1},${x.x2},${x.y2}`).join(';')}`).join('|');
  return hex(hash(`${imgs}#${plan}#${b.seed}`), 8) + hex(hash(`${b.seed}#${plan}#${imgs.length}`), 8);
}

/**
 * An agent entry's default name, as the server makes it (crates/folo-server/src/bench/entries.rs): the agent's name
 * with its model in parentheses, `Claude Code (haiku)`, `Codex (gpt-6.1-sol)`, `Antigravity (gemini-4-neon)`; the
 * agent's name alone without a model.
 */
export const agentEntryName = (agent: LabelAgentId, model: string | null) => (model ? `${AGENT_NAME[agent]} (${model})` : AGENT_NAME[agent]);

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
export let openBoxesMax = OPEN_BOXES_MAX;
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
