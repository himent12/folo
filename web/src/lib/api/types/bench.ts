// Vision benchmarks: plans, entries, results and self-improvement loops (docs/API.md, Vision benchmarks).

import type { Box, Split } from './datasets';
import type { LabelAgentId, LabelEffort, LabelUsage } from './label';
import type { Arch } from './models';
import type { Time } from './time';

// ---------- Vision benchmarks (docs/API.md); servers that have them list `vision_bench` in `SystemInfo.features` ----------
// `folo_api::bench` field for field (docs/BENCH.md defines every number).

/** What kind of trouble a `BenchCondition` is. */
export type BenchConditionGroup = 'light' | 'blur' | 'noise' | 'medium' | 'digital' | 'geometry';

/** One stress condition: a deterministic change to an image that keeps its size, at severities 1 to 5. */
export interface BenchCondition {
  /** Stable id, e.g. `haze`; what a `BenchPlanCondition` names. */
  id: string;
  /** Display name, e.g. `Haze`. */
  name: string;
  group: BenchConditionGroup;
  /** What it does to an image and what it stands for, in a sentence. */
  description: string;
}

/** An axis of the score. */
export type BenchAxis = 'accuracy' | 'robustness' | 'self_labeling' | 'calibration' | 'localization';

/** An axis as the catalog lists it. */
export interface BenchAxisInfo {
  axis: BenchAxis;
  /** Display name, e.g. `Self-labeling`. */
  name: string;
  /** Its share of the score when every axis is measured; the five sum to 1. */
  weight: number;
  /** What it measures, in a sentence. */
  description: string;
}

/** `GET /api/bench/catalog`: what a benchmark can be made of and how it is scored. */
export interface BenchCatalog {
  /** Every stress condition, in display order. */
  conditions: BenchCondition[];
  /** The highest severity (5). */
  severities: number;
  /** The plan a benchmark gets when it names none (every condition at 1, 3 and 5, weight 1). */
  default_plan: BenchPlan;
  /** The axes of the score, in display order, with their weights. */
  axes: BenchAxisInfo[];
}

/** One condition of a `BenchPlan`. */
export interface BenchPlanCondition {
  /** A `BenchCondition.id`. */
  condition: string;
  /** The severities to run: 1 to 5 distinct values in `1..=5`, ascending. */
  severities: number[];
  /** How much the condition counts in the robustness axis, relative to the others (`0 < weight <= 10`; default 1). */
  weight: number;
  /** Why it is in the plan, when someone said (an agent's design does). */
  why?: string | null;
}

/** The agent behind a designed `BenchPlan`. */
export interface BenchDesigner {
  agent: LabelAgentId;
  /** The model it was asked for; null means the CLI's default. */
  model: string | null;
}

/** Which conditions a benchmark runs, how hard and how much each counts. */
export interface BenchPlan {
  /** Each condition at most once. Empty: the unchanged images only (no robustness axis). */
  conditions: BenchPlanCondition[];
  /** Why this plan fits the benchmark, when someone said (an agent's design does). */
  rationale?: string | null;
  /** The agent that proposed the plan; null for the default plan and one written by hand. */
  designed_by?: BenchDesigner | null;
}

/**
 * Where a set of images comes from: a benchmark's images with their truth, or a self-improvement loop's pool (whose
 * labels are never read). A benchmark's split defaults to `val`, a pool's to `train`.
 */
export type BenchSource = { kind: 'dataset'; dataset_id: string; split?: Split | null } | { kind: 'label_job'; job_id: string };

/** The leading entry of a `Bench`. */
export interface BenchBest {
  entry_id: string;
  name: string;
  /** `0..=100`. */
  score: number;
  /** The leader's `score.partial`: its number compares only loosely with complete entries. */
  partial: boolean;
}

/**
 * A benchmark: images with their true boxes, copied in and frozen, and the plan of conditions to run them under. Its
 * images and truth never change; its plan can change only while it has no entries.
 */
export interface Bench {
  /** Opaque id, e.g. `bn_3f9a`. */
  id: string;
  name: string;
  /** Where the camera is and what makes the job hard, in the user's words (may be empty). An agent designing the plan reads it. */
  description: string;
  /** A box's `class` indexes this list. */
  classes: string[];
  created: Time;
  source: BenchSource;
  /** The source's display name when the benchmark was made (the dataset's or the labeling job's). */
  source_name: string;
  images: number;
  /** True boxes over all images. */
  boxes: number;
  /** True boxes per class, aligned with `classes`. */
  class_boxes: number[];
  plan: BenchPlan;
  /** Seed of everything random in the benchmark (which images were sampled, where noise falls). */
  seed: number;
  /** A hash of the images, truth, plan and seed, as hex: two entries compare when their fingerprints agree. */
  fingerprint: string;
  /** Entries, whatever their status. */
  entries: number;
  /** The finished entry with the highest score; null before the first. */
  best: BenchBest | null;
  /** An entry or a loop of it is queued, running or stopping (exactly when DELETE answers `409 bench_active`). */
  active: boolean;
}

/** `POST /api/bench/benchmarks`. */
export interface CreateBench {
  name: string;
  /** At most 4000 characters. */
  description?: string;
  source: BenchSource;
  /** At most this many images (1 to 5000, default 500), sampled with `seed` when the source has more. */
  max_images?: number | null;
  /** The sample's seed (default 0): a whole number from 0 to 2^53 − 1 (JSON's exact integers; larger is a `400`). */
  seed?: number;
  /** Absent: the catalog's `default_plan`. */
  plan?: BenchPlan | null;
}

/** `PUT /api/bench/benchmarks/{id}`: absent fields stay. `plan` only while the benchmark has no entries (`409 bench_frozen`). */
export interface UpdateBench {
  name?: string;
  description?: string;
  plan?: BenchPlan;
}

/** `POST /api/bench/benchmarks/{id}/design`: ask a coding agent to propose the plan. */
export interface DesignBench {
  agent: LabelAgentId;
  /** Absent or null: the agent's default. */
  model?: string | null;
  /** Absent or null: the agent's default. */
  effort?: LabelEffort | null;
  /** Anything to add to the benchmark's description for this request. */
  note?: string;
}

/** The answer of `POST …/design`: a proposal (valid as it is, `designed_by` and `rationale` set), not saved until it is `PUT`. */
export interface BenchDesign {
  plan: BenchPlan;
  /** What the call used. */
  usage: LabelUsage;
}

/** One image of a `Bench` with its truth. */
export interface BenchImage {
  /** Opaque, unique within the benchmark. */
  id: string;
  /** File name, unique within the benchmark. */
  file: string;
  /** Pixels (after EXIF orientation); the same under every condition. */
  width: number;
  height: number;
  /** The true boxes, pixel xyxy, under the condition asked for (a geometric condition moves them). */
  boxes: Box[];
}

/** `GET /api/bench/benchmarks/{id}/images`: `limit` at most 1000; `condition` and `severity` go together. */
export interface BenchImagesQuery {
  offset?: number;
  limit?: number;
  condition?: string;
  severity?: number;
}

/** `GET /api/bench/benchmarks/{id}/images`. */
export interface BenchImagePage {
  /** All images of the benchmark (before offset / limit). */
  total: number;
  /** This page, in the benchmark's order. */
  images: BenchImage[];
}

/** What `benchImageUrl` asks for: the image under a condition (both or neither), and a thumbnail size. */
export interface BenchImageOptions {
  condition?: string;
  severity?: number;
  thumb?: number;
}

/** A detection: a box, its class and how sure the contestant is. */
export interface BenchDetection extends Box {
  /** Confidence in `0..=1`; 1 when absent, and for a contestant that gives none. */
  score: number;
}

/** The detections of one image under one condition, as an external contestant submits them. */
export interface BenchPrediction {
  /** A `BenchImage.id`. */
  image: string;
  /** The condition the image was fetched under; absent or null for the unchanged image. */
  condition?: string | null;
  /** Its severity (required with `condition`). */
  severity?: number | null;
  /** What the model found, in pixels of the image as it was served; `score` may be left out (1). */
  boxes: (Box & { score?: number })[];
}

/** Who a new entry is (`StartBenchEntry.contestant`). */
export type BenchContestantSpec =
  /** A model of this server, run by FOLO. */
  | { kind: 'model'; model_id: string }
  /** A coding agent on the server's machine, asked for boxes as agent labeling asks. */
  | { kind: 'agent'; agent: LabelAgentId; model?: string | null; effort?: LabelEffort | null }
  /**
   * Any other vision model: its predictions, made wherever it runs. `scored: false` when it gives no confidences
   * (default true). Every unchanged image must be answered for; a condition of the plan is answered for every image at
   * every planned severity, or left out. One record carries at most 1000 boxes.
   * `open: false` (the default): `predictions` is the whole entry, scored in the request (up to 64 MiB). `open: true`:
   * the entry is created `running`, to be filled in parts (`POST …/entries/{entry_id}/predictions`, `BenchPredictions`)
   * and scored by `POST …/entries/{entry_id}/finish`; `predictions` here, if any, is its first part. An open entry is
   * active (it freezes the plan); stopping it discards what it received and leaves it `stopped` until deleted; with no
   * part for 30 minutes it fails ("no predictions arrived for 30 minutes"); it holds at most 20 million boxes (a `400`
   * says to send pass by pass).
   */
  | { kind: 'external'; scored?: boolean; open?: boolean; predictions?: BenchPrediction[] };

/**
 * `POST /api/bench/benchmarks/{id}/entries/{entry_id}/predictions`: a part of an open external entry (up to 64 MiB),
 * all or nothing: `400 invalid_body` names the first bad record (a record for an image, condition and severity already
 * received with other boxes is one) and adds nothing. A record received before with the same boxes is taken again and
 * adds nothing, so a part whose answer was lost can be sent again. `409 bench_entry_not_open` on any entry that is not
 * an open external one.
 */
export interface BenchPredictions {
  predictions: BenchPrediction[];
}

/** `POST /api/bench/benchmarks/{id}/entries`: run a contestant on the benchmark. */
export interface StartBenchEntry {
  contestant: BenchContestantSpec;
  /**
   * The name on the leaderboard (one line, at most 100 characters); absent uses the model's name, or the agent's with
   * its model in parentheses, `Claude Code (haiku)`, `Codex (<model>)` (the agent's name alone when it has no default
   * model). Required for an external contestant.
   */
  name?: string | null;
  /** Only the first this many images (a seeded shuffle, so a prefix is a fair sample). Absent: all for a model, 48 for an agent. Not for external. */
  max_images?: number | null;
  /** Whether to run the plan's conditions. Absent: true for a model, false for an agent. Not for external. */
  conditions?: boolean | null;
}

/** Who an entry is. */
export type BenchContestant =
  /** A model of this server (it may have been deleted since). */
  | { kind: 'model'; model_id: string; arch: Arch; params: number; imgsz: number }
  /** A coding agent; `model` null means the CLI's default. */
  | { kind: 'agent'; agent: LabelAgentId; model: string | null; effort: LabelEffort }
  /** Predictions made elsewhere. */
  | { kind: 'external' };

/** Where a `BenchEntry` stands; `queued`, `running` and `stopping` are active. */
export type BenchEntryStatus = 'queued' | 'running' | 'stopping' | 'done' | 'failed' | 'stopped';

/** How far a running entry is. */
export interface BenchProgress {
  /** Images answered so far, over all conditions; for an open external entry, the records received so far. */
  done: number;
  /** The entry's images times one plus its conditions' severities (for an open external entry, the plan's). */
  total: number;
  /** The condition being run; null for the unchanged images. */
  condition: string | null;
  severity: number | null;
}

/** How fast a model contestant ran on the server (once `done`). Not part of the score: it depends on the machine. */
export interface BenchSpeed {
  /** e.g. `gpu:0`. */
  device: string;
  /**
   * Wall time per image on the unchanged images, decode to boxes, in milliseconds: run in batches (images prepared on
   * several threads while the previous batch runs) and counted from the end of the first batch, so loading the model
   * and warming up the device are left out. A throughput, not the latency of a single image.
   */
  ms_per_image: number;
}

/** One axis of a `BenchScore`. */
export interface BenchAxisScore {
  axis: BenchAxis;
  /** `0..=1`; null when it was not measured. */
  value: number | null;
  /** Its share of the score. */
  weight: number;
}

/** The score of an entry: one number and what it is made of. */
export interface BenchScore {
  /** `0..=100`: the weighted mean of the measured axes, times 100. */
  total: number;
  /** An axis was not measured, or the entry ran a part of the images: it compares loosely with complete entries. */
  partial: boolean;
  /** Every axis, in display order (accuracy, robustness, self-labeling, calibration, localization). */
  axes: BenchAxisScore[];
}

/** A contestant's run of a benchmark, on its leaderboard. */
export interface BenchEntry {
  /** Opaque id, e.g. `be_91c2`. */
  id: string;
  bench_id: string;
  /** Name on the leaderboard. */
  name: string;
  contestant: BenchContestant;
  status: BenchEntryStatus;
  created: Time;
  /** When it finished, failed or was stopped; null until then. */
  finished: Time | null;
  progress: BenchProgress;
  /** Images it runs: all of the benchmark's, or a prefix. */
  images: number;
  /** Whether it runs the plan's conditions. */
  conditions: boolean;
  /** Whether the contestant gives confidences. */
  scored: boolean;
  /** The benchmark's fingerprint when the entry ran. */
  fingerprint: string;
  /** Classes of the benchmark the contestant does not know (a model matched by class name); its AP on them is 0. */
  missing_classes: string[];
  /** Null until `done`. */
  score: BenchScore | null;
  /** mAP@0.5:0.95 on the unchanged images, repeated here for lists; null until `done`. */
  map50_95: number | null;
  /** The 95% interval of `map50_95` (`BenchAccuracy.ci95`), for lists; null until `done`. */
  ci95: [number, number] | null;
  /** How many of the plan's conditions it ran completely; 0 when none ran or it has not finished. */
  conditions_run: number;
  /** A model contestant, once `done`. */
  speed?: BenchSpeed | null;
  /** What an agent contestant's calls used. */
  usage?: LabelUsage | null;
  /** Why it failed (`status: failed`), else null. */
  error: string | null;
  /** How to fix `error`, when there is something useful to say. */
  hint?: string | null;
}

/** `GET …/entries/{id}/result`: accuracy on one class. */
export interface BenchClassAccuracy {
  /** Class index. */
  class: number;
  /** True boxes of the class. */
  truths: number;
  /** AP@0.5. */
  map50: number;
  /** AP@0.5:0.95. */
  map50_95: number;
}

/** Accuracy on the unchanged images (docs/METRICS.md for AP). */
export interface BenchAccuracy {
  /** mAP@0.5:0.95: the accuracy axis. */
  map50_95: number;
  /** Its 95% bootstrap interval over images, `[low, high]`. */
  ci95: [number, number];
  map50: number;
  map75: number;
  /** Class-mean precision, recall and F1 at `best_conf`. */
  precision: number;
  recall: number;
  f1: number;
  /** The confidence with the best class-mean F1; 1 when no confidence gives a right detection (`f1` 0: no best F1). */
  best_conf: number;
  /** mAP@0.5:0.95 of objects under 32² px, 32²–96² px, over 96² px; null when the benchmark has none. */
  map_small: number | null;
  map_medium: number | null;
  map_large: number | null;
  /** Per class with at least one true box, in class order. */
  per_class: BenchClassAccuracy[];
}

/** Accuracy at one severity of a condition. */
export interface BenchSeverityPoint {
  /** `1..=5`. */
  severity: number;
  map50_95: number;
  map50: number;
  /** `min(1, map50_95 / the unchanged images' map50_95)`; 0 when that is 0. */
  retention: number;
}

/** One condition's result. */
export interface BenchConditionResult {
  /** The condition's id. */
  condition: string;
  /** Its weight in the plan. */
  weight: number;
  /** The mean of its severities' `retention`. */
  retention: number;
  /** Per severity run, ascending. */
  points: BenchSeverityPoint[];
}

/** Accuracy under the plan's conditions. */
export interface BenchRobustness {
  /** The robustness axis: the weighted mean of the conditions' `retention`. */
  retention: number;
  /** Per condition, in plan order (only those that ran). */
  conditions: BenchConditionResult[];
}

/** One confidence bin of the reliability diagram. */
export interface BenchCalibrationBin {
  lo: number;
  hi: number;
  /** Detections in the bin. */
  count: number;
  /** Their mean confidence; null for an empty bin. */
  confidence: number | null;
  /** The share of them that are right; null for an empty bin. */
  precision: number | null;
}

/** Reliability of the confidences on the unchanged images. */
export interface BenchCalibration {
  /** Expected calibration error, `0..=1` (0 is perfect); the axis is `1 - ece`. */
  ece: number;
  /** Detections that entered it (confidence at least 0.1). */
  detections: number;
  /** Ten bins of confidence, `[0, 0.1)` to `[0.9, 1]`. */
  bins: BenchCalibrationBin[];
}

/** Box quality of the correct detections on the unchanged images. */
export interface BenchLocalization {
  /** Mean IoU of the correct detections with their true boxes, `0.5..=1`; null when nothing was correct. The axis is `(mean_iou - 0.5) / 0.5`. */
  mean_iou: number | null;
  /** Correct detections it is the mean of. */
  matches: number;
  /** How many fall in each tenth of the IoU range 0.5..1: `[0.5, 0.55)`, …, `[0.95, 1]` (ten counts). */
  histogram: number[];
}

/** Whether a contestant's own labels are good enough to train on. */
export type BenchReadiness = 'ready' | 'partial' | 'not_ready';

/** One confidence threshold of `BenchSelfLabeling.curve`. */
export interface BenchLabelPoint {
  /** Detections with at least this confidence are kept. */
  confidence: number;
  /** The share of them that are right; null when none is kept. */
  precision: number | null;
  /** The share of all true boxes they find. */
  recall: number;
  /** How many are kept. */
  labels: number;
}

/** What the contestant's own labels would be worth, answered on images whose truth is known. */
export interface BenchSelfLabeling {
  /** The precision labels must reach (0.9). */
  target_precision: number;
  /**
   * The lowest confidence on the grid at which the kept detections are that precise and at least 10 of them are kept
   * (a precision on fewer is no support); null when none is (and without confidences, whose one operating point
   * reaches the target only with at least 10 detections).
   */
  threshold: number | null;
  /** Precision of the labels at `threshold` (of all detections when there is none). */
  precision: number;
  /** The share of all true boxes those labels find: the self-labeling axis, 0 when the target is not reached. */
  recall: number;
  /** Mean IoU of the right ones with their true boxes; null when there are none. */
  mean_iou: number | null;
  /** Labels per image at `threshold` (of all detections when there is none). */
  labels_per_image: number;
  /** Labels (detections) kept at `threshold`; all detections when there is none. */
  labels: number;
  readiness: BenchReadiness;
  /** Precision and recall at each confidence 0.00, 0.01, …, 1.00 (one point without confidences). */
  curve: BenchLabelPoint[];
}

/** `GET …/entries/{id}/result`: everything measured, for a finished entry (`409 bench_entry_not_done` before). */
export interface BenchResult {
  entry: BenchEntry;
  accuracy: BenchAccuracy;
  /** Null when the conditions were not run. */
  robustness: BenchRobustness | null;
  /** Null without confidences. */
  calibration: BenchCalibration | null;
  localization: BenchLocalization;
  self_labeling: BenchSelfLabeling;
}

/** One image of a finished entry: the truth and what the contestant found, unchanged image. */
export interface BenchEntryImage {
  /** The image with its true boxes. */
  image: BenchImage;
  /** The detections at or above the entry's `best_conf` (every detection when its `f1` is 0), highest confidence first. */
  detections: BenchDetection[];
  /** True boxes none of them matches (IoU at least 0.5, same class). */
  missed: number;
  /** Detections that match no true box. */
  wrong: number;
}

/** `GET …/entries/{id}/images`: `order` is the benchmark's order, `worst` the most `missed + wrong` first. */
export interface BenchEntryImagesQuery {
  sort?: 'order' | 'worst';
  offset?: number;
  limit?: number;
}

/** `GET …/entries/{id}/images`. */
export interface BenchEntryImagePage {
  /** Images the entry ran (before offset / limit). */
  total: number;
  images: BenchEntryImage[];
}

/** How the students of a loop train: the fields of `TrainConfig` a loop lets you choose. */
export interface BenchStudent {
  /** Absent: the first teacher's. A larger one is "the next bigger model". */
  arch?: Arch | null;
  /** Epochs per generation (default 30). */
  epochs?: number | null;
  /** Default: the first teacher's. */
  imgsz?: number | null;
  batch?: number | null;
  /** Default `auto`. */
  device?: string | null;
  /** Starting weights as `TrainConfig.weights` takes them; absent trains from scratch. */
  weights?: string | null;
  /** Default 0. */
  seed?: number;
}

/**
 * `POST /api/bench/benchmarks/{id}/loops`: each generation, the current model labels the pool by itself, a student
 * trains on those labels alone, and the benchmark grades the student, the next generation's teacher.
 */
export interface StartBenchLoop {
  /**
   * Absent: `<teacher's name> loop`. Its generations' datasets and training runs are named `<name> gen <n>`, so each
   * student model is `<name> gen <n>-best` (a run's best checkpoint), and so is its benchmark entry.
   */
  name?: string | null;
  /** The first teacher: a model of this server whose classes are the benchmark's (same names, same order). */
  teacher: string;
  /** The images the teachers label; its own labels are never read, images also in the benchmark are left out. */
  pool: BenchSource;
  /** 1 to 5 (default 1). */
  generations?: number | null;
  /**
   * Keep a teacher's detections from this confidence up; absent: the threshold its benchmark entry found. A loop never
   * guesses one: with none set and no threshold in the teacher's entry, it fails before labeling (its `hint` says to
   * set this). A teacher that keeps a box on fewer than 10 pool images at it fails the loop at labeling.
   */
  min_confidence?: number | null;
  student?: BenchStudent;
}

/** Where a `BenchLoop` stands; `running` and `stopping` are active. */
export type BenchLoopStatus = 'running' | 'stopping' | 'done' | 'failed' | 'stopped';

/** What a running loop is doing. */
export type BenchLoopPhase = 'benchmarking' | 'labeling' | 'training';

/** The last generation against the first teacher on mAP@0.5:0.95: above its 95% interval, inside it, below it. */
export type BenchLoopVerdict = 'improved' | 'held' | 'regressed';

/**
 * The labels a teacher made of a loop's pool. A teacher's silence is not a label: a pool image it kept no box on is left
 * out of the student's dataset (never written as a background image), so the dataset is `labeled_images` images. With
 * fewer than 10 the loop fails at labeling instead (its `error` says how many boxes on how many images, at which
 * confidence; its `hint` points at a lower `min_confidence`), and the generation has no labels.
 */
export interface BenchLabels {
  /** The confidence from which detections were kept. */
  threshold: number;
  /** Images of the pool the teacher looked at (the pool less the benchmark's own images). */
  images: number;
  /** Of those, the images it kept at least one box on: the student's dataset (train and val). */
  labeled_images: number;
  /** Boxes kept, on those images. */
  boxes: number;
  /** Pool images left out because the benchmark has them. */
  excluded: number;
  /** The dataset written from them (train and val both from the pool, both with the teacher's labels). */
  dataset_id: string;
}

/** One generation of a `BenchLoop`: 0 is the first teacher, then one per student. */
export interface BenchGeneration {
  generation: number;
  /** The labels its teacher made for it; null for generation 0 and until they are made. */
  labels: BenchLabels | null;
  /** The training run of this generation's model; null for generation 0 and until it starts. */
  run_id: string | null;
  /** The generation's model; null until its training has finished. */
  model_id: string | null;
  model_name: string | null;
  /** Its benchmark entry; null until it is started. */
  entry_id: string | null;
  /** Its score; null until the entry is done. */
  score: number | null;
  /** Its mAP@0.5:0.95 on the unchanged images, and that number's 95% interval. */
  map50_95: number | null;
  ci95: [number, number] | null;
}

/** A self-improvement loop of a benchmark. */
export interface BenchLoop {
  /** Opaque id, e.g. `bl_07d1`. */
  id: string;
  bench_id: string;
  name: string;
  status: BenchLoopStatus;
  /** Null once it has ended. */
  phase: BenchLoopPhase | null;
  created: Time;
  /** Null while it runs. */
  finished: Time | null;
  /** The request, defaults filled in. */
  config: StartBenchLoop;
  /** Generation 0 (the first teacher), then one per student, as far as the loop has come. */
  generations: BenchGeneration[];
  /** Null until `done`. */
  verdict: BenchLoopVerdict | null;
  /** Why it failed (`status: failed`), else null. */
  error: string | null;
  /** How to fix `error`, when there is something useful to say. */
  hint?: string | null;
}

/** SSE events on `GET /api/bench/benchmarks/{id}/events`, keyed by the SSE `event:` name. */
export interface BenchEventMap {
  /** The benchmark after a change (name, plan, entry count, best entry); also first on connect, and to a subscriber that fell behind (load the lists again). */
  bench: Bench;
  /** An entry after a change (created, progress at most twice a second, done, failed). */
  entry: BenchEntry;
  loop: BenchLoop;
  /** An entry or a loop was deleted. */
  removed: { id: string };
}
