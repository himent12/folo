// Vision benchmarks' rules (docs/API.md, Vision benchmarks), shared by the forms that check what a person typed, the
// screens that order and word entries, and the in-browser mock that answers as the server would.

import type { BenchAxis, BenchConditionGroup, BenchEntry, BenchEntryStatus, BenchLoop, BenchLoopStatus, BenchSource } from '../api/types';

/** The feature a server lists in `/api/system` when it has the `/api/bench` endpoints. */
export const BENCH_FEATURE = 'vision_bench';
/** Feature advertised when `POST /api/bench/benchmarks` accepts `source: { kind: "generated", ... }` (docs/API.md). */
export const BENCH_GENERATED_FEATURE = 'bench_generated_source';
/** Synthetic `dataset_id` persisted in `Bench.source` for generated-scene benchmarks (crates/folo-server/src/bench/benchmarks.rs). */
export const GENERATED_SOURCE_ID = 'generated';
/** Defaults and bounds for generated-scene benchmarks (crates/folo-bench/src/source.rs). */
export const GENERATED_IMAGES_DEFAULT = 64;
export const GENERATED_IMGSZ_DEFAULT = 640;
export const GENERATED_IMGSZ_MIN = 32;
export const GENERATED_IMGSZ_MAX = 4096;
export const GENERATED_CLASSES_MAX = 8;
export const GENERATED_CLASSES_DEFAULT = ['circle', 'square', 'triangle'] as const;

/** True when a persisted `Bench.source` came from deterministic generated scenes rather than a registered dataset. */
export function isGeneratedBenchSource(s: BenchSource): boolean {
  return s.kind === 'dataset' && s.dataset_id === GENERATED_SOURCE_ID && (s.split === null || s.split === undefined);
}

/** Human-readable `source_name` matching `folo_bench::GeneratedSourceSpec::display_name`. */
export function generatedSourceName(spec: { images: number; imgsz: number; seed: number; vary_aspect?: boolean }): string {
  const aspect = spec.vary_aspect ? ', varied aspect' : '';
  return `generated (${spec.images}×${spec.imgsz}px${aspect}, seed ${spec.seed})`;
}

/** Severities run from 1 (barely visible) to 5 (severe, still solvable by a person). */
export const MAX_SEVERITY = 5;
export const SEVERITIES = [1, 2, 3, 4, 5] as const;

/** A benchmark's name, as a labeling job's: not blank, at most this long. */
export const BENCH_NAME_MAX = 100;
export const DESCRIPTION_MAX = 4000;
/** A benchmark holds 1 to 5000 images, 500 when the request names no limit. */
export const IMAGES_MAX = 5000;
export const IMAGES_DEFAULT = 500;
/** A plan condition's weight: `0 < weight <= 10`. */
export const WEIGHT_MAX = 10;
/** An agent entry's images when the request names none. */
export const AGENT_IMAGES = 48;
/** The precision a contestant's own labels must reach for the self-labeling axis. */
export const TARGET_PRECISION = 0.9;
/** A loop runs 1 to 5 generations. */
export const GENERATIONS_MAX = 5;
/** A design request's note, as a labeling run's (at most 2000 characters). */
export const DESIGN_NOTE_MAX = 2000;
/** Most images one `…/images` request returns. */
export const IMAGES_PAGE_MAX = 1000;

/** The axes of the score in display order, with their weights (docs/BENCH.md, The score). */
export const AXES: readonly BenchAxis[] = ['accuracy', 'robustness', 'self_labeling', 'calibration', 'localization'];
export const AXIS_WEIGHT: Record<BenchAxis, number> = { accuracy: 0.35, robustness: 0.25, self_labeling: 0.2, calibration: 0.1, localization: 0.1 };
export const AXIS_NAME: Record<BenchAxis, string> = {
  accuracy: 'Accuracy',
  robustness: 'Robustness',
  self_labeling: 'Self-labeling',
  calibration: 'Calibration',
  localization: 'Localization',
};
/** What each axis asks, in one line (the catalog's own description is the server's; this is the screens' short one). */
export const AXIS_ASKS: Record<BenchAxis, string> = {
  accuracy: 'Does it find the objects, with tight boxes?',
  robustness: 'How much of that survives hard conditions?',
  self_labeling: 'Could its own labels train a model?',
  calibration: 'Can you trust its confidence?',
  localization: 'How tight are its correct boxes?',
};

/** Condition groups in display order, with their names. */
export const GROUPS: readonly BenchConditionGroup[] = ['light', 'blur', 'noise', 'medium', 'digital', 'geometry'];
export const GROUP_NAME: Record<BenchConditionGroup, string> = {
  light: 'Light',
  blur: 'Blur',
  noise: 'Noise',
  medium: 'Medium',
  digital: 'Digital',
  geometry: 'Geometry',
};

const ACTIVE_ENTRY: readonly BenchEntryStatus[] = ['queued', 'running', 'stopping'];
const ACTIVE_LOOP: readonly BenchLoopStatus[] = ['running', 'stopping'];

/** True while the entry waits or runs. */
export const isEntryActive = (e: Pick<BenchEntry, 'status'>) => ACTIVE_ENTRY.includes(e.status);
/** True while the loop runs. */
export const isLoopActive = (l: Pick<BenchLoop, 'status'>) => ACTIVE_LOOP.includes(l.status);

/** A finished entry with a score: one the leaderboard ranks. */
const ranked = (e: BenchEntry) => e.status === 'done' && !!e.score;

/**
 * The leaderboard's order, as the server sends it (docs/API.md, entries): complete finished entries (`score.partial`
 * false) by score, then partial finished ones by score (each a tie broken by age, the older first: it got there first),
 * then every other entry, newest first. A partial entry never ranks above a complete one: its number compares only
 * loosely.
 */
export function rankEntries(entries: readonly BenchEntry[]): BenchEntry[] {
  const byScore = (a: BenchEntry, b: BenchEntry) => (b.score?.total ?? 0) - (a.score?.total ?? 0) || a.created.localeCompare(b.created) || a.id.localeCompare(b.id);
  const complete = entries.filter((e) => ranked(e) && !e.score!.partial).sort(byScore);
  const partial = entries.filter((e) => ranked(e) && e.score!.partial).sort(byScore);
  const rest = entries.filter((e) => !ranked(e)).sort((a, b) => b.created.localeCompare(a.created) || b.id.localeCompare(a.id));
  return [...complete, ...partial, ...rest];
}

/** What is wrong with a benchmark's name, or null. */
export function benchNameError(name: string): string | null {
  const n = name.trim();
  if (!n) return 'A benchmark needs a name.';
  if (n.length > BENCH_NAME_MAX) return `At most ${BENCH_NAME_MAX} characters.`;
  return null;
}

/** What is wrong with a description, or null. */
export function descriptionError(text: string): string | null {
  return text.length > DESCRIPTION_MAX ? `At most ${DESCRIPTION_MAX.toLocaleString('en-US')} characters (now ${text.length.toLocaleString('en-US')}).` : null;
}

/** Whether an entry in this state freezes its benchmark's plan: one that finished, or is under way. */
export const planFreezes = (status: BenchEntryStatus) => status === 'done' || ACTIVE_ENTRY.includes(status);

/**
 * Whether a benchmark's plan can still change (docs/API.md, `409 bench_frozen`): it has no entry that finished or is
 * under way. A failed or stopped entry graded nothing that a new plan would make incomparable.
 */
export const planEditable = (entries: readonly Pick<BenchEntry, 'status'>[]) => !entries.some((e) => planFreezes(e.status));
