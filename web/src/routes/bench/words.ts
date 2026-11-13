// StatusBadge's OwnStatus is restated here (structurally the same), so this module needs no component import.
// What the benchmark screens say about a benchmark, an entry, a loop and their numbers: one wording everywhere (the
// leaderboard, the entry's page, the compare page, the palette), in docs/BENCH.md's vocabulary.

import type {
  Bench,
  BenchAxis,
  BenchAxisScore,
  BenchCalibration,
  BenchCalibrationBin,
  BenchCatalog,
  BenchEntry,
  BenchGeneration,
  BenchLabelPoint,
  BenchLabels,
  BenchLoop,
  BenchLoopPhase,
  BenchLoopVerdict,
  BenchReadiness,
  BenchResult,
  BenchSelfLabeling,
  BenchSource,
  LabelAgent,
  Model,
  RunStatus,
} from '$lib/api/types';
import { AXIS_NAME, TARGET_PRECISION } from '$lib/bench/rules';
import { MIN_LABELS } from '$lib/bench/score';
import { compact, conf, metric, ms, num, pct, sentence, usd } from '$lib/format';
import { AGENT_NAME } from '$lib/label';
import { modelLabel, tokens } from '../label/words';

/** StatusBadge's `OwnStatus` (ui/StatusBadge.tsx). */
interface OwnStatus {
  status: RunStatus;
  label: string;
}

/**
 * The fewest decimals (from `min`, at most `max`) at which `a` and `b` print differently, so a sentence that says one
 * is above or below the other is true as printed ("0.8661 is above 0.8660", never "0.866 is above … 0.866").
 */
export function digitsApart(a: number, b: number, min = 3, max = 7): number {
  for (let d = min; d < max; d++) if (Math.round(a * 10 ** d) !== Math.round(b * 10 ** d)) return d;
  return max;
}

/** A share as a reader can see it: "29%", a small one with its decimal ("0.2%"), never a bare "0%" that is not 0. */
export const share = (x: number) => (x > 0 && x < 0.01 ? pct(x, 1) : pct(x, 0));

/** A share printed with as many decimals as it takes to differ from `ref` ("89.6%" beside "90%", never "90%" twice). */
export function shareApart(x: number, ref: number): string {
  if (x === ref) return share(x);
  const d = Math.max(x > 0 && x < 0.01 ? 1 : 0, digitsApart(x * 100, ref * 100, 0, 3));
  return pct(x, d);
}

/** A share with a decimal under 10% ("1.6%", "0.4%"), whole above ("62%"): a small share is read, not rounded away. */
export const fine = (x: number) => (x > 0 && x < 0.1 ? pct(x, 1) : pct(x, 0));

/** "1,363 labels", "1 label". */
export const labelsText = (n: number) => `${num(n)} ${n === 1 ? 'label' : 'labels'}`;

/** Labels per image as a reader can see it: "4.5 per image", "under 0.1 per image" (never "0.0" beside some labels). */
export const perImageText = (perImage: number, labels: number) => (labels > 0 && perImage < 0.05 ? 'under 0.1 per image' : `${num(perImage, 1)} per image`);

/** "71.9": a score, 0–100, one decimal. */
export const scoreText = (total: number) => num(total, 1);

/** "1 image", "360 images". */
export const imagesText = (n: number) => `${num(n)} ${n === 1 ? 'image' : 'images'}`;
export const entriesText = (n: number) => `${num(n)} ${n === 1 ? 'entry' : 'entries'}`;
export const classesText = (n: number) => `${num(n)} ${n === 1 ? 'class' : 'classes'}`;
export const conditionsText = (n: number) => `${num(n)} ${n === 1 ? 'condition' : 'conditions'}`;

/** A split's name as a sentence has it: "val" → "validation split". */
const SPLIT: Record<string, string> = { train: 'training split', val: 'validation split', test: 'test split' };

/** Where a benchmark's images came from, as two facts: the source's name and what it is ("validation split", "labeling job"). */
export function sourceFacts(b: Pick<Bench, 'source' | 'source_name'>): [string, string] {
  return [b.source_name, sourceKind(b.source)];
}

/** "validation split", "labeling job". */
export function sourceKind(s: BenchSource): string {
  return s.kind === 'dataset' ? (SPLIT[s.split ?? 'val'] ?? `${s.split} split`) : 'labeling job';
}

/** Where the source lives in Studio, when it is still there. */
export function sourcePath(s: BenchSource): string {
  return s.kind === 'dataset' ? `/datasets/${encodeURIComponent(s.dataset_id)}` : `/datasets/label/${encodeURIComponent(s.job_id)}`;
}

/** Names as a sentence lists them: "RBC", "RBC or WBC", "RBC, WBC or Platelets", "a, b, c, d or 3 more". */
export function orList(names: readonly string[], max = 4): string {
  if (names.length <= 1) return names[0] ?? '';
  const shown = names.length > max ? [...names.slice(0, max), `${num(names.length - max)} more`] : [...names];
  return `${shown.slice(0, -1).join(', ')} or ${shown.at(-1)}`;
}

/**
 * Add entry when no model of the server knows a single class of the benchmark: what is wrong, and how to get a model
 * that does (train one on the source dataset, or first make a dataset from the labeling job), with the page to go to.
 */
export function noModelHelp(b: Pick<Bench, 'classes' | 'source' | 'source_name'>): { title: string; text: string; action: { label: string; path: string } } {
  const title = `No model here knows ${orList(b.classes)}`;
  if (b.source.kind === 'dataset')
    return {
      title,
      text: `Train one on ${b.source_name}, or run a coding agent or another model's predictions instead.`,
      action: { label: `Train on ${b.source_name}`, path: `/runs/new?dataset=${encodeURIComponent(b.source.dataset_id)}` },
    };
  return {
    title,
    text: `Make a dataset from the labeling job ${b.source_name} and train a model on it, or run a coding agent or another model's predictions instead.`,
    action: { label: 'Open the labeling job', path: sourcePath(b.source) },
  };
}

/** The short fingerprint a person compares by eye: its first 4 and last 4 hex digits ("9c41…e2a0"). */
export const shortFingerprint = (f: string) => (f.length > 10 ? `${f.slice(0, 4)}…${f.slice(-4)}` : f);

/**
 * An entry's status as a status mark and its word: "Done", "Running", "Queued", "Stopping", "Failed", "Stopped" (the run
 * status whose mark it shares: a finished entry is a completed run).
 */
export function entryStatus(e: Pick<BenchEntry, 'status'> & Partial<Pick<BenchEntry, 'contestant'>>): OwnStatus {
  switch (e.status) {
    case 'done':
      return { status: 'completed', label: 'Done' };
    case 'running':
      return { status: 'running', label: receiving(e) ? 'Receiving' : 'Running' };
    case 'queued':
      return { status: 'queued', label: 'Queued' };
    case 'stopping':
      return { status: 'stopping', label: 'Stopping' };
    case 'failed':
      return { status: 'failed', label: 'Failed' };
    default:
      return { status: 'stopped', label: 'Stopped' };
  }
}

/**
 * A loop's status: while it runs, what it does and for which generation ("Benchmarking the teacher", "Labeling for gen
 * 1", "Training gen 1", "Benchmarking gen 1", read from its generations as `loopSteps` reads them); else how it ended.
 * Without its generations, the phase alone ("Labeling").
 */
export function loopStatus(l: Pick<BenchLoop, 'status' | 'phase'> & Partial<Pick<BenchLoop, 'generations'>>): OwnStatus {
  if (l.status === 'running') {
    const now = l.generations ? loopSteps({ status: l.status, phase: l.phase, generations: l.generations }) : null;
    const step = now?.steps.find((x) => x.state === 'now');
    if (now && step) return { status: 'running', label: nowLabel(now.generation, step.phase) };
    return { status: 'running', label: l.phase ? PHASE[l.phase] : 'Running' };
  }
  if (l.status === 'stopping') return { status: 'stopping', label: 'Stopping' };
  if (l.status === 'done') return { status: 'completed', label: 'Done' };
  if (l.status === 'failed') return { status: 'failed', label: 'Failed' };
  return { status: 'stopped', label: 'Stopped' };
}

export const PHASE: Record<BenchLoopPhase, string> = { benchmarking: 'Benchmarking', labeling: 'Labeling', training: 'Training' };

/** A loop's verdict in one word, and what it means in one line. */
export const VERDICT: Record<BenchLoopVerdict, { word: string; means: string }> = {
  improved: { word: 'Improved', means: "The last generation is above the first teacher's 95% interval: its own labels made a better model." },
  held: { word: 'Held', means: "The last generation is inside the first teacher's 95% interval: no measurable change." },
  regressed: { word: 'Regressed', means: "The last generation is below the first teacher's 95% interval." },
};

export const READINESS: Record<BenchReadiness, string> = { ready: 'Ready', partial: 'Partly ready', not_ready: 'Not ready' };

/**
 * Who an entry is, as facts: a model's architecture and size ("folo-s", "9.1M params"), an agent's CLI, model and
 * effort ("Claude Code", "Sonnet", "medium effort"), or "External" with whether it gave confidences. The first fact
 * of a model is an identifier (`arch`), for ui/Ident.
 */
export function contestantFacts(e: Pick<BenchEntry, 'contestant' | 'scored' | 'name'>, agents?: readonly LabelAgent[]): { arch?: string; facts: string[] } {
  const c = e.contestant;
  if (c.kind === 'model') return { arch: c.arch, facts: [`${compact(c.params)} params`] };
  if (c.kind === 'agent') {
    // What the entry's own name already says (the default "Claude Code · Sonnet") is not said again.
    const a = agents?.find((x) => x.id === c.agent);
    const model = modelLabel(c.model, a);
    const name = e.name.toLowerCase();
    return {
      facts: ['Agent', ...(name.includes(AGENT_NAME[c.agent].toLowerCase()) ? [] : [AGENT_NAME[c.agent]]), ...(name.includes(model.toLowerCase()) ? [] : [model]), `${c.effort} effort`],
    };
  }
  return { facts: ['External', e.scored ? 'with confidences' : 'no confidences'] };
}

/** "Model", "Agent", "External": what kind of contestant, in one word. */
export const KIND: Record<BenchEntry['contestant']['kind'], string> = { model: 'Model', agent: 'Agent', external: 'External' };

/**
 * What an entry was graded on, where it differs from the whole benchmark, one fact each: "48 of 360 images",
 * "unchanged only", "no confidences", "4 of 7 conditions" (with the plan's count, when the result says how many ran),
 * "knows 2 of 8 classes". Empty for an entry that ran everything. A partial score with none of these reasons (an
 * external entry that left conditions out: the entry does not say which) reads "some conditions left out".
 */
export function coverageFacts(e: BenchEntry, b: Pick<Bench, 'images' | 'classes' | 'plan'> & Partial<Pick<Bench, 'fingerprint'>>, conditionsRun: number | undefined = e.status === 'done' ? e.conditions_run : undefined): string[] {
  const axis = (a: BenchAxis) => e.score?.axes.find((x) => x.axis === a);
  const out: string[] = [];
  // Graded on an earlier version of the benchmark (another fingerprint: the engine changed): said first.
  if (b.fingerprint && e.fingerprint && e.fingerprint !== b.fingerprint && e.status === 'done') out.push(STALE_FACT);
  if (e.images < b.images) out.push(`${num(e.images)} of ${num(b.images)} images`);
  const planned = b.plan.conditions.length;
  // An open external entry has not said yet which conditions it answers: nothing to say until it is finished.
  if (planned && !e.conditions && !receiving(e)) out.push('unchanged only');
  else if (planned && conditionsRun !== undefined && conditionsRun < planned) out.push(`${conditionsRun} of ${planned} conditions`);
  if (!e.scored) out.push('no confidences');
  if (e.missing_classes.length) out.push(`knows ${b.classes.length - e.missing_classes.length} of ${b.classes.length} classes`);
  // A scored entry whose detections never reach confidence 0.1 has no calibration: partial for that, whatever it ran.
  if (e.scored && e.score && axis('calibration')?.value === null) out.push(CALIBRATION_NONE);
  // Robustness not measured though it ran the conditions: none was answered completely.
  if (planned && e.conditions && e.score && axis('robustness')?.value === null) out.push('no condition answered whole');
  if (e.score?.partial && !out.some((f) => !f.startsWith('knows') && f !== STALE_FACT)) out.push('some conditions left out');
  return out;
}

/** The fact for an entry graded on an earlier version of its benchmark (its fingerprint is not the benchmark's). */
export const STALE_FACT = 'earlier version of this benchmark';

/** What that means, and what to do (docs/BENCH.md: two entries compare when their fingerprints agree). */
export const STALE_HELP =
  'Graded on an earlier version of this benchmark (its fingerprint differs: the engine that grades changed since): its numbers do not compare with the current entries. Run it again to compare.';

/** The fact for an entry that gave confidences but none from 0.1 up: calibration could not be measured. */
export const CALIBRATION_NONE = 'calibration not measured';

/** Why an entry is partial, in a sentence (its score's tooltip and accessible description), each reason its own. */
export function partialWhy(e: BenchEntry, b: Pick<Bench, 'images' | 'classes' | 'plan'>): string {
  const f = coverageFacts(e, b)
    .filter((x) => !x.startsWith('knows') && x !== STALE_FACT)
    .map((x) => (x === CALIBRATION_NONE ? 'calibration not measured (no detection reached confidence 0.1)' : x));
  return `Partial: ${f.join(', ')}. It compares only loosely with complete entries.`;
}

/**
 * An open external entry: its sender is sending its predictions in parts (docs/API.md, `contestant.open`); it runs
 * nothing itself, so it is never worded as a model running a condition.
 */
export const receiving = (e: Pick<BenchEntry, 'status'> & Partial<Pick<BenchEntry, 'contestant'>>) => e.status === 'running' && e.contestant?.kind === 'external';

/**
 * The server's words for an API client in a failed entry's or loop's `hint`, and what Studio calls the same thing: the
 * setting by its name in New loop, the run's log instead of its endpoint.
 */
const HINT_WORDS: [RegExp, string][] = [
  [/^set min_confidence: its entry's/, 'start a new loop with “Keep labels from a lower confidence” set: its entry’s'],
  [/\ba lower min_confidence\b/, 'a lower confidence (“Keep labels from a lower confidence” in New loop)'],
  [/\bGET \/api\/runs\/\{id\}\/logs says why\b/, 'the log of its training run says why'],
  [/\ba smaller max_images costs\b/, 'fewer images cost'],
];

/**
 * A failed entry's or loop's `hint` as Studio says it, or null when nothing of it is for people. The server writes its
 * hints for API clients as well ("start the entry again: POST /api/bench/benchmarks/{id}/entries", "set
 * min_confidence: …"); the page has a button or a setting for what they ask, so a clause that names an endpoint is
 * left out and an API field becomes the setting's name (HINT_WORDS). A sentence, capitalised and ended.
 */
export function hintText(hint: string | null | undefined): string | null {
  let h = hint ?? '';
  for (const [api, words] of HINT_WORDS) h = h.replace(api, words);
  h = h
    // "start it again (POST /api/… with contestant.open: true) and send …": the bracket goes.
    .replace(/\s*\((?:GET|POST|PUT|DELETE) \/api\/[^)]*\)/g, '')
    // "start the entry again: POST /api/…", "stop it first: POST …/stop": the endpoint and what follows it go.
    .replace(/:?\s*\b(?:GET|POST|PUT|DELETE) (?:\/api\/|…\/)\S*.*$/, '')
    .trim();
  return h ? sentence(h) : null;
}

/**
 * Where a running entry is, in words: "Unchanged images" or "Haze, severity 3" (the catalog's name), and how far, as a
 * whole percent rounded down ("54%"). A queued entry: "Waiting for the GPU" (model entries run one at a time). An open
 * external entry: "Receiving predictions" and "1,200 of 6,000 records".
 */
export function progressFacts(e: BenchEntry, catalog?: BenchCatalog): { where: string; pct: string | null } {
  if (e.status === 'queued') return { where: 'Waiting for the GPU', pct: null };
  if (receiving(e)) return { where: 'Receiving predictions', pct: `${num(e.progress.done)} of ${num(e.progress.total)} records` };
  if (e.status === 'stopping') return { where: 'Stopping: ending what is in flight', pct: null };
  const p = e.progress;
  const where = p.condition ? `${catalog?.conditions.find((c) => c.id === p.condition)?.name ?? p.condition}, severity ${p.severity}` : 'Unchanged images';
  return { where, pct: p.total ? `${Math.floor((100 * p.done) / p.total)}%` : null };
}

/** A running entry's share done, 0..1. */
export const progressShare = (e: Pick<BenchEntry, 'progress'>) => (e.progress.total ? Math.min(1, e.progress.done / e.progress.total) : 0);

/**
 * What a model's "ms per image" is (docs/API.md, `speed.ms_per_image`): a throughput over the unchanged images in
 * batches, the first batch (loading the model, warming up the device) left out; not the latency of one image.
 */
export const SPEED_HELP =
  'Wall time per image over the unchanged images, run in batches with the first batch (loading the model, warming up the device) left out: a throughput on this server, not how long one image takes alone.';

/** What an entry cost or how fast it ran: a model's "2.35 ms" per image, an agent's "$0.42" (or its tokens, Codex reports no cost). */
export function costFacts(e: Pick<BenchEntry, 'speed' | 'usage' | 'contestant'>): { value: string; sub: string } | null {
  if (e.speed) return { value: ms(e.speed.ms_per_image), sub: `per image · ${e.speed.device}` };
  if (e.usage && e.usage.calls) {
    const t = `${tokens(e.usage.input_tokens + e.usage.output_tokens)} tokens`;
    return e.usage.cost_usd !== null ? { value: usd(e.usage.cost_usd), sub: `${num(e.usage.calls)} calls · ${t}` } : { value: t, sub: `${num(e.usage.calls)} calls` };
  }
  return null;
}

/** "9c41…e2a0" means: two entries compare when their fingerprints agree. */
export const FINGERPRINT_HELP = 'A hash of the images, their truth, the plan and the seed. Entries compare when their fingerprints agree.';

/** Why a plan is fixed, in one line (the conditions tab, the design action's reason). */
export const FROZEN = 'This benchmark has finished or running entries; its plan is fixed so they stay comparable.';


// ---------- reading a result: docs/BENCH.md's "The score", "Definitions" and "Reading it", said for one entry ----------

/** Half a 95% interval's width. */
export const halfWidth = (ci: readonly [number, number]) => Math.max(0, (ci[1] - ci[0]) / 2);

/**
 * The "± 0.018" of a value whose interval is symmetric about it as printed (three decimals: 0.641 in 0.623–0.659), else
 * null: the engine centres the interval's resamples on the value, which does not make the interval itself symmetric
 * (0.354 in 0.337–0.402), and a "±" would then misstate both ends.
 */
export function plusMinus(v: number, ci: readonly [number, number]): string | null {
  const m = Math.round(v * 1000);
  const lo = Math.round(ci[0] * 1000);
  const hi = Math.round(ci[1] * 1000);
  return m - lo === hi - m ? `± ${num((hi - m) / 1000, 3)}` : null;
}

/** "mAP50-95 0.641 ± 0.018" when the interval is symmetric as printed, else "mAP50-95 0.354 (0.337–0.402)". */
export function mapWithInterval(map: number, ci: readonly [number, number] | null): string {
  if (!ci) return `mAP50-95 ${metric(map)}`;
  return `mAP50-95 ${metric(map)} ${plusMinus(map, ci) ?? `(${intervalText(ci)})`}`;
}

/** "0.623–0.659". */
export const intervalText = (ci: readonly [number, number]) => `${metric(ci[0])}–${metric(ci[1])}`;

/**
 * Each axis's points of the total: its share of the score as computed (its weight over the measured axes' weights) ×
 * its value × 100, so the rows add up to the score. Null for an axis that was not measured.
 */
export function axisPoints(axes: readonly BenchAxisScore[]): Record<BenchAxis, number | null> {
  const w = axes.filter((a) => a.value !== null).reduce((s, a) => s + a.weight, 0);
  const out = {} as Record<BenchAxis, number | null>;
  for (const a of axes) out[a.axis] = a.value === null || w <= 0 ? null : (100 * a.weight * a.value) / w;
  return out;
}

/** The number behind an axis, in its own unit ("keeps 83% of its accuracy", "ECE 0.06"); null when not measured. */
export function axisRaw(
  axis: BenchAxis,
  r: Pick<BenchResult, 'accuracy' | 'robustness' | 'calibration' | 'localization' | 'self_labeling'>,
  scored: boolean,
): string | null {
  switch (axis) {
    case 'accuracy':
      return mapWithInterval(r.accuracy.map50_95, r.accuracy.ci95);
    case 'robustness':
      return r.robustness ? `keeps ${pct(r.robustness.retention, 0)} of its accuracy` : null;
    case 'self_labeling': {
      const s = r.self_labeling;
      if (s.recall > 0) return `labels ${share(s.recall)} of objects, ${labelsText(s.labels)} at ${pct(scored ? TARGET_PRECISION : s.precision, 0)} precision`;
      if (scored) return `never ${pct(TARGET_PRECISION, 0)} right on ${MIN_LABELS} labels or more`;
      if (s.labels < MIN_LABELS) return `${labelsText(s.labels)}: too few to judge`;
      return `${shareApart(s.precision, TARGET_PRECISION)} of labels right, under ${pct(TARGET_PRECISION, 0)}`;
    }
    case 'calibration':
      return r.calibration ? `ECE ${num(r.calibration.ece, 2)}` : null;
    case 'localization':
      return r.localization.mean_iou !== null ? `mean IoU ${num(r.localization.mean_iou, 2)}` : 'nothing right to measure';
  }
}

/** Why an axis was not measured for this entry, in a sentence; null when it was (or always is). */
export function axisWhyNot(axis: BenchAxis, e: Pick<BenchEntry, 'conditions' | 'scored' | 'contestant'>, b: Pick<Bench, 'plan'>): string | null {
  if (axis === 'robustness') {
    if (!b.plan.conditions.length) return 'This benchmark’s plan has no conditions, so no entry has this axis.';
    if (!e.conditions)
      return e.contestant.kind === 'agent' ? 'It ran the unchanged images only: for an agent every condition is another call per image.' : 'It ran the unchanged images only.';
    return 'No condition of the plan was answered completely.';
  }
  if (axis === 'calibration') {
    if (!e.scored)
      return e.contestant.kind === 'agent'
        ? 'Agents give no confidences: an agent answers with the boxes it is sure of, so there is nothing to calibrate.'
        : 'Its predictions came without confidences.';
    return 'No detection reached confidence 0.1.';
  }
  return null;
}

/** When axes were left out, how the others share the score. */
export function sharesNote(axes: readonly BenchAxisScore[]): string | null {
  const left = axes.filter((a) => a.value === null);
  if (!left.length) return null;
  const kept = axes.length - left.length;
  const w = axes.filter((a) => a.value !== null).reduce((s, a) => s + a.weight, 0);
  const names = left.map((a) => AXIS_NAME[a.axis]);
  const list = names.length === 1 ? names[0]! : `${names.slice(0, -1).join(', ')} and ${names.at(-1)}`;
  const n = ['no', 'one', 'two', 'three', 'four', 'five'][kept] ?? String(kept);
  const verb = kept === 1 ? 'axis carries' : 'axes share';
  return `${list} ${left.length === 1 ? 'was' : 'were'} not measured: the other ${n} ${verb} the score, each counting its weight over ${pct(w, 0)}.`;
}

/**
 * What a 95% interval says, under the number, in docs/BENCH.md's words (Accuracy): where resampling these images puts
 * the value, and how to read an overlap.
 */
export function intervalReading(ci: readonly [number, number], images: number): string {
  return `Resampling these ${imagesText(images)} puts it between ${metric(ci[0])} and ${metric(ci[1])} (95%). Entries whose intervals overlap are not clearly different on this benchmark; more images make the interval narrower.`;
}

/** How its confidence reads against how often it is right. */
export type CalibrationTone = 'close' | 'over' | 'under';

/** A bin holds enough detections to quote when it has at least this many and 5% of them all. */
const QUOTE_MIN = 20;

/**
 * The reliability diagram read in one sentence: "Overconfident: when it says 0.94, it is right 78% of the time." The
 * tone is the detections' mean gap (over: it says more than it gets right), or close when the ECE is under 0.03. A bin
 * is quoted only when it holds enough detections to mean something (QUOTE_MIN and 5% of them) and says what the tone
 * says: for over or under the bin furthest off the diagonal on that side (at least 2 points off, so the quote reads that
 * way as printed); when close, the fullest bin within 5 points of the diagonal. With no such bin the sentence quotes
 * the ECE alone, never a bin that contradicts it.
 */
export function calibrationReading(c: Pick<BenchCalibration, 'ece' | 'bins'>): { tone: CalibrationTone; sentence: string } {
  const filled = c.bins.filter((b) => b.count > 0 && b.confidence !== null && b.precision !== null);
  const n = filled.reduce((s, b) => s + b.count, 0);
  if (!n) return { tone: 'close', sentence: 'No detection reached confidence 0.1, so there is nothing to read.' };
  const gapOf = (b: BenchCalibrationBin) => b.confidence! - b.precision!;
  const meaningful = filled.filter((b) => b.count >= Math.max(QUOTE_MIN, 0.05 * n));
  const says = (b: BenchCalibrationBin) => `when it says ${num(b.confidence!, 2)}, it is right ${pct(b.precision!, 0)} of the time`;
  const ece = `ECE ${num(c.ece, 2)}`;
  if (c.ece < 0.03) {
    const near = meaningful.filter((b) => Math.abs(gapOf(b)) <= 0.05).sort((a, b) => b.count - a.count)[0];
    const far = meaningful.some((b) => Math.abs(gapOf(b)) > 0.1);
    const head = near ? `Close to the diagonal (${ece}): ${says(near)}.` : `Close to the diagonal on average (${ece}).`;
    return { tone: 'close', sentence: `${head} ${far ? 'Some of its scores are off by more: read the bins.' : 'Its confidence can be taken at its word.'}` };
  }
  const gap = filled.reduce((s, b) => s + b.count * gapOf(b), 0) / n;
  const over = gap >= 0;
  const side = meaningful.filter((b) => (over ? gapOf(b) >= 0.02 : gapOf(b) <= -0.02));
  const pick = side.reduce<BenchCalibrationBin | null>((a, b) => (!a || Math.abs(gapOf(b)) > Math.abs(gapOf(a)) ? b : a), null);
  const what = pick ? `${says(pick)}` : `on average it is right ${num(Math.abs(gap) * 100, 0)} points ${over ? 'less' : 'more'} often than it says (${ece})`;
  return over
    ? { tone: 'over', sentence: `Overconfident: ${what}. Trust its high scores less than they claim.` }
    : { tone: 'under', sentence: `Underconfident: ${what}. Its scores undersell it.` };
}

/** What a readiness verdict means (docs/BENCH.md, Self-labeling). */
export const READINESS_MEANS: Record<BenchReadiness, string> = {
  ready: 'It labels at least half of all objects by itself with 9 in 10 labels right: enough to train its successor on.',
  partial: 'It labels a quarter to half of all objects that way: its labels help, and people label the rest.',
  not_ready: 'It labels less than a quarter of all objects that way, or never gets 9 in 10 right: its own labels would not train a model.',
};

/**
 * The self-labeling readout in words, for every case: a threshold found (with the labels it rests on), none (with the
 * best supported precision, and when a few labels alone reach 90%, that they are too few), no confidences. Every share
 * as a reader can see it (never "0%" that is not 0, never "90%" beside "short of 90%"), never "0.0 labels per image"
 * beside some labels.
 */
export function selfLabelReading(
  s: Pick<BenchSelfLabeling, 'threshold' | 'precision' | 'recall' | 'mean_iou' | 'labels_per_image' | 'labels' | 'curve'>,
  scored: boolean,
): string {
  const per = perImageText(s.labels_per_image, s.labels);
  const iou = s.mean_iou !== null ? `, mean IoU ${num(s.mean_iou, 2)}` : '';
  if (!scored) {
    const head = `It gives no confidences, so it has one operating point: its ${labelsText(s.labels)} (${per}) ${s.labels === 1 ? 'is' : 'are'}`;
    if (s.recall > 0) return `${head} ${pct(s.precision, 0)} right and cover ${share(s.recall)} of all objects${iou}.`;
    if (s.labels < MIN_LABELS) return `${head} ${pct(s.precision, 0)} right: too few to rest a label set on (it takes at least ${MIN_LABELS}).`;
    return `${head} ${shareApart(s.precision, TARGET_PRECISION)} right, short of the ${pct(TARGET_PRECISION, 0)} a label set to train on needs.`;
  }
  if (s.threshold !== null)
    return `From confidence ${conf(s.threshold)} up: ${labelsText(s.labels)} (${per}), ${pct(s.precision, 0)} right, ${share(s.recall)} of all objects${iou}.`;
  // No threshold: the best precision on enough labels, and a precise handful said to be a handful.
  const supported = s.curve.filter((p) => p.precision !== null && p.labels >= MIN_LABELS);
  const best = supported.reduce<BenchLabelPoint | null>((a, p) => (!a || p.precision! > a.precision! ? p : a), null);
  const few = s.curve.filter((p) => p.precision !== null && p.labels > 0 && p.labels < MIN_LABELS && p.precision >= TARGET_PRECISION).sort((a, b) => a.confidence - b.confidence)[0];
  const most = best ? `: the best is ${shareApart(best.precision!, TARGET_PRECISION)} right, from ${conf(best.confidence)} up (${labelsText(best.labels)})` : '';
  const handful = few ? ` From ${conf(few.confidence)} up its ${labelsText(few.labels)} ${few.labels === 1 ? 'is' : 'are'} ${pct(few.precision!, 0)} right, too few to rest a threshold on.` : '';
  // Without a threshold nothing is kept as a label: what follows is all its detections, said as detections (the real
  // weak teacher gave 74,593 of them, 248.6 an image, 1.6% right: never "labels per image" it would keep).
  const all = `${num(s.labels)} ${s.labels === 1 ? 'detection' : 'detections'}`;
  return `No confidence keeps ${MIN_LABELS} labels or more of which ${pct(TARGET_PRECISION, 0)} are right${most}, so none is kept as a label.${handful} All its detections together: ${all} (${per}), ${fine(s.precision)} right.`;
}

/** The robustness axis read: where it breaks most and where least. */
export function robustnessReading(
  conditions: readonly { condition: string; retention: number; points: readonly { severity: number; retention: number }[] }[],
  name: (id: string) => string,
): string {
  if (!conditions.length) return '';
  const worst = conditions.reduce((a, c) => (c.retention < a.retention ? c : a));
  const best = conditions.reduce((a, c) => (c.retention > a.retention ? c : a));
  const low = worst.points.reduce((a, p) => (p.retention < a.retention ? p : a), worst.points[0] ?? { severity: 0, retention: worst.retention });
  const hard = `Hardest: ${name(worst.condition)}, ${pct(low.retention, 0)} kept at severity ${low.severity}.`;
  return conditions.length === 1 ? hard : `${hard} Easiest: ${name(best.condition)}, ${pct(best.retention, 0)} kept on average.`;
}

/** Retention as the heat table and the readouts write it: "83%". */
export const keptText = (r: number) => pct(r, 0);

// ---------- comparing entries ----------

const listOf = (xs: readonly string[]) => (xs.length <= 1 ? (xs[0] ?? '') : `${xs.slice(0, -1).join(', ')} and ${xs.at(-1)}`);

/** Whether two intervals overlap as printed (three decimals): what a reader checks by eye against the numbers shown. */
export const overlapAsPrinted = (x: readonly [number, number], y: readonly [number, number]) => {
  const r = (v: number) => Math.round(v * 1000);
  return r(x[0]) <= r(y[1]) && r(y[0]) <= r(x[1]);
};

/**
 * Whether entries' mAP intervals tell them apart from the highest, in one sentence that says what was compared
 * (docs/BENCH.md: overlapping intervals are not clearly different on this benchmark): each other entry's interval
 * against the leader's, as printed. Two entries behind the leader are not compared with each other, so nothing is said
 * of them as a pair.
 */
export function intervalsVerdict(items: readonly { name: string; map: number; ci: readonly [number, number] | null }[]): string {
  const known = items.filter((i): i is { name: string; map: number; ci: readonly [number, number] } => !!i.ci);
  if (known.length < 2) return '';
  const sorted = [...known].sort((a, b) => b.map - a.map);
  const lead = sorted[0]!;
  const rest = sorted.slice(1);
  const tied = rest.filter((x) => overlapAsPrinted(x.ci, lead.ci));
  const behind = rest.filter((x) => !tied.includes(x));
  const its = `${lead.name}’s`;
  if (known.length === 2)
    return tied.length
      ? `Not clearly different on this benchmark: their 95% intervals overlap.`
      : `${lead.name} is clearly ahead on mAP50-95: its interval is above ${rest[0]!.name}’s.`;
  if (!behind.length) return `Against ${lead.name}, the highest: every other interval overlaps ${its}, so none is clearly different from it on this benchmark.`;
  if (!tied.length) return `${lead.name} is clearly ahead on mAP50-95: its interval is above each of the others’.`;
  const verb = (xs: readonly unknown[], one: string, many: string) => (xs.length === 1 ? one : many);
  return `Against ${lead.name}, the highest: ${listOf(tied.map((x) => x.name))} ${verb(tied, 'overlaps', 'overlap')} its interval, so ${verb(tied, 'is', 'are')} not clearly different from it; ${listOf(behind.map((x) => x.name))} ${verb(behind, 'is', 'are')} clearly below it.`;
}

/**
 * What makes entries not like for like, one line each, said up front on Compare (docs/BENCH.md, Reading it): fewer
 * images, conditions left out, no confidences (compare on precision, recall and self-labeling, not AP), classes it
 * does not know.
 */
export function compareCaveats(entries: readonly BenchEntry[], b: Pick<Bench, 'images' | 'classes' | 'plan'> & Partial<Pick<Bench, 'fingerprint'>>): { id: string; text: string }[] {
  const out: { id: string; text: string }[] = [];
  const planned = b.plan.conditions.length;
  // First: an entry graded on an earlier version of the benchmark does not compare at all (another fingerprint).
  for (const e of entries)
    if (b.fingerprint && e.fingerprint && e.fingerprint !== b.fingerprint)
      out.push({ id: e.id, text: `${e.name} was graded on an earlier version of this benchmark (another fingerprint: the engine changed since): its numbers do not compare with the others’, and it is left out of “best” and the intervals’ verdict. Run it again to compare.` });
  for (const e of entries) {
    if (e.images < b.images) out.push({ id: e.id, text: `${e.name} ran ${num(e.images)} of ${num(b.images)} images: its interval is wider and its score partial.` });
    if (planned && !e.conditions) out.push({ id: e.id, text: `${e.name} ran the unchanged images only: it has no robustness to compare.` });
    else if (planned && e.conditions_run < planned) out.push({ id: e.id, text: `${e.name} ran ${e.conditions_run} of ${planned} conditions: its robustness is over those.` });
    if (!e.scored)
      out.push({ id: e.id, text: `${e.name} gives no confidences: its AP comes from one precision–recall point. Compare it on precision, recall and self-labeling rather than mAP.` });
    if (e.missing_classes.length) out.push({ id: e.id, text: `${e.name} knows ${b.classes.length - e.missing_classes.length} of ${b.classes.length} classes: its AP on the others is 0.` });
  }
  return out;
}

// ---------- loops ----------

/** The question a loop answers, in one line. */
export function loopQuestion(teacher: string, pool: string, studentArch: string | null, generations: number): string {
  const whom = studentArch ? `a better ${studentArch}` : 'a better model';
  return `Can ${teacher} label ${pool} well enough, by itself, to train ${whom}${generations > 1 ? ', generation after generation' : ''}?`;
}

/** A loop's verdict as a reading of its numbers: the last generation against the first teacher's interval. */
export function loopVerdictText(l: Pick<BenchLoop, 'verdict' | 'generations'>): string | null {
  const first = l.generations[0];
  const last = l.generations.at(-1);
  if (!l.verdict || !first?.ci95 || last?.map50_95 == null) return null;
  // As many decimals as it takes for "above" or "below" to be true as printed (0.8661 above 0.8660, never 0.866 above
  // 0.866); inside, three.
  const edge = l.verdict === 'improved' ? first.ci95[1] : l.verdict === 'regressed' ? first.ci95[0] : null;
  const d = edge === null ? 3 : digitsApart(last.map50_95, edge);
  const g = `generation ${last.generation} (${num(last.map50_95, d)})`;
  const iv = `the first teacher’s 95% interval (${num(first.ci95[0], d)}–${num(first.ci95[1], d)})`;
  if (l.verdict === 'improved') return `Improved: ${g} is above ${iv}. Its own labels made a better model.`;
  if (l.verdict === 'regressed') return `Regressed: ${g} is below ${iv}.`;
  return `Held: ${g} is inside ${iv}: no measurable change.`;
}

/** docs/BENCH.md's caveat, one wording. */
export const LOOP_CAVEAT = 'A loop is as honest as its benchmark is held out: if the teacher trained on the benchmark’s images, every generation inherits that.';

/** What starting a loop does and costs: "3 generations: 3 training runs of 30 epochs on up to 4,200 images, 4 benchmark entries". */
export function loopPlanText(o: { generations: number; epochs: number; poolImages: number | null; teacherEntry: boolean }): string {
  const g = o.generations;
  const entries = g + (o.teacherEntry ? 0 : 1);
  const on = o.poolImages !== null ? ` on up to ${imagesText(o.poolImages)}` : '';
  const runs = `${g} training ${g === 1 ? 'run' : 'runs'} of ${num(o.epochs)} ${o.epochs === 1 ? 'epoch' : 'epochs'}${on}`;
  return `${g} ${g === 1 ? 'generation' : 'generations'}: ${runs}, ${num(entries)} benchmark ${entries === 1 ? 'entry' : 'entries'}`;
}

/** Whether a model can teach on this benchmark: its classes must be the benchmark's, same names (without case), same order. */
export function teacherEligibility(m: Pick<Model, 'classes'>, b: Pick<Bench, 'classes'>): { ok: true } | { ok: false; why: string } {
  const mine = m.classes.map((c) => c.toLowerCase());
  const theirs = b.classes.map((c) => c.toLowerCase());
  if (mine.length === theirs.length && mine.every((c, i) => c === theirs[i])) return { ok: true };
  const known = theirs.filter((c) => mine.includes(c)).length;
  if (known === theirs.length && mine.length === theirs.length) return { ok: false, why: 'Same classes in another order: labels are written by class index' };
  if (!known) return { ok: false, why: `None of these classes (it knows ${classesText(m.classes.length)})` };
  if (known === theirs.length) return { ok: false, why: `Knows ${classesText(m.classes.length)}, more than these ${theirs.length}` };
  return { ok: false, why: `Knows ${known} of these ${theirs.length} classes` };
}

/** A student generation's steps, in order: its teacher labels the pool, the student trains, the benchmark grades it. */
export const LOOP_STEPS: { phase: BenchLoopPhase; name: string }[] = [
  { phase: 'labeling', name: 'Teacher labels the pool' },
  { phase: 'training', name: 'Student trains' },
  { phase: 'benchmarking', name: 'Benchmark grades the student' },
];

/** Generation 0's one step: the benchmark grades the first teacher (nothing is labeled or trained for it). */
export const TEACHER_STEP: { phase: BenchLoopPhase; name: string } = { phase: 'benchmarking', name: 'Benchmark grades the teacher' };

export type StepState = 'done' | 'now' | 'next';

/** Whether a step of generation `g` has happened, by what it left in the generation (never by the phase alone). */
function stepDone(g: BenchGeneration, phase: BenchLoopPhase): boolean {
  if (phase === 'labeling') return g.labels !== null;
  if (phase === 'training') return g.model_id !== null;
  return g.map50_95 !== null;
}

/**
 * The steps of the generation a running loop is on, as far as that generation's fields say (docs/API.md,
 * Self-improvement loops): generation 0 has one, the benchmark grading the first teacher; a student's generation three,
 * its teacher labels the pool (done once `labels` is in), the student trains (once its model is), the benchmark grades
 * the student (once its score is). A step is done only when what it makes is there, so nothing reads done that has not
 * happened; the first step not done is the one it is on, the rest come next. Null for a loop that has ended or has no
 * generation yet.
 */
export function loopSteps(l: Pick<BenchLoop, 'status' | 'phase' | 'generations'>): { generation: number; steps: { phase: BenchLoopPhase; name: string; state: StepState }[] } | null {
  if (l.status !== 'running' && l.status !== 'stopping') return null;
  const g = l.generations.at(-1);
  if (!g) return null;
  const plan = g.generation === 0 ? [TEACHER_STEP] : LOOP_STEPS;
  let now = false;
  const steps = plan.map((st) => {
    const done = stepDone(g, st.phase);
    const state: StepState = done ? 'done' : now ? 'next' : 'now';
    if (!done) now = true;
    return { ...st, state };
  });
  return { generation: g.generation, steps };
}

/** What a running loop does now, as its status says it: "Benchmarking the teacher", "Labeling for gen 1", "Training gen 1". */
function nowLabel(generation: number, phase: BenchLoopPhase): string {
  if (generation === 0) return 'Benchmarking the teacher';
  if (phase === 'labeling') return `Labeling for gen ${generation}`;
  return `${PHASE[phase]} gen ${generation}`;
}

/** Under this share of the pool kept, a generation's labels are thin: the line says so. */
export const THIN_LABELS = 0.1;

/**
 * What a teacher's labels of the pool were, in one line: "kept 243 boxes on 48 of 1,920 pool images from confidence
 * 0.31; 12 left out as benchmark images", and whether that is thin (under a tenth of the pool kept: a student learns
 * from a handful of images).
 */
export function labelsLine(l: Pick<BenchLabels, 'threshold' | 'images' | 'labeled_images' | 'boxes' | 'excluded'>): { text: string; thin: boolean } {
  const boxes = `${num(l.boxes)} ${l.boxes === 1 ? 'box' : 'boxes'}`;
  const pool = `${num(l.images)} pool ${l.images === 1 ? 'image' : 'images'}`;
  const left = l.excluded ? `; ${num(l.excluded)} left out as benchmark ${l.excluded === 1 ? 'image' : 'images'}` : '';
  return {
    text: `kept ${boxes} on ${num(l.labeled_images)} of ${pool} from confidence ${conf(l.threshold)}${left}`,
    thin: l.images > 0 && l.labeled_images / l.images < THIN_LABELS,
  };
}

/** The fewest images a teacher must keep a box on for a loop to train a student (the server's MIN_LABELED_IMAGES). */
export const MIN_LABELED_IMAGES = 10;
/** Under this share of all objects labeled, a teacher's labels are too few to learn from (so is a verdict of not ready). */
export const STARVED_RECALL = 0.1;
/**
 * What a teacher's labels of a pool would be at a confidence, read from its entry's self-labeling curve. A loop never
 * guesses a confidence: with none set and no threshold in its entry, it fails before labeling (`from: 'none'`).
 */
export interface LabelForecast {
  /** The confidence the labels are kept from; null when there is none (the loop would fail before labeling). */
  threshold: number | null;
  /** Where it comes from: the entry's own threshold t*, the person's, or none (its entry found no threshold). */
  from: 'entry' | 'chosen' | 'none';
  /** The share of the labels kept there that are right; null when it keeps none. */
  precision: number | null;
  /** The share of all objects they label. */
  recall: number;
  /** Labels per image there. */
  perImage: number;
  /** The entry's verdict, at its own threshold. */
  readiness: BenchReadiness;
  /** About how many of the pool's images would get at least one box (each image's labels taken as a Poisson count). */
  images: number | null;
  /** Too few to learn from: not ready at its own threshold, or under STARVED_RECALL of all objects. */
  starved: boolean;
  /** The loop would fail: before labeling with no confidence to keep labels from, or at labeling with under
   *  MIN_LABELED_IMAGES pool images labeled. */
  stops: boolean;
}

/** The curve's point at a confidence (its grid is 0.00, 0.01, ..., 1.00). */
export function labelPointAt(curve: readonly BenchLabelPoint[], c: number): BenchLabelPoint | null {
  if (!curve.length) return null;
  return curve.reduce((a, p) => (Math.abs(p.confidence - c) < Math.abs(a.confidence - c) ? p : a));
}

/**
 * What the teacher's labels would be in a loop, by the server's rules (docs/BENCH.md, Self-improvement loops): at
 * `chosen` when the person sets a confidence, else at its entry's threshold, read off the entry's curve; about how many
 * of `poolImages` would get a box (each image's labels taken as a Poisson count); whether that is too few to train on;
 * and whether the loop would fail: before labeling when there is no confidence at all (no `min_confidence` and no
 * threshold in its entry: a loop never guesses one), at labeling when fewer than MIN_LABELED_IMAGES pool images would
 * get a box.
 */
export function labelForecast(
  s: Pick<BenchSelfLabeling, 'threshold' | 'readiness' | 'curve'>,
  entryImages: number,
  poolImages: number | null,
  chosen: number | null,
): LabelForecast {
  const from: LabelForecast['from'] = chosen !== null ? 'chosen' : s.threshold !== null ? 'entry' : 'none';
  const threshold = chosen ?? s.threshold;
  if (threshold === null) return { threshold: null, from, precision: null, recall: 0, perImage: 0, readiness: s.readiness, images: null, starved: true, stops: true };
  const p = labelPointAt(s.curve, threshold);
  const recall = p?.recall ?? 0;
  const perImage = p && entryImages > 0 ? p.labels / entryImages : 0;
  const images = poolImages !== null ? Math.round(poolImages * (1 - Math.exp(-perImage))) : null;
  const stops = images !== null && images < MIN_LABELED_IMAGES;
  const starved = stops || recall < STARVED_RECALL || (from !== 'chosen' && s.readiness === 'not_ready');
  return { threshold, from, precision: p?.precision ?? null, recall, perImage, readiness: s.readiness, images, starved, stops };
}

/**
 * The forecast in words, for New loop: where the labels are kept from, how many are right and how many objects they
 * cover there, how many pool images would get a box; and, when that is too few, what happens and the way out.
 */
export function forecastText(f: LabelForecast, poolImages: number | null): { title: string | null; text: string; remedy: string | null } {
  if (f.threshold === null)
    return {
      title: 'Its entry found no threshold: choose a confidence',
      text: `No confidence keeps ${MIN_LABELS} labels or more of which ${pct(TARGET_PRECISION, 0)} are right, so there is no threshold to keep its labels from, and a loop never guesses one: without a confidence set here it stops before labeling.`,
      remedy: 'What the one you type keeps, and how many of those are right, shows here.',
    };
  // "62% of its labels are right and they cover 41% of all objects", or "it keeps no detection there".
  const right = (x: number) => (f.from === 'chosen' ? shareApart(x, TARGET_PRECISION) : share(x));
  const what = f.precision === null ? 'it keeps no detection there' : `${right(f.precision)} of its labels are right ${f.starved ? 'but' : 'and'} they cover ${share(f.recall)} of all objects`;
  const pool = f.images !== null && poolImages !== null ? `: about ${num(f.images)} of the pool’s ${num(poolImages)} images would get a box` : '';
  const at = f.from === 'entry' ? `At its own threshold, ${conf(f.threshold)}, ${what} (${READINESS[f.readiness].toLowerCase()})${pool}.` : `From ${conf(f.threshold)}, ${what}${pool}.`;
  // A confidence of the person's below the 90% a label set to train on needs: the cost said, not hidden.
  const cost = f.from === 'chosen' && f.precision !== null && f.precision < TARGET_PRECISION ? ` That is under the ${pct(TARGET_PRECISION, 0)} right a label set to train on needs: the student learns from its wrong labels too.` : '';
  if (!f.starved) return { title: null, text: at + cost, remedy: null };
  return {
    title: f.stops ? 'Too few labels: the loop would stop at labeling' : 'Too few labels to train a student on',
    text: `${at}${cost} ${f.stops ? `A loop stops when fewer than ${MIN_LABELED_IMAGES} pool images get a box.` : 'A student learns little from so few.'}`,
    remedy: f.from === 'chosen' ? 'A still lower confidence keeps more of its detections, fewer of them right.' : 'A lower confidence keeps more of its detections, fewer of them right.',
  };
}

/**
 * The confidence "Keep labels from a lower confidence" proposes: the entry's best-F1 confidence when it is lower than
 * the threshold (a balance of right and found), else a step of 0.2 down; never under 0.05.
 */
export function lowerConfidence(threshold: number, bestConf: number | null): number {
  const c = bestConf !== null && bestConf < threshold ? bestConf : threshold - 0.2;
  return Math.max(0.05, Math.round(c * 100) / 100);
}

/**
 * Values rounded to `decimals` so they add up to `total` as written (largest remainder): an axis's points shown in
 * rows, which a reader adds up to the score above them.
 */
export function roundToTotal(values: readonly (number | null)[], total: number, decimals = 1): (number | null)[] {
  const k = 10 ** decimals;
  const want = Math.round(total * k);
  const raw = values.map((v) => (v === null ? null : v * k));
  const floor = raw.map((v) => (v === null ? null : Math.floor(v)));
  let left = want - floor.reduce<number>((s, v) => s + (v ?? 0), 0);
  const order = raw
    .map((v, i) => ({ i, rem: v === null ? -1 : v - Math.floor(v) }))
    .filter((x) => x.rem >= 0)
    .sort((a, b) => b.rem - a.rem);
  const out = [...floor];
  for (const { i } of order) {
    if (left <= 0) break;
    out[i] = out[i]! + 1;
    left--;
  }
  return out.map((v) => (v === null ? null : v / k));
}

/** The New loop field a server error belongs to, when the error says (`POST …/loops` checks the students' settings as
 *  `POST /api/runs` does): `model_not_found` with starting weights is the weights' field, an `invalid_config` naming a
 *  field is that field; anything else (a `device_not_found`: the form has no device) is the form's error. */
export function loopErrorField(e: { code?: string; message?: string } | null | undefined, hasWeights: boolean): 'epochs' | 'imgsz' | 'weights' | 'arch' | null {
  if (!e?.code) return null;
  const m = (e.message ?? '').toLowerCase();
  if (e.code === 'model_not_found') return hasWeights && !/teacher/.test(m) ? 'weights' : null;
  if (e.code !== 'invalid_config' && e.code !== 'invalid_body') return null;
  if (/\bimgsz\b|image size/.test(m)) return 'imgsz';
  if (/\bepochs?\b/.test(m)) return 'epochs';
  if (/\bweights\b/.test(m)) return 'weights';
  if (/\barch\b|architecture/.test(m)) return 'arch';
  return null;
}
