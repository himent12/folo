// What the benchmark screens say about a benchmark, an entry, a loop and their numbers: one wording everywhere (the
// leaderboard, the entry's page, the compare page, the palette), in docs/BENCH.md's vocabulary.

import type {
  Bench,
  BenchAxis,
  BenchAxisScore,
  BenchCalibration,
  BenchCatalog,
  BenchEntry,
  BenchLoop,
  BenchLoopVerdict,
  BenchReadiness,
  BenchResult,
  BenchSelfLabeling,
  BenchSource,
  LabelAgent,
  Model,
} from '$lib/api/types';
import { AXIS_NAME, TARGET_PRECISION } from '$lib/bench/rules';
import type { OwnStatus } from '$lib/ui/StatusBadge.svelte';
import { compact, conf, metric, ms, num, pct, usd } from '$lib/format';
import { AGENT_NAME } from '$lib/label';
import { modelLabel, tokens } from '../label/words';

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

/** The short fingerprint a person compares by eye: its first 4 and last 4 hex digits ("9c41…e2a0"). */
export const shortFingerprint = (f: string) => (f.length > 10 ? `${f.slice(0, 4)}…${f.slice(-4)}` : f);

/**
 * An entry's status as a status mark and its word: "Done", "Running", "Queued", "Stopping", "Failed", "Stopped" (the run
 * status whose mark it shares: a finished entry is a completed run).
 */
export function entryStatus(e: Pick<BenchEntry, 'status'>): OwnStatus {
  switch (e.status) {
    case 'done':
      return { status: 'completed', label: 'Done' };
    case 'running':
      return { status: 'running', label: 'Running' };
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

/** A loop's status: "Running" with its phase as the word ("Labeling", "Training", "Benchmarking"), else how it ended. */
export function loopStatus(l: Pick<BenchLoop, 'status' | 'phase'>): OwnStatus {
  if (l.status === 'running') return { status: 'running', label: l.phase ? PHASE[l.phase] : 'Running' };
  if (l.status === 'stopping') return { status: 'stopping', label: 'Stopping' };
  if (l.status === 'done') return { status: 'completed', label: 'Done' };
  if (l.status === 'failed') return { status: 'failed', label: 'Failed' };
  return { status: 'stopped', label: 'Stopped' };
}

export const PHASE: Record<NonNullable<BenchLoop['phase']>, string> = { benchmarking: 'Benchmarking', labeling: 'Labeling', training: 'Training' };

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
export function coverageFacts(e: BenchEntry, b: Pick<Bench, 'images' | 'classes' | 'plan'>, conditionsRun: number | undefined = e.status === 'done' ? e.conditions_run : undefined): string[] {
  const out: string[] = [];
  if (e.images < b.images) out.push(`${num(e.images)} of ${num(b.images)} images`);
  const planned = b.plan.conditions.length;
  if (planned && !e.conditions) out.push('unchanged only');
  else if (planned && conditionsRun !== undefined && conditionsRun < planned) out.push(`${conditionsRun} of ${planned} conditions`);
  if (!e.scored) out.push('no confidences');
  if (e.missing_classes.length) out.push(`knows ${b.classes.length - e.missing_classes.length} of ${b.classes.length} classes`);
  if (e.score?.partial && !out.some((f) => !f.startsWith('knows'))) out.push('some conditions left out');
  return out;
}

/** Why an entry is partial, in a sentence (its score's tooltip and accessible description). */
export function partialWhy(e: BenchEntry, b: Pick<Bench, 'images' | 'classes' | 'plan'>): string {
  const f = coverageFacts(e, b).filter((x) => !x.startsWith('knows'));
  return `Partial: ${f.join(', ')}. It compares only loosely with complete entries.`;
}

/**
 * Where a running entry is, in words: "Unchanged images" or "Haze, severity 3" (the catalog's name), and how far, as a
 * whole percent rounded down ("54%"). A queued entry: "Waiting for the GPU" (model entries run one at a time).
 */
export function progressFacts(e: BenchEntry, catalog?: BenchCatalog): { where: string; pct: string | null } {
  if (e.status === 'queued') return { where: 'Waiting for the GPU', pct: null };
  if (e.status === 'stopping') return { where: 'Stopping: ending what is in flight', pct: null };
  const p = e.progress;
  const where = p.condition ? `${catalog?.conditions.find((c) => c.id === p.condition)?.name ?? p.condition}, severity ${p.severity}` : 'Unchanged images';
  return { where, pct: p.total ? `${Math.floor((100 * p.done) / p.total)}%` : null };
}

/** A running entry's share done, 0..1. */
export const progressShare = (e: Pick<BenchEntry, 'progress'>) => (e.progress.total ? Math.min(1, e.progress.done / e.progress.total) : 0);

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
export const FROZEN = 'This benchmark has entries; its plan is fixed so they stay comparable.';


// ---------- reading a result: docs/BENCH.md's "The score", "Definitions" and "Reading it", said for one entry ----------

/** Half a 95% interval's width, the "± 0.018" after an mAP. */
export const halfWidth = (ci: readonly [number, number]) => Math.max(0, (ci[1] - ci[0]) / 2);

/** "mAP50-95 0.641 ± 0.018". */
export const mapWithInterval = (map: number, ci: readonly [number, number] | null) => `mAP50-95 ${metric(map)}${ci ? ` ± ${num(halfWidth(ci), 3)}` : ''}`;

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
      if (s.recall > 0) return `labels ${pct(s.recall, 0)} of objects at ${pct(scored ? TARGET_PRECISION : s.precision, 0)} precision`;
      return scored ? `never ${pct(TARGET_PRECISION, 0)} of labels right` : `${pct(s.precision, 0)} of labels right, under ${pct(TARGET_PRECISION, 0)}`;
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

/** What a 95% interval says, in one line under the number. */
export function intervalReading(ci: readonly [number, number], images: number): string {
  return `With 95% confidence between ${metric(ci[0])} and ${metric(ci[1])} on these ${imagesText(images)}: an entry whose interval overlaps it is not clearly different here.`;
}

/** How its confidence reads against how often it is right. */
export type CalibrationTone = 'close' | 'over' | 'under';

/**
 * The reliability diagram read in one sentence: "Overconfident: when it says 0.94, it is right 78% of the time." The
 * bin quoted is the one furthest off the diagonal among those holding at least 5% of the detections (a bin of three
 * detections proves nothing); the tone is the detections' mean gap, unless the ECE is under 0.03 (close).
 */
export function calibrationReading(c: Pick<BenchCalibration, 'ece' | 'bins'>): { tone: CalibrationTone; sentence: string } {
  const filled = c.bins.filter((b) => b.count > 0 && b.confidence !== null && b.precision !== null);
  const n = filled.reduce((s, b) => s + b.count, 0);
  if (!n) return { tone: 'close', sentence: 'No detection reached confidence 0.1, so there is nothing to read.' };
  const big = filled.filter((b) => b.count >= 0.05 * n);
  const pool = big.length ? big : filled;
  if (c.ece < 0.03) {
    const top = pool.at(-1)!;
    return {
      tone: 'close',
      sentence: `Close to the diagonal: when it says ${num(top.confidence!, 2)}, it is right ${pct(top.precision!, 0)} of the time. Its confidence can be taken at its word.`,
    };
  }
  const gap = filled.reduce((s, b) => s + b.count * (b.confidence! - b.precision!), 0) / n;
  const pick = pool.reduce((a, b) => (Math.abs(b.confidence! - b.precision!) > Math.abs(a.confidence! - a.precision!) ? b : a));
  const says = `when it says ${num(pick.confidence!, 2)}, it is right ${pct(pick.precision!, 0)} of the time`;
  return gap >= 0
    ? { tone: 'over', sentence: `Overconfident: ${says}. Trust its high scores less than they claim.` }
    : { tone: 'under', sentence: `Underconfident: ${says}. Its scores undersell it.` };
}

/** What a readiness verdict means (docs/BENCH.md, Self-labeling). */
export const READINESS_MEANS: Record<BenchReadiness, string> = {
  ready: 'It labels at least half of all objects by itself with 9 in 10 labels right: enough to train its successor on.',
  partial: 'It labels a quarter to half of all objects that way: its labels help, and people label the rest.',
  not_ready: 'It labels less than a quarter of all objects that way, or never gets 9 in 10 right: its own labels would not train a model.',
};

/** The self-labeling readout in words, for every case: a threshold found, never reaching 90%, no confidences. */
export function selfLabelReading(
  s: Pick<BenchSelfLabeling, 'threshold' | 'precision' | 'recall' | 'mean_iou' | 'labels_per_image' | 'curve'>,
  scored: boolean,
): string {
  const per = `${num(s.labels_per_image, 1)} ${s.labels_per_image === 1 ? 'label' : 'labels'} per image`;
  const iou = s.mean_iou !== null ? `, mean IoU ${num(s.mean_iou, 2)}` : '';
  if (!scored) {
    const head = 'It gives no confidences, so it has one operating point';
    return s.recall > 0
      ? `${head}: ${pct(s.precision, 0)} of its labels are right and they cover ${pct(s.recall, 0)} of all objects (${per}${iou}).`
      : `${head}: ${pct(s.precision, 0)} of its labels are right, short of the ${pct(TARGET_PRECISION, 0)} a label set to train on needs.`;
  }
  if (s.threshold !== null)
    return `From confidence ${conf(s.threshold)} up, ${pct(s.precision, 0)} of its labels are right and they cover ${pct(s.recall, 0)} of all objects: ${per}${iou}.`;
  const best = s.curve.reduce<{ c: number; p: number } | null>(
    (a, p) => (p.precision !== null && p.labels > 0 && (!a || p.precision > a.p) ? { c: p.confidence, p: p.precision } : a),
    null,
  );
  const most = best ? `: the most is ${pct(best.p, 0)}, from ${conf(best.c)} up` : '';
  return `At no confidence are ${pct(TARGET_PRECISION, 0)} of its labels right${most}. All its detections together are ${pct(s.precision, 0)} right, ${per}.`;
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

/**
 * Whether entries' mAP intervals tell them apart, in one sentence (docs/BENCH.md: overlapping intervals are not
 * clearly different on this benchmark), read against the highest.
 */
export function intervalsVerdict(items: readonly { name: string; map: number; ci: readonly [number, number] | null }[]): string {
  const known = items.filter((i): i is { name: string; map: number; ci: readonly [number, number] } => !!i.ci);
  if (known.length < 2) return '';
  const sorted = [...known].sort((a, b) => b.map - a.map);
  const lead = sorted[0]!;
  const rest = sorted.slice(1);
  const tied = rest.filter((x) => x.ci[0] <= lead.ci[1] && lead.ci[0] <= x.ci[1]);
  const behind = rest.filter((x) => !tied.includes(x));
  if (!behind.length)
    return known.length === 2 ? 'Not clearly different on this benchmark: their 95% intervals overlap.' : `Not clearly different on this benchmark: every interval overlaps ${lead.name}’s.`;
  if (!tied.length) return `${lead.name} is clearly ahead on mAP50-95: its interval is above ${known.length === 2 ? 'the other’s' : 'the others’'}.`;
  const verb = behind.length === 1 ? 'is' : 'are';
  return `${listOf([lead, ...tied].map((x) => x.name))} are not clearly different (their intervals overlap); ${listOf(behind.map((x) => x.name))} ${verb} clearly behind.`;
}

/**
 * What makes entries not like for like, one line each, said up front on Compare (docs/BENCH.md, Reading it): fewer
 * images, conditions left out, no confidences (compare on precision, recall and self-labeling, not AP), classes it
 * does not know.
 */
export function compareCaveats(entries: readonly BenchEntry[], b: Pick<Bench, 'images' | 'classes' | 'plan'>): { id: string; text: string }[] {
  const out: { id: string; text: string }[] = [];
  const planned = b.plan.conditions.length;
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
  const g = `generation ${last.generation} (${metric(last.map50_95)})`;
  const iv = `the first teacher’s 95% interval (${intervalText(first.ci95)})`;
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
export const LOOP_STEPS: { phase: NonNullable<BenchLoop['phase']>; name: string }[] = [
  { phase: 'labeling', name: 'Teacher labels the pool' },
  { phase: 'training', name: 'Student trains' },
  { phase: 'benchmarking', name: 'Benchmark grades it' },
];

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
