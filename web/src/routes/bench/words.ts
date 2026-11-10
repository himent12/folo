// What the benchmark screens say about a benchmark, an entry, a loop and their numbers: one wording everywhere (the
// leaderboard, the entry's page, the compare page, the palette), in docs/BENCH.md's vocabulary.

import type { Bench, BenchCatalog, BenchEntry, BenchLoop, BenchLoopVerdict, BenchReadiness, BenchSource, LabelAgent } from '$lib/api/types';
import type { OwnStatus } from '$lib/ui/StatusBadge.svelte';
import { compact, ms, num, usd } from '$lib/format';
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
export function coverageFacts(e: BenchEntry, b: Pick<Bench, 'images' | 'classes' | 'plan'>, conditionsRun?: number): string[] {
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
