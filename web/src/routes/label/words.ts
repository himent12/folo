// StatusBadge's OwnStatus is restated here (structurally the same), so this module needs no component import.
// What the labeling screens say about a job, a run and what it cost, one wording everywhere.

import type { LabelAgent, LabelImage, LabelJob, LabelRun, LabelUsage, RunStatus } from '$lib/api/types';
import { num, usd } from '$lib/format';
import { AGENT_NAME, CLASSES_MAX } from '$lib/label';

/** StatusBadge's `OwnStatus` (ui/StatusBadge.tsx). */
interface OwnStatus {
  status: RunStatus;
  label: string;
}

/**
 * A job's status as a status mark (the run status whose mark it shares) and its word: "Labeling" while an agent
 * works, "Stopping", "Last run failed" when its last run ended early and images are still to label (once they are all
 * labeled, by hand or by a later run, the failure is history), else "Idle".
 */
export function jobStatus(job: LabelJob): OwnStatus {
  if (job.status === 'running') return { status: 'running', label: 'Labeling' };
  if (job.status === 'stopping') return { status: 'stopping', label: 'Stopping' };
  if (job.run?.outcome === 'failed' && toLabel(job) > 0) return { status: 'failed', label: 'Last run failed' };
  return { status: 'queued', label: 'Idle' };
}

/** What an addition of class names could not take: "Only 80 classes fit: c81 and c82 were not added." */
export function leftOutText(names: readonly string[]): string {
  const shown = names.length > 4 ? [...names.slice(0, 3), `${num(names.length - 3)} more`] : names;
  const list = shown.length > 1 ? `${shown.slice(0, -1).join(', ')} and ${shown.at(-1)}` : (shown[0] ?? '');
  return `Only ${num(CLASSES_MAX)} classes fit: ${list} ${names.length === 1 ? 'was' : 'were'} not added.`;
}

/** The images an unscoped run labels: those without labels (pending and failed). */
export const toLabel = (job: LabelJob) => job.counts.pending + job.counts.failed;

/** "1 image", "24 images". */
export const images = (n: number) => `${num(n)} ${n === 1 ? 'image' : 'images'}`;

/** "Sonnet" (the agent's own label for its model), the model id it does not list, or "Default model" (the CLI's own). */
export function modelLabel(model: string | null, agent: LabelAgent | undefined): string {
  if (!model) return 'Default model';
  return agent?.models.find((m) => m.id === model)?.label ?? model;
}

/** The run's agent and model as two facts: "Claude Code", "Sonnet". */
export function runAgent(run: LabelRun, agents: readonly LabelAgent[] | undefined): [string, string] {
  return [AGENT_NAME[run.agent], modelLabel(run.model, agents?.find((a) => a.id === run.agent))];
}

/** How a finished run ended, in one word. */
export const OUTCOME: Record<NonNullable<LabelRun['outcome']>, string> = { completed: 'Completed', stopped: 'Stopped', failed: 'Failed' };

/** A token count: "940", "3.2k", "18.4k", "126k", "1.4M". */
export function tokens(n: number): string {
  if (n < 1000) return num(n);
  if (n < 1e6) return `${num(n / 1000, n < 1e5 ? 1 : 0)}k`;
  return `${num(n / 1e6, 1)}M`;
}

/**
 * What the agent reported a run (or a job) used, one fact each: "4 calls", "18.4k tokens in (3.2k cached)", "5.1k out",
 * "$0.02" (when it reported a cost, in the one dollar format, `$lib/format` `usd`). Nothing for a run that made no call.
 */
export function usageFacts(u: LabelUsage): string[] {
  if (u.calls === 0) return [];
  return [
    `${num(u.calls)} ${u.calls === 1 ? 'call' : 'calls'}`,
    `${tokens(u.input_tokens)} tokens in${u.cached_input_tokens ? ` (${tokens(u.cached_input_tokens)} cached)` : ''}`,
    `${tokens(u.output_tokens)} out`,
    ...(u.cost_usd !== null ? [usd(u.cost_usd)] : []),
  ];
}

/**
 * What the editor says once the run "Ask the agent again" started has ended: the image's new boxes, why the agent
 * could not label it (the run's error and hint, or the image's own), or that the run was stopped. Null while the
 * image is still with the agent (its last event may come after the run's end).
 */
export function askOutcome(run: LabelRun, img: LabelImage | undefined): { tone: 'success' | 'info' | 'error'; title: string; detail?: string } | null {
  const failed = `${AGENT_NAME[run.agent]} could not label it again`;
  if (run.outcome === 'failed') return { tone: 'error', title: failed, detail: [run.error, run.hint].filter(Boolean).join(' ') || undefined };
  if (run.outcome === 'stopped') return { tone: 'info', title: 'The run was stopped', detail: 'The image keeps the boxes it had.' };
  if (!img || img.status === 'labeling') return null;
  if (img.status === 'failed') return { tone: 'error', title: failed, detail: img.error ?? undefined };
  return { tone: 'success', title: 'New boxes', detail: imageState(img) };
}

/** An image's state in words, as the editor says it: a failed one with the reason the server gives. */
export function imageState(img: LabelImage): string {
  if (img.status === 'labeling') return 'The agent is labeling it';
  if (img.status === 'failed') return img.error ? `The agent could not label it: ${img.error}` : 'The agent could not label it';
  if (img.status === 'pending') return 'Not labeled yet';
  const n = img.boxes.length;
  const what = n === 0 ? 'No objects' : `${num(n)} ${n === 1 ? 'box' : 'boxes'}`;
  return img.source === 'manual' ? `${what}, edited by hand` : `${what}, drawn by the agent`;
}
