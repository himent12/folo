// What changed between two polls of the run list, said out loud: a run that was training and has now completed, failed
// or run out of GPU memory. A run the user stopped is not news (they did it), and a run first seen already finished
// (the first poll, a page reload) is history, not a change.

import type { Run } from '$lib/api/types';
import { metric } from '$lib/format';
import { isOom } from '$lib/oom';

export type RunEndKind = 'completed' | 'failed' | 'oom';
export interface RunEnd {
  run: Run;
  kind: RunEndKind;
}

const ACTIVE = new Set(['running', 'stopping']);

/** Runs that were training in `prev` and have ended in `next` (none on the first poll, when `prev` is undefined). */
export function runEnds(prev: readonly Run[] | undefined, next: readonly Run[]): RunEnd[] {
  if (!prev) return [];
  const was = new Map(prev.map((r) => [r.id, r.status]));
  const out: RunEnd[] = [];
  for (const run of next) {
    const before = was.get(run.id);
    if (!before || !ACTIVE.has(before)) continue;
    if (run.status === 'completed') out.push({ run, kind: 'completed' });
    else if (run.status === 'failed') out.push({ run, kind: isOom(run.error) ? 'oom' : 'failed' });
  }
  return out;
}

/** The toast for a run that ended: a title naming the run and the outcome, and one line of detail. */
export function runEndMessage(e: RunEnd): { tone: 'success' | 'error'; title: string; detail: string } {
  const { run } = e;
  if (e.kind === 'completed') {
    const best = run.best ? `Best mAP50-95 ${metric(run.best.map50_95)} at epoch ${run.best.epoch}.` : `${run.epochs} epochs done.`;
    return { tone: 'success', title: `${run.name} finished training`, detail: best };
  }
  if (e.kind === 'oom') return { tone: 'error', title: `${run.name} ran out of GPU memory`, detail: `At epoch ${run.epoch}. Its page has smaller settings to retry with.` };
  return { tone: 'error', title: `${run.name} failed`, detail: `At epoch ${run.epoch}. Its page has the error and the log.` };
}
