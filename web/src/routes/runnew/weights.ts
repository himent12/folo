/** The weights a run can start from, in the Start from field's words: which are published, and the field's hint. */
import type { Resource } from '$lib/api/resource.svelte';
import type { Dataset, Model, Run } from '$lib/api/types';
import { publishedSummary } from '$lib/published';
import { scoreContext, scoreMeta, type ScoreContext } from '$lib/scoreSummary';

export const isPublished = (m: Model) => m.source === 'published';

// Each starting point's score with what it is on (lib/scoreSummary): grouped by dataset and split, best first.
export function scoreContextOf(runs: Resource<Run[]>, datasets: Resource<Dataset[]>) {
  const runById = new Map<string, Run>((runs.data ?? []).map((r) => [r.id, r]));
  const dsById = new Map<string, Dataset>((datasets.data ?? []).map((d) => [d.id, d]));
  return (m: Model) => scoreContext(m, runById, dsById);
}

/** Start from's hint: the picked weights' score with all of its context, and what published weights are. */
export function startHintOf(picked: Model | undefined, scoreCtx: (m: Model) => ScoreContext): string {
  if (!picked) return 'Fine-tuning from a trained model converges much faster.';
  // Published weights: what they are for, in plain words; their provenance and score are in the ⓘ beside the label
  // (pickedAbout). Trained ones are their score with its context.
  if (isPublished(picked)) return publishedSummary(picked);
  return scoreMeta(picked.metrics?.map50_95, scoreCtx(picked)) || 'Fine-tuning from a trained model converges much faster.';
}
