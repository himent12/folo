/** The weights a run can start from, in the Start from field's words: which are published, and the field's hint. */
import type { Resource } from '$lib/api/resource';
import type { Arch, Dataset, Model, Run } from '$lib/api/types';
import { publishedSummary } from '$lib/published';
import { modelChoices, scoreContext, scoreMeta, type ModelChoice, type ScoreContext } from '$lib/scoreSummary';
import { arches } from './estimate';

export const isPublished = (m: Model) => m.source === 'published';

// Each starting point's score with what it is on (lib/scoreSummary): grouped by dataset and split, best first.
export function scoreContextOf(runs: Resource<Run[]>, datasets: Resource<Dataset[]>) {
  const runById = new Map<string, Run>((runs.data ?? []).map((r) => [r.id, r]));
  const dsById = new Map<string, Dataset>((datasets.data ?? []).map((d) => [d.id, d]));
  return (m: Model) => scoreContext(m, runById, dsById);
}

/**
 * Trained models of the other sizes, as Start from lists them after this size's: one group per size, best first, each
 * a way to switch the size (a fine-tune trains the weights' own).
 */
export function otherSizeChoices(models: readonly Model[], arch: Arch, scoreCtx: (m: Model) => ScoreContext): ModelChoice<Model>[] {
  return arches
    .filter((a) => a.arch !== arch)
    .flatMap((a) => modelChoices(models.filter((m) => m.arch === a.arch), scoreCtx).map((c) => ({ ...c, group: `${a.arch}: switches the size` })));
}

/**
 * Start from's hint: the picked weights' score with all of its context, and what published weights are. With none
 * picked and no trained model of this size, where the trained ones are (`others`, of the other sizes). When the
 * picked weights' class list is not the dataset's (`dataset`), what the run keeps of them: the trainer starts the class
 * outputs fresh and loads the rest (docs/API.md, `weights`), which the run's log says only once it has started.
 */
export function startHintOf(
  picked: Model | undefined,
  scoreCtx: (m: Model) => ScoreContext,
  arch?: Arch,
  mine = 0,
  others: readonly Model[] = [],
  dataset?: Pick<Dataset, 'name' | 'classes'>,
): string {
  if (!picked && !mine && others.length) {
    const sizes = [...new Set(others.map((m) => m.arch))];
    const what = sizes.length === 1 ? `${others.length} trained ${sizes[0]} ${others.length === 1 ? 'model is' : 'models are'}` : `${others.length} trained models of other sizes are`;
    return `No trained ${arch} models yet. ${what} in the list: picking one switches the size.`;
  }
  if (!picked) return 'Fine-tuning from a trained model converges much faster.';
  // Published weights: what they are for, in plain words; their provenance and score are in the ⓘ beside the label
  // (pickedAbout). Trained ones are their score with its context.
  const what = isPublished(picked) ? publishedSummary(picked) : scoreMeta(picked.metrics?.map50_95, scoreCtx(picked)) || 'Fine-tuning from a trained model converges much faster.';
  // A model whose classes are not recorded (an older file) says nothing it cannot know.
  const same = !dataset || !picked.classes.length || (picked.classes.length === dataset.classes.length && picked.classes.every((c, i) => c === dataset.classes[i]));
  return same ? what : `${what.replace(/\.$/, '')}. Its classes are not ${dataset.name}'s: the class outputs start fresh, the rest is kept.`;
}
