// What a model's score is on, wherever a summary shows one (Overview's Best models, the Models list, the Playground's
// model picker, New run's Start from, the palette): a score never travels without its evaluation context.
//
// `Model.metrics` of a trained model is its run's best epoch, scored on the run's dataset: its validation split, or its
// training split when the dataset has no validation images (docs/API.md, Split fallback; `runEvalSource`). Imported
// files and published weights carry no dataset of this server's, so their split is unknown (only the model's own
// `/eval` can say, and a summary does not fetch it); nothing here infers provenance it was not given. Scores rank only
// within one comparable group, the same dataset and the same split. Validation is the unmarked case: any other split,
// or an unknown one, is tagged beside the score ("Training split", "Split unknown").

import type { Dataset, Model, Run, Split } from '$lib/api/types';
import { metric } from '$lib/format';
import { splitWord } from '$lib/evaluated';

export interface ScoreContext {
  /** `run`: trained here, from its run's best epoch; `published` or `imported`: no dataset of this server's. */
  kind: 'run' | 'published' | 'imported';
  datasetId: string | null;
  datasetName: string | null;
  /** The split the score is on; null when it is not known (never assumed to be `val`). */
  split: Split | null;
}

/**
 * A model's score context, from its run and that run's dataset. A trained model whose run or dataset this client cannot
 * see (not loaded, or deleted) has an unknown split: whether its dataset had validation images is not known.
 */
export function scoreContext(
  m: Pick<Model, 'run_id' | 'source'>,
  runs: ReadonlyMap<string, Pick<Run, 'config'>>,
  datasets: ReadonlyMap<string, Pick<Dataset, 'id' | 'name' | 'splits'>>,
): ScoreContext {
  if (m.source === 'published') return { kind: 'published', datasetId: null, datasetName: null, split: null };
  if (!m.run_id) return { kind: 'imported', datasetId: null, datasetName: null, split: null };
  const dsId = runs.get(m.run_id)?.config.dataset_id ?? null;
  const ds = dsId ? datasets.get(dsId) : undefined;
  if (!ds) return { kind: 'run', datasetId: dsId, datasetName: dsId, split: null };
  return { kind: 'run', datasetId: ds.id, datasetName: ds.name, split: ds.splits.val === 0 ? 'train' : 'val' };
}

/** The comparable group's key (same dataset, same split); null when the score cannot be ranked against any other. */
export function rankKey(c: ScoreContext): string | null {
  return c.kind === 'run' && c.datasetId && c.split ? `${c.datasetId}:${c.split}` : null;
}

/** A split as a label: "Validation split", "Training split", "Test split". */
export function splitLabel(s: Split): string {
  const w = splitWord(s);
  return `${w.charAt(0).toUpperCase()}${w.slice(1)} split`;
}

/** The tag beside a score that is not on a validation split: "Training split", "Split unknown"; null on validation. */
export function scoreTag(c: ScoreContext): string | null {
  if (c.split === 'val') return null;
  return c.split ? splitLabel(c.split) : 'Split unknown';
}

/**
 * A comparable group's heading: "Validation split of fasteners", "Training split of fasteners-mini". Groups that are
 * never ranked say what they are and why: "Imported · split unknown", "Published weights · split unknown".
 */
export function groupLabel(c: ScoreContext): string {
  if (c.kind === 'published') return 'Published weights · split unknown';
  if (c.kind === 'imported') return 'Imported · split unknown';
  if (!c.split) return `${c.datasetName ?? 'Unknown dataset'} · split unknown`;
  return `${splitLabel(c.split)} of ${c.datasetName}`;
}

/** A filter's short name for a group: "fasteners", "fasteners-mini · training split", "Imported", "Published weights". */
export function filterLabel(c: ScoreContext): string {
  if (c.kind === 'published') return 'Published weights';
  if (c.kind === 'imported') return 'Imported';
  if (!c.split) return `${c.datasetName ?? 'Unknown dataset'} · split unknown`;
  return c.split === 'val' ? (c.datasetName ?? '') : `${c.datasetName} · ${splitWord(c.split)} split`;
}

/** A key for every group, ranked or not: the comparable group's, else one per kind (and per dataset, split unknown). */
export function groupKey(c: ScoreContext): string {
  return rankKey(c) ?? (c.kind === 'run' ? `~unknown:${c.datasetId ?? ''}` : `~${c.kind}`);
}

/**
 * A score with its context in one line of text, for places that hold no markup (a native option, a palette hint):
 * "mAP 0.604 on fasteners", "mAP 0.701 on fasteners-mini · training split", "mAP 0.448 · split unknown".
 */
export function scoreText(score: number | null | undefined, c: ScoreContext): string {
  const s = `mAP ${metric(score)}`;
  const tag = scoreTag(c)?.toLowerCase();
  if (c.kind === 'run' && c.datasetName && c.split) return `${s} on ${c.datasetName}${tag ? ` · ${tag}` : ''}`;
  return `${s} · ${tag ?? 'split unknown'}`;
}

export interface ScoreGroup<T> {
  key: string;
  ctx: ScoreContext;
  /** Best first (on a tie the earlier `created`), then those without a score. */
  items: T[];
}

/**
 * Rank items only against comparable ones: one group per dataset and split, best first; groups ordered by dataset name,
 * validation before other splits. Whatever has no comparable group (imported, published, an unknown split) is returned
 * apart, unranked, in the order given.
 */
export function rankByContext<T>(
  items: readonly T[],
  ctxOf: (t: T) => ScoreContext,
  scoreOf: (t: T) => number | null | undefined,
  createdOf: (t: T) => string = () => '',
): { groups: ScoreGroup<T>[]; unranked: T[] } {
  const by = new Map<string, ScoreGroup<T>>();
  const unranked: T[] = [];
  for (const t of items) {
    const ctx = ctxOf(t);
    const key = rankKey(ctx);
    if (!key) {
      unranked.push(t);
      continue;
    }
    if (!by.has(key)) by.set(key, { key, ctx, items: [] });
    by.get(key)!.items.push(t);
  }
  const groups = [...by.values()];
  for (const g of groups) g.items.sort((a, b) => (scoreOf(b) ?? -1) - (scoreOf(a) ?? -1) || createdOf(a).localeCompare(createdOf(b)));
  groups.sort((a, b) => (a.ctx.datasetName ?? '').localeCompare(b.ctx.datasetName ?? '') || splitOrder(a.ctx.split) - splitOrder(b.ctx.split));
  return { groups, unranked };
}

const splitOrder = (s: Split | null) => (s === 'val' ? 0 : s === 'test' ? 1 : 2);

export interface ModelChoice<M> {
  model: M;
  ctx: ScoreContext;
  /** The `<optgroup>` it is listed under: what its score is on ("Validation split of fasteners"). */
  group: string;
  /** "fasteners-n-best · mAP 0.604 on fasteners": the score never without its context, even in a closed select. */
  label: string;
}

/**
 * Models as a picker lists them (Playground, New run's Start from): one group per comparable dataset and split, best
 * first within it, then the groups that are never ranked (published, imported, split unknown), each model's label
 * carrying its score's context.
 */
export function modelChoices<M extends Pick<Model, 'id' | 'name' | 'created' | 'metrics'>>(
  models: readonly M[],
  ctxOf: (m: M) => ScoreContext,
): ModelChoice<M>[] {
  const { groups, unranked } = rankByContext(models, ctxOf, (m) => m.metrics?.map50_95, (m) => m.created);
  const choice = (model: M, ctx: ScoreContext): ModelChoice<M> => ({
    model,
    ctx,
    group: groupLabel(ctx),
    label: model.metrics ? `${model.name} · ${scoreText(model.metrics.map50_95, ctx)}` : model.name,
  });
  const order = (c: ScoreContext) => (c.kind === 'published' ? 0 : c.kind === 'imported' ? 1 : 2);
  const rest = unranked
    .map((m) => ({ m, c: ctxOf(m) }))
    .sort((a, b) => order(a.c) - order(b.c) || groupKey(a.c).localeCompare(groupKey(b.c)) || a.m.name.localeCompare(b.m.name));
  return [...groups.flatMap((g) => g.items.map((m) => choice(m, g.ctx))), ...rest.map(({ m, c }) => choice(m, c))];
}
