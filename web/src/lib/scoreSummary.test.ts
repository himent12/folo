import { describe, expect, it } from 'vitest';
import type { Model } from '$lib/api/types';
import {
  filterLabel,
  groupKey,
  groupLabel,
  modelChoices,
  rankByContext,
  rankKey,
  scoreContext,
  scoreTag,
  scoreText,
  type ScoreContext,
} from './scoreSummary';

const ds = (id: string, val: number) => ({ id, name: id.replace('ds_', '').replace('_', '-'), splits: { train: 100, val, test: 0 } });
const datasets = new Map([ds('ds_fasteners', 360), ds('ds_fasteners_mini', 0), ds('ds_shapes', 200)].map((d) => [d.id, d]));
const runs = new Map(
  [
    ['r_fast', 'ds_fasteners'],
    ['r_fast2', 'ds_fasteners'],
    ['r_mini', 'ds_fasteners_mini'],
    ['r_shapes', 'ds_shapes'],
    ['r_gone', 'ds_deleted'],
  ].map(([id, dataset_id]) => [id!, { config: { dataset_id } } as never]),
);

function model(id: string, run_id: string | null, map: number | null, extra: Partial<Model> = {}): Model {
  return {
    id,
    name: id,
    arch: 'folo-n',
    imgsz: 640,
    classes: [],
    params: 0,
    flops_g: 0,
    bytes: 0,
    created: `2026-11-0${id.length % 9}T00:00:00Z`,
    run_id,
    metrics: map === null ? null : { map50: map, map50_95: map, precision: 0, recall: 0 },
    source: run_id ? 'run' : 'file',
    ...extra,
  };
}

const ctx = (m: Model) => scoreContext(m, runs, datasets);

describe('scoreContext', () => {
  it('puts a trained model on its run\'s dataset: the validation split, or training when the dataset has none', () => {
    expect(ctx(model('a', 'r_fast', 0.6))).toEqual({ kind: 'run', datasetId: 'ds_fasteners', datasetName: 'fasteners', split: 'val' });
    expect(ctx(model('b', 'r_mini', 0.7))).toEqual({ kind: 'run', datasetId: 'ds_fasteners_mini', datasetName: 'fasteners-mini', split: 'train' });
  });
  it('never infers a split it was not given: imported, published, and a run whose dataset is not known', () => {
    expect(ctx(model('c', null, 0.7)).split).toBeNull();
    expect(ctx(model('c', null, 0.7)).kind).toBe('imported');
    expect(ctx(model('d', null, 0.3, { source: 'published' })).kind).toBe('published');
    expect(ctx(model('e', 'r_gone', 0.5))).toEqual({ kind: 'run', datasetId: 'ds_deleted', datasetName: 'ds_deleted', split: null });
    expect(ctx(model('f', 'r_unknown', 0.5))).toEqual({ kind: 'run', datasetId: null, datasetName: null, split: null });
  });
});

describe('the tags and labels', () => {
  const val = ctx(model('a', 'r_fast', 0.6));
  const train = ctx(model('b', 'r_mini', 0.7));
  const imported = ctx(model('c', null, 0.7));
  const published = ctx(model('d', null, 0.3, { source: 'published' }));
  it('tags every score that is not on a validation split, and only those', () => {
    expect(scoreTag(val)).toBeNull();
    expect(scoreTag(train)).toBe('Training split');
    expect(scoreTag(imported)).toBe('Split unknown');
    expect(scoreTag(published)).toBe('Split unknown');
  });
  it('heads each group with what its scores are on', () => {
    expect(groupLabel(val)).toBe('Validation split of fasteners');
    expect(groupLabel(train)).toBe('Training split of fasteners-mini');
    expect(groupLabel(imported)).toBe('Imported · split unknown');
    expect(groupLabel(published)).toBe('Published weights · split unknown');
    expect(filterLabel(val)).toBe('fasteners');
    expect(filterLabel(train)).toBe('fasteners-mini · training split');
  });
  it('carries the context in one line of text', () => {
    expect(scoreText(0.604, val)).toBe('mAP 0.604 on fasteners');
    expect(scoreText(0.701, train)).toBe('mAP 0.701 on fasteners-mini · training split');
    expect(scoreText(0.448, imported)).toBe('mAP 0.448 · split unknown');
  });
  it('gives only a known dataset and split a rank key', () => {
    expect(rankKey(val)).toBe('ds_fasteners:val');
    expect(rankKey(train)).toBe('ds_fasteners_mini:train');
    expect([imported, published, ctx(model('e', 'r_gone', 0.5))].map(rankKey)).toEqual([null, null, null]);
    expect(groupKey(imported)).toBe('~imported');
    expect(groupKey(published)).toBe('~published');
  });
});

describe('rankByContext', () => {
  const list = [
    model('mini-n', null, 0.701),
    model('fast-a', 'r_fast', 0.6),
    model('mini-run', 'r_mini', 0.74),
    model('shapes-s', 'r_shapes', 0.854),
    model('fast-b', 'r_fast2', 0.604),
    model('coco', null, 0.314, { source: 'published' }),
  ];
  const r = rankByContext(list, ctx, (m) => m.metrics?.map50_95, (m) => m.created);
  it('ranks only within one dataset and split, best first', () => {
    expect(r.groups.map((g) => [g.key, g.items.map((m) => m.id)])).toEqual([
      ['ds_fasteners:val', ['fast-b', 'fast-a']],
      ['ds_fasteners_mini:train', ['mini-run']],
      ['ds_shapes:val', ['shapes-s']],
    ]);
  });
  it('keeps imported and published weights apart, unranked', () => {
    expect(r.unranked.map((m) => m.id)).toEqual(['mini-n', 'coco']);
  });
  it('breaks a tie with the earlier checkpoint', () => {
    const tie = rankByContext([model('fast-zz', 'r_fast', 0.6), model('fast-a', 'r_fast', 0.6)], ctx, (m) => m.metrics?.map50_95, (m) => m.created);
    expect(tie.groups[0]!.items.map((m) => m.id)).toEqual(['fast-a', 'fast-zz']);
  });
});

describe('modelChoices', () => {
  it('lists the ranked groups first, then the unranked ones, each label carrying its context', () => {
    const c = modelChoices([model('mini-n', null, 0.701), model('fast-a', 'r_fast', 0.6), model('mini-run', 'r_mini', 0.74)], ctx);
    expect(c.map((x) => [x.group, x.label])).toEqual([
      ['Validation split of fasteners', 'fast-a · mAP 0.600 on fasteners'],
      ['Training split of fasteners-mini', 'mini-run · mAP 0.740 on fasteners-mini · training split'],
      ['Imported · split unknown', 'mini-n · mAP 0.701 · split unknown'],
    ]);
    // Closed, a select shows the name alone; the score and all of its context go in the meta line under it.
    expect(c.map((x) => [x.short, x.meta])).toEqual([
      ['fast-a', 'mAP 0.600 · fasteners, validation split'],
      ['mini-run', 'mAP 0.740 · fasteners-mini, training split'],
      ['mini-n', 'mAP 0.701 · imported, split unknown'],
    ]);
  });
});

it('a context type with no split is never shown as validation', () => {
  const c: ScoreContext = { kind: 'run', datasetId: 'ds_x', datasetName: 'x', split: null };
  expect(scoreTag(c)).toBe('Split unknown');
  expect(scoreText(0.5, c)).toBe('mAP 0.500 · split unknown');
});
