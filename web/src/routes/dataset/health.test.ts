import { describe, expect, it } from 'vitest';
import type { Dataset, DatasetStats } from '$lib/api/types';
import { bySeverity, countBinLabel, groupSummary, healthChecks, issueLabel, reviewCount, shareBelow, sizeBinLabel } from './health';

const ds = (over: Partial<Dataset> = {}): Dataset => ({
  id: 'ds_x',
  name: 'x',
  path: '/x',
  format: 'yolo',
  classes: ['a', 'b', 'c'],
  created: '2026-11-01T00:00:00Z',
  splits: { train: 800, val: 200, test: 0 },
  ...over,
});

const st = (over: Partial<DatasetStats> = {}): DatasetStats => ({
  boxes: 3000,
  empty_images: 0,
  broken_labels: 0,
  class_counts: [1000, 1000, 1000],
  box_size_hist: { edges: [0, 0.05, 0.1, 1], counts: [10, 990, 2000] },
  boxes_per_image_hist: { edges: [0, 1, 2, 4], counts: [0, 500, 500] },
  ...over,
});

describe('reviewCount', () => {
  it('counts exactly the error and warning findings', () => {
    const h = healthChecks(ds(), st({ broken_labels: 3, empty_images: 20, box_size_hist: { edges: [0, 0.05, 1], counts: [500, 2500] } }));
    expect(h.map((x) => x.severity)).toEqual(['danger', 'warning', 'info']);
    expect(reviewCount(h)).toBe(2);
    expect(reviewCount(healthChecks(ds(), st()))).toBe(0);
  });

  it('gives tiny boxes a concrete action', () => {
    const tiny = healthChecks(ds(), st({ box_size_hist: { edges: [0, 0.05, 1], counts: [500, 2500] } })).find((x) => x.id === 'tiny');
    expect(tiny?.action).toEqual({ label: 'Train at 960 px', path: '/runs/new?dataset=ds_x&imgsz=960' });
  });
});

describe('healthChecks', () => {
  it('reports ok for a clean dataset', () => {
    const h = healthChecks(ds(), st());
    expect(h.map((x) => x.id)).toEqual(['ok']);
  });

  it('flags imbalance above 5x, naming both classes', () => {
    const h = healthChecks(ds(), st({ class_counts: [1200, 600, 200] }));
    const im = h.find((x) => x.id === 'imbalance');
    expect(im?.severity).toBe('warning');
    expect(im?.title).toContain('6.0×');
    expect(im?.detail).toContain('a has 1,200');
    expect(h.some((x) => x.id === 'ok')).toBe(false);
  });

  it('puts broken labels and missing classes first', () => {
    const h = healthChecks(ds(), st({ broken_labels: 3, class_counts: [1000, 0, 1000], empty_images: 5 }));
    expect(h[0]?.severity).toBe('danger');
    expect(h[1]?.severity).toBe('danger');
    expect(h.at(-1)?.id).toBe('empty');
  });

  it('flags a tiny-box share above 10%', () => {
    const h = healthChecks(ds(), st({ box_size_hist: { edges: [0, 0.05, 0.1, 1], counts: [400, 600, 2000] } }));
    expect(h.find((x) => x.id === 'tiny')?.title).toBe('13% of boxes are tiny');
  });

  it('treats a few empty images as info, many as a warning', () => {
    expect(healthChecks(ds(), st({ empty_images: 12 })).find((x) => x.id === 'empty')?.severity).toBe('info');
    expect(healthChecks(ds(), st({ empty_images: 300 })).find((x) => x.id === 'empty')?.severity).toBe('warning');
  });

  it('warns without a validation split', () => {
    expect(healthChecks(ds({ splits: { train: 1000, val: 0, test: 0 } }), st()).some((x) => x.id === 'no-val')).toBe(true);
  });
});

describe('healthChecks with the check counts', () => {
  const counts = (error: number, warning: number) => ({ error, warning, info: 0, by_kind: {} });

  it('adds the check warnings, with their issues to list', () => {
    const h = healthChecks(ds(), st(), counts(0, 4));
    const w = h.find((x) => x.id === 'check-warnings');
    expect(w?.title).toBe('4 warnings from the dataset check');
    expect(w?.short).toBe('4 check warnings');
    expect(w?.issues).toBe('warning');
    expect(reviewCount(h)).toBe(1);
  });

  it('counts errors the stats do not (unreadable images), but never twice', () => {
    expect(healthChecks(ds(), st(), counts(2, 0)).map((x) => x.id)).toEqual(['check-errors']);
    const broken = healthChecks(ds(), st({ broken_labels: 3 }), counts(3, 0));
    expect(broken.map((x) => x.id)).toEqual(['broken']);
    expect(broken[0]?.issues).toBe('error');
    // Label errors of every kind are the broken files; only image errors (unreadable, missing) are a finding of their own.
    const labels = healthChecks(ds(), st({ broken_labels: 3 }), { ...counts(3, 0), by_kind: { broken_label: 1, class_overflow: 1, out_of_range_box: 1 } });
    expect(labels.map((x) => x.id)).toEqual(['broken']);
    const mixed = healthChecks(ds(), st({ broken_labels: 3 }), { ...counts(5, 0), by_kind: { broken_label: 3, corrupt_image: 1, missing_image: 1 } });
    expect(mixed.map((x) => x.id)).toEqual(['broken', 'check-errors']);
    expect(mixed.map((x) => x.files)).toEqual([3, 2]);
  });

  it('stays as before without counts (older servers)', () => {
    expect(healthChecks(ds(), st()).map((x) => x.id)).toEqual(['ok']);
    expect(healthChecks(ds(), st(), counts(0, 0)).map((x) => x.id)).toEqual(['ok']);
  });
});

describe('labels', () => {
  it("labels the server's box-size bins in percent of the image side", () => {
    const edges = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1];
    const labels = edges.slice(0, -1).map((lo, i) => sizeBinLabel(lo, edges[i + 1] ?? 1));
    expect(labels).toEqual(['<1%', '1–2%', '2–5%', '5–10%', '10–20%', '20–50%', '≥50%']);
  });

  it('labels integer count bins', () => {
    expect(countBinLabel(0, 1, false)).toBe('0');
    expect(countBinLabel(2, 4, false)).toBe('2–3');
    expect(countBinLabel(32, 64, true)).toBe('32+');
  });

  it('computes the share below a limit', () => {
    expect(shareBelow({ edges: [0, 1, 2, 3], counts: [1, 1, 2] }, 2)).toBe(0.5);
  });
});

describe('bySeverity', () => {
  it('groups findings into errors, warnings and notes, skipping empty groups and ok', () => {
    const h = healthChecks(ds(), st({ broken_labels: 3, empty_images: 20, box_size_hist: { edges: [0, 0.05, 1], counts: [500, 2500] } }));
    const g = bySeverity(h);
    expect(g.map((x) => [x.label, x.items.map((i) => i.id)])).toEqual([
      ['Errors', ['broken']],
      ['Warnings', ['tiny']],
      ['Notes', ['empty']],
    ]);
    expect(bySeverity(healthChecks(ds(), st()))).toEqual([]);
  });
});

describe('summary copy', () => {
  it('says the issue total once, as issues (each finding that needs attention is one)', () => {
    expect(issueLabel(0)).toBeUndefined();
    expect(issueLabel(1)).toBe('1 issue');
    expect(issueLabel(3)).toBe('3 issues');
    const h = healthChecks(
      ds(),
      st({ broken_labels: 3, empty_images: 20, box_size_hist: { edges: [0, 0.05, 1], counts: [1050, 1950] } }),
      { error: 3, warning: 3, info: 20, by_kind: {} },
    );
    // Errors: the broken files; warnings: the check's warnings and the tiny boxes; the note is not an issue.
    expect(reviewCount(h)).toBe(3);
    expect(issueLabel(reviewCount(healthChecks(ds(), st({ empty_images: 1 }))))).toBeUndefined();
  });

  it('folds a severity into a complete short line, units named, not the joined titles', () => {
    const h = healthChecks(ds(), st({ box_size_hist: { edges: [0, 0.05, 1], counts: [1050, 1950] } }), { error: 0, warning: 3, info: 0, by_kind: {} });
    const warnings = bySeverity(h).find((g) => g.severity === 'warning')!;
    expect(groupSummary(warnings)).toBe('3 check warnings · 35% tiny boxes');
  });
});
