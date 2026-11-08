import { describe, expect, it } from 'vitest';
import type { Dataset, DatasetStats } from '$lib/api/types';
import { countBinLabel, healthChecks, shareBelow, sizeBinLabel } from './health';

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

describe('labels', () => {
  it('labels box-size bins in percent of the image side', () => {
    const edges = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1];
    const labels = edges.slice(0, -1).map((lo, i) => sizeBinLabel(lo, edges[i + 1] ?? 1));
    expect(labels).toEqual(['<2%', '2–5%', '5–10%', '10–15%', '15–20%', '20–30%', '30–50%', '≥50%']);
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
