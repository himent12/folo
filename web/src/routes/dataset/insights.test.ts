import { describe, expect, it } from 'vitest';
import type { Dataset, DatasetStats, IssueCounts } from '$lib/api/types';
import { computeDatasetInsights } from './insights';

function makeDataset(over: Partial<Dataset> = {}): Dataset {
  return {
    id: 'ds_test',
    name: 'test-ds',
    path: '/data/test-ds',
    format: 'yolo',
    classes: ['bolt', 'nut', 'washer'],
    created: '2026-11-01T00:00:00Z',
    splits: { train: 800, val: 150, test: 50 },
    ...over,
  };
}

function makeStats(over: Partial<DatasetStats> = {}): DatasetStats {
  return {
    boxes: 3000,
    empty_images: 0,
    broken_labels: 0,
    class_counts: [1100, 1000, 900],
    box_size_hist: { edges: [0, 0.05, 0.2, 1], counts: [50, 950, 2000] },
    boxes_per_image_hist: { edges: [0, 1, 2, 4], counts: [0, 400, 600] },
    ...over,
  };
}

describe('computeDatasetInsights', () => {
  it('reports ok balance and clean quality for a well-balanced dataset', () => {
    const ins = computeDatasetInsights(makeDataset(), makeStats());
    expect(ins.totalImages).toBe(1000);
    expect(ins.boxesPerImage).toBeCloseTo(3.0);
    expect(ins.imbalanceRatio).toBe(1.2);
    expect(ins.imbalanced).toBe(false);
    expect(ins.balanceTone).toBe('ok');
    expect(ins.qualityTone).toBe('ok');
    expect(ins.underrepresented).toEqual([]);
    expect(ins.missingClasses).toEqual([]);
    expect(ins.mostCommon?.name).toBe('bolt');
    expect(ins.rarest?.name).toBe('washer');
    expect(ins.splits.map((s) => [s.split, Math.round(s.pct)])).toEqual([
      ['train', 80],
      ['val', 15],
      ['test', 5],
    ]);
    expect(ins.noVal).toBe(false);
    expect(ins.noTrain).toBe(false);
  });

  it('flags severe class imbalance (>5x) and orders underrepresented classes rarest-first', () => {
    const ins = computeDatasetInsights(
      makeDataset(),
      makeStats({ boxes: 1270, class_counts: [1000, 220, 50] }),
    );
    expect(ins.imbalanceRatio).toBe(20.0);
    expect(ins.imbalanced).toBe(true);
    expect(ins.balanceTone).toBe('warning');
    expect(ins.underrepresented.map((c) => c.name)).toEqual(['washer']);
    expect(ins.rarest?.cls).toBe(2);
  });

  it('flags zero-box classes as danger and surfaces broken, empty, tiny, and check findings', () => {
    const counts: IssueCounts = { error: 2, warning: 4, info: 15, by_kind: { clipped_box: 4 } };
    const ins = computeDatasetInsights(
      makeDataset({ splits: { train: 100, val: 0, test: 0 } }),
      makeStats({
        boxes: 500,
        empty_images: 15,
        broken_labels: 2,
        class_counts: [400, 100, 0],
        box_size_hist: { edges: [0, 0.05, 1], counts: [120, 380] },
      }),
      counts,
    );
    expect(ins.missingClasses).toEqual([{ cls: 2, name: 'washer' }]);
    expect(ins.balanceTone).toBe('danger');
    expect(ins.qualityTone).toBe('danger');
    expect(ins.emptyWarn).toBe(true);
    expect(ins.tinyWarn).toBe(true);
    expect(ins.checkWarnings).toBe(4);
    expect(ins.noVal).toBe(true);
  });

  it('handles an empty dataset cleanly with zero NaNs or divisions by zero', () => {
    const ins = computeDatasetInsights(
      makeDataset({ splits: { train: 0, val: 0, test: 0 } }),
      makeStats({ boxes: 0, class_counts: [0, 0, 0], box_size_hist: { edges: [0, 1], counts: [0] } }),
    );
    expect(ins.totalImages).toBe(0);
    expect(ins.boxesPerImage).toBeNull();
    expect(ins.imbalanceRatio).toBeNull();
    expect(ins.emptyPct).toBe(0);
    expect(ins.tinyBoxPct).toBe(0);
    expect(ins.noTrain).toBe(true);
    expect(ins.noVal).toBe(true);
  });
});
