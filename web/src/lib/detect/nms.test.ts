import { describe, expect, it } from 'vitest';
import { effectiveMinConf, filterDetections, iou, nms, thresholdForClass, type NmsBox } from './nms';

const box = (cls: number, score: number, x1: number, y1: number, x2: number, y2: number): NmsBox => ({ class: cls, score, x1, y1, x2, y2 });

describe('iou', () => {
  it('is 1 for identical boxes and 0 for disjoint or touching ones', () => {
    const a = box(0, 1, 0, 0, 10, 10);
    expect(iou(a, a)).toBe(1);
    expect(iou(a, box(0, 1, 20, 20, 30, 30))).toBe(0);
    expect(iou(a, box(0, 1, 10, 0, 20, 10))).toBe(0);
  });

  it('computes partial overlap', () => {
    // 5x10 overlap = 50; union = 100 + 100 - 50 = 150.
    expect(iou(box(0, 1, 0, 0, 10, 10), box(0, 1, 5, 0, 15, 10))).toBeCloseTo(1 / 3, 10);
  });

  it('is 0 for degenerate boxes', () => {
    expect(iou(box(0, 1, 0, 0, 0, 0), box(0, 1, 0, 0, 0, 0))).toBe(0);
  });
});

describe('nms', () => {
  it('suppresses lower-scored overlapping boxes of the same class', () => {
    const a = box(0, 0.9, 0, 0, 10, 10);
    const b = box(0, 0.8, 1, 0, 11, 10); // IoU 0.82 with a
    const c = box(0, 0.7, 50, 50, 60, 60);
    expect(nms([b, c, a], 0.5)).toEqual([a, c]);
  });

  it('is class aware: overlapping boxes of different classes both survive', () => {
    const a = box(0, 0.9, 0, 0, 10, 10);
    const b = box(1, 0.8, 0, 0, 10, 10);
    expect(nms([a, b], 0.5)).toEqual([a, b]);
  });

  it('keeps a box whose overlap equals the threshold (suppresses only above it)', () => {
    const a = box(0, 0.9, 0, 0, 10, 10);
    const b = box(0, 0.8, 5, 0, 15, 10); // IoU exactly 1/3
    expect(nms([a, b], 1 / 3)).toHaveLength(2);
    expect(nms([a, b], 0.3)).toEqual([a]);
  });

  it('only compares against kept boxes (greedy chains)', () => {
    // b overlaps a and c; a suppresses b, so c (which overlaps only b) survives.
    const a = box(0, 0.9, 0, 0, 10, 10);
    const b = box(0, 0.8, 4, 0, 14, 10);
    const c = box(0, 0.7, 8, 0, 18, 10);
    expect(nms([a, b, c], 0.3)).toEqual([a, c]);
  });

  it('caps the output at maxDet and orders by score with stable ties', () => {
    const xs = [box(0, 0.5, 0, 0, 1, 1), box(1, 0.5, 0, 0, 1, 1), box(2, 0.9, 0, 0, 1, 1)];
    expect(nms(xs, 0.5)).toEqual([xs[2], xs[0], xs[1]]);
    expect(nms(xs, 0.5, 2)).toEqual([xs[2], xs[0]]);
    expect(nms(xs, 0.5, 0)).toEqual([]);
  });

  it('does not mutate its input', () => {
    const xs = [box(0, 0.1, 0, 0, 1, 1), box(0, 0.9, 0, 0, 1, 1)];
    const copy = [...xs];
    nms(xs, 0.5);
    expect(xs).toEqual(copy);
  });
});

describe('filterDetections', () => {
  const xs = [
    box(0, 0.95, 0, 0, 100, 100),
    box(0, 0.6, 5, 5, 105, 105), // duplicate of the first (IoU ~0.82)
    box(1, 0.4, 200, 200, 260, 260),
    box(2, 0.05, 300, 0, 340, 40),
  ];

  it('filters by confidence before NMS', () => {
    expect(filterDetections(xs, { conf: 0.5, iou: 0.7, maxDet: 300 })).toEqual([xs[0]]);
    expect(filterDetections(xs, { conf: 0.01, iou: 0.7, maxDet: 300 })).toEqual([xs[0], xs[2], xs[3]]);
  });

  it('a loose IoU threshold keeps duplicates', () => {
    expect(filterDetections(xs, { conf: 0.5, iou: 0.9, maxDet: 300 })).toEqual([xs[0], xs[1]]);
  });

  it('applies max detections last', () => {
    expect(filterDetections(xs, { conf: 0, iou: 0.7, maxDet: 1 })).toEqual([xs[0]]);
  });

  it('applies per-class confidence overrides with global fallback and inclusive/strict boundary control', () => {
    expect(thresholdForClass(1, 0.25, { 1: 0.5 })).toBe(0.5);
    expect(thresholdForClass(0, 0.25, { 1: 0.5 })).toBe(0.25);
    expect(effectiveMinConf(0.25, { 0: 0.6, 2: 0.04 })).toBe(0.04);
    expect(effectiveMinConf(0.25, { 0: 0.6 })).toBe(0.25);

    // Raise class 1 to 0.5 (drops score 0.4), lower class 2 to 0.05 (keeps score 0.05 inclusively)
    expect(
      filterDetections(xs, {
        conf: 0.25,
        iou: 0.7,
        maxDet: 300,
        perClassConf: { 1: 0.5, 2: 0.05 },
      }),
    ).toEqual([xs[0], xs[3]]);

    // Strict (`confInclusive: false`) excludes exact boundary score 0.05
    expect(
      filterDetections(xs, {
        conf: 0.25,
        iou: 0.7,
        maxDet: 300,
        perClassConf: { 1: 0.5, 2: 0.05 },
        confInclusive: false,
      }),
    ).toEqual([xs[0]]);
  });
});
