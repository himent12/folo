import { describe, expect, it } from 'vitest';
import type { BenchEntryImage } from '$lib/api/types';
import { diagnoseImage, filterAndSortImages, summarizeGalleryErrors } from './errorAnalysis';

const CLASSES = ['bolt', 'nut', 'washer'];

function makeImage(
  id: string,
  truth: BenchEntryImage['image']['boxes'],
  detections: BenchEntryImage['detections'],
  missed: number,
  wrong: number,
): BenchEntryImage {
  return {
    image: { id, file: `${id}.jpg`, width: 200, height: 200, boxes: truth },
    detections,
    missed,
    wrong,
  };
}

describe('Per-image error-analysis diagnostics', () => {
  it('classifies TP, duplicate FP, localization FP, class-mismatch FP/FN, background FP, and pure missed FN', () => {
    const im = makeImage(
      'img_diag',
      [
        { class: 0, x1: 0, y1: 0, x2: 20, y2: 20 }, // matched by det 0 (TP) + overlapped by det 1 (duplicate FP)
        { class: 1, x1: 50, y1: 50, x2: 70, y2: 70 }, // missed GT overlapped by det 2 of class 0 (class-mismatch)
        { class: 2, x1: 100, y1: 100, x2: 120, y2: 120 }, // missed GT overlapped by low-IoU same-class det 3 (localization)
        { class: 0, x1: 160, y1: 160, x2: 180, y2: 180 }, // completely missed GT
      ],
      [
        { class: 0, score: 0.95, x1: 0, y1: 0, x2: 20, y2: 20 }, // TP IoU 1.0
        { class: 0, score: 0.72, x1: 1, y1: 0, x2: 21, y2: 20 }, // duplicate FP
        { class: 0, score: 0.81, x1: 50, y1: 50, x2: 70, y2: 70 }, // class mismatch (GT: nut)
        { class: 2, score: 0.64, x1: 100, y1: 100, x2: 110, y2: 110 }, // localization FP (IoU 0.25)
        { class: 1, score: 0.52, x1: 0, y1: 160, x2: 10, y2: 170 }, // background FP
      ],
      3,
      4,
    );

    const d = diagnoseImage(im, CLASSES);
    expect([d.tp, d.fp, d.fn, d.errors]).toEqual([1, 4, 3, 7]);
    expect(d.meanIou).toBeCloseTo(1, 5);
    expect(d.minIou).toBeCloseTo(1, 5);
    expect(d.minScore).toBeCloseTo(0.52, 5);

    const byKey = new Map(d.boxes.map((b) => [b.key, b]));
    expect(byKey.get('d0')?.cause).toBe('tp');
    expect(byKey.get('d1')?.cause).toBe('duplicate');
    expect(byKey.get('d2')?.cause).toBe('class-mismatch');
    expect(byKey.get('d2')?.detail).toContain('GT: nut');
    expect(byKey.get('d3')?.cause).toBe('localization');
    expect(byKey.get('d4')?.cause).toBe('background');

    expect(byKey.get('t1')?.cause).toBe('class-mismatch');
    expect(byKey.get('t1')?.detail).toContain('pred bolt');
    expect(byKey.get('t2')?.cause).toBe('localization');
    expect(byKey.get('t3')?.cause).toBe('missed');
  });

  it('filters by error focus and sorts deterministically by worst, FN, FP, lowest IoU, lowest score, and order', () => {
    const cleanHighIou = makeImage(
      'clean',
      [{ class: 0, x1: 0, y1: 0, x2: 20, y2: 20 }],
      [{ class: 0, score: 0.92, x1: 0, y1: 0, x2: 20, y2: 20 }],
      0,
      0,
    );
    const lowIouWithFp = makeImage(
      'low_iou_fp',
      [{ class: 0, x1: 0, y1: 0, x2: 20, y2: 20 }],
      [
        { class: 0, score: 0.85, x1: 0, y1: 0, x2: 20, y2: 12 }, // IoU 0.60
        { class: 1, score: 0.31, x1: 80, y1: 80, x2: 90, y2: 90 }, // FP (lowest score 0.31)
        { class: 1, score: 0.45, x1: 120, y1: 120, x2: 130, y2: 130 }, // FP
      ],
      0,
      2,
    );
    const heavyFn = makeImage(
      'heavy_fn',
      [
        { class: 0, x1: 0, y1: 0, x2: 20, y2: 20 },
        { class: 0, x1: 40, y1: 40, x2: 60, y2: 60 },
        { class: 0, x1: 80, y1: 80, x2: 100, y2: 100 },
      ],
      [{ class: 0, score: 0.78, x1: 0, y1: 0, x2: 20, y2: 18 }], // IoU 0.90, 2 FN
      2,
      0,
    );
    const mixedBoth = makeImage(
      'mixed',
      [
        { class: 0, x1: 0, y1: 0, x2: 20, y2: 20 },
        { class: 1, x1: 50, y1: 50, x2: 70, y2: 70 },
      ],
      [
        { class: 0, score: 0.65, x1: 0, y1: 0, x2: 20, y2: 16 }, // IoU 0.80
        { class: 2, score: 0.55, x1: 100, y1: 100, x2: 120, y2: 120 }, // FP
        { class: 2, score: 0.5, x1: 130, y1: 130, x2: 150, y2: 150 }, // FP
        { class: 2, score: 0.48, x1: 160, y1: 160, x2: 180, y2: 180 }, // FP
      ],
      1,
      3,
    );

    const list = [cleanHighIou, lowIouWithFp, heavyFn, mixedBoth];
    const summary = summarizeGalleryErrors(list);
    expect(summary).toEqual({ total: 4, withFn: 2, withFp: 2, withErrors: 3, tp: 4, fp: 5, fn: 3 });

    const ids = (filter: Parameters<typeof filterAndSortImages>[1], sort: Parameters<typeof filterAndSortImages>[2]) =>
      filterAndSortImages(list, filter, sort, CLASSES).map((x) => x.item.image.id);

    expect(ids('all', 'order')).toEqual(['clean', 'low_iou_fp', 'heavy_fn', 'mixed']);
    expect(ids('all', 'worst')).toEqual(['mixed', 'low_iou_fp', 'heavy_fn', 'clean']);
    expect(ids('fn', 'fn')).toEqual(['heavy_fn', 'mixed']);
    expect(ids('fp', 'fp')).toEqual(['mixed', 'low_iou_fp']);
    expect(ids('errors', 'order')).toEqual(['low_iou_fp', 'heavy_fn', 'mixed']);
    expect(ids('all', 'lowest-iou')).toEqual(['low_iou_fp', 'mixed', 'heavy_fn', 'clean']);
    expect(ids('all', 'lowest-score')).toEqual(['low_iou_fp', 'mixed', 'heavy_fn', 'clean']);
  });
});
