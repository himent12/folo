import { describe, expect, it } from 'vitest';
import type { Model, Prediction } from '$lib/api/types';
import {
  comparePredictionJson,
  formatCompareTimingMeta,
  formatLatencyComparison,
  formatSignedDelta,
  pickDefaultModelB,
  summarizeComparison,
} from './compare';
import { fittedImageRect, panToPx, pxToNormPan, viewportSplitToImagePct } from './zoomState';

const makeModel = (id: string, name: string, classes: string[], head: 'o2m' | 'o2o' = 'o2m'): Model =>
  ({ id, name, arch: 'folo-n', classes, head }) as unknown as Model;

const predA: Prediction = {
  model_id: 'm_a',
  image: { width: 640, height: 480 },
  boxes: [
    { class: 0, name: 'circle', score: 0.9, x1: 10, y1: 10, x2: 50, y2: 50 },
    { class: 1, name: 'square', score: 0.8, x1: 60, y1: 60, x2: 120, y2: 120 },
  ],
  timing: { preprocess_ms: 2, inference_ms: 18, postprocess_ms: 1, total_ms: 21 },
  device: 'gpu:0',
};

const predB: Prediction = {
  model_id: 'm_b',
  image: { width: 640, height: 480 },
  boxes: [
    { class: 0, name: 'circle', score: 0.95, x1: 12, y1: 12, x2: 52, y2: 52 },
    { class: 0, name: 'circle', score: 0.72, x1: 200, y1: 200, x2: 240, y2: 240 },
    { class: 2, name: 'triangle', score: 0.85, x1: 300, y1: 100, x2: 360, y2: 160 },
  ],
  timing: { preprocess_ms: 2, inference_ms: 7.5, postprocess_ms: 0.5, total_ms: 10 },
  device: 'gpu:0',
};

describe('playground/compare', () => {
  it('picks a distinct Model B with matching classes first, then any distinct model, then Model A', () => {
    const a = makeModel('m_a', 'shapes-s', ['circle', 'square']);
    const other = makeModel('m_other', 'coco-n', ['person', 'car']);
    const sameClasses = makeModel('m_b', 'shapes-n', ['circle', 'square']);

    expect(pickDefaultModelB(a, [a, other, sameClasses])).toBe(sameClasses);
    expect(pickDefaultModelB(a, [a, other])).toBe(other);
    expect(pickDefaultModelB(a, [a])).toBe(a);
    expect(pickDefaultModelB(a, [])).toBeNull();
  });

  it('summarizes detection count deltas, per-class counts, and latency speedup', () => {
    const sum = summarizeComparison({
      boxesA: predA.boxes,
      boxesB: predB.boxes,
      predA,
      predB,
    });

    expect(sum.countA).toBe(2);
    expect(sum.countB).toBe(3);
    expect(sum.countDelta).toBe(1);
    expect(sum.latencyA).toBe(21);
    expect(sum.latencyB).toBe(10);
    expect(sum.latencyDeltaMs).toBe(-11);
    expect(sum.speedupRatio).toBeCloseTo(2.1);

    expect(sum.classes).toEqual([
      { name: 'circle', colorClass: 0, countA: 1, countB: 2, delta: 1 },
      { name: 'square', colorClass: 1, countA: 1, countB: 0, delta: -1 },
      { name: 'triangle', colorClass: 2, countA: 0, countB: 1, delta: 1 },
    ]);
  });

  it('formats signed deltas and latency comparisons clearly', () => {
    expect(formatSignedDelta(3)).toBe('+3');
    expect(formatSignedDelta(-2)).toBe('-2');
    expect(formatSignedDelta(0)).toBe('0');

    expect(formatLatencyComparison({ latencyA: 20, latencyB: 10, latencyDeltaMs: -10, speedupRatio: 2 })).toContain('B 2.0× faster');
    expect(formatLatencyComparison({ latencyA: 10, latencyB: 25, latencyDeltaMs: 15, speedupRatio: 0.4 })).toContain('A 2.5× faster');
    expect(formatLatencyComparison({ latencyA: 10, latencyB: 10.01, latencyDeltaMs: 0.01, speedupRatio: 0.999 })).toBe('Same latency');
    expect(formatLatencyComparison({ latencyA: null, latencyB: 10, latencyDeltaMs: null, speedupRatio: null })).toBe('—');
  });

  it('builds structured Compare A/B JSON with per-model metadata and deltas', () => {
    const summary = summarizeComparison({ boxesA: predA.boxes, boxesB: predB.boxes, predA, predB });
    const payload = comparePredictionJson({
      view: 'split',
      predA,
      predB,
      hasResultA: true,
      hasResultB: true,
      boxesA: predA.boxes,
      boxesB: predB.boxes,
      modelAName: 'shapes-s',
      modelBName: 'shapes-n',
      runOnA: 'server',
      runOnB: 'cpu',
      o2oA: true,
      o2oB: false,
      filter: { conf: 0.3, iou: 0.65, maxDet: 100 },
      summary,
    });

    expect(payload).not.toBeNull();
    expect(payload?.mode).toBe('compare');
    expect(payload?.view).toBe('split');
    expect(payload?.filter).toEqual({ conf: 0.3, iou: 0.65, max_det: 100 });
    expect(payload?.comparison.count_delta).toBe(1);
    expect(payload?.a?.model_name).toBe('shapes-s');
    expect(payload?.a?.run_on).toBe('server');
    expect('iou' in (payload?.a?.filter ?? {})).toBe(false);
    expect(payload?.b?.model_name).toBe('shapes-n');
    expect(payload?.b?.run_on).toBe('cpu');
    expect(payload?.b?.filter).toEqual({ conf: 0.3, iou: 0.65, max_det: 100 });
  });

  it('tracks unavailable model results as null without fabricating zero counts or misleading deltas', () => {
    const sum = summarizeComparison({
      boxesA: predA.boxes,
      boxesB: [],
      predA,
      predB: null,
    });
    expect(sum.countA).toBe(2);
    expect(sum.countB).toBeNull();
    expect(sum.countDelta).toBeNull();
    expect(formatSignedDelta(sum.countDelta)).toBe('—');
    expect(sum.classes).toEqual([
      { name: 'circle', colorClass: 0, countA: 1, countB: null, delta: null },
      { name: 'square', colorClass: 1, countA: 1, countB: null, delta: null },
    ]);

    const payload = comparePredictionJson({
      view: 'side-by-side',
      predA,
      predB: null,
      hasResultA: true,
      hasResultB: false,
      boxesA: predA.boxes,
      boxesB: [],
      modelAName: 'shapes-s',
      modelBName: 'shapes-n',
      runOnA: 'server',
      runOnB: 'server',
      o2oA: false,
      o2oB: false,
      filter: { conf: 0.25, iou: 0.7, maxDet: 300 },
      summary: sum,
    });
    expect(payload?.a).not.toBeNull();
    expect(payload?.b).toBeNull();
    expect(payload?.comparison.count_a).toBe(2);
    expect(payload?.comparison.count_b).toBeNull();
    expect(payload?.comparison.count_delta).toBeNull();
  });

  it('converts live split-overlay divider through letterbox, zoom, and pan into exact image space', () => {
    // Portrait image 200x400 inside a wide 1000x400 viewport -> ox = 400, rw = 200.
    // A 40% viewport split sits at x=400 (the left edge of the portrait image = 0% image space).
    expect(
      viewportSplitToImagePct({ splitPct: 40, viewportW: 1000, viewportH: 400, imgW: 200, imgH: 400, z: 1, nx: 0 }),
    ).toBeCloseTo(0);
    // 50% viewport split sits at x=500 -> center of portrait image = 50% image space.
    expect(
      viewportSplitToImagePct({ splitPct: 50, viewportW: 1000, viewportH: 400, imgW: 200, imgH: 400, z: 1, nx: 0 }),
    ).toBeCloseTo(50);
    // 60% viewport split sits at x=600 -> right edge of portrait image = 100% image space.
    expect(
      viewportSplitToImagePct({ splitPct: 60, viewportW: 1000, viewportH: 400, imgW: 200, imgH: 400, z: 1, nx: 0 }),
    ).toBeCloseTo(100);
  });

  it('keeps an off-center image landmark centered when switching between Side-by-side and Split viewports', () => {
    // Wide 1200x400 image in a narrow 400x400 Side-by-side pane vs wide 800x400 Split viewport.
    const sideRect = fittedImageRect(400, 400, 1200, 400);
    const splitRect = fittedImageRect(800, 400, 1200, 400);
    const z = 2;
    // Center on a landmark offset (+0.15 * rw, -0.20 * rh) from image center -> tx = -0.15 * rw * z, ty = +0.20 * rh * z.
    const nx = pxToNormPan(-0.15 * sideRect.rw * z, sideRect.rw, 400, z);
    const ny = pxToNormPan(0.2 * sideRect.rh * z, sideRect.rh, 400, z);

    const splitTx = panToPx(nx, splitRect.rw, 800, z);
    const splitTy = panToPx(ny, splitRect.rh, 400, z);
    // In Split mode, the exact same relative image landmark remains centered.
    expect(splitTx / (splitRect.rw * z)).toBeCloseTo(-0.15);
    expect(splitTy / (splitRect.rh * z)).toBeCloseTo(0.2);
  });

  it('formats collapsed mobile Timing header for both Compare sides without fabricating missing ones', () => {
    expect(formatCompareTimingMeta(null, null)).toBeNull();
    expect(formatCompareTimingMeta(predA, null)).toBe('A 21.0 ms');
    expect(formatCompareTimingMeta(null, predB)).toBe('B 10.0 ms');
    expect(formatCompareTimingMeta(predA, predB)).toBe('A 21.0 ms · B 10.0 ms');
  });
});
