import { describe, expect, it } from 'vitest';
import type { Model, Prediction } from '$lib/api/types';
import { DEFAULTS } from './predict';
import {
  areDefaultThresholds,
  classCounts,
  detectionIndex,
  deviceLabel,
  nextRowIndex,
  predictionJson,
  smallerModelOf,
  sourceDims,
  sourceFullName,
  sourceShortName,
  visibleClasses,
  visibleRows,
  whereLabel,
} from './viewState';

const makeModel = (id: string, arch: string, classes: string[]): Model =>
  ({ id, name: id, arch, classes, head: 'o2m' }) as unknown as Model;

const samplePred: Prediction = {
  model_id: 'm1',
  image: { width: 640, height: 480 },
  boxes: [
    { class: 1, name: 'square', score: 0.9, x1: 0, y1: 0, x2: 10, y2: 10 },
    { class: 0, name: 'circle', score: 0.8, x1: 20, y1: 20, x2: 40, y2: 40 },
    { class: 1, name: 'square', score: 0.7, x1: 50, y1: 50, x2: 70, y2: 70 },
  ],
  timing: { preprocess_ms: 1, inference_ms: 2, postprocess_ms: 0.5, total_ms: 3.5 },
  device: 'gpu:0',
};

describe('playground/viewState', () => {
  it('finds a smaller model with matching classes only', () => {
    const small = makeModel('m_n', 'folo-n', ['a', 'b']);
    const otherClasses = makeModel('m_n_other', 'folo-n', ['a', 'c']);
    const medium = makeModel('m_m', 'folo-m', ['a', 'b']);
    expect(smallerModelOf(medium, [otherClasses, small, medium])).toBe(small);
    expect(smallerModelOf(small, [small, medium])).toBeNull();
    expect(smallerModelOf(null, [small])).toBeNull();
  });

  it('formats where and device labels for server and browser predictions', () => {
    expect(whereLabel(null)).toBe('server');
    expect(whereLabel(samplePred)).toBe('server');
    expect(whereLabel({ ...samplePred, backend: 'webgpu' })).toBe('browser · WebGPU');
    expect(deviceLabel(samplePred, null)).toBe('gpu:0');
    expect(deviceLabel({ ...samplePred, backend: 'webgpu' }, 'RTX 4080')).toBe('WebGPU · RTX 4080');
    expect(deviceLabel({ ...samplePred, backend: 'cpu' }, 'ignored')).toBe('CPU');
  });

  it('derives source dimensions and names from prediction or source', () => {
    const fileSource = { kind: 'file' as const, file: new Blob(), name: 'photo.jpg', url: 'blob:1', width: 320, height: 240 };
    expect(sourceDims(null, fileSource)).toEqual({ width: 320, height: 240 });
    expect(sourceDims(samplePred, fileSource)).toEqual({ width: 640, height: 480 });
    expect(sourceFullName(fileSource)).toBe('photo.jpg');
    expect(sourceShortName(fileSource)).toBe('photo.jpg');
  });

  it('computes visible classes, rows, class counts, and detection indices', () => {
    const hidden = new Set([0]);
    expect(visibleClasses(['circle', 'square'], new Set())).toBeNull();
    expect([...(visibleClasses(['circle', 'square'], hidden) ?? [])]).toEqual([1]);
    expect(visibleRows(samplePred.boxes, hidden).map((r) => r.i)).toEqual([0, 2]);
    expect(classCounts(samplePred.boxes)).toEqual([
      [1, 2],
      [0, 1],
    ]);
    expect(detectionIndex(samplePred.boxes[1]!, samplePred.boxes, hidden)).toBeNull();
    expect(detectionIndex(samplePred.boxes[2]!, samplePred.boxes, hidden)).toBe(2);
  });

  it('clamps keyboard row navigation indices', () => {
    expect(nextRowIndex('ArrowDown', 0, 3)).toBe(1);
    expect(nextRowIndex('ArrowDown', 2, 3)).toBe(2);
    expect(nextRowIndex('ArrowUp', 0, 3)).toBe(0);
    expect(nextRowIndex('Home', 2, 3)).toBe(0);
    expect(nextRowIndex('End', 0, 3)).toBe(2);
    expect(nextRowIndex('Enter', 1, 3)).toBeNull();
    expect(nextRowIndex('ArrowDown', 0, 0)).toBeNull();
  });

  it('omits IoU from default check and JSON payload for end-to-end (o2o) models', () => {
    expect(areDefaultThresholds(DEFAULTS.conf, 0.3, DEFAULTS.maxDet, true)).toBe(true);
    expect(areDefaultThresholds(DEFAULTS.conf, 0.3, DEFAULTS.maxDet, false)).toBe(false);
    expect(areDefaultThresholds(0.5, DEFAULTS.iou, DEFAULTS.maxDet, true)).toBe(false);

    const o2oJson = predictionJson(samplePred, true, samplePred.boxes, { conf: 0.25, iou: 0.7, maxDet: 300, o2o: true });
    expect(o2oJson?.filter).toEqual({ conf: 0.25, max_det: 300 });
    expect('iou' in (o2oJson?.filter ?? {})).toBe(false);

    const o2mJson = predictionJson(samplePred, true, samplePred.boxes, { conf: 0.25, iou: 0.7, maxDet: 300, o2o: false });
    expect(o2mJson?.filter).toEqual({ conf: 0.25, iou: 0.7, max_det: 300 });
  });
});
