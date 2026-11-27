// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Prediction } from '$lib/api/types';
import { toCsv } from '$lib/tableExport';
import { toast } from '$lib/ui/toast';
import {
  BATCH_CSV_HEADERS,
  BATCH_SUMMARY_CSV_HEADERS,
  batchPredKey,
  batchQaCounts,
  buildBatchCsvRows,
  buildBatchJson,
  buildBatchSummaryCsvRows,
  exportBatchCsv,
  exportBatchJson,
  exportBatchSummaryCsv,
  filterBatchBoxes,
  filterBatchItems,
  matchesBatchQaFilter,
  summarizeBatch,
  type BatchFilter,
  type BatchItem,
} from './batch';

function makePred(boxes: Prediction['boxes'], width = 640, height = 480): Prediction {
  return {
    model_id: 'm_shapes_s',
    image: { width, height },
    boxes,
    timing: { preprocess_ms: 1.2, inference_ms: 8.5, postprocess_ms: 0.3, total_ms: 10 },
    device: 'cpu',
  };
}

function makeItem(over: Partial<BatchItem> = {}): BatchItem {
  return {
    id: 'batch_1',
    file: new Blob(['img'], { type: 'image/png' }),
    name: 'frame-01.png',
    url: 'blob:frame-01',
    width: 640,
    height: 480,
    pred: null,
    predKey: null,
    status: 'idle',
    error: null,
    ...over,
  };
}

const DEFAULT_FILTER: BatchFilter = { conf: 0.25, iou: 0.7, maxDet: 300, o2o: false };

afterEach(() => {
  for (const t of toast.items) toast.dismiss(t.id);
  vi.restoreAllMocks();
});

describe('playground batch helpers', () => {
  it('filters cached predictions with live thresholds, o2o NMS bypass, and hidden classes', () => {
    const pred = makePred([
      { class: 0, name: 'circle', score: 0.92, x1: 10, y1: 10, x2: 110, y2: 110 },
      { class: 0, name: 'circle', score: 0.85, x1: 12, y1: 12, x2: 112, y2: 112 }, // high IoU overlap with box 0
      { class: 1, name: 'square', score: 0.6, x1: 200, y1: 50, x2: 300, y2: 150 },
      { class: 2, name: 'triangle', score: 0.15, x1: 320, y1: 60, x2: 400, y2: 140 }, // below conf 0.25
    ]);

    expect(filterBatchBoxes(pred, DEFAULT_FILTER, new Set())).toHaveLength(2);
    expect(filterBatchBoxes(pred, { ...DEFAULT_FILTER, o2o: true }, new Set())).toHaveLength(3);
    expect(filterBatchBoxes(pred, DEFAULT_FILTER, new Set([1]))).toHaveLength(1);
    expect(filterBatchBoxes(null, DEFAULT_FILTER, new Set())).toEqual([]);
  });

  it('summarizes batch progress and visible detection totals across items', () => {
    const key = batchPredKey('m_shapes_s', 'server');
    const items: BatchItem[] = [
      makeItem({
        id: 'b1',
        name: 'a.png',
        status: 'done',
        predKey: key,
        pred: makePred([
          { class: 0, name: 'circle', score: 0.9, x1: 0, y1: 0, x2: 40, y2: 40 },
          { class: 1, name: 'square', score: 0.1, x1: 50, y1: 50, x2: 90, y2: 90 },
        ]),
      }),
      makeItem({ id: 'b2', name: 'b.png', status: 'running' }),
      makeItem({ id: 'b3', name: 'c.png', status: 'error', error: 'HTTP 500' }),
      makeItem({ id: 'b4', name: 'd.png', status: 'idle' }),
    ];

    expect(summarizeBatch(items, DEFAULT_FILTER, new Set())).toEqual({
      total: 4,
      done: 1,
      running: 1,
      failed: 1,
      pending: 2,
      totalDetections: 1,
      truncated: 0,
    });
  });

  it('builds structured Batch JSON and RFC-4180 Batch CSV rows with expected columns', () => {
    const key = batchPredKey('m_shapes_s', 'server');
    const items: BatchItem[] = [
      makeItem({
        id: 'b1',
        name: '=evil.png',
        width: 800,
        height: 600,
        status: 'done',
        predKey: key,
        pred: makePred(
          [
            { class: 0, name: 'circle', score: 0.91234, x1: 10.126, y1: 20.5, x2: 110.25, y2: 120.75 },
            { class: 1, name: 'square', score: 0.55, x1: 150, y1: 40, x2: 250, y2: 140 },
          ],
          800,
          600,
        ),
      }),
      makeItem({
        id: 'b2',
        name: 'empty.png',
        status: 'done',
        predKey: key,
        pred: makePred([], 640, 480),
      }),
      makeItem({
        id: 'b3',
        name: 'broken.png',
        status: 'error',
        error: 'Decode failed',
      }),
    ];

    const json = buildBatchJson({
      items,
      modelId: 'm_shapes_s',
      modelName: 'shapes-s-best',
      runOn: 'server',
      filter: DEFAULT_FILTER,
      hidden: new Set([1]),
      classes: ['circle', 'square'],
    });
    expect(json.summary).toEqual({
      total_images: 3,
      completed_images: 2,
      failed_images: 1,
      total_detections: 1,
    });
    expect(json.filter).toEqual({ conf: 0.25, iou: 0.7, max_det: 300, hidden_classes: ['square'] });
    expect(json.images[0]!.boxes).toHaveLength(1);
    expect(json.images[1]!.boxes).toEqual([]);
    expect(json.images[2]!.error).toBe('Decode failed');

    const rows = buildBatchCsvRows({ items, filter: DEFAULT_FILTER, hidden: new Set([1]) });
    expect(rows).toEqual([['=evil.png', 'circle', 0.9123, 10.13, 20.5, 110.25, 120.75, 800, 600]]);
    const csv = toCsv(BATCH_CSV_HEADERS, rows);
    expect(csv).toContain('filename,class,confidence,x1,y1,x2,y2,image_width,image_height\r\n');
    // Spreadsheet formula neutralization prefixes leading '=' with a single quote.
    expect(csv).toContain("'=evil.png,circle,0.9123,10.13,20.5,110.25,120.75,800,600\r\n");
  });

  it('refuses export when no images have completed, and downloads JSON/CSV when done', () => {
    const createUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:export');
    const revokeUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    const idleInput = {
      items: [makeItem()],
      modelId: 'm_shapes_s',
      modelName: 'shapes-s',
      runOn: 'server' as const,
      filter: DEFAULT_FILTER,
      hidden: new Set<number>(),
      classes: ['circle'],
    };
    expect(exportBatchJson(idleInput)).toBe(false);
    expect(exportBatchCsv(idleInput)).toBe(false);
    expect(click).not.toHaveBeenCalled();

    const doneInput = {
      ...idleInput,
      items: [
        makeItem({
          status: 'done',
          predKey: batchPredKey('m_shapes_s', 'server'),
          pred: makePred([{ class: 0, name: 'circle', score: 0.88, x1: 1, y1: 2, x2: 30, y2: 40 }]),
        }),
      ],
    };
    vi.useFakeTimers();
    try {
      expect(exportBatchJson(doneInput)).toBe(true);
      expect(toast.items).toHaveLength(1);
      expect(toast.items[0]?.key).toBe('playground-export');
      expect(toast.items[0]?.title).toBe('Batch JSON exported');
      // Repeated Playground exports replace the prior export toast in place instead of stacking over stage controls.
      expect(exportBatchCsv(doneInput)).toBe(true);
      expect(toast.items).toHaveLength(1);
      expect(toast.items[0]?.key).toBe('playground-export');
      expect(toast.items[0]?.title).toBe('Batch CSV exported');
      // A failed download replaces the prior success toast instead of stacking, and a subsequent retry replaces the error.
      createUrl.mockImplementationOnce(() => {
        throw new Error('Blocked');
      });
      expect(exportBatchCsv(doneInput)).toBe(false);
      expect(toast.items).toHaveLength(1);
      expect(toast.items[0]?.key).toBe('playground-export');
      expect(toast.items[0]?.tone).toBe('error');
      expect(toast.items[0]?.title).toBe('Download failed');

      expect(exportBatchCsv(doneInput)).toBe(true);
      expect(toast.items).toHaveLength(1);
      expect(toast.items[0]?.key).toBe('playground-export');
      expect(toast.items[0]?.tone).toBe('success');
      expect(toast.items[0]?.title).toBe('Batch CSV exported');

      expect(createUrl).toHaveBeenCalledTimes(4);
      expect(click).toHaveBeenCalledTimes(3);
      vi.advanceTimersByTime(1000);
      expect(revokeUrl).toHaveBeenCalledTimes(3);
    } finally {
      vi.useRealTimers();
    }
  });

  it('computes live QA filter counts, filters batch items by detections/empty/class, and builds per-image summary CSV + filtered JSON', () => {
    const key = batchPredKey('m_shapes_s', 'server');
    const items: BatchItem[] = [
      makeItem({
        id: 'b1',
        name: 'both.png',
        status: 'done',
        predKey: key,
        pred: makePred([
          { class: 0, name: 'circle', score: 0.91, x1: 0, y1: 0, x2: 30, y2: 30 },
          { class: 1, name: 'square', score: 0.65, x1: 40, y1: 10, x2: 80, y2: 50 },
          { class: 0, name: 'circle', score: 0.52, x1: 90, y1: 10, x2: 120, y2: 40 },
        ]),
      }),
      makeItem({
        id: 'b2',
        name: 'empty.png',
        status: 'done',
        predKey: key,
        pred: makePred([]),
      }),
      makeItem({
        id: 'b3',
        name: 'only-square.png',
        status: 'done',
        predKey: key,
        pred: makePred([{ class: 1, name: 'square', score: 0.84, x1: 10, y1: 10, x2: 50, y2: 50 }]),
      }),
      makeItem({
        id: 'b4',
        name: '=failed.png',
        status: 'error',
        error: 'HTTP 500',
      }),
    ];

    const counts = batchQaCounts(items, DEFAULT_FILTER, new Set(), ['circle', 'square']);
    expect(counts.all).toBe(4);
    expect(counts.withDetections).toBe(2);
    expect(counts.empty).toBe(1);
    expect(counts.byClass).toEqual([
      { classId: 1, name: 'square', imageCount: 2, boxCount: 2 },
      { classId: 0, name: 'circle', imageCount: 1, boxCount: 2 },
    ]);

    expect(filterBatchItems(items, { kind: 'with-detections' }, DEFAULT_FILTER, new Set()).map((e) => e.index)).toEqual([0, 2]);
    expect(filterBatchItems(items, { kind: 'empty' }, DEFAULT_FILTER, new Set()).map((e) => e.index)).toEqual([1]);
    expect(filterBatchItems(items, { kind: 'class', classId: 0 }, DEFAULT_FILTER, new Set()).map((e) => e.index)).toEqual([0]);
    expect(matchesBatchQaFilter(items[3]!, { kind: 'empty' }, DEFAULT_FILTER, new Set())).toBe(false);

    // Per-image summary CSV includes empty and failed rows with timing and formula neutralization.
    const summaryRows = buildBatchSummaryCsvRows({ items, filter: DEFAULT_FILTER, hidden: new Set() });
    expect(summaryRows).toEqual([
      ['both.png', 'done', 3, 'circle:2; square:1', 0.91, 10, 640, 480, ''],
      ['empty.png', 'done', 0, '', '', 10, 640, 480, ''],
      ['only-square.png', 'done', 1, 'square:1', 0.84, 10, 640, 480, ''],
      ['=failed.png', 'error', 0, '', '', '', 640, 480, 'HTTP 500'],
    ]);
    const csv = toCsv(BATCH_SUMMARY_CSV_HEADERS, summaryRows);
    expect(csv).toContain('filename,status,detections,classes,top_confidence,total_ms,image_width,image_height,error\r\n');
    expect(csv).toContain("'=failed.png,error,0,,,,640,480,HTTP 500\r\n");

    // Filtered JSON includes qa_filter metadata.
    const json = buildBatchJson({
      items: [items[0]!],
      modelId: 'm_shapes_s',
      modelName: 'shapes-s',
      runOn: 'server',
      filter: DEFAULT_FILTER,
      hidden: new Set(),
      classes: ['circle', 'square'],
      qaFilter: { kind: 'class', classId: 0 },
    });
    expect(json.qa_filter).toEqual({ kind: 'class', class: 0, name: 'circle' });

    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:summary');
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    expect(
      exportBatchSummaryCsv({
        items,
        modelId: 'm_shapes_s',
        modelName: 'shapes-s',
        runOn: 'server',
        filter: DEFAULT_FILTER,
        hidden: new Set(),
        classes: ['circle', 'square'],
      }),
    ).toBe(true);
    expect(click).toHaveBeenCalledTimes(1);
    expect(toast.items[0]?.title).toBe('Batch summary CSV exported');
  });
});
