// @vitest-environment happy-dom
// Dataset Health & Class-Balance Report export builders (CSV & JSON) and formula-safe RFC-4180 serialization.
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Dataset, DatasetStats, IssueCounts } from '$lib/api/types';
import { toCsv } from '$lib/tableExport';
import { toast } from '$lib/ui/toast';
import {
  buildDatasetHealthReportCsvRows,
  buildDatasetHealthReportJson,
  DATASET_REPORT_CSV_HEADERS,
  exportDatasetReportCsv,
  exportDatasetReportJson,
} from './reportExport';

const SAMPLE_DATASET: Dataset = {
  id: 'ds_parts',
  name: 'parts-qa',
  path: 'D:/datasets/parts-qa',
  format: 'yolo',
  classes: ['bolt', '=nut', 'washer'],
  created: '2026-11-10T00:00:00Z',
  splits: { train: 80, val: 20, test: 0 },
};

const SAMPLE_STATS: DatasetStats = {
  boxes: 650,
  empty_images: 15,
  broken_labels: 2,
  class_counts: [600, 50, 0],
  box_size_hist: { edges: [0, 0.05, 1], counts: [100, 550] },
  boxes_per_image_hist: { edges: [0, 1, 4, 8], counts: [15, 60, 25] },
};

const SAMPLE_COUNTS: IssueCounts = {
  error: 2,
  warning: 4,
  info: 15,
  by_kind: {
    broken_label: 2,
    clipped_box: 3,
    duplicate_image: 1,
    empty_image: 15,
  },
};

afterEach(() => {
  for (const t of toast.items) toast.dismiss(t.id);
  vi.restoreAllMocks();
});

describe('dataset health report export', () => {
  it('builds a structured JSON report with splits, sorted classes, health findings, and check issue counts', () => {
    const json = buildDatasetHealthReportJson(SAMPLE_DATASET, SAMPLE_STATS, SAMPLE_COUNTS);
    expect(json.dataset).toMatchObject({ id: 'ds_parts', name: 'parts-qa', format: 'yolo' });
    expect(json.summary).toMatchObject({
      total_images: 100,
      total_boxes: 650,
      boxes_per_image: 6.5,
      empty_images: 15,
      empty_image_pct: 15,
      broken_labels: 2,
      class_count: 3,
      present_classes: 2,
      missing_classes: 1,
      imbalance_ratio: 12,
      tiny_boxes: 100,
    });
    expect(json.splits).toEqual({
      train: 80,
      val: 20,
      test: 0,
      train_pct: 80,
      val_pct: 20,
      test_pct: 0,
    });
    expect(json.classes).toEqual([
      { class_id: 0, name: 'bolt', boxes: 600, share_pct: 92.31, status: 'ok' },
      { class_id: 1, name: '=nut', boxes: 50, share_pct: 7.69, status: 'rare' },
      { class_id: 2, name: 'washer', boxes: 0, share_pct: 0, status: 'missing' },
    ]);
    expect(json.health_findings.some((f) => f.id === 'broken')).toBe(true);
    expect(json.health_findings.some((f) => f.id === 'missing')).toBe(true);
    expect(json.health_findings.some((f) => f.id === 'imbalance')).toBe(true);
    expect(json.check_issues?.by_kind).toEqual([
      { kind: 'broken_label', label: 'Broken label', severity: 'error', count: 2 },
      { kind: 'clipped_box', label: 'Clipped box', severity: 'warning', count: 3 },
      { kind: 'duplicate_image', label: 'Duplicate image', severity: 'warning', count: 1 },
      { kind: 'empty_image', label: 'Empty image', severity: 'info', count: 15 },
    ]);
  });

  it('builds RFC-4180 CSV rows with spreadsheet formula neutralization and triggers CSV/JSON downloads with toast feedback', () => {
    const rows = buildDatasetHealthReportCsvRows(SAMPLE_DATASET, SAMPLE_STATS, SAMPLE_COUNTS);
    expect(rows.some((r) => r[0] === 'split' && r[1] === 'train' && r[2] === 80)).toBe(true);
    expect(rows.some((r) => r[0] === 'class' && r[1] === '=nut' && r[2] === 50)).toBe(true);
    expect(rows.some((r) => r[0] === 'check_issue' && r[1] === 'Clipped box' && r[2] === 3)).toBe(true);

    const csv = toCsv(DATASET_REPORT_CSV_HEADERS, rows);
    expect(csv).toContain('section,name,count,share_pct,severity,detail\r\n');
    expect(csv).toContain("class,'=nut,50,7.69,warning,fewer than 100 boxes\r\n");

    const createUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:ds-report');
    const revokeUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    vi.useFakeTimers();
    try {
      expect(exportDatasetReportCsv(SAMPLE_DATASET, SAMPLE_STATS, SAMPLE_COUNTS)).toBe(true);
      expect(toast.items[0]?.title).toBe('Health report CSV exported');
      expect(exportDatasetReportJson(SAMPLE_DATASET, SAMPLE_STATS, SAMPLE_COUNTS)).toBe(true);
      expect(toast.items).toHaveLength(1);
      expect(toast.items[0]?.title).toBe('Health report JSON exported');
      expect(createUrl).toHaveBeenCalledTimes(2);
      expect(click).toHaveBeenCalledTimes(2);
      vi.advanceTimersByTime(1000);
      expect(revokeUrl).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });
});
