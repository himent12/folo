// @vitest-environment happy-dom
// Unit tests for Benchmark Entry Error-Analysis CSV / JSON export (FP / FN / localization failure rows).
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Bench, BenchEntry, BenchEntryImage } from '$lib/api/types';
import { pendingBlobRevokeCount, revokePendingBlobUrls, toCsv } from '$lib/tableExport';
import { toast } from '$lib/ui/toast';
import { filterAndSortImages } from './errorAnalysis';
import {
  buildBenchErrorExportCsvRows,
  buildBenchErrorExportJson,
  collectBenchErrorRows,
  ERROR_EXPORT_CSV_HEADERS,
  exportBenchErrorsCsv,
  exportBenchErrorsJson,
} from './errorExport';

const CLASSES = ['bolt', '=nut', 'washer'];

const BENCH: Pick<Bench, 'id' | 'name' | 'classes'> = {
  id: 'bn_1',
  name: 'Fasteners QA',
  classes: CLASSES,
};

const ENTRY: Pick<BenchEntry, 'id' | 'name' | 'scored'> = {
  id: 'be_1',
  name: 'folo-s',
  scored: true,
};

const IMAGES: BenchEntryImage[] = [
  {
    image: {
      id: 'im_1',
      file: 'img_01.jpg',
      width: 200,
      height: 200,
      boxes: [
        { class: 0, x1: 0, y1: 0, x2: 20, y2: 20 }, // matched TP
        { class: 1, x1: 50, y1: 50, x2: 70, y2: 70 }, // class mismatch FN
        { class: 2, x1: 100, y1: 100, x2: 120, y2: 120 }, // localization FN
      ],
    },
    detections: [
      { class: 0, score: 0.95, x1: 0, y1: 0, x2: 20, y2: 20 }, // TP
      { class: 0, score: 0.81, x1: 50, y1: 50, x2: 70, y2: 70 }, // class mismatch FP
      { class: 2, score: 0.64, x1: 100, y1: 100, x2: 110, y2: 110 }, // localization FP
      { class: 1, score: 0.42, x1: 160, y1: 160, x2: 180, y2: 180 }, // background FP
    ],
    missed: 2,
    wrong: 3,
  },
  {
    image: {
      id: 'im_clean',
      file: 'clean.jpg',
      width: 200,
      height: 200,
      boxes: [{ class: 0, x1: 10, y1: 10, x2: 30, y2: 30 }],
    },
    detections: [{ class: 0, score: 0.99, x1: 10, y1: 10, x2: 30, y2: 30 }],
    missed: 0,
    wrong: 0,
  },
];

afterEach(() => {
  revokePendingBlobUrls();
  for (const t of toast.items) toast.dismiss(t.id);
  vi.restoreAllMocks();
});

describe('bench/entry/errorExport', () => {
  it('collects only FP/FN error boxes matching the active filter, excludes TPs, and handles unscored entries', () => {
    const analyzedAll = filterAndSortImages(IMAGES, 'all', 'worst', CLASSES);
    const allErrors = collectBenchErrorRows(analyzedAll, 'all', CLASSES, true);
    expect(allErrors).toHaveLength(5);
    expect(allErrors.every((r) => r.error_type === 'FP' || r.error_type === 'FN')).toBe(true);
    expect(allErrors.map((r) => [r.box_key, r.error_type, r.error_category])).toEqual([
      ['d1', 'FP', 'class-mismatch'],
      ['d2', 'FP', 'localization'],
      ['d3', 'FP', 'background'],
      ['t1', 'FN', 'class-mismatch'],
      ['t2', 'FN', 'localization'],
    ]);
    expect(allErrors[0]).toMatchObject({
      image_id: 'im_1',
      image_file: 'img_01.jpg',
      condition: 'unchanged',
      severity: null,
      class_name: 'bolt',
      other_class_name: '=nut',
      confidence: 0.81,
      iou: 1,
      x1: 50,
      y1: 50,
      x2: 70,
      y2: 70,
    });

    // FN filter keeps only missed GT boxes; FP filter keeps only wrong detection boxes.
    const fnOnly = collectBenchErrorRows(filterAndSortImages(IMAGES, 'fn', 'fn', CLASSES), 'fn', CLASSES, true);
    expect(fnOnly.map((r) => r.error_type)).toEqual(['FN', 'FN']);
    const fpOnly = collectBenchErrorRows(filterAndSortImages(IMAGES, 'fp', 'fp', CLASSES), 'fp', CLASSES, true);
    expect(fpOnly.map((r) => r.error_type)).toEqual(['FP', 'FP', 'FP']);

    // Unscored contestant omits confidence cleanly.
    const unscored = collectBenchErrorRows(analyzedAll, 'all', CLASSES, false);
    expect(unscored.every((r) => r.confidence === null)).toBe(true);
  });

  it('builds formula-safe RFC-4180 CSV and structured JSON downloads with toast feedback and Blob cleanup', async () => {
    const analyzed = filterAndSortImages(IMAGES, 'errors', 'worst', CLASSES);
    const input = {
      bench: BENCH,
      entry: ENTRY,
      analyzed,
      filter: 'errors' as const,
      sort: 'worst' as const,
      conf: 0.5,
      loadedImages: 2,
      totalImages: 10,
    };

    const json = buildBenchErrorExportJson(input);
    expect(json.summary).toEqual({
      total_errors: 5,
      fp: 3,
      fn: 2,
      by_category: {
        duplicate: 0,
        'class-mismatch': 2,
        localization: 2,
        background: 1,
        missed: 0,
      },
    });

    const csv = toCsv(ERROR_EXPORT_CSV_HEADERS, buildBenchErrorExportCsvRows(json.errors));
    expect(csv).toContain('image_id,image_file,image_width,image_height,condition,severity,box_key,error_type,error_category');
    // Formula neutralization protects spreadsheet users on `=nut`.
    expect(csv).toContain("'=nut");

    const blobs: Blob[] = [];
    vi.spyOn(URL, 'createObjectURL').mockImplementation((b) => {
      blobs.push(b as Blob);
      return `blob:err-${blobs.length}`;
    });
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const downloads: string[] = [];
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      downloads.push(this.download);
    });

    expect(exportBenchErrorsCsv(input)).toBe(true);
    expect(downloads[0]).toBe('bench-fasteners-qa-folo-s-errors.csv');
    expect(toast.items[0]?.title).toBe('Error analysis CSV exported');
    expect(toast.items[0]?.detail).toContain('5 errors across 1 image (2 of 10 loaded)');

    // Under filter='all' (which includes clean.jpg), the toast still counts only the 1 image with exported errors.
    const analyzedAll = filterAndSortImages(IMAGES, 'all', 'worst', CLASSES);
    expect(exportBenchErrorsJson({ ...input, filter: 'all', analyzed: analyzedAll })).toBe(true);
    expect(toast.items[0]?.title).toBe('Error analysis JSON exported');
    expect(toast.items[0]?.detail).toContain('5 errors across 1 image (2 of 10 loaded)');
    expect(click).toHaveBeenCalledTimes(2);
    expect(pendingBlobRevokeCount()).toBe(2);
    revokePendingBlobUrls();

    // Clean-only gallery returns false without downloading.
    const cleanOnly = filterAndSortImages([IMAGES[1]!], 'all', 'worst', CLASSES);
    expect(exportBenchErrorsCsv({ ...input, analyzed: cleanOnly })).toBe(false);
    expect(exportBenchErrorsJson({ ...input, analyzed: cleanOnly })).toBe(false);
  });
});
