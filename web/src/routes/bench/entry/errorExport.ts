// Export builders (CSV & JSON) for Benchmark Entry Per-Image Error Analysis (FP / FN / localization failure rows).
// Exports only the currently filtered error boxes in the active diagnostic sort order, using Studio's formula-safe
// RFC-4180 CSV and structured JSON helpers so failure cases drop straight into dataset curation & retraining workflows.
import type { Bench, BenchEntry } from '$lib/api/types';
import { exportTableCsv, exportTableJson, type CsvCell } from '$lib/tableExport';
import { toast } from '$lib/ui/toast';
import type { AnalyzedImage, BoxCause, ErrorFilter, ErrorSort } from './errorAnalysis';

export type ErrorCategory = Exclude<BoxCause, 'tp'>;

export interface BenchErrorExportRow {
  image_id: string;
  image_file: string;
  image_width: number;
  image_height: number;
  condition: string;
  severity: number | null;
  box_key: string;
  error_type: 'FP' | 'FN';
  error_category: ErrorCategory;
  class_id: number;
  class_name: string;
  other_class_id: number | null;
  other_class_name: string | null;
  confidence: number | null;
  iou: number | null;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  detail: string;
}

export const ERROR_EXPORT_CSV_HEADERS = [
  'image_id',
  'image_file',
  'image_width',
  'image_height',
  'condition',
  'severity',
  'box_key',
  'error_type',
  'error_category',
  'class_id',
  'class_name',
  'other_class_id',
  'other_class_name',
  'confidence',
  'iou',
  'x1',
  'y1',
  'x2',
  'y2',
  'detail',
] as const;

export interface BenchErrorExportInput {
  bench: Pick<Bench, 'id' | 'name' | 'classes'>;
  entry: Pick<BenchEntry, 'id' | 'name' | 'scored'>;
  analyzed: readonly AnalyzedImage[];
  filter: ErrorFilter;
  sort: ErrorSort;
  conf: number | null;
  loadedImages: number;
  totalImages: number | null;
}

function round(v: number, digits: number): number {
  return Number(v.toFixed(digits));
}

function classNameOf(classes: readonly string[], cls: number): string {
  return classes[cls] ?? `class ${cls}`;
}

/** Collects FP / FN / localization failure rows matching the active gallery filter in the active sorted image order. */
export function collectBenchErrorRows(
  analyzed: readonly AnalyzedImage[],
  filter: ErrorFilter,
  classes: readonly string[],
  scored: boolean,
): BenchErrorExportRow[] {
  const rows: BenchErrorExportRow[] = [];
  for (const { item, diag } of analyzed) {
    for (const b of diag.boxes) {
      if (b.outcome === 'right' || b.cause === 'tp') continue;
      if (filter === 'fn' && b.outcome !== 'missed') continue;
      if (filter === 'fp' && b.outcome !== 'wrong') continue;
      rows.push({
        image_id: item.image.id,
        image_file: item.image.file,
        image_width: item.image.width,
        image_height: item.image.height,
        condition: 'unchanged',
        severity: null,
        box_key: b.key,
        error_type: b.outcome === 'wrong' ? 'FP' : 'FN',
        error_category: b.cause,
        class_id: b.cls,
        class_name: classNameOf(classes, b.cls),
        other_class_id: b.otherCls,
        other_class_name: b.otherCls !== null ? classNameOf(classes, b.otherCls) : null,
        confidence: scored && b.score !== null && Number.isFinite(b.score) ? round(b.score, 4) : null,
        iou: b.iou !== null && Number.isFinite(b.iou) ? round(b.iou, 4) : null,
        x1: Number.isFinite(b.box.x1) ? round(b.box.x1, 2) : 0,
        y1: Number.isFinite(b.box.y1) ? round(b.box.y1, 2) : 0,
        x2: Number.isFinite(b.box.x2) ? round(b.box.x2, 2) : 0,
        y2: Number.isFinite(b.box.y2) ? round(b.box.y2, 2) : 0,
        detail: b.detail,
      });
    }
  }
  return rows;
}

/** Builds the structured JSON error-analysis report for the currently filtered benchmark entry gallery. */
export function buildBenchErrorExportJson(input: BenchErrorExportInput) {
  const errors = collectBenchErrorRows(input.analyzed, input.filter, input.bench.classes, input.entry.scored);
  let fp = 0;
  let fn = 0;
  const byCategory: Record<ErrorCategory, number> = {
    duplicate: 0,
    'class-mismatch': 0,
    localization: 0,
    background: 0,
    missed: 0,
  };
  for (const row of errors) {
    if (row.error_type === 'FP') fp++;
    else fn++;
    byCategory[row.error_category]++;
  }
  return {
    benchmark: { id: input.bench.id, name: input.bench.name },
    entry: { id: input.entry.id, name: input.entry.name, scored: input.entry.scored, conf: input.conf },
    filter: input.filter,
    sort: input.sort,
    loaded_images: input.loadedImages,
    total_images: input.totalImages,
    matching_images: input.analyzed.length,
    summary: {
      total_errors: errors.length,
      fp,
      fn,
      by_category: byCategory,
    },
    errors,
  };
}

/** Builds RFC-4180 CSV rows aligned with `ERROR_EXPORT_CSV_HEADERS`. */
export function buildBenchErrorExportCsvRows(rows: readonly BenchErrorExportRow[]): CsvCell[][] {
  return rows.map((r) => [
    r.image_id,
    r.image_file,
    r.image_width,
    r.image_height,
    r.condition,
    r.severity,
    r.box_key,
    r.error_type,
    r.error_category,
    r.class_id,
    r.class_name,
    r.other_class_id,
    r.other_class_name,
    r.confidence,
    r.iou,
    r.x1,
    r.y1,
    r.x2,
    r.y2,
    r.detail,
  ]);
}

function errorFilenameBase(input: Pick<BenchErrorExportInput, 'bench' | 'entry' | 'filter'>): string {
  const filterTag = input.filter === 'all' || input.filter === 'errors' ? 'errors' : `${input.filter}-errors`;
  return `bench-${input.bench.name}-${input.entry.name}-${filterTag}`;
}

function exportSummaryDetail(
  rows: readonly BenchErrorExportRow[],
  input: Pick<BenchErrorExportInput, 'loadedImages' | 'totalImages'>,
): string {
  const rowCount = rows.length;
  const imgCount = new Set(rows.map((r) => r.image_id)).size;
  const base = `${rowCount} ${rowCount === 1 ? 'error' : 'errors'} across ${imgCount} ${imgCount === 1 ? 'image' : 'images'}`;
  if (input.totalImages !== null && input.loadedImages < input.totalImages) {
    return `${base} (${input.loadedImages} of ${input.totalImages} loaded).`;
  }
  return `${base}.`;
}

/** Downloads filtered FP / FN / localization failure cases as formula-safe RFC-4180 CSV. */
export function exportBenchErrorsCsv(input: BenchErrorExportInput): boolean {
  const rows = collectBenchErrorRows(input.analyzed, input.filter, input.bench.classes, input.entry.scored);
  if (!rows.length) return false;
  const ok = exportTableCsv(
    errorFilenameBase(input),
    ERROR_EXPORT_CSV_HEADERS,
    buildBenchErrorExportCsvRows(rows),
    'bench-error-export',
  );
  if (ok) {
    toast.success('Error analysis CSV exported', exportSummaryDetail(rows, input), undefined, {
      key: 'bench-error-export',
    });
  }
  return ok;
}

/** Downloads filtered FP / FN / localization failure cases as structured JSON. */
export function exportBenchErrorsJson(input: BenchErrorExportInput): boolean {
  const payload = buildBenchErrorExportJson(input);
  if (!payload.errors.length) return false;
  const ok = exportTableJson(errorFilenameBase(input), payload, 'bench-error-export');
  if (ok) {
    toast.success('Error analysis JSON exported', exportSummaryDetail(payload.errors, input), undefined, {
      key: 'bench-error-export',
    });
  }
  return ok;
}
