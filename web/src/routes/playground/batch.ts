// Pure state types, threshold filtering, summary stats, and JSON/CSV export helpers for Playground multi-image batch mode.
import type { Prediction } from '$lib/api/types';
import { filterDetections } from '$lib/detect/nms';
import { exportTableCsv, exportTableJson, type CsvCell } from '$lib/tableExport';
import { toast } from '$lib/ui/toast';
import { isPredictionCutAt } from './allCandidates';
import type { RunOn } from './predict';
import type { Source } from './source';
import type { Det } from './viewState';

export type BatchItemStatus = 'idle' | 'running' | 'done' | 'error';

export interface BatchItem {
  id: string;
  file: Blob;
  name: string;
  url: string;
  width: number;
  height: number;
  /** Raw loose-threshold prediction cached for `predKey` (re-filtered in the browser on threshold changes). */
  pred: Prediction | null;
  /** `${modelId}|${runOn}` that produced `pred`, or null when unrun / invalidated. */
  predKey: string | null;
  status: BatchItemStatus;
  error: string | null;
}

export interface BatchFilter {
  conf: number;
  iou: number;
  maxDet: number;
  o2o: boolean;
}

export interface BatchSummary {
  total: number;
  done: number;
  running: number;
  failed: number;
  pending: number;
  totalDetections: number;
  truncated: number;
}

export const BATCH_CSV_HEADERS = [
  'filename',
  'class',
  'confidence',
  'x1',
  'y1',
  'x2',
  'y2',
  'image_width',
  'image_height',
] as const;

/** Cache key binding a batch item's raw prediction to the model and backend that produced it. */
export function batchPredKey(modelId: string, runOn: RunOn): string {
  return `${modelId}|${runOn}`;
}

/** Builds a Playground `Source` pointing at a batch item whose object URL is owned by the batch session. */
export function batchItemSource(item: BatchItem): Extract<Source, { kind: 'file' }> {
  return {
    kind: 'file',
    file: item.file,
    name: item.name,
    url: item.url,
    width: item.width,
    height: item.height,
    sharedUrl: true,
  };
}

/** Filters a batch item's cached raw prediction using Playground's live thresholds and hidden classes. */
export function filterBatchBoxes(
  pred: Prediction | null,
  filter: BatchFilter,
  hidden: ReadonlySet<number>,
): Det[] {
  if (!pred) return [];
  const kept = filterDetections(pred.boxes, {
    conf: filter.conf,
    iou: filter.o2o ? 1 : filter.iou,
    maxDet: filter.maxDet,
  });
  return hidden.size ? kept.filter((b) => !hidden.has(b.class)) : kept;
}

/** Counts batch progress and total visible detections across all completed items. */
export function summarizeBatch(
  items: readonly BatchItem[],
  filter: BatchFilter,
  hidden: ReadonlySet<number>,
): BatchSummary {
  let done = 0;
  let running = 0;
  let failed = 0;
  let totalDetections = 0;
  let truncated = 0;
  for (const item of items) {
    if (item.status === 'done' && item.pred) {
      done++;
      totalDetections += filterBatchBoxes(item.pred, filter, hidden).length;
      if (isPredictionCutAt(item.pred, filter.conf)) truncated++;
    } else if (item.status === 'running') {
      running++;
    } else if (item.status === 'error') {
      failed++;
    }
  }
  const total = items.length;
  return {
    total,
    done,
    running,
    failed,
    pending: Math.max(0, total - done - running),
    totalDetections,
    truncated,
  };
}

export interface BatchExportInput {
  items: readonly BatchItem[];
  modelId: string | null;
  modelName: string | null;
  runOn: RunOn;
  filter: BatchFilter;
  hidden: ReadonlySet<number>;
  classes: readonly string[];
}

/** Builds the structured JSON payload for the current batch under active thresholds and class filters. */
export function buildBatchJson(input: BatchExportInput) {
  const summary = summarizeBatch(input.items, input.filter, input.hidden);
  const hiddenClasses = input.hidden.size
    ? [...input.hidden].sort((a, b) => a - b).map((c) => input.classes[c] ?? `class ${c}`)
    : undefined;
  const baseFilter = input.filter.o2o
    ? { conf: input.filter.conf, max_det: input.filter.maxDet }
    : { conf: input.filter.conf, iou: input.filter.iou, max_det: input.filter.maxDet };

  return {
    model_id: input.modelId,
    model_name: input.modelName,
    backend: input.runOn,
    filter: hiddenClasses?.length ? { ...baseFilter, hidden_classes: hiddenClasses } : baseFilter,
    summary: {
      total_images: summary.total,
      completed_images: summary.done,
      failed_images: summary.failed,
      total_detections: summary.totalDetections,
      ...(summary.truncated > 0 ? { truncated_images: summary.truncated } : {}),
    },
    images: input.items.map((item) => {
      const boxes = item.status === 'done' ? filterBatchBoxes(item.pred, input.filter, input.hidden) : [];
      const truncated = item.status === 'done' && isPredictionCutAt(item.pred, input.filter.conf);
      return {
        filename: item.name,
        width: item.pred?.image.width ?? item.width,
        height: item.pred?.image.height ?? item.height,
        status: item.status,
        ...(truncated ? { truncated: true } : {}),
        ...(item.error ? { error: item.error } : {}),
        timing: item.pred?.timing ?? null,
        boxes: boxes.map((b) => ({
          class: b.class,
          name: b.name,
          confidence: b.score,
          x1: b.x1,
          y1: b.y1,
          x2: b.x2,
          y2: b.y2,
        })),
      };
    }),
  };
}

/** Builds RFC-4180 CSV rows (one row per visible detection across completed batch images). */
export function buildBatchCsvRows(input: Pick<BatchExportInput, 'items' | 'filter' | 'hidden'>): CsvCell[][] {
  const rows: CsvCell[][] = [];
  for (const item of input.items) {
    if (item.status !== 'done' || !item.pred) continue;
    const width = item.pred.image.width || item.width;
    const height = item.pred.image.height || item.height;
    const boxes = filterBatchBoxes(item.pred, input.filter, input.hidden);
    for (const b of boxes) {
      rows.push([
        item.name,
        b.name,
        Number(b.score.toFixed(4)),
        Number(b.x1.toFixed(2)),
        Number(b.y1.toFixed(2)),
        Number(b.x2.toFixed(2)),
        Number(b.y2.toFixed(2)),
        width,
        height,
      ]);
    }
  }
  return rows;
}

function batchFilenameBase(modelName: string | null, count: number): string {
  return `playground-batch-${modelName || 'detections'}-${count}img`;
}

/** Exports batch detections as pretty-printed JSON and reports toast feedback. */
export function exportBatchJson(input: BatchExportInput): boolean {
  const summary = summarizeBatch(input.items, input.filter, input.hidden);
  if (summary.done === 0) return false;
  const payload = buildBatchJson(input);
  const truncated = payload.summary.truncated_images ?? 0;
  const ok = exportTableJson(batchFilenameBase(input.modelName, input.items.length), payload);
  if (ok) {
    toast.success(
      'Batch JSON exported',
      `${summary.totalDetections} ${summary.totalDetections === 1 ? 'detection' : 'detections'} across ${summary.done} ${summary.done === 1 ? 'image' : 'images'}${truncated ? ` (${truncated} capped at 1,000 candidates)` : ''}.`,
    );
  }
  return ok;
}

/** Exports batch detections as an RFC-4180 CSV table (with spreadsheet formula neutralization) and reports toast feedback. */
export function exportBatchCsv(input: BatchExportInput): boolean {
  const summary = summarizeBatch(input.items, input.filter, input.hidden);
  if (summary.done === 0) return false;
  const rows = buildBatchCsvRows(input);
  const ok = exportTableCsv(batchFilenameBase(input.modelName, input.items.length), BATCH_CSV_HEADERS, rows);
  if (ok) {
    toast.success(
      'Batch CSV exported',
      `${rows.length} ${rows.length === 1 ? 'detection' : 'detections'} across ${summary.done} ${summary.done === 1 ? 'image' : 'images'}${summary.truncated ? ` (${summary.truncated} capped at 1,000 candidates)` : ''}.`,
    );
  }
  return ok;
}
