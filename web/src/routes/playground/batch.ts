// Pure state types, threshold filtering, summary stats, and JSON/CSV export helpers for Playground multi-image batch mode.
import type { Prediction } from '$lib/api/types';
import { filterDetections } from '$lib/detect/nms';
import type { InferenceMeta } from '$lib/inference/browser';
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

export type BatchQaFilter =
  | { kind: 'all' }
  | { kind: 'with-detections' }
  | { kind: 'empty' }
  | { kind: 'class'; classId: number };

export interface BatchQaClassCount {
  classId: number;
  name: string;
  imageCount: number;
  boxCount: number;
}

export interface BatchQaCounts {
  all: number;
  withDetections: number;
  empty: number;
  byClass: BatchQaClassCount[];
}

export interface BatchEntry {
  item: BatchItem;
  index: number;
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

export const BATCH_SUMMARY_CSV_HEADERS = [
  'filename',
  'status',
  'detections',
  'classes',
  'top_confidence',
  'total_ms',
  'image_width',
  'image_height',
  'error',
] as const;

/** Cache key binding a batch item's raw prediction to the model, backend, and inference mode that produced it. */
export function batchPredKey(modelId: string, runOn: RunOn, inferenceKey = 'standard'): string {
  return inferenceKey === 'standard' ? `${modelId}|${runOn}` : `${modelId}|${runOn}|${inferenceKey}`;
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

/** Whether a batch item matches the active QA filmstrip filter under live thresholds and hidden classes. */
export function matchesBatchQaFilter(
  item: BatchItem,
  qa: BatchQaFilter,
  filter: BatchFilter,
  hidden: ReadonlySet<number>,
): boolean {
  if (qa.kind === 'all') return true;
  if (item.status !== 'done' || !item.pred) return false;
  const boxes = filterBatchBoxes(item.pred, filter, hidden);
  if (qa.kind === 'with-detections') return boxes.length > 0;
  if (qa.kind === 'empty') return boxes.length === 0;
  return boxes.some((b) => b.class === qa.classId);
}

/** Returns batch items matching `qa` while preserving each item's original index in `items`. */
export function filterBatchItems(
  items: readonly BatchItem[],
  qa: BatchQaFilter,
  filter: BatchFilter,
  hidden: ReadonlySet<number>,
): BatchEntry[] {
  const out: BatchEntry[] = [];
  for (let index = 0; index < items.length; index++) {
    const item = items[index]!;
    if (matchesBatchQaFilter(item, qa, filter, hidden)) out.push({ item, index });
  }
  return out;
}

/** Counts batch images matching each QA filter bucket (all, with detections, empty, and per detected class). */
export function batchQaCounts(
  items: readonly BatchItem[],
  filter: BatchFilter,
  hidden: ReadonlySet<number>,
  classes: readonly string[],
): BatchQaCounts {
  let withDetections = 0;
  let empty = 0;
  const classMap = new Map<number, BatchQaClassCount>();

  for (const item of items) {
    if (item.status !== 'done' || !item.pred) continue;
    const boxes = filterBatchBoxes(item.pred, filter, hidden);
    if (boxes.length === 0) {
      empty++;
      continue;
    }
    withDetections++;
    const seenInImage = new Set<number>();
    for (const b of boxes) {
      let entry = classMap.get(b.class);
      if (!entry) {
        entry = {
          classId: b.class,
          name: classes[b.class] ?? b.name ?? `class ${b.class}`,
          imageCount: 0,
          boxCount: 0,
        };
        classMap.set(b.class, entry);
      }
      entry.boxCount++;
      if (!seenInImage.has(b.class)) {
        seenInImage.add(b.class);
        entry.imageCount++;
      }
    }
  }

  const byClass = [...classMap.values()].sort(
    (a, b) => b.imageCount - a.imageCount || b.boxCount - a.boxCount || a.classId - b.classId,
  );
  return { all: items.length, withDetections, empty, byClass };
}

export interface BatchExportInput {
  items: readonly BatchItem[];
  modelId: string | null;
  modelName: string | null;
  runOn: RunOn;
  inference?: InferenceMeta;
  filter: BatchFilter;
  hidden: ReadonlySet<number>;
  classes: readonly string[];
  qaFilter?: BatchQaFilter;
}

function qaFilterMeta(qa: BatchQaFilter | undefined, classes: readonly string[]) {
  if (!qa || qa.kind === 'all') return undefined;
  if (qa.kind === 'class') {
    return { kind: 'class' as const, class: qa.classId, name: classes[qa.classId] ?? `class ${qa.classId}` };
  }
  return { kind: qa.kind };
}

function qaFilterSlug(qa: BatchQaFilter | undefined, classes: readonly string[]): string {
  if (!qa || qa.kind === 'all') return '';
  if (qa.kind === 'with-detections') return '-detected';
  if (qa.kind === 'empty') return '-empty';
  return `-${classes[qa.classId] ?? `class-${qa.classId}`}`;
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
  const qa = qaFilterMeta(input.qaFilter, input.classes);

  return {
    model_id: input.modelId,
    model_name: input.modelName,
    backend: input.runOn,
    ...(input.inference ? { inference: input.inference } : {}),
    filter: hiddenClasses?.length ? { ...baseFilter, hidden_classes: hiddenClasses } : baseFilter,
    ...(qa ? { qa_filter: qa } : {}),
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

/** Builds RFC-4180 per-image summary CSV rows (including empty images, class counts, top confidence, and timing). */
export function buildBatchSummaryCsvRows(input: Pick<BatchExportInput, 'items' | 'filter' | 'hidden'>): CsvCell[][] {
  const rows: CsvCell[][] = [];
  for (const item of input.items) {
    const width = item.pred?.image.width || item.width;
    const height = item.pred?.image.height || item.height;
    if (item.status !== 'done' || !item.pred) {
      rows.push([item.name, item.status, 0, '', '', '', width, height, item.error ?? '']);
      continue;
    }
    const boxes = filterBatchBoxes(item.pred, input.filter, input.hidden);
    const counts = new Map<string, number>();
    let topScore = -1;
    for (const b of boxes) {
      counts.set(b.name, (counts.get(b.name) ?? 0) + 1);
      if (b.score > topScore) topScore = b.score;
    }
    const classBreakdown = [...counts.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map(([name, n]) => `${name}:${n}`)
      .join('; ');
    rows.push([
      item.name,
      item.status,
      boxes.length,
      classBreakdown,
      topScore >= 0 ? Number(topScore.toFixed(4)) : '',
      Number(item.pred.timing.total_ms.toFixed(2)),
      width,
      height,
      '',
    ]);
  }
  return rows;
}

function batchFilenameBase(
  modelName: string | null,
  count: number,
  qa?: BatchQaFilter,
  classes: readonly string[] = [],
  suffix = '',
): string {
  return `playground-batch-${modelName || 'detections'}${qaFilterSlug(qa, classes)}${suffix}-${count}img`;
}

/** Exports batch detections as pretty-printed JSON and reports toast feedback. */
export function exportBatchJson(input: BatchExportInput): boolean {
  const summary = summarizeBatch(input.items, input.filter, input.hidden);
  if (summary.done === 0) return false;
  const payload = buildBatchJson(input);
  const truncated = payload.summary.truncated_images ?? 0;
  const ok = exportTableJson(
    batchFilenameBase(input.modelName, input.items.length, input.qaFilter, input.classes),
    payload,
    'playground-export',
  );
  if (ok) {
    toast.success(
      'Batch JSON exported',
      `${summary.totalDetections} ${summary.totalDetections === 1 ? 'detection' : 'detections'} across ${summary.done} ${summary.done === 1 ? 'image' : 'images'}${truncated ? ` (${truncated} capped at the candidate limit)` : ''}.`,
      undefined,
      { key: 'playground-export' },
    );
  }
  return ok;
}

/** Exports batch detections as an RFC-4180 CSV table (with spreadsheet formula neutralization) and reports toast feedback. */
export function exportBatchCsv(input: BatchExportInput): boolean {
  const summary = summarizeBatch(input.items, input.filter, input.hidden);
  if (summary.done === 0) return false;
  const rows = buildBatchCsvRows(input);
  const ok = exportTableCsv(
    batchFilenameBase(input.modelName, input.items.length, input.qaFilter, input.classes),
    BATCH_CSV_HEADERS,
    rows,
    'playground-export',
  );
  if (ok) {
    toast.success(
      'Batch CSV exported',
      `${rows.length} ${rows.length === 1 ? 'detection' : 'detections'} across ${summary.done} ${summary.done === 1 ? 'image' : 'images'}${summary.truncated ? ` (${summary.truncated} capped at the candidate limit)` : ''}.`,
      undefined,
      { key: 'playground-export' },
    );
  }
  return ok;
}

/** Exports one-row-per-image batch QA summary (status, detection count, classes, top confidence, timing) as CSV. */
export function exportBatchSummaryCsv(input: BatchExportInput): boolean {
  const summary = summarizeBatch(input.items, input.filter, input.hidden);
  if (summary.done === 0) return false;
  const rows = buildBatchSummaryCsvRows(input);
  const ok = exportTableCsv(
    batchFilenameBase(input.modelName, input.items.length, input.qaFilter, input.classes, '-summary'),
    BATCH_SUMMARY_CSV_HEADERS,
    rows,
    'playground-export',
  );
  if (ok) {
    toast.success(
      'Batch summary CSV exported',
      `${rows.length} ${rows.length === 1 ? 'image' : 'images'} (${summary.done} completed, ${summary.totalDetections} ${summary.totalDetections === 1 ? 'detection' : 'detections'})${summary.truncated ? ` (${summary.truncated} capped at the candidate limit)` : ''}.`,
      undefined,
      { key: 'playground-export' },
    );
  }
  return ok;
}
