// View-state and DOM helpers extracted from Playground.tsx: detection filtering/grouping, source metadata,
// smaller-model fallback lookup, keyboard row navigation, row scrolling, and JSON export formatting.
import type { Model, Prediction } from '$lib/api/types';
import { BACKEND_NAME, DEFAULTS } from './predict';
import type { Source } from './source';

export type Det = Prediction['boxes'][number];

/** Model sizes, smallest first: a smaller model of the same classes may fit where this one does not. */
export const ARCH_ORDER = ['folo-n', 'folo-s', 'folo-m', 'folo-l', 'folo-x'];

/** Finds a smaller model with the same class list when the current model runs out of GPU memory. */
export function smallerModelOf(model: Model | null, ranked: readonly Model[]): Model | null {
  if (!model) return null;
  const rank = (x: Model) => ARCH_ORDER.indexOf(x.arch);
  const wantRank = rank(model);
  if (wantRank <= 0) return null;
  const classesKey = model.classes.join('|');
  return ranked.find((x) => {
    const r = rank(x);
    return x.id !== model.id && r >= 0 && r < wantRank && x.classes.join('|') === classesKey;
  }) ?? null;
}

/** Label for where the latest prediction ran ("server" or "browser · WebGPU"). */
export function whereLabel(pred: Prediction | null): string {
  return pred?.backend ? `browser · ${BACKEND_NAME[pred.backend]}` : 'server';
}

/** Label for the device that ran the latest prediction. */
export function deviceLabel(pred: Prediction | null, adapter?: string | null): string {
  if (!pred) return '';
  return pred.backend ? `${BACKEND_NAME[pred.backend]}${pred.backend === 'webgpu' && adapter ? ` · ${adapter}` : ''}` : (pred.device ?? '');
}

/** Pixel dimensions of the active image, preferring the prediction payload when available. */
export function sourceDims(pred: Prediction | null, source: Source | null): { width: number; height: number } | null {
  if (pred) return pred.image;
  if (source?.kind === 'file') return { width: source.width, height: source.height };
  if (source?.kind === 'dataset') return { width: source.img.width, height: source.img.height };
  return null;
}

/** Full display name of the active image source (for tooltips and alt text). */
export function sourceFullName(source: Source | null): string {
  if (!source) return '';
  return source.kind === 'file' ? source.name : `${source.ds.name} / ${source.img.file}`;
}

/** Short filename of the active image source (for the viewer toolbar). */
export function sourceShortName(source: Source | null): string {
  if (!source) return '';
  return source.kind === 'file' ? source.name : (source.img.file.split('/').pop() ?? source.img.file);
}

/** Set of visible class indices when any class is hidden, or null when all classes are shown. */
export function visibleClasses(classes: readonly string[], hidden: ReadonlySet<number>): Set<number> | null {
  return hidden.size ? new Set(classes.map((_, i) => i).filter((i) => !hidden.has(i))) : null;
}

/** Filtered detection rows paired with their index in `kept`. */
export function visibleRows(kept: readonly Det[], hidden: ReadonlySet<number>): { b: Det; i: number }[] {
  return kept.map((b, i) => ({ b, i })).filter((r) => !hidden.has(r.b.class));
}

/** Per-class detection counts sorted descending by count, then ascending by class index. */
export function classCounts(kept: readonly Det[]): [number, number][] {
  const m = new Map<number, number>();
  for (const b of kept) m.set(b.class, (m.get(b.class) ?? 0) + 1);
  return [...m.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
}

/** Index of `det` in `kept` when visible, or null when absent or hidden. */
export function detectionIndex(det: Det | null, kept: readonly Det[], hidden: ReadonlySet<number>): number | null {
  const i = det && !hidden.has(det.class) ? kept.indexOf(det) : -1;
  return i < 0 ? null : i;
}

/** Target row index for Arrow/Home/End navigation inside the detections table, or null for other keys. */
export function nextRowIndex(key: string, current: number, length: number): number | null {
  if (length <= 0) return null;
  const to = key === 'ArrowDown' ? current + 1 : key === 'ArrowUp' ? current - 1 : key === 'Home' ? 0 : key === 'End' ? length - 1 : null;
  if (to === null) return null;
  return Math.max(0, Math.min(length - 1, to));
}

/** Brings a picked detection row into view inside either the desktop rail or the narrow table scroller. */
export function scrollRowIntoView(row: HTMLTableRowElement | null, narrow: boolean): void {
  if (!row) return;
  if (!narrow) {
    row.scrollIntoView({ block: 'nearest' });
    return;
  }
  const box = row.closest<HTMLElement>('.scroll');
  if (!box) return;
  const head = box.querySelector('thead')?.offsetHeight ?? 0;
  if (row.offsetTop - head < box.scrollTop) box.scrollTop = row.offsetTop - head;
  else if (row.offsetTop + row.offsetHeight > box.scrollTop + box.clientHeight) box.scrollTop = row.offsetTop + row.offsetHeight - box.clientHeight;
}

/** Whether active threshold sliders are at their default values (ignoring IoU for end-to-end o2o models). */
export function areDefaultThresholds(conf: number, iou: number, maxDet: number, o2o: boolean): boolean {
  return conf === DEFAULTS.conf && (o2o || iou === DEFAULTS.iou) && maxDet === DEFAULTS.maxDet;
}

/** Summary of visible detection confidence scores for the live distribution badge. */
export interface ConfidenceSummary {
  count: number;
  min: number;
  median: number;
  max: number;
  mean: number;
  /** High confidence detections (score >= 0.75). */
  high: number;
  /** Medium confidence detections (0.50 <= score < 0.75). */
  mid: number;
  /** Low confidence detections (score < 0.50). */
  low: number;
}

/** Computes live confidence distribution stats for currently visible detections, or null when empty. */
export function confidenceSummary(rows: readonly { b: Pick<Det, 'score'> }[]): ConfidenceSummary | null {
  const count = rows.length;
  if (count === 0) return null;
  const scores = rows.map((r) => r.b.score).sort((a, b) => a - b);
  let sum = 0;
  let high = 0;
  let mid = 0;
  let low = 0;
  for (const s of scores) {
    sum += s;
    if (s >= 0.75) high++;
    else if (s >= 0.5) mid++;
    else low++;
  }
  const midIdx = Math.floor(count / 2);
  const median = count % 2 === 1 ? scores[midIdx]! : (scores[midIdx - 1]! + scores[midIdx]!) / 2;
  return {
    count,
    min: scores[0]!,
    median,
    max: scores[count - 1]!,
    mean: sum / count,
    high,
    mid,
    low,
  };
}

/** Structured payload shown in the Playground's JSON tab and copied via Copy JSON (matches visible detections). */
export function predictionJson(
  pred: Prediction | null,
  hasResult: boolean,
  kept: readonly Det[],
  filter: { conf: number; iou: number; maxDet: number; o2o: boolean; hiddenClasses?: readonly string[] },
) {
  if (!pred || !hasResult) return null;
  const baseFilter = filter.o2o ? { conf: filter.conf, max_det: filter.maxDet } : { conf: filter.conf, iou: filter.iou, max_det: filter.maxDet };
  return {
    model_id: pred.model_id,
    image: pred.image,
    boxes: kept.map((b) => ({ class: b.class, name: b.name, score: b.score, x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2 })),
    timing: pred.timing,
    device: pred.device,
    filter: filter.hiddenClasses?.length ? { ...baseFilter, hidden_classes: [...filter.hiddenClasses] } : baseFilter,
  };
}
