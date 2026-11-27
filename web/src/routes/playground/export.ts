// Playground annotated PNG and structured JSON export helpers for both Single and Compare A/B modes.
// Draws boxes, ground truth, and label chips onto an offscreen canvas using Studio's design-system class palette,
// then triggers a clean browser download with object-URL revocation and toast feedback.
import { classChipText, classColor, classHalo } from '$lib/design/tokens';
import { ms } from '$lib/format';
import { downloadBlob as downloadBlobShared, downloadText, exportTableCsv, exportTableJson, safeFilename, toJson } from '$lib/tableExport';
import { toast } from '$lib/ui/toast';
import type { ViewBox } from '$lib/detect/DetectionView';
import type { CompareViewMode } from './compare';
import { imageBlob, type Source } from './source';
import {
  buildTrackExportCsvRows,
  formatBoxChipText,
  TRACK_CSV_HEADERS,
  type TrackExportPayload,
  type TrackObservation,
  type TrackTrail,
} from './tracking';

export interface AnnotatedLayer {
  boxes: readonly ViewBox[];
  truth?: readonly ViewBox[];
  /** Optional short per-track motion trails drawn underneath tracked boxes. */
  trails?: readonly TrackTrail[];
  classes: readonly string[];
  labels: boolean;
  scores: boolean;
  /** Optional model badge header text rendered in Compare exports (e.g. "A · shapes-s · 6 det · 14 ms"). */
  caption?: string;
}

export interface ExportSinglePngOptions {
  source: Source;
  src: string | null;
  width: number;
  height: number;
  layer: AnnotatedLayer;
  filenameBase: string;
}

export interface ExportComparePngOptions {
  source: Source;
  src: string | null;
  width: number;
  height: number;
  view: CompareViewMode;
  /** Split position percentage (0..100), used when `view === 'split'`. */
  splitPct: number;
  layerA: AnnotatedLayer;
  layerB: AnnotatedLayer;
  filenameBase: string;
}

/** Builds a canvas font declaration dynamically so typography stays proportional to the exported image resolution. */
function makeCanvasFont(weight: number, sizePx: number): string {
  return `${weight} ${Math.max(12, Math.round(sizePx))}px ui-sans-serif, system-ui, -apple-system, sans-serif`;
}

/** Formats a safe lowercase filename with any supported export extension. */
export function exportFilename(base: string, ext: 'png' | 'json'): string {
  const jsonName = safeFilename(base, 'json');
  return ext === 'json' ? jsonName : `${jsonName.slice(0, -5)}.png`;
}

/** Triggers a browser download for a binary Blob and schedules safe object-URL revocation. */
export function downloadBlob(blob: Blob, filename: string): boolean {
  return downloadBlobShared(blob, filename, 'playground-export', 'Export failed');
}

/** Downloads a structured detections or A/B comparison JSON file. */
export function exportDetectionsJson(filenameBase: string, payload: unknown): boolean {
  if (!payload) return false;
  let content: string;
  try {
    content = toJson(payload);
  } catch (err) {
    toast.error(err instanceof Error ? err : new Error('Could not serialize export JSON.'), 'Export failed', { key: 'playground-export' });
    return false;
  }
  return downloadText(content, exportFilename(filenameBase, 'json'), 'application/json;charset=utf-8', 'playground-export');
}

/** Downloads frame-indexed ByteTrack trajectory observations and summary metadata as pretty-printed JSON. */
export function exportTracksJson(filenameBase: string, payload: TrackExportPayload): boolean {
  if (!payload.observations.length) return false;
  const ok = exportTableJson(filenameBase, payload, 'playground-export');
  if (ok) {
    const n = payload.observations.length;
    const u = new Set(payload.observations.map((o) => o.track_id)).size;
    toast.success(
      'Tracks JSON exported',
      `${u} unique ${u === 1 ? 'track' : 'tracks'} · ${n} ${n === 1 ? 'observation' : 'observations'}.`,
      undefined,
      { key: 'playground-export' },
    );
  }
  return ok;
}

/** Downloads frame-indexed ByteTrack trajectory observations as formula-safe RFC-4180 CSV. */
export function exportTracksCsv(filenameBase: string, observations: readonly TrackObservation[]): boolean {
  if (!observations.length) return false;
  const rows = buildTrackExportCsvRows(observations);
  const ok = exportTableCsv(filenameBase, TRACK_CSV_HEADERS, rows, 'playground-export');
  if (ok) {
    const u = new Set(observations.map((o) => o.track_id)).size;
    toast.success(
      'Tracks CSV exported',
      `${u} unique ${u === 1 ? 'track' : 'tracks'} · ${rows.length} ${rows.length === 1 ? 'row' : 'rows'}.`,
      undefined,
      { key: 'playground-export' },
    );
  }
  return ok;
}

/** Draws ground-truth boxes, predicted boxes, and readable label chips onto `ctx` within `(ox, oy, width, height)`. */
export function drawAnnotations(
  ctx: CanvasRenderingContext2D,
  layer: AnnotatedLayer,
  bounds: { ox: number; oy: number; width: number; height: number },
): void {
  const { ox, oy, width, height } = bounds;
  const scale = Math.max(1, Math.min(width, height) / 640);
  const strokeW = Math.max(2, Math.round(2 * scale));
  const haloW = strokeW + Math.max(2, Math.round(2 * scale));
  const fontPx = Math.max(12, Math.round(12 * scale));
  const padX = Math.max(5, Math.round(5 * scale));
  const chipH = Math.round(fontPx * 1.45);
  const labelFont = makeCanvasFont(600, fontPx);

  ctx.save();
  ctx.beginPath();
  ctx.rect(ox, oy, width, height);
  ctx.clip();

  // Ground truth (dashed) underneath predictions.
  if (layer.truth?.length) {
    ctx.save();
    ctx.setLineDash([Math.max(4, Math.round(5 * scale)), Math.max(3, Math.round(4 * scale))]);
    for (const b of layer.truth) {
      const x = ox + b.x1;
      const y = oy + b.y1;
      const bw = Math.max(1, b.x2 - b.x1);
      const bh = Math.max(1, b.y2 - b.y1);
      ctx.lineWidth = Math.max(2, Math.round(1.5 * scale));
      ctx.strokeStyle = classColor(b.class);
      ctx.strokeRect(x, y, bw, bh);
    }
    ctx.restore();
  }

  // Short per-track motion trails underneath boxes (fading tail-to-head with dark halo for contrast).
  if (layer.trails?.length) {
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (const trail of layer.trails) {
      const pts = trail.points;
      if (pts.length < 2) continue;
      const color = classColor(trail.class);
      for (let i = 1; i < pts.length; i++) {
        const a = pts[i - 1]!;
        const b = pts[i]!;
        const t = i / (pts.length - 1);
        ctx.globalAlpha = 0.2 + 0.65 * t;
        ctx.beginPath();
        ctx.moveTo(ox + a.x, oy + a.y);
        ctx.lineTo(ox + b.x, oy + b.y);
        ctx.lineWidth = haloW;
        ctx.strokeStyle = classHalo;
        ctx.stroke();
        ctx.lineWidth = Math.max(1.5, strokeW * 0.9);
        ctx.strokeStyle = color;
        ctx.stroke();
      }
      const head = pts[pts.length - 1]!;
      ctx.globalAlpha = 0.95;
      ctx.beginPath();
      ctx.arc(ox + head.x, oy + head.y, Math.max(2.5, strokeW * 1.25), 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }
    ctx.restore();
  }

  // Predictions: halo + class-colored stroke.
  for (const b of layer.boxes) {
    const x = ox + b.x1;
    const y = oy + b.y1;
    const bw = Math.max(1, b.x2 - b.x1);
    const bh = Math.max(1, b.y2 - b.y1);
    ctx.lineWidth = haloW;
    ctx.strokeStyle = classHalo;
    ctx.strokeRect(x, y, bw, bh);
    ctx.lineWidth = strokeW;
    ctx.strokeStyle = classColor(b.class);
    ctx.strokeRect(x, y, bw, bh);
  }

  // Label chips drawn after all strokes so labels stay crisp and unobstructed.
  if (layer.labels) {
    ctx.font = labelFont;
    ctx.textBaseline = 'middle';
    const placed: { x: number; y: number; w: number; h: number }[] = [];
    for (const b of layer.boxes) {
      const text = formatBoxChipText(b, layer.classes, layer.scores);
      const tw = ctx.measureText(text).width;
      const cw = Math.ceil(tw + padX * 2);
      const cx = Math.max(ox + 2, Math.min(ox + width - cw - 2, ox + b.x1));
      let cy = oy + b.y1 - chipH - 2;
      if (cy < oy + 2) cy = Math.min(oy + height - chipH - 2, oy + b.y1 + 2);

      // Nudge downward slightly if overlapping an earlier chip.
      for (let tries = 0; tries < 4; tries++) {
        const hit = placed.find((r) => cx < r.x + r.w + 2 && cx + cw + 2 > r.x && cy < r.y + r.h + 2 && cy + chipH + 2 > r.y);
        if (!hit) break;
        const nextY = hit.y + hit.h + 2;
        if (nextY + chipH <= oy + height - 2) cy = nextY;
        else break;
      }
      placed.push({ x: cx, y: cy, w: cw, h: chipH });

      ctx.fillStyle = classHalo;
      ctx.fillRect(cx - 1, cy - 1, cw + 2, chipH + 2);
      ctx.fillStyle = classColor(b.class);
      ctx.fillRect(cx, cy, cw, chipH);
      ctx.fillStyle = classChipText;
      ctx.fillText(text, cx + padX, cy + chipH / 2 + 0.5);
    }
  }
  ctx.restore();
}

/** Draws a high-contrast caption pill inside a pane for Compare exports, clamped to `maxWidth` to prevent collisions. */
function drawBadge(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, align: 'left' | 'right', scale: number, maxWidth: number): void {
  const fontPx = Math.max(12, Math.round(13 * scale));
  const padX = Math.max(8, Math.round(10 * scale));
  const h = Math.round(fontPx * 1.8);
  ctx.save();
  ctx.font = makeCanvasFont(600, fontPx);
  ctx.textBaseline = 'middle';
  const availText = Math.max(16, maxWidth - padX * 2);
  let label = text;
  if (ctx.measureText(label).width > availText) {
    while (label.length > 1 && ctx.measureText(`${label}…`).width > availText) label = label.slice(0, -1);
    label = `${label}…`;
  }
  const w = Math.min(maxWidth, Math.ceil(ctx.measureText(label).width + padX * 2));
  const left = align === 'right' ? x - w : x;
  ctx.fillStyle = 'rgba(10, 12, 16, 0.86)';
  ctx.fillRect(left, y, w, h);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
  ctx.lineWidth = Math.max(1, Math.round(scale));
  ctx.strokeRect(left, y, w, h);
  ctx.fillStyle = '#EDEFF2';
  ctx.fillText(label, left + padX, y + h / 2 + 0.5, availText);
  ctx.restore();
}

/** Loads a drawable image source safely without tainting the canvas. */
async function loadDrawable(source: Source, src: string | null): Promise<{ img: CanvasImageSource; width: number; height: number; close: () => void }> {
  const ctrl = new AbortController();
  if (typeof createImageBitmap === 'function') {
    try {
      const blob = await imageBlob(source, ctrl.signal);
      const bmp = await createImageBitmap(blob);
      return { img: bmp, width: bmp.width, height: bmp.height, close: () => bmp.close() };
    } catch {
      // Fall back to HTMLImageElement below (e.g. data: URLs in mock or tests without ImageBitmap).
    }
  }
  if (!src) throw new Error('Image is not loaded yet.');
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const el = new Image();
    el.crossOrigin = 'anonymous';
    el.onload = () => resolve(el);
    el.onerror = () => reject(new Error('Could not decode image for PNG export.'));
    el.src = src;
  });
  return { img, width: img.naturalWidth || img.width, height: img.naturalHeight || img.height, close: () => {} };
}

function canvasToPng(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('PNG encoding failed.'))), 'image/png');
  });
}

/** Formats a concise caption badge for Compare PNG exports (never reports 0 dets when a model has no result). */
export function compareBadgeText(side: 'A' | 'B', modelName: string, count: number | null, totalMs: number | null): string {
  if (count === null) return `${side} · ${modelName} · unavailable`;
  const det = `${count} ${count === 1 ? 'det' : 'dets'}`;
  return totalMs !== null ? `${side} · ${modelName} · ${det} · ${ms(totalMs)}` : `${side} · ${modelName} · ${det}`;
}

/** Renders the active single-model image + boxes onto an offscreen canvas and downloads an annotated PNG. */
export async function exportSinglePng(opts: ExportSinglePngOptions): Promise<boolean> {
  const loaded = await loadDrawable(opts.source, opts.src);
  try {
    const w = opts.width || loaded.width;
    const h = opts.height || loaded.height;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas is unavailable in this browser.');
    ctx.drawImage(loaded.img, 0, 0, w, h);
    drawAnnotations(ctx, opts.layer, { ox: 0, oy: 0, width: w, height: h });
    const blob = await canvasToPng(canvas);
    return downloadBlob(blob, exportFilename(opts.filenameBase, 'png'));
  } finally {
    loaded.close();
  }
}

/** Renders a Compare A/B composite (side-by-side or split-overlay at `splitPct`) onto a canvas and downloads PNG. */
export async function exportComparePng(opts: ExportComparePngOptions): Promise<boolean> {
  const loaded = await loadDrawable(opts.source, opts.src);
  try {
    const w = opts.width || loaded.width;
    const h = opts.height || loaded.height;
    const scale = Math.max(1, Math.min(w, h) / 640);
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas is unavailable in this browser.');

    if (opts.view === 'side-by-side') {
      const gap = Math.max(4, Math.round(4 * scale));
      const headH = Math.max(28, Math.round(32 * scale));
      canvas.width = w * 2 + gap;
      canvas.height = h + headH;
      ctx.fillStyle = '#0E1013';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Pane A
      ctx.drawImage(loaded.img, 0, headH, w, h);
      drawAnnotations(ctx, opts.layerA, { ox: 0, oy: headH, width: w, height: h });
      // Pane B
      ctx.drawImage(loaded.img, w + gap, headH, w, h);
      drawAnnotations(ctx, opts.layerB, { ox: w + gap, oy: headH, width: w, height: h });

      // Crisp header captions
      ctx.font = makeCanvasFont(600, Math.max(12, Math.round(13 * scale)));
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#EDEFF2';
      const pad = Math.max(10, Math.round(12 * scale));
      if (opts.layerA.caption) ctx.fillText(opts.layerA.caption, pad, headH / 2 + 0.5, w - pad * 2);
      if (opts.layerB.caption) ctx.fillText(opts.layerB.caption, w + gap + pad, headH / 2 + 0.5, w - pad * 2);
    } else {
      // Split-overlay composite at exact splitPct
      const pct = Math.max(0, Math.min(100, opts.splitPct));
      const cutX = Math.round((w * pct) / 100);
      canvas.width = w;
      canvas.height = h;

      // Right half (Model B)
      ctx.save();
      ctx.beginPath();
      ctx.rect(cutX, 0, w - cutX, h);
      ctx.clip();
      ctx.drawImage(loaded.img, 0, 0, w, h);
      drawAnnotations(ctx, opts.layerB, { ox: 0, oy: 0, width: w, height: h });
      ctx.restore();

      // Left half (Model A)
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, cutX, h);
      ctx.clip();
      ctx.drawImage(loaded.img, 0, 0, w, h);
      drawAnnotations(ctx, opts.layerA, { ox: 0, oy: 0, width: w, height: h });
      ctx.restore();

      // Divider line
      const lineW = Math.max(2, Math.round(2 * scale));
      ctx.fillStyle = '#C5F82A';
      ctx.fillRect(Math.max(0, cutX - Math.floor(lineW / 2)), 0, lineW, h);

      const inset = Math.max(4, Math.round(10 * scale));
      const availHalf = Math.floor((w - inset * 3) / 2);
      if (availHalf >= 64) {
        if (opts.layerA.caption) drawBadge(ctx, opts.layerA.caption, inset, inset, 'left', scale, availHalf);
        if (opts.layerB.caption) drawBadge(ctx, opts.layerB.caption, w - inset, inset, 'right', scale, availHalf);
      } else if (w >= 96) {
        const fullW = Math.max(32, w - inset * 2);
        const pillH = Math.round(Math.max(12, Math.round(13 * scale)) * 1.8);
        if (opts.layerA.caption) drawBadge(ctx, opts.layerA.caption, inset, inset, 'left', scale, fullW);
        if (opts.layerB.caption) drawBadge(ctx, opts.layerB.caption, inset, inset + pillH + 4, 'left', scale, fullW);
      }
    }

    const blob = await canvasToPng(canvas);
    return downloadBlob(blob, exportFilename(opts.filenameBase, 'png'));
  } finally {
    loaded.close();
  }
}
