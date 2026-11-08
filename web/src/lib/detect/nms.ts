// Client-side post-processing for the Playground: the server is asked once with a very low confidence, then the
// confidence, IoU and max-detections controls re-filter its boxes here instantly, without another round trip.
// Mirrors the server's class-aware greedy NMS (crates/folo-vision), so what you see matches `folo predict`.

export interface NmsBox {
  class: number;
  score: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** Intersection over union of two pixel xyxy boxes (0 when either has no area). */
export function iou(a: NmsBox, b: NmsBox): number {
  const iw = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
  const ih = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * Class-aware greedy non-maximum suppression: visit boxes by descending score and keep one unless it overlaps an
 * already kept box of the same class by more than `iouThreshold`. Returns kept boxes, highest score first.
 * Ties keep input order, so the result is deterministic.
 */
export function nms<T extends NmsBox>(boxes: readonly T[], iouThreshold: number, maxDet = Infinity): T[] {
  const order = boxes.map((_, i) => i).sort((i, j) => boxes[j]!.score - boxes[i]!.score || i - j);
  const kept: T[] = [];
  const byClass = new Map<number, T[]>();
  for (const i of order) {
    if (kept.length >= maxDet) break;
    const b = boxes[i]!;
    const same = byClass.get(b.class);
    if (same?.some((k) => iou(k, b) > iouThreshold)) continue;
    kept.push(b);
    if (same) same.push(b);
    else byClass.set(b.class, [b]);
  }
  return kept;
}

export interface FilterOptions {
  /** Minimum score (inclusive). */
  conf: number;
  /** NMS IoU threshold. */
  iou: number;
  maxDet: number;
}

/** The Playground's pipeline: confidence filter, then NMS, then the max-detections cap. */
export function filterDetections<T extends NmsBox>(boxes: readonly T[], o: FilterOptions): T[] {
  return nms(
    boxes.filter((b) => b.score >= o.conf),
    o.iou,
    Math.max(0, Math.floor(o.maxDet)),
  );
}
