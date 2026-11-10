// Which detections of an image are right and which true boxes they missed, matched as docs/BENCH.md's Definitions
// (Matching) say: per class, detections in descending confidence are matched one by one to the still unmatched true box
// of that class with the highest IoU, if that IoU is at least 0.5. The server sends an entry's images with their counts
// (`missed`, `wrong`) but not the pairs; the failure gallery draws the pairs, so it matches them here the same way (and
// its tests hold the counts to the server's).

import type { Box } from '../api/types';

/** The IoU a detection needs with a true box of its class to be right. */
export const MATCH_IOU = 0.5;

/** Intersection over union of two pixel xyxy boxes (0 when either has no area). */
export function iou(a: Pick<Box, 'x1' | 'y1' | 'x2' | 'y2'>, b: Pick<Box, 'x1' | 'y1' | 'x2' | 'y2'>): number {
  const w = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const h = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = w * h;
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return union > 0 ? inter / union : 0;
}

export interface MatchedDetection<D> {
  det: D;
  /** Its index in the detections given. */
  index: number;
  /** The true box it matched (an index into the truth), or null: a wrong detection. */
  truth: number | null;
  /** The IoU with that true box; 0 for a wrong one. */
  iou: number;
}

export interface ImageMatch<D> {
  /** In the order given (the server sends them highest confidence first). */
  detections: MatchedDetection<D>[];
  /** Per true box, the detection that found it (an index into the detections), or null: missed. */
  found: (number | null)[];
  missed: number;
  wrong: number;
}

/** Matches an image's detections to its truth (docs/BENCH.md, Matching). */
export function matchImage<D extends Box & { score?: number }>(truth: readonly Box[], detections: readonly D[]): ImageMatch<D> {
  const order = detections.map((_, i) => i).sort((a, b) => (detections[b]!.score ?? 1) - (detections[a]!.score ?? 1) || a - b);
  const found: (number | null)[] = truth.map(() => null);
  const out: MatchedDetection<D>[] = detections.map((det, index) => ({ det, index, truth: null, iou: 0 }));
  for (const i of order) {
    const d = detections[i]!;
    let best = -1;
    let bestIou = MATCH_IOU;
    truth.forEach((t, k) => {
      if (t.class !== d.class || found[k] !== null) return;
      const v = iou(t, d);
      if (v >= bestIou) {
        // Ties keep the first true box: the same rule every time.
        if (v > bestIou || best < 0) best = k;
        bestIou = v;
      }
    });
    if (best >= 0) {
      found[best] = i;
      out[i] = { det: d, index: i, truth: best, iou: bestIou };
    }
  }
  const missed = found.filter((f) => f === null).length;
  return { detections: out, found, missed, wrong: out.filter((m) => m.truth === null).length };
}
