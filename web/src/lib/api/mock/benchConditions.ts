// The geometry of the mock's conditions, apart from their pixels (./benchRender): where `distance` moves the true boxes
// and where `occlusion` hides parts of them. Pure and seeded, so the image the renderer draws and the truth the API
// returns for it always agree.

import type { Box } from '../types';
import { hash, range, rng } from './rng';

/** How much `distance` shrinks the scene towards its centre at a severity: 0.88 at 1, 0.4 at 5. */
export const distanceScale = (severity: number) => 1 - 0.12 * Math.max(1, Math.min(5, severity));

/** The true boxes under a condition: `distance` moves them with the image, every other condition keeps them. */
export function conditionBoxes(boxes: readonly Box[], width: number, height: number, condition: string | undefined, severity: number | undefined): Box[] {
  if (condition !== 'distance' || !severity) return boxes.map((b) => ({ ...b }));
  const k = distanceScale(severity);
  const cx = width / 2;
  const cy = height / 2;
  const r1 = (v: number) => Math.round(v * 10) / 10;
  return boxes.map((b) => ({ class: b.class, x1: r1(cx + (b.x1 - cx) * k), y1: r1(cy + (b.y1 - cy) * k), x2: r1(cx + (b.x2 - cx) * k), y2: r1(cy + (b.y2 - cy) * k) }));
}

/** A patch drawn over part of a box (`occlusion`): a rectangle in image pixels and whether it is leafy or a plain shape. */
export interface Patch {
  x: number;
  y: number;
  w: number;
  h: number;
  round: boolean;
}

/** The patches `occlusion` draws at a severity: over each box, a share of it that grows from about 12% to 60%. */
export function occlusionPatches(boxes: readonly Box[], severity: number, seed: string): Patch[] {
  const share = 0.12 * Math.max(1, Math.min(5, severity));
  const r = rng(hash(`occlusion:${seed}`));
  return boxes.map((b) => {
    const w = b.x2 - b.x1;
    const h = b.y2 - b.y1;
    // A patch of `share` of the box's area, cut in from one side (which side is the seed's).
    const side = Math.floor(r() * 4);
    const along = range(r, 0.85, 1.15);
    if (side < 2) {
      const pw = Math.min(w, w * share * along);
      return { x: side === 0 ? b.x1 - w * 0.04 : b.x2 - pw + w * 0.04, y: b.y1 - h * 0.06, w: pw, h: h * 1.12, round: r() < 0.5 };
    }
    const ph = Math.min(h, h * share * along);
    return { x: b.x1 - w * 0.06, y: side === 2 ? b.y1 - h * 0.04 : b.y2 - ph + h * 0.04, w: w * 1.12, h: ph, round: r() < 0.5 };
  });
}
