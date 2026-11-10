// Boxes the mock draws: a model's predictions for an image with known ground truth, invented ones for an upload it
// knows nothing about, and the size of an uploaded image.

import { ApiError } from '../client';
import type { Box } from '../types';
import { range, rng } from './rng';

/** A prediction before it gets its class name. */
export type Scored = Box & { score: number };

export function round(x: number, d: number) {
  const k = 10 ** d;
  return Math.round(x * k) / k;
}

export async function imageSize(file: Blob): Promise<{ width: number; height: number }> {
  try {
    const bmp = await createImageBitmap(file);
    const out = { width: bmp.width, height: bmp.height };
    bmp.close();
    return out;
  } catch {
    throw new ApiError(400, 'bad_image', 'That file is not an image FOLO can decode.', 'Use a JPEG, PNG or WebP.');
  }
}

/** Predictions for an image with known ground truth: misses, jitter, confusions and low-score false positives. */
export function predictBoxes(truth: Box[], W: number, H: number, quality: number, seed: number, classes: number): Scored[] {
  const r = rng(seed);
  const q = Math.max(0, Math.min(1, quality));
  const out: Scored[] = [];
  for (const t of truth) {
    if (r() > 0.25 + 0.75 * Math.min(1, q * 1.25)) continue;
    const w = t.x2 - t.x1;
    const h = t.y2 - t.y1;
    const j = (1 - q) * 0.22;
    const nx1 = t.x1 + w * j * (r() - 0.5) * 2;
    const ny1 = t.y1 + h * j * (r() - 0.5) * 2;
    const nx2 = t.x2 + w * j * (r() - 0.5) * 2;
    const ny2 = t.y2 + h * j * (r() - 0.5) * 2;
    const cls = r() < (1 - q) * 0.25 ? Math.floor(r() * classes) : t.class;
    out.push(clampBox({ class: cls, x1: nx1, y1: ny1, x2: nx2, y2: ny2, score: Math.min(0.99, 0.3 + 0.62 * q + (r() - 0.5) * 0.3) }, W, H));
  }
  const fps = Math.round((1 - q) * 4 * r() + r() * 1.5);
  for (let i = 0; i < fps; i++) {
    const w = range(r, W * 0.04, W * 0.2);
    const h = w * range(r, 0.6, 1.5);
    const x1 = range(r, 0, W - w);
    const y1 = range(r, 0, H - h);
    out.push(clampBox({ class: Math.floor(r() * classes), x1, y1, x2: x1 + w, y2: y1 + h, score: range(r, 0.04, 0.42) }, W, H));
  }
  return out.sort((a, b) => b.score - a.score);
}

/** Adds lower-scored, slightly shifted copies of some boxes (IoU ≈ 0.75–0.9 with the original). */
export function withDuplicates(boxes: Scored[], W: number, H: number, seed: number): Scored[] {
  const r = rng(seed);
  const out = [...boxes];
  for (const b of boxes) {
    if (b.score < 0.3 || r() > 0.55) continue;
    const w = b.x2 - b.x1;
    const h = b.y2 - b.y1;
    const j = () => (r() - 0.5) * 0.08;
    out.push(clampBox({ ...b, x1: b.x1 + w * j(), y1: b.y1 + h * j(), x2: b.x2 + w * j(), y2: b.y2 + h * j(), score: b.score * range(r, 0.45, 0.8) }, W, H));
  }
  return out.sort((a, b) => b.score - a.score);
}

/** Plausible boxes for an uploaded image the mock knows nothing about. */
export function inventBoxes(W: number, H: number, seed: number, classes: number): Scored[] {
  const r = rng(seed);
  const n = 3 + Math.floor(r() * 6);
  const out: Scored[] = [];
  for (let i = 0; i < n; i++) {
    const w = range(r, W * 0.08, W * 0.35);
    const h = range(r, H * 0.1, H * 0.45);
    const x1 = range(r, 0, W - w);
    const y1 = range(r, 0, H - h);
    out.push(clampBox({ class: Math.floor(r() * Math.min(classes, 12)), x1, y1, x2: x1 + w, y2: y1 + h, score: i < 3 ? range(r, 0.55, 0.96) : range(r, 0.05, 0.5) }, W, H));
  }
  return out.sort((a, b) => b.score - a.score);
}

function clampBox<T extends Scored>(b: T, W: number, H: number): T {
  const x1 = Math.max(0, Math.min(W, b.x1));
  const x2 = Math.max(0, Math.min(W, b.x2));
  const y1 = Math.max(0, Math.min(H, b.y1));
  const y2 = Math.max(0, Math.min(H, b.y2));
  return { ...b, x1: round(Math.min(x1, x2), 1), y1: round(Math.min(y1, y2), 1), x2: round(Math.max(x1, x2), 1), y2: round(Math.max(y1, y2), 1), score: round(b.score, 4) };
}
