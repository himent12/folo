// Where an image and its boxes sit in a frame of another shape (a benchmark's gallery tile, the large view, the
// condition preview): the whole image, scaled to fit and centred on the frame's well (`object-fit: contain`), and every
// box mapped through the same scale and offset, so an overlay stays on its object whatever the image's shape.

import type { Box } from '$lib/api/types';

/** The scale of an image in a frame and where its top-left corner falls, in the frame's pixels. */
export interface Fit {
  s: number;
  ox: number;
  oy: number;
}

/** An image of `width × height` contained in a `cw × ch` frame; null until both have a size. */
export function containFit(cw: number, ch: number, width: number, height: number): Fit | null {
  if (!(cw > 0 && ch > 0 && width > 0 && height > 0)) return null;
  const s = Math.min(cw / width, ch / height);
  return { s, ox: (cw - width * s) / 2, oy: (ch - height * s) / 2 };
}

/** A box of the image (pixels) in the frame's pixels. */
export function placeBox(f: Fit, b: Pick<Box, 'x1' | 'y1' | 'x2' | 'y2'>): { left: number; top: number; width: number; height: number } {
  return { left: f.ox + b.x1 * f.s, top: f.oy + b.y1 * f.s, width: (b.x2 - b.x1) * f.s, height: (b.y2 - b.y1) * f.s };
}

/**
 * A placed box pulled in so its line, drawn `edge` px outside it (its stroke and halo, ./MatchView), stays inside the
 * `cw × ch` frame, which clips: a box along the image's edge, a whole-image box above all, drew its line where the
 * frame cut it off and showed nothing (issue #382). A box `edge` px or more inside the frame is unchanged.
 */
export function insideFrame(r: { left: number; top: number; width: number; height: number }, cw: number, ch: number, edge: number): { left: number; top: number; width: number; height: number } {
  const left = Math.max(r.left, edge);
  const top = Math.max(r.top, edge);
  const right = Math.min(r.left + r.width, cw - edge);
  const bottom = Math.min(r.top + r.height, ch - edge);
  return { left, top, width: Math.max(0, right - left), height: Math.max(0, bottom - top) };
}
