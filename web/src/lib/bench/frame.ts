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
 * The room to keep free between the image and the frame's edge, which clips, so that every box's marks, drawn up to
 * `reach` px outside it (its line and halo, a wrong box's cross: ./MatchView), show (issue #382: a whole-image box on a
 * thumbnail the image fills drew its line where the frame cut it off). 0 when no box comes that close to the edge in
 * `f`; else the largest reach of those that do. Fitting the image into the frame less that room on every side
 * (`insetFit`) moves every box, the image and the chips together, so none is drawn anywhere but on its object.
 */
export function edgeRoom(f: Fit, cw: number, ch: number, boxes: readonly (Pick<Box, 'x1' | 'y1' | 'x2' | 'y2'> & { reach: number })[]): number {
  let room = 0;
  for (const b of boxes) {
    const r = placeBox(f, b);
    if (Math.min(r.left, r.top, cw - r.left - r.width, ch - r.top - r.height) < b.reach) room = Math.max(room, b.reach);
  }
  return room;
}

/** `containFit` in the `cw × ch` frame less `room` px on every side (null when nothing is left). */
export function insetFit(cw: number, ch: number, width: number, height: number, room: number): Fit | null {
  const f = containFit(cw - 2 * room, ch - 2 * room, width, height);
  return f && { s: f.s, ox: f.ox + room, oy: f.oy + room };
}
