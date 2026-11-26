// Drawing and correcting boxes on one image (the labeling editor). Everything here is in image pixels, as the API's
// boxes are: the component maps a pointer into the image (`toImage`), asks what a press grabs (`grab`), and draws the
// box `dragTo` gives back while the pointer moves; `settle` decides what the gesture leaves. Every box stays inside the
// image, and a box under 2 px on either side is not a box. A press that moves the pointer less than `CLICK_SLOP` screen
// px is a click, whatever the image's size: it selects, and changes nothing.

import type { Box } from '$lib/api/types';

/** Boxes smaller than this on either side (image px) are dropped. */
export const MIN_SIDE = 2;
/** How far from a handle (screen px) a press still takes it; a fingertip gets more. */
export const HANDLE_REACH = 8;
export const HANDLE_REACH_TOUCH = 16;
/** A press whose pointer stays within this many screen px of where it went down is a click, not a drag. */
export const CLICK_SLOP = 4;

/** Whether the pointer went from screen point `a` to `b` without leaving the click's slop. */
export const isClick = (a: readonly [number, number], b: readonly [number, number]) => Math.max(Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1])) < CLICK_SLOP;

/** How the image sits on screen: a screen point is `o + image point · s`. */
export interface Placement {
  s: number;
  ox: number;
  oy: number;
}

/** The image `width × height` scaled to fit a `cw × ch` view and centred in it. */
export function fit(width: number, height: number, cw: number, ch: number): Placement {
  if (!Number.isFinite(width) || !Number.isFinite(height) || !Number.isFinite(cw) || !Number.isFinite(ch) || width <= 0 || height <= 0 || cw <= 0 || ch <= 0) {
    return { s: 0, ox: 0, oy: 0 };
  }
  const s = Math.min(cw / width, ch / height);
  return { s, ox: (cw - width * s) / 2, oy: (ch - height * s) / 2 };
}

export function toImage(p: Placement, x: number, y: number): [number, number] {
  if (!Number.isFinite(p.s) || p.s <= 0) return [0, 0];
  return [(x - p.ox) / p.s, (y - p.oy) / p.s];
}

export function toScreen(p: Placement, x: number, y: number): [number, number] {
  return [p.ox + x * p.s, p.oy + y * p.s];
}

/** The edges a handle moves: n / s the top / bottom, w / e the left / right edge, corners two of them. */
export type Handle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w';
export const HANDLES: readonly Handle[] = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];

/** Where a handle sits on a box. */
export function handleAt(b: Box, h: Handle): [number, number] {
  const x = h.includes('w') ? b.x1 : h.includes('e') ? b.x2 : (b.x1 + b.x2) / 2;
  const y = h.includes('n') ? b.y1 : h.includes('s') ? b.y2 : (b.y1 + b.y2) / 2;
  return [x, y];
}

/** What a press takes: a handle of the selected box, a box to move, or empty image to draw a new box on. */
export type Grip = { kind: 'resize'; index: number; handle: Handle } | { kind: 'move'; index: number } | { kind: 'draw' };

/**
 * What a press at image point (x, y) takes. The selected box's handles come first (within `reach` image px), then the
 * smallest box under the point (so a box inside another can be taken), else drawing.
 */
export function grab(boxes: readonly Box[], selected: number | null, x: number, y: number, reach: number): Grip {
  const sel = selected === null ? undefined : boxes[selected];
  if (sel && selected !== null) {
    let best: { handle: Handle; d: number } | null = null;
    for (const handle of HANDLES) {
      const [hx, hy] = handleAt(sel, handle);
      const d = Math.max(Math.abs(hx - x), Math.abs(hy - y));
      if (d <= reach && (!best || d < best.d)) best = { handle, d };
    }
    if (best) return { kind: 'resize', index: selected, handle: best.handle };
  }
  let hit = -1;
  let area = Infinity;
  boxes.forEach((b, i) => {
    const a = (b.x2 - b.x1) * (b.y2 - b.y1);
    if (x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2 && a < area) [hit, area] = [i, a];
  });
  return hit >= 0 ? { kind: 'move', index: hit } : { kind: 'draw' };
}

/** `b` with its corners in order (x1 ≤ x2, y1 ≤ y2) and inside a `width × height` image. */
export function clampBox(b: Box, width: number, height: number): Box {
  const cx = (v: number) => Math.max(0, Math.min(width, v));
  const cy = (v: number) => Math.max(0, Math.min(height, v));
  return { class: b.class, x1: cx(Math.min(b.x1, b.x2)), y1: cy(Math.min(b.y1, b.y2)), x2: cx(Math.max(b.x1, b.x2)), y2: cy(Math.max(b.y1, b.y2)) };
}

export const tooSmall = (b: Box) => b.x2 - b.x1 < MIN_SIDE || b.y2 - b.y1 < MIN_SIDE;

/** A gesture in progress: what it took, where it started (image px), and the box as it was (none when drawing). */
export interface Drag {
  grip: Grip;
  from: [number, number];
  start: Box | null;
  /** The class a drawn box takes. */
  cls: number;
}

/**
 * The box a drag shows with the pointer at image point (x, y): moved (kept whole inside the image, sliding along an
 * edge), resized by its handle (the box may flip over), or drawn from the press point.
 */
export function dragTo(d: Drag, x: number, y: number, width: number, height: number): Box {
  const [fx, fy] = d.from;
  if (d.grip.kind === 'draw' || !d.start) return clampBox({ class: d.cls, x1: fx, y1: fy, x2: x, y2: y }, width, height);
  const b = d.start;
  if (d.grip.kind === 'move') {
    const c = clampBox(b, width, height);
    const w = c.x2 - c.x1;
    const h = c.y2 - c.y1;
    const x1 = Math.max(0, Math.min(width - w, c.x1 + x - fx));
    const y1 = Math.max(0, Math.min(height - h, c.y1 + y - fy));
    return { class: b.class, x1, y1, x2: x1 + w, y2: y1 + h };
  }
  const h = d.grip.handle;
  const dx = x - fx;
  const dy = y - fy;
  return clampBox(
    {
      class: b.class,
      x1: h.includes('w') ? b.x1 + dx : b.x1,
      x2: h.includes('e') ? b.x2 + dx : b.x2,
      y1: h.includes('n') ? b.y1 + dy : b.y1,
      y2: h.includes('s') ? b.y2 + dy : b.y2,
    },
    width,
    height,
  );
}

/**
 * The boxes once a gesture ends with `shown` drawn, and which one is selected: a drawn box is added (and selected)
 * unless it is too small, which leaves nothing selected (a press on empty image clears the selection); a move or a
 * resize replaces its box, or keeps the old one when the result is too small. A `click` (the pointer hardly moved) only
 * selects: it leaves every box as it was, so looking at a box never counts as a change.
 */
export function settle(boxes: readonly Box[], d: Drag, shown: Box, click = false): { boxes: Box[]; selected: number | null } {
  if (d.grip.kind === 'draw') return click || tooSmall(shown) ? { boxes: [...boxes], selected: null } : { boxes: [...boxes, shown], selected: boxes.length };
  const i = d.grip.index;
  if (click || tooSmall(shown)) return { boxes: [...boxes], selected: i };
  return { boxes: boxes.map((b, k) => (k === i ? shown : b)), selected: i };
}

// ---------- the keyboard's reach: a new box, and moving or resizing the selected one ----------

/**
 * One arrow press, in image px, for an image shown at `scale` screen px per image px: about one screen pixel, never
 * under one image pixel (a 12000 px photo moves visibly, a 40 px icon by its own pixels).
 */
export function keyStep(scale: number): number {
  return scale > 0 ? Math.max(1, Math.round(1 / scale)) : 1;
}

/** A new box of class `cls` for Add box: a square a quarter of the image's shorter side, in its middle. */
export function newBox(width: number, height: number, cls: number): Box {
  const short = Math.max(0, Math.min(width, height));
  const side = Math.min(short, Math.max(MIN_SIDE, Math.round(short / 4)));
  const x1 = Math.max(0, Math.round((width - side) / 2));
  const y1 = Math.max(0, Math.round((height - side) / 2));
  return { class: cls, x1, y1, x2: x1 + side, y2: y1 + side };
}

/** `b` moved by (dx, dy) image px, kept whole inside the image (it slides along an edge it reaches). */
export function nudge(b: Box, dx: number, dy: number, width: number, height: number): Box {
  const c = clampBox(b, width, height);
  const w = c.x2 - c.x1;
  const h = c.y2 - c.y1;
  const x1 = Math.max(0, Math.min(width - w, c.x1 + dx));
  const y1 = Math.max(0, Math.min(height - h, c.y1 + dy));
  return { class: b.class, x1, y1, x2: x1 + w, y2: y1 + h };
}

/** `b` resized from its bottom-right corner by (dw, dh) image px: inside the image, never under `MIN_SIDE` a side. */
export function stretch(b: Box, dw: number, dh: number, width: number, height: number): Box {
  const x1 = Math.max(0, Math.min(width, b.x1));
  const y1 = Math.max(0, Math.min(height, b.y1));
  const x2 = Math.min(width, Math.max(x1 + MIN_SIDE, b.x2 + dw));
  const y2 = Math.min(height, Math.max(y1 + MIN_SIDE, b.y2 + dh));
  return { class: b.class, x1, y1, x2, y2 };
}

/** Whether two box lists are the same labels (order included), to tell an edited image from a saved one. */
export function sameBoxes(a: readonly Box[], b: readonly Box[]): boolean {
  return a.length === b.length && a.every((x, i) => {
    const y = b[i] as Box;
    return x.class === y.class && x.x1 === y.x1 && x.y1 === y.y1 && x.x2 === y.x2 && x.y2 === y.y2;
  });
}

/** Coordinates to a tenth of a pixel, as the server keeps them: no float noise from the screen mapping is saved. */
export function tidy(b: Box): Box {
  const r = (v: number) => Math.round(v * 10) / 10;
  return { class: b.class, x1: r(b.x1), y1: r(b.y1), x2: r(b.x2), y2: r(b.y2) };
}
