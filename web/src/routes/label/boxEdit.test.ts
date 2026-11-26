import { describe, expect, it } from 'vitest';
import type { Box } from '$lib/api/types';
import { CLICK_SLOP, clampBox, dragTo, fit, grab, handleAt, isClick, keyStep, MIN_SIDE, newBox, nudge, sameBoxes, settle, stretch, tidy, toImage, toScreen, tooSmall, type Drag } from './boxEdit';

const box = (x1: number, y1: number, x2: number, y2: number, cls = 0): Box => ({ class: cls, x1, y1, x2, y2 });
const W = 640;
const H = 480;

describe('placement', () => {
  it('fits the image in the view, centred, and maps both ways', () => {
    const p = fit(640, 480, 800, 800);
    expect(p).toEqual({ s: 1.25, ox: 0, oy: 100 });
    expect(toScreen(p, 320, 240)).toEqual([400, 400]);
    expect(toImage(p, 400, 400)).toEqual([320, 240]);
    expect(toImage(p, 0, 100)).toEqual([0, 0]);
  });
});

describe('grab', () => {
  const boxes = [box(100, 100, 300, 300), box(150, 150, 200, 200)];

  it("takes the selected box's nearest handle within reach", () => {
    expect(grab(boxes, 0, 103, 98, 6)).toEqual({ kind: 'resize', index: 0, handle: 'nw' });
    expect(grab(boxes, 0, 300, 201, 6)).toEqual({ kind: 'resize', index: 0, handle: 'e' });
    expect(grab(boxes, 0, 200, 305, 6)).toEqual({ kind: 'resize', index: 0, handle: 's' });
    expect(handleAt(boxes[0]!, 'se')).toEqual([300, 300]);
  });

  it('takes the smallest box under the point to move, so a box inside another can be taken', () => {
    expect(grab(boxes, null, 175, 175, 6)).toEqual({ kind: 'move', index: 1 });
    expect(grab(boxes, null, 120, 250, 6)).toEqual({ kind: 'move', index: 0 });
    // Handles belong to the selected box only.
    expect(grab(boxes, null, 100, 100, 6)).toEqual({ kind: 'move', index: 0 });
  });

  it('draws on empty image', () => {
    expect(grab(boxes, 1, 400, 50, 6)).toEqual({ kind: 'draw' });
    expect(grab([], null, 1, 1, 6)).toEqual({ kind: 'draw' });
  });
});

describe('dragTo', () => {
  const start = box(100, 100, 200, 160, 2);

  it('moves a box whole, sliding along the edge it meets', () => {
    const d: Drag = { grip: { kind: 'move', index: 0 }, from: [150, 130], start, cls: 0 };
    expect(dragTo(d, 170, 120, W, H)).toEqual(box(120, 90, 220, 150, 2));
    expect(dragTo(d, -500, 130, W, H)).toEqual(box(0, 100, 100, 160, 2));
    expect(dragTo(d, 2000, 2000, W, H)).toEqual(box(540, 420, 640, 480, 2));
  });

  it('resizes by a corner or an edge, clamped to the image, flipping when dragged past the other side', () => {
    const at = (handle: 'se' | 'n' | 'w', x: number, y: number) => {
      const [hx, hy] = handleAt(start, handle);
      return dragTo({ grip: { kind: 'resize', index: 0, handle }, from: [hx, hy], start, cls: 0 }, x, y, W, H);
    };
    expect(at('se', 260, 200)).toEqual(box(100, 100, 260, 200, 2));
    expect(at('n', 999, 40)).toEqual(box(100, 40, 200, 160, 2));
    expect(at('n', 150, -50)).toEqual(box(100, 0, 200, 160, 2));
    expect(at('w', 260, 130)).toEqual(box(200, 100, 260, 160, 2));
  });

  it('draws from the press point in the class last used, inside the image', () => {
    const d: Drag = { grip: { kind: 'draw' }, from: [600, 400], start: null, cls: 3 };
    expect(dragTo(d, 700, 300, W, H)).toEqual(box(600, 300, 640, 400, 3));
  });
});

describe('settle', () => {
  const boxes = [box(10, 10, 50, 50)];

  it('adds a drawn box and selects it; a click on empty image clears the selection', () => {
    const draw: Drag = { grip: { kind: 'draw' }, from: [100, 100], start: null, cls: 1 };
    expect(settle(boxes, draw, box(100, 100, 140, 130, 1))).toEqual({ boxes: [...boxes, box(100, 100, 140, 130, 1)], selected: 1 });
    expect(settle(boxes, draw, box(100, 100, 101, 130, 1))).toEqual({ boxes, selected: null });
  });

  it('replaces a moved or resized box, and keeps the old one when the result is under 2 px', () => {
    const resize: Drag = { grip: { kind: 'resize', index: 0, handle: 'e' }, from: [50, 30], start: boxes[0]!, cls: 0 };
    expect(settle(boxes, resize, box(10, 10, 80, 50))).toEqual({ boxes: [box(10, 10, 80, 50)], selected: 0 });
    expect(settle(boxes, resize, box(10, 10, 11, 50))).toEqual({ boxes, selected: 0 });
  });

  it('changes nothing on a click: a big box drawn by a wobble on a large image, or a box pressed to select it', () => {
    const draw: Drag = { grip: { kind: 'draw' }, from: [100, 100], start: null, cls: 1 };
    expect(settle(boxes, draw, box(100, 100, 129, 129, 1), true)).toEqual({ boxes, selected: null });
    const move: Drag = { grip: { kind: 'move', index: 0 }, from: [30, 30], start: boxes[0]!, cls: 0 };
    const r = settle(boxes, move, box(10, 10, 50, 50.00000000000001), true);
    expect(r.selected).toBe(0);
    expect(r.boxes[0]).toBe(boxes[0]);
  });

  it('tells a click from a drag by the screen distance the pointer went', () => {
    expect(isClick([10, 10], [10 + CLICK_SLOP - 1, 10 - CLICK_SLOP + 1])).toBe(true);
    expect(isClick([10, 10], [10, 10 + CLICK_SLOP])).toBe(false);
  });
});

describe('helpers', () => {
  it('orders and clamps corners, and tells a box under 2 px', () => {
    expect(clampBox(box(700, -5, 600, 20), W, H)).toEqual(box(600, 0, 640, 20));
    expect(tooSmall(box(0, 0, 1.9, 10))).toBe(true);
    expect(tooSmall(box(0, 0, 2, 2))).toBe(false);
  });

  it('compares labels and rounds coordinates to a tenth of a pixel', () => {
    expect(sameBoxes([box(1, 2, 3, 4)], [box(1, 2, 3, 4)])).toBe(true);
    expect(sameBoxes([box(1, 2, 3, 4)], [box(1, 2, 3, 4, 1)])).toBe(false);
    expect(sameBoxes([], [box(1, 2, 3, 4)])).toBe(false);
    expect(tidy(box(1.04, 2.06, 3.15, 4.449))).toEqual(box(1, 2.1, 3.2, 4.4));
  });
});

describe('the keyboard: Add box, arrows and Alt+arrows', () => {
  it('steps about a screen pixel, never under an image pixel', () => {
    expect(keyStep(0.069)).toBe(14); // a 12000 px photo in a 830 px frame
    expect(keyStep(1.3)).toBe(1);
    expect(keyStep(20)).toBe(1); // a 40 px icon
    expect(keyStep(0)).toBe(1); // not laid out yet
  });

  it('adds a square a quarter of the shorter side in the middle, in the class given', () => {
    expect(newBox(W, H, 2)).toEqual(box(260, 180, 380, 300, 2));
    // A tiny image still gets a box it can hold.
    const tiny = newBox(3, 5, 0);
    expect(tooSmall(tiny)).toBe(false);
    expect(clampBox(tiny, 3, 5)).toEqual(tiny);
  });

  it('moves a box whole, sliding along the edge it reaches', () => {
    const b = box(100, 100, 200, 150, 1);
    expect(nudge(b, 1, 0, W, H)).toEqual(box(101, 100, 201, 150, 1));
    expect(nudge(b, -10, 10, W, H)).toEqual(box(90, 110, 190, 160, 1));
    expect(nudge(box(0, 0, 50, 50), -1, -10, W, H)).toEqual(box(0, 0, 50, 50));
    expect(nudge(box(600, 470, 640, 480), 10, 10, W, H)).toEqual(box(600, 470, 640, 480));
  });

  it('resizes from the bottom-right corner, inside the image and never under the minimum', () => {
    const b = box(100, 100, 200, 150);
    expect(stretch(b, 1, 0, W, H)).toEqual(box(100, 100, 201, 150));
    expect(stretch(b, -10, 10, W, H)).toEqual(box(100, 100, 190, 160));
    expect(stretch(b, 1000, 1000, W, H)).toEqual(box(100, 100, W, H));
    expect(stretch(b, -1000, -1000, W, H)).toEqual(box(100, 100, 100 + MIN_SIDE, 100 + MIN_SIDE));
  });

  it('guards degenerate image dimensions and oversized/inverted boxes during fit, move, nudge, stretch, and newBox', () => {
    expect(fit(0, 480, 320, 240)).toEqual({ s: 0, ox: 0, oy: 0 });
    expect(fit(640, -1, 320, 240)).toEqual({ s: 0, ox: 0, oy: 0 });
    expect(toImage({ s: 0, ox: 10, oy: 20 }, 100, 100)).toEqual([0, 0]);

    // Oversized / inverted boxes stay strictly inside [0, W] x [0, H] on move and nudge.
    const oversized = box(700, -20, -10, 600, 2);
    const moved = dragTo({ grip: { kind: 'move', index: 0 }, from: [50, 50], start: oversized, cls: 2 }, 120, 80, W, H);
    expect(moved).toEqual(box(0, 0, W, H, 2));
    expect(nudge(oversized, 15, -15, W, H)).toEqual(box(0, 0, W, H, 2));

    // Stretch keeps x2 >= x1 and y2 >= y1 even when x1 starts past the image right edge.
    const stretched = stretch(box(W + 10, H + 5, W + 20, H + 15), -5, -5, W, H);
    expect(stretched.x1).toBe(W);
    expect(stretched.x2).toBe(W);
    expect(stretched.y1).toBe(H);
    expect(stretched.y2).toBe(H);

    // Sub-MIN_SIDE image (1x1) produces a non-negative in-bounds box.
    const tiny = newBox(1, 1, 0);
    expect(tiny).toEqual(box(0, 0, 1, 1, 0));
  });
});
