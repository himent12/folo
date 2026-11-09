import { describe, expect, it } from 'vitest';
import type { Box } from '$lib/api/types';
import { clampBox, dragTo, fit, grab, handleAt, sameBoxes, settle, tidy, toImage, toScreen, tooSmall, type Drag } from './boxEdit';

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
