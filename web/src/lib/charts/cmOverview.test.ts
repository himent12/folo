import { describe, expect, it } from 'vitest';
import { overviewGeometry, overviewHit, overviewWindow, pageFit, pageRange } from './cmOverview';

describe('confusion overview', () => {
  it('fits every cell into the size it is given', () => {
    // 13 classes + background (a reduced COCO matrix).
    const g = overviewGeometry(14, 168);
    expect(g).toEqual({ cell: 11, gap: 1, size: 167 });
    expect(g.size).toBeLessThanOrEqual(168);
    // Small matrices cap the cell size; huge ones keep a visible minimum.
    expect(overviewGeometry(3).cell).toBe(20);
    expect(overviewGeometry(81).cell).toBe(6);
  });

  it('maps a tap to its row and column', () => {
    const g = overviewGeometry(9, 168); // 17 px cells, step 18
    expect(g.cell).toBe(17);
    expect(overviewHit(0, 0, g, 9)).toEqual({ r: 0, c: 0 });
    expect(overviewHit(17.5, 40, g, 9)).toEqual({ r: 2, c: 0 }); // in the gap after column 0
    expect(overviewHit(18, 40, g, 9)).toEqual({ r: 2, c: 1 });
    expect(overviewHit(g.size - 1, g.size - 1, g, 9)).toEqual({ r: 8, c: 8 });
    expect(overviewHit(g.size, 4, g, 9)).toBeNull();
    expect(overviewHit(-1, 4, g, 9)).toBeNull();
  });

  it('marks the columns the full matrix shows', () => {
    const g = overviewGeometry(14, 168);
    expect(overviewWindow(1, 6, g)).toEqual({ x: 0, w: 71 });
    expect(overviewWindow(9, 14, g)).toEqual({ x: 96, w: 71 });
  });
});

describe('matrix pages', () => {
  it('fits whole columns beside the row labels, so no column is half in view', () => {
    // A 390 px phone: the matrix box is 324 px, 96 px of row labels, 32 px cells and 2 px gaps: six whole columns.
    const fit = pageFit(324, 98, 32, 2, 9)!;
    expect(fit.cols).toBe(6);
    // Six columns fill the width exactly: nothing of a seventh shows.
    expect(98 + fit.cols * (fit.cell + 2)).toBeLessThanOrEqual(324);
    expect(98 + fit.cols * (fit.cell + 2)).toBeGreaterThan(324 - 0.1 * fit.cols);
    expect(fit.cell).toBeGreaterThanOrEqual(32);
    // On touch (44 px cells), four.
    expect(pageFit(324, 98, 44, 2, 9)!.cols).toBe(4);
    // Every column fits: no paging.
    expect(pageFit(1200, 98, 32, 2, 9)).toBeNull();
  });

  it('counts the columns in view from the scroll offset, the last page ending on the last column', () => {
    const step = 36;
    expect(pageRange(0, step, 6, 9)).toEqual({ first: 1, last: 6 });
    expect(pageRange(3 * step, step, 6, 9)).toEqual({ first: 4, last: 9 });
    // Mid-snap the counter names the columns nearest the offset, and never runs past the matrix.
    expect(pageRange(1.4 * step, step, 6, 9)).toEqual({ first: 2, last: 7 });
    expect(pageRange(99 * step, step, 6, 9)).toEqual({ first: 4, last: 9 });
  });
});
