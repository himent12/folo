import { describe, expect, it } from 'vitest';
import { overviewGeometry, overviewHit, overviewWindow } from './cmOverview';

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
