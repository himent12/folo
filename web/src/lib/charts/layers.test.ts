import { describe, expect, it } from 'vitest';
import { paintOrder } from './layers';

const s = (id: string, end: number, extra: object = {}) => ({ id, points: [{ x: 0 }, { x: end }], ...extra });

describe('paintOrder', () => {
  it('draws shorter series last, so they sit on top', () => {
    const out = paintOrder([s('a', 38), s('b', 150), s('c', 61), s('d', 23)], (t) => t);
    expect(out.map((t) => t.id)).toEqual(['b', 'c', 'a', 'd']);
  });

  it('draws patterned series over solid ones, whatever their length', () => {
    // Compare: sky, amber solid; magenta dashed, teal dotted. The dotted 23-epoch run must not lie under a solid one.
    const out = paintOrder([s('sky', 38), s('amber', 150), s('magenta', 61, { pattern: 'dash' }), s('teal', 23, { pattern: 'dot' }), s('short', 10)], (t) => t);
    expect(out.map((t) => t.id)).toEqual(['amber', 'sky', 'short', 'magenta', 'teal']);
  });

  it('keeps context lines at the back and emphasised lines in front, each in the given order', () => {
    const out = paintOrder([s('mean', 1, { emphasis: true }), s('c0', 1, { context: true }), s('lit', 0.5, { emphasis: true }), s('c1', 0.2, { context: true })], (t) => t);
    expect(out.map((t) => t.id)).toEqual(['c0', 'c1', 'mean', 'lit']);
  });

  it('keeps the given order for series of the same length', () => {
    const out = paintOrder([s('total', 10), s('box', 10), s('cls', 10)], (t) => t);
    expect(out.map((t) => t.id)).toEqual(['total', 'box', 'cls']);
  });
});
