import { describe, expect, it } from 'vitest';
import { boxRevision } from './label';

describe('boxRevision', () => {
  it('is the server’s revision of the boxes (folo_api::label_box_revision)', () => {
    // Revisions folo label printed for these boxes (sweep 4, studio-data-label).
    expect(boxRevision([])).toBe('0-811c9dc5');
    expect(boxRevision([{ class: 0, x1: 4, y1: 5, x2: 18, y2: 20 }])).toBe('1-961f3115');
    // Any change is another revision: a class, a coordinate, the order.
    const two = [
      { class: 0, x1: 4, y1: 5, x2: 18, y2: 20 },
      { class: 1, x1: 0.5, y1: 1.25, x2: 9.75, y2: 30 },
    ];
    const revs = [two, [two[1]!, two[0]!], [{ ...two[0]!, class: 1 }, two[1]!], [two[0]!, { ...two[1]!, x2: 9.5 }]].map(boxRevision);
    expect(new Set(revs).size).toBe(4);
    expect(revs[0]).toMatch(/^2-[0-9a-f]{8}$/);
  });
});
