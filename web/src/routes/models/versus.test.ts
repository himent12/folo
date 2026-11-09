import { describe, expect, it } from 'vitest';
import { unlike, versusYolo26n } from '$lib/benchmarks';

describe('engine speed rows', () => {
  it('keep each qualifier to one short line; longer notes go to the measurement fold', () => {
    for (const v of versusYolo26n) expect(v.note?.length ?? 0, v.metric).toBeLessThanOrEqual(60);
    expect(versusYolo26n.filter((v) => v.caveat).length).toBeGreaterThan(0);
  });

  it("name each unlike row's difference in a few words, for rows that exist", () => {
    const metrics = new Set(versusYolo26n.map((v) => v.metric));
    for (const [m, u] of Object.entries(unlike)) {
      expect(metrics.has(m), m).toBe(true);
      expect(u.kinds.length, m).toBeGreaterThan(0);
      expect(u.differs.length, m).toBeLessThanOrEqual(24);
    }
  });
});
