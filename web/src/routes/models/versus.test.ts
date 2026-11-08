import { describe, expect, it } from 'vitest';
import { versusYolo26n } from '$lib/benchmarks';

describe('engine speed rows', () => {
  it('keep each qualifier to one short line; longer notes go to the measurement fold', () => {
    for (const v of versusYolo26n) expect(v.note?.length ?? 0, v.metric).toBeLessThanOrEqual(60);
    expect(versusYolo26n.filter((v) => v.caveat).length).toBeGreaterThan(0);
  });
});
