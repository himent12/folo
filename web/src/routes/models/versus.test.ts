import { describe, expect, it } from 'vitest';
import { unlike, unlikeCondition, versusYolo26n } from '$lib/benchmarks';

describe('engine speed rows', () => {
  it('keep each qualifier to one short line; longer notes go to the measurement fold', () => {
    for (const v of versusYolo26n) expect(v.note?.length ?? 0, v.metric).toBeLessThanOrEqual(60);
    expect(versusYolo26n.filter((v) => v.caveat).length).toBeGreaterThan(0);
  });

  it("keep a sub-line only where a row differs from the table's default, never restating it or the conclusion", () => {
    // The default (both f32, end to end with NMS) is said once over the columns; an unlike row's difference is its
    // conclusion column's words. A wall of identical "f32, NMS included" lines was round 27's Engine benchmarks.
    for (const v of versusYolo26n) {
      if (!v.note) continue;
      expect(v.note, v.metric).not.toMatch(/^f32\b|\bf32 vs f32\b|NMS included/);
      const u = unlike[v.metric];
      if (u) expect(v.note.includes(u.differs), v.metric).toBe(false);
    }
    expect(versusYolo26n.filter((v) => v.note).length).toBeLessThan(versusYolo26n.length / 2);
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

describe('unlikeCondition', () => {
  it('says each row its own condition only ("no winner" is the legend, said once)', () => {
    expect(unlikeCondition(unlike['GPU forward throughput, batch 16']!)).toBe('Different precision: f32 vs TensorFloat-32');
    expect(unlikeCondition(unlike['GPU end-to-end throughput, batch 16']!)).toBe('Different precision and work: f32 vs fp16');
    expect(unlikeCondition(unlike['Training peak GPU memory']!)).toBe('Different work: train step vs trainer');
  });
});
