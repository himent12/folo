import { describe, expect, it } from 'vitest';
import { advantage, versusYolo26n } from './benchmarks';

describe('benchmarks (docs/BENCHMARKS.md, 2026-11-08)', () => {
  it('reads each comparison in its own direction', () => {
    const gpu = versusYolo26n.find((v) => v.metric === 'GPU forward, batch 1')!;
    expect(advantage(gpu)).toBeCloseTo(4.95 / 2.99);
    const tput = versusYolo26n.find((v) => v.metric === 'GPU end-to-end throughput, batch 16')!;
    expect(advantage(tput)).toBeCloseTo(722 / 615);
  });
});
