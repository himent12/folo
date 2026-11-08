import { describe, expect, it } from 'vitest';
import { advantage, e2eMs, versusYolo26n } from './benchmarks';

describe('benchmarks (docs/BENCHMARKS.md, 2026-11-08)', () => {
  it('reads each comparison in its own direction', () => {
    const gpu = versusYolo26n.find((v) => v.metric === 'GPU forward, batch 1')!;
    expect(advantage(gpu)).toBeCloseTo(4.95 / 2.99);
    const tput = versusYolo26n.find((v) => v.metric === 'GPU end-to-end throughput, batch 16')!;
    expect(advantage(tput)).toBeCloseTo(722 / 615);
  });

  it('keeps the rows FOLO trails (GPU forward throughput, CPU end to end)', () => {
    const behind = versusYolo26n.filter((v) => advantage(v) < 1).map((v) => v.metric);
    expect(behind).toEqual(['GPU forward throughput, batch 16', 'CPU end to end, batch 1']);
  });

  it('uses the document figures for training memory', () => {
    const mem = versusYolo26n.find((v) => v.unit === 'MiB')!;
    expect([mem.folo, mem.yolo]).toEqual([3474, 4353]);
  });

  it('has measured latency only for the sizes the document measured', () => {
    expect(Object.keys(e2eMs).sort()).toEqual(['folo-n', 'folo-s']);
  });
});
