import { describe, expect, it, vi } from 'vitest';
import type { Api } from './client';
import { MAX_RUNS, mapHistory } from './metricsCache';
import type { Run } from './types';

function fakeApi() {
  const runMetrics = vi.fn(async (_id: string) => ({ steps: [], epochs: [{ val: { map50_95: 0.5 } }] }));
  return { api: { runMetrics } as unknown as Api, runMetrics };
}
const run = (i: number, status = 'completed', epoch = 3) => ({ id: `run_${i}`, status, epoch }) as unknown as Run;

describe('mapHistory', () => {
  it('fetches each finished run once across visits of a list longer than 200 runs', async () => {
    const { api, runMetrics } = fakeApi();
    const runs = Array.from({ length: 304 }, (_, i) => run(i));
    for (let visit = 0; visit < 3; visit++) await Promise.all(runs.map((r) => mapHistory(api, r)));
    expect(runMetrics).toHaveBeenCalledTimes(304);
  });

  it('replaces an active run entry as it advances instead of keeping one per epoch', async () => {
    const { api, runMetrics } = fakeApi();
    for (let e = 1; e <= 3; e++) {
      await mapHistory(api, run(1, 'running', e));
      await mapHistory(api, run(1, 'running', e));
    }
    await mapHistory(api, run(1, 'completed', 3));
    await mapHistory(api, run(1, 'completed', 3));
    expect(runMetrics).toHaveBeenCalledTimes(4);
  });

  it('evicts the least recently used run past the bound', async () => {
    const { api, runMetrics } = fakeApi();
    await mapHistory(api, run(0));
    for (let i = 1; i < MAX_RUNS; i++) void mapHistory(api, run(i));
    await mapHistory(api, run(0)); // used again: now the most recent
    await mapHistory(api, run(MAX_RUNS)); // evicts run_1, not run_0
    runMetrics.mockClear();
    await mapHistory(api, run(0));
    expect(runMetrics).not.toHaveBeenCalled();
    await mapHistory(api, run(1));
    expect(runMetrics).toHaveBeenCalledTimes(1);
  });

  it('forgets a failed fetch so the next use retries', async () => {
    const { api, runMetrics } = fakeApi();
    runMetrics.mockRejectedValueOnce(new Error('offline'));
    await expect(mapHistory(api, run(7))).rejects.toThrow('offline');
    await expect(mapHistory(api, run(7))).resolves.toEqual([0.5]);
    expect(runMetrics).toHaveBeenCalledTimes(2);
  });
});
