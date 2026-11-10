import { describe, expect, it } from 'vitest';
import { scheduleWarnings as found } from '../../../routes/run/warnings';
import { MockApi } from './index';

describe('mock API: short-run scenario', () => {
  it('adds a 5-epoch run whose log carries the trainer’s two schedule warnings after its first line', async () => {
    // 800 images at batch 32: 25 batches an epoch, 2 per optimizer step, so 13 steps an epoch, as the trainer counts.
    const api = new MockApi('short-run', 0);
    expect((await api.runs()).map((r) => r.id)).toContain('run_s5q1');
    const { lines } = await api.runLogs('run_s5q1');
    expect(lines[1]).toMatchObject({ level: 'warn' });
    expect(lines[1]!.message).toMatch(/65 optimizer steps \(5 epochs × 13/);
    expect(lines[2]!.message).toMatch(/warmup takes 39 of the run's 65 optimizer steps/);
    expect(found(lines)).toHaveLength(2);
  });

  it('warns about no run in the normal demo', async () => {
    const api = new MockApi('normal', 0);
    for (const r of await api.runs()) expect(found((await api.runLogs(r.id)).lines), r.id).toEqual([]);
  });
});
