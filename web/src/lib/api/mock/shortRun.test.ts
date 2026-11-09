import { describe, expect, it } from 'vitest';
import { scheduleWarnings as found } from '../../../routes/run/warnings';
import { MockApi } from './index';

describe('mock API: short-run scenario', () => {
  it('adds a 5-epoch run whose log carries the trainer’s two schedule warnings after its first line', async () => {
    const api = new MockApi('short-run', 0);
    expect((await api.runs()).map((r) => r.id)).toContain('run_s5q1');
    const { lines } = await api.runLogs('run_s5q1');
    expect(lines[1]).toMatchObject({ level: 'warn' });
    expect(lines[1]!.message).toMatch(/^this run takes only 125 optimizer steps \(5 epochs × 25: 800 training images, batch 32\);/);
    expect(lines[1]!.message).toMatch(/Train for at least 8 epochs/);
    expect(lines[2]!.message).toBe("warmup takes 75 of the run's 125 optimizer steps, so the learning rate reaches lr0 only in the second half of the run; lower warmup_epochs (now 3) or train for more epochs");
    expect(found(lines)).toHaveLength(2);
  });

  it('warns about no run in the normal demo', async () => {
    const api = new MockApi('normal', 0);
    for (const r of await api.runs()) expect(found((await api.runLogs(r.id)).lines), r.id).toEqual([]);
  });
});
