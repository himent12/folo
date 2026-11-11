// Best models refetches when a run ends or goes, and only then (runs F4).
import { describe, expect, it } from 'vitest';
import type { Run } from '$lib/api/types';
import { modelsMayHaveChanged } from './runChanges';

const run = (id: string, status: Run['status']) => ({ id, status }) as unknown as Run;

describe('modelsMayHaveChanged', () => {
  it('is false for the first list and for an unchanged one', () => {
    expect(modelsMayHaveChanged(undefined, [run('a', 'completed')])).toBe(false);
    expect(modelsMayHaveChanged([run('a', 'running'), run('b', 'completed')], [run('a', 'running'), run('b', 'completed')])).toBe(false);
  });

  it('is true when a run completes, stops or fails', () => {
    for (const end of ['completed', 'stopped', 'failed'] as const) {
      expect(modelsMayHaveChanged([run('a', 'running')], [run('a', end)])).toBe(true);
    }
    expect(modelsMayHaveChanged([run('a', 'stopping')], [run('a', 'stopped')])).toBe(true);
    expect(modelsMayHaveChanged([run('a', 'queued')], [run('a', 'failed')])).toBe(true);
  });

  it('is true when a run is deleted, false when one is added or starts', () => {
    expect(modelsMayHaveChanged([run('a', 'completed'), run('b', 'failed')], [run('b', 'failed')])).toBe(true);
    expect(modelsMayHaveChanged([run('a', 'completed')], [run('b', 'queued'), run('a', 'completed')])).toBe(false);
    expect(modelsMayHaveChanged([run('a', 'queued')], [run('a', 'running')])).toBe(false);
  });
});
