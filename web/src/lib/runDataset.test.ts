// A run's dataset in a list: a name, a placeholder or the removed one's recorded name, never a raw id (shell F15).
import { describe, expect, it } from 'vitest';
import type { Run } from '$lib/api/types';
import { runDatasetLabel } from './runDataset';

const loaded = new Map([['ds_cbb5', { name: 'shapes-small' }]]);
const run = (dataset_id: string, recorded?: string) => ({ config: { dataset_id }, dataset: recorded ? { name: recorded, classes: [] } : undefined }) as Pick<Run, 'config' | 'dataset'>;

describe('runDatasetLabel', () => {
  it('is the name once the datasets loaded', () => {
    expect(runDatasetLabel(run('ds_cbb5', 'old-name'), loaded, false)).toBe('shapes-small');
  });

  it('is a placeholder while they load, and the recorded name or nothing when they failed', () => {
    expect(runDatasetLabel(run('ds_cbb5', 'shapes-small'), undefined, false)).toBeUndefined();
    expect(runDatasetLabel(run('ds_cbb5', 'shapes-small'), undefined, true)).toBe('shapes-small');
    expect(runDatasetLabel(run('ds_cbb5'), undefined, true)).toBeNull();
  });

  it('names a removed dataset as the run recorded it, marked, never by its id', () => {
    expect(runDatasetLabel(run('ds_977d', 'fasteners'), loaded, false)).toBe('fasteners (removed)');
    expect(runDatasetLabel(run('ds_977d'), loaded, false)).toBe('removed dataset');
  });
});
