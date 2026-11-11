// A run's dataset on Overview: a name, a placeholder or "deleted dataset", never a raw id (shell F15).
import { describe, expect, it } from 'vitest';
import { datasetLabel } from './datasetLabel';

const loaded = new Map([['ds_cbb5', { name: 'shapes-small' }]]);

describe('datasetLabel', () => {
  it('is the name once the datasets loaded', () => {
    expect(datasetLabel('ds_cbb5', loaded, false)).toBe('shapes-small');
  });

  it('is a placeholder while they load, and nothing when they failed', () => {
    expect(datasetLabel('ds_cbb5', undefined, false)).toBeUndefined();
    expect(datasetLabel('ds_cbb5', undefined, true)).toBeNull();
  });

  it('says the dataset was deleted, not its id', () => {
    expect(datasetLabel('ds_977d', loaded, false)).toBe('deleted dataset');
  });
});
