import { describe, expect, it } from 'vitest';
import { DEFAULT_STATE, emptyFilterSummary, leftForModel, readState, returnQuery, writeState } from './listState';

describe('the Models list state in the URL', () => {
  it('round-trips with defaults omitted and picks sensible default sort direction per column', () => {
    const s = { group: '~published', sort: 'bytes', dir: 'asc', q: 'folo-n' } as const;
    const q = writeState(s);
    expect(q).toBe('group=%7Epublished&q=folo-n&sort=bytes&dir=asc');
    expect(readState(new URLSearchParams(q))).toEqual(s);
    expect(writeState(DEFAULT_STATE)).toBe('');

    // Text columns default to asc when dir is omitted; numeric columns default to desc.
    expect(readState(new URLSearchParams('sort=name'))).toEqual({ ...DEFAULT_STATE, sort: 'name', dir: 'asc' });
    expect(readState(new URLSearchParams('sort=bytes'))).toEqual({ ...DEFAULT_STATE, sort: 'bytes', dir: 'desc' });
    expect(readState(new URLSearchParams('group=%20%20&sort=unknown&dir=sideways'))).toEqual(DEFAULT_STATE);
    expect(readState(new URLSearchParams('group=%20~published%20&q=fo%0Alo'))).toEqual({
      ...DEFAULT_STATE,
      group: '~published',
      q: 'folo',
    });
  });

  it('summarizes active group and search filters for empty-state recovery', () => {
    expect(emptyFilterSummary({ group: '~published', q: 'zebra' }, 'Published weights')).toBe('No models in Published weights match "zebra".');
    expect(emptyFilterSummary({ group: 'ds_missing:val', q: 'zebra' }, null)).toBe('No models match the selected group filter and "zebra".');
    expect(emptyFilterSummary({ group: '~imported', q: '' }, 'Imported')).toBe('No models in Imported right now.');
    expect(emptyFilterSummary({ group: 'ds_missing:val', q: '' }, null)).toBe('No models match the selected group filter.');
    expect(emptyFilterSummary({ group: 'all', q: 'zebra' })).toBe('No models match "zebra".');
    expect(emptyFilterSummary({ group: 'all', q: '' })).toBe('Try another dataset or search term.');
  });

  it('restores the query once when returning from a model page to bare /models', () => {
    leftForModel(7, 'group=~published&q=coco');
    expect(returnQuery(8, new URLSearchParams())).toBe('group=~published&q=coco');
    expect(returnQuery(8, new URLSearchParams())).toBeNull();
    leftForModel(7, 'q=coco');
    expect(returnQuery(9, new URLSearchParams())).toBeNull();
  });
});
