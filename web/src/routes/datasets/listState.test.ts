import { describe, expect, it } from 'vitest';
import { DEFAULT_STATE, emptyFilterSummary, leftForDataset, readState, returnQuery, writeState } from './listState';

describe('the Datasets list state in the URL', () => {
  it('round-trips with defaults omitted and invalid params falling back to defaults', () => {
    const s = { format: 'yolo', sort: 'images', q: 'fasteners v2' } as const;
    const q = writeState(s);
    expect(q).toBe('format=yolo&sort=images&q=fasteners+v2');
    expect(readState(new URLSearchParams(q))).toEqual(s);
    expect(writeState(DEFAULT_STATE)).toBe('');
    expect(readState(new URLSearchParams('format=voc&sort=size'))).toEqual(DEFAULT_STATE);
  });

  it('preserves transient params such as ?add=folder while replacing filter keys', () => {
    const existing = new URLSearchParams('add=folder&format=coco&q=old');
    expect(writeState({ format: 'yolo', sort: 'boxes', q: 'new' }, existing)).toBe('add=folder&format=yolo&sort=boxes&q=new');
    expect(writeState(DEFAULT_STATE, existing)).toBe('add=folder');
  });

  it('summarizes active filters for empty-state recovery', () => {
    expect(emptyFilterSummary({ format: 'yolo', q: 'zebra' })).toBe('No YOLO datasets match "zebra".');
    expect(emptyFilterSummary({ format: 'coco', q: '   ' })).toBe('No COCO datasets found.');
    expect(emptyFilterSummary({ format: 'all', q: 'zebra' })).toBe('No datasets match "zebra".');
    expect(emptyFilterSummary({ format: 'all', q: '' })).toBe('Try another format or search term.');
  });

  it('restores the query once when returning from a dataset page to bare /datasets', () => {
    leftForDataset(3, 'format=yolo&q=fast');
    expect(returnQuery(4, new URLSearchParams())).toBe('format=yolo&q=fast');
    expect(returnQuery(4, new URLSearchParams())).toBeNull();
    leftForDataset(3, 'format=yolo');
    expect(returnQuery(4, new URLSearchParams('q=other'))).toBeNull();
    leftForDataset(3, '');
    expect(returnQuery(4, new URLSearchParams())).toBeNull();
  });
});
