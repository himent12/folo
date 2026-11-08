import { describe, expect, it } from 'vitest';
import type { DatasetIssue, DatasetIssuesQuery } from '$lib/api/types';
import { loadIssues } from './issues';

// A server with 1,500 empty-image notes first and two errors after them, paged at most 1000 per request.
const all: DatasetIssue[] = [
  ...Array.from({ length: 1500 }, (_, i): DatasetIssue => ({ kind: 'empty_image', severity: 'info', file: `images/train/${String(i).padStart(6, '0')}.png`, line: null, message: 'no boxes' })),
  { kind: 'broken_label', severity: 'error', file: 'labels/val/000002.txt', line: 3, message: 'bad' },
  { kind: 'class_overflow', severity: 'error', file: 'labels/val/000001.txt', line: 1, message: 'bad' },
];
const api = {
  async datasetIssues(_id: string, q: DatasetIssuesQuery) {
    const m = all.filter((i) => !q.kind || i.kind === q.kind);
    const off = q.offset ?? 0;
    return { total: m.length, issues: m.slice(off, off + Math.min(1000, q.limit ?? 100)) };
  },
};

describe('loadIssues', () => {
  it('finds errors that sit after the first 1000 issues', async () => {
    const r = await loadIssues(api, 'ds', 'error');
    expect(r.total).toBe(2);
    expect(r.list.map((i) => i.file)).toEqual(['labels/val/000001.txt', 'labels/val/000002.txt']);
  });

  it('pages through a kind and reports the total past the cap', async () => {
    const r = await loadIssues(api, 'ds', 'info', 1200);
    expect(r.total).toBe(1500);
    expect(r.list).toHaveLength(1200);
  });
});
