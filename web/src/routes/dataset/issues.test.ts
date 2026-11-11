import { describe, expect, it } from 'vitest';
import type { DatasetIssue, DatasetIssuesQuery } from '$lib/api/types';
import { issueCaption, loadIssues, loadKinds } from './issues';

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

describe('loadKinds', () => {
  it("reads a finding's own kinds only, merged in dataset order", async () => {
    const withImage = {
      async datasetIssues(id: string, q: DatasetIssuesQuery) {
        const r = await api.datasetIssues(id, q);
        return q.kind === 'corrupt_image'
          ? { total: 1, issues: [{ kind: 'corrupt_image' as const, severity: 'error' as const, file: 'images/fake.jpg', line: null, message: 'cannot decode' }] }
          : r;
      },
    };
    const r = await loadKinds(withImage, 'ds', 'error', ['broken_label', 'class_overflow']);
    expect(r.total).toBe(2);
    expect(r.list.map((i) => i.kind)).toEqual(['class_overflow', 'broken_label']);
    expect(r.byKind).toEqual({ broken_label: 1, class_overflow: 1 });
  });
});

describe('loadIssues with counts', () => {
  const issues: DatasetIssue[] = [
    { kind: 'broken_label', severity: 'error', file: 'labels/a.txt', line: 3, message: 'bad', fix: 'delete line 3' },
    { kind: 'clipped_box', severity: 'warning', file: 'labels/b.txt', line: 1, message: 'clipped', fix: 'clip the box on line 1 to the image' },
    { kind: 'duplicate_image', severity: 'warning', file: 'images/c.png', line: null, message: 'same bytes as images/a.png', fix: 'remove this copy' },
    { kind: 'empty_image', severity: 'info', file: 'images/d.png', line: null, message: 'no boxes' },
  ];
  const counts = { error: 1, warning: 2, info: 1, by_kind: { broken_label: 1, clipped_box: 1, duplicate_image: 1, empty_image: 1 } };
  const queries: DatasetIssuesQuery[] = [];
  const server = {
    async datasetIssues(_id: string, q: DatasetIssuesQuery) {
      queries.push(q);
      const m = issues.filter((i) => (!q.kind || i.kind === q.kind) && (!q.severity || i.severity === q.severity));
      return { total: m.length, issues: m.slice(q.offset ?? 0), counts };
    },
  };

  it('asks for a whole severity in one query and passes the counts on', async () => {
    queries.length = 0;
    const r = await loadIssues(server, 'ds', 'warning');
    expect(queries).toEqual([{ severity: 'warning', offset: 0, limit: 1000 }]);
    expect(r.total).toBe(2);
    expect(r.list.map((i) => i.fix)).toEqual(['remove this copy', 'clip the box on line 1 to the image']);
    expect(r.counts).toEqual(counts);
    expect(r.byKind).toEqual({ clipped_box: 1, duplicate_image: 1 });
  });

  it('narrows to one kind', async () => {
    queries.length = 0;
    const r = await loadIssues(server, 'ds', 'warning', 2000, undefined, 'clipped_box');
    expect(queries).toEqual([{ kind: 'clipped_box', offset: 0, limit: 1000 }]);
    expect(r.list).toHaveLength(1);
    expect(r.byKind).toEqual({ clipped_box: 1, duplicate_image: 1 });
  });

  it('reads an older server kind by kind and counts per kind itself', async () => {
    const r = await loadIssues(api, 'ds', 'error');
    expect(r.counts).toBeUndefined();
    expect(r.byKind).toEqual({ broken_label: 1, class_overflow: 1 });
  });
});

describe('issueCaption', () => {
  it('names the rows shown: severity, count and what they mean for training', () => {
    expect(issueCaption('error', 3)).toBe('3 errors: these files are left out of training until fixed.');
    expect(issueCaption('warning', 1)).toBe('1 warning: training still uses these files.');
    expect(issueCaption('warning', 2, 'clipped_box')).toBe('2 clipped box warnings: training still uses these files.');
    expect(issueCaption('info', 1200)).toBe('1,200 notes: nothing to fix.');
    expect(issueCaption('error', 0)).toBeNull();
  });
});
