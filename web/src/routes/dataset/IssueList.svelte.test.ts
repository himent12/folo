// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index.svelte';
import { MockApi } from '$lib/api/mock';
import IssueList from './IssueList.svelte';

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;
let api: MockApi;

async function until<T>(get: () => T | null | undefined | false, what: string): Promise<T> {
  let v: T | null | undefined | false = undefined;
  await vi.waitFor(
    () => {
      flushSync();
      v = get();
      if (!v) throw new Error(`still waiting for ${what}`);
    },
    { timeout: 3000, interval: 10 },
  );
  return v as T;
}

const rows = () => [...target.querySelectorAll('li')];
const radios = () => [...target.querySelectorAll<HTMLButtonElement>('[role="radio"]')];

beforeEach(() => {
  api = new MockApi('normal', 0);
  hub.api = api;
  hub.mode = 'mock';
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
});

describe('IssueList', () => {
  it('lists each issue with its fix, and counts every severity from the server', async () => {
    c = mount(IssueList, { target, props: { datasetId: 'ds_shapes_hard', path: 'D:/data/shapes-hard', severity: 'error' } });
    await until(() => rows().length === 3, 'the three errors');
    expect(target.querySelector('.caption')?.textContent).toBe('3 errors: these files are left out of training until fixed.');
    for (const li of rows()) expect(li.querySelector('.fix')?.textContent).toMatch(/^Fix: delete line \d+$/);
    const counts = await until(() => (radios()[0]?.querySelector('.meta') ? radios().map((b) => b.textContent?.replace(/\s+/g, '')) : null), 'the counts');
    expect(counts[0]).toBe('Errors3');
    expect(counts[1]).toBe('Warnings3');
    expect(counts[2]).toMatch(/^Notes\d+$/);
    // Three kinds of error: a sub-filter by check type (not a second severity filter) lists them with their counts.
    const kinds = [...target.querySelectorAll('select option')].map((o) => o.textContent);
    expect(target.querySelector('select')?.getAttribute('aria-label')).toBe('Check type');
    expect(kinds).toEqual(['All error types', 'Broken label · 1', 'Class out of range · 1', 'Box out of range · 1']);
  });

  it('asks the server for one severity when the filter moves', async () => {
    c = mount(IssueList, { target, props: { datasetId: 'ds_shapes_hard', path: 'D:/data/shapes-hard', severity: 'error' } });
    await until(() => rows().length === 3, 'the errors');
    const read = vi.spyOn(api, 'datasetIssues');
    radios()[1]!.click();
    await until(() => (rows().length === 3 && rows().every((li) => li.querySelector('.fix')?.textContent?.includes('clip') || li.querySelector('.fix')?.textContent?.includes('remove this copy')) ? true : null), 'the warnings');
    expect(read.mock.calls[0]?.[1]).toEqual({ severity: 'warning', offset: 0, limit: 1000 });
    // The caption names exactly the rows shown.
    expect(target.querySelector('.caption')?.textContent).toBe('3 warnings: training still uses these files.');
    // Notes (infos) have no fix.
    radios()[2]!.click();
    await until(() => (rows().length > 0 && rows()[0]?.querySelector('.msg')?.textContent === 'no boxes (a background image)' ? true : null), 'the notes');
    expect(target.querySelector('.fix')).toBeNull();
  });

  it('falls back to reading kind by kind when the server sends no counts', async () => {
    const real = api.datasetIssues.bind(api);
    const read = vi.spyOn(api, 'datasetIssues').mockImplementation(async (id, q) => {
      const { counts: _, ...older } = await real(id, q);
      // An older server ignores ?severity=.
      return q.severity && !q.kind ? await real(id, {}).then(({ counts: _c, ...o }) => o) : older;
    });
    c = mount(IssueList, { target, props: { datasetId: 'ds_shapes_hard', path: 'D:/data/shapes-hard', severity: 'error' } });
    await until(() => rows().length === 3, 'the errors');
    expect(read.mock.calls.some(([, q]) => q.kind === 'broken_label')).toBe(true);
    expect(radios()[0]?.querySelector('.meta')).toBeNull();
    expect(target.querySelectorAll('select option')).toHaveLength(4);
  });
});
