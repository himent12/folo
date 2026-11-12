// @vitest-environment happy-dom
import { createRoot, flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { watched } from '$lib/shell/runList';
import { hub, type Api } from './index';
import { modelsKey, modelsResource } from './models';
import type { Model, Run } from './types';

const run = (id: string, status: Run['status'], model_id: string | null = null) => ({ id, status, model_id }) as Run;
const model = (id: string) => ({ id, name: id }) as Model;

describe('modelsKey', () => {
  it('changes when a run ends or is deleted, not while one trains', () => {
    const before = modelsKey([run('a', 'completed', 'm_a'), run('b', 'running')]);
    expect(modelsKey([run('b', 'running'), run('a', 'completed', 'm_a')])).toBe(before);
    expect(modelsKey([run('a', 'completed', 'm_a'), run('b', 'stopping')])).toBe(before);
    expect(modelsKey([run('a', 'completed', 'm_a'), run('b', 'completed', 'm_b')])).not.toBe(before);
    expect(modelsKey([run('b', 'running')])).not.toBe(before);
  });
});

describe('modelsResource', () => {
  let lists: Model[][];
  let stop: () => void;
  beforeEach(() => {
    lists = [[model('m_a')], [model('m_a'), model('m_b')], [model('m_b')]];
    let i = 0;
    hub.api = { mock: true, models: async () => lists[Math.min(i++, lists.length - 1)]! } as unknown as Api;
    hub.mode = 'mock';
    watched.runs = undefined;
  });
  afterEach(() => stop?.());

  it('fetches again when a run finishes, is deleted, or the tab is shown again', async () => {
    let r!: ReturnType<typeof modelsResource>;
    stop = createRoot((dispose) => ((r = modelsResource()), dispose));
    flushSync();
    await vi.waitFor(() => expect(r.data?.map((m) => m.id)).toEqual(['m_a']));
    watched.runs = [run('a', 'completed', 'm_a'), run('b', 'running')];
    flushSync();
    // Where things stand when the list first arrives: no refetch.
    await Promise.resolve();
    expect(r.data?.map((m) => m.id)).toEqual(['m_a']);
    watched.runs = [run('a', 'completed', 'm_a'), run('b', 'completed', 'm_b')];
    flushSync();
    await vi.waitFor(() => expect(r.data?.map((m) => m.id)).toEqual(['m_a', 'm_b']));
    // The tab comes back: files may have changed meanwhile.
    document.dispatchEvent(new Event('visibilitychange'));
    flushSync();
    await vi.waitFor(() => expect(r.data?.map((m) => m.id)).toEqual(['m_b']));
    expect(r.loading).toBe(false);
  });
});
