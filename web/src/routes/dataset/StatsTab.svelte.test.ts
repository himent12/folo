// @vitest-environment happy-dom
import { flushSync, mount, tick, unmount } from 'svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { hub } from '$lib/api/index.svelte';
import { MockApi } from '$lib/api/mock';
import StatsTab from './StatsTab.svelte';

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

beforeEach(() => {
  const api = new MockApi('normal', 0);
  hub.api = api;
  hub.mode = 'mock';
  target = document.createElement('div');
  document.body.appendChild(target);
  Element.prototype.scrollIntoView ??= () => {};
  window.matchMedia ??= (() => ({ matches: false })) as unknown as typeof window.matchMedia;
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
});

describe('Dataset stats: health findings', () => {
  it("takes keyboard focus to the check findings when a finding's evidence is asked for", async () => {
    const dataset = await hub.api.dataset('ds_shapes_hard');
    const counts = (await hub.api.datasetIssues('ds_shapes_hard', { limit: 1 })).counts;
    c = mount(StatsTab, { target, props: { dataset, stats: dataset.stats!, counts, onclass: () => {} } });
    flushSync();
    const show = target.querySelector<HTMLButtonElement>('button.haction');
    expect(show).not.toBeNull();
    show!.click();
    await tick();
    await tick();
    expect(document.activeElement?.id).toBe('ds-check-title');
  });

  it('folds each severity into a complete summary with what it counts, and names the units', async () => {
    const dataset = await hub.api.dataset('ds_shapes_hard');
    const counts = (await hub.api.datasetIssues('ds_shapes_hard', { limit: 1 })).counts;
    c = mount(StatsTab, { target, props: { dataset, stats: dataset.stats!, counts, onclass: () => {} } });
    flushSync();
    const line = (sev: string) => {
      const s = target.querySelector(`#ds-health li.${sev} summary`);
      return { sum: s?.querySelector('.htitles')?.textContent?.trim(), count: s?.querySelector('.hcount')?.textContent?.trim() };
    };
    expect(line('warning')).toEqual({ sum: '3 file issues · 35% tiny boxes', count: '3 file issues · 1 dataset-wide' });
    expect(line('danger')).toEqual({ sum: '3 broken label files', count: '3 file issues' });
    expect(target.querySelector('#ds-health')?.textContent).toContain('3 issue categories · 6 file issues · 1 note');
  });
});
