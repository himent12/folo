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
  it("opens a finding's check issues under the finding, filtered to its severity, one finding at a time", async () => {
    const dataset = await hub.api.dataset('ds_shapes_hard');
    const counts = (await hub.api.datasetIssues('ds_shapes_hard', { limit: 1 })).counts;
    c = mount(StatsTab, { target, props: { dataset, stats: dataset.stats!, counts, onclass: () => {} } });
    flushSync();
    // No separate findings section: the list lives under the findings.
    expect(target.querySelector('#ds-check')).toBeNull();
    const shows = [...target.querySelectorAll<HTMLButtonElement>('button.haction')];
    expect(shows.map((b) => b.textContent?.trim())).toEqual(['Show the files', 'Show the warnings', 'Show the images']);
    shows[1]!.click();
    flushSync();
    const ev = target.querySelector(`#${shows[1]!.getAttribute('aria-controls')}`);
    expect(shows[1]!.getAttribute('aria-expanded')).toBe('true');
    expect(shows[1]!.textContent?.trim()).toBe('Hide the warnings');
    expect(ev?.closest('li')?.classList.contains('warning')).toBe(true);
    // Filtered: no severity control inside a finding.
    expect(ev?.querySelector('[role="radio"]')).toBeNull();
    shows[0]!.click();
    flushSync();
    expect(target.querySelectorAll('.evidence').length).toBe(1);
    expect(shows[1]!.getAttribute('aria-expanded')).toBe('false');
    await tick();
  });

  it('rests on one line with each finding once, units named, and no second accounting', async () => {
    const dataset = await hub.api.dataset('ds_shapes_hard');
    const counts = (await hub.api.datasetIssues('ds_shapes_hard', { limit: 1 })).counts;
    c = mount(StatsTab, { target, props: { dataset, stats: dataset.stats!, counts, onclass: () => {} } });
    flushSync();
    const items = [...target.querySelectorAll('#ds-health summary .h-sev')].map((e) => e.textContent?.replace(/\s+/g, ' ').trim());
    // shapes-hard's 3 check errors are all label errors (broken_label, class_overflow, out_of_range_box): the 3 broken files.
    expect(items).toEqual(['Error:3 broken label files', 'Warning:3 check warnings', 'Warning:35% tiny boxes', 'Note:42 empty images (0.8%)']);
    const text = target.querySelector('#ds-health')?.textContent ?? '';
    expect(text).not.toMatch(/categor|dataset-wide|file issues/);
    expect(target.querySelector('#ds-health details')?.hasAttribute('open')).toBe(false);
  });

  it('leaves Health out when there is nothing to report', async () => {
    const dataset = await hub.api.dataset('ds_shapes_hard');
    const stats = { ...dataset.stats!, broken_labels: 0, empty_images: 0, class_counts: dataset.stats!.class_counts.map(() => 1000), box_size_hist: { edges: [0, 0.05, 1], counts: [0, 6000] } };
    c = mount(StatsTab, { target, props: { dataset, stats, onclass: () => {} } });
    flushSync();
    expect(target.querySelector('#ds-health')).toBeNull();
    expect(target.querySelector('.kpis')).not.toBeNull();
  });
});

describe('Dataset stats: Health heading', () => {
  it('sums the findings up at rest and, open, is a plain heading with their count instead of repeating them', async () => {
    const dataset = await hub.api.dataset('ds_shapes_hard');
    const counts = (await hub.api.datasetIssues('ds_shapes_hard', { limit: 1 })).counts;
    c = mount(StatsTab, { target, props: { dataset, stats: dataset.stats!, counts, onclass: () => {} } });
    flushSync();
    const summary = target.querySelector<HTMLElement>('#ds-health summary')!;
    expect(summary.querySelectorAll('.h-sev').length).toBeGreaterThan(0);
    expect(summary.querySelector('.h-count')).toBeNull();
    const details = target.querySelector<HTMLDetailsElement>('#ds-health details')!;
    details.open = true;
    details.dispatchEvent(new Event('toggle'));
    flushSync();
    const n = target.querySelectorAll('.hitems > li').length;
    expect(summary.querySelector('.h-count')?.textContent).toBe(`${n} findings`);
    expect(summary.querySelectorAll('.h-sev').length).toBe(0);
  });
});
