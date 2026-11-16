// @vitest-environment happy-dom
import { flush as flushSync, tick } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import StatsTab from './StatsTab';

/** Port of `mount(C, { target, props })`: renders into `target`; `unmount(c)` is `c.unmount()`. */
const mount = <P extends object>(C: (p: P) => JSX.Element, o: { target: HTMLElement; props: P }) => render(() => C(o.props), o.target);
const unmount = (c: { unmount(): void }) => c.unmount();

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
    expect(shows).toHaveLength(3);
    shows[1]!.click();
    flushSync();
    const ev = target.querySelector(`#${shows[1]!.getAttribute('aria-controls')}`);
    expect(shows[1]!.getAttribute('aria-expanded')).toBe('true');
    // A full-width row of its own, right after its finding (the grid packs the next finding beside that one).
    expect(ev?.matches('li.evidence')).toBe(true);
    expect(ev?.previousElementSibling?.classList.contains('warning')).toBe(true);

    // Filtered: no severity control inside a finding.
    expect(ev?.querySelector('[role="radio"]')).toBeNull();
    shows[0]!.click();
    flushSync();
    expect(target.querySelectorAll('.evidence').length).toBe(1);
    expect(shows[1]!.getAttribute('aria-expanded')).toBe('false');
    await tick();
  });

  it('never repeats a count Health lists: the strip gives those tiles to others, and keeps them as the all-clear', async () => {
    const dataset = await hub.api.dataset('ds_shapes_hard');
    const counts = (await hub.api.datasetIssues('ds_shapes_hard', { limit: 1 })).counts;
    c = mount(StatsTab, { target, props: { dataset, stats: dataset.stats!, counts, onclass: () => {} } });
    flushSync();
    const labels = [...target.querySelectorAll('.kpis .k-label')].map((e) => e.textContent);
    expect(labels).toEqual(['Images', 'Boxes', 'Boxes per image', 'Classes']);
    const strip = target.querySelector('.kpis')!.textContent!;
    // Health says "42 empty images (0.8%)" and "3 images with broken labels"; the strip says neither.
    expect(strip).not.toMatch(/\b42\b|0\.8%|broken/);
    // Health lists neither: the zero tiles stay.
    unmount(c);
    const stats = { ...dataset.stats!, broken_labels: 0, empty_images: 0 };
    c = mount(StatsTab, { target, props: { dataset, stats, onclass: () => {} } });
    flushSync();
    expect([...target.querySelectorAll('.kpis .k-label')].map((e) => e.textContent)).toEqual(['Images', 'Boxes', 'Empty images', 'Broken labels']);
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
