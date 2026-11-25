// @vitest-environment happy-dom
// URL query filter/sort sync (without history pollution) and one-click empty-state filter recovery + focus restoration
// across Datasets, Models, and Runs.
import { flush as flushSync, type Component } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import Datasets from './Datasets';
import Models from './Models';
import Runs from './Runs';

let target: HTMLElement;
let view: Rendered | undefined;

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  target = document.createElement('div');
  document.body.appendChild(target);
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
});

afterEach(() => {
  view?.unmount();
  view = undefined;
  target.remove();
  vi.restoreAllMocks();
});

function mountRoute<P extends object>(comp: Component<P>, path: string) {
  history.replaceState(history.state, '', `#${path}`);
  router.route = parse(`#${path}`);
  const Comp = comp as Component<Record<string, unknown>>;
  view = render(() => <Comp />, target);
  flushSync();
}

async function settle(rounds = 14) {
  for (let i = 0; i < rounds; i++) {
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
  }
}

function typeInto(input: HTMLInputElement, value: string) {
  input.value = value;
  input.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
}

describe('URL filter/sort sync and empty-state recovery', () => {
  it('syncs Datasets filter, sort, and search with the URL via replace without history pollution and recovers cleanly from empty states', async () => {
    const push = vi.spyOn(history, 'pushState');
    const go = vi.spyOn(router, 'go').mockImplementation((path) => {
      router.route = parse(`#${path}`);
    });

    mountRoute(Datasets, '/datasets?format=yolo&sort=images&q=fast');
    await settle();

    const search = target.querySelector<HTMLInputElement>('input[aria-label="Filter datasets"]')!;
    const sort = target.querySelector<HTMLSelectElement>('#datasets-sort')!;
    expect(search.value).toBe('fast');
    expect(sort.value).toBe('images');
    expect(target.querySelector('[role="radio"][aria-checked="true"]')?.textContent).toContain('YOLO');

    // Typing updates the URL with replace: true (no history.pushState calls).
    typeInto(search, 'zebra');
    expect(go).toHaveBeenLastCalledWith('/datasets?format=yolo&sort=images&q=zebra', { replace: true });
    expect(push).not.toHaveBeenCalled();

    // Empty state describes the exact active filters; Clear filters resets, updates URL, announces, and restores focus.
    expect(target.querySelector('.empty')?.textContent).toContain('No YOLO datasets match "zebra".');
    const clear = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Clear filters')!;
    clear.focus();
    clear.click();
    flushSync();
    await Promise.resolve();
    expect(search.value).toBe('');
    expect(go).toHaveBeenLastCalledWith('/datasets?sort=images', { replace: true });
    expect(target.querySelector('[data-filter-status]')?.textContent).toMatch(/^Showing all \d+ datasets\.$/);
    expect(document.activeElement).toBe(search);

    // Back/Forward route change updates local controls reactively.
    router.route = parse('#/datasets?format=coco&q=shapes');
    flushSync();
    expect(search.value).toBe('shapes');
    expect(target.querySelector('[role="radio"][aria-checked="true"]')?.textContent).toContain('COCO');
  });

  it('preserves ?add=folder on Datasets when filters change before the Add dialog closes', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation((path) => {
      router.route = parse(`#${path}`);
    });
    mountRoute(Datasets, '/datasets?add=folder');
    await settle();

    const search = target.querySelector<HTMLInputElement>('input[aria-label="Filter datasets"]')!;
    typeInto(search, 'fast');
    expect(go).toHaveBeenLastCalledWith('/datasets?add=folder&q=fast', { replace: true });
  });

  it('syncs Models group, sort, and search with the URL, handles unknown ?group= gracefully, and recovers focus on Clear filters', async () => {
    const push = vi.spyOn(history, 'pushState');
    const go = vi.spyOn(router, 'go').mockImplementation((path) => {
      router.route = parse(`#${path}`);
    });

    mountRoute(Models, '/models?group=~published&sort=bytes&dir=asc&q=coco');
    await settle(18);

    const search = target.querySelector<HTMLInputElement>('input[aria-label="Filter models"]')!;
    expect(search.value).toBe('coco');
    const sizeHead = [...target.querySelectorAll('table.models thead th')].find((th) => th.textContent?.trim() === 'Size')!;
    expect(sizeHead.getAttribute('aria-sort')).toBe('ascending');
    expect(target.querySelectorAll('table.models tbody tr[data-row]').length).toBe(1);

    // Changing search updates URL via replace without pushState.
    typeInto(search, 'no-match-xyz');
    expect(go).toHaveBeenLastCalledWith('/models?group=%7Epublished&q=no-match-xyz&sort=bytes&dir=asc', { replace: true });
    expect(push).not.toHaveBeenCalled();
    expect(target.querySelector('.empty')?.textContent).toContain('No models in Published weights match "no-match-xyz".');

    // Clear filters resets group & query while preserving sort, and restores focus to search.
    const clear = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Clear filters')!;
    clear.focus();
    clear.click();
    flushSync();
    await Promise.resolve();
    expect(search.value).toBe('');
    expect(go).toHaveBeenLastCalledWith('/models?sort=bytes&dir=asc', { replace: true });
    expect(document.activeElement).toBe(search);

    // Unknown group param from a stale link shows clear recovery summary and never pretends "All" is selected on phones.
    const win = window as unknown as { happyDOM?: { setViewport(v: { width: number }): void } };
    const prevWidth = window.innerWidth;
    win.happyDOM?.setViewport({ width: 390 });
    try {
      router.route = parse('#/models?group=ds_deleted:val');
      flushSync();
      await Promise.resolve();
      expect(target.querySelector('.empty')?.textContent).toContain('No models match the selected group filter.');
      const select = target.querySelector<HTMLSelectElement>('#models-ds')!;
      expect(select.value).toBe('ds_deleted:val');
      expect(select.selectedOptions[0]?.textContent).toContain('Unknown group');
    } finally {
      win.happyDOM?.setViewport({ width: prevWidth });
    }
  });

  it('shows active filter summary on Runs empty state and restores focus + URL on Clear filters', async () => {
    const push = vi.spyOn(history, 'pushState');
    const go = vi.spyOn(router, 'go').mockImplementation((path) => {
      router.route = parse(`#${path}`);
    });

    mountRoute(Runs, '/runs?status=failed&q=no-such-run');
    await settle(16);

    const search = target.querySelector<HTMLInputElement>('input[aria-label="Filter runs"]')!;
    expect(search.value).toBe('no-such-run');
    expect(target.querySelector('.empty')?.textContent).toContain('No failed runs match "no-such-run".');

    const clear = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Clear filters')!;
    clear.focus();
    clear.click();
    flushSync();
    await Promise.resolve();
    expect(search.value).toBe('');
    expect(go).toHaveBeenLastCalledWith('/runs', { replace: true });
    expect(push).not.toHaveBeenCalled();
    expect(target.querySelector('[data-filter-status]')?.textContent).toMatch(/^Showing all \d+ runs\.$/);
    expect(document.activeElement).toBe(search);
  });
});
