// @vitest-environment happy-dom
// URL query filter/sort sync through the real router (without history pollution), pending-navigation safety,
// Escape-return restoration, and one-click empty-state filter recovery + focus restoration across Datasets, Models, and Runs.
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
  view?.unmount();
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
  it('syncs Datasets filter, sort, and search through the real router without pushState, restores on Escape return, and recovers cleanly from empty states', async () => {
    const push = vi.spyOn(history, 'pushState');
    const go = vi.spyOn(router, 'go');

    mountRoute(Datasets, '/datasets?format=yolo&sort=images&q=fast');
    await settle();

    const search = target.querySelector<HTMLInputElement>('input[aria-label="Filter datasets"]')!;
    const sort = target.querySelector<HTMLSelectElement>('#datasets-sort')!;
    expect(search.value).toBe('fast');
    expect(sort.value).toBe('images');
    expect(target.querySelector('[role="radio"][aria-checked="true"]')?.textContent).toContain('YOLO');

    // Typing updates the real browser hash via replaceState (no history.pushState calls).
    typeInto(search, 'zebra');
    expect(go).toHaveBeenLastCalledWith('/datasets?format=yolo&sort=images&q=zebra', { replace: true });
    expect(location.hash).toBe('#/datasets?format=yolo&sort=images&q=zebra');
    expect(push).not.toHaveBeenCalled();

    // Empty state describes active filters; Clear filters resets, updates URL, announces politely, and restores focus.
    expect(target.querySelector('.empty')?.textContent).toContain('No YOLO datasets match "zebra".');
    const filterStatus = target.querySelector<HTMLElement>('[data-filter-status]')!;
    const liveNote = target.querySelector<HTMLElement>('[data-live-note]')!;
    expect(filterStatus.getAttribute('role')).toBe('status');
    expect(filterStatus.getAttribute('aria-live')).toBe('polite');
    expect(liveNote.getAttribute('role')).toBe('status');
    expect(liveNote.getAttribute('aria-live')).toBe('polite');

    const clear = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Clear filters')!;
    clear.focus();
    clear.click();
    flushSync();
    await Promise.resolve();
    expect(search.value).toBe('');
    expect(location.hash).toBe('#/datasets?sort=images');
    expect(filterStatus.textContent).toMatch(/^Showing all \d+ datasets\.$/);
    expect(liveNote.textContent).toBe('Cleared dataset filters.');
    expect(document.activeElement).toBe(search);

    // Real router navigation to a detail route, unmount (recording return state), and Escape-like bare `/datasets` restores filters.
    router.go('/datasets/ds_fasteners');
    view!.unmount();
    view = undefined;
    router.go('/datasets');
    view = render(() => <Datasets />, target);
    flushSync();
    await settle();
    expect(location.hash).toBe('#/datasets?sort=images');
    expect(target.querySelector<HTMLSelectElement>('#datasets-sort')!.value).toBe('images');

    // Back/Forward route change updates local controls reactively.
    router.go('/datasets?format=coco&q=shapes', { replace: true });
    flushSync();
    expect(target.querySelector<HTMLInputElement>('input[aria-label="Filter datasets"]')!.value).toBe('shapes');
    expect(target.querySelector('[role="radio"][aria-checked="true"]')?.textContent).toContain('COCO');
  });

  it('preserves ?add=folder when filtering, keeps filters when closing the dialog, and closes a URL-opened dialog when Back drops ?add=', async () => {
    mountRoute(Datasets, '/datasets?add=folder');
    await settle();
    const dialog = () => target.querySelector<HTMLDialogElement>('dialog')!;
    expect(dialog().open).toBe(true);

    const search = target.querySelector<HTMLInputElement>('input[aria-label="Filter datasets"]')!;
    typeInto(search, 'fast');
    expect(location.hash).toBe('#/datasets?add=folder&q=fast');

    // Back removing ?add= closes the URL-opened dialog while keeping the filter.
    router.go('/datasets?q=fast', { replace: true });
    flushSync();
    expect(dialog().open).toBe(false);
    expect(search.value).toBe('fast');

    // Reopening via URL, then closing via the dialog button drops ?add= while preserving ?q=fast.
    router.go('/datasets?add=folder&q=fast', { replace: true });
    flushSync();
    expect(dialog().open).toBe(true);
    const close = target.querySelector<HTMLButtonElement>('dialog button[aria-label="Close"]')!;
    close.click();
    flushSync();
    expect(dialog().open).toBe(false);
    expect(location.hash).toBe('#/datasets?q=fast');
  });

  it('does not overwrite a pending detail navigation if a filter changes while the next chunk is held', async () => {
    mountRoute(Datasets, '/datasets');
    await settle();
    const search = target.querySelector<HTMLInputElement>('input[aria-label="Filter datasets"]')!;

    const release = router.holdWhile((to) => to.name === 'dataset');
    try {
      router.go('/datasets/ds_fasteners');
      expect(router.location.name).toBe('dataset');
      expect(router.route.name).toBe('datasets');
      typeInto(search, 'fast');
      expect(router.location.name).toBe('dataset');
      expect(location.hash).toBe('#/datasets/ds_fasteners');
    } finally {
      release();
    }
  });

  it('syncs Models group, sort, and search through the real router, keeps desktop & phone unknown-group controls accessible, and defers empty flash while metadata loads', async () => {
    const push = vi.spyOn(history, 'pushState');

    // 1. Slow runs/datasets metadata on a deep-linked dataset group must hold loading instead of flashing "No models match".
    let resolveRuns!: (v: Awaited<ReturnType<MockApi['runs']>>) => void;
    const baseApi = new MockApi('normal', 0);
    const realRuns = await baseApi.runs();
    vi.spyOn(baseApi, 'runs').mockImplementation(() => new Promise((r) => (resolveRuns = r)));
    hub.api = baseApi;
    mountRoute(Models, '/models?group=ds_fasteners:val');
    await settle(6);
    expect(target.querySelector('.empty')).toBeNull();
    expect(target.querySelector('[data-filter-status]')?.textContent).toBe('');
    resolveRuns(realRuns);
    await settle(10);
    expect(target.querySelectorAll('table.models tbody tr[data-row]').length).toBeGreaterThan(0);

    // 2. Real router URL sync on sort/group/search without pushState.
    hub.api = new MockApi('normal', 0);
    mountRoute(Models, '/models?group=~published&sort=bytes&dir=asc&q=coco');
    await settle(18);
    push.mockClear();

    const search = target.querySelector<HTMLInputElement>('input[aria-label="Filter models"]')!;
    expect(search.value).toBe('coco');
    const sizeHead = [...target.querySelectorAll('table.models thead th')].find((th) => th.textContent?.trim() === 'Size')!;
    expect(sizeHead.getAttribute('aria-sort')).toBe('ascending');
    expect(target.querySelectorAll('table.models tbody tr[data-row]').length).toBe(1);

    typeInto(search, 'no-match-xyz');
    expect(location.hash).toBe('#/models?group=%7Epublished&q=no-match-xyz&sort=bytes&dir=asc');
    expect(push).not.toHaveBeenCalled();
    expect(target.querySelector('.empty')?.textContent).toContain('No models in Published weights match "no-match-xyz".');

    // Clear filters resets group & query while preserving sort, and restores focus to search.
    const clear = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Clear filters')!;
    clear.focus();
    clear.click();
    flushSync();
    await Promise.resolve();
    expect(search.value).toBe('');
    expect(location.hash).toBe('#/models?sort=bytes&dir=asc');
    expect(document.activeElement).toBe(search);

    // Desktop unknown ?group= keeps a tabbable radio option ("Unknown group") and clear empty summary.
    router.go('/models?group=ds_deleted:val', { replace: true });
    flushSync();
    expect(target.querySelector('.empty')?.textContent).toContain('No models match the selected group filter.');
    const activeDesktopRadio = target.querySelector<HTMLButtonElement>('[role="radiogroup"] button[role="radio"][tabindex="0"]');
    expect(activeDesktopRadio).not.toBeNull();
    expect(activeDesktopRadio!.textContent).toContain('Unknown group');

    // Escape-return from /models/:id restores the filter state.
    router.go('/models/m_fast_n');
    view!.unmount();
    view = undefined;
    router.go('/models');
    view = render(() => <Models />, target);
    flushSync();
    await settle(10);
    expect(location.hash).toBe('#/models?group=ds_deleted%3Aval');

    // Phone unknown ?group= shows "Unknown group · 0" in the native <select> instead of falsely showing "All".
    const win = window as unknown as { happyDOM?: { setViewport(v: { width: number }): void } };
    const prevWidth = window.innerWidth;
    win.happyDOM?.setViewport({ width: 390 });
    try {
      flushSync();
      await Promise.resolve();
      const select = target.querySelector<HTMLSelectElement>('#models-ds')!;
      expect(select.value).toBe('ds_deleted:val');
      expect(select.selectedOptions[0]?.textContent).toContain('Unknown group');
    } finally {
      win.happyDOM?.setViewport({ width: prevWidth });
    }
  });

  it('shows active filter summary on Runs empty state, announces politely, suppresses empty-workspace announcements, and restores focus + URL on Clear filters', async () => {
    // Empty workspace with a filtered URL must never announce "Showing all 0 runs.".
    hub.api = new MockApi('empty', 0);
    mountRoute(Runs, '/runs?status=failed');
    await settle(12);
    expect(target.querySelector('[data-filter-status]')?.textContent).toBe('');

    // Normal workspace: real router sync, polite status, clear-message text, and focus restoration.
    hub.api = new MockApi('normal', 0);
    const push = vi.spyOn(history, 'pushState');
    mountRoute(Runs, '/runs?status=failed&q=no-such-run');
    await settle(16);
    push.mockClear();

    const search = target.querySelector<HTMLInputElement>('input[aria-label="Filter runs"]')!;
    const filterStatus = target.querySelector<HTMLElement>('[data-filter-status]')!;
    const liveNote = target.querySelector<HTMLElement>('[data-live-note]')!;
    expect(filterStatus.getAttribute('role')).toBe('status');
    expect(filterStatus.getAttribute('aria-live')).toBe('polite');
    expect(liveNote.getAttribute('role')).toBe('status');
    expect(liveNote.getAttribute('aria-live')).toBe('polite');
    expect(search.value).toBe('no-such-run');
    expect(target.querySelector('.empty')?.textContent).toContain('No failed runs match "no-such-run".');

    const clear = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Clear filters')!;
    clear.focus();
    clear.click();
    flushSync();
    await Promise.resolve();
    expect(search.value).toBe('');
    expect(location.hash).toBe('#/runs');
    expect(push).not.toHaveBeenCalled();
    expect(filterStatus.textContent).toMatch(/^Showing all \d+ runs\.$/);
    expect(liveNote.textContent).toBe('Cleared run filters.');
    expect(document.activeElement).toBe(search);
  });
});
