// @vitest-environment happy-dom
// BenchEntry and BenchLoop async stop/remove actions: settling after unmount or a server switch (hub.epoch) must not
// mutate live state, show stale toasts, or hijack navigation.
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import BenchEntryPage from '../BenchEntry';
import BenchLoopPage from '../BenchLoop';

let target: HTMLElement;
let view: Rendered | undefined;

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  toast.items = [];
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

async function until<T>(fn: () => T | null | undefined | false, what: string): Promise<T> {
  return vi.waitFor(() => {
    flushSync();
    const v = fn();
    if (!v) throw new Error(`waiting for ${what}`);
    return v;
  });
}

describe('BenchEntry / BenchLoop stale async stop and delete guards', () => {
  it('ignores a deleteBenchEntry completion after a server switch (hub.epoch change)', async () => {
    const api = hub.api as MockApi;
    const benches = await api.benches();
    let benchId = '';
    let doneEntryId = '';
    for (const b of benches) {
      const entries = await api.benchEntries(b.id);
      const found = entries.find((e) => e.status === 'done' || e.status === 'stopped' || e.status === 'failed');
      if (found) {
        benchId = b.id;
        doneEntryId = found.id;
        break;
      }
    }
    expect(doneEntryId).toBeTruthy();

    let resolveDel!: () => void;
    vi.spyOn(api, 'deleteBenchEntry').mockImplementation(
      () =>
        new Promise<void>((r) => {
          resolveDel = r;
        }),
    );

    const path = `/benchmarks/${benchId}/entries/${doneEntryId}`;
    history.replaceState(history.state, '', `#${path}`);
    router.route = parse(`#${path}`);
    view = render(() => <BenchEntryPage />, target);
    flushSync();

    // Open overflow menu -> Delete entry -> confirm delete.
    const trigger = await until(() => target.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"]'), 'entry menu');
    trigger.click();
    flushSync();
    const delItem = await until(
      () => [...target.querySelectorAll<HTMLButtonElement>('[role="menuitem"]')].find((b) => b.textContent?.includes('Delete entry')),
      'Delete entry item',
    );
    delItem.click();
    flushSync();
    const confirmBtn = await until(
      () => [...document.querySelectorAll<HTMLButtonElement>('dialog[open] button')].find((b) => b.textContent?.trim() === 'Delete entry'),
      'confirm delete button',
    );
    confirmBtn.click();
    flushSync();

    // Server switches while staying on the same route (same ID could exist on the new server): dialog must close & reset.
    hub.epoch++;
    flushSync();
    expect(document.querySelector('dialog[open]')).toBeNull();
    resolveDel();
    await new Promise((r) => setTimeout(r, 10));
    flushSync();

    expect(toast.items).toEqual([]);
    expect(document.querySelector('dialog[open]')).toBeNull();
  });

  it('refuses deleteBenchEntry when submitted while lazy navigation is already heading elsewhere', async () => {
    const api = hub.api as MockApi;
    const benches = await api.benches();
    let benchId = '';
    let doneEntryId = '';
    for (const b of benches) {
      const entries = await api.benchEntries(b.id);
      const found = entries.find((e) => e.status === 'done' || e.status === 'stopped' || e.status === 'failed');
      if (found) {
        benchId = b.id;
        doneEntryId = found.id;
        break;
      }
    }
    const delSpy = vi.spyOn(api, 'deleteBenchEntry').mockResolvedValue(undefined);

    const path = `/benchmarks/${benchId}/entries/${doneEntryId}`;
    history.replaceState(history.state, '', `#${path}`);
    router.route = parse(`#${path}`);
    view = render(() => <BenchEntryPage />, target);
    flushSync();

    const trigger = await until(() => target.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"]'), 'entry menu');
    trigger.click();
    flushSync();
    const delItem = await until(
      () => [...target.querySelectorAll<HTMLButtonElement>('[role="menuitem"]')].find((b) => b.textContent?.includes('Delete entry')),
      'Delete entry item',
    );
    delItem.click();
    flushSync();
    const confirmBtn = await until(
      () => [...document.querySelectorAll<HTMLButtonElement>('dialog[open] button')].find((b) => b.textContent?.trim() === 'Delete entry'),
      'confirm delete button',
    );

    const hold = router.holdWhile((to) => to.name === 'runs');
    try {
      router.go('/runs');
      expect(router.location.path).toBe('/runs');
      expect(router.route.path).toBe(path);
      confirmBtn.click();
      await new Promise((r) => setTimeout(r, 10));
      flushSync();
      expect(delSpy).not.toHaveBeenCalled();
      expect(router.location.path).toBe('/runs');
    } finally {
      hold();
    }
  });

  it('ignores a deleteBenchLoop completion after a server switch (hub.epoch change)', async () => {
    const api = hub.api as MockApi;
    const benches = await api.benches();
    let benchId = '';
    let doneLoopId = '';
    for (const b of benches) {
      const loops = await api.benchLoops(b.id);
      const found = loops.find((l) => l.status === 'done' || l.status === 'stopped' || l.status === 'failed');
      if (found) {
        benchId = b.id;
        doneLoopId = found.id;
        break;
      }
    }
    expect(doneLoopId).toBeTruthy();

    let resolveDel!: () => void;
    vi.spyOn(api, 'deleteBenchLoop').mockImplementation(
      () =>
        new Promise<void>((r) => {
          resolveDel = r;
        }),
    );

    const path = `/benchmarks/${benchId}/loops/${doneLoopId}`;
    history.replaceState(history.state, '', `#${path}`);
    router.route = parse(`#${path}`);
    view = render(() => <BenchLoopPage />, target);
    flushSync();

    const trigger = await until(() => target.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"]'), 'loop menu');
    trigger.click();
    flushSync();
    const delItem = await until(
      () => [...target.querySelectorAll<HTMLButtonElement>('[role="menuitem"]')].find((b) => b.textContent?.includes('Delete loop')),
      'Delete loop item',
    );
    delItem.click();
    flushSync();
    const confirmBtn = await until(
      () => [...document.querySelectorAll<HTMLButtonElement>('dialog[open] button')].find((b) => b.textContent?.trim() === 'Delete loop'),
      'confirm delete loop button',
    );
    confirmBtn.click();
    flushSync();

    // Server switches while staying on the same route (same ID could exist on the new server): dialog must close & reset.
    hub.epoch++;
    flushSync();
    expect(document.querySelector('dialog[open]')).toBeNull();
    resolveDel();
    await new Promise((r) => setTimeout(r, 10));
    flushSync();

    expect(toast.items).toEqual([]);
    expect(document.querySelector('dialog[open]')).toBeNull();
  });

  it('refuses deleteBenchLoop when submitted while lazy navigation is already heading elsewhere', async () => {
    const api = hub.api as MockApi;
    const benches = await api.benches();
    let benchId = '';
    let doneLoopId = '';
    for (const b of benches) {
      const loops = await api.benchLoops(b.id);
      const found = loops.find((l) => l.status === 'done' || l.status === 'stopped' || l.status === 'failed');
      if (found) {
        benchId = b.id;
        doneLoopId = found.id;
        break;
      }
    }
    const delSpy = vi.spyOn(api, 'deleteBenchLoop').mockResolvedValue(undefined);

    const path = `/benchmarks/${benchId}/loops/${doneLoopId}`;
    history.replaceState(history.state, '', `#${path}`);
    router.route = parse(`#${path}`);
    view = render(() => <BenchLoopPage />, target);
    flushSync();

    const trigger = await until(() => target.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"]'), 'loop menu');
    trigger.click();
    flushSync();
    const delItem = await until(
      () => [...target.querySelectorAll<HTMLButtonElement>('[role="menuitem"]')].find((b) => b.textContent?.includes('Delete loop')),
      'Delete loop item',
    );
    delItem.click();
    flushSync();
    const confirmBtn = await until(
      () => [...document.querySelectorAll<HTMLButtonElement>('dialog[open] button')].find((b) => b.textContent?.trim() === 'Delete loop'),
      'confirm delete loop button',
    );

    const hold = router.holdWhile((to) => to.name === 'runs');
    try {
      router.go('/runs');
      expect(router.location.path).toBe('/runs');
      expect(router.route.path).toBe(path);
      confirmBtn.click();
      await new Promise((r) => setTimeout(r, 10));
      flushSync();
      expect(delSpy).not.toHaveBeenCalled();
      expect(router.location.path).toBe('/runs');
    } finally {
      hold();
    }
  });
});
