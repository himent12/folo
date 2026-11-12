// @vitest-environment happy-dom
// Back / Forward return to where the user was (shell F3): the scroll of each history entry, applied once the screen is
// tall enough for it; a new navigation keeps nothing; a value a screen keeps with its entry comes back with it.
import { signal } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { router } from '$lib/router';
import { tabHistory, type TabHistory } from '$lib/tabHistory.testing';
import { entryState, installEntryMemory, restoringScroll } from './entryMemory';
import { focusRoute } from './routeFocus';

describe('entryMemory', () => {
  let tab: TabHistory;
  let uninstall: () => void;
  let height = 5000;
  let y = 0;

  const scroll = (to: number) => {
    y = to;
    window.dispatchEvent(new Event('scroll'));
  };
  /** Lets the moves happen (the tab's traversals are timers here, faked like the restore's frames). */
  const settle = () => vi.advanceTimersByTimeAsync(1);
  /** Lets the restore's frames run. */
  const frames = (n: number) => vi.advanceTimersByTimeAsync(n * 16);

  beforeEach(() => {
    vi.useFakeTimers();
    sessionStorage.clear();
    tab = tabHistory();
    router.go('/', { replace: true });
    height = 5000;
    y = 0;
    vi.spyOn(window, 'scrollY', 'get').mockImplementation(() => y);
    vi.spyOn(window, 'scrollTo').mockImplementation(((_x: number, to: number) => scroll(Math.min(to, Math.max(0, height - window.innerHeight)))) as typeof window.scrollTo);
    vi.spyOn(document.documentElement, 'scrollHeight', 'get').mockImplementation(() => height);
    uninstall = installEntryMemory();
  });
  afterEach(() => {
    uninstall();
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it('brings an entry back to its scroll on Back and Forward, once the page is tall enough', async () => {
    router.go('/runs');
    scroll(1200);
    router.go('/runs/run_1');
    scroll(0);
    scroll(700);

    // Back to the list: its screen is short at first (loading), so the restore waits for the rows.
    height = 900;
    tab.back();
    await settle();
    expect(router.route.path).toBe('/runs');
    expect(restoringScroll()).toBe(true);
    height = 5000;
    await frames(2);
    expect(window.scrollY).toBe(1200);
    await frames(40);
    expect(restoringScroll()).toBe(false);

    tab.forward();
    await settle();
    await frames(40);
    expect(window.scrollY).toBe(700);
  });

  it("waits for the entry's own screen while the one before is held on display, and route focus keeps it (review R3)", async () => {
    router.go('/runs');
    scroll(600);
    router.go('/datasets/ds_1');
    scroll(0);

    // Back to Runs, whose chunk is loading: App holds the (tall, still) dataset page on display meanwhile.
    const loading = signal(true);
    const release = router.holdWhile(() => loading.get());
    tab.back();
    await settle();
    expect(router.route.path).toBe('/datasets/ds_1');
    await frames(60);
    expect([restoringScroll(), window.scrollY]).toEqual([true, 0]);

    // The chunk is there: Runs shows, short at first, then with its rows.
    loading.set(false);
    height = 900;
    expect(router.route.path).toBe('/runs');
    await frames(2);
    height = 5000;
    await frames(2);
    expect(window.scrollY).toBe(600);
    // App focuses the new screen's heading a frame later: that does not send it back to the top.
    focusRoute(document.body);
    expect(window.scrollY).toBe(600);
    await frames(40);
    expect([restoringScroll(), window.scrollY]).toEqual([false, 600]);
    release();
  });

  it('gives way to the user, and starts a new entry at a place an old one had with nothing kept', async () => {
    router.go('/runs');
    const runs = router.entry;
    scroll(1200);
    router.go('/models');
    tab.back();
    await settle();
    window.dispatchEvent(new Event('wheel'));
    expect(restoringScroll()).toBe(false);

    // Back on Runs, a new navigation takes the place Models had: it keeps nothing of Models.
    router.go('/models');
    scroll(400);
    tab.back();
    await settle();
    router.go('/settings');
    expect(router.entry).toBe(runs + 1);
    tab.back();
    await settle();
    tab.forward();
    await settle();
    await frames(40);
    expect(window.scrollY).toBe(0);
  });

  it('keeps no screen value from another server: its ids are not the one in use (review R4)', async () => {
    hub.mode = 'live';
    router.go('/runs');
    const picked = entryState<string[]>('runs.compare');
    picked.set(['run_1', 'run_2']);
    expect(picked.get()).toEqual(['run_1', 'run_2']);
    const url = vi.spyOn(hub, 'serverUrl', 'get').mockReturnValue('http://other:7878');
    expect(picked.get()).toBeUndefined();
    picked.set(['run_9']);
    url.mockRestore();
    expect(picked.get()).toBeUndefined();
    hub.mode = 'mock';
    expect(picked.get()).toBeUndefined();
  });

  it('keeps a screen value with its entry', async () => {
    router.go('/runs');
    const picked = entryState<string[]>('runs.compare');
    expect(picked.get()).toBeUndefined();
    picked.set(['run_1', 'run_2']);
    router.go('/runs/run_1');
    expect(picked.get()).toBeUndefined();
    tab.back();
    await settle();
    expect(picked.get()).toEqual(['run_1', 'run_2']);
  });
});
