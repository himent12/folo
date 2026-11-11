// @vitest-environment happy-dom
// Back / Forward return to where the user was (shell F3): the scroll of each history entry, applied once the screen is
// tall enough for it; a new navigation keeps nothing; a value a screen keeps with its entry comes back with it.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { router } from '$lib/router';
import { tabHistory, type TabHistory } from '$lib/tabHistory.testing';
import { entryState, installEntryMemory, restoringScroll } from './entryMemory';

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
