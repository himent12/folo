// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { router, type NavigationGuard } from './router.svelte';
import { tabHistory, type TabHistory } from './tabHistory.testing';

const JOB = '/datasets/label/lb_1';

describe('router guards', () => {
  let tab: TabHistory;
  let remove: (() => void) | undefined;
  beforeEach(() => {
    tab = tabHistory();
    // Overview, then Datasets, then a labeling job with an image open.
    router.go('/', { replace: true });
    router.go('/datasets');
    router.go(`${JOB}?image=3`);
  });
  afterEach(() => {
    remove?.();
    vi.restoreAllMocks();
  });

  /** A guard that holds every navigation and keeps each one's `proceed`. */
  function hold() {
    const held: { to: string; proceed: () => void }[] = [];
    const guard: NavigationGuard = (to, proceed) => {
      held.push({ to: to.path, proceed });
      return false;
    };
    remove = router.guard(guard);
    return held;
  }

  it('holds router.go, and the page continues it later', () => {
    const held = hold();
    router.go('/runs');
    expect(router.route.path).toBe(JOB);
    expect(location.hash).toBe(`#${JOB}?image=3`);
    expect(held.map((h) => h.to)).toEqual(['/runs']);
    held[0]!.proceed();
    expect(router.route.path).toBe('/runs');
    expect(tab.hashes()).toEqual(['#/', '#/datasets', `#${JOB}?image=3`, '#/runs']);
  });

  it('holds a replace too (another ?image= of the same page)', () => {
    const held = hold();
    router.go(`${JOB}?image=4`, { replace: true });
    expect(router.route.query.get('image')).toBe('3');
    held[0]!.proceed();
    expect(router.route.query.get('image')).toBe('4');
    expect(tab.hashes()).toEqual(['#/', '#/datasets', `#${JOB}?image=4`]);
  });

  it('holds a followed link: back on the page until it continues, then the link is where the tab goes', async () => {
    const held = hold();
    tab.follow('#/runs');
    await tab.settle();
    expect(router.route.path).toBe(JOB);
    expect(location.hash).toBe(`#${JOB}?image=3`);
    expect(held.map((h) => h.to)).toEqual(['/runs']);
    held[0]!.proceed();
    await tab.settle();
    expect(router.route.path).toBe('/runs');
    expect(location.hash).toBe('#/runs');
    expect(tab.hashes()).toEqual(['#/', '#/datasets', `#${JOB}?image=3`, '#/runs']);
    expect(tab.at()).toBe(3);
  });

  it('holds Back without losing the history: dropped, the tab is where it was and Back still leads back', async () => {
    const held = hold();
    tab.back();
    await tab.settle();
    expect(held.map((h) => h.to)).toEqual(['/datasets']);
    expect(router.route.path).toBe(JOB);
    expect(location.hash).toBe(`#${JOB}?image=3`);
    // Nothing was overwritten: Datasets is still the entry before, and nothing was added.
    expect(tab.hashes()).toEqual(['#/', '#/datasets', `#${JOB}?image=3`]);
    expect(tab.at()).toBe(2);

    // The page lets go (its work saved): Back now goes through, and Forward comes back.
    remove?.();
    tab.back();
    await tab.settle();
    expect(router.route.path).toBe('/datasets');
    tab.forward();
    await tab.settle();
    expect(router.route.path).toBe(JOB);
    expect(router.route.query.get('image')).toBe('3');
  });

  it('continues a held Back by travelling there, two entries back too', async () => {
    const held = hold();
    history.go(-2);
    await tab.settle();
    expect(held.map((h) => h.to)).toEqual(['/']);
    expect(router.route.path).toBe(JOB);
    expect(tab.at()).toBe(2);
    held[0]!.proceed();
    await tab.settle();
    expect(router.route.path).toBe('/');
    expect(location.hash).toBe('#/');
    expect(tab.hashes()).toEqual(['#/', '#/datasets', `#${JOB}?image=3`]);
    expect(tab.at()).toBe(0);
    // The guard is asked again for what comes next, from where the tab now is.
    tab.forward();
    await tab.settle();
    expect(held.map((h) => h.to)).toEqual(['/', '/datasets']);
    expect(router.route.path).toBe('/');
    expect(tab.at()).toBe(0);
  });

  it('lets everything through once the guard is removed, or while it says yes', async () => {
    remove = router.guard(() => true);
    router.go('/models');
    expect(router.route.path).toBe('/models');
    remove();
    hold();
    remove?.();
    router.go('/runs');
    expect(router.route.path).toBe('/runs');
    tab.follow('#/settings');
    await tab.settle();
    expect(router.route.path).toBe('/settings');
    tab.back();
    await tab.settle();
    expect(router.route.path).toBe('/runs');
  });
});
