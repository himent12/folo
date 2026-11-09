// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { router, type NavigationGuard } from './router.svelte';

describe('router guards', () => {
  let remove: (() => void) | undefined;
  beforeEach(() => {
    history.replaceState(null, '', '#/datasets/label/lb_1?image=3');
    window.dispatchEvent(new HashChangeEvent('hashchange'));
  });
  afterEach(() => remove?.());

  /** A guard that holds every navigation and keeps the last one's `proceed`. */
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
    router.go('/datasets');
    expect(router.route.path).toBe('/datasets/label/lb_1');
    expect(location.hash).toBe('#/datasets/label/lb_1?image=3');
    expect(held.map((h) => h.to)).toEqual(['/datasets']);
    held[0]!.proceed();
    expect(router.route.path).toBe('/datasets');
    expect(location.hash).toBe('#/datasets');
  });

  it('holds a replace too (another ?image= of the same page)', () => {
    const held = hold();
    router.go('/datasets/label/lb_1?image=4', { replace: true });
    expect(router.route.query.get('image')).toBe('3');
    held[0]!.proceed();
    expect(router.route.query.get('image')).toBe('4');
  });

  it('holds a link or Back (the URL already moved): puts the page back in the URL until it continues', () => {
    const held = hold();
    // What a link click or Back does: the URL changes, then hashchange.
    history.replaceState(null, '', '#/runs');
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    expect(router.route.path).toBe('/datasets/label/lb_1');
    expect(location.hash).toBe('#/datasets/label/lb_1?image=3');
    expect(held.map((h) => h.to)).toEqual(['/runs']);
    held[0]!.proceed();
    expect(router.route.path).toBe('/runs');
    expect(location.hash).toBe('#/runs');
  });

  it('lets everything through once the guard is removed, or while it says yes', () => {
    remove = router.guard(() => true);
    router.go('/models');
    expect(router.route.path).toBe('/models');
    remove();
    hold();
    remove?.();
    router.go('/runs');
    expect(router.route.path).toBe('/runs');
    history.replaceState(null, '', '#/settings');
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    expect(router.route.path).toBe('/settings');
  });
});
