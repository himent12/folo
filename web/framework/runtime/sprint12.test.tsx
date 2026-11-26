// @vitest-environment happy-dom
// Sprint 12 regression tests for fw compiler + runtime:
// 1. Nested conditional / && branches preserve inner DOM nodes across truthy changes and never render "0" for falsy left operands.
// 2. MediaQuery invalidates cached computed() derivations when matchMedia changes while unwatched (initial or between unmount/remount).
// 3. spread() normalizes camelCase event handlers (onClick / onKeyDown / onScroll) identically to lowercase props.
import { describe, expect, it, vi } from 'vitest';
import { render } from '../testing';
import { computed, createRoot, effect, flush, signal } from './core';
import { MediaQuery } from './reactivity';

describe('fw Sprint 12 compiler & runtime hardening', () => {
  it('preserves inner JSX across truthy updates in nested conditional/&& expressions and never renders "0"', () => {
    const outer = signal(true);
    const count = signal(1);
    const staticZero = 0;
    let childMounts = 0;

    function Child() {
      childMounts++;
      return <span class="child">ok</span>;
    }

    const host = document.createElement('div');
    const c = render(
      () => (
        <div>
          <section class="nested-ternary">{outer.get() && (count.get() ? <Child /> : <em class="alt">no</em>)}</section>
          <section class="nested-and">{outer.get() ? count.get() && <b class="kept">yes</b> : <i class="off">off</i>}</section>
          <section class="static-zero">{staticZero && <u class="never">nope</u>}</section>
        </div>
      ),
      host,
    );
    flush();

    const firstChild = host.querySelector('.child');
    const firstKept = host.querySelector('.kept');
    expect(firstChild).not.toBeNull();
    expect(firstKept).not.toBeNull();
    expect(childMounts).toBe(1);
    expect(host.querySelector('.static-zero')?.textContent).toBe('');

    // Truthy -> truthy (1 -> 2) must keep the exact same DOM nodes without remounting Child.
    count.set(2);
    flush();
    expect(host.querySelector('.child')).toBe(firstChild);
    expect(host.querySelector('.kept')).toBe(firstKept);
    expect(childMounts).toBe(1);

    // Falsy 0 must render the alternate / empty, never the literal text "0".
    count.set(0);
    flush();
    expect(host.querySelector('.nested-ternary')?.textContent).toBe('no');
    expect(host.querySelector('.nested-and')?.textContent).toBe('');

    c.unmount();
  });

  it('MediaQuery updates cached computed() when matchMedia changes while unwatched', () => {
    let matches = false;
    const listeners = new Set<() => void>();
    const fakeMql = {
      get matches() {
        return matches;
      },
      media: '(max-width: 600px)',
      addEventListener: (_: string, fn: () => void) => listeners.add(fn),
      removeEventListener: (_: string, fn: () => void) => listeners.delete(fn),
    };
    const spy = vi.spyOn(window, 'matchMedia').mockImplementation(() => fakeMql as unknown as MediaQueryList);
    try {
      const mq = new MediaQuery('(max-width: 600px)', false);
      const label = computed(() => (mq.current ? 'phone' : 'desktop'));

      // Change before first reactive subscription: computed must observe the live match, not stale constructor snapshot.
      matches = true;
      const seen1: string[] = [];
      const stop1 = createRoot((dispose) => {
        effect(() => {
          seen1.push(label.get());
        });
        return dispose;
      });
      flush();
      expect(seen1).toEqual(['phone']);
      expect(listeners.size).toBe(1);
      stop1();
      expect(listeners.size).toBe(0);

      // Change while unwatched between unmount and remount: computed must recompute on re-subscribe.
      matches = false;
      const seen2: string[] = [];
      const stop2 = createRoot((dispose) => {
        effect(() => {
          seen2.push(label.get());
        });
        return dispose;
      });
      flush();
      expect(seen2).toEqual(['desktop']);

      // Live change while subscribed still fires immediately.
      matches = true;
      for (const fn of listeners) fn();
      flush();
      expect(seen2).toEqual(['desktop', 'phone']);

      // Change synchronously right before last subscriber unsubscribes (before `change` fires):
      // remounting must still observe the new value instead of caching stale `#last`.
      matches = false;
      stop2();
      expect(listeners.size).toBe(0);

      const seen3: string[] = [];
      const stop3 = createRoot((dispose) => {
        effect(() => {
          seen3.push(label.get());
        });
        return dispose;
      });
      flush();
      expect(seen3).toEqual(['desktop']);
      stop3();
    } finally {
      spy.mockRestore();
    }
  });

  it('spread normalizes camelCase DOM events without casing collisions and preserves custom event casing', () => {
    const host = document.createElement('div');
    document.body.appendChild(host);
    const clicks: string[] = [];
    const scrolls: number[] = [];
    const custom: string[] = [];
    const props = signal<Record<string, unknown>>({
      onclick: () => clicks.push('lower'),
      onScroll: () => scrolls.push(1),
      onValueChanged: () => custom.push('changed'),
    });

    const c = render(() => <button {...props.get()}>go</button>, host);
    flush();
    const el = host.querySelector('button')!;

    el.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    el.dispatchEvent(new Event('scroll'));
    el.dispatchEvent(new CustomEvent('ValueChanged'));
    expect(clicks).toEqual(['lower']);
    expect(scrolls).toEqual([1]);
    expect(custom).toEqual(['changed']);

    // Switching `{ onclick }` -> `{ onClick }` must NOT wipe the delegated click handler.
    props.set({ onClick: () => clicks.push('camel') });
    flush();
    el.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    el.dispatchEvent(new Event('scroll'));
    el.dispatchEvent(new CustomEvent('ValueChanged'));
    expect(clicks).toEqual(['lower', 'camel']);
    expect(scrolls).toEqual([1]);
    expect(custom).toEqual(['changed']);

    // Co-existing `{ onclick: local, onClick: ext }` then dropping `onClick` restores `local`.
    const local = () => clicks.push('local');
    props.set({ onclick: local, onClick: () => clicks.push('ext') });
    flush();
    el.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(clicks).toEqual(['lower', 'camel', 'ext']);

    props.set({ onclick: local });
    flush();
    el.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(clicks).toEqual(['lower', 'camel', 'ext', 'local']);

    c.unmount();
    el.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(clicks).toEqual(['lower', 'camel', 'ext', 'local']);
    host.remove();
  });
});
