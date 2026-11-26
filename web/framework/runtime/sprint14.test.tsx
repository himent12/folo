// @vitest-environment happy-dom
// Sprint 14 fw runtime regression tests: throwing-effect partial rollback, unsubscribed computed NOTIFIED hygiene,
// and detached/moved marker resilience in insert/cleanChildren/appendNodes.
import { computed, createRoot, effect, flush, onCleanup, signal } from 'fw';
import { insert } from './dom';
import { describe, expect, it } from 'vitest';

describe('fw runtime sprint 14 hardening', () => {
  it('rolls back partial onCleanup callbacks and child effects when an effect throws mid-execution, then recovers on next write', () => {
    const fail = signal(true);
    const childTick = signal(0);
    const events: string[] = [];

    const stop = createRoot((dispose) => {
      effect(() => {
        const shouldFail = fail.get();
        events.push(`outer:start:${shouldFail}`);
        onCleanup(() => events.push(`outer:cleanup:${shouldFail}`));
        effect(() => {
          events.push(`child:run:${shouldFail}:${childTick.get()}`);
          return () => events.push(`child:cleanup:${shouldFail}`);
        });
        if (shouldFail) throw new Error('outer boom');
        events.push('outer:done');
      });
      return dispose;
    });

    // First flush throws, but must roll back the partial cleanup and dispose the half-initialized child effect
    // BEFORE the child effect can run in the same cycle.
    expect(() => flush()).toThrow('outer boom');
    expect(events).toEqual(['outer:start:true', 'outer:cleanup:true']);

    // Mutating the child signal must NOT trigger any leaked child effect.
    childTick.set(1);
    flush();
    expect(events).toEqual(['outer:start:true', 'outer:cleanup:true']);

    // Recovering the outer signal re-runs the outer effect and mounts the child cleanly.
    fail.set(false);
    flush();
    expect(events).toEqual([
      'outer:start:true',
      'outer:cleanup:true',
      'outer:start:false',
      'outer:done',
      'child:run:false:1',
    ]);

    stop();
    expect(events).toEqual([
      'outer:start:true',
      'outer:cleanup:true',
      'outer:start:false',
      'outer:done',
      'child:run:false:1',
      'child:cleanup:false',
      'outer:cleanup:false',
    ]);
  });

  it('clears stale NOTIFIED flags when a non-owning computed loses its last subscriber before flush', () => {
    const s = signal(1);
    const doubled = computed(() => s.get() * 2);
    const seen: number[] = [];

    const stop1 = createRoot((dispose) => {
      effect(() => {
        seen.push(doubled.get());
      });
      return dispose;
    });
    flush();
    expect(seen).toEqual([2]);

    // Notify `doubled` while subscribed, then tear down its only subscriber before flushing.
    s.set(2);
    stop1();
    flush();

    // A new subscriber later sees the latest value AND receives subsequent updates normally.
    const seen2: number[] = [];
    const stop2 = createRoot((dispose) => {
      effect(() => {
        seen2.push(doubled.get());
      });
      return dispose;
    });
    flush();
    expect(seen2).toEqual([4]);

    s.set(3);
    flush();
    expect(seen2).toEqual([4, 6]);
    stop2();
  });

  it('handles detached or moved insert markers in appendNodes and cleanChildren without throwing NotFoundError', () => {
    // 1. appendNodes: initial array insert with a marker already detached from `parent` (threw NotFoundError previously).
    const hostAppend = document.createElement('div');
    const detachedMarker = document.createComment('detached');
    const first = document.createElement('span');
    first.textContent = 'appended';
    const stopAppend = createRoot((dispose) => {
      insert(hostAppend, () => [first], detachedMarker);
      return dispose;
    });
    expect(() => flush()).not.toThrow();
    expect(hostAppend.querySelector('span')?.textContent).toBe('appended');
    stopAppend();

    // 2. cleanChildren: replacing a region after both the prior node was removed and `marker` was reparented elsewhere.
    const hostA = document.createElement('div');
    const hostB = document.createElement('div');
    const marker = document.createComment('end');
    hostA.appendChild(marker);

    const val = signal<Node | string | null>(document.createElement('span'));
    const stop = createRoot((dispose) => {
      insert(hostA, () => val.get(), marker);
      return dispose;
    });
    flush();
    const initialSpan = hostA.querySelector('span')!;
    expect(initialSpan).toBeTruthy();

    // Externally detach the rendered node AND move `marker` to `hostB` so `cleanChildren` takes its `insertBefore(node, ref)` path.
    initialSpan.remove();
    hostB.appendChild(marker);

    expect(() => {
      val.set('replacement');
      flush();
    }).not.toThrow();
    expect(hostA.textContent).toBe('replacement');

    // Also verify null cleanup when `marker` is detached.
    marker.remove();
    expect(() => {
      val.set(null);
      flush();
    }).not.toThrow();

    stop();
  });
});
