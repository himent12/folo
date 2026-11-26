// @vitest-environment happy-dom
// Hardening tests for fw scheduler batching, deterministic disposal/cleanup, delegated multi-root lifecycles,
// keyed list SameValueZero/exception recovery, and single-node outro transitions.
import { describe, expect, it, vi } from 'vitest';
import {
  batch,
  computed,
  createRoot,
  dispose,
  effect,
  flush,
  getOwner,
  memo,
  onCleanup,
  onMount,
  renderEffect,
  runWithOwner,
  signal,
  watchedSignal,
  type Owner,
} from './core';
import { use } from './bind';
import { createComponent } from './component';
import { insert, setLeave } from './dom';
import { delegateEvents, delegateRoot, setDelegated } from './events';
import { For, Index, Show } from './flow';
import { deep } from './reactivity';
import { render as rawRender } from './render';
import { spread } from './spread';
import { transition } from './transition';
import { render } from '../testing';

describe('batch() scheduler hardening', () => {
  it('defers flush() inside batch and nested batches until the outermost batch completes', () => {
    const a = signal(1);
    const b = signal(10);
    const sum = computed(() => a.get() + b.get());
    const seen: number[] = [];

    const stop = createRoot((d) => {
      effect(() => void seen.push(sum.get()));
      return d;
    });
    flush();
    expect(seen).toEqual([11]);

    batch(() => {
      a.set(2);
      // Computeds pull fresh values synchronously inside batch, while flush() defers effects.
      expect(sum.get()).toBe(12);
      flush();
      expect(seen).toEqual([11]);
      batch(() => {
        b.set(20);
        flush();
        expect(seen).toEqual([11]);
      });
      // Still inside outer batch: effects must not see torn state yet.
      expect(seen).toEqual([11]);
      a.set(3);
    });

    // Outermost batch exits and runs the deferred flush once with the final state.
    expect(seen).toEqual([11, 23]);
    stop();
  });

  it('restores batchDepth on throw without masking the thrown error', () => {
    const s = signal(0);
    const seen: number[] = [];
    const stop = createRoot((d) => {
      effect(() => void seen.push(s.get()));
      return d;
    });
    flush();
    expect(seen).toEqual([0]);

    expect(() =>
      batch(() => {
        s.set(1);
        flush();
        throw new Error('batch-boom');
      }),
    ).toThrow('batch-boom');

    // Subsequent flushes work normally (batchDepth was restored to 0).
    flush();
    expect(seen).toEqual([0, 1]);
    stop();
  });
});

describe('deterministic owner disposal & cleanup', () => {
  it('unsubscribes every source and runs returned cleanup when an effect disposes its own root mid-run', () => {
    const subs = { a: 0, b: 0, c: 0 };
    const a = watchedSignal(1, (on) => (subs.a += on ? 1 : -1));
    const b = watchedSignal(2, (on) => (subs.b += on ? 1 : -1));
    const c = watchedSignal(3, (on) => (subs.c += on ? 1 : -1));
    const cleaned = vi.fn();
    const otherSeen: number[] = [];

    const stopOther = createRoot((d) => {
      effect(() => void otherSeen.push(a.get() + b.get() + c.get()));
      return d;
    });

    let stopSelf!: () => void;
    createRoot((d) => {
      stopSelf = d;
      effect(() => {
        const av = a.get();
        const bv = b.get();
        if (av === 2) {
          stopSelf();
          // Reading another signal and returning a cleanup after self-disposal must not leak.
          void c.get();
          return cleaned;
        }
        return () => void bv;
      });
    });

    flush();
    expect(subs).toEqual({ a: 1, b: 1, c: 1 });
    expect(otherSeen).toEqual([6]);

    // Trigger self-disposal mid-run.
    a.set(2);
    flush();
    expect(cleaned).toHaveBeenCalledTimes(1);

    // Disposing the surviving sibling confirms zero leaked subscriptions remain on a, b, or c.
    stopOther();
    expect(subs).toEqual({ a: 0, b: 0, c: 0 });

    // Source link pointers (`_node`) were cleanly restored: a fresh effect tracks all three normally.
    const fresh: number[] = [];
    const stopFresh = createRoot((d) => {
      effect(() => void fresh.push(a.get() + b.get() + c.get()));
      return d;
    });
    flush();
    expect(fresh).toEqual([7]);
    b.set(20);
    c.set(30);
    flush();
    expect(fresh).toEqual([7, 52]);
    stopFresh();
    expect(subs).toEqual({ a: 0, b: 0, c: 0 });
  });

  it('does not run an effect body if its own cleanup disposes the root before re-run', () => {
    const s = signal(0);
    const runs = vi.fn();
    let stop!: () => void;
    createRoot((d) => {
      stop = d;
      effect(() => {
        runs(s.get());
        return () => {
          stop();
        };
      });
    });
    flush();
    expect(runs).toHaveBeenCalledTimes(1);

    s.set(1);
    flush();
    expect(runs).toHaveBeenCalledTimes(1);
  });

  it('runs onCleanup immediately and skips zombie effects on an already-disposed owner', () => {
    let deadOwner: Owner | null = null;
    const stop = createRoot((d) => {
      deadOwner = getOwner();
      return d;
    });
    stop();

    const lateCleanup = vi.fn();
    const zombieEffect = vi.fn();
    const zombieRender = vi.fn();
    const s = signal(1);

    runWithOwner(deadOwner, () => {
      onCleanup(lateCleanup);
      effect(() => zombieEffect(s.get()));
      renderEffect(() => zombieRender(s.get()));
    });
    flush();
    s.set(2);
    flush();

    expect(lateCleanup).toHaveBeenCalledTimes(1);
    expect(zombieEffect).not.toHaveBeenCalled();
    expect(zombieRender).not.toHaveBeenCalled();
  });

  it('propagates DISPOSED through attach() so a dead memo rendering a child component leaks no effects or subscriptions', () => {
    let subs = 0;
    const s = watchedSignal(1, (on) => (subs += on ? 1 : -1));
    const dep = signal(0);
    const childRuns = vi.fn();
    const childMounted = vi.fn();

    const Child = () => {
      renderEffect(() => childRuns(s.get()));
      onMount(childMounted);
      return null;
    };

    let retained!: () => unknown;
    const stop = createRoot((d) => {
      retained = memo(() => {
        const v = dep.get();
        if (v > 0) createComponent(Child, {});
        return v;
      });
      return d;
    });
    expect(retained()).toBe(0);
    stop();

    // Mutate dependency and read retained memo after its parent root was disposed.
    dep.set(1);
    expect(retained()).toBe(1);
    flush();
    s.set(2);
    flush();

    expect(subs).toBe(0);
    expect(childRuns).not.toHaveBeenCalled();
    expect(childMounted).not.toHaveBeenCalled();
  });

  it('runs cleanups with no active owner so nested onCleanup/effect cannot attach to an ambient owner', () => {
    const leaked = vi.fn();
    let innerOwner: Owner | null = null;
    const outerStop = createRoot((d) => {
      createRoot((dInner) => {
        innerOwner = getOwner();
        onCleanup(() => {
          expect(getOwner()).toBeNull();
          onCleanup(leaked);
        });
        return dInner;
      }, null);
      // Disposing `innerOwner` while `outer` is the active owner must not attach `leaked` to `outer`.
      dispose(innerOwner!);
      return d;
    });
    outerStop();
    expect(leaked).not.toHaveBeenCalled();
  });

  it('disposes partial renderEffects and cleanups when a component or memo throws', () => {
    let watched = 0;
    const s = watchedSignal(1, (on) => (watched += on ? 1 : -1));
    const cleaned = vi.fn();
    const runs = vi.fn();

    const Failing = () => {
      onCleanup(cleaned);
      renderEffect(() => runs(s.get()));
      throw new Error('component-boom');
    };

    const stop = createRoot((d) => {
      expect(() => createComponent(Failing, {})).toThrow('component-boom');
      return d;
    });
    expect(cleaned).toHaveBeenCalledTimes(1);
    expect(runs).toHaveBeenCalledTimes(1);
    expect(watched).toBe(0);

    s.set(2);
    flush();
    expect(runs).toHaveBeenCalledTimes(1);

    // Same for a throwing memo branch.
    const memoCleaned = vi.fn();
    const memoRuns = vi.fn();
    const gate = signal(true);
    const stop2 = createRoot((d) => {
      const m = memo(() => {
        onCleanup(memoCleaned);
        renderEffect(() => memoRuns(s.get()));
        if (gate.get()) throw new Error('memo-boom');
        return 'ok';
      });
      expect(() => m()).toThrow('memo-boom');
      return d;
    });
    expect(memoCleaned).toHaveBeenCalledTimes(1);
    expect(watched).toBe(0);
    s.set(3);
    flush();
    expect(memoRuns).toHaveBeenCalledTimes(1);

    stop();
    stop2();
  });
});

describe('multi-root delegated events & render() cleanup', () => {
  it('keeps delegated events and late-registered types working when co-mounted roots unmount out of order or twice', () => {
    const container = document.body.appendChild(document.createElement('div'));
    let clicksA = 0;
    let clicksB = 0;
    let dblClicksA = 0;

    const a = render(() => <button id="a" onclick={() => clicksA++}>A</button>, container);
    const b = render(() => <button id="b" onclick={() => clicksB++}>B</button>, container);

    container.querySelector<HTMLButtonElement>('#a')!.click();
    container.querySelector<HTMLButtonElement>('#b')!.click();
    expect(clicksA).toBe(1);
    expect(clicksB).toBe(1);

    // Unmount `b` twice (idempotency): must not strip `container` or `document` listeners from `a`.
    b.unmount();
    b.unmount();

    const btnA = container.querySelector<HTMLButtonElement>('#a')!;
    btnA.click();
    expect(clicksA).toBe(2);

    // Register a brand-new delegated event type AFTER `b` unmounted: `container` must still receive it for `a`.
    setDelegated(btnA, 'dblclick', () => dblClicksA++);
    delegateEvents(['dblclick']);
    btnA.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    expect(dblClicksA).toBe(1);

    a.unmount();
    container.remove();
  });

  it('keeps delegateRoot undo strictly idempotent across sibling roots', () => {
    const r1 = document.body.appendChild(document.createElement('div'));
    const r2 = document.body.appendChild(document.createElement('div'));
    const undo1 = delegateRoot(r1);
    const undo2 = delegateRoot(r2);
    let hits = 0;
    const btn = r2.appendChild(document.createElement('button'));
    setDelegated(btn, 'click', () => hits++);

    // Calling undo1 repeatedly must never decrement document's listener refcount below r2's share.
    undo1();
    undo1();
    undo1();

    btn.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(hits).toBe(1);

    undo2();
    r1.remove();
    r2.remove();
  });

  it('removes anchor nodes when initial render throws', () => {
    const container = document.createElement('div');
    expect(() =>
      rawRender(() => {
        throw new Error('render-boom');
      }, container),
    ).toThrow('render-boom');
    expect(container.childNodes.length).toBe(0);
  });
});

describe('For / Index keyed diff & single-node outro hardening', () => {
  it('preserves row identity for NaN and -0/+0 keys across prefix, suffix, swap, and rotate paths', () => {
    const items = signal<{ k: number; label: string }[]>([
      { k: Number.NaN, label: 'nan-0' },
      { k: -0, label: 'zero-0' },
      { k: 1, label: 'one-0' },
    ]);
    let created = 0;
    const { container, unmount } = render(() => (
      <ul>
        <For each={items.get()} key={(x) => x.k}>
          {(item) => {
            created++;
            return <li>{item().label}</li>;
          }}
        </For>
      </ul>
    ));
    flush();
    const [liNan, liZero, liOne] = [...container.querySelectorAll('li')];
    expect(created).toBe(3);

    // Same keys with updated labels (prefix fast path): NaN and +0 match NaN and -0 without recreating DOM nodes.
    items.set([
      { k: Number.NaN, label: 'nan-1' },
      { k: +0, label: 'zero-1' },
      { k: 1, label: 'one-1' },
    ]);
    flush();
    expect(created).toBe(3);
    expect([...container.querySelectorAll('li')]).toEqual([liNan, liZero, liOne]);
    expect(container.textContent).toBe('nan-1zero-1one-1');

    // End-for-end swap with NaN key.
    items.set([
      { k: 1, label: 'one-2' },
      { k: 0, label: 'zero-2' },
      { k: Number.NaN, label: 'nan-2' },
    ]);
    flush();
    expect(created).toBe(3);
    expect([...container.querySelectorAll('li')]).toEqual([liOne, liZero, liNan]);

    // Rotate right with NaN key.
    items.set([
      { k: Number.NaN, label: 'nan-3' },
      { k: 1, label: 'one-3' },
      { k: -0, label: 'zero-3' },
    ]);
    flush();
    expect(created).toBe(3);
    expect([...container.querySelectorAll('li')]).toEqual([liNan, liOne, liZero]);

    unmount();
  });

  it('disposes partially built rows on throw in For and Index and recovers cleanly on next update', () => {
    const cleanups: string[] = [];
    const list = signal<string[]>(['a', 'b']);
    let failOn = 'bad';

    const view = render(() => (
      <div>
        <For each={list.get()} key={(x) => x}>
          {(item) => {
            const v = item();
            onCleanup(() => cleanups.push(v));
            if (v === failOn) throw new Error(`row:${v}`);
            return <span>{v}</span>;
          }}
        </For>
      </div>
    ));
    flush();
    expect(view.container.textContent).toBe('ab');

    // Middle reconcile that drops 'a', creates 'x', then throws on 'bad': 'x' must be disposed immediately.
    list.set(['x', 'bad', 'b']);
    expect(() => flush()).toThrow('row:bad');
    expect(cleanups).toContain('a');
    expect(cleanups).toContain('x');

    // Recovery with valid list works cleanly without touching already-disposed 'a'.
    failOn = '';
    list.set(['b', 'c']);
    flush();
    expect(view.container.textContent).toBe('bc');
    view.unmount();
    expect(cleanups.filter((x) => x === 'a')).toHaveLength(1);
    expect(cleanups.filter((x) => x === 'x')).toHaveLength(1);
    expect(cleanups).toContain('b');
    expect(cleanups).toContain('c');

    // Same for Index multi-row grow throw: earlier row in failed batch is disposed and never mounts.
    const idxCleanups: number[] = [];
    const idxMounts: number[] = [];
    const idxList = signal<string[]>(['ok']);
    const idxView = render(() => (
      <div>
        <Index each={idxList.get()}>
          {(item, i) => {
            onCleanup(() => idxCleanups.push(i));
            onMount(() => void idxMounts.push(i));
            if (item() === 'boom') throw new Error('idx-boom');
            return <span>{item()}</span>;
          }}
        </Index>
      </div>
    ));
    flush();
    expect(idxMounts).toEqual([0]);
    idxList.set(['ok', 'partial', 'boom']);
    expect(() => flush()).toThrow('idx-boom');
    expect(idxCleanups).toEqual([2, 1]);
    expect(idxMounts).toEqual([0]);
    idxList.set(['ok', 'good']);
    flush();
    expect(idxView.container.textContent).toBe('okgood');
    expect(idxMounts).toEqual([0, 1]);
    idxView.unmount();
  });

  it('recovers from throwing row cleanups in For middle reconcile and failed multi-row appends', () => {
    // 1. Throwing row cleanup during middle reconcile must compact `rows` (no `undefined` holes).
    const list = signal<string[]>(['a', 'b', 'c']);
    let throwOnA = true;
    const view = render(() => (
      <div>
        <For each={list.get()} key={(x) => x}>
          {(item) => {
            const v = item();
            onCleanup(() => {
              if (v === 'a' && throwOnA) {
                throwOnA = false;
                throw new Error('cleanup:a');
              }
            });
            return <span>{v}</span>;
          }}
        </For>
      </div>
    ));
    flush();
    expect(view.container.textContent).toBe('abc');

    list.set(['b', 'x', 'y']);
    expect(() => flush()).toThrow('cleanup:a');

    // Subsequent update must not crash on an `undefined` hole and must reconcile cleanly.
    list.set(['b', 'z']);
    flush();
    expect(view.container.textContent).toBe('bz');
    view.unmount();

    // 2. Failed multi-row pure append in For disposes earlier rows in that batch before user effects run.
    const mounted: string[] = [];
    const disposed: string[] = [];
    const appendList = signal<string[]>(['a']);
    const appendView = render(() => (
      <div>
        <For each={appendList.get()} key={(x) => x}>
          {(item) => {
            const v = item();
            onMount(() => void mounted.push(v));
            onCleanup(() => void disposed.push(v));
            if (v === 'bad') throw new Error('append-boom');
            return <span>{v}</span>;
          }}
        </For>
      </div>
    ));
    flush();
    expect(mounted).toEqual(['a']);

    appendList.set(['a', 'x', 'bad']);
    expect(() => flush()).toThrow('append-boom');
    expect(disposed).toEqual(['bad', 'x']);
    expect(mounted).toEqual(['a']);

    appendList.set(['a', 'ok']);
    flush();
    expect(appendView.container.textContent).toBe('aok');
    expect(mounted).toEqual(['a', 'ok']);
    appendView.unmount();
  });

  it('keeps a running single-node outro intact across consecutive text and element updates until done()', () => {
    const mode = signal<'outro' | 'text1' | 'text2' | 'elem1' | 'elem2'>('outro');
    let finishOutro: (() => void) | undefined;
    const leaveSpy = vi.fn((done: () => void) => {
      finishOutro = done;
    });

    const host = document.createElement('div');
    const stop = createRoot((d) => {
      insert(host, () => {
        const m = mode.get();
        if (m === 'outro') {
          const el = document.createElement('p');
          el.textContent = 'leaving';
          setLeave(el, leaveSpy);
          return el;
        }
        if (m === 'text1') return 'first';
        if (m === 'text2') return 'second';
        if (m === 'elem1') {
          const el = document.createElement('em');
          el.textContent = 'first-el';
          return el;
        }
        const el = document.createElement('strong');
        el.textContent = 'second-el';
        return el;
      });
      return d;
    });
    flush();
    expect(host.querySelector('p')?.textContent).toBe('leaving');

    // Replace single leaving node with "first", then update to "second" while outro is still running.
    mode.set('text1');
    flush();
    expect(leaveSpy).toHaveBeenCalledTimes(1);
    expect(host.querySelector('p')).not.toBeNull();
    expect(host.textContent).toContain('first');

    mode.set('text2');
    flush();
    expect(host.querySelector('p')).not.toBeNull();
    expect(host.textContent).toContain('second');
    expect(host.textContent).not.toContain('first');

    finishOutro!();
    expect(host.querySelector('p')).toBeNull();
    expect(host.textContent).toBe('second');

    // Mount leaving node again and replace with elem1, then elem2 while outro is still running.
    finishOutro = undefined;
    mode.set('outro');
    flush();
    expect(host.querySelector('p')).not.toBeNull();

    mode.set('elem1');
    flush();
    expect(leaveSpy).toHaveBeenCalledTimes(2);
    expect(host.querySelector('em')?.textContent).toBe('first-el');
    expect(host.querySelector('p')).not.toBeNull();

    mode.set('elem2');
    flush();
    expect(host.querySelector('p')).not.toBeNull();
    expect(host.querySelector('em')).toBeNull();
    expect(host.querySelector('strong')?.textContent).toBe('second-el');

    finishOutro!();
    expect(host.querySelector('p')).toBeNull();
    expect(host.innerHTML).toBe('<strong>second-el</strong>');

    stop();
  });
});

describe('runtime action, spread, transition, deep, and unmount hardening', () => {
  it('use:action records initial primitive param so an unchanged first reactive re-evaluation does not spuriously call update()', () => {
    const count = signal(0);
    const updates: unknown[] = [];
    const action = (_node: HTMLElement, init: string | null) => {
      expect(init).toBeNull();
      return {
        update: (v: string | null) => {
          updates.push(v);
        },
      };
    };
    const el = document.createElement('div');
    const stop = createRoot((d) => {
      use(el, action, () => (count.get() >= 2 ? 'capped' : null));
      return d;
    });
    flush();
    expect(updates).toEqual([]);

    // Dependency changes from 0 -> 1, param still evaluates to null: update() must NOT be called.
    count.set(1);
    flush();
    expect(updates).toEqual([]);

    // Dependency changes from 1 -> 2, param flips to 'capped': update() fires once.
    count.set(2);
    flush();
    expect(updates).toEqual(['capped']);
    stop();
  });

  it('spread with class: directives does not rewrite className when only unrelated props change', () => {
    const title = signal('a');
    const active = signal(true);
    const el = document.createElement('div');
    let classWrites = 0;
    const desc = Object.getOwnPropertyDescriptor(Element.prototype, 'className')!;
    Object.defineProperty(el, 'className', {
      configurable: true,
      get() {
        return desc.get!.call(this);
      },
      set(v: string) {
        classWrites++;
        desc.set!.call(this, v);
      },
    });
    const stop = createRoot((d) => {
      spread(el, () => ({ class: 'base', title: title.get() }), 'fw-test', () => ({ on: active.get() }));
      return d;
    });
    flush();
    expect(el.className).toBe('base fw-test on');
    const initialWrites = classWrites;

    // Unrelated prop changes: must not rewrite className.
    title.set('b');
    flush();
    expect(el.getAttribute('title')).toBe('b');
    expect(classWrites).toBe(initialWrites);

    // Directive change toggles class without rewriting className.
    active.set(false);
    flush();
    expect(el.classList.contains('on')).toBe(false);
    expect(classWrites).toBe(initialWrites);
    stop();
  });

  it('transition preserves CSS values containing colons and ignores aborted onfinish callbacks', async () => {
    const seenKeyframes: Keyframe[][] = [];
    const anims: { onfinish: (() => void) | null; cancel: ReturnType<typeof vi.fn>; currentTime: number; effect: unknown }[] = [];
    const el = document.createElement('div');
    (el as unknown as { animate: (k: Keyframe[]) => unknown }).animate = (k: Keyframe[]) => {
      seenKeyframes.push(k);
      const a = { onfinish: null as (() => void) | null, cancel: vi.fn(), currentTime: 0, effect: {} };
      anims.push(a);
      return a;
    };
    const onIntroEnd = vi.fn();
    el.addEventListener('introend', onIntroEnd);
    const show = signal(false);
    const view = render(() => (
      <Show when={show.get()}>
        {() => {
          transition(
            el,
            () => ({
              delay: 10,
              duration: 16,
              css: () => '; background-image: url(https://example.com/a.png); opacity: 0.5;',
            }),
            undefined,
            1,
          );
          return el;
        }}
      </Show>
    ));
    flush();
    show.set(true);
    flush();
    // Wait for intro microtask to start delay animation.
    await Promise.resolve();
    expect(anims).toHaveLength(1);
    const firstFinish = anims[0]!.onfinish!;
    // Finish delay animation -> builds real keyframes with colon preserved and skips empty `;` segments.
    firstFinish();
    expect(anims).toHaveLength(2);
    expect(seenKeyframes[1]![0]).toEqual({
      backgroundImage: 'url(https://example.com/a.png)',
      opacity: '0.5',
    });
    // Completing the second animation fires introend and aborts cleanly; re-invoking a stale onfinish does nothing.
    const secondFinish = anims[1]!.onfinish!;
    secondFinish();
    expect(onIntroEnd).toHaveBeenCalledTimes(1);
    secondFinish();
    expect(onIntroEnd).toHaveBeenCalledTimes(1);
    view.unmount();
  });

  it('deep() ignores deleting inherited properties and notifies truncated indices on string array length', () => {
    const obj = deep<{ a: number }>({ a: 1 });
    let objRuns = 0;
    const stopObj = createRoot((d) => {
      effect(() => {
        Object.keys(obj);
        objRuns++;
      });
      return d;
    });
    flush();
    expect(objRuns).toBe(1);
    // Deleting an inherited property that is not own must not trigger reactivity.
    Reflect.deleteProperty(obj, 'toString');
    flush();
    expect(objRuns).toBe(1);
    stopObj();

    const arr = deep(['x', 'y', 'z']);
    const seenSecond: (string | undefined)[] = [];
    const stopArr = createRoot((d) => {
      effect(() => {
        seenSecond.push(arr[1]);
      });
      return d;
    });
    flush();
    expect(seenSecond).toEqual(['y']);
    (arr as unknown as { length: unknown }).length = '1';
    flush();
    expect(seenSecond).toEqual(['y', undefined]);
    stopArr();
  });

  it('unmounting a render() root clears pending outro leave handlers so leaving counter does not leak', () => {
    const container = document.createElement('div');
    const unmount = rawRender(() => {
      const el = document.createElement('div');
      setLeave(el, () => {});
      return el;
    }, container);
    const leavingEl = container.querySelector('div') as (HTMLDivElement & { $$leave?: unknown }) | null;
    expect(leavingEl?.$$leave).toBeDefined();
    unmount();
    expect(leavingEl?.$$leave).toBeUndefined();
    expect(container.childNodes.length).toBe(0);
  });
});
