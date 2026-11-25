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
  renderEffect,
  runWithOwner,
  signal,
  watchedSignal,
  type Owner,
} from './core';
import { createComponent } from './component';
import { insert, setLeave } from './dom';
import { delegateEvents, delegateRoot, setDelegated } from './events';
import { For, Index, Show } from './flow';
import { render as rawRender } from './render';
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

    // Same for Index row creation throw.
    const idxCleanups: number[] = [];
    const idxList = signal<string[]>(['ok']);
    const idxView = render(() => (
      <div>
        <Index each={idxList.get()}>
          {(item, i) => {
            onCleanup(() => idxCleanups.push(i));
            if (item() === 'boom') throw new Error('idx-boom');
            return <span>{item()}</span>;
          }}
        </Index>
      </div>
    ));
    flush();
    idxList.set(['ok', 'boom']);
    expect(() => flush()).toThrow('idx-boom');
    expect(idxCleanups).toEqual([1]);
    idxView.unmount();
  });

  it('runs single-node outro transitions when replaced by text or another single element without leaking leaving count', () => {
    const mode = signal<'outro' | 'text' | 'elem'>('outro');
    let finishOutro: (() => void) | undefined;
    const leaveSpy = vi.fn((done: () => void) => {
      finishOutro = done;
    });

    const host = document.createElement('div');
    const stop = createRoot((d) => {
      insert(
        host,
        <Show
          when={mode.get() === 'outro'}
          fallback={mode.get() === 'text' ? 'plain-text' : <em>replacement</em>}
        >
          {(() => {
            const el = document.createElement('p');
            el.textContent = 'leaving';
            setLeave(el, leaveSpy);
            return el;
          })()}
        </Show>,
      );
      return d;
    });
    flush();
    expect(host.querySelector('p')?.textContent).toBe('leaving');

    // Replace single leaving node with text: outro must fire and keep <p> in DOM until done().
    mode.set('text');
    flush();
    expect(leaveSpy).toHaveBeenCalledTimes(1);
    expect(host.querySelector('p')).not.toBeNull();
    expect(host.textContent).toContain('plain-text');
    finishOutro!();
    expect(host.querySelector('p')).toBeNull();
    expect(host.textContent).toBe('plain-text');

    // Mount leaving node again and replace with another single element.
    finishOutro = undefined;
    mode.set('outro');
    flush();
    expect(host.querySelector('p')).not.toBeNull();

    mode.set('elem');
    flush();
    expect(leaveSpy).toHaveBeenCalledTimes(2);
    expect(host.querySelector('em')?.textContent).toBe('replacement');
    expect(host.querySelector('p')).not.toBeNull();
    finishOutro!();
    expect(host.querySelector('p')).toBeNull();
    expect(host.innerHTML).toBe('<em>replacement</em>');

    stop();
  });
});
