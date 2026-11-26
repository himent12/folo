// @vitest-environment happy-dom
// Sprint 13 regression tests for fw runtime hardening:
// 1. Show, Switch, For, and Index keep direct reactive fallback expressions live while falsy/unmatched/empty.
// 2. For preserves duplicate-key FIFO order on suffix trims, duplicate insertions, and end-for-end swaps.
// 3. Portal tolerates synchronous mid-normalize self-disposal without throwing NotFoundError or leaking DOM nodes.
import { describe, expect, it } from 'vitest';
import { render } from '../testing';
import { createRoot, effect, flush, onCleanup, signal } from './core';
import { For, Index, Match, Portal, Show, Switch } from './flow';

describe('fw Sprint 13 runtime hardening', () => {
  it('updates direct reactive primitive and conditional fallbacks in Show and Switch while falsy/unmatched', () => {
    const ok = signal(false);
    const mode = signal<'a' | 'none'>('none');
    const msg = signal('loading');
    const alt = signal(false);
    let showFbDisposed = 0;
    let switchFbDisposed = 0;

    function ShowFb(props: { label: string }) {
      onCleanup(() => showFbDisposed++);
      return <span class="show-fb">{props.label}</span>;
    }

    function SwitchFb(props: { label: string }) {
      onCleanup(() => switchFbDisposed++);
      return <span class="switch-fb">{props.label}</span>;
    }

    const host = document.createElement('div');
    const view = render(
      () => (
        <div>
          <section class="show-prim">
            <Show when={ok.get()} fallback={msg.get()}>
              {() => <span class="show-main">main</span>}
            </Show>
          </section>
          <section class="show-cond">
            <Show when={ok.get()} fallback={alt.get() ? <ShowFb label="second" /> : <em class="show-first">first</em>}>
              {() => <span class="show-main2">main2</span>}
            </Show>
          </section>
          <section class="switch-prim">
            <Switch fallback={msg.get()}>
              <Match when={mode.get() === 'a'}>{() => <span class="switch-a">A</span>}</Match>
            </Switch>
          </section>
          <section class="switch-cond">
            <Switch fallback={alt.get() ? <SwitchFb label="s2" /> : <em class="switch-first">s1</em>}>
              <Match when={mode.get() === 'a'}>{() => <span class="switch-a2">A2</span>}</Match>
            </Switch>
          </section>
        </div>
      ),
      host,
    );
    flush();

    expect(host.querySelector('.show-prim')?.textContent).toBe('loading');
    expect(host.querySelector('.switch-prim')?.textContent).toBe('loading');
    expect(host.querySelector('.show-first')?.textContent).toBe('first');
    expect(host.querySelector('.switch-first')?.textContent).toBe('s1');

    // Updating signals read directly by fallback expressions updates live while falsy/unmatched.
    msg.set('still loading');
    alt.set(true);
    flush();
    expect(host.querySelector('.show-prim')?.textContent).toBe('still loading');
    expect(host.querySelector('.switch-prim')?.textContent).toBe('still loading');
    expect(host.querySelector('.show-fb')?.textContent).toBe('second');
    expect(host.querySelector('.switch-fb')?.textContent).toBe('s2');

    ok.set(true);
    mode.set('a');
    flush();
    expect(host.querySelector('.show-main')?.textContent).toBe('main');
    expect(host.querySelector('.switch-a')?.textContent).toBe('A');
    expect(showFbDisposed).toBeGreaterThanOrEqual(1);
    expect(switchFbDisposed).toBeGreaterThanOrEqual(1);

    view.unmount();
  });

  it('updates direct reactive fallbacks in For and Index while empty and recovers if fallback throws', () => {
    const items = signal<string[]>([]);
    const emptyText = signal('nothing yet');
    const boom = signal(true);

    const host = document.createElement('div');
    const view = render(
      () => (
        <div>
          <section class="for-box">
            <For each={items.get()} key={(x) => x} fallback={emptyText.get()}>
              {(x) => <span class="for-item">{x()}</span>}
            </For>
          </section>
          <section class="idx-box">
            <Index each={items.get()} fallback={emptyText.get()}>
              {(x) => <span class="idx-item">{x()}</span>}
            </Index>
          </section>
        </div>
      ),
      host,
    );
    flush();

    expect(host.querySelector('.for-box')?.textContent).toBe('nothing yet');
    expect(host.querySelector('.idx-box')?.textContent).toBe('nothing yet');

    emptyText.set('filtered to 0');
    flush();
    expect(host.querySelector('.for-box')?.textContent).toBe('filtered to 0');
    expect(host.querySelector('.idx-box')?.textContent).toBe('filtered to 0');

    // Empty -> empty update keeps DOM intact.
    items.set([]);
    flush();
    expect(host.querySelector('.for-box')?.textContent).toBe('filtered to 0');

    // Transition to populated, then back to empty with updated fallback text.
    items.set(['one']);
    flush();
    expect(host.querySelectorAll('.for-item')).toHaveLength(1);

    emptyText.set('back to empty');
    items.set([]);
    flush();
    expect(host.querySelector('.for-box')?.textContent).toBe('back to empty');
    expect(host.querySelector('.idx-box')?.textContent).toBe('back to empty');

    view.unmount();

    // Throwing fallback recovers once the list updates after the condition stops throwing.
    const errList = signal<string[]>([]);
    createRoot((dispose) => {
      try {
        const listMemo = For({
          get each() {
            return errList.get();
          },
          key: (x) => x,
          get fallback() {
            if (boom.get()) throw new Error('fallback boom');
            return document.createTextNode('recovered');
          },
          children: (x) => document.createTextNode(x()),
        }) as unknown as () => Array<() => Node>;
        expect(() => listMemo()).toThrow('fallback boom');
        boom.set(false);
        errList.set(['x']);
        expect(listMemo().length).toBe(1);
        errList.set([]);
        const nodes = listMemo();
        expect(nodes.length).toBe(1);
        expect(nodes[0]!().textContent).toBe('recovered');
      } finally {
        dispose();
      }
    });
  });

  it('For preserves duplicate-key FIFO order across suffix trims, duplicate inserts, and end-for-end swaps', () => {
    // Case 1 (suffix trim): [a1, a2] -> [b, a] must retain a1 (FIFO) and dispose a2.
    const trimList = signal([
      { id: 'a', v: 'a1' },
      { id: 'a', v: 'a2' },
    ]);
    const disposed: string[] = [];
    const host1 = document.createElement('div');
    const view1 = render(
      () => (
        <ul>
          <For each={trimList.get()} key={(x) => x.id}>
            {(x) => {
              const initial = x().v;
              onCleanup(() => disposed.push(initial));
              return <li data-init={initial}>{x().v}</li>;
            }}
          </For>
        </ul>
      ),
      host1,
    );
    flush();
    const [a1Node] = [...host1.querySelectorAll('li')];
    trimList.set([
      { id: 'b', v: 'b1' },
      { id: 'a', v: 'a-kept' },
    ]);
    flush();
    const afterTrim = [...host1.querySelectorAll('li')];
    expect(afterTrim[1]).toBe(a1Node);
    expect(afterTrim[1]?.getAttribute('data-init')).toBe('a1');
    expect(disposed).toEqual(['a2']);
    view1.unmount();

    // Case 2 (duplicate insert before suffix): [b, a1] -> [a, a] must retain a1 at index 0 (FIFO).
    const insList = signal([
      { id: 'b', v: 'b1' },
      { id: 'a', v: 'a1' },
    ]);
    const host2 = document.createElement('div');
    const view2 = render(
      () => (
        <ul>
          <For each={insList.get()} key={(x) => x.id}>
            {(x) => <li data-init={x().v}>{x().v}</li>}
          </For>
        </ul>
      ),
      host2,
    );
    flush();
    const origA1 = host2.querySelectorAll('li')[1]!;
    insList.set([
      { id: 'a', v: 'a-first' },
      { id: 'a', v: 'a-second' },
    ]);
    flush();
    const afterIns = [...host2.querySelectorAll('li')];
    expect(afterIns[0]).toBe(origA1);
    expect(afterIns[1]).not.toBe(origA1);
    view2.unmount();

    // Case 3 (end-for-end swap with duplicates in middle): [a1, c1, b1, a2, b2] -> [b, c, a, b, a].
    const swapList = signal([
      { id: 'a', v: 'a1' },
      { id: 'c', v: 'c1' },
      { id: 'b', v: 'b1' },
      { id: 'a', v: 'a2' },
      { id: 'b', v: 'b2' },
    ]);
    const host3 = document.createElement('div');
    const view3 = render(
      () => (
        <ul>
          <For each={swapList.get()} key={(x) => x.id}>
            {(x) => <li data-init={x().v}>{x().v}</li>}
          </For>
        </ul>
      ),
      host3,
    );
    flush();
    const [sa1, sc1, sb1, sa2, sb2] = [...host3.querySelectorAll('li')];
    swapList.set([
      { id: 'b', v: 'b-0' },
      { id: 'c', v: 'c-1' },
      { id: 'a', v: 'a-2' },
      { id: 'b', v: 'b-3' },
      { id: 'a', v: 'a-4' },
    ]);
    flush();
    expect([...host3.querySelectorAll('li')]).toEqual([sb1, sc1, sa1, sb2, sa2]);
    view3.unmount();

    // Case 4 (unique-key end-for-end swap fast-path): [a, b, c] -> [c, b, a] reuses all DOM nodes.
    const uniq = signal(['a', 'b', 'c']);
    const host4 = document.createElement('div');
    const view4 = render(
      () => (
        <ul>
          <For each={uniq.get()} key={(x) => x}>
            {(x) => <li>{x()}</li>}
          </For>
        </ul>
      ),
      host4,
    );
    flush();
    const [na, nb, nc] = [...host4.querySelectorAll('li')];
    uniq.set(['c', 'b', 'a']);
    flush();
    expect([...host4.querySelectorAll('li')]).toEqual([nc, nb, na]);
    view4.unmount();
  });

  it('Portal tolerates synchronous mid-normalize self-disposal without throwing or leaking target nodes', () => {
    const mountTarget = document.createElement('div');
    document.body.appendChild(mountTarget);

    // Synchronous disposal inside a nested reader in a child array during normalize().
    let disposeRoot!: () => void;
    expect(() => {
      createRoot((dispose) => {
        disposeRoot = dispose;
        const killerReader = () => {
          disposeRoot();
          const el = document.createElement('span');
          el.className = 'leaked';
          return el;
        };
        Portal({
          mount: mountTarget,
          children: [document.createElement('i'), killerReader] as unknown as JSX.Element,
        });
      });
    }).not.toThrow();
    expect(mountTarget.querySelector('.leaked')).toBeNull();
    expect(mountTarget.childNodes.length).toBe(0);

    // Effect-driven self-disposal also leaves zero nodes in the portal target.
    const open = signal(true);
    function SelfDisposing() {
      effect(() => {
        open.set(false);
      });
      return <div class="ephemeral">gone</div>;
    }

    const host = document.createElement('div');
    const view = render(
      () => (
        <Show when={open.get()}>
          {() => (
            <Portal mount={mountTarget}>
              <SelfDisposing />
            </Portal>
          )}
        </Show>
      ),
      host,
    );
    expect(() => flush()).not.toThrow();
    expect(mountTarget.querySelector('.ephemeral')).toBeNull();
    expect(mountTarget.childNodes.length).toBe(0);

    view.unmount();
    mountTarget.remove();
  });

  it('scopes For and Index fallback transitions inside enterBranch(ran, false) without leaking to an enclosing Show', () => {
    const outer = signal(true);
    const forItems = signal<string[]>([]);
    const idxItems = signal<string[]>(['a']);
    const events: string[] = [];

    const spy = (tag: string) => (_node: Element, _params: unknown, opts: { direction: 'in' | 'out' | 'both' }) => {
      events.push(`${tag}:${opts.direction}`);
      return { duration: 10, css: () => '' };
    };
    const forSpy = spy('for');
    const idxSpy = spy('idx');

    const host = document.createElement('div');
    const view = render(
      () => (
        <Show when={outer.get()}>
          {() => (
            <div>
              <For each={forItems.get()} key={(x) => x} fallback={<span transition:forSpy={{}}>for-empty</span>}>
                {(x) => <b>{x()}</b>}
              </For>
              <Index each={idxItems.get()} fallback={<span in:idxSpy={{}}>idx-empty</span>}>
                {(x) => <i>{x()}</i>}
              </Index>
            </div>
          )}
        </Show>
      ),
      host,
    );
    flush();
    // Initial empty mount of For must not play intro (ran === false).
    expect(events).toEqual([]);

    // Populated -> empty on Index must play intro because ran === true.
    idxItems.set([]);
    flush();
    expect(events).toEqual(['idx:in']);

    // Populate For so its fallback is disposed, then close the enclosing Show:
    // the disposed For fallback transition must NOT have leaked into Show's outro list.
    forItems.set(['x']);
    flush();
    events.length = 0;
    outer.set(false);
    flush();
    expect(events).toEqual([]);

    view.unmount();
  });
});
