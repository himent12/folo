// @vitest-environment happy-dom
// Sprint 13 regression tests for fw runtime hardening:
// 1. Show, Switch, For, and Index keep reactive fallback expressions live while falsy/unmatched/empty.
// 2. For preserves duplicate-key FIFO order on end-for-end swaps while retaining unique-key swap reuse.
// 3. Portal tolerates mid-mount self-disposal without throwing NotFoundError or leaking DOM nodes.
import { describe, expect, it } from 'vitest';
import { render } from '../testing';
import { createRoot, effect, flush, onCleanup, signal } from './core';
import { For, Index, Match, Portal, Show, Switch } from './flow';

describe('fw Sprint 13 runtime hardening', () => {
  it('updates reactive fallbacks in Show and Switch while falsy/unmatched and disposes on branch switch', () => {
    const ok = signal(false);
    const mode = signal<'a' | 'none'>('none');
    const msg = signal('loading');
    let showFbDisposed = 0;
    let switchFbDisposed = 0;

    function ShowFb(props: { text: string }) {
      onCleanup(() => showFbDisposed++);
      return <span class="show-fb">{props.text}</span>;
    }

    function SwitchFb(props: { text: string }) {
      onCleanup(() => switchFbDisposed++);
      return <span class="switch-fb">{props.text}</span>;
    }

    const host = document.createElement('div');
    const view = render(
      () => (
        <div>
          <Show when={ok.get()} fallback={<ShowFb text={msg.get()} />}>
            {() => <span class="show-main">main</span>}
          </Show>
          <Switch fallback={<SwitchFb text={msg.get()} />}>
            <Match when={mode.get() === 'a'}>{() => <span class="switch-a">A</span>}</Match>
          </Switch>
        </div>
      ),
      host,
    );
    flush();

    expect(host.querySelector('.show-fb')?.textContent).toBe('loading');
    expect(host.querySelector('.switch-fb')?.textContent).toBe('loading');

    msg.set('still loading');
    flush();
    expect(host.querySelector('.show-fb')?.textContent).toBe('still loading');
    expect(host.querySelector('.switch-fb')?.textContent).toBe('still loading');

    ok.set(true);
    mode.set('a');
    flush();
    expect(host.querySelector('.show-main')?.textContent).toBe('main');
    expect(host.querySelector('.switch-a')?.textContent).toBe('A');
    expect(showFbDisposed).toBeGreaterThanOrEqual(1);
    expect(switchFbDisposed).toBeGreaterThanOrEqual(1);

    view.unmount();
  });

  it('updates reactive fallbacks in For and Index while empty and recovers if fallback throws', () => {
    const items = signal<string[]>([]);
    const emptyText = signal('nothing yet');
    const boom = signal(true);

    const host = document.createElement('div');
    const view = render(
      () => (
        <div>
          <section class="for-box">
            <For each={items.get()} key={(x) => x} fallback={<p class="for-fb">{emptyText.get()}</p>}>
              {(x) => <span class="for-item">{x()}</span>}
            </For>
          </section>
          <section class="idx-box">
            <Index each={items.get()} fallback={<p class="idx-fb">{emptyText.get()}</p>}>
              {(x) => <span class="idx-item">{x()}</span>}
            </Index>
          </section>
        </div>
      ),
      host,
    );
    flush();

    expect(host.querySelector('.for-fb')?.textContent).toBe('nothing yet');
    expect(host.querySelector('.idx-fb')?.textContent).toBe('nothing yet');

    emptyText.set('filtered to 0');
    flush();
    expect(host.querySelector('.for-fb')?.textContent).toBe('filtered to 0');
    expect(host.querySelector('.idx-fb')?.textContent).toBe('filtered to 0');

    // Empty -> empty update with unchanged fallback text does not throw or lose DOM.
    items.set([]);
    flush();
    expect(host.querySelector('.for-fb')?.textContent).toBe('filtered to 0');

    // Transition to populated, then back to empty with updated text.
    items.set(['one']);
    flush();
    expect(host.querySelectorAll('.for-item')).toHaveLength(1);
    expect(host.querySelector('.for-fb')).toBeNull();

    emptyText.set('back to empty');
    items.set([]);
    flush();
    expect(host.querySelector('.for-fb')?.textContent).toBe('back to empty');
    expect(host.querySelector('.idx-fb')?.textContent).toBe('back to empty');

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

  it('For preserves duplicate-key FIFO order on end-for-end swap while reusing DOM nodes on unique swaps', () => {
    const dupList = signal([
      { id: 'a', label: 'a1' },
      { id: 'b', label: 'b1' },
      { id: 'a', label: 'a2' },
    ]);
    const host = document.createElement('div');
    const view = render(
      () => (
        <ul class="dup">
          <For each={dupList.get()} key={(x) => x.id}>
            {(x) => <li data-key={x().id}>{x().label}</li>}
          </For>
        </ul>
      ),
      host,
    );
    flush();

    const firstA = host.querySelectorAll('li')[0]!;
    const secondA = host.querySelectorAll('li')[2]!;
    // Reorder [a, b, a] -> [a, a, b]: the first 'a' node must stay before the second 'a' node (FIFO).
    dupList.set([
      { id: 'a', label: 'a1-next' },
      { id: 'a', label: 'a2-next' },
      { id: 'b', label: 'b1-next' },
    ]);
    flush();
    const afterDup = [...host.querySelectorAll('li')];
    expect(afterDup[0]).toBe(firstA);
    expect(afterDup[1]).toBe(secondA);
    expect(afterDup.map((n) => n.textContent)).toEqual(['a1-next', 'a2-next', 'b1-next']);
    view.unmount();

    // Unique-key end-for-end swap [a, b, c] -> [c, b, a] still reuses every DOM node.
    const uniq = signal(['a', 'b', 'c']);
    const host2 = document.createElement('div');
    const view2 = render(
      () => (
        <ul>
          <For each={uniq.get()} key={(x) => x}>
            {(x) => <li>{x()}</li>}
          </For>
        </ul>
      ),
      host2,
    );
    flush();
    const [na, nb, nc] = [...host2.querySelectorAll('li')];
    uniq.set(['c', 'b', 'a']);
    flush();
    expect([...host2.querySelectorAll('li')]).toEqual([nc, nb, na]);
    view2.unmount();
  });

  it('Portal tolerates mid-mount self-disposal without throwing or leaking target nodes', () => {
    const mountTarget = document.createElement('div');
    document.body.appendChild(mountTarget);
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
});
