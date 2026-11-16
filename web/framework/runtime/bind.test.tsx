// @vitest-environment happy-dom
import { setFlagsFromString } from 'node:v8';
import { runInNewContext } from 'node:vm';
import { describe, expect, it, vi } from 'vitest';
import { bindable, defaults, effect, fade, flush, fly, For, Key, onWindow, Show, signal, splitProps, type Action, type Signal } from 'fw';
import { render } from '../testing';

describe('bind:', () => {
  it('bind:value on an input is two-way; number inputs give numbers', () => {
    const text = signal('a');
    const n = signal<number | null>(3);
    const { container, unmount } = render(() => (
      <div>
        <input bind:value={text} />
        <input type="number" bind:value={n} />
      </div>
    ));
    const [t, num] = [...container.querySelectorAll('input')];
    expect(t!.value).toBe('a');
    t!.value = 'abc';
    t!.dispatchEvent(new Event('input', { bubbles: true }));
    expect(text.get()).toBe('abc');
    text.set('x');
    flush();
    expect(t!.value).toBe('x');
    num!.value = '7';
    num!.dispatchEvent(new Event('input', { bubbles: true }));
    expect(n.get()).toBe(7);
    unmount();
  });

  it('bind:checked and bind:group', () => {
    const on = signal(false);
    const pick = signal('b');
    const many = signal<string[]>(['x']);
    const { container, unmount } = render(() => (
      <div>
        <input type="checkbox" bind:checked={on} />
        <input type="radio" value="a" bind:group={pick} />
        <input type="radio" value="b" bind:group={pick} />
        <input type="checkbox" value="x" bind:group={many} />
        <input type="checkbox" value="y" bind:group={many} />
      </div>
    ));
    const inputs = [...container.querySelectorAll('input')];
    inputs[0]!.click();
    expect(on.get()).toBe(true);
    expect(inputs[2]!.checked).toBe(true);
    inputs[1]!.click();
    expect(pick.get()).toBe('a');
    inputs[4]!.click();
    expect(many.get()).toEqual(['x', 'y']);
    many.set([]);
    flush();
    expect(inputs[3]!.checked).toBe(false);
    unmount();
  });

  it('bind:value on a select picks the option after the options render', () => {
    const v = signal('two');
    const opts = ['one', 'two', 'three'];
    const { container, unmount } = render(() => (
      <select bind:value={v}>
        <For each={opts}>{(o) => <option value={o()}>{o()}</option>}</For>
      </select>
    ));
    flush();
    const sel = container.querySelector('select')!;
    expect(sel.value).toBe('two');
    sel.value = 'three';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    expect(v.get()).toBe('three');
    unmount();
  });

  it('ref gets the element; a signal ref is cleared when the element goes', () => {
    const el = signal<HTMLElement>();
    const show = signal(true);
    let fnRef: HTMLElement | undefined;
    const { unmount } = render(() => (
      <div ref={(e) => (fnRef = e)}>
        <Show when={show.get()}>
          <span ref={el} />
        </Show>
      </div>
    ));
    expect(fnRef?.tagName).toBe('DIV');
    expect(el.get()?.tagName).toBe('SPAN');
    show.set(false);
    flush();
    expect(el.get()).toBeUndefined();
    unmount();
  });
});

describe('bind:clientWidth', () => {
  // stress x4-soak-3: the shared observer's callback was a closure made in the first caller's effect, so it kept that
  // first element (on the Run page: the run chart, and through it the whole page left long ago) alive for good.
  it('keeps no element alive once it has left the page, and still serves the elements after it', async () => {
    setFlagsFromString('--expose-gc');
    const gc = runInNewContext('gc') as () => void;
    let notify: ResizeObserverCallback | undefined;
    class FakeObserver {
      constructor(cb: ResizeObserverCallback) {
        notify ??= cb;
      }
      observe() {}
      unobserve() {}
      disconnect() {}
    }
    vi.stubGlobal('ResizeObserver', FakeObserver);
    try {
      const width = signal(0);
      const first = render(() => <div class="first" bind:clientWidth={width} />);
      flush();
      const gone = new WeakRef(first.container.firstElementChild!);
      first.unmount();
      const w2 = signal(0);
      const second = render(() => <div class="second" bind:clientWidth={w2} />);
      flush();
      const el = second.container.querySelector('.second')!;
      Object.defineProperty(el, 'clientWidth', { value: 321 });
      notify!([{ target: el } as ResizeObserverEntry], {} as ResizeObserver);
      expect(w2.get()).toBe(321);
      second.unmount();
      await new Promise((r) => setTimeout(r, 0));
      gc();
      expect(gone.deref()).toBeUndefined();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe('props', () => {
  it('defaults fill undefined props and stay reactive', () => {
    const size = signal<string | undefined>(undefined);
    let seen: string[] = [];
    const C = (props: { size?: string; other?: number }) => {
      const p = defaults(props, { size: 'md' });
      effect(() => void seen.push(p.size));
      return null;
    };
    const { unmount } = render(() => <C size={size.get()} />);
    flush();
    size.set('lg');
    flush();
    size.set(undefined);
    flush();
    expect(seen).toEqual(['md', 'lg', 'md']);
    unmount();
  });

  it('bindable: bound writes reach the parent; unbound writes are local until the parent changes', () => {
    const parent = signal(1);
    let inner!: Signal<number>;
    const C = (props: { value?: number }) => {
      inner = bindable(props, 'value', 0);
      return null;
    };
    const a = render(() => <C bind:value={parent} />);
    inner.set(5);
    expect(parent.get()).toBe(5);
    a.unmount();
    const b = render(() => <C value={parent.get()} />);
    inner.set(9);
    expect(inner.get()).toBe(9);
    expect(parent.get()).toBe(5);
    parent.set(6);
    expect(inner.get()).toBe(6);
    b.unmount();
  });

  it('a rest spread onto an element and into a component stays reactive', () => {
    const title = signal('a');
    const Inner = (props: { title?: string }) => <i title={props.title} />;
    const Outer = (props: { label: string; title?: string }) => {
      const [, rest] = splitProps(props, ['label']);
      return (
        <div {...rest}>
          <Inner {...rest} />
        </div>
      );
    };
    const { container, unmount } = render(() => <Outer label="x" title={title.get()} />);
    title.set('b');
    flush();
    expect(container.querySelector('div')!.title).toBe('b');
    expect(container.querySelector('i')!.title).toBe('b');
    expect(container.querySelector('div')!.hasAttribute('label')).toBe(false);
    unmount();
  });
});

describe('use:', () => {
  it('runs after mount, updates with the param, destroys on removal', () => {
    const log: string[] = [];
    const action: Action<HTMLElement, string> = (node, p) => {
      log.push(`init ${p} ${node.isConnected}`);
      return { update: (q) => log.push(`update ${q}`), destroy: () => log.push('destroy') };
    };
    const p = signal('a');
    const show = signal(true);
    const { unmount } = render(() => <Show when={show.get()}>{() => <b use:action={p.get()} />}</Show>);
    expect(log).toEqual([]);
    flush();
    expect(log).toEqual(['init a true']);
    p.set('b');
    flush();
    show.set(false);
    flush();
    expect(log).toEqual(['init a true', 'update b', 'destroy']);
    unmount();
  });

  it('use={[action, param]} and lists', () => {
    const seen: string[] = [];
    const a: Action<HTMLElement, string> = (_, p) => void seen.push(`a ${p}`);
    const b: Action<HTMLElement> = () => void seen.push('b');
    const { unmount } = render(() => (
      <div>
        <i use={[a, 'x']} />
        <i use={[[a, 'y'], b]} />
      </div>
    ));
    flush();
    expect(seen).toEqual(['a x', 'a y', 'b']);
    unmount();
  });
});

describe('window listeners', () => {
  it('onWindow listens for the component life', () => {
    const fn = vi.fn();
    const C = () => {
      onWindow('keydown', fn);
      return null;
    };
    const { unmount } = render(() => <C />);
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'a' }));
    unmount();
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'a' }));
    expect(fn).toHaveBeenCalledOnce();
  });
});

describe('transitions', () => {
  it('an outro keeps a removed row in the DOM until it ends; intros only after the first render', async () => {
    const list = signal([1, 2]);
    const ends: string[] = [];
    const slow = (node: Element) => {
      node.addEventListener('outroend', () => ends.push('out'));
      node.addEventListener('introend', () => ends.push('in'));
      return { duration: 10, css: (t: number) => `opacity: ${t}` };
    };
    // happy-dom runs Web Animations on real timers: a quiet moment shows nothing ran that should not, and what should
    // run is waited for (a loaded machine fires a 10 ms animation's end later than any fixed margin).
    const settle = () => new Promise((r) => setTimeout(r, 80));
    const ended = (want: string[]) => vi.waitFor(() => expect(ends).toEqual(want), { timeout: 3000, interval: 10 });
    const { container, unmount } = render(() => (
      <ul>
        <For each={list.get()}>{(n) => <li transition:slow={{}}>{n()}</li>}</For>
      </ul>
    ));
    flush();
    await settle();
    expect(ends).toEqual([]);
    list.set([1, 2, 3]);
    flush();
    await ended(['in']);
    list.set([1, 3]);
    flush();
    expect(container.querySelectorAll('li').length).toBe(3);
    await ended(['in', 'out']);
    expect(container.textContent).toBe('13');
    unmount();
  });

  it('fade and fly produce Svelte’s css', () => {
    const el = document.createElement('div');
    el.style.opacity = '1';
    document.body.appendChild(el);
    expect(fade(el, { duration: 160 }).css!(0.5, 0.5)).toBe('opacity: 0.5');
    const f = fly(el, { y: 8 });
    expect(f.css!(0.5, 0.5).replace(/\s+/g, ' ').trim()).toBe('transform: translate(0px, 4px); opacity: 0.5');
    expect(f.duration).toBe(400);
    el.remove();
  });

  it('Key re-creation plays an in: transition', async () => {
    const k = signal(1);
    const ends: number[] = [];
    const t = (node: Element) => {
      node.addEventListener('introend', () => ends.push(k.peek()));
      return { duration: 10, css: (x: number) => `opacity: ${x}` };
    };
    const { unmount } = render(() => (
      <Key when={k.get()}>
        <div in:t={{}}>{k.get()}</div>
      </Key>
    ));
    flush();
    // The first render plays no intro: a quiet moment shows none ran.
    await new Promise((r) => setTimeout(r, 60));
    expect(ends).toEqual([]);
    k.set(2);
    flush();
    // The re-created row's intro ends when its animation does (later on a loaded machine): waited for.
    await vi.waitFor(() => expect(ends).toEqual([2]), { timeout: 3000, interval: 10 });
    unmount();
  });
});
