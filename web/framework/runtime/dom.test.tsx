// @vitest-environment happy-dom
import { describe, expect, it, vi } from 'vitest';
import { Dynamic, flush, mergeProps, onCleanup, signal, splitProps, type Signal } from 'fw';
import { render } from '../testing';

const html = (el: Element) => el.innerHTML.replace(/<!---->/g, '');

describe('templates and text', () => {
  it('renders static HTML and reactive text, reusing the text node', () => {
    const n = signal(1);
    const { container, unmount } = render(() => <p class="a">Count: {n.get()} items</p>);
    expect(html(container)).toBe('<p class="a">Count: 1 items</p>');
    const text = container.querySelector('p')!.childNodes[1];
    n.set(2);
    flush();
    expect(container.textContent).toBe('Count: 2 items');
    expect(container.querySelector('p')!.childNodes[1]).toBe(text);
    unmount();
  });

  it('follows JSX whitespace rules', () => {
    const { container, unmount } = render(() => (
      <div>
        <b>a</b> <i>b</i>
        <u>c</u>
        {' '}
        text   spaced
        over lines
      </div>
    ));
    expect(container.firstElementChild!.innerHTML).toBe('<b>a</b> <i>b</i><u>c</u> text   spaced over lines');
    unmount();
  });

  it('decodes entities in text and attributes', () => {
    const { container, unmount } = render(() => <p title="a &amp; b">x&nbsp;y &lt; z</p>);
    const p = container.querySelector('p')!;
    expect(p.title).toBe('a & b');
    expect(p.textContent).toBe('x y < z');
    unmount();
  });

  it('inserts nodes, arrays and nothing; keeps sibling positions', () => {
    const v = signal<unknown>('x');
    const { container, unmount } = render(() => (
      <div>
        <span>a</span>
        {v.get() as JSX.Element}
        <span>b</span>
      </div>
    ));
    const div = container.firstElementChild!;
    const values: [unknown, string][] = [
      [null, '<span>a</span><span>b</span>'],
      [['p', 'q'], '<span>a</span>pq<span>b</span>'],
      [document.createElement('hr'), '<span>a</span><hr><span>b</span>'],
      [false, '<span>a</span><span>b</span>'],
      [7, '<span>a</span>7<span>b</span>'],
    ];
    for (const [value, expected] of values) {
      v.set(value);
      flush();
      expect(html(div)).toBe(expected);
    }
    unmount();
  });

  it('adjacent dynamic children keep their order through updates', () => {
    const a = signal('a');
    const b = signal('b');
    const { container, unmount } = render(() => (
      <p>
        {a.get()}
        {b.get()}
      </p>
    ));
    a.set('A');
    flush();
    b.set('B');
    flush();
    a.set('');
    flush();
    a.set('x');
    flush();
    expect(container.textContent).toBe('xB');
    unmount();
  });

  it('renders fragments and components returning fragments', () => {
    const n = signal(1);
    const Pair = (p: { n: number }) => (
      <>
        <i>{p.n}</i>
        {p.n + 1}
      </>
    );
    const { container, unmount } = render(() => (
      <div>
        <Pair n={n.get()} />
      </div>
    ));
    expect(html(container.firstElementChild!)).toBe('<i>1</i>2');
    const i = container.querySelector('i');
    n.set(5);
    flush();
    expect(html(container.firstElementChild!)).toBe('<i>5</i>6');
    expect(container.querySelector('i')).toBe(i);
    unmount();
  });
});

describe('attributes', () => {
  it('follows Svelte: null removes, false/true are strings for aria, properties for DOM props', () => {
    const v = signal<boolean | undefined>(false);
    const { container, unmount } = render(() => <button aria-expanded={v.get()} disabled={v.get()} title={v.get() ? 'on' : undefined} tabindex={v.get() ? 0 : -1} />);
    const b = container.querySelector('button')!;
    expect(b.getAttribute('aria-expanded')).toBe('false');
    expect(b.disabled).toBe(false);
    expect(b.hasAttribute('title')).toBe(false);
    expect(b.getAttribute('tabindex')).toBe('-1');
    v.set(true);
    flush();
    expect(b.getAttribute('aria-expanded')).toBe('true');
    expect(b.disabled).toBe(true);
    expect(b.title).toBe('on');
    expect(b.getAttribute('tabindex')).toBe('0');
    v.set(undefined);
    flush();
    expect(b.hasAttribute('aria-expanded')).toBe(false);
    unmount();
  });

  it('class strings, clsx values and class: directives', () => {
    const on = signal(false);
    const k = signal<string | string[]>('a b');
    const { container, unmount } = render(() => (
      <div>
        <span class="s" class:on={on.get()} />
        <i class={k.get()} class:on={on.get()} />
      </div>
    ));
    const [s, i] = [container.querySelector('span')!, container.querySelector('i')!];
    expect(s.className).toBe('s');
    expect(i.className).toBe('a b');
    on.set(true);
    flush();
    expect(s.className).toBe('s on');
    expect(i.className).toBe('a b on');
    k.set(['c', 'd']);
    flush();
    expect(i.className).toBe('c d on');
    unmount();
  });

  it('style strings, objects and style: directives (custom properties too)', () => {
    const top = signal<number | undefined>(3);
    const c = signal('red');
    const { container, unmount } = render(() => (
      <div>
        <p style="color: blue" style:top={top.get() === undefined ? undefined : `${top.get()}px`} />
        <i style={{ color: c.get(), '--x': '2px' }} />
      </div>
    ));
    const p = container.querySelector('p')!;
    const i = container.querySelector('i')!;
    expect(p.style.color).toBe('blue');
    expect(p.style.top).toBe('3px');
    expect(i.style.getPropertyValue('--x')).toBe('2px');
    expect(i.style.color).toBe('red');
    top.set(undefined);
    c.set('green');
    flush();
    expect(p.style.top).toBe('');
    expect(i.style.color).toBe('green');
    unmount();
  });

  it('spreads props reactively, with events and class', () => {
    const extra = signal<Record<string, unknown>>({ title: 'a', 'data-x': 1 });
    const click = vi.fn();
    const { container, unmount } = render(() => <button class="base" {...extra.get()} onclick={click} />);
    const b = container.querySelector('button')!;
    expect(b.title).toBe('a');
    expect(b.getAttribute('data-x')).toBe('1');
    expect(b.className).toBe('base');
    extra.set({ class: 'other', 'aria-label': 'L' });
    flush();
    expect(b.hasAttribute('title')).toBe(false);
    expect(b.getAttribute('aria-label')).toBe('L');
    b.click();
    expect(click).toHaveBeenCalledOnce();
    unmount();
  });
});

describe('events', () => {
  it('delegates with currentTarget and stopPropagation', () => {
    const log: string[] = [];
    const { container, unmount } = render(() => (
      <div onclick={(e) => log.push(`div ${(e.currentTarget as Element).tagName}`)}>
        <button
          onclick={(e) => {
            log.push(`button ${(e.currentTarget as Element).tagName}`);
          }}
        >
          <span>x</span>
        </button>
        <a
          onclick={(e) => {
            log.push('a');
            e.stopPropagation();
          }}
        >
          y
        </a>
      </div>
    ));
    container.querySelector('span')!.click();
    container.querySelector('a')!.click();
    expect(log).toEqual(['button BUTTON', 'div DIV', 'a']);
    unmount();
  });

  it('non-delegated and capture events, removed on unmount', () => {
    const scroll = vi.fn();
    const capture = vi.fn();
    const { container, unmount } = render(() => (
      <div onscroll={scroll} onclickcapture={capture}>
        <i />
      </div>
    ));
    const div = container.firstElementChild!;
    div.dispatchEvent(new Event('scroll'));
    container.querySelector('i')!.click();
    expect(scroll).toHaveBeenCalledOnce();
    expect(capture).toHaveBeenCalledOnce();
    unmount();
    div.dispatchEvent(new Event('scroll'));
    expect(scroll).toHaveBeenCalledOnce();
  });

  it('a handler prop read late is the current one', () => {
    const h = signal<() => void>(() => {});
    const a = vi.fn();
    const Btn = (p: { onclick: () => void }) => <button onclick={p.onclick} />;
    const { container, unmount } = render(() => <Btn onclick={h.get()} />);
    h.set(() => a);
    flush();
    container.querySelector('button')!.click();
    expect(a).toHaveBeenCalledOnce();
    unmount();
  });
});

describe('svg', () => {
  it('creates SVG elements in the SVG namespace, also from separate templates', () => {
    const show = signal(true);
    const Part = () => <circle r="2" />;
    const { container, unmount } = render(() => (
      <svg width={10} viewBox="0 0 10 10">
        {show.get() ? <path d="M0 0" class="l" /> : <line x1="1" />}
        <Part />
      </svg>
    ));
    const svg = container.querySelector('svg')!;
    const NS = 'http://www.w3.org/2000/svg';
    expect(svg.namespaceURI).toBe(NS);
    expect(svg.querySelector('path')!.namespaceURI).toBe(NS);
    expect(svg.querySelector('path')!.getAttribute('class')).toBe('l');
    expect(svg.querySelector('circle')!.namespaceURI).toBe(NS);
    show.set(false);
    flush();
    expect(svg.querySelector('line')!.namespaceURI).toBe(NS);
    unmount();
  });
});

describe('components and props', () => {
  it('props are live getters; children is a snippet', () => {
    const label = signal('a');
    const Box = (p: { label: string; children?: JSX.Element }) => (
      <div>
        <b>{p.label}</b>
        {p.children}
      </div>
    );
    const { container, unmount } = render(() => (
      <Box label={label.get()}>
        <i>kid</i>
      </Box>
    ));
    expect(html(container)).toBe('<div><b>a</b><i>kid</i></div>');
    const i = container.querySelector('i');
    label.set('b');
    flush();
    expect(html(container)).toBe('<div><b>b</b><i>kid</i></div>');
    expect(container.querySelector('i')).toBe(i);
    unmount();
  });

  it('mergeProps and splitProps keep reactivity', () => {
    const v = signal(1);
    const props = { get a() { return v.get(); }, b: 2 };
    const [mine, rest] = splitProps(mergeProps(props, { c: 3 }), ['a']);
    expect(mine.a).toBe(1);
    expect(Object.keys(rest).sort()).toEqual(['b', 'c']);
    v.set(5);
    expect(mine.a).toBe(5);
  });

  it('bind: on a component reads and writes the parent signal', () => {
    const open = signal(false);
    const Child = (p: { open: boolean }) => <button onclick={() => (p.open = !p.open)}>{String(p.open)}</button>;
    const { container, unmount } = render(() => <Child bind:open={open} />);
    container.querySelector('button')!.click();
    flush();
    expect(open.get()).toBe(true);
    expect(container.textContent).toBe('true');
    unmount();
  });

  it('Dynamic renders a tag or a component', () => {
    const tag = signal<string>('h2');
    const { container, unmount } = render(() => (
      <Dynamic component={tag.get()} class="t" id="x">
        hi
      </Dynamic>
    ));
    expect(container.querySelector('h2#x.t')!.textContent).toBe('hi');
    tag.set('h3');
    flush();
    expect(container.querySelector('h3#x.t')).not.toBeNull();
    unmount();
  });

  it('disposes what a component created on unmount', () => {
    const cleanup = vi.fn();
    const s: Signal<number> = signal(0);
    const C = () => {
      onCleanup(cleanup);
      return <i>{s.get()}</i>;
    };
    const { unmount } = render(() => <C />);
    unmount();
    expect(cleanup).toHaveBeenCalledOnce();
    expect((s as unknown as { _targets: unknown })._targets).toBeUndefined();
  });
});
