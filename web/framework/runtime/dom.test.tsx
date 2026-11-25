// @vitest-environment happy-dom
import { describe, expect, it, vi } from 'vitest';
import { createRoot, Dynamic, effect, flush, mergeProps, onCleanup, signal, splitProps, type Signal } from 'fw';
import { spread } from 'fw/internal';
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

  it('inlined style: and static class: directives handle nullish/empty/0 values, updates, and deduped templates', () => {
    const top = signal<string | number | undefined | null>(undefined);
    const width = signal<string | undefined>('12px');
    const staticFlag = false;
    const staticOn = true;
    const staticOpacity = 0;
    const staticEmpty = '';
    const { container, unmount } = render(() => (
      <div>
        <span class="base" class:base={staticFlag} class:on={staticOn} style:top={top.get()} style:width={width.get()} />
        <span style:color="red" style:opacity={staticOpacity} style:margin={staticEmpty} />
        <em style:top={top.get()} />
      </div>
    ));
    const [s1, s2] = container.querySelectorAll('span');
    const em = container.querySelector('em')!;
    expect(s1!.className).toBe('on');
    expect(s1!.style.top).toBe('');
    expect(s1!.style.width).toBe('12px');
    // Unset initial dynamic style: leaves no style attribute at all.
    expect(em.hasAttribute('style')).toBe(false);
    expect(s2!.style.color).toBe('red');
    expect(s2!.style.opacity).toBe('0');
    expect(s2!.style.margin).toBe('');

    top.set(0);
    width.set('');
    flush();
    expect(s1!.style.top).toBe('0px');
    expect(s1!.style.width).toBe('');
    expect(em.style.top).toBe('0px');

    top.set(null);
    flush();
    expect(s1!.style.top).toBe('');
    expect(em.style.top).toBe('');
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

  it('a spread listener outlives updates of other keys, takes a new handler, and goes with its key', () => {
    const h = vi.fn();
    const h2 = vi.fn();
    const props = signal<Record<string, unknown>>({ onfocus: h, title: 'one' });
    const { container, unmount } = render(() => <input {...props.get()} />);
    const input = container.querySelector('input')!;
    const focus = () => input.dispatchEvent(new FocusEvent('focus'));
    focus();
    props.set({ onfocus: h, title: 'two' });
    flush();
    expect(input.title).toBe('two');
    focus();
    expect(h).toHaveBeenCalledTimes(2);
    props.set({ onfocus: h2, title: 'two' });
    flush();
    focus();
    expect(h).toHaveBeenCalledTimes(2);
    expect(h2).toHaveBeenCalledTimes(1);
    props.set({ title: 'three' });
    flush();
    focus();
    expect(h2).toHaveBeenCalledTimes(1);
    props.set({ onfocus: h, title: 'three' });
    flush();
    focus();
    expect(h).toHaveBeenCalledTimes(3);
    unmount();
    focus();
    expect(h).toHaveBeenCalledTimes(3);
    expect(h2).toHaveBeenCalledTimes(1);
  });

  it('spread ongotpointercapture / onlostpointercapture listen to those events (not a capture phase)', () => {
    const got = vi.fn();
    const lost = vi.fn();
    const props = { ongotpointercapture: got, onlostpointercapture: lost };
    const { container, unmount } = render(() => <div {...props} />);
    const div = container.querySelector('div')!;
    div.dispatchEvent(new Event('gotpointercapture'));
    div.dispatchEvent(new Event('lostpointercapture'));
    expect(got).toHaveBeenCalledOnce();
    expect(lost).toHaveBeenCalledOnce();
    unmount();
  });

  it('a spread class that goes away is removed; the scope class and class: directives stay', () => {
    const props = signal<Record<string, unknown>>({ class: 'old', title: 't' });
    const active = signal(true);
    const el = document.createElement('div');
    const dispose = createRoot((d) => {
      spread(el, () => props.get(), 'fw-scope', () => ({ active: active.get() }));
      return d;
    });
    expect(el.className).toBe('old fw-scope active');
    props.set({ title: 't' });
    flush();
    expect(el.className).toBe('fw-scope active');
    props.set({ class: 'new' });
    flush();
    expect(el.className).toBe('new fw-scope active');
    props.set({});
    active.set(false);
    flush();
    expect(el.className).toBe('fw-scope');
    const bare = document.createElement('div');
    const p2 = signal<Record<string, unknown>>({ class: 'old' });
    createRoot(() => spread(bare, () => p2.get()));
    p2.set({});
    flush();
    expect(bare.hasAttribute('class')).toBe(false);
    dispose();
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

describe('text runs', () => {
  it('render adjacent text and expressions as one text node, updated in place', () => {
    const a = signal(1);
    const b = signal(60);
    const { container, unmount } = render(() => (
      <p>
        Epoch {a.get()}/{b.get()}
      </p>
    ));
    const p = container.querySelector('p')!;
    expect(p.childNodes.length).toBe(1);
    expect(p.textContent).toBe('Epoch 1/60');
    const node = p.firstChild;
    a.set(2);
    flush();
    expect(p.textContent).toBe('Epoch 2/60');
    expect(p.firstChild).toBe(node);
    expect(p.childNodes.length).toBe(1);
    unmount();
  });

  it('keep neighbouring elements and blocks in place', () => {
    const n = signal(3);
    const show = signal(true);
    const { container, unmount } = render(() => (
      <p>
        <b>x</b>
        {n.get()} items{show.get() ? <i>!</i> : null} left
      </p>
    ));
    const p = container.querySelector('p')!;
    expect(p.textContent).toBe('x3 items! left');
    n.set(4);
    show.set(false);
    flush();
    expect(p.textContent).toBe('x4 items left');
    unmount();
  });

  it('a part that yields a node splits the run, and then nothing is re-created when text changes', () => {
    let made = 0;
    const label = signal('L');
    const withIcon = signal(false);
    const Icon = () => {
      made++;
      return <svg />;
    };
    const icon = () => (withIcon.get() ? <Icon /> : null);
    const { container, unmount } = render(() => (
      <button>
        {icon()}
        {label.get()}
      </button>
    ));
    const btn = container.querySelector('button')!;
    expect(btn.textContent).toBe('L');
    withIcon.set(true);
    flush();
    expect(btn.querySelector('svg')).not.toBeNull();
    expect(btn.textContent).toBe('L');
    const before = made;
    label.set('M');
    flush();
    expect(btn.textContent).toBe('M');
    expect(made).toBe(before);
    expect(btn.firstChild).toBe(btn.querySelector('svg'));
    unmount();
  });

  it('a snippet in a run renders as it is', () => {
    const Box = (props: { children?: JSX.Element }) => <span>Save {props.children}</span>;
    const { container, unmount } = render(() => (
      <Box>
        <b>now</b>
      </Box>
    ));
    expect(container.querySelector('span')!.innerHTML.replace(/<!---->/g, '')).toBe('Save <b>now</b>');
    unmount();
  });

  it('fragment runs: one node when text, the parts when not', () => {
    const n = signal(1);
    const C = () => <>Count {n.get()}</>;
    const { container, unmount } = render(() => (
      <div>
        <C />
      </div>
    ));
    const div = container.querySelector('div')!;
    expect(div.textContent).toBe('Count 1');
    expect([...div.childNodes].filter((x) => x.nodeType === 3 && (x as Text).data).length).toBe(1);
    n.set(2);
    flush();
    expect(div.textContent).toBe('Count 2');
    unmount();
  });
});

describe('Svelte attribute and prop semantics', () => {
  it('spellcheck={false} renders "false"; aria booleans stringify', () => {
    const { container, unmount } = render(() => <input spellcheck={false} aria-hidden={true} disabled={false} />);
    const i = container.querySelector('input')!;
    expect(i.getAttribute('spellcheck')).toBe('false');
    expect(i.getAttribute('aria-hidden')).toBe('true');
    expect(i.hasAttribute('disabled')).toBe(false);
    unmount();
  });

  it('a non-trivial prop is memoized: the child re-runs only when its value changes', () => {
    const phase = signal<'idle' | 'busy' | 'error'>('idle');
    const runs: (string | null)[] = [];
    const Child = (props: { src: string | null }) => {
      effect(() => void runs.push(props.src));
      return null;
    };
    const src = 'a.png';
    const { unmount } = render(() => <Child src={phase.get() === 'error' ? null : src} />);
    flush();
    phase.set('busy');
    flush();
    phase.set('error');
    flush();
    expect(runs).toEqual(['a.png', null]);
    unmount();
  });

  it('style={...} and style: directives combine, the directives winning', () => {
    const color = signal('red');
    const top = signal<string | undefined>('2px');
    const { container, unmount } = render(() => <i style={{ color: color.get(), top: '9px' }} style:top={top.get()} />);
    const i = container.querySelector('i')!;
    expect(i.style.color).toBe('red');
    expect(i.style.top).toBe('2px');
    color.set('blue');
    flush();
    expect(i.style.color).toBe('blue');
    expect(i.style.top).toBe('2px');
    top.set('5px');
    flush();
    expect(i.style.top).toBe('5px');
    const s = signal('color: green');
    const r = render(() => <b style={s.get()} style:left="1px" />);
    const b = r.container.querySelector('b')!;
    s.set('color: navy');
    flush();
    expect(b.style.color).toBe('navy');
    expect(b.style.left).toBe('1px');
    r.unmount();
    unmount();
  });

  it('<option value={1}> keeps the number: a select bound to 1 selects it', () => {
    const v = signal<number>(1);
    const { container, unmount } = render(() => (
      <div>
        <select bind:value={v}>
          <option value={0}>zero</option>
          <option value={1}>one</option>
        </select>
        <select value={1}>
          <option value={0}>zero</option>
          <option value={1}>one</option>
        </select>
      </div>
    ));
    flush();
    const [bound, oneWay] = container.querySelectorAll('select');
    expect(bound!.selectedIndex).toBe(1);
    expect(oneWay!.selectedIndex).toBe(1);
    bound!.selectedIndex = 0;
    bound!.dispatchEvent(new Event('change'));
    expect(v.get()).toBe(0);
    unmount();
  });

  it('a spread <option> value keeps the number: single and multiple selects bound to numbers select it', () => {
    const one = signal<number>(1);
    const many = signal<number[]>([1]);
    const extra = { title: 'opt' };
    const { container, unmount } = render(() => (
      <div>
        <select bind:value={one}>
          <option {...extra} value={0}>zero</option>
          <option {...extra} value={1}>one</option>
        </select>
        <select multiple bind:value={many}>
          <option {...extra} value={0}>zero</option>
          <option {...extra} value={1}>one</option>
        </select>
      </div>
    ));
    flush();
    const [single, multi] = container.querySelectorAll('select');
    expect(single!.selectedIndex).toBe(1);
    expect([...multi!.options].map((o) => o.selected)).toEqual([false, true]);
    single!.selectedIndex = 0;
    single!.dispatchEvent(new Event('change'));
    expect(one.get()).toBe(0);
    unmount();
  });

  it('a spread option value that goes away leaves no typed value behind', () => {
    const p = signal<Record<string, unknown>>({ value: 1 });
    const { container, unmount } = render(() => (
      <select>
        <option {...p.get()}>one</option>
      </select>
    ));
    const option = container.querySelector('option')! as HTMLOptionElement & { __value?: unknown };
    expect(option.__value).toBe(1);
    expect(option.value).toBe('1');
    p.set({});
    flush();
    // As `<option value={undefined}>`: '' (Svelte), the number gone.
    expect(option.__value).toBeUndefined();
    expect(option.value).toBe('');
    unmount();
  });

  it('property-only values are set when static too (indeterminate, checked, muted, defaultValue)', () => {
    const { container, unmount } = render(() => (
      <div>
        <input type="checkbox" indeterminate={true} />
        <input type="checkbox" class="c" checked={true} />
        <input class="d" defaultValue="dv" />
        <video muted />
      </div>
    ));
    expect(container.querySelector('input')!.indeterminate).toBe(true);
    expect(container.querySelector<HTMLInputElement>('.c')!.checked).toBe(true);
    expect(container.querySelector<HTMLInputElement>('.d')!.defaultValue).toBe('dv');
    expect(container.querySelector<HTMLInputElement>('.d')!.hasAttribute('defaultvalue')).toBe(false);
    expect(container.querySelector('video')!.muted).toBe(true);
    unmount();
  });
});
