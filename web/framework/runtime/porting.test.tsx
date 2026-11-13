// @vitest-environment happy-dom
// Regressions the ports found: each test is the smallest repro of a difference from Svelte that a screen hit.
import { describe, expect, it, vi } from 'vitest';
import { bindable, createUniqueId, deep, effect, flush, For, onMount, Show, signal } from 'fw';
import { render } from '../testing';

const html = (el: Element) => el.innerHTML.replace(/<!---->/g, '');

describe('conditional children', () => {
  it("a conditional as a component's children builds its branch once per flip", () => {
    let made = 0;
    const Leaf = () => {
      made++;
      return <b>leaf</b>;
    };
    const Box = (p: { children?: JSX.Element }) => <div>{p.children}</div>;
    const on = signal(false);
    const { container, unmount } = render(() => <Box>{on.get() ? <Leaf /> : null}</Box>);
    expect(made).toBe(0);
    on.set(true);
    flush();
    expect(made).toBe(1);
    expect(html(container)).toBe('<div><b>leaf</b></div>');
    on.set(false);
    flush();
    on.set(true);
    flush();
    expect(made).toBe(2);
    unmount();
  });

  it("a <Show> as a component's children builds its branch once per flip", () => {
    let made = 0;
    const Leaf = () => {
      made++;
      return <b>leaf</b>;
    };
    const Box = (p: { children?: JSX.Element }) => <div>{p.children}</div>;
    const on = signal(false);
    const { unmount } = render(() => (
      <Box>
        <Show when={on.get()}>
          <Leaf />
        </Show>
      </Box>
    ));
    on.set(true);
    flush();
    expect(made).toBe(1);
    unmount();
  });

  it("a snippet whose whole content is a conditional builds its branch once per flip", () => {
    let made = 0;
    const Leaf = () => {
      made++;
      return <b>leaf</b>;
    };
    const on = signal(false);
    const body = () => <>{on.get() ? <Leaf /> : null}</>;
    const Panel = (p: { body: () => JSX.Element }) => <section>{p.body()}</section>;
    const { unmount } = render(() => (
      <div>
        {body()}
        <Panel body={body} />
      </div>
    ));
    on.set(true);
    flush();
    expect(made).toBe(2);
    unmount();
  });
});

describe('creation order', () => {
  it('a conditional in children is created before the components that follow it (Svelte order)', () => {
    const order: string[] = [];
    const ids: string[] = [];
    const A = (p: { n: string; children?: JSX.Element }) => {
      order.push(p.n);
      ids.push(createUniqueId());
      return <i>{p.children}</i>;
    };
    const c = signal(true);
    const { unmount } = render(() => (
      <A n="page">
        {c.get() ? <A n="cond" /> : null}
        <A n="static" />
      </A>
    ));
    expect(order).toEqual(['page', 'cond', 'static']);
    const n = (id: string) => Number(id.slice(1));
    expect(n(ids[1]!)).toBe(n(ids[0]!) + 1);
    expect(n(ids[2]!)).toBe(n(ids[0]!) + 2);
    unmount();
  });
});

describe('branches re-created in document order', () => {
  it("a conditional among a fragment's components flips before one inside a later component (both on one change)", () => {
    const order: string[] = [];
    const A = (p: { n: string }) => {
      order.push(p.n);
      createUniqueId();
      return <i />;
    };
    const Panel = (p: { children?: JSX.Element }) => <section>{p.children}</section>;
    const Page = (p: { children?: JSX.Element }) => <main>{p.children}</main>;
    const data = signal(false);
    const { unmount } = render(() => (
      <Page>
        {data.get() ? <A n="callout" /> : null}
        <Panel>{data.get() ? <A n="curves" /> : null}</Panel>
        <Panel>{data.get() ? <A n="matrix" /> : null}</Panel>
      </Page>
    ));
    data.set(true);
    flush();
    expect(order).toEqual(['callout', 'curves', 'matrix']);
    unmount();
  });
});

describe('text runs', () => {
  it('a fragment run whose part turns into a node later leaves nothing behind when its branch goes', () => {
    const v = signal<JSX.Element>('text');
    const shown = signal(true);
    const { container, unmount } = render(() => (
      <div>
        <Show when={shown.get()}>
          <>start {v.get()} end</>
        </Show>
        <i>last</i>
      </div>
    ));
    const div = container.querySelector('div')!;
    expect(div.textContent).toBe('start text endlast');
    const b = document.createElement('b');
    b.textContent = 'bold';
    v.set(b);
    flush();
    expect(html(div)).toBe('start <b>bold</b> end<i>last</i>');
    shown.set(false);
    flush();
    expect(html(div)).toBe('<i>last</i>');
    shown.set(true);
    flush();
    expect(div.textContent).toBe('start bold endlast');
    v.set('plain');
    flush();
    expect(div.textContent).toBe('start plain endlast');
    shown.set(false);
    flush();
    expect(html(div)).toBe('<i>last</i>');
    unmount();
  });

  it('a later split elsewhere in a fragment run keeps the parts split before (their DOM and state)', () => {
    const a = signal<JSX.Element>('a');
    const b = signal<JSX.Element>('b');
    const made = vi.fn();
    const button = () => {
      made();
      return <button>btn</button>;
    };
    const Frag = () => <>start {a.get()} mid {b.get()} end</>;
    const { container, unmount } = render(() => (
      <div>
        <Frag />
      </div>
    ));
    const div = container.querySelector('div')!;
    a.set(() => button);
    flush();
    const btn = div.querySelector('button')!;
    btn.dataset.state = 'pressed';
    expect(div.textContent).toBe('start btn mid b end');
    const i = document.createElement('i');
    i.textContent = 'i';
    b.set(i);
    flush();
    expect(div.textContent).toBe('start btn mid i end');
    expect(div.querySelector('button')).toBe(btn);
    expect(btn.dataset.state).toBe('pressed');
    expect(made).toHaveBeenCalledOnce();
    b.set('plain');
    flush();
    expect(div.textContent).toBe('start btn mid plain end');
    expect(div.querySelector('button')).toBe(btn);
    expect(made).toHaveBeenCalledOnce();
    unmount();
  });

  it('snippets in an array part (nested, of any length) keep their nodes through a later split elsewhere', () => {
    const a = signal<JSX.Element>('a');
    const b = signal<JSX.Element>('b');
    const made: string[] = [];
    const button = () => {
      made.push('button');
      return <button>btn</button>;
    };
    const em = () => {
      made.push('em');
      return <em>em</em>;
    };
    const Frag = () => <>start {a.get()} mid {b.get()} end</>;
    const { container, unmount } = render(() => (
      <div>
        <Frag />
      </div>
    ));
    const div = container.querySelector('div')!;
    a.set([button, [' and ', em]] as unknown as JSX.Element);
    flush();
    expect(div.textContent).toBe('start btn and em mid b end');
    const btn = div.querySelector('button')!;
    const e = div.querySelector('em')!;
    const i = document.createElement('i');
    i.textContent = 'i';
    b.set(i);
    flush();
    expect(div.textContent).toBe('start btn and em mid i end');
    expect(div.querySelector('button')).toBe(btn);
    expect(div.querySelector('em')).toBe(e);
    expect(made).toEqual(['button', 'em']);
    // A new array, longer: its snippets render once each, nothing of the old one is left.
    a.set([em, button, button] as unknown as JSX.Element);
    flush();
    expect(div.textContent).toBe('start embtnbtn mid i end');
    expect(made).toEqual(['button', 'em', 'em', 'button', 'button']);
    const buttons = [...div.querySelectorAll('button')];
    b.set('plain');
    flush();
    expect(div.textContent).toBe('start embtnbtn mid plain end');
    expect([...div.querySelectorAll('button')]).toEqual(buttons);
    expect(made).toHaveLength(5);
    a.set([] as unknown as JSX.Element);
    flush();
    expect(div.textContent).toBe('start  mid plain end');
    unmount();
  });

  it('snippets a snippet returns (in arrays at any depth) keep their nodes through a later split elsewhere', () => {
    const a = signal<JSX.Element>('a');
    const b = signal<JSX.Element>('b');
    let made = 0;
    const inner = () => {
      made++;
      return <input />;
    };
    const outer = () => [[inner]] as unknown as JSX.Element;
    const Frag = () => <>start {a.get()} mid {b.get()} end</>;
    const { container, unmount } = render(() => (
      <div>
        <Frag />
      </div>
    ));
    const div = container.querySelector('div')!;
    a.set([outer] as unknown as JSX.Element);
    flush();
    const input = div.querySelector('input')!;
    input.value = 'typed';
    const i = document.createElement('i');
    i.textContent = 'i';
    b.set(i);
    flush();
    expect(div.querySelector('input')).toBe(input);
    expect(input.value).toBe('typed');
    expect(made).toBe(1);
    a.set('gone');
    flush();
    expect(div.querySelector('input')).toBeNull();
    expect(div.textContent).toBe('start gone mid i end');
    unmount();
  });

  it("a split part's outro keeps it in the DOM with the run's text until the outro ends", async () => {
    const ends: string[] = [];
    const slow = (node: Element) => {
      node.addEventListener('outroend', () => ends.push('out'));
      return { duration: 10, css: (t: number) => `opacity: ${t}` };
    };
    // The compiler cannot see `label` as a snippet: its call is a part of the run, which splits there.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const Frag = (p: { label: () => any }) => <>start {p.label()} end</>;
    const shown = signal(true);
    const { container, unmount } = render(() => (
      <div>
        <Show when={shown.get()}>
          <Frag label={() => <b out:slow={{}}>bold</b>} />
        </Show>
      </div>
    ));
    const div = container.querySelector('div')!;
    expect(html(div)).toBe('start <b>bold</b> end');
    shown.set(false);
    flush();
    // Leaving (made inert, as Svelte does), the part with the text around it.
    expect(div.querySelector('b')?.inert).toBe(true);
    expect(div.textContent).toBe('start bold end');
    // The outro ends when its animation does (happy-dom runs it on real timers; later on a loaded machine).
    await vi.waitFor(() => expect(ends).toEqual(['out']), { timeout: 3000, interval: 10 });
    expect(html(div)).toBe('');
    unmount();
  });

  it('a late split in a For row moves with its row', () => {
    const rows = signal(['x', 'y', 'z']);
    const v = signal<JSX.Element>('t');
    const { container, unmount } = render(() => (
      <p>
        <For each={rows.get()}>{(r) => <>{r()}:{v.get()};</>}</For>
      </p>
    ));
    const p = container.querySelector('p')!;
    expect(p.textContent).toBe('x:t;y:t;z:t;');
    // A snippet: each row renders a <b> of its own (`set` takes a function as an updater).
    const snippet = () => <b>B</b>;
    v.set(() => snippet);
    flush();
    expect(p.textContent).toBe('x:B;y:B;z:B;');
    rows.set(['z', 'x', 'y']);
    flush();
    expect(p.textContent).toBe('z:B;x:B;y:B;');
    rows.set([]);
    flush();
    expect(p.textContent).toBe('');
    unmount();
  });

  it('static text between blocks and a run keeps its own nodes (no template merge)', () => {
    const a = signal(true);
    const b = signal(true);
    const c = signal('c');
    const { container, unmount } = render(() => (
      <p>
        {a.get() ? <b /> : null}
        {' '}
        {b.get() ? <i /> : null}
        {' '}
        {c.get()}
      </p>
    ));
    const p = container.querySelector('p')!;
    expect(html(p)).toBe('<b></b> <i></i> c');
    c.set('d');
    a.set(false);
    flush();
    expect(html(p)).toBe(' <i></i> d');
    unmount();
  });

  it('two local snippet calls side by side are each rendered once', () => {
    let made = 0;
    const a = () => {
      made++;
      return <b>a</b>;
    };
    const b = () => {
      made++;
      return <i>b</i>;
    };
    const { container, unmount } = render(() => (
      <p>
        {a()}
        {b()}
      </p>
    ));
    expect(made).toBe(2);
    expect(html(container.querySelector('p')!)).toBe('<b>a</b><i>b</i>');
    unmount();
  });

  it('a snippet call in a run splits it as Svelte does: the text after it stays one node', () => {
    let made = 0;
    const link = () => {
      made++;
      return <a>run</a>;
    };
    const x = signal('7m');
    const { container, unmount } = render(() => (
      <span>
        {' '}
        {link()} ({x.get()}) and
      </span>
    ));
    const span = container.querySelector('span')!;
    const texts = [...span.childNodes].filter((n) => n.nodeType === 3 && (n as Text).data !== '').map((n) => (n as Text).data);
    expect(texts).toEqual([' ', ' (7m) and']);
    expect(made).toBe(1);
    const tail = [...span.childNodes].find((n) => n.nodeType === 3 && (n as Text).data === ' (7m) and');
    x.set('8m');
    flush();
    expect(tail!.textContent).toBe(' (8m) and');
    expect(made).toBe(1);
    unmount();
  });

  it('a run part that renders a node (unknown to the compiler) is created once and kept when the text changes', () => {
    let made = 0;
    const Tag = () => {
      made++;
      return <b>t</b>;
    };
    const Box = (p: { lead: string; tag?: JSX.Element }) => (
      <span>
        {p.lead}: {p.tag} {p.lead}!
      </span>
    );
    const lead = signal('x');
    const { container, unmount } = render(() => <Box lead={lead.get()} tag={<Tag />} />);
    const span = container.querySelector('span')!;
    expect(span.textContent).toBe('x: t x!');
    expect(made).toBe(1);
    lead.set('y');
    flush();
    expect(span.textContent).toBe('y: t y!');
    expect(made).toBe(1);
    unmount();
  });

  it('two snippet props side by side in a run are each rendered once', () => {
    let made = 0;
    const Leaf = (p: { n: string }) => {
      made++;
      return <b>{p.n}</b>;
    };
    const Row = (p: { left: unknown; right: unknown }) => (
      <p>
        {p.left as string}
        {p.right as string}
      </p>
    );
    const { container, unmount } = render(() => <Row left={() => <Leaf n="l" />} right={() => <Leaf n="r" />} />);
    expect(made).toBe(2);
    expect(container.querySelector('p')!.textContent).toBe('lr');
    unmount();
  });

  it('a ternary of snippets is a block: its branch is created only when the condition flips', () => {
    let made = 0;
    const on = signal(true);
    const n = signal(1);
    const a = () => {
      made++;
      return <b>a</b>;
    };
    const b = () => {
      made++;
      return <i>b</i>;
    };
    const { container, unmount } = render(() => (
      <p>
        {on.get() ? a() : b()} n={n.get()}
      </p>
    ));
    expect(made).toBe(1);
    n.set(2);
    flush();
    expect(made).toBe(1);
    expect(container.querySelector('p')!.textContent).toBe('a n=2');
    on.set(false);
    flush();
    expect(made).toBe(2);
    expect(container.querySelector('p')!.textContent).toBe('b n=2');
    unmount();
  });
});

describe('branches settle before their effects', () => {
  it("a branch's bindings never run with the state that removes it (the branch is created before its reader)", () => {
    const short = signal<{ scratch: boolean } | null>({ scratch: true });
    const Box = (p: { children?: JSX.Element }) => <div>{p.children}</div>;
    const { container, unmount } = render(() => (
      <Box>
        {short.get() ? <span>{short.get()!.scratch ? 'short' : 'warmup'}</span> : null}
        <i />
      </Box>
    ));
    expect(container.textContent).toBe('short');
    short.set(null);
    expect(() => flush()).not.toThrow();
    expect(container.textContent).toBe('');
    short.set({ scratch: false });
    flush();
    expect(container.textContent).toBe('warmup');
    unmount();
  });
});

describe('bindable', () => {
  it('a prop the parent does not bind: the parent wins on any change, even back to the value a write was made over', () => {
    let child!: { set(v: boolean): void; get(): boolean };
    const Check = (p: { checked?: boolean }) => {
      const checked = bindable(p, 'checked', false);
      child = checked;
      return <i>{String(checked.get())}</i>;
    };
    // A plain read (not memoized, as in Svelte): the prop's getter reads the parent's state directly.
    const parent = deep({ on: false });
    const { container, unmount } = render(() => <Check checked={parent.on} />);
    child.set(true);
    flush();
    expect(container.textContent).toBe('true');
    parent.on = true;
    flush();
    parent.on = false;
    flush();
    expect(container.textContent).toBe('false');
    // A -> B -> A with no read in between: still the parent's.
    child.set(true);
    flush();
    expect(container.textContent).toBe('true');
    parent.on = true;
    parent.on = false;
    flush();
    expect(container.textContent).toBe('false');
    unmount();
  });
});

describe('select bindings re-select when the options change (Svelte init_select)', () => {
  const settle = async () => {
    flush();
    await Promise.resolve();
    await Promise.resolve();
  };

  it("value '' with options arriving later leaves nothing selected", async () => {
    const v = signal('');
    const opts = signal<string[]>([]);
    const { container, unmount } = render(() => (
      <select bind:value={v}>
        <For each={opts.get()}>{(o) => <option value={o()}>{o()}</option>}</For>
      </select>
    ));
    await settle();
    opts.set(['a', 'b']);
    await settle();
    expect(container.querySelector('select')!.selectedIndex).toBe(-1);
    expect(v.get()).toBe('');
    unmount();
  });

  it("value 'b' with options going ['a'] -> ['a', 'b'] selects b once it exists", async () => {
    const v = signal('b');
    const opts = signal(['a']);
    const { container, unmount } = render(() => (
      <select bind:value={v}>
        <For each={opts.get()}>{(o) => <option value={o()}>{o()}</option>}</For>
      </select>
    ));
    await settle();
    const sel = container.querySelector('select')!;
    expect(sel.selectedIndex).toBe(-1);
    opts.set(['a', 'b']);
    await settle();
    expect(sel.value).toBe('b');
    expect(v.get()).toBe('b');
    unmount();
  });

  it('a one-way value={...} re-selects too', async () => {
    const opts = signal(['a']);
    const { container, unmount } = render(() => (
      <select value="b">
        <For each={opts.get()}>{(o) => <option value={o()}>{o()}</option>}</For>
      </select>
    ));
    await settle();
    opts.set(['a', 'b']);
    await settle();
    expect(container.querySelector('select')!.value).toBe('b');
    unmount();
  });
});

describe('styles as Svelte writes them (set_style)', () => {
  it('first the attribute text (cssText), then setProperty for a directive or style object key that changed', () => {
    const top = signal('1px');
    const c = signal('red');
    const proto = CSSStyleDeclaration.prototype;
    const setProperty = vi.spyOn(proto, 'setProperty');
    const cssText = Object.getOwnPropertyDescriptor(proto, 'cssText')!;
    const texts: string[] = [];
    Object.defineProperty(proto, 'cssText', {
      configurable: true,
      get: cssText.get,
      set(v: string) {
        texts.push(v);
        cssText.set!.call(this, v);
      },
    });
    try {
      const { container, unmount } = render(() => (
        <div>
          <i style="color: blue" style:top={top.get()} />
          <b style={{ '--c': c.get(), width: '2px' }} />
        </div>
      ));
      expect(texts).toEqual(['color: blue; top: 1px;', '--c: red; width: 2px;']);
      expect(setProperty).not.toHaveBeenCalled();
      top.set('2px');
      c.set('blue');
      flush();
      expect(setProperty.mock.calls).toEqual([
        ['top', '2px', undefined],
        ['--c', 'blue', undefined],
      ]);
      expect(texts.length).toBe(2);
      expect(container.querySelector('i')!.style.top).toBe('2px');
      unmount();
    } finally {
      Object.defineProperty(proto, 'cssText', cssText);
      setProperty.mockRestore();
    }
  });
});

describe("a component's own effects run after what it renders (Svelte defers them to its mount)", () => {
  it("children's effects and actions first, then the parent's; effects inside effects are not deferred", () => {
    const order: string[] = [];
    const act = (_: Element, n: string) => void order.push(`action ${n}`);
    const Child = (p: { n: string }) => {
      effect(() => void order.push(`child ${p.n}`));
      return <b use:act={p.n} />;
    };
    const Parent = () => {
      effect(() => void order.push('parent'));
      onMount(() => void order.push('parent mount'));
      return (
        <div use:act={'parent-div'}>
          <Child n="a" />
          <Child n="b" />
        </div>
      );
    };
    const { unmount } = render(() => <Parent />);
    flush();
    expect(order).toEqual(['action parent-div', 'action a', 'child a', 'action b', 'child b', 'parent', 'parent mount']);
    unmount();
  });
});

describe('effects run in tree order, as Svelte walks its effect tree', () => {
  it('an effect in a branch created late runs at the branch place, before the effects of the component around it', () => {
    const order: string[] = [];
    const data = signal(0);
    const shown = signal(false);
    const Dialog = () => {
      effect(() => void order.push(`dialog ${data.get()}`));
      return <i />;
    };
    const Page = () => {
      effect(() => void order.push(`page ${data.get()}`));
      return <div>{shown.get() ? <Dialog /> : null}</div>;
    };
    const { unmount } = render(() => <Page />);
    flush();
    shown.set(true);
    flush();
    order.length = 0;
    data.set(1);
    flush();
    expect(order).toEqual(['dialog 1', 'page 1']);
    unmount();
  });
});
