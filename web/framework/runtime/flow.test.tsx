// @vitest-environment happy-dom
import { describe, expect, it, vi } from 'vitest';
import { For, Index, Key, Match, Show, Switch, effect, flush, onCleanup, signal } from 'fw';
import { render } from '../testing';

const text = (el: Element) => el.textContent;

describe('Show', () => {
  it('creates on truthy, disposes on falsy, keeps content while it stays truthy', () => {
    const v = signal<number | null>(1);
    const created = vi.fn();
    const disposed = vi.fn();
    const Inner = (p: { v: () => number }) => {
      created();
      onCleanup(disposed);
      return <b>{p.v()}</b>;
    };
    const { container, unmount } = render(() => (
      <div>
        <Show when={v.get()} fallback={<i>none</i>}>
          {(n) => <Inner v={n} />}
        </Show>
      </div>
    ));
    expect(text(container)).toBe('1');
    const b = container.querySelector('b');
    v.set(2);
    flush();
    expect(text(container)).toBe('2');
    expect(container.querySelector('b')).toBe(b);
    expect(created).toHaveBeenCalledTimes(1);
    v.set(null);
    flush();
    expect(text(container)).toBe('none');
    expect(disposed).toHaveBeenCalledTimes(1);
    unmount();
  });

  it('keyed re-creates on every value change', () => {
    const v = signal('a');
    const { container, unmount } = render(() => (
      <Show when={v.get()} keyed>
        {(x) => <b>{x()}</b>}
      </Show>
    ));
    const b = container.querySelector('b');
    v.set('b');
    flush();
    expect(text(container)).toBe('b');
    expect(container.querySelector('b')).not.toBe(b);
    unmount();
  });

  it('nested Show inside For: inner effects die with their row', () => {
    const rows = signal([1, 2, 3]);
    const flag = signal(true);
    const runs: number[] = [];
    const { container, unmount } = render(() => (
      <ul>
        <For each={rows.get()}>
          {(r) => (
            <li>
              <Show when={flag.get()}>
                {() => {
                  effect(() => void runs.push(r() * (flag.get() ? 1 : -1)));
                  return <span>{r()}</span>;
                }}
              </Show>
            </li>
          )}
        </For>
      </ul>
    ));
    flush();
    expect(runs).toEqual([1, 2, 3]);
    rows.set([1, 3]);
    flush();
    flag.set(false);
    flush();
    expect(runs).toEqual([1, 2, 3]);
    expect(container.querySelectorAll('span').length).toBe(0);
    unmount();
  });
});

describe('conditionals in JSX', () => {
  it('a ternary re-creates its branch only when the condition flips', () => {
    const n = signal(1);
    const made = vi.fn();
    const Big = () => {
      made();
      return <b>big</b>;
    };
    const { container, unmount } = render(() => <div>{n.get() > 5 ? <Big /> : n.get() > 2 ? 'mid' : <i>{n.get()}</i>}</div>);
    expect(text(container)).toBe('1');
    n.set(2);
    flush();
    expect(text(container)).toBe('2');
    n.set(3);
    flush();
    expect(text(container)).toBe('mid');
    n.set(6);
    flush();
    n.set(7);
    flush();
    expect(made).toHaveBeenCalledTimes(1);
    unmount();
  });

  it('a branch in a fragment next to changing text is not re-created', () => {
    const t = signal('a');
    const made = vi.fn();
    const C = () => {
      made();
      return <b />;
    };
    const Frag = () => (
      <>
        {t.get()}
        {t.get() !== 'zzz' && <C />}
      </>
    );
    const { unmount } = render(() => (
      <div>
        <Frag />
      </div>
    ));
    t.set('b');
    flush();
    t.set('c');
    flush();
    expect(made).toHaveBeenCalledTimes(1);
    unmount();
  });
});

describe('For', () => {
  const setup = (initial: { id: number; label: string }[]) => {
    const list = signal(initial);
    const r = render(() => (
      <ul>
        <For each={list.get()} key={(x) => x.id} fallback={<li>empty</li>}>
          {(item, i) => (
            <li>
              {i()}:{item().label}
            </li>
          )}
        </For>
      </ul>
    ));
    return { list, ...r, lis: () => [...r.container.querySelectorAll('li')] };
  };

  it('keeps DOM nodes by key, moves them, updates items and indices', () => {
    const items = [1, 2, 3, 4, 5].map((id) => ({ id, label: `L${id}` }));
    const { list, lis, container, unmount } = setup(items);
    const before = lis();
    list.set([items[4]!, items[1]!, items[2]!, items[3]!, items[0]!]);
    flush();
    const after = lis();
    expect(after[0]).toBe(before[4]);
    expect(after[4]).toBe(before[0]);
    expect(after[1]).toBe(before[1]);
    expect(text(container)).toBe('0:L51:L22:L33:L44:L1');
    // Same key, new object: the row stays and shows the new item.
    list.set([{ id: 5, label: 'five' }, ...list.peek().slice(1)]);
    flush();
    expect(lis()[0]).toBe(before[4]);
    expect(lis()[0]!.textContent).toBe('0:five');
    list.set([]);
    flush();
    expect(text(container)).toBe('empty');
    list.set(items.slice(0, 2));
    flush();
    expect(text(container)).toBe('0:L11:L2');
    unmount();
  });

  it('moves with the fewest DOM operations (swap = 2 moves)', () => {
    const items = Array.from({ length: 20 }, (_, i) => ({ id: i, label: String(i) }));
    const { list, container, unmount } = setup(items);
    const ul = container.querySelector('ul')!;
    const spy = vi.spyOn(ul, 'insertBefore');
    const next = items.slice();
    [next[1], next[18]] = [next[18]!, next[1]!];
    list.set(next);
    flush();
    expect(spy).toHaveBeenCalledTimes(2);
    expect(text(container)).toBe(next.map((x, i) => `${i}:${x.label}`).join(''));
    unmount();
  });

  it('random reorders end in the right order', () => {
    const items = Array.from({ length: 30 }, (_, i) => ({ id: i, label: String(i) }));
    const { list, lis, unmount } = setup(items);
    let seed = 7;
    const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
    for (let round = 0; round < 20; round++) {
      const next = items.filter(() => rand() > 0.3).sort(() => rand() - 0.5);
      list.set(next);
      flush();
      expect(lis().map((li) => li.textContent)).toEqual(next.map((x, i) => `${i}:${x.label}`));
    }
    unmount();
  });

  it('keeps focus on a moved row', () => {
    const list = signal([1, 2, 3]);
    const { container, unmount } = render(() => (
      <div>
        <For each={list.get()}>{(n) => <button>{n()}</button>}</For>
      </div>
    ));
    const third = container.querySelectorAll('button')[2]!;
    third.focus();
    list.set([3, 1, 2]);
    flush();
    expect(container.querySelectorAll('button')[0]).toBe(third);
    expect(document.activeElement).toBe(third);
    unmount();
  });

  it('disposes removed rows and everything on unmount', () => {
    const list = signal([1, 2, 3]);
    const disposed: number[] = [];
    const { unmount } = render(() => (
      <For each={list.get()}>
        {(n) => {
          onCleanup(() => disposed.push(n()));
          return <i>{n()}</i>;
        }}
      </For>
    ));
    list.set([1, 3]);
    flush();
    expect(disposed).toEqual([2]);
    unmount();
    expect(disposed.sort()).toEqual([1, 2, 3]);
  });

  it('handles duplicate identities', () => {
    const list = signal(['a', 'b', 'a']);
    const { container, unmount } = render(() => (
      <p>
        <For each={list.get()}>{(x) => <i>{x()}</i>}</For>
      </p>
    ));
    expect(text(container)).toBe('aba');
    list.set(['a', 'a', 'b', 'a']);
    flush();
    expect(text(container)).toBe('aaba');
    unmount();
  });
});

describe('Index', () => {
  it('keeps rows by position and updates their items', () => {
    const list = signal(['a', 'b']);
    const { container, unmount } = render(() => (
      <p>
        <Index each={list.get()}>{(x, i) => <i>{i}{x()}</i>}</Index>
      </p>
    ));
    const first = container.querySelector('i');
    list.set(['z', 'b', 'c']);
    flush();
    expect(text(container)).toBe('0z1b2c');
    expect(container.querySelector('i')).toBe(first);
    list.set(['q']);
    flush();
    expect(text(container)).toBe('0q');
    unmount();
  });
});

describe('Switch / Key', () => {
  it('Switch shows the first matching case', () => {
    const n = signal(1);
    const { container, unmount } = render(() => (
      <Switch fallback={<i>other</i>}>
        <Match when={n.get() === 1}>one</Match>
        <Match when={n.get() === 2}>two</Match>
      </Switch>
    ));
    expect(text(container)).toBe('one');
    n.set(2);
    flush();
    expect(text(container)).toBe('two');
    n.set(3);
    flush();
    expect(text(container)).toBe('other');
    unmount();
  });

  it('Key re-creates its content when the key changes', () => {
    const k = signal('a');
    const other = signal(0);
    const made = vi.fn();
    const C = () => {
      made();
      return <b>{other.get()}</b>;
    };
    const { unmount } = render(() => (
      <Key when={k.get()}>
        <C />
      </Key>
    ));
    other.set(1);
    flush();
    expect(made).toHaveBeenCalledTimes(1);
    k.set('b');
    flush();
    expect(made).toHaveBeenCalledTimes(2);
    unmount();
  });
});
