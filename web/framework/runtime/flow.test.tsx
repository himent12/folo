// @vitest-environment happy-dom
import { describe, expect, it, vi } from 'vitest';
import { For, Index, Key, Match, Portal, Show, Switch, deep, effect, flush, onCleanup, signal } from 'fw';
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

  it('hoists focus check out of row moves when focus is outside the list and preserves nested focus on swaps and reorders', () => {
    const list = signal([1, 2, 3, 4, 5]);
    const { container, unmount } = render(() => (
      <section>
        <button id="outside">outside</button>
        <ul>
          <For each={list.get()}>
            {(n) => (
              <li>
                <input value={String(n())} />
              </li>
            )}
          </For>
        </ul>
      </section>
    ));
    const outside = container.querySelector<HTMLButtonElement>('#outside')!;
    const ul = container.querySelector('ul')!;
    const rowSpies = [...ul.querySelectorAll('li')].map((li) => vi.spyOn(li, 'contains'));
    const parentSpy = vi.spyOn(ul, 'contains');

    // Focus outside the list: LIS reorder checks parent.contains once, skips per-row walks, and keeps focus outside.
    outside.focus();
    list.set([5, 3, 1, 4, 2]);
    flush();
    expect(parentSpy).toHaveBeenCalledTimes(1);
    expect(document.activeElement).toBe(outside);
    for (const s of rowSpies) expect(s).not.toHaveBeenCalled();

    // End-for-end swap with focus still outside: checks parent.contains once, zero per-row contains walks.
    parentSpy.mockClear();
    list.set([2, 3, 1, 4, 5]);
    flush();
    expect(parentSpy).toHaveBeenCalledTimes(1);
    expect(document.activeElement).toBe(outside);
    for (const s of rowSpies) expect(s).not.toHaveBeenCalled();

    // Focus on document.body or on the parent itself: never calls parent.contains.
    parentSpy.mockClear();
    outside.blur();
    list.set([5, 3, 1, 4, 2]);
    flush();
    expect(parentSpy).not.toHaveBeenCalled();

    ul.tabIndex = -1;
    ul.focus();
    parentSpy.mockClear();
    list.set([2, 3, 1, 4, 5]);
    flush();
    expect(parentSpy).not.toHaveBeenCalled();
    expect(document.activeElement).toBe(ul);

    // Pure append / remove: never calls parent.contains at all.
    parentSpy.mockClear();
    outside.focus();
    list.set([2, 3, 1, 4, 5, 6]);
    flush();
    list.set([2, 3, 1, 4, 5]);
    flush();
    expect(parentSpy).not.toHaveBeenCalled();

    // Focus inside a deeply nested input: preserved across both end-for-end swap and LIS reorder.
    const targetInput = ul.querySelectorAll('input')[0]!;
    targetInput.focus();
    list.set([5, 3, 1, 4, 2]);
    flush();
    expect(document.activeElement).toBe(targetInput);
    list.set([4, 2, 5, 1, 3]);
    flush();
    expect(document.activeElement).toBe(targetInput);

    // Focus inside a removed row during a mixed remove + reorder: detached input is not refocused.
    const doomed = ul.querySelectorAll('input')[0]!;
    doomed.focus();
    list.set([3, 1, 5, 2]);
    flush();
    expect(document.activeElement).not.toBe(doomed);

    parentSpy.mockRestore();
    for (const s of rowSpies) s.mockRestore();
    unmount();
  });

  it('preserves focus redirected to another moved row by a focus listener during swap', () => {
    const list = signal(['A', 'M', 'B']);
    const { container, unmount } = render(() => (
      <ul>
        <For each={list.get()}>
          {(id) => (
            <li>
              <input data-id={id()} />
            </li>
          )}
        </For>
      </ul>
    ));
    const inputA = container.querySelector<HTMLInputElement>('input[data-id="A"]')!;
    const inputB = container.querySelector<HTMLInputElement>('input[data-id="B"]')!;
    inputB.focus();
    // When moving B restores focus on B, its focus listener redirects focus to A before A moves.
    inputB.addEventListener('focus', () => inputA.focus());
    list.set(['B', 'M', 'A']);
    flush();
    expect(document.activeElement).toBe(inputA);
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
    // Reorder duplicates past a shared prefix so `chain` offsets (`j - start`) are exercised,
    // and assert surviving duplicate nodes keep their exact DOM identities in order.
    list.set(['a', 'x', 'a', 'b', 'y', 'a']);
    flush();
    expect(text(container)).toBe('axabya');
    const before = [...container.querySelectorAll('i')];
    list.set(['a', 'b', 'a', 'x', 'a', 'y']);
    flush();
    expect(text(container)).toBe('abaxay');
    const after = [...container.querySelectorAll('i')];
    expect(after[0]).toBe(before[0]);
    expect(after[1]).toBe(before[3]);
    expect(after[2]).toBe(before[2]);
    expect(after[3]).toBe(before[1]);
    expect(after[4]).toBe(before[5]);
    expect(after[5]).toBe(before[4]);
    unmount();
  });

  it('fast-paths pure append, prepend, middle insert, and pure removals while keeping surviving nodes', () => {
    const mk = (id: number) => ({ id, label: `L${id}` });
    const { list, lis, container, unmount } = setup([mk(2), mk(3)]);
    const orig = lis();

    // Pure append
    list.set([mk(2), mk(3), mk(5), mk(6)]);
    flush();
    expect(lis().slice(0, 2)).toEqual(orig);
    expect(text(container)).toBe('0:L21:L32:L53:L6');

    // Pure prepend
    const beforePrepend = lis();
    list.set([mk(0), mk(1), mk(2), mk(3), mk(5), mk(6)]);
    flush();
    expect(lis().slice(2)).toEqual(beforePrepend);
    expect(text(container)).toBe('0:L01:L12:L23:L34:L55:L6');

    // Pure middle insert
    const beforeMid = lis();
    list.set([mk(0), mk(1), mk(2), mk(3), mk(4), mk(5), mk(6)]);
    flush();
    expect(lis().slice(0, 4)).toEqual(beforeMid.slice(0, 4));
    expect(lis().slice(5)).toEqual(beforeMid.slice(4));
    expect(text(container)).toBe('0:L01:L12:L23:L34:L45:L56:L6');

    // Pure removals (middle, start, end)
    const beforeRemove = lis();
    list.set([mk(0), mk(1), mk(5), mk(6)]);
    flush();
    expect(lis()).toEqual([beforeRemove[0], beforeRemove[1], beforeRemove[5], beforeRemove[6]]);
    expect(text(container)).toBe('0:L01:L12:L53:L6');

    list.set([mk(5)]);
    flush();
    expect(lis()).toEqual([beforeRemove[5]]);
    expect(text(container)).toBe('0:L5');
    unmount();
  });

  it('stores function items in keyed For without invoking them as signal updaters', () => {
    const f1 = vi.fn(() => 'one');
    const f2 = vi.fn(() => 'two');
    const list = signal<Array<() => string>>([f1]);
    const { container, unmount } = render(() => (
      <p>
        <For each={list.get()} key={(_, i) => i}>
          {(fn) => <i>{fn()()}</i>}
        </For>
      </p>
    ));
    expect(text(container)).toBe('one');
    expect(f1).toHaveBeenCalledTimes(1);
    list.set([f2]);
    flush();
    expect(text(container)).toBe('two');
    expect(f1).toHaveBeenCalledTimes(1);
    expect(f2).toHaveBeenCalledTimes(1);
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

  it('skips DOM reconciliation on same-length updates and handles function items and fallback transitions', () => {
    const f1 = vi.fn(() => 'f1');
    const f2 = vi.fn(() => 'f2');
    const list = signal<Array<() => string>>([]);
    const disposed: number[] = [];
    const { container, unmount } = render(() => (
      <p>
        <Index each={list.get()} fallback={<em>empty</em>}>
          {(fn, i) => {
            onCleanup(() => disposed.push(i));
            return <i>{fn()()}</i>;
          }}
        </Index>
      </p>
    ));
    const p = container.querySelector('p')!;
    expect(text(p)).toBe('empty');

    // Empty -> empty keeps fallback without DOM churn.
    const insertSpy = vi.spyOn(p, 'insertBefore');
    list.set([]);
    flush();
    expect(insertSpy).not.toHaveBeenCalled();

    // Empty -> populated
    list.set([f1, f2]);
    flush();
    expect(text(p)).toBe('f1f2');
    expect(f1).toHaveBeenCalledTimes(1);
    expect(f2).toHaveBeenCalledTimes(1);

    // Same-length update: updates row signals in place without reconciling `p`'s children.
    insertSpy.mockClear();
    const removeSpy = vi.spyOn(p, 'removeChild');
    list.set([f2, f1]);
    flush();
    expect(text(p)).toBe('f2f1');
    expect(insertSpy).not.toHaveBeenCalled();
    expect(removeSpy).not.toHaveBeenCalled();
    expect(f1).toHaveBeenCalledTimes(2);
    expect(f2).toHaveBeenCalledTimes(2);

    // Shrink disposes only truncated indices, empty restores fallback.
    list.set([f1]);
    flush();
    expect(disposed).toEqual([1]);
    list.set([]);
    flush();
    expect(disposed).toEqual([1, 0]);
    expect(text(p)).toBe('empty');

    insertSpy.mockRestore();
    removeSpy.mockRestore();
    unmount();
  });

  it('returns the same cached array reference on same-length and empty-to-empty updates', () => {
    const list = signal<string[]>([]);
    const seen: unknown[] = [];
    const reader = Index({
      get each() {
        return list.get();
      },
      fallback: 'empty',
      children: (item) => item,
    }) as () => unknown[];
    const { unmount } = render(() => {
      effect(() => void seen.push(reader()));
      return null;
    });
    flush();
    const emptyArr = reader();
    expect(seen).toEqual([emptyArr]);

    // Empty -> empty keeps exact array identity and does not notify observers.
    list.set([]);
    flush();
    expect(reader()).toBe(emptyArr);
    expect(seen).toHaveLength(1);

    // Populated -> same-length update keeps exact array identity and does not notify observers.
    list.set(['a', 'b']);
    flush();
    const twoArr = reader();
    expect(twoArr).not.toBe(emptyArr);
    expect(seen).toHaveLength(2);
    list.set(['x', 'y']);
    flush();
    expect(reader()).toBe(twoArr);
    expect(seen).toHaveLength(2);

    unmount();
  });

  it('rebuilds output cleanly after partial row creation or fallback errors', () => {
    const list = signal<string[]>([]);
    const { container, unmount } = render(() => (
      <p>
        <Index each={list.get()} fallback={<em>empty</em>}>
          {(item, i) => {
            if (item() === 'boom') throw new Error(`row ${i}`);
            return <i>{item()}</i>;
          }}
        </Index>
      </p>
    ));
    expect(text(container)).toBe('empty');

    // Partial creation failure: row 0 succeeds while row 1 throws (leaving rows.length === 1 and out = [<em>]).
    list.set(['ok', 'boom']);
    expect(() => flush()).toThrow('row 1');

    // Recovering to length 1 must rebuild output with row 0 instead of reusing the disposed 1-element fallback.
    list.set(['ok']);
    flush();
    expect(text(container)).toBe('ok');
    expect(container.querySelector('em')).toBeNull();
    expect(container.querySelector('i')?.textContent).toBe('ok');
    unmount();
  });
});

describe('lists of a deep() array', () => {
  it('For and Index follow in-place mutations (push, index writes, splice)', () => {
    const list = deep(['a']);
    const { container, unmount } = render(() => (
      <div>
        <ul>
          <For each={list}>{(x) => <li>{x()}</li>}</For>
        </ul>
        <ol>
          <Index each={list}>{(x) => <li>{x()}</li>}</Index>
        </ol>
      </div>
    ));
    const ul = container.querySelector('ul')!;
    const ol = container.querySelector('ol')!;
    list.push('b');
    flush();
    expect(text(ul)).toBe('ab');
    expect(text(ol)).toBe('ab');
    list[0] = 'c';
    flush();
    expect(text(ul)).toBe('cb');
    expect(text(ol)).toBe('cb');
    list.splice(0, 1);
    flush();
    expect(text(ul)).toBe('b');
    expect(text(ol)).toBe('b');
    unmount();
  });

  it('a keyed For follows a key read from a deep() row', () => {
    const list = deep([{ id: 1, name: 'a' }]);
    const made = vi.fn();
    const { container, unmount } = render(() => (
      <ul>
        <For each={list} key={(r) => r.id}>
          {(r) => {
            made();
            return <li>{r().name}</li>;
          }}
        </For>
      </ul>
    ));
    list[0]!.id = 2;
    flush();
    expect(made).toHaveBeenCalledTimes(2);
    expect(text(container)).toBe('a');
    unmount();
  });
});

describe('Portal', () => {
  it('takes its nodes out of the mount when it goes', () => {
    const m = document.body.appendChild(document.createElement('div'));
    const { unmount } = render(() => (
      <Portal mount={m}>
        <b>portal</b>
      </Portal>
    ));
    expect(m.textContent).toBe('portal');
    unmount();
    expect(m.textContent).toBe('');
    m.remove();
  });

  it("removes exactly its own nodes, keeping the mount's other content, also when a branch hides it", () => {
    const m = document.body.appendChild(document.createElement('div'));
    m.append('before ');
    const shown = signal(true);
    const label = signal('one');
    const { unmount } = render(() => (
      <Show when={shown.get()}>
        <Portal mount={m}>
          <b>{label.get()}</b>
          <i>{label.get()}</i>
        </Portal>
      </Show>
    ));
    m.append(' after');
    expect(m.textContent).toBe('before oneone after');
    label.set('two');
    flush();
    expect(m.textContent).toBe('before twotwo after');
    shown.set(false);
    flush();
    expect(m.textContent).toBe('before  after');
    shown.set(true);
    flush();
    expect(m.textContent).toBe('before  aftertwotwo');
    unmount();
    expect(m.textContent).toBe('before  after');
    m.remove();
  });
});

describe('Portal outros', () => {
  it('a node leaving with an outro inside a portal goes with the portal, and only it', async () => {
    const slow = () => ({ duration: 50, css: (t: number) => `opacity: ${t}` });
    const m = document.body.appendChild(document.createElement('div'));
    m.append('keep');
    const shown = signal(true);
    const { unmount } = render(() => (
      <Portal mount={m}>
        <Show when={shown.get()}>
          <b out:slow={{}}>leaving</b>
        </Show>
      </Portal>
    ));
    expect(m.textContent).toBe('keepleaving');
    shown.set(false);
    flush();
    // The outro runs: the node stays for now.
    expect(m.querySelector('b')).not.toBeNull();
    unmount();
    expect(m.querySelector('b')).toBeNull();
    expect(m.textContent).toBe('keep');
    await new Promise((r) => setTimeout(r, 80));
    expect(m.textContent).toBe('keep');
    m.remove();
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
