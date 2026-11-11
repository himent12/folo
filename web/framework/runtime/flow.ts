// Control flow: Show ({#if}), For ({#each} with a key), Index ({#each} without one), Switch/Match ({#if}/{:else if}),
// Key ({#key}), Dynamic (<svelte:element> / a component chosen at runtime), Portal.

import { createRoot, computed, getOwner, isReader, memo, onCleanup, owning, reader, signal, untrack, type Owner, type ReadSignal, type Reader, type Signal } from './core';
import { createComponent, splitProps } from './component';
import { insert, setLeave } from './dom';
import { spread } from './spread';
import { inBranch, leaveBranch, type Branch } from './transition';

type Falsy = false | 0 | '' | null | undefined;

/** Creates a list's rows now, in document order, as Svelte does (an error is kept for the first real read). */
function eager(list: { peek(): unknown }): void {
  try {
    list.peek();
  } catch {
    // Thrown again where the list is read.
  }
}

/** A child function's result as stored: a snippet result that is itself a function is wrapped once (see `memo`). */
function settle(v: JSX.Element): JSX.Element {
  return typeof v === 'function' && !isReader(v) ? memo(v as () => JSX.Element) : v;
}

// ---------------------------------------------------------------------------------------------------------------------
// Show

export interface ShowProps<T> {
  /** Shows `children` while truthy. */
  when: T | Falsy;
  /** Re-create the content whenever `when` changes value, not only when it turns truthy (default false). */
  keyed?: boolean;
  /** Shown while `when` is falsy (port of `{:else}`). */
  fallback?: JSX.Element;
  /** The content, or a function of the (narrowed) value. */
  children: JSX.Element | ((value: () => NonNullable<Exclude<T, Falsy>>) => JSX.Element);
}

/**
 * `{#if when}children{:else}fallback{/if}`. The content is created when `when` turns truthy and disposed when it turns
 * falsy; in between, a function child reads the current value through its accessor.
 */
export function Show<T>(props: ShowProps<T>): JSX.Element {
  const keyed = props.keyed;
  const value = computed(() => props.when);
  const cond = keyed ? value : computed(() => value.get(), { equals: (a, b) => !a === !b });
  let ran = false;
  let prev: { b: Branch; v: JSX.Element } | null = null;
  return memo(() => {
    const c = cond.get();
    const b: Branch = { t: null };
    if (prev !== null) leaveBranch(prev.b, prev.v);
    const v = inBranch(b, ran, () => {
      if (!c) return untrack(() => props.fallback);
      const ch = props.children;
      return untrack(() =>
        typeof ch === 'function' && !isReader(ch)
          ? (ch as (v: () => unknown) => JSX.Element)(keyed ? () => c : () => value.get())
          : (ch as JSX.Element),
      );
    });
    ran = true;
    prev = { b, v };
    return settle(v);
  });
}

// ---------------------------------------------------------------------------------------------------------------------
// For

export interface ForProps<T> {
  each: readonly T[] | null | undefined | false;
  /** A row's identity (port of `{#each list as item (key)}`); default: the item itself. */
  key?: (item: T, index: number) => unknown;
  /** Shown while the list is empty (port of `{:else}`). */
  fallback?: JSX.Element;
  /** Renders a row. `item()` is the row's current item (a keyed row keeps its DOM when its item is replaced). */
  children: (item: () => T, index: () => number) => JSX.Element;
}

interface Row<T> {
  key: unknown;
  item: T;
  sig: Signal<T> | null;
  idx: Signal<number> | null;
  dispose: () => void;
  value: JSX.Element;
  b: Branch;
}

/**
 * A keyed list (port of `{#each list as item, i (key)}`). Rows are matched by key; a kept row keeps its DOM nodes (and
 * focus) and only moves, with the fewest moves; a row whose item was replaced under the same key gets the new item.
 */
export function For<T>(props: ForProps<T>): JSX.Element {
  const parent = getOwner();
  const keyFn = props.key;
  const fn = props.children;
  const withIndex = fn.length > 1;
  let rows: Row<T>[] = [];
  let fallback: { dispose: () => void; value: JSX.Element } | null = null;
  let ran = false;

  const make = (item: T, i: number, key: unknown): Row<T> =>
    createRoot((dispose) => {
      const sig = keyFn ? signal(item) : null;
      const idx = withIndex ? signal(i) : null;
      const row: Row<T> = { key, item, sig, idx, dispose, value: undefined, b: { t: null } };
      row.value = settle(
        inBranch(row.b, ran, () =>
          fn(sig ? () => sig.get() : () => item, idx ? () => idx.get() : () => i),
        ),
      );
      return row;
    }, parent);

  const drop = (row: Row<T>) => {
    leaveBranch(row.b, row.value);
    row.dispose();
  };

  const update = (row: Row<T>, item: T, i: number) => {
    if (row.item !== item) {
      row.item = item;
      row.sig?.set(item);
    }
    row.idx?.set(i);
  };

  const mapped: ReadSignal<JSX.Element[]> = computed(
    () => {
      // The list's length and entries (and the keys) are read tracked, so a `deep()` array changed in place (`push`,
      // `list[i] = x`) updates the rows; building, updating and dropping rows is untracked.
      const list = props.each || [];
      const n = list.length;
      const keys = new Array<unknown>(n);
      for (let j = 0; j < n; j++) keys[j] = keyFn ? keyFn(list[j]!, j) : list[j];
      return untrack(() => {
        if (n === 0) {
          for (const r of rows) drop(r);
          rows = [];
          if (fallback === null && props.fallback !== undefined) {
            fallback = createRoot((dispose) => ({ dispose, value: settle(inBranch(null, ran, () => props.fallback)) }), parent);
          }
          ran = true;
          return fallback !== null ? [fallback.value] : [];
        }
        if (fallback !== null) {
          fallback.dispose();
          fallback = null;
        }
        const next = new Array<Row<T>>(n);
        if (rows.length === 0) {
          for (let j = 0; j < n; j++) next[j] = make(list[j]!, j, keys[j]);
        } else {
          let start = 0;
          let end = rows.length - 1;
          let newEnd = n - 1;
          const min = Math.min(rows.length, n);
          for (;;) {
            while (start < min && start <= end && start <= newEnd && rows[start]!.key === keys[start]) {
              update((next[start] = rows[start]!), list[start]!, start);
              start++;
            }
            while (end >= start && newEnd >= start && rows[end]!.key === keys[newEnd]) {
              update((next[newEnd] = rows[end]!), list[newEnd]!, newEnd);
              end--;
              newEnd--;
            }
            // The first and last rows swapped: match them directly.
            if (end > start && newEnd > start && rows[start]!.key === keys[newEnd] && rows[end]!.key === keys[start]) {
              update((next[newEnd] = rows[start]!), list[newEnd]!, newEnd);
              update((next[start] = rows[end]!), list[start]!, start);
              start++;
              end--;
              newEnd--;
              continue;
            }
            break;
          }
          // Middle: match the rest by key (duplicates chained in order).
          const first = new Map<unknown, number>();
          const chain = new Int32Array(newEnd + 1 > start ? newEnd + 1 : 0);
          for (let j = newEnd; j >= start; j--) {
            const k = keys[j];
            const at = first.get(k);
            chain[j] = at === undefined ? -1 : at;
            first.set(k, j);
          }
          for (let i = start; i <= end; i++) {
            const row = rows[i]!;
            const j = first.get(row.key);
            if (j !== undefined && j !== -1) {
              next[j] = row;
              const c = chain[j]!;
              if (c === -1) first.delete(row.key);
              else first.set(row.key, c);
            } else drop(row);
          }
          for (let j = start; j <= newEnd; j++) {
            const row = next[j];
            if (row !== undefined) update(row, list[j]!, j);
            else next[j] = make(list[j]!, j, keys[j]);
          }
        }
        rows = next;
        ran = true;
        const out = new Array<JSX.Element>(n);
        for (let j = 0; j < n; j++) out[j] = next[j]!.value;
        return out;
      });
    },
    { equals: false },
  );
  owning(mapped);
  eager(mapped);
  return reader(() => mapped.get());
}

// ---------------------------------------------------------------------------------------------------------------------
// Index

export interface IndexProps<T> {
  each: readonly T[] | null | undefined | false;
  fallback?: JSX.Element;
  /** Renders the row at `index`; `item()` is its current item (port of `{#each list as item, i}` without a key). */
  children: (item: () => T, index: number) => JSX.Element;
}

/** A list by position (port of an unkeyed `{#each}`): row `i` stays and its item updates in place. */
export function Index<T>(props: IndexProps<T>): JSX.Element {
  const parent = getOwner();
  const fn = props.children;
  let rows: { sig: Signal<T>; dispose: () => void; value: JSX.Element; b: Branch }[] = [];
  let fallback: { dispose: () => void; value: JSX.Element } | null = null;
  let ran = false;
  const mapped: ReadSignal<JSX.Element[]> = computed(
    () => {
      // Length and entries tracked (a `deep()` array changed in place updates the rows), the rows' upkeep untracked.
      const list = props.each || [];
      const n = list.length;
      const items = new Array<T>(n);
      for (let i = 0; i < n; i++) items[i] = list[i]!;
      return untrack(() => {
        for (let i = 0; i < Math.min(n, rows.length); i++) rows[i]!.sig.set(() => items[i]!);
        for (let i = rows.length; i < n; i++) {
          rows.push(
            createRoot((dispose) => {
              const sig = signal<T>(items[i]!);
              const b: Branch = { t: null };
              return { sig, dispose, b, value: settle(inBranch(b, ran, () => fn(() => sig.get(), i))) };
            }, parent),
          );
        }
        for (let i = n; i < rows.length; i++) {
          leaveBranch(rows[i]!.b, rows[i]!.value);
          rows[i]!.dispose();
        }
        rows.length = n;
        if (n === 0 && fallback === null && props.fallback !== undefined) {
          fallback = createRoot((dispose) => ({ dispose, value: settle(inBranch(null, ran, () => props.fallback)) }), parent);
        } else if (n > 0 && fallback !== null) {
          fallback.dispose();
          fallback = null;
        }
        ran = true;
        return n === 0 ? (fallback !== null ? [fallback.value] : []) : rows.map((r) => r.value);
      });
    },
    { equals: false },
  );
  owning(mapped);
  eager(mapped);
  return reader(() => mapped.get());
}

// ---------------------------------------------------------------------------------------------------------------------
// Switch / Match

const MATCH = Symbol('fw.match');

interface MatchProps<T> {
  when: T | Falsy;
  children: JSX.Element | ((value: () => NonNullable<Exclude<T, Falsy>>) => JSX.Element);
}

/** One case of a `Switch`. */
export function Match<T>(props: MatchProps<T>): JSX.Element {
  return { [MATCH]: props } as unknown as JSX.Element;
}

/** The first `Match` whose `when` is truthy, else `fallback` (port of `{#if a}{:else if b}{:else}{/if}`). */
export function Switch(props: { fallback?: JSX.Element; children: JSX.Element }): JSX.Element {
  const cases: MatchProps<unknown>[] = [];
  const gather = (v: unknown): void => {
    while (typeof v === 'function') v = (v as () => unknown)();
    if (Array.isArray(v)) for (const x of v) gather(x);
    else if (v && typeof v === 'object' && MATCH in v) cases.push((v as Record<symbol, MatchProps<unknown>>)[MATCH]!);
  };
  untrack(() => gather(props.children));
  const which = computed(() => {
    for (let i = 0; i < cases.length; i++) if (cases[i]!.when) return i;
    return -1;
  });
  let ran = false;
  let prev: { b: Branch; v: JSX.Element } | null = null;
  return memo(() => {
    const i = which.get();
    const b: Branch = { t: null };
    if (prev !== null) leaveBranch(prev.b, prev.v);
    const v = inBranch(b, ran, () =>
      untrack(() => {
        if (i < 0) return props.fallback;
        const m = cases[i]!;
        const ch = m.children;
        return typeof ch === 'function' && !isReader(ch) ? (ch as (v: () => unknown) => JSX.Element)(() => m.when) : (ch as JSX.Element);
      }),
    );
    ran = true;
    prev = { b, v };
    return settle(v);
  });
}

// ---------------------------------------------------------------------------------------------------------------------
// Key

/** Re-creates its content whenever `when` changes (port of `{#key when}`); intros in it play on each re-creation. */
export function Key<T>(props: { when: T; children: JSX.Element }): JSX.Element {
  const key = computed(() => props.when);
  let ran = false;
  let prev: { b: Branch; v: JSX.Element } | null = null;
  return memo(() => {
    key.get();
    const b: Branch = { t: null };
    if (prev !== null) leaveBranch(prev.b, prev.v);
    const v = inBranch(b, ran, () =>
      untrack(() => {
        const ch = props.children;
        // A snippet is created here, inside the branch (its intros play, its outros are this branch's).
        return typeof ch === 'function' && !isReader(ch) ? (ch as () => JSX.Element)() : ch;
      }),
    );
    ran = true;
    prev = { b, v };
    return settle(v);
  });
}

// ---------------------------------------------------------------------------------------------------------------------
// Dynamic

const SVG_TAGS = new Set(['svg', 'g', 'path', 'circle', 'rect', 'line', 'polyline', 'polygon', 'ellipse', 'text', 'tspan', 'defs', 'use', 'symbol', 'clipPath', 'mask', 'pattern', 'linearGradient', 'radialGradient', 'stop', 'marker', 'foreignObject']);

export type DynamicProps<P> = { component: ((props: P) => JSX.Element) | string | null | undefined | false; $$scope?: string } & Record<string, unknown>;

/**
 * A component, or an element by tag name, chosen at runtime (port of `<svelte:element this={tag}>` and
 * `<svelte:component>`). The other props go to it; for a tag they are its attributes and `children` its content.
 */
export function Dynamic<P>(props: DynamicProps<P>): JSX.Element {
  const [local, others] = splitProps(props, ['component', '$$scope']);
  const c = computed(() => local.component);
  return memo(() => {
    const comp = c.get();
    if (!comp) return undefined;
    if (typeof comp === 'function') return untrack(() => createComponent(comp, others as P));
    return untrack(() => {
      const el = SVG_TAGS.has(comp) ? document.createElementNS('http://www.w3.org/2000/svg', comp) : document.createElement(comp);
      const [kids, attrs] = splitProps(others as Record<string, unknown>, ['children']);
      spread(el, attrs, local.$$scope);
      insert(el, () => kids.children);
      return el;
    });
  });
}

// ---------------------------------------------------------------------------------------------------------------------
// Portal

/**
 * Renders its children into `mount` (default `document.body`) instead of in place, after what the mount holds. It
 * renders between two empty comments of its own, so everything between them is its own: when it goes, all of that
 * goes, nodes still playing an outro too, and what else the mount holds stays.
 */
export function Portal(props: { mount?: Node; children: JSX.Element }): JSX.Element {
  const mount = props.mount ?? document.body;
  const owner: Owner | null = getOwner();
  createRoot(() => {
    const start = mount.appendChild(document.createComment(''));
    const end = mount.appendChild(document.createComment(''));
    insert(mount, () => props.children, end);
    onCleanup(() => {
      const parent = start.parentNode;
      if (parent === null) return;
      for (let n = start.nextSibling; n !== null && n !== end; ) {
        const next: ChildNode | null = n.nextSibling;
        setLeave(n, undefined);
        parent.removeChild(n);
        n = next;
      }
      start.remove();
      end.remove();
    });
  }, owner);
  return undefined;
}

export type { Reader };
