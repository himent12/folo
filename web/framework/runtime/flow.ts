// Control flow: Show ({#if}), For ({#each} with a key), Index ({#each} without one), Switch/Match ({#if}/{:else if}),
// Key ({#key}), Dynamic (<svelte:element> / a component chosen at runtime), Portal.

import { computed, createRoot, dispose, getOwner, isReader, memo, onCleanup, ownedRoot, owning, reader, signal, swapOwner, untrack, type Owner, type ReadSignal, type Reader, type Signal } from './core';
import { createComponent, splitProps } from './component';
import { clearLeaves, insert } from './dom';
import { spread } from './spread';
import { enterBranch, exitBranch, leaveBranch, type BranchTransitions } from './transition';

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
  let prevT: BranchTransitions = null;
  let prevV: JSX.Element = undefined;
  return memo(() => {
    const c = cond.get();
    if (prevT !== null) {
      const t0 = prevT;
      const v0 = prevV;
      prevT = null;
      prevV = undefined;
      leaveBranch(t0, v0);
    }
    enterBranch(ran);
    let v: JSX.Element;
    let t: BranchTransitions = null;
    try {
      v = untrack(() => {
        if (!c) return props.fallback;
        const ch = props.children;
        return typeof ch === 'function' && !isReader(ch)
          ? (ch as (v: () => unknown) => JSX.Element)(keyed ? () => c : () => value.get())
          : (ch as JSX.Element);
      });
    } finally {
      t = exitBranch();
    }
    ran = true;
    prevT = t;
    prevV = v;
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
  root: Owner;
  value: JSX.Element;
  t: BranchTransitions;
}

const EMPTY_KEYS: readonly unknown[] = [];

/** SameValueZero key equality, matching `Map` (`NaN` equals `NaN`, `-0` equals `+0`). */
const sameKey = (a: unknown, b: unknown): boolean => a === b || (a !== a && b !== b);

/** Whether `rows[rowFrom..rowTo]` matches `keys[keyFrom..]` element-for-element with no key equal to `excluded`. */
function matchesShifted<T>(rows: readonly Row<T>[], keys: readonly unknown[], rowFrom: number, rowTo: number, keyFrom: number, excluded: unknown): boolean {
  for (let i = rowFrom, j = keyFrom; i <= rowTo; i++, j++) {
    const k = rows[i]!.key;
    if (!sameKey(k, keys[j]) || sameKey(k, excluded)) return false;
  }
  return true;
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
  let fallback: { root: Owner; value: JSX.Element } | null = null;
  let out: JSX.Element[] = [];
  let outDirty = false;
  let ran = false;

  const make = (item: T, i: number, key: unknown): Row<T> => {
    const root = ownedRoot(parent);
    const prevOwner = swapOwner(root);
    const sig = keyFn ? signal(item) : null;
    const idx = withIndex ? signal(i) : null;
    let value: JSX.Element;
    let t: BranchTransitions = null;
    try {
      enterBranch(ran);
      let raw: JSX.Element;
      try {
        raw = fn(sig ? () => sig.get() : () => item, idx ? () => idx.get() : () => i);
      } finally {
        t = exitBranch();
      }
      value = settle(raw);
    } catch (e) {
      try {
        dispose(root);
      } catch {
        // Keep the row error.
      }
      throw e;
    } finally {
      swapOwner(prevOwner);
    }
    return { key, item, sig, idx, root, value, t };
  };

  const makeFallback = (): { root: Owner; value: JSX.Element } => {
    const root = ownedRoot(parent);
    const prevOwner = swapOwner(root);
    let value: JSX.Element;
    try {
      enterBranch(ran, false);
      let raw: JSX.Element;
      try {
        raw = props.fallback;
      } finally {
        exitBranch();
      }
      value = settle(raw);
    } catch (e) {
      try {
        dispose(root);
      } catch {
        // Keep the fallback error.
      }
      throw e;
    } finally {
      swapOwner(prevOwner);
    }
    return { root, value };
  };

  const drop = (row: Row<T>) => {
    if (row.t !== null) leaveBranch(row.t, row.value);
    dispose(row.root);
  };

  const update = (row: Row<T>, item: T, i: number) => {
    if (row.item !== item) {
      row.item = item;
      row.sig?.set(typeof item === 'function' ? () => item : item);
    }
    row.idx?.set(i);
  };

  const mapped: ReadSignal<JSX.Element[]> = computed(() => {
    // The list's length and entries (and the keys) are read tracked, so a `deep()` array changed in place (`push`,
    // `list[i] = x`) updates the rows; building, updating and dropping rows is untracked.
    const list = props.each || [];
    const n = list.length;
    let keys: readonly unknown[] = EMPTY_KEYS;
    if (n > 0) {
      const ks = new Array<unknown>(n);
      for (let j = 0; j < n; j++) ks[j] = keyFn ? keyFn(list[j]!, j) : list[j];
      keys = ks;
    }
    return untrack(() => {
      if (n === 0) {
        if (rows.length > 0) {
          for (let i = 0; i < rows.length; i++) drop(rows[i]!);
          rows = [];
          outDirty = true;
        }
        if (fallback === null && props.fallback !== undefined) {
          outDirty = true;
          fallback = makeFallback();
        }
        ran = true;
        if (outDirty) {
          out = fallback !== null ? [fallback.value] : [];
          outDirty = false;
        }
        return out;
      }
      if (fallback !== null) {
        const prevFallback = fallback;
        fallback = null;
        outDirty = true;
        dispose(prevFallback.root);
      }
      const prevLen = rows.length;
      if (prevLen === 0) {
        outDirty = true;
        const next = new Array<Row<T>>(n);
        const nextOut = new Array<JSX.Element>(n);
        let built = 0;
        try {
          for (let j = 0; j < n; j++) {
            const r = make(list[j]!, j, keys[j]);
            next[j] = r;
            nextOut[j] = r.value;
            built = j + 1;
          }
        } catch (e) {
          for (let j = 0; j < built; j++) {
            try {
              dispose(next[j]!.root);
            } catch {
              // Keep the original row error.
            }
          }
          throw e;
        }
        rows = next;
        ran = true;
        outDirty = false;
        return (out = nextOut);
      }
      // Scan common prefix before allocating `next` so identical-order updates, pure appends, and pure truncations
      // skip full array reconciliation.
      let start = 0;
      const min = Math.min(prevLen, n);
      while (start < min && sameKey(rows[start]!.key, keys[start])) {
        update(rows[start]!, list[start]!, start);
        start++;
      }
      if (start === n && n === prevLen) {
        ran = true;
        if (outDirty) {
          const nextOut = new Array<JSX.Element>(n);
          for (let j = 0; j < n; j++) nextOut[j] = rows[j]!.value;
          out = nextOut;
          outDirty = false;
        }
        return out;
      }
      if (start === prevLen) {
        outDirty = true;
        const nextOut = new Array<JSX.Element>(n);
        for (let j = 0; j < start; j++) nextOut[j] = rows[j]!.value;
        try {
          for (let j = start; j < n; j++) {
            const r = make(list[j]!, j, keys[j]);
            rows.push(r);
            nextOut[j] = r.value;
          }
        } catch (e) {
          for (let j = prevLen; j < rows.length; j++) {
            try {
              dispose(rows[j]!.root);
            } catch {
              // Keep the original row error.
            }
          }
          rows.length = prevLen;
          throw e;
        }
        ran = true;
        outDirty = false;
        return (out = nextOut);
      }
      if (start === n) {
        outDirty = true;
        for (let i = n; i < prevLen; i++) drop(rows[i]!);
        rows.length = n;
        const nextOut = new Array<JSX.Element>(n);
        for (let j = 0; j < n; j++) nextOut[j] = rows[j]!.value;
        ran = true;
        outDirty = false;
        return (out = nextOut);
      }
      const next = new Array<Row<T>>(n);
      for (let j = 0; j < start; j++) next[j] = rows[j]!;
      let end = prevLen - 1;
      let newEnd = n - 1;
      for (;;) {
        while (start < min && start <= end && start <= newEnd && sameKey(rows[start]!.key, keys[start])) {
          update((next[start] = rows[start]!), list[start]!, start);
          start++;
        }
        while (end >= start && newEnd >= start && sameKey(rows[end]!.key, keys[newEnd])) {
          update((next[newEnd] = rows[end]!), list[newEnd]!, newEnd);
          end--;
          newEnd--;
        }
        // The first and last rows swapped: match them directly.
        if (end > start && newEnd > start && sameKey(rows[start]!.key, keys[newEnd]) && sameKey(rows[end]!.key, keys[start])) {
          update((next[newEnd] = rows[start]!), list[newEnd]!, newEnd);
          update((next[start] = rows[end]!), list[start]!, start);
          start++;
          end--;
          newEnd--;
          continue;
        }
        break;
      }
      if (start > end) {
        // Pure insertions (or everything matched in prefix/suffix/swap).
        let built = start;
        try {
          for (let j = start; j <= newEnd; j++) {
            next[j] = make(list[j]!, j, keys[j]);
            built = j + 1;
          }
        } catch (e) {
          for (let j = start; j < built; j++) {
            try {
              dispose(next[j]!.root);
            } catch {
              // Keep the original row error.
            }
          }
          throw e;
        }
      } else if (start > newEnd) {
        // Pure removals.
        for (let i = start; i <= end; i++) drop(rows[i]!);
      } else if (start === end && start === newEnd) {
        // Single-row slot: preserve SameValueZero parity before replacing.
        if (sameKey(rows[start]!.key, keys[start])) update((next[start] = rows[start]!), list[start]!, start);
        else {
          try {
            drop(rows[start]!);
            next[start] = make(list[start]!, start, keys[start]);
          } catch (e) {
            rows.splice(start, 1);
            outDirty = true;
            throw e;
          }
        }
      } else if (
        end - start === newEnd - start &&
        sameKey(rows[end]!.key, keys[start]) &&
        matchesShifted(rows, keys, start, end - 1, start + 1, keys[start])
      ) {
        // Single-row move earlier (rotate right), unique rotated key: exact FIFO parity without a map.
        update((next[start] = rows[end]!), list[start]!, start);
        for (let i = start; i < end; i++) update((next[i + 1] = rows[i]!), list[i + 1]!, i + 1);
      } else if (
        end - start === newEnd - start &&
        sameKey(rows[start]!.key, keys[newEnd]) &&
        matchesShifted(rows, keys, start + 1, end, start, keys[newEnd])
      ) {
        // Single-row move later (rotate left), unique rotated key: exact FIFO parity without a map.
        for (let i = start; i < end; i++) update((next[i] = rows[i + 1]!), list[i]!, i);
        update((next[newEnd] = rows[start]!), list[newEnd]!, newEnd);
      } else {
        // Middle: match the rest by key (duplicates chained in order).
        const first = new Map<unknown, number>();
        const chain = new Int32Array(newEnd - start + 1);
        for (let j = newEnd; j >= start; j--) {
          const k = keys[j];
          const at = first.get(k);
          chain[j - start] = at === undefined ? -1 : at;
          first.set(k, j);
        }
        try {
          for (let i = start; i <= end; i++) {
            const row = rows[i]!;
            const j = first.get(row.key);
            if (j !== undefined) {
              next[j] = row;
              const c = chain[j - start]!;
              if (c === -1) first.delete(row.key);
              else first.set(row.key, c);
            } else {
              rows[i] = undefined!;
              drop(row);
            }
          }
          for (let j = start; j <= newEnd; j++) {
            const row = next[j];
            if (row !== undefined) update(row, list[j]!, j);
            else next[j] = make(list[j]!, j, keys[j]);
          }
        } catch (e) {
          const kept = new Set(rows);
          for (let j = start; j <= newEnd; j++) {
            const r = next[j];
            if (r !== undefined && !kept.has(r)) {
              try {
                dispose(r.root);
              } catch {
                // Keep the original row error.
              }
            }
          }
          rows = rows.filter(Boolean);
          outDirty = true;
          throw e;
        }
      }
      rows = next;
      ran = true;
      const nextOut = new Array<JSX.Element>(n);
      for (let j = 0; j < n; j++) nextOut[j] = next[j]!.value;
      outDirty = false;
      return (out = nextOut);
    });
  });
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
  let rows: { sig: Signal<T>; root: Owner; value: JSX.Element; t: BranchTransitions }[] = [];
  let fallback: { root: Owner; value: JSX.Element } | null = null;
  let out: JSX.Element[] = [];
  let outDirty = false;
  let ran = false;
  const mapped: ReadSignal<JSX.Element[]> = computed(() => {
    // Length and entries tracked (a `deep()` array changed in place updates the rows), the rows' upkeep untracked.
    const list = props.each || [];
    const n = list.length;
    const items = n > 0 ? new Array<T>(n) : (EMPTY_KEYS as unknown as T[]);
    for (let i = 0; i < n; i++) items[i] = list[i]!;
    return untrack(() => {
      const prevLen = rows.length;
      const common = Math.min(n, prevLen);
      for (let i = 0; i < common; i++) {
        const item = items[i]!;
        rows[i]!.sig.set(typeof item === 'function' ? () => item : item);
      }
      if (n > prevLen) {
        outDirty = true;
        try {
          for (let i = prevLen; i < n; i++) {
            const root = ownedRoot(parent);
            const prevOwner = swapOwner(root);
            const sig = signal<T>(items[i]!);
            let value: JSX.Element;
            let t: BranchTransitions = null;
            try {
              enterBranch(ran);
              let raw: JSX.Element;
              try {
                raw = fn(() => sig.get(), i);
              } finally {
                t = exitBranch();
              }
              value = settle(raw);
            } catch (e) {
              try {
                dispose(root);
              } catch {
                // Keep the original row error.
              }
              throw e;
            } finally {
              swapOwner(prevOwner);
            }
            rows.push({ sig, root, t, value });
          }
        } catch (e) {
          for (let j = prevLen; j < rows.length; j++) {
            try {
              dispose(rows[j]!.root);
            } catch {
              // Keep the original row error.
            }
          }
          rows.length = prevLen;
          throw e;
        }
      } else if (n < prevLen) {
        outDirty = true;
        for (let i = n; i < prevLen; i++) {
          const row = rows[i]!;
          if (row.t !== null) leaveBranch(row.t, row.value);
          dispose(row.root);
        }
        rows.length = n;
      }
      if (n === 0 && fallback === null && props.fallback !== undefined) {
        outDirty = true;
        const root = ownedRoot(parent);
        const prevOwner = swapOwner(root);
        let value: JSX.Element;
        try {
          enterBranch(ran, false);
          let raw: JSX.Element;
          try {
            raw = props.fallback;
          } finally {
            exitBranch();
          }
          value = settle(raw);
        } catch (e) {
          try {
            dispose(root);
          } catch {
            // Keep the original fallback error.
          }
          throw e;
        } finally {
          swapOwner(prevOwner);
        }
        fallback = { root, value };
      } else if (n > 0 && fallback !== null) {
        outDirty = true;
        const prevFallback = fallback;
        fallback = null;
        dispose(prevFallback.root);
      }
      ran = true;
      if (outDirty) {
        if (n === 0) out = fallback !== null ? [fallback.value] : [];
        else {
          const next = new Array<JSX.Element>(n);
          for (let i = 0; i < n; i++) next[i] = rows[i]!.value;
          out = next;
        }
        outDirty = false;
      }
      return out;
    });
  });
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
  let prevT: BranchTransitions = null;
  let prevV: JSX.Element = undefined;
  return memo(() => {
    const i = which.get();
    if (prevT !== null) {
      const t0 = prevT;
      const v0 = prevV;
      prevT = null;
      prevV = undefined;
      leaveBranch(t0, v0);
    }
    enterBranch(ran);
    let v: JSX.Element;
    let t: BranchTransitions = null;
    try {
      v = untrack(() => {
        if (i < 0) return props.fallback;
        const m = cases[i]!;
        const ch = m.children;
        return typeof ch === 'function' && !isReader(ch) ? (ch as (v: () => unknown) => JSX.Element)(() => m.when) : (ch as JSX.Element);
      });
    } finally {
      t = exitBranch();
    }
    ran = true;
    prevT = t;
    prevV = v;
    return settle(v);
  });
}

// ---------------------------------------------------------------------------------------------------------------------
// Key

/** Re-creates its content whenever `when` changes (port of `{#key when}`); intros in it play on each re-creation. */
export function Key<T>(props: { when: T; children: JSX.Element }): JSX.Element {
  const key = computed(() => props.when);
  let ran = false;
  let prevT: BranchTransitions = null;
  let prevV: JSX.Element = undefined;
  return memo(() => {
    key.get();
    if (prevT !== null) {
      const t0 = prevT;
      const v0 = prevV;
      prevT = null;
      prevV = undefined;
      leaveBranch(t0, v0);
    }
    enterBranch(ran);
    let v: JSX.Element;
    let t: BranchTransitions = null;
    try {
      v = untrack(() => {
        const ch = props.children;
        // A snippet is created here, inside the branch (its intros play, its outros are this branch's).
        return typeof ch === 'function' && !isReader(ch) ? (ch as () => JSX.Element)() : ch;
      });
    } finally {
      t = exitBranch();
    }
    ran = true;
    prevT = t;
    prevV = v;
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
        clearLeaves(n);
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
