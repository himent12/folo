// Element refs, actions (`use:`) and two-way bindings (`bind:`), with Svelte 5's behaviour.

import { effect, onCleanup, renderEffect, untrack } from './core';
import { on } from './events';

/** Something `bind:` can read and write: a signal, a `bindable` prop, `field(obj, key)`. */
export interface Bindable<T> {
  get(): T;
  set(v: T): void;
}

/** A ref: a function called with the element, or a signal-like object set to it (and back to `undefined` on removal). */
export type Ref<T> = ((el: T) => void) | { set(v: T | undefined): void; peek?(): unknown } | null | undefined;

/** Internal: `ref={r}` (port of `bind:this`). Called once, synchronously, when the element is created. */
export function ref<T>(el: T, r: Ref<T>): void {
  if (typeof r === 'function') r(el);
  else if (r != null) {
    r.set(el);
    // As Svelte's bind:this does, clear the binding when the element goes away (if it still points at it).
    onCleanup(() => {
      if (r.peek === undefined || r.peek() === el) r.set(undefined);
    });
  }
}

/** A Svelte action: `(node, param) => { update?(param), destroy?() }`. */
export type Action<E extends Element = HTMLElement, P = undefined> = (node: E, param: P) => void | ActionReturn<P>;
export interface ActionReturn<P = undefined> {
  update?: (param: P) => void;
  destroy?: () => void;
}

/**
 * Internal: `use:action={param}` (port of Svelte's `use:`). Runs after mount; `update` is called whenever `param`
 * is re-evaluated to a different value (always, for an object literal), `destroy` on removal.
 */
export function use<E extends Element, P>(el: E, action: Action<E, P>, param?: () => P): void {
  effect(() => {
    const payload = untrack(() => action(el, param?.() as P)) || {};
    if (param && payload.update) {
      let inited = false;
      let prev: unknown = {};
      renderEffect(() => {
        const value = param();
        if (inited && safeNotEqual(prev, value)) {
          prev = value;
          untrack(() => payload.update!(value));
        }
      });
      inited = true;
    }
    if (payload.destroy) return () => payload.destroy!();
  });
}

function safeNotEqual(a: unknown, b: unknown): boolean {
  // eslint-disable-next-line no-self-compare
  return a != a ? b == b : a !== b || (a !== null && typeof a === 'object') || typeof a === 'function';
}

/** Internal: `use={...}` with a value: an action, `[action, param]`, or a list of those. */
export function useValue<E extends Element>(el: E, value: unknown): void {
  if (typeof value === 'function') use(el, value as Action<E, unknown>);
  else if (Array.isArray(value)) {
    if (typeof value[0] === 'function') {
      const p = value[1];
      use(el, value[0] as Action<E, unknown>, typeof p === 'function' ? (p as () => unknown) : () => p);
    } else for (const v of value) useValue(el, v);
  }
}

function isNumberLike(input: HTMLInputElement): boolean {
  const type = input.type;
  return type === 'number' || type === 'range';
}

function toNumber(value: string): number | null {
  return value === '' ? null : +value;
}

/** Internal: `bind:value={sig}` on `<input>`, `<textarea>` or `<select>`. */
export function bindValue(el: Element, sig: Bindable<unknown>): void {
  if (el.nodeName === 'SELECT') {
    bindSelect(el as HTMLSelectElement, sig);
    return;
  }
  const input = el as HTMLInputElement;
  on(input, 'input', () => {
    const value = isNumberLike(input) ? toNumber(input.value) : input.value;
    sig.set(value);
  });
  if (untrack(() => sig.get()) == null && input.value) sig.set(isNumberLike(input) ? toNumber(input.value) : input.value);
  renderEffect(() => {
    const value = sig.get();
    if (isNumberLike(input) && value === toNumber(input.value)) return;
    if (input.type === 'date' && !value && !input.value) return;
    if (value !== input.value) input.value = (value ?? '') as string;
  });
}

function selectOption(select: HTMLSelectElement, value: unknown): void {
  if (select.multiple) {
    const values = Array.isArray(value) ? value : [];
    for (const option of select.options) option.selected = values.includes(optionValue(option));
    return;
  }
  for (const option of select.options) {
    if (Object.is(optionValue(option), value)) {
      option.selected = true;
      return;
    }
  }
  select.selectedIndex = -1;
}

function optionValue(option: HTMLOptionElement): unknown {
  return '__value' in option ? (option as HTMLOptionElement & { __value: unknown }).__value : option.value;
}

function bindSelect(select: HTMLSelectElement, sig: Bindable<unknown>): void {
  on(select, 'change', () => {
    if (select.multiple) sig.set([...select.selectedOptions].map(optionValue));
    else {
      const o = select.options[select.selectedIndex];
      sig.set(o ? optionValue(o) : undefined);
    }
  });
  // After the options are in (a user effect runs after the render effects that insert them), then on every change.
  effect(() => {
    const value = sig.get();
    selectOption(select, value);
    if (value === undefined && !select.multiple) {
      const o = select.options[select.selectedIndex];
      if (o) untrack(() => sig.set(optionValue(o)));
    }
  });
}

/** Internal: `<select value={v}>` (one-way): selects the matching option after the options are rendered. */
export function selectValue(select: Element, fn: () => unknown): void {
  effect(() => selectOption(select as HTMLSelectElement, fn()));
}

/** Internal: `bind:checked={sig}` on a checkbox. */
export function bindChecked(el: Element, sig: Bindable<boolean>): void {
  const input = el as HTMLInputElement;
  on(input, 'change', () => sig.set(input.checked));
  if (untrack(() => sig.get()) == null) sig.set(input.checked);
  renderEffect(() => {
    input.checked = !!sig.get();
  });
}

/** Internal: `bind:group={sig}` on a radio (value) or a checkbox (array of values) with `value={...}`. */
export function bindGroup(el: Element, sig: Bindable<unknown>, value: () => unknown): void {
  const input = el as HTMLInputElement;
  const isCheckbox = input.type === 'checkbox';
  on(input, 'change', () => {
    const v = value();
    if (!isCheckbox) {
      if (input.checked) sig.set(v);
      return;
    }
    const list = (untrack(() => sig.get()) as unknown[] | undefined) ?? [];
    sig.set(input.checked ? (list.includes(v) ? list : [...list, v]) : list.filter((x) => x !== v));
  });
  renderEffect(() => {
    const g = sig.get();
    const v = value();
    input.checked = isCheckbox ? Array.isArray(g) && g.includes(v) : Object.is(g, v);
  });
}

/** Internal: `bind:open={sig}` on `<details>` or `<dialog>`. */
export function bindOpen(el: Element, sig: Bindable<boolean>): void {
  const e = el as HTMLDetailsElement;
  on(e, el.nodeName === 'DIALOG' ? 'close' : 'toggle', () => sig.set(e.open));
  renderEffect(() => {
    const open = !!sig.get();
    if (e.open !== open) e.open = open;
  });
}

type SizeKey = 'clientWidth' | 'clientHeight' | 'offsetWidth' | 'offsetHeight';
let observer: ResizeObserver | undefined;
const watchers = new WeakMap<Element, Set<() => void>>();

/** Internal: `bind:clientWidth={sig}` and co. (read-only), via a shared ResizeObserver, as Svelte does. */
export function bindSize(el: Element, key: SizeKey, sig: { set(v: number): void }): void {
  const update = () => sig.set((el as HTMLElement)[key]);
  effect(() => {
    untrack(update);
    if (typeof ResizeObserver === 'undefined') return;
    observer ??= new ResizeObserver((entries) => {
      for (const entry of entries) for (const fn of watchers.get(entry.target) ?? []) fn();
    });
    let set = watchers.get(el);
    if (set === undefined) {
      watchers.set(el, (set = new Set()));
      observer.observe(el);
    }
    set.add(update);
    return () => {
      set!.delete(update);
      if (set!.size === 0) {
        watchers.delete(el);
        observer!.unobserve(el);
      }
    };
  });
}
