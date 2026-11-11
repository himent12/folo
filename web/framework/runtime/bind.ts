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

type SelectEl = HTMLSelectElement & { __value?: unknown };

/** Svelte's `select_option`: selects the option(s) holding `value`; none (index -1) when no option does. */
function selectOption(select: HTMLSelectElement, value: unknown, mounting = false): void {
  if (select.multiple) {
    // Not an array (or nothing): the selection stays as it is.
    if (!Array.isArray(value)) return;
    for (const option of select.options) option.selected = value.includes(optionValue(option));
    return;
  }
  for (const option of select.options) {
    if (Object.is(optionValue(option), value)) {
      option.selected = true;
      return;
    }
  }
  if (!mounting || value !== undefined) select.selectedIndex = -1;
}

function optionValue(option: HTMLOptionElement): unknown {
  return '__value' in option ? (option as HTMLOptionElement & { __value: unknown }).__value : option.value;
}

/**
 * Svelte's `init_select`: when the options change (a list rendering them, an option's value), the select's value is
 * selected again, so a value set before its option exists is shown once the option arrives, and a value no option
 * holds keeps nothing selected (the browser would select the first option added).
 */
function initSelect(select: SelectEl): void {
  if (typeof MutationObserver === 'undefined') return;
  const observer = new MutationObserver(() => {
    if ('__value' in select) selectOption(select, select.__value);
  });
  observer.observe(select, { childList: true, subtree: true, attributes: true, attributeFilter: ['value'] });
  onCleanup(() => observer.disconnect());
}

function bindSelect(select: SelectEl, sig: Bindable<unknown>): void {
  let mounting = true;
  on(select, 'change', () => {
    let value: unknown;
    if (select.multiple) value = [...select.selectedOptions].map(optionValue);
    else {
      // Nothing selected: the first option that is not disabled, as Svelte reads it.
      const o = select.options[select.selectedIndex] ?? select.querySelector<HTMLOptionElement>('option:not([disabled])');
      value = o ? optionValue(o) : null;
    }
    sig.set(value);
    select.__value = value;
  });
  // After the options are in (a user effect runs after the render effects that insert them), then on every change.
  effect(() => {
    let value = sig.get();
    selectOption(select, value, mounting);
    if (mounting && value === undefined) {
      // Mounting with no value: the value is what the DOM selected.
      const o = select.options[select.selectedIndex];
      if (o) {
        value = optionValue(o);
        untrack(() => sig.set(value));
      }
    }
    select.__value = value;
    mounting = false;
  });
  initSelect(select);
}

/** Internal: `<select value={v}>` (one-way): selects the matching option after the options are rendered, and again when they change. */
export function selectValue(select: Element, fn: () => unknown): void {
  const el = select as SelectEl;
  let prev: unknown = {};
  effect(() => {
    const value = fn();
    if (value === prev) return;
    prev = value;
    el.value = ((el.__value = value) ?? '') as string;
    selectOption(el, value);
  });
  initSelect(el);
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
