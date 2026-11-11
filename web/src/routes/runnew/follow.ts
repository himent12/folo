// Two Svelte behaviours New run's controls rely on that fw's runtime does not have yet (both reported; delete this file
// once framework/runtime has them):
//
// - `follow`: Svelte's un-bound `$bindable` prop. The child writes the value locally (a Switch's checkbox, a Select's
//   pick) until the parent's value changes, whatever it changes to. fw's `bindable` keeps the local value whenever the
//   parent's is the one it was written over, so a parent that goes A → B → A brings back the child's stale write (Mixed
//   precision: switched on, then the CPU turns it off, and the checkbox stays ticked). Passed as `bind:checked` /
//   `bind:value`, it is that prop with Svelte's semantics.
// - `reselect`: Svelte's `<select bind:value>` selects its value again when its options change (`init_select`'s
//   MutationObserver), so a value set before its option exists (a copied run's connection, device or weights, before
//   the lists load) is shown once the option arrives. fw's selects the value only when the value changes.

import { computed, effect, signal, untrack, type Signal } from 'fw';

/** A value the parent passes, written locally by the child until the parent's value changes (Svelte's un-bound prop). */
export function follow<T>(read: () => T): Signal<T> {
  // A new box per change of the parent's value: a local write holds while the box it was made over is current.
  const parent = computed(() => ({ v: read() }));
  const local = signal<{ v: T; over: { v: T } } | null>(null, { equals: false });
  const get = () => {
    const p = parent.get();
    const l = local.get();
    return l !== null && l.over === p ? l.v : p.v;
  };
  return {
    get,
    peek: () => untrack(get),
    set(v) {
      const next = typeof v === 'function' ? (v as (p: T) => T)(untrack(get)) : v;
      local.set({ v: next, over: parent.peek() });
    },
  };
}

/**
 * Selects `value()` in the `<select id={id}>` again whenever `options()` (what its options are made from) changes, as
 * Svelte's select binding does. Call in the component body that renders the select: the options are in by the time it
 * runs (user effects run after the DOM updates).
 */
export function reselect(id: string, value: () => string, options: () => unknown) {
  effect(() => {
    options();
    const v = value();
    const el = document.getElementById(id);
    if (!(el instanceof HTMLSelectElement)) return;
    for (const o of el.options) {
      if (o.value === v) {
        o.selected = true;
        return;
      }
    }
    el.selectedIndex = -1;
  });
}
