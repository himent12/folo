// What compiled JSX calls. Not for application code.

import { computed, reader } from './core';

export { template, insert, attr, prop, setAttribute, setProperty, setValue, setChecked, className, toggleClass, setStyleProp, styleProp, bindStyle } from './dom';
export { createComponent, mergeProps, bound, propMemo, propJsx, derive } from './component';
export { spread } from './spread';
export { setEvent, setDelegated, delegateEvents } from './events';
export { ref, use, useValue, bindValue, bindChecked, bindGroup, bindOpen, bindSize, selectValue } from './bind';
export { transition } from './transition';
export { run, runNode } from './text';
export { memo, untrack } from './core';

/** A condition's truthiness as a reader: only flips re-run what reads it. */
export function cond(fn: () => unknown) {
  const c = computed(() => !!fn());
  return reader(() => c.get());
}
