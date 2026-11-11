// Text runs: `Epoch {a}/{b}` is one text node, as Svelte renders it (Chrome shapes each text node on its own, so
// four nodes antialias differently from one). The compiler hands the run's parts over: static strings, and a function
// per expression. One render effect concatenates them into the node's `data`.
//
// An expression may also yield a node, a list or a snippet (JSX cannot tell `{label}` from `{props.children}`). The
// first time a part is not text, the run switches for good to one `insert` per part, placed before the (then empty)
// text node, so nothing it renders is re-created when another part changes.

import { createRoot, getOwner, isReader, renderEffect, type Owner } from './core';
import { insert } from './dom';

type Part = string | (() => unknown);

/** The run's text, or `null` when a part is not text. */
function join(parts: Part[]): string | null {
  let s = '';
  for (let i = 0; i < parts.length; i++) {
    const p = parts[i]!;
    if (typeof p === 'string') {
      s += p;
      continue;
    }
    let v = p();
    while (isReader(v)) v = v();
    if (v == null || v === true || v === false) continue;
    const t = typeof v;
    if (t === 'string' || t === 'number' || t === 'bigint') s += v as string;
    else return null;
  }
  return s;
}

/** Binds `node` (the run's text node) to `parts`; returns false when the run is not text (it was split instead). */
function bind(node: Text, parts: Part[], owner: Owner | null, canSplit: () => boolean): boolean {
  let split = false;
  let alive = true;
  createRoot((d) => {
    renderEffect(() => {
      if (split) return;
      const s = join(parts);
      if (s !== null) {
        if (node.data !== s) node.data = s;
        return;
      }
      split = true;
      // This evaluation's nodes are thrown away (its root goes next); each part gets an insert of its own.
      queueMicrotask(d);
      alive = false;
      node.data = '';
      if (canSplit()) createRoot(() => parts.forEach((p) => insert(node.parentNode!, p, node)), owner);
    });
  }, owner);
  return alive;
}

/** Internal: a run whose text node is in the template. */
export function run(node: Text, parts: Part[]): void {
  bind(node, parts, getOwner(), () => true);
}

/**
 * Internal: a run inside a fragment: its text node, or, when a part is not text at creation, the parts themselves (an
 * array `insert` renders). A part that turns non-text later is split in place, before the node, like a template run.
 */
export function runNode(parts: Part[]): JSX.Element {
  const node = document.createTextNode('');
  let created = false;
  const owner = getOwner();
  const text = bind(node, parts, owner, () => created && node.parentNode !== null);
  created = true;
  return text ? node : (parts as JSX.Element);
}

