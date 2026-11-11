// Text runs: `Epoch {a}/{b}` is one text node, as Svelte renders it (Chrome shapes each text node on its own, so
// four nodes antialias differently from one). The compiler hands the run's parts over: static strings, and a function
// per expression. One render effect concatenates them into the node's `data`.
//
// An expression may also yield a node, a list or a snippet (JSX cannot always tell `{label}` from `{props.footer}`).
// Then the run splits where Svelte's text sequence would have ended at a `{@render}`: the text before that part stays
// in the run's node, the part is rendered after it, and the text after it is a run of its own. Nothing is evaluated
// twice: the effect that found the part becomes that part's own effect, keeping only what the part read, so a change
// in the text around it never re-creates it.

import { dropSourcesThrough, getOwner, isReader, reader, renderEffect, runWithOwner, signal, sourceMark, untrack, type Owner, type Signal } from './core';
import { insert } from './dom';

type Part = string | (() => unknown);

function isText(v: unknown): boolean {
  if (v == null) return true;
  const t = typeof v;
  return t === 'string' || t === 'number' || t === 'bigint' || t === 'boolean';
}

function unwrap(v: unknown): unknown {
  while (isReader(v)) v = v();
  return v;
}

function staticText(parts: Part[]): string | null {
  let s = '';
  for (const p of parts) {
    if (typeof p !== 'string') return null;
    s += p;
  }
  return s;
}

/**
 * Binds `node` to `parts`. `out` is set while a fragment run is created (its node not in the DOM yet): a part that is
 * not text, and what follows it, are pushed there, for the fragment to render after `node`.
 */
function bind(node: Text, parts: Part[], owner: Owner | null, out: JSX.Element[] | null): void {
  const fixed = staticText(parts);
  if (fixed !== null) {
    node.data = fixed;
    return;
  }
  let at = -1;
  let content: Signal<unknown> | undefined;
  let creating = true;
  renderEffect(() => {
    if (at >= 0) {
      // The part's own effect now: what it read changed.
      const v = (parts[at] as () => unknown)();
      content!.set(() => v);
      return;
    }
    let s = '';
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i]!;
      if (typeof p === 'string') {
        s += p;
        continue;
      }
      const mark = sourceMark();
      const raw = p();
      let v = raw;
      if (isReader(raw)) {
        // A reader is looked at untracked first: if it renders more than text, the part's effect must not read it.
        v = untrack(() => unwrap(raw));
        if (isText(v)) unwrap(raw);
      }
      if (isText(v)) {
        if (v != null && typeof v !== 'boolean') s += v as string;
        continue;
      }
      // Not text: split here.
      dropSourcesThrough(mark);
      at = i;
      content = signal<unknown>(raw);
      const before = parts.slice(0, i);
      const after = parts.slice(i + 1);
      const fragment = out !== null && creating;
      runWithOwner(owner, () => {
        bind(node, before, owner, null);
        const rendered = reader(() => content!.get());
        const rest = after.length > 0 || !fragment ? document.createTextNode('') : null;
        if (fragment) {
          out!.push(rendered as JSX.Element);
          if (rest !== null) {
            out!.push(rest);
            bind(rest, after, owner, out);
          }
        } else {
          const parent = node.parentNode!;
          parent.insertBefore(rest!, node.nextSibling);
          insert(parent, rendered, rest);
          if (after.length > 0) bind(rest!, after, owner, null);
        }
      });
      return;
    }
    if (node.data !== s) node.data = s;
  });
  creating = false;
}

/** Internal: a run whose text node is in the template. */
export function run(node: Text, parts: Part[]): void {
  bind(node, parts, getOwner(), null);
}

/**
 * Internal: a run inside a fragment: its text node, or, when a part is not text at creation, the text node followed by
 * what the split rendered. A part that turns non-text later is split in place, after the node, like a template run.
 */
export function runNode(parts: Part[]): JSX.Element {
  const node = document.createTextNode('');
  const out: JSX.Element[] = [];
  bind(node, parts, getOwner(), out);
  return out.length === 0 ? node : ([node, ...out] as JSX.Element);
}
