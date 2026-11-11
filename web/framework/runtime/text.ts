// Text runs: `Epoch {a}/{b}` is one text node, as Svelte renders it (Chrome shapes each text node on its own, so
// four nodes antialias differently from one). The compiler hands the run's parts over: static strings, and a function
// per expression. One render effect concatenates them into the node's `data`.
//
// An expression may also yield a node, a list or a snippet (JSX cannot always tell `{label}` from `{props.footer}`).
// Then the run splits where Svelte's text sequence would have ended at a `{@render}`: the text before that part stays
// in the run's node, the part is rendered after it, and the text after it is a run of its own. Nothing is evaluated
// twice: the effect that found the part becomes that part's own effect, keeping only what the part read, so a change
// in the text around it never re-creates it. What the part renders is a memo the run owns, so reading it again (a
// later split elsewhere in the run) never re-creates it either.
//
// A run in a template is inside an element: what a split renders goes in beside it. A run in a fragment has no element
// of its own, and the insert that holds the fragment knows only the nodes it was given: such a run is a reader of its
// nodes, which a split (at creation or later) extends, so that insert places, moves and removes them all.

import { dropSourcesThrough, getOwner, isReader, memo, reader, renderEffect, runWithOwner, signal, sourceMark, untrack, type Owner, type Reader, type Signal } from './core';
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

/** A fragment run's nodes, in order: its text node, then what splits added (each split's part, then its own run). */
interface Nodes {
  list: JSX.Element[];
  /** Set once the run has been created: a split after that tells the insert holding the run. */
  sig: Signal<JSX.Element> | null;
}

/**
 * Binds `node` to `parts`. `nodes` is set for a fragment run: a part that is not text, and a run for what follows it,
 * are placed after `node` in that list (and so rendered by the insert that holds the fragment).
 */
function bind(node: Text, parts: Part[], owner: Owner | null, nodes: Nodes | null): void {
  const fixed = staticText(parts);
  if (fixed !== null) {
    node.data = fixed;
    return;
  }
  let at = -1;
  let content: Signal<unknown> | undefined;
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
      runWithOwner(owner, () => {
        bind(node, before, owner, nodes);
        // The part's content, owned by the run and created once per value (a snippet is called once): an insert that
        // reads it again (the fragment's, when another part splits later) gets the same nodes, never new ones.
        const rendered = memo(() => content!.get());
        if (nodes !== null) {
          const rest = after.length > 0 ? document.createTextNode('') : null;
          const added: JSX.Element[] = rest !== null ? [rendered as JSX.Element, rest] : [rendered as JSX.Element];
          nodes.list.splice(nodes.list.indexOf(node) + 1, 0, ...added);
          if (rest !== null) bind(rest, after, owner, nodes);
          nodes.sig?.set(nodes.list.slice());
        } else {
          const rest = document.createTextNode('');
          const parent = node.parentNode!;
          parent.insertBefore(rest, node.nextSibling);
          insert(parent, rendered, rest);
          if (after.length > 0) bind(rest, after, owner, null);
        }
      });
      return;
    }
    if (node.data !== s) node.data = s;
  });
}

/** Internal: a run whose text node is in the template. */
export function run(node: Text, parts: Part[]): void {
  bind(node, parts, getOwner(), null);
}

/**
 * Internal: a run inside a fragment: its text node, or, when one of its parts may render more than text, a reader of
 * its nodes (the text node, then what splits rendered, at creation or later), for the insert that holds the fragment.
 */
export function runNode(parts: Part[]): JSX.Element {
  const node = document.createTextNode('');
  if (staticText(parts) !== null) {
    bind(node, parts, null, null);
    return node;
  }
  const nodes: Nodes = { list: [node], sig: null };
  bind(node, parts, getOwner(), nodes);
  const sig = (nodes.sig = signal<JSX.Element>(nodes.list.length === 1 ? node : (nodes.list.slice() as JSX.Element)));
  const read = reader(() => sig.get()) as Reader<JSX.Element> & { $$run?: 1 };
  // Marked, so a branch leaving with an outro keeps these nodes until it ends (transition.ts, `collect`).
  read.$$run = 1;
  return read as unknown as JSX.Element;
}
