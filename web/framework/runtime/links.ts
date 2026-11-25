// Doubly linked dependency graph shared between reactive sources and targets.
//
// Each `Link` lives in both its target's source list (`_ps`/`_ns`, in read order) and its source's subscriber list
// (`_pt`/`_nt`). Re-evaluating a target marks existing links unread (`_v = -1`), moves re-read links to the tail
// without allocating, and unlinks anything left unread in `cleanupSources`.

export const TRACKING = 32;

export interface Link {
  _s: Source;
  _ps: Link | undefined;
  _ns: Link | undefined;
  _t: Target;
  _pt: Link | undefined;
  _nt: Link | undefined;
  _v: number;
  _rb: Link | undefined;
}

export interface Source {
  _value: unknown;
  _version: number;
  _node: Link | undefined;
  _targets: Link | undefined;
  _refresh(): boolean;
  _subscribe(link: Link): void;
  _unsubscribe(link: Link): void;
}

export interface Target {
  _sources: Link | undefined;
  _flags: number;
  _notify(): void;
}

/** Appends or reuses the link from `source` to `target` at the tail of `target._sources`. */
export function linkDependency(source: Source, target: Target): Link | undefined {
  let node = source._node;
  if (node === undefined || node._t !== target) {
    // A new dependency: append it to the target's source list (its tail is `_sources` while evaluating).
    node = { _v: 0, _s: source, _ps: target._sources, _ns: undefined, _t: target, _pt: undefined, _nt: undefined, _rb: node };
    if (target._sources !== undefined) target._sources._ns = node;
    target._sources = node;
    source._node = node;
    if (target._flags & TRACKING) source._subscribe(node);
    return node;
  }
  if (node._v === -1) {
    // Read again in this evaluation: reuse the link, moved to the tail so the list stays in read order.
    node._v = 0;
    if (node._ns !== undefined) {
      node._ns._ps = node._ps;
      if (node._ps !== undefined) node._ps._ns = node._ns;
      node._ps = target._sources;
      node._ns = undefined;
      target._sources!._ns = node;
      target._sources = node;
    }
    return node;
  }
  return undefined;
}

/** Before an evaluation: mark every current link unread (-1) and point `_sources` at the tail. */
export function prepareSources(target: Target): void {
  for (let node = target._sources; node !== undefined; node = node._ns) {
    const rb = node._s._node;
    if (rb !== undefined) node._rb = rb;
    node._s._node = node;
    node._v = -1;
    if (node._ns === undefined) {
      target._sources = node;
      break;
    }
  }
}

/** After an evaluation: drop links that were not read, restore the sources' outer links, point `_sources` at the head. */
export function cleanupSources(target: Target): void {
  let node = target._sources;
  let head: Link | undefined;
  while (node !== undefined) {
    const prev = node._ps;
    if (node._v === -1) {
      node._s._unsubscribe(node);
      if (prev !== undefined) prev._ns = node._ns;
      if (node._ns !== undefined) node._ns._ps = prev;
    } else head = node;
    node._s._node = node._rb;
    if (node._rb !== undefined) node._rb = undefined;
    node = prev;
  }
  target._sources = head;
}

/** Whether any source of `target` changed since it last ran (pulls computeds up to date on the way). */
export function needsToRecompute(target: Target): boolean {
  for (let node = target._sources; node !== undefined; node = node._ns) {
    if (node._s._version !== node._v || !node._s._refresh() || node._s._version !== node._v) return true;
  }
  return false;
}

export function subscribeLink(source: Source, node: Link): void {
  if (source._targets !== node && node._pt === undefined) {
    node._nt = source._targets;
    if (source._targets !== undefined) source._targets._pt = node;
    source._targets = node;
  }
}

export function unsubscribeLink(source: Source, node: Link): void {
  if (source._targets === undefined) return;
  const prev = node._pt;
  const next = node._nt;
  if (prev !== undefined) {
    prev._nt = next;
    node._pt = undefined;
  }
  if (next !== undefined) {
    next._pt = prev;
    node._nt = undefined;
  }
  if (node === source._targets) source._targets = next;
}
