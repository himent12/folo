// Zero-allocation tree-order comparator and queue sorter for the reactive scheduler.
//
// Effects and owning computeds form a tree via `_parent` and `_depth` (0 at a root), with `_id` giving sibling order
// (a memo's branch root copies its memo's `_id` so a recreated branch keeps its original place). Comparing two jobs
// without allocating path arrays walks up to their lowest common ancestor; tracking the shallowest differing `_id`
// along the walk makes the result identical to lexicographic root-to-node `_id` path comparison even when a memo and
// its branch root share the same `_id`.

/** Minimal tree-position shape shared by `Owner` and owning `ComputedNode` jobs. */
export interface TreeNode {
  readonly _id: number;
  readonly _parent: TreeNode | null;
  readonly _depth: number;
}

/**
 * Compares two jobs in tree order (parents before children, siblings in creation order; a branch created late stands
 * at its memo's place). Matches lexicographic comparison of root-to-node `_id` paths with zero allocations.
 */
export function compareJobs(a: TreeNode, b: TreeNode): number {
  if (a._parent === b._parent) return a._id - b._id;
  let pa = a;
  let pb = b;
  let da = a._depth;
  let db = b._depth;
  while (da > db) {
    pa = pa._parent!;
    da--;
  }
  while (db > da) {
    pb = pb._parent!;
    db--;
  }
  let diff = 0;
  while (pa !== pb) {
    if (pa._id !== pb._id) diff = pa._id - pb._id;
    const ap = pa._parent;
    const bp = pb._parent;
    if (ap === bp) break;
    pa = ap!;
    pb = bp!;
  }
  return diff !== 0 ? diff : a._depth - b._depth;
}

/** Sorts a flush queue in tree order in place, fast-pathing 2-element and already-ordered queues. */
export function sortJobs<T extends TreeNode>(q: T[]): void {
  const len = q.length;
  if (len <= 1) return;
  if (len === 2) {
    const a = q[0]!;
    const b = q[1]!;
    if ((a._parent === b._parent ? a._id - b._id : compareJobs(a, b)) > 0) {
      q[0] = b;
      q[1] = a;
    }
    return;
  }
  for (let i = 1; i < len; i++) {
    const prev = q[i - 1]!;
    const cur = q[i]!;
    if ((prev._parent === cur._parent ? prev._id - cur._id : compareJobs(prev, cur)) > 0) {
      q.sort(compareJobs);
      return;
    }
  }
}
