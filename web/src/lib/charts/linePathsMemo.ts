/**
 * `pathsOf` (./linePaths) with a memory per series: a series whose points (and band, and area flag) are the same as
 * last time, under the same scales and plot geometry, keeps the path strings it had, so a live run that grows one line
 * (or a poll that hands back equal data in new arrays) rebuilds only the lines that changed. Same output as `pathsOf`.
 */
import type { Pt } from './downsample';
import type { Margin, Scale } from './lineAxes';
import { pathsOf } from './linePaths';
import type { Series } from './lineSeries';

type Paths = ReturnType<typeof pathsOf>;

/** Equal point lists: the same array, or the same x and y at every index (`Object.is`, so NaN equals NaN). */
export function samePoints(a: readonly Pt[] | undefined, b: readonly Pt[] | undefined): boolean {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const p = a[i]!;
    const q = b[i]!;
    if (!Object.is(p.x, q.x) || !Object.is(p.y, q.y)) return false;
  }
  return true;
}

interface PointStamp {
  arr: Pt[] | undefined;
  len: number;
  first: Pt | undefined;
  last: Pt | undefined;
}

function stampOf(arr: Pt[] | undefined): PointStamp {
  return { arr, len: arr?.length ?? 0, first: arr?.[0], last: arr?.[arr.length - 1] };
}

function sameStamped(cur: Pt[] | undefined, prev: PointStamp): boolean {
  if (prev.arr && (prev.arr.length !== prev.len || prev.arr[0] !== prev.first || prev.arr[prev.arr.length - 1] !== prev.last)) {
    return false;
  }
  if (cur === prev.arr) return true;
  return samePoints(cur, prev.arr);
}

/** A memoised `pathsOf`; `stats` counts the series rebuilt and reused (tests, measurements). */
export function memoPathsOf() {
  type Entry = { points: PointStamp; lower: PointStamp; rev: number | undefined; area: boolean; out: Paths[number] };
  let geo: unknown[] = [];
  let cache = new Map<string, Entry>();
  const stats = { built: 0, reused: 0 };
  function paths(visible: Series[], curve: 'linear' | 'monotone', width: number, margin: Margin, height: number, x: Scale, y: Scale, yLog: boolean): Paths {
    const key = [curve, width, margin.top, margin.right, margin.bottom, margin.left, height, x, y, yLog];
    if (key.length !== geo.length || key.some((v, i) => v !== geo[i])) {
      cache = new Map();
      geo = key;
    }
    const next = new Map<string, Entry>();
    const out = visible.map((s) => {
      const hit = cache.get(s.id);
      if (hit && !!s.area === hit.area && hit.rev === s.rev && sameStamped(s.points, hit.points) && sameStamped(s.lower, hit.lower)) {
        stats.reused++;
        const o = { ...hit.out, s };
        next.set(s.id, { points: stampOf(s.points), lower: stampOf(s.lower), rev: s.rev, area: !!s.area, out: o });
        return o;
      }
      stats.built++;
      const o = pathsOf([s], curve, width, margin, height, x, y, yLog)[0]!;
      next.set(s.id, {
        points: stampOf(s.points),
        lower: stampOf(s.lower),
        rev: s.rev,
        area: !!s.area,
        out: o,
      });
      return o;
    });
    cache = next;
    return out;
  }
  return Object.assign(paths, { stats });
}
