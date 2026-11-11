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

/** A memoised `pathsOf`; `stats` counts the series rebuilt and reused (tests, measurements). */
export function memoPathsOf() {
  let geo: unknown[] = [];
  let cache = new Map<string, { points: Pt[]; lower: Pt[] | undefined; area: boolean; out: Paths[number] }>();
  const stats = { built: 0, reused: 0 };
  function paths(visible: Series[], curve: 'linear' | 'monotone', width: number, margin: Margin, height: number, x: Scale, y: Scale, yLog: boolean): Paths {
    const key = [curve, width, margin.top, margin.right, margin.bottom, margin.left, height, x, y, yLog];
    if (key.length !== geo.length || key.some((v, i) => v !== geo[i])) {
      cache = new Map();
      geo = key;
    }
    const next = new Map<
      string,
      {
        points: Pt[];
        lower: Pt[] | undefined;
        area: boolean;
        out: Paths[number];
      }
    >();
    const out = visible.map((s) => {
      const hit = cache.get(s.id);
      if (hit && !!s.area === hit.area && samePoints(s.points, hit.points) && samePoints(s.lower, hit.lower)) {
        stats.reused++;
        const o = { ...hit.out, s };
        next.set(s.id, { ...hit, points: s.points, lower: s.lower, out: o });
        return o;
      }
      stats.built++;
      const o = pathsOf([s], curve, width, margin, height, x, y, yLog)[0]!;
      next.set(s.id, {
        points: s.points,
        lower: s.lower,
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
