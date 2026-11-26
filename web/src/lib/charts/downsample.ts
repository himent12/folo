// Point helpers shared by the charts.

export interface Pt {
  x: number;
  y: number;
}

/**
 * Min/max bucket downsampling: keeps the extremes of each bucket so spikes survive, which matters for loss curves.
 * Returns the input unchanged when it is already small enough.
 */
export function downsample(points: Pt[], buckets: number): Pt[] {
  if (points.length <= buckets * 2 || buckets < 2) return points;
  const out: Pt[] = [];
  const size = points.length / buckets;
  for (let b = 0; b < buckets; b++) {
    const start = Math.floor(b * size);
    const end = Math.min(points.length, Math.floor((b + 1) * size));
    let min: Pt | undefined;
    let max: Pt | undefined;
    for (let i = start; i < end; i++) {
      const p = points[i] as Pt;
      if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
      if (!min || p.y < min.y) min = p;
      if (!max || p.y > max.y) max = p;
    }
    if (!min || !max) continue;
    if (min === max) out.push(min);
    else if (min.x < max.x) out.push(min, max);
    else out.push(max, min);
  }
  const last = points[points.length - 1];
  if (last && Number.isFinite(last.x) && Number.isFinite(last.y) && out[out.length - 1] !== last) out.push(last);
  return out;
}

/** Exponential moving average with bias correction (TensorBoard-style smoothing), `alpha` in [0, 1). */
export function smooth(points: Pt[], alpha: number): Pt[] {
  if (alpha <= 0) return points;
  let avg = 0;
  let n = 0;
  return points.map((p) => {
    if (!Number.isFinite(p.y)) return p;
    n++;
    avg = alpha * avg + (1 - alpha) * p.y;
    return { x: p.x, y: avg / (1 - alpha ** n) };
  });
}

/** Index of the point with the x closest to `x` (points sorted by x). */
export function nearest(points: Pt[], x: number): number {
  let lo = 0;
  let hi = points.length - 1;
  if (hi < 0) return -1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((points[mid] as Pt).x < x) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs((points[lo - 1] as Pt).x - x) <= Math.abs((points[lo] as Pt).x - x)) return lo - 1;
  return lo;
}

/** "Nice" tick values for a linear domain. */
export function ticks(min: number, max: number, count: number): number[] {
  if (!(max > min)) return [min];
  const span = max - min;
  const raw = span / Math.max(1, count);
  const pow = 10 ** Math.floor(Math.log10(raw));
  const err = raw / pow;
  const step = pow * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
  const decimals = Math.max(0, -Math.floor(Math.log10(step)));
  const out: number[] = [];
  for (let k = Math.ceil(min / step - 1e-9); k * step <= max + step * 1e-9; k++) out.push(+(k * step).toFixed(decimals));
  return out;
}
