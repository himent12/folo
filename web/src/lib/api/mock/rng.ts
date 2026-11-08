// Seeded randomness for the mock: everything it produces is a pure function of ids, indices and time.

/** mulberry32: small, fast, good enough for mock data. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** FNV-1a hash of a string, for turning ids into seeds. */
export function hash(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

export const range = (r: () => number, lo: number, hi: number) => lo + (hi - lo) * r();
export const int = (r: () => number, lo: number, hi: number) => Math.floor(range(r, lo, hi + 1));
export const pick = <T>(r: () => number, xs: readonly T[]): T => xs[Math.floor(r() * xs.length)] as T;

/** Approximately normal (sum of uniforms). */
export function gauss(r: () => number): number {
  return r() + r() + r() + r() - 2;
}

/** Picks an index with the given relative weights. */
export function weighted(r: () => number, weights: readonly number[]): number {
  const total = weights.reduce((a, b) => a + b, 0);
  let x = r() * total;
  for (let i = 0; i < weights.length; i++) {
    x -= weights[i] ?? 0;
    if (x <= 0) return i;
  }
  return weights.length - 1;
}

/** Smooth deterministic noise in [-1, 1] for a continuous parameter (value noise). */
export function smoothNoise(seed: number, x: number): number {
  const i = Math.floor(x);
  const f = x - i;
  const a = rng(seed + i * 7919)() * 2 - 1;
  const b = rng(seed + (i + 1) * 7919)() * 2 - 1;
  const u = f * f * (3 - 2 * f);
  return a + (b - a) * u;
}
