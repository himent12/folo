import { afterEach, describe, expect, it, vi } from 'vitest';
import { motion, reducedMotion } from './motion';

const media = (reduce: boolean) => vi.fn((q: string) => ({ matches: reduce && q.includes('reduce'), media: q }) as unknown as MediaQueryList);

afterEach(() => vi.unstubAllGlobals());

describe('motion', () => {
  it('keeps the duration by default', () => {
    vi.stubGlobal('matchMedia', media(false));
    expect(reducedMotion()).toBe(false);
    expect(motion(160)).toBe(160);
  });

  it('is 0 under prefers-reduced-motion', () => {
    vi.stubGlobal('matchMedia', media(true));
    expect(reducedMotion()).toBe(true);
    expect(motion(160)).toBe(0);
  });

  it('keeps the duration where matchMedia does not exist (tests, SSR)', () => {
    vi.stubGlobal('matchMedia', undefined);
    expect(motion(200)).toBe(200);
  });
});
