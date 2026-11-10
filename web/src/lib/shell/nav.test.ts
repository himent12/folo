import { describe, expect, it } from 'vitest';
import { mainNav, tabNav, tabOf } from './nav';

describe('the phone tab bar', () => {
  it('holds five tabs; every other section lives under one of them', () => {
    expect(tabNav.map((n) => n.id)).toEqual(['overview', 'datasets', 'runs', 'models', 'playground']);
    for (const n of mainNav) expect(tabNav.some((t) => t.id === tabOf(n.id))).toBe(true);
    expect(tabOf('benchmarks')).toBe('models');
    expect(tabOf('runs')).toBe('runs');
  });
});
