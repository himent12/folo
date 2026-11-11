import { describe, expect, it } from 'vitest';
import { mainNav, tabNav, tabOf } from './nav.fw';

describe('the phone tab bar', () => {
  it('puts every section under one of its tabs, the benchmarks under Models', () => {
    for (const n of mainNav) expect(tabNav.some((t) => t.id === tabOf(n.id)), n.id).toBe(true);
    expect(tabOf('benchmarks')).toBe('models');
  });
});
