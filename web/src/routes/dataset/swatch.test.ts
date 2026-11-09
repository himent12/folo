import { describe, expect, it } from 'vitest';
import { classInk } from '$lib/design/classInk.svelte';
import { swatchStyle } from './swatch';

describe('swatchStyle', () => {
  it('fills a present class with its hue', () => {
    expect(swatchStyle(2, false)).toBe(`background: ${classInk(2)}`);
  });

  it('draws an absent class as a 40 % fill inside a 1 px outline, not a faded square', () => {
    const s = swatchStyle(2, true);
    expect(s).toContain(`color-mix(in srgb, ${classInk(2)} 40%, transparent)`);
    expect(s).toContain('box-shadow: inset 0 0 0 1px');
    expect(s).not.toContain('opacity');
  });
});
