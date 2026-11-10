// Series hues in text take their text ink (DESIGN_DIRECTION §2.5): `--chart-N` and `--chart-fill-N` are marks (3:1),
// `--chart-text-N` is held to 4.5:1 by design/contrast.mjs.
import { describe, expect, it } from 'vitest';
import { seriesText } from './ink';

describe('series text ink', () => {
  it('maps a series line or fill to its text ink and keeps anything else', () => {
    expect(seriesText('var(--chart-1)')).toBe('var(--chart-text-1)');
    expect(seriesText('var(--chart-fill-4)')).toBe('var(--chart-text-4)');
    expect(seriesText('var(--text-primary)')).toBe('var(--text-primary)');
    expect(seriesText('var(--chart-context)')).toBe('var(--chart-context)');
  });
});
