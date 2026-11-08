import { describe, expect, it } from 'vitest';
import { formatKeys } from './keys';

describe('formatKeys', () => {
  it('splits chords, sequences and literal plus', () => {
    expect(formatKeys('G R')).toEqual(['G', 'R']);
    expect(formatKeys('+ -')).toEqual(['+', '-']);
    expect(formatKeys('Mod+K').at(-1)).toBe('K');
    expect(formatKeys('Mod+Shift+K')).toHaveLength(3);
  });
});
