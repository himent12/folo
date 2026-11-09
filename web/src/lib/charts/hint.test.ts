import { describe, expect, it } from 'vitest';
import { chartHint, claimOnce, KEY_HINT, TAP_HINT } from './hint';

function memory(): Storage {
  const m = new Map<string, string>();
  return {
    get length() {
      return m.size;
    },
    clear: () => m.clear(),
    getItem: (k) => m.get(k) ?? null,
    key: (i) => [...m.keys()][i] ?? null,
    removeItem: (k) => void m.delete(k),
    setItem: (k, v) => void m.set(k, v),
  };
}

describe('claimOnce', () => {
  it('is true once per storage', () => {
    const s = memory();
    expect(claimOnce('k', () => s)).toBe(true);
    expect(claimOnce('k', () => s)).toBe(false);
    expect(claimOnce('other', () => s)).toBe(true);
  });

  it('shows nothing when storage throws', () => {
    expect(
      claimOnce('k', () => {
        throw new Error('blocked');
      }),
    ).toBe(false);
    const broken = { ...memory(), setItem: () => { throw new Error('quota'); } } as Storage;
    expect(claimOnce('k', () => broken)).toBe(false);
  });
});

describe('chartHint', () => {
  it('says how to read the chart until a value is read out', () => {
    expect(chartHint({ reading: false, keyFocus: true, tap: false })).toBe(KEY_HINT);
    expect(chartHint({ reading: false, keyFocus: false, tap: true })).toBe(TAP_HINT);
    expect(chartHint({ reading: false, keyFocus: true, tap: true })).toBe(KEY_HINT);
    expect(chartHint({ reading: true, keyFocus: true, tap: true })).toBeNull();
    expect(chartHint({ reading: false, keyFocus: false, tap: false })).toBeNull();
  });
});
