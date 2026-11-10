import { describe, expect, it } from 'vitest';
import { asOf, etaText } from './freshness.svelte';

const NOW = Date.parse('2026-11-09T12:00:00Z');

describe('freshness', () => {
  it('says how old a state is, never "just now" for one that went quiet', () => {
    expect(asOf(NOW - 10_000, NOW)).toBe('seconds ago');
    expect(asOf(NOW - 120_000, NOW)).toBe('2m ago');
  });
  it('qualifies an estimate whose source went quiet, and leaves a current one alone', () => {
    expect(etaText(420, null, NOW)).toBe('7m left');
    expect(etaText(420, NOW - 120_000, NOW)).toBe('~7m left, as of 2m ago');
    expect(etaText(null, NOW - 120_000, NOW)).toBe('—');
  });
});
