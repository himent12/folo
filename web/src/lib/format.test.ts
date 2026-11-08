import { describe, expect, it } from 'vitest';
import { ago, bytes, compact, duration, lr, metric, ms, pct } from './format';

describe('format', () => {
  it('formats metrics and percents', () => {
    expect(metric(0.74123)).toBe('0.741');
    expect(metric(null)).toBe('—');
    expect(pct(0.7412)).toBe('74.1%');
  });
  it('compacts counts', () => {
    expect(compact(940)).toBe('940');
    expect(compact(12_400)).toBe('12.4k');
    expect(compact(2_630_000)).toBe('2.63M');
    expect(compact(56_970_000)).toBe('57.0M');
  });
  it('formats bytes in both unit systems', () => {
    expect(bytes(16 * 2 ** 30)).toBe('16.0 GiB');
    expect(bytes(5_400_000, 'metric')).toBe('5.40 MB');
    expect(bytes(512)).toBe('512 B');
  });
  it('formats durations', () => {
    expect(duration(42)).toBe('42s');
    expect(duration(750)).toBe('12m 30s');
    expect(duration(3840)).toBe('1h 04m');
  });
  it('formats milliseconds and learning rates', () => {
    expect(ms(3.14159)).toBe('3.14 ms');
    expect(ms(12.34)).toBe('12.3 ms');
    expect(lr(0.0098)).toBe('9.80e-3');
  });
  it('formats relative times', () => {
    const now = Date.parse('2026-11-08T14:30:00Z');
    expect(ago('2026-11-08T14:29:40Z', now)).toBe('just now');
    expect(ago('2026-11-08T14:27:00Z', now)).toBe('3m ago');
    expect(ago('2026-11-08T12:30:00Z', now)).toBe('2h ago');
    expect(ago('2026-11-07T10:00:00Z', now)).toBe('yesterday');
    expect(ago('2026-11-03T14:30:00Z', now)).toBe('5d ago');
  });
});
