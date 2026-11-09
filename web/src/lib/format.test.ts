import { describe, expect, it } from 'vitest';
import { ago, atBestF1, bytes, compact, conf, deviceName, duration, eta, lr, metric, ms, pct } from './format';

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
    expect(compact(56_970_000)).toBe('57M');
    expect(compact(20_000)).toBe('20k');
  });
  it('formats bytes in both unit systems', () => {
    expect(bytes(16 * 2 ** 30)).toBe('16 GiB');
    expect(bytes(5_400_000, 'metric')).toBe('5.4 MB');
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
    expect(lr(0.0098)).toBe('9.80 × 10⁻³');
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

describe('eta', () => {
  it('shows seconds under two minutes, then whole minutes rounded up', () => {
    expect(eta(45)).toBe('45s');
    expect(eta(119)).toBe('1m 59s');
    expect(eta(373)).toBe('7m');
    expect(eta(376)).toBe('7m');
    expect(eta(3600)).toBe('1h 00m');
    expect(eta(3661)).toBe('1h 02m');
    expect(eta(null)).toBe('—');
  });
});

describe('deviceName', () => {
  it('names a device the same way everywhere', () => {
    expect(deviceName('NVIDIA GeForce RTX 4080 SUPER (vulkan)')).toBe('RTX 4080 SUPER');
    expect(deviceName('NVIDIA GeForce RTX 4080 SUPER')).toBe('RTX 4080 SUPER');
    expect(deviceName('AMD Radeon RX 7900 XTX (dx12)')).toBe('RX 7900 XTX');
    expect(deviceName('cpu')).toBe('CPU');
    expect(deviceName('CPU (24 threads)')).toBe('CPU');
    expect(deviceName('AMD Ryzen 9 7950X (cpu)')).toBe('Ryzen 9 7950X (CPU)');
  });
});

describe('operating point (docs/API.md, best_f1_conf)', () => {
  it('names the threshold to two decimals, three below 0.1, and falls back honestly', () => {
    expect(conf(0.312)).toBe('0.31');
    expect(conf(0.045)).toBe('0.045');
    expect(atBestF1(0.312)).toBe('at conf 0.31 (best F1)');
    expect(atBestF1(null)).toBe('at the best-F1 confidence');
    expect(atBestF1(undefined)).toBe('at the best-F1 confidence');
  });
});
