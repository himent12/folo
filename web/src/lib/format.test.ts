import { describe, expect, it } from 'vitest';
import { ago, atBestF1, bytes, compact, conf, deviceName, duration, eta, hoursText, lr, metric, ms, pct, pricePerHour, timeOfDay, usd, usdRange } from './format';

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
  it('moves a count that rounds up to 1000 of a unit to the next unit', () => {
    expect(compact(999_999)).toBe('1M');
    expect(compact(999_499)).toBe('999k');
    expect(compact(999_950_000)).toBe('1B');
    expect(compact(-999_999)).toBe('-1M');
  });
  it('formats bytes in both unit systems', () => {
    expect(bytes(16 * 2 ** 30)).toBe('16.0 GiB');
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
  it('never writes a learning rate with a mantissa of 10', () => {
    // 0.009996 rounds to 10.00 × 10⁻³ in the old exponent: that is 1.00 × 10⁻².
    expect(lr(0.009996)).toBe('1.00 × 10⁻²');
    expect(lr(0.00099951)).toBe('1.00 × 10⁻³');
    expect(lr(-0.009996)).toBe('-1.00 × 10⁻²');
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

describe('money and time limits (cloud compute)', () => {
  it('writes a price per hour with two decimals, and a dash for compute that is not billed', () => {
    expect(pricePerHour(0.69)).toBe('$0.69/h');
    expect(pricePerHour(2.5)).toBe('$2.50/h');
    expect(pricePerHour(null)).toBe('—');
  });

  it('writes a cost with two decimals, under a cent as <$0.01, from $100 in whole dollars', () => {
    expect(usd(0.42)).toBe('$0.42');
    expect(usd(0)).toBe('$0.00');
    expect(usd(0.004)).toBe('<$0.01');
    expect(usd(0.005)).toBe('$0.01');
    expect(usd(99.994)).toBe('$99.99');
    expect(usd(1240.4)).toBe('$1,240');
    expect(usd(undefined)).toBe('—');
  });

  it('writes a range low to high, one amount when both ends read the same', () => {
    expect(usdRange(0.12, 0.93)).toBe('$0.12–$0.93');
    expect(usdRange(0.93, 0.12)).toBe('$0.12–$0.93');
    expect(usdRange(0.421, 0.419)).toBe('$0.42');
  });

  it('writes hours as the limit is set', () => {
    expect(hoursText(12)).toBe('12 h');
    expect(hoursText(0.5)).toBe('0.5 h');
    expect(hoursText(0.25)).toBe('0.25 h');
  });

  it('writes a deadline as a clock time, with the date when it is not today', () => {
    const now = new Date(2026, 10, 8, 9, 0).getTime();
    expect(timeOfDay(new Date(2026, 10, 8, 14, 32).toISOString(), now)).toBe('14:32');
    expect(timeOfDay(new Date(2026, 10, 9, 2, 5).toISOString(), now)).toBe('Nov 9, 02:05');
    expect(timeOfDay(null, now)).toBe('—');
  });
});
