import { describe, expect, it } from 'vitest';
import { ago, atBestF1, date, bytes, sizeParts, compact, conf, count, deviceName, duration, eta, hoursText, lr, metric, ms, pct, pricePerHour, sentence, timeOfDay, usd, usdRange } from './format';

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
  it('moves a size that rounds up to 1024 of a unit to the next unit', () => {
    expect(bytes(2 ** 20 - 1)).toBe('1.0 MiB');
    expect(bytes(2 ** 30 - 1)).toBe('1.0 GiB');
    expect(bytes(999_999, 'metric')).toBe('1.0 MB');
    expect(bytes(1023)).toBe('1,023 B');
  });
  it("splits a size into its number and its own unit at Desktop's steps (Models' Size cells)", () => {
    expect(sizeParts(38_797_312)).toEqual(['37.0', 'MiB']);
    expect(sizeParts(1.25 * 2 ** 30)).toEqual(['1.3', 'GiB']);
    expect(sizeParts(300 * 1024 + 300)).toEqual(['300', 'KiB']);
    expect(sizeParts(512)).toEqual(['512', 'B']);
    expect(sizeParts(5_400_000, 'metric')).toEqual(['5.4', 'MB']);
    expect(sizeParts(null)).toEqual(['—', '']);
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

  // x1-parity-2: a unit that rounds up reads as the next one, as in Desktop's time.rs.
  it('never says 60m, 24h or 14d ago', () => {
    const now = Date.parse('2026-11-15T12:00:00Z');
    expect(ago('2026-11-15T11:00:20Z', now)).toBe('1h ago');
    expect(ago('2026-11-14T12:15:00Z', now)).toBe('yesterday');
    expect(ago('2026-11-01T23:00:00Z', now)).toBe(date('2026-11-01T23:00:00Z'));
    expect(ago('2026-11-09T14:00:00Z', now)).toBe('6d ago');
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

describe('sentence', () => {
  it('capitalises and closes a message, but never an identifier it starts with', () => {
    expect(sentence('device gpu:3 is not available')).toBe('Device gpu:3 is not available.');
    expect(sentence('`claude` printed no result')).toBe('`claude` printed no result.');
    expect(sentence('`claude` printed no result', true)).toBe('claude printed no result.');
    expect(sentence('run `folo serve` first', true)).toBe('Run folo serve first.');
  });
});

describe('count', () => {
  it('says one of a thing in the singular', () => {
    expect(count(1, 'class')).toBe('1 class');
    expect(count(2, 'class')).toBe('2 classes');
    expect(count(0, 'image')).toBe('0 images');
    expect(count(2400, 'image')).toBe('2,400 images');
    expect(count(1, 'box')).toBe('1 box');
    expect(count(3, 'box')).toBe('3 boxes');
  });
});
