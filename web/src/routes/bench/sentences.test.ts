import { describe, expect, it } from 'vitest';
import type { BenchCalibrationBin, BenchEntry, BenchLabelPoint, BenchLoop } from '$lib/api/types';
import { catalog } from '$lib/api/mock/benchCatalog';
import { metric } from '$lib/format';
import { bestAsPrinted, diffAsPrinted } from './charts';
import {
  axisRaw,
  calibrationReading,
  coverageFacts,
  digitsApart,
  forecastText,
  intervalsVerdict,
  labelForecast,
  loopVerdictText,
  mapWithInterval,
  partialWhy,
  selfLabelReading,
  share,
  shareApart,
} from './words';

// The generated sentences hold what they claim as printed: for a grid of inputs, each claim ("short of 90%", "above",
// "overconfident", "± h") is checked against the numbers the sentence itself prints, never against the raw values a
// reader cannot see (docs/BENCH.md defines the numbers; this file holds the words to them).

/** Every number a sentence prints, in order (percents as shares). */
const numbers = (s: string) => [...s.matchAll(/(\d[\d,]*(?:\.\d+)?)(%?)/g)].map((m) => Number(m[1]!.replace(/,/g, '')) / (m[2] ? 100 : 1));

describe('printing so the claim is true', () => {
  it('prints two numbers with the digits that tell them apart', () => {
    for (const [a, b] of [
      [0.8661, 0.866],
      [0.6201, 0.62],
      [0.5, 0.4999999],
      [0.1, 0.2],
    ]) {
      const d = digitsApart(a!, b!);
      expect(Math.sign(Number(a!.toFixed(d)) - Number(b!.toFixed(d)))).toBe(Math.sign(a! - b!));
    }
  });

  it('never prints a share under 90% as "90%", nor a small share as "0%"', () => {
    for (let x = 0.85; x < 0.9; x += 0.0007) {
      const t = shareApart(x, 0.9);
      expect(Number(t.replace('%', '')), t).toBeLessThan(90);
    }
    for (const x of [0.0001, 0.0008, 0.004, 0.0099]) expect(share(x)).not.toBe('0%');
    expect(share(0)).toBe('0%');
  });
});

const pt = (confidence: number, precision: number | null, recall: number, labels: number): BenchLabelPoint => ({ confidence, precision, recall, labels });

describe('self-labeling sentences', () => {
  it('say "short of 90%" only of a share printed under 90%, for any precision under it', () => {
    for (let p = 0.5; p < 0.9; p += 0.0013) {
      const s = selfLabelReading({ threshold: null, precision: p, recall: 0, mean_iou: null, labels_per_image: 4, labels: 200, curve: [pt(0, p, 0.5, 200)] }, false);
      const right = Number(/are ([\d.]+)% right, short of/.exec(s)![1]);
      expect(right, s).toBeLessThan(90);
      const raw = axisRaw('self_labeling', { self_labeling: { recall: 0, precision: p, threshold: null, labels: 200 } } as never, false)!;
      expect(Number(raw.split('%')[0]), raw).toBeLessThan(90);
    }
  });

  it('state the label count wherever a threshold is read, and never "0.0 per image" beside labels', () => {
    for (const labels of [10, 11, 99, 1363]) {
      for (const lpi of [0.001, 0.04, 0.05, 2.5]) {
        const s = selfLabelReading({ threshold: 0.32, precision: 0.91, recall: 0.0008, mean_iou: 0.8, labels_per_image: lpi, labels, curve: [] }, true);
        expect(s).toContain(`${labels.toLocaleString('en-US')} labels`);
        expect(s).not.toMatch(/0\.0 per image/);
        expect(s).not.toMatch(/ 0% of all objects/);
      }
    }
  });

  it('never claim a precise handful is a threshold', () => {
    const curve = [pt(0, 0.4, 0.6, 5000), pt(0.5, 0.7, 0.2, 800), pt(0.78, 1, 0.0002, 3)];
    const s = selfLabelReading({ threshold: null, precision: 0.4, recall: 0, mean_iou: 0.7, labels_per_image: 20, labels: 5000, curve }, true);
    expect(s).toMatch(/^No confidence keeps 10 labels or more/);
    expect(s).toMatch(/its 3 labels are 100% right, too few to rest a threshold on/);
  });
});

describe('the forecast of a loop', () => {
  const curve = Array.from({ length: 101 }, (_, i) => {
    const t = i / 100;
    const precision = 0.3 + 0.65 * t;
    const recall = 0.7 * (1 - t) ** 2;
    const labels = Math.round((3000 * recall) / precision);
    return pt(t, labels ? precision : null, labels ? recall : 0, labels);
  });

  it('agrees with the server’s rules at every confidence: under 10 pool images stops it, no confidence stops it earlier', () => {
    for (let c = 0; c <= 1.0001; c += 0.05) {
      for (const pool of [8, 50, 1920]) {
        const f = labelForecast({ threshold: null, readiness: 'not_ready', curve }, 300, pool, Math.round(c * 100) / 100);
        expect(f.stops).toBe(f.images !== null && f.images < 10);
        const w = forecastText(f, pool);
        if (f.stops) expect(w.text).toMatch(/A loop stops when fewer than 10 pool images get a box\./);
        // "under the 90%" said only of a share printed under 90%.
        const m = /, ([\d.]+)% of its labels are right/.exec(w.text);
        if (m && /under the 90%/.test(w.text)) expect(Number(m[1])).toBeLessThan(90);
      }
    }
    const none = labelForecast({ threshold: null, readiness: 'not_ready', curve }, 300, 1920, null);
    expect(none).toMatchObject({ threshold: null, stops: true });
    expect(forecastText(none, 1920).text).toMatch(/stops before labeling/);
  });
});

describe('calibration sentences', () => {
  const bin = (lo: number, count: number, confidence: number, precision: number): BenchCalibrationBin => ({ lo, hi: lo + 0.1, count, confidence, precision });

  it('quote only a bin whose own gap has the tone’s sign, as printed', () => {
    const grid: BenchCalibrationBin[][] = [];
    for (const lowGap of [-0.15, -0.05, 0.03, 0.12])
      for (const highGap of [-0.3, -0.04, 0.05, 0.4])
        for (const highCount of [3, 30, 300]) grid.push([bin(0.1, 800, 0.15, 0.15 - lowGap), bin(0.5, 400, 0.55, 0.55 - (lowGap + highGap) / 2), bin(0.9, highCount, 0.95, Math.max(0, Math.min(1, 0.95 - highGap)))]);
    for (const bins of grid) {
      const n = bins.reduce((s, b) => s + b.count, 0);
      const ece = bins.reduce((s, b) => s + (b.count / n) * Math.abs(b.confidence! - b.precision!), 0);
      const r = calibrationReading({ ece, bins });
      const m = /when it says ([\d.]+), it is right (\d+)%/.exec(r.sentence);
      if (!m) continue;
      const said = Number(m[1]);
      const right = Number(m[2]) / 100;
      if (r.tone === 'over') expect(said, r.sentence).toBeGreaterThan(right);
      if (r.tone === 'under') expect(said, r.sentence).toBeLessThan(right);
      if (r.tone === 'close') expect(Math.abs(said - right), r.sentence).toBeLessThanOrEqual(0.06);
      // A quoted bin holds enough detections to mean something.
      const quoted = bins.find((b) => b.confidence!.toFixed(2) === m[1]);
      expect(quoted!.count, r.sentence).toBeGreaterThanOrEqual(20);
    }
  });
});

describe('interval sentences', () => {
  it('say "above" or "below" only when the printed numbers say so, for a loop', () => {
    for (let k = 0; k < 400; k++) {
      const lo = 0.5 + k * 0.0011;
      const hi = lo + 0.02;
      for (const last of [hi + 0.00004, hi + 0.0004, lo - 0.00004, lo - 0.0005]) {
        const verdict = last > hi ? 'improved' : last < lo ? 'regressed' : 'held';
        const t = loopVerdictText({ verdict, generations: [{ generation: 0, map50_95: lo + 0.01, ci95: [lo, hi] }, { generation: 1, map50_95: last, ci95: null }] } as unknown as BenchLoop)!;
        const [g, a, b] = numbers(t.replace(/generation \d+ /, ''));
        if (verdict === 'improved') expect(g!, t).toBeGreaterThan(b!);
        if (verdict === 'regressed') expect(g!, t).toBeLessThan(a!);
      }
    }
  });

  it('print "±" only for an interval symmetric as printed, and both ends otherwise', () => {
    for (let k = 0; k < 300; k++) {
      const m = 0.3 + k * 0.00137;
      const ci: [number, number] = [m - 0.02 - (k % 3) * 0.0004, m + 0.02 + (k % 5) * 0.0003];
      const t = mapWithInterval(m, ci);
      const nums = numbers(t.replace('mAP50-95', ''));
      if (t.includes('±')) expect(Math.round(nums[0]! * 1000) - Math.round(ci[0] * 1000)).toBe(Math.round(ci[1] * 1000) - Math.round(nums[0]! * 1000));
      else expect(t).toContain(`${metric(ci[0])}–${metric(ci[1])}`);
    }
  });

  it('call entries clearly apart from the leader only when their printed intervals do not meet', () => {
    for (let k = 0; k < 200; k++) {
      const gap = (k - 100) * 0.00003;
      const lead = { name: 'lead', map: 0.7, ci: [0.62 + gap, 0.75] as [number, number] };
      const other = { name: 'other', map: 0.55, ci: [0.5, 0.62] as [number, number] };
      const t = intervalsVerdict([lead, other]);
      const meet = Math.round(lead.ci[0] * 1000) <= Math.round(other.ci[1] * 1000);
      expect(/clearly ahead/.test(t), `${gap}: ${t}`).toBe(!meet);
    }
  });
});

describe('a partial score says why', () => {
  const b = { images: 240, classes: ['koi', 'goldfish', 'frog'], plan: catalog().default_plan };
  const axes = (cal: number | null) => (['accuracy', 'robustness', 'self_labeling', 'calibration', 'localization'] as const).map((axis) => ({ axis, value: axis === 'calibration' ? cal : 0.5, weight: 0.2 }));
  const entry = (o: Partial<BenchEntry>) => ({ id: 'e', name: 'x', images: 240, conditions: true, scored: true, missing_classes: [], status: 'done', conditions_run: 12, contestant: { kind: 'external' }, score: { total: 50, partial: true, axes: axes(null) }, ...o }) as BenchEntry;

  it('names calibration not measured, not "conditions left out", for a complete scored entry with no confidence from 0.1', () => {
    const e = entry({});
    expect(coverageFacts(e, b)).toEqual(['calibration not measured']);
    expect(partialWhy(e, b)).toBe('Partial: calibration not measured (no detection reached confidence 0.1). It compares only loosely with complete entries.');
    // An external entry that left a condition out and is otherwise whole still says so.
    expect(coverageFacts(entry({ score: { total: 50, partial: true, axes: axes(0.9) } }), b)).toEqual(['some conditions left out']);
  });
});

describe('Compare’s best and differences', () => {
  it('compare as printed: a tie at the shown precision has no best, and the difference of equal cells is ±0', () => {
    expect([...bestAsPrinted([0.6801, 0.6804, 0.5], 2)]).toEqual([]);
    expect([...bestAsPrinted([0.69, 0.6804, 0.5], 2)]).toEqual([0]);
    expect(diffAsPrinted(0.6804, 0.6751, 2)).toBe('±0');
    expect(diffAsPrinted(0.6851, 0.6749, 2)).toBe('+0.02');
    expect(diffAsPrinted(null, 0.5, 2)).toBeNull();
  });

  it('mark best on mAP only when its interval clears every other’s', () => {
    expect([...bestAsPrinted([0.641, 0.606], 3, [[0.62, 0.66], [0.58, 0.63]])]).toEqual([]);
    expect([...bestAsPrinted([0.641, 0.506], 3, [[0.62, 0.66], [0.48, 0.53]])]).toEqual([0]);
    // An entry without an interval cannot be cleared.
    expect([...bestAsPrinted([0.641, 0.506], 3, [[0.62, 0.66], null])]).toEqual([]);
  });
});
