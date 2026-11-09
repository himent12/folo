import { describe, expect, it } from 'vitest';
import { area, synthEvaluation } from './evaluation';

const classes = ['bolt', 'nut', 'washer', 'screw', 'spring', 'rivet', 'pin', 'bearing'];
const metrics = { map50: 0.78, map50_95: 0.6, precision: 0.82, recall: 0.74 };

describe('synthEvaluation', () => {
  const ev = synthEvaluation({ seed: 'm_test', classes, metrics });

  it('is deterministic', () => {
    expect(synthEvaluation({ seed: 'm_test', classes, metrics })).toEqual(ev);
    expect(synthEvaluation({ seed: 'm_other', classes, metrics })).not.toEqual(ev);
  });

  it('draws monotone PR curves whose area is close to the class AP50', () => {
    ev.curves.forEach((curve, c) => {
      for (let i = 1; i < curve.length; i++) {
        expect(curve[i]!.x).toBeGreaterThanOrEqual(curve[i - 1]!.x);
        expect(curve[i]!.y).toBeLessThanOrEqual(curve[i - 1]!.y);
      }
      expect(area(curve)).toBeCloseTo(ev.perClass[c]!.ap50, 1);
    });
  });

  it('centres synthesised per-class AP exactly on the model metrics', () => {
    const mean = ev.perClass.reduce((a, c) => a + c.ap50, 0) / classes.length;
    expect(Math.abs(mean - metrics.map50)).toBeLessThan(0.001);
    const mean95 = ev.perClass.reduce((a, c) => a + c.ap50_95, 0) / classes.length;
    expect(Math.abs(mean95 - metrics.map50_95)).toBeLessThan(0.001);
    // The 101-point AP of the mean curve (the "All classes" line) is the model's mAP50.
    expect(Math.abs(ev.mean.reduce((a, p) => a + p.y, 0) / ev.mean.length - metrics.map50)).toBeLessThan(0.001);
  });

  it('uses given per-class metrics and instances', () => {
    const perClass = classes.map((_, c) => ({ class: c, map50: 0.5, map50_95: 0.3, precision: 0.7, recall: 0.6 }));
    const instances = classes.map((_, c) => 100 + c);
    const e = synthEvaluation({ seed: 'm', classes, metrics, perClass, instances });
    expect(e.perClass.map((c) => c.ap50)).toEqual(classes.map(() => 0.5));
    expect(e.perClass.map((c) => c.instances)).toEqual(instances);
  });

  it('builds a mostly diagonal confusion matrix whose rows add up to the instances', () => {
    const n = classes.length;
    expect(ev.confusion).toHaveLength(n + 1);
    for (let t = 0; t < n; t++) {
      const row = ev.confusion[t]!;
      expect(row.reduce((a, b) => a + b, 0)).toBe(ev.perClass[t]!.instances);
      expect(row[t]).toBe(Math.max(...row));
    }
    // Lookalikes are confused more than unrelated classes: bolt (0) ↔ screw (3) vs bolt ↔ nut (1).
    expect(ev.confusion[0]![3]!).toBeGreaterThan(ev.confusion[0]![1]!);
    expect(ev.confusion[n]![n]).toBe(0);
  });
});
