import { describe, expect, it } from 'vitest';
import type { BenchLabelPoint } from '../api/types';
import { axisScores, axisShares, conditionRetention, expectedCalibrationError, histogramMeanIou, labelOperatingPoint, localizationAxis, readiness, retention, robustnessAxis, scoreTotal } from './score';

// Hand-worked cases of docs/BENCH.md's definitions.
describe('the definitions', () => {
  it('retention: min(1, under / unchanged), 0 when unchanged is 0; a condition is the mean of its severities', () => {
    expect(retention(0.3, 0.6)).toBe(0.5);
    expect(retention(0.7, 0.6)).toBe(1);
    expect(retention(0.2, 0)).toBe(0);
    expect(conditionRetention([{ retention: 1 }, { retention: 0.5 }, { retention: 0.3 }])).toBeCloseTo(0.6, 9);
  });

  it('robustness: the weighted mean of the conditions; null when none ran', () => {
    expect(robustnessAxis([{ weight: 3, retention: 0.5 }, { weight: 1, retention: 0.9 }])).toBeCloseTo(0.6, 9);
    expect(robustnessAxis([])).toBeNull();
  });

  it('ECE: the share-weighted gap between confidence and precision over the bins', () => {
    const bins = [
      { lo: 0, hi: 0.1, count: 0, confidence: null, precision: null },
      { lo: 0.5, hi: 0.6, count: 30, confidence: 0.55, precision: 0.45 },
      { lo: 0.9, hi: 1, count: 10, confidence: 0.95, precision: 0.95 },
    ];
    expect(expectedCalibrationError(bins)).toBeCloseTo(0.075, 9);
    expect(expectedCalibrationError([])).toBe(0);
  });

  it('localization: mean IoU rescaled from 0.5–1, 0 with nothing right; a histogram stands for its bin centres', () => {
    expect(localizationAxis(0.75)).toBe(0.5);
    expect(localizationAxis(null)).toBe(0);
    expect(histogramMeanIou([0, 0, 0, 0, 0, 2, 2, 0, 0, 0])).toBeCloseTo(0.8, 9);
    expect(histogramMeanIou(new Array(10).fill(0))).toBeNull();
  });

  it('self-labeling: the lowest confidence with 90% precision and labels kept; one point without confidences', () => {
    const p = (confidence: number, precision: number | null, recall: number, labels: number): BenchLabelPoint => ({ confidence, precision, recall, labels });
    const curve = [p(0, 0.5, 0.9, 100), p(0.3, 0.88, 0.7, 60), p(0.4, 0.92, 0.6, 50), p(0.5, 0.97, 0.4, 30), p(1, null, 0, 0)];
    expect(labelOperatingPoint(curve)).toEqual({ threshold: 0.4, point: curve[2] });
    expect(labelOperatingPoint([p(0, 0.5, 0.9, 1), p(1, null, 0, 0)])).toEqual({ threshold: null, point: null });
    expect(labelOperatingPoint([p(1, 0.93, 0.7, 40)]).threshold).toBeNull();
    expect(labelOperatingPoint([p(1, 0.93, 0.7, 40)]).point?.recall).toBe(0.7);
    expect(labelOperatingPoint([p(1, 0.85, 0.7, 40)]).point).toBeNull();
    expect([readiness(0.5), readiness(0.25), readiness(0.249)]).toEqual(['ready', 'partial', 'not_ready']);
  });

  it('the score: 100 · Σ w·axis / Σ w over the measured axes', () => {
    const all = axisScores({ accuracy: 0.6, robustness: 0.8, self_labeling: 0.5, calibration: 0.95, localization: 0.6 });
    expect(scoreTotal(all)).toBeCloseTo(100 * (0.35 * 0.6 + 0.25 * 0.8 + 0.2 * 0.5 + 0.1 * 0.95 + 0.1 * 0.6), 9);
    const agent = axisScores({ accuracy: 0.4, robustness: null, self_labeling: 0.7, calibration: null, localization: 0.45 });
    expect(scoreTotal(agent)).toBeCloseTo((100 * (0.35 * 0.4 + 0.2 * 0.7 + 0.1 * 0.45)) / 0.65, 9);
    const shares = axisShares(agent);
    expect(shares.robustness).toBe(0);
    expect(shares.accuracy).toBeCloseTo(0.35 / 0.65, 9);
  });
});
