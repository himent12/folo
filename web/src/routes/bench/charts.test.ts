import { describe, expect, it } from 'vitest';
import type { BenchEntry, BenchGeneration, BenchResult } from '$lib/api/types';
import { MockApi } from '$lib/api/mock/index';
import {
  assignSlots,
  byWorstKept,
  classRows,
  compareShown,
  conditionRows,
  dotDomain,
  generationDomain,
  generationPoints,
  labelCurves,
  labelTradeoff,
  reliabilityPoints,
  reliabilityStrip,
  severityDomain,
  severityPoints,
} from './charts';

describe('the reliability diagram', () => {
  it('draws every filled bin at its mean confidence and share right, leaving out the empty [0, 0.1) bin', async () => {
    const r = await new MockApi('normal', 0).benchResult('bn_3c90', 'be_f101');
    const cal = r.calibration!;
    const pts = reliabilityPoints(cal);
    const filled = cal.bins.filter((b) => b.count > 0);
    expect(pts.map((p) => [p.x, p.y])).toEqual(filled.map((b) => [b.confidence, b.precision]));
    expect(pts.every((p) => p.x >= 0.1)).toBe(true);
    // The fullest bin is the biggest dot; the strip has one bar per bin from 0.1, the counts as sent.
    const most = filled.reduce((a, b) => (b.count > a.count ? b : a));
    expect(Math.max(...pts.map((p) => p.r!))).toBe(pts[filled.indexOf(most)]!.r);
    expect(reliabilityStrip(cal).map((s) => s.v)).toEqual(cal.bins.slice(1).map((b) => b.count));
  });

  it('draws a bin with under 5% of the detections hollow', () => {
    const bins = [
      { lo: 0.1, hi: 0.2, count: 3, confidence: 0.15, precision: 0.9 },
      { lo: 0.5, hi: 0.6, count: 600, confidence: 0.55, precision: 0.5 },
    ];
    expect(reliabilityPoints({ bins }).map((p) => p.hollow)).toEqual([true, false]);
  });
});

describe('the label curves', () => {
  it('draws precision and recall where labels are kept, a gap where none is', () => {
    const curve = [
      { confidence: 0, precision: 0.6, recall: 0.8, labels: 100 },
      { confidence: 0.5, precision: 0.92, recall: 0.5, labels: 40 },
      { confidence: 1, precision: null, recall: 0, labels: 0 },
    ];
    const c = labelCurves(curve);
    expect(c.precision.map((p) => p.y)).toEqual([0.6, 0.92, null]);
    expect(c.recall.map((p) => p.y)).toEqual([0.8, 0.5, null]);
    expect(labelTradeoff(curve)).toEqual([
      { x: 0.5, y: 0.92 },
      { x: 0.8, y: 0.6 },
    ]);
  });
});

describe('robustness', () => {
  it('starts each condition at the unchanged mAP and orders the table from the worst kept', () => {
    const c = { condition: 'haze', weight: 1, retention: 0.7, points: [{ severity: 1, map50_95: 0.6, map50: 0.8, retention: 0.94 }, { severity: 5, map50_95: 0.3, map50: 0.5, retention: 0.47 }] };
    expect(severityPoints(c, 0.64)).toEqual([
      { x: 0, y: 0.64 },
      { x: 1, y: 0.6 },
      { x: 5, y: 0.3 },
    ]);
    expect(severityDomain(0.64, [c], 200).domain[0]).toBe(0);
    expect(severityDomain(0.64, [c], 200).domain[1]).toBeGreaterThanOrEqual(0.64);
    expect(byWorstKept([{ condition: 'a', retention: 0.9 }, { condition: 'b', retention: 0.4 }]).map((x) => x.condition)).toEqual(['b', 'a']);
  });
});

describe('the generations chart', () => {
  const g = (generation: number, map: number | null, ci: [number, number] | null): BenchGeneration =>
    ({ generation, map50_95: map, ci95: ci, labels: null, run_id: null, model_id: null, model_name: null, entry_id: null, score: null }) as BenchGeneration;

  it('draws each generation with its interval, and fits the axis to the whiskers without cutting one', () => {
    const gens = [g(0, 0.794, [0.752, 0.831]), g(1, 0.816, [0.776, 0.854]), g(2, null, null)];
    expect(generationPoints(gens)).toEqual([
      { x: 0, y: 0.794, lo: 0.752, hi: 0.831 },
      { x: 1, y: 0.816, lo: 0.776, hi: 0.854 },
      { x: 2, y: null },
    ]);
    const fit = generationDomain(gens, 200);
    expect(fit.domain[0]).toBeLessThanOrEqual(0.752);
    expect(fit.domain[1]).toBeGreaterThanOrEqual(0.854);
    expect(fit.domain[0]).toBeGreaterThan(0);
  });
});

describe('comparing entries', () => {
  it('keeps each entry’s colour when another leaves', () => {
    const a = assignSlots(['x', 'y', 'z'], {});
    expect(a).toEqual({ x: 0, y: 1, z: 2 });
    expect(assignSlots(['x', 'z'], a)).toEqual({ x: 0, z: 2 });
    expect(assignSlots(['x', 'z', 'w'], { x: 0, z: 2 })).toEqual({ x: 0, z: 2, w: 1 });
  });

  it('lines conditions up in plan order, null where an entry did not run one, dropping those none ran', () => {
    const res = (conds: [string, number][]) => ({ robustness: { retention: 0, conditions: conds.map(([condition, retention]) => ({ condition, retention, weight: 1, points: [] })) } }) as unknown as BenchResult;
    expect(conditionRows(['low_light', 'haze', 'jpeg'], [res([['low_light', 0.8], ['haze', 0.7]]), res([['haze', 0.6]]), undefined])).toEqual([
      { key: 'low_light', values: [0.8, null, null] },
      { key: 'haze', values: [0.7, 0.6, null] },
    ]);
  });

  it('lines classes up by their AP, the hardest first', () => {
    const res = (aps: number[]) => ({ accuracy: { per_class: aps.map((map50_95, c) => ({ class: c, truths: 10 + c, map50: 0, map50_95 })) } }) as unknown as BenchResult;
    const rows = classRows(['bolt', 'nut', 'pin'], [605, 410, 77], [res([0.8, 0.3, 0.6]), res([0.7, 0.1, 0.65])]);
    expect(rows.map((r) => r.name)).toEqual(['nut', 'pin', 'bolt']);
    // The benchmark's own count of the class, never one entry's (an agent on 24 images counted 11 of them).
    expect(rows[0]).toMatchObject({ truths: 410, graded: [11, 11], values: [0.3, 0.1] });
  });

  it('fits a dot plot to its values, from 0 when they spread low, at least 0.2 wide', () => {
    expect(dotDomain([0.62, 0.95, 0.88])).toEqual([0.6, 1]);
    expect(dotDomain([0.1, 0.9])).toEqual([0, 0.95]);
    expect(dotDomain([0.9, 0.91])).toEqual([0.75, 0.95]);
    expect(dotDomain([null])).toEqual([0, 1]);
  });

  it('shows the finished entries the URL names and lists the others', () => {
    const e = (id: string, status: BenchEntry['status']) => ({ id, status, score: status === 'done' ? { total: 1, partial: false, axes: [] } : null }) as BenchEntry;
    const r = compareShown(['a', 'b', 'nope'], [e('a', 'done'), e('b', 'running')]);
    expect(r.shown.map((x) => x.id)).toEqual(['a']);
    expect(r.missing).toEqual(['b', 'nope']);
  });
});
