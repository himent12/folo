import { describe, expect, it } from 'vitest';
import type { BenchAxisScore, BenchCalibrationBin, BenchEntry, BenchLabelPoint, BenchLoop, BenchResult } from '$lib/api/types';
import { catalog } from '$lib/api/mock/benchCatalog';
import { scoreTotal } from '$lib/bench/score';
import {
  axisPoints,
  axisRaw,
  axisWhyNot,
  calibrationReading,
  compareCaveats,
  intervalsVerdict,
  loopPlanText,
  loopQuestion,
  loopVerdictText,
  robustnessReading,
  selfLabelReading,
  sharesNote,
  teacherEligibility,
} from './words';

// The sentences the result screens generate to read their charts (./words, "reading a result" onwards).

const entry = (o: Partial<BenchEntry>) =>
  ({ id: 'e', name: 'x', images: 360, conditions: true, scored: true, missing_classes: [], status: 'done', conditions_run: 12, score: { total: 60, partial: false, axes: [] }, ...o }) as BenchEntry;

const axes = (v: (number | null)[]): BenchAxisScore[] =>
  (['accuracy', 'robustness', 'self_labeling', 'calibration', 'localization'] as const).map((axis, i) => ({ axis, value: v[i] ?? null, weight: [0.35, 0.25, 0.2, 0.1, 0.1][i]! }));

describe('the score in rows', () => {
  it('gives each axis its points, which add up to the total, measured or not', () => {
    for (const a of [axes([0.64, 0.83, 0.58, 0.94, 0.62]), axes([0.39, null, 0.6, null, 0.4])]) {
      const p = axisPoints(a);
      const sum = Object.values(p).reduce<number>((s, x) => s + (x ?? 0), 0);
      expect(sum).toBeCloseTo(scoreTotal(a), 6);
    }
    expect(axisPoints(axes([0.39, null, 0.6, null, 0.4])).robustness).toBeNull();
  });

  it('says how the others share the score when axes are left out', () => {
    expect(sharesNote(axes([0.6, 0.8, 0.5, 0.9, 0.6]))).toBeNull();
    expect(sharesNote(axes([0.6, 0.8, 0.5, null, 0.6]))).toBe('Calibration was not measured: the other four axes share the score, each counting its weight over 90%.');
    expect(sharesNote(axes([0.6, null, 0.5, null, 0.6]))).toBe(
      'Robustness and Calibration were not measured: the other three axes share the score, each counting its weight over 65%.',
    );
  });

  it('names the number behind each axis in its own unit', () => {
    const r = {
      accuracy: { map50_95: 0.641, ci95: [0.623, 0.659] },
      robustness: { retention: 0.832, conditions: [] },
      calibration: { ece: 0.061, detections: 10, bins: [] },
      localization: { mean_iou: 0.84, matches: 10, histogram: [] },
      self_labeling: { recall: 0.58, precision: 0.91, threshold: 0.62 },
    } as unknown as BenchResult;
    expect(axisRaw('accuracy', r, true)).toBe('mAP50-95 0.641 ± 0.018');
    expect(axisRaw('robustness', r, true)).toBe('keeps 83% of its accuracy');
    expect(axisRaw('self_labeling', r, true)).toBe('labels 58% of objects at 90% precision');
    expect(axisRaw('calibration', r, true)).toBe('ECE 0.06');
    expect(axisRaw('localization', r, true)).toBe('mean IoU 0.84');
    const none = { ...r, robustness: null, calibration: null, self_labeling: { recall: 0, precision: 0.84, threshold: null } } as unknown as BenchResult;
    expect(axisRaw('robustness', none, false)).toBeNull();
    expect(axisRaw('self_labeling', none, false)).toBe('84% of labels right, under 90%');
    expect(axisRaw('self_labeling', none, true)).toBe('never 90% of labels right');
  });

  it('says why an axis was not measured', () => {
    const plan = catalog().default_plan;
    const agent = { conditions: false, scored: false, contestant: { kind: 'agent' } } as BenchEntry;
    expect(axisWhyNot('robustness', agent, { plan })).toMatch(/unchanged images only: for an agent/);
    expect(axisWhyNot('robustness', agent, { plan: { conditions: [] } })).toMatch(/no conditions/);
    expect(axisWhyNot('calibration', agent, { plan })).toMatch(/^Agents give no confidences/);
    expect(axisWhyNot('calibration', { ...agent, contestant: { kind: 'external' } }, { plan })).toBe('Its predictions came without confidences.');
    expect(axisWhyNot('accuracy', agent, { plan })).toBeNull();
  });
});

const bin = (lo: number, count: number, confidence: number | null, precision: number | null): BenchCalibrationBin => ({ lo, hi: lo + 0.1, count, confidence, precision });

describe('calibrationReading', () => {
  it('reads an overconfident model at its bin furthest off the diagonal, among well-filled bins', () => {
    const bins = [bin(0, 0, null, null), bin(0.5, 400, 0.55, 0.5), bin(0.9, 300, 0.94, 0.78), bin(0.1, 3, 0.15, 0.9)];
    const r = calibrationReading({ ece: 0.09, bins });
    expect(r.tone).toBe('over');
    // The 3-detection bin is further off (0.75) but holds too few to say anything.
    expect(r.sentence).toBe('Overconfident: when it says 0.94, it is right 78% of the time. Trust its high scores less than they claim.');
  });

  it('reads an underconfident model, and one close to the diagonal', () => {
    expect(calibrationReading({ ece: 0.08, bins: [bin(0.3, 500, 0.35, 0.52), bin(0.6, 200, 0.65, 0.7)] })).toEqual({
      tone: 'under',
      sentence: 'Underconfident: when it says 0.35, it is right 52% of the time. Its scores undersell it.',
    });
    const close = calibrationReading({ ece: 0.02, bins: [bin(0.5, 300, 0.55, 0.54), bin(0.9, 300, 0.95, 0.93)] });
    expect(close.tone).toBe('close');
    expect(close.sentence).toBe('Close to the diagonal: when it says 0.95, it is right 93% of the time. Its confidence can be taken at its word.');
  });
});

const pt = (confidence: number, precision: number | null, recall: number, labels: number): BenchLabelPoint => ({ confidence, precision, recall, labels });

describe('selfLabelReading', () => {
  it('reads the threshold it found', () => {
    expect(selfLabelReading({ threshold: 0.62, precision: 0.912, recall: 0.58, mean_iou: 0.84, labels_per_image: 3.14, curve: [] }, true)).toBe(
      'From confidence 0.62 up, 91% of its labels are right and they cover 58% of all objects: 3.1 labels per image, mean IoU 0.84.',
    );
  });

  it('says when no confidence reaches 90%, with the most it gets and all its detections', () => {
    const curve = [pt(0, 0.62, 0.8, 900), pt(0.5, 0.86, 0.4, 300), pt(0.9, null, 0, 0)];
    expect(selfLabelReading({ threshold: null, precision: 0.62, recall: 0, mean_iou: 0.8, labels_per_image: 2.5, curve }, true)).toBe(
      'At no confidence are 90% of its labels right: the most is 86%, from 0.50 up. All its detections together are 62% right, 2.5 labels per image.',
    );
  });

  it('reads a contestant without confidences at its one point', () => {
    expect(selfLabelReading({ threshold: null, precision: 0.93, recall: 0.61, mean_iou: 0.81, labels_per_image: 4, curve: [pt(0, 0.93, 0.61, 200)] }, false)).toBe(
      'It gives no confidences, so it has one operating point: 93% of its labels are right and they cover 61% of all objects (4.0 labels per image, mean IoU 0.81).',
    );
    expect(selfLabelReading({ threshold: null, precision: 0.84, recall: 0, mean_iou: null, labels_per_image: 4, curve: [pt(0, 0.84, 0.66, 200)] }, false)).toBe(
      'It gives no confidences, so it has one operating point: 84% of its labels are right, short of the 90% a label set to train on needs.',
    );
  });
});

describe('robustnessReading', () => {
  it('names the hardest condition at its worst severity, and the easiest on average', () => {
    const names: Record<string, string> = { distance: 'Distance', color_cast: 'Colour cast', haze: 'Haze' };
    const name = (id: string) => names[id] ?? id;
    const c = [
      { condition: 'haze', retention: 0.8, points: [{ severity: 1, retention: 0.95 }, { severity: 5, retention: 0.62 }] },
      { condition: 'distance', retention: 0.6, points: [{ severity: 1, retention: 0.9 }, { severity: 5, retention: 0.41 }] },
      { condition: 'color_cast', retention: 0.97, points: [{ severity: 3, retention: 0.97 }] },
    ];
    expect(robustnessReading(c, name)).toBe('Hardest: Distance, 41% kept at severity 5. Easiest: Colour cast, 97% kept on average.');
    expect(robustnessReading([], name)).toBe('');
  });
});

describe('comparing entries', () => {
  it('says whether intervals tell entries apart', () => {
    const a = { name: 'a', map: 0.64, ci: [0.62, 0.66] as [number, number] };
    const b = { name: 'b', map: 0.63, ci: [0.61, 0.65] as [number, number] };
    const c = { name: 'c', map: 0.4, ci: [0.35, 0.45] as [number, number] };
    expect(intervalsVerdict([a, b])).toBe('Not clearly different on this benchmark: their 95% intervals overlap.');
    expect(intervalsVerdict([c, a])).toBe('a is clearly ahead on mAP50-95: its interval is above the other’s.');
    expect(intervalsVerdict([a, b, c])).toBe('a and b are not clearly different (their intervals overlap); c is clearly behind.');
    expect(intervalsVerdict([a, b, { ...a, name: 'd' }])).toBe('Not clearly different on this benchmark: every interval overlaps a’s.');
    expect(intervalsVerdict([a])).toBe('');
  });

  it('calls out what makes entries not like for like', () => {
    const b = { images: 360, classes: ['a', 'b', 'c', 'd'], plan: catalog().default_plan };
    const agent = entry({ id: 'x', name: 'Claude', images: 48, conditions: false, scored: false, conditions_run: 0 });
    const model = entry({ id: 'y', name: 'mini', conditions_run: 12, missing_classes: ['c', 'd'] });
    const ext = entry({ id: 'z', name: 'ext', conditions_run: 4 });
    expect(compareCaveats([agent, model, ext], b).map((c) => c.text)).toEqual([
      'Claude ran 48 of 360 images: its interval is wider and its score partial.',
      'Claude ran the unchanged images only: it has no robustness to compare.',
      'Claude gives no confidences: its AP comes from one precision–recall point. Compare it on precision, recall and self-labeling rather than mAP.',
      'mini knows 2 of 4 classes: its AP on the others is 0.',
      'ext ran 4 of 12 conditions: its robustness is over those.',
    ]);
    expect(compareCaveats([entry({}), entry({ id: 'f' })], b)).toEqual([]);
  });
});

describe('loops in words', () => {
  it('states the question, the verdict against the first interval and what a new loop costs', () => {
    expect(loopQuestion('fasteners-n-best', 'the fasteners training split', 'folo-s', 3)).toBe(
      'Can fasteners-n-best label the fasteners training split well enough, by itself, to train a better folo-s, generation after generation?',
    );
    const gens = [
      { generation: 0, map50_95: 0.794, ci95: [0.752, 0.831] },
      { generation: 3, map50_95: 0.857, ci95: [0.824, 0.89] },
    ] as BenchLoop['generations'];
    expect(loopVerdictText({ verdict: 'improved', generations: gens })).toBe(
      'Improved: generation 3 (0.857) is above the first teacher’s 95% interval (0.752–0.831). Its own labels made a better model.',
    );
    expect(loopVerdictText({ verdict: 'held', generations: [gens[0]!, { ...gens[1]!, generation: 1, map50_95: 0.8 }] })).toBe(
      'Held: generation 1 (0.800) is inside the first teacher’s 95% interval (0.752–0.831): no measurable change.',
    );
    expect(loopVerdictText({ verdict: null, generations: gens })).toBeNull();
    expect(loopPlanText({ generations: 3, epochs: 30, poolImages: 4200, teacherEntry: false })).toBe(
      '3 generations: 3 training runs of 30 epochs on up to 4,200 images, 4 benchmark entries',
    );
    expect(loopPlanText({ generations: 1, epochs: 1, poolImages: null, teacherEntry: true })).toBe('1 generation: 1 training run of 1 epoch, 1 benchmark entry');
  });

  it('lets a model teach only with the benchmark’s classes, same names in the same order', () => {
    const b = { classes: ['bolt', 'nut', 'washer'] };
    expect(teacherEligibility({ classes: ['Bolt', 'nut', 'washer'] }, b)).toEqual({ ok: true });
    expect(teacherEligibility({ classes: ['nut', 'bolt', 'washer'] }, b)).toEqual({ ok: false, why: 'Same classes in another order: labels are written by class index' });
    expect(teacherEligibility({ classes: ['bolt', 'nut'] }, b)).toEqual({ ok: false, why: 'Knows 2 of these 3 classes' });
    expect(teacherEligibility({ classes: ['bolt', 'nut', 'washer', 'pin'] }, b)).toEqual({ ok: false, why: 'Knows 4 classes, more than these 3' });
    expect(teacherEligibility({ classes: ['cat'] }, b)).toEqual({ ok: false, why: 'None of these classes (it knows 1 class)' });
  });
});

describe('roundToTotal', () => {
  it('rounds the rows so they add up to the total as written', async () => {
    const { roundToTotal } = await import('./words');
    const r = roundToTotal([22.44, 20.74, 11.64, 9.44, 7.64], 71.9);
    expect(r.reduce<number>((s, v) => s + (v ?? 0), 0)).toBeCloseTo(71.9, 6);
    expect(roundToTotal([10.05, null, 9.95], 20)).toEqual([10.1, null, 9.9]);
  });
});
