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
  loopVerdictText,
  robustnessReading,
  roundToTotal,
  selfLabelReading,
  sharesNote,
  teacherEligibility,
} from './words';

// The sentences the result screens generate to read their charts (./words, "reading a result" onwards): what each
// picks to say and the numbers it prints, not its wording.

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

  it('rounds the rows so they add up to the total as written', () => {
    const r = roundToTotal([22.44, 20.74, 11.64, 9.44, 7.64], 71.9);
    expect(r.reduce<number>((s, v) => s + (v ?? 0), 0)).toBeCloseTo(71.9, 6);
    expect(roundToTotal([10.05, null, 9.95], 20)).toEqual([10.1, null, 9.9]);
  });

  it('says how the others share the score when axes are left out', () => {
    expect(sharesNote(axes([0.6, 0.8, 0.5, 0.9, 0.6]))).toBeNull();
    expect(sharesNote(axes([0.6, 0.8, 0.5, null, 0.6]))).toMatch(/^Calibration was not measured: .* over 90%/);
    expect(sharesNote(axes([0.6, null, 0.5, null, 0.6]))).toMatch(/^Robustness and Calibration were not measured: .* over 65%/);
  });

  it('names the number behind each axis in its own unit', () => {
    const r = {
      accuracy: { map50_95: 0.641, ci95: [0.623, 0.659] },
      robustness: { retention: 0.832, conditions: [] },
      calibration: { ece: 0.061, detections: 10, bins: [] },
      localization: { mean_iou: 0.84, matches: 10, histogram: [] },
      self_labeling: { recall: 0.58, precision: 0.91, threshold: 0.62, labels: 1363 },
    } as unknown as BenchResult;
    expect(axisRaw('accuracy', r, true)).toContain('0.641 ± 0.018');
    expect(axisRaw('robustness', r, true)).toContain('83%');
    expect(axisRaw('self_labeling', r, true)).toMatch(/58%.*1,363 labels/);
    expect(axisRaw('calibration', r, true)).toContain('0.06');
    expect(axisRaw('localization', r, true)).toContain('0.84');
    const none = { ...r, robustness: null, calibration: null, self_labeling: { recall: 0, precision: 0.84, threshold: null, labels: 105 } } as unknown as BenchResult;
    expect(axisRaw('robustness', none, false)).toBeNull();
    expect(axisRaw('self_labeling', none, false)).toMatch(/^84% .*under 90%/);
    // As printed: 89.6% is not "90% … under 90%"; a handful of labels is too few to judge.
    expect(axisRaw('self_labeling', { ...none, self_labeling: { ...none.self_labeling, precision: 0.896 } }, false)).toMatch(/^89\.6% /);
    expect(axisRaw('self_labeling', { ...none, self_labeling: { ...none.self_labeling, precision: 1, labels: 6 } }, false)).toMatch(/too few/);
  });

  it('says why an axis was not measured, and only for one that was not', () => {
    const plan = catalog().default_plan;
    const agent = { conditions: false, scored: false, contestant: { kind: 'agent' } } as BenchEntry;
    const unchangedOnly = axisWhyNot('robustness', agent, { plan });
    expect(unchangedOnly).toBeTruthy();
    // A plan without conditions is another reason than an agent's.
    expect(axisWhyNot('robustness', agent, { plan: { conditions: [] } })).not.toBe(unchangedOnly);
    const agentCal = axisWhyNot('calibration', agent, { plan });
    expect(agentCal).toBeTruthy();
    expect(axisWhyNot('calibration', { ...agent, contestant: { kind: 'external' } }, { plan })).not.toBe(agentCal);
    expect(axisWhyNot('accuracy', agent, { plan })).toBeNull();
  });
});

const bin = (lo: number, count: number, confidence: number | null, precision: number | null): BenchCalibrationBin => ({ lo, hi: lo + 0.1, count, confidence, precision });
/** The bin a calibration sentence quotes, as "said → right". */
const quoted = (sentence: string) => /when it says ([\d.]+), it is right (\d+)%/.exec(sentence)?.slice(1).join(' → ') ?? null;

describe('calibrationReading', () => {
  it('reads an overconfident model at its bin furthest off the diagonal, among well-filled bins', () => {
    const bins = [bin(0, 0, null, null), bin(0.5, 400, 0.55, 0.5), bin(0.9, 300, 0.94, 0.78), bin(0.1, 3, 0.15, 0.9)];
    const r = calibrationReading({ ece: 0.09, bins });
    expect(r.tone).toBe('over');
    // The 3-detection bin is further off (0.75) but holds too few to say anything.
    expect(quoted(r.sentence)).toBe('0.94 → 78');
  });

  it('reads an underconfident model, and one close to the diagonal', () => {
    const under = calibrationReading({ ece: 0.08, bins: [bin(0.3, 500, 0.35, 0.52), bin(0.6, 200, 0.65, 0.7)] });
    expect([under.tone, quoted(under.sentence)]).toEqual(['under', '0.35 → 52']);
    const close = calibrationReading({ ece: 0.02, bins: [bin(0.5, 300, 0.55, 0.54), bin(0.9, 300, 0.95, 0.93)] });
    expect([close.tone, quoted(close.sentence)]).toEqual(['close', '0.55 → 54']);
  });

  it('never quotes a bin that contradicts its tone, nor a thin one, nor a far one as "close"', () => {
    // Mostly underconfident (the low bins), a thin overconfident top that is further off: the old reading said
    // "Underconfident: when it says 0.95, it is right 40% of the time".
    const r = calibrationReading({ ece: 0.09, bins: [bin(0.1, 900, 0.15, 0.24), bin(0.2, 600, 0.25, 0.33), bin(0.9, 12, 0.95, 0.4)] });
    expect([r.tone, quoted(r.sentence)]).toEqual(['under', '0.15 → 24']);
    const only = calibrationReading({ ece: 0.05, bins: [bin(0.1, 1000, 0.15, 0.1), bin(0.9, 10, 0.95, 0.99)] });
    expect([only.tone, quoted(only.sentence)]).toEqual(['over', '0.15 → 10']);
    // No bin with enough detections in it: the average alone.
    const flat = calibrationReading({ ece: 0.05, bins: [bin(0.1, 10, 0.15, 0.1), bin(0.5, 10, 0.55, 0.5)] });
    expect(quoted(flat.sentence)).toBeNull();
    expect(flat.sentence).toContain('ECE 0.05');
    // Close on average, but a well-filled bin far off: the far bin is not quoted as close, and the reader is sent to
    // the bins rather than told to take it at its word.
    const far = calibrationReading({ ece: 0.025, bins: [bin(0.2, 2000, 0.25, 0.26), bin(0.9, 200, 0.95, 0.6)] });
    expect([far.tone, quoted(far.sentence)]).toEqual(['close', '0.25 → 26']);
    expect(far.sentence).not.toBe(calibrationReading({ ece: 0.025, bins: [bin(0.2, 2000, 0.25, 0.26)] }).sentence);
  });
});

const pt = (confidence: number, precision: number | null, recall: number, labels: number): BenchLabelPoint => ({ confidence, precision, recall, labels });

describe('selfLabelReading', () => {
  it('reads the threshold it found, with the labels it rests on', () => {
    const s = selfLabelReading({ threshold: 0.32, precision: 0.904, recall: 0.94, mean_iou: 0.84, labels_per_image: 4.54, labels: 1363, curve: [] }, true);
    expect(s).toMatch(/^From confidence 0\.32 up: 1,363 labels \(4\.5 per image\), 90% right, 94% of all objects/);
  });

  it('says when no confidence keeps 10 labels 90% right, with the best supported and a precise handful as a handful', () => {
    const curve = [pt(0, 0.62, 0.8, 900), pt(0.5, 0.86, 0.4, 300), pt(0.78, 1, 0.0008, 1), pt(0.9, null, 0, 0)];
    const s = selfLabelReading({ threshold: null, precision: 0.62, recall: 0, mean_iou: 0.8, labels_per_image: 2.5, labels: 900, curve }, true);
    // The best supported: 86% from 0.50 (300 labels); the handful: 1 label from 0.78; then all its detections.
    expect(s).toMatch(/86% right, from 0\.50 up \(300 labels\)/);
    expect(s).toMatch(/From 0\.78 up its 1 label is 100% right, too few/);
    expect(s).toMatch(/900 detections \(2\.5 per image\), 62% right\.$/);
  });

  it('reads the real weak teacher: all its detections, 1.6% right, never as labels it would keep', () => {
    const curve = [pt(0, 0.016, 0.52, 74593), pt(0.5, 0.31, 0.05, 900), pt(0.78, 1, 0.0008, 1)];
    const t = selfLabelReading({ threshold: null, precision: 0.016, recall: 0, mean_iou: 0.71, labels_per_image: 248.6, labels: 74593, curve }, true);
    expect(t).toMatch(/74,593 detections \(248\.6 per image\), 1\.6% right\.$/);
    expect(t).not.toMatch(/248\.6 labels/);
  });

  it('reads a contestant without confidences at its one point', () => {
    expect(selfLabelReading({ threshold: null, precision: 0.93, recall: 0.61, mean_iou: 0.81, labels_per_image: 4, labels: 200, curve: [pt(0, 0.93, 0.61, 200)] }, false)).toMatch(
      /200 labels \(4\.0 per image\) are 93% right and cover 61% of all objects/,
    );
    expect(selfLabelReading({ threshold: null, precision: 0.84, recall: 0, mean_iou: null, labels_per_image: 4, labels: 200, curve: [pt(0, 0.84, 0.66, 200)] }, false)).toMatch(/are 84% right, short of the 90%/);
    // A handful all right is not a label set, and never "0.0 per image" beside them.
    const handful = selfLabelReading({ threshold: null, precision: 1, recall: 0, mean_iou: 0.9, labels_per_image: 0.02, labels: 6, curve: [pt(0, 1, 0.01, 6)] }, false);
    expect(handful).toMatch(/6 labels \(under 0\.1 per image\) are 100% right: too few/);
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
    expect(robustnessReading(c, name)).toMatch(/^Hardest: Distance, 41% kept at severity 5\. Easiest: Colour cast, 97% /);
    expect(robustnessReading([], name)).toBe('');
  });
});

describe('comparing entries', () => {
  it('says whether intervals tell entries apart, each against the leader', () => {
    const a = { name: 'a', map: 0.64, ci: [0.62, 0.66] as [number, number] };
    const b = { name: 'b', map: 0.63, ci: [0.61, 0.65] as [number, number] };
    const c = { name: 'c', map: 0.4, ci: [0.35, 0.45] as [number, number] };
    const apart = (t: string) => /clearly (ahead|below)/.test(t) && !/not clearly/.test(t);
    expect(apart(intervalsVerdict([a, b]))).toBe(false);
    expect(intervalsVerdict([c, a])).toMatch(/^a is clearly ahead/);
    // Each against the leader: b and c are not compared with each other.
    expect(intervalsVerdict([a, b, c])).toMatch(/^Against a, .*b overlaps .*c is clearly below/);
    expect(intervalsVerdict([a, b, { ...a, name: 'd' }])).toMatch(/none is clearly different/);
    // As printed: 0.6201 and 0.6204 both print 0.620, so they overlap.
    expect(apart(intervalsVerdict([{ name: 'p', map: 0.7, ci: [0.6204, 0.72] }, { name: 'q', map: 0.5, ci: [0.45, 0.6201] }]))).toBe(false);
    expect(intervalsVerdict([a])).toBe('');
  });

  it('calls out what makes entries not like for like, one caveat for each', () => {
    const b = { images: 360, classes: ['a', 'b', 'c', 'd'], plan: catalog().default_plan };
    const agent = entry({ id: 'x', name: 'Claude', images: 48, conditions: false, scored: false, conditions_run: 0 });
    const model = entry({ id: 'y', name: 'mini', conditions_run: 12, missing_classes: ['c', 'd'] });
    const ext = entry({ id: 'z', name: 'ext', conditions_run: 4 });
    const caveats = compareCaveats([agent, model, ext], b).map((c) => c.text);
    // The agent's images, its unchanged images only and its missing confidences; the model's classes; the conditions
    // the external entry ran.
    expect(caveats).toHaveLength(5);
    expect(caveats[0]).toMatch(/^Claude ran 48 of 360 images/);
    expect(caveats[3]).toMatch(/^mini knows 2 of 4 classes/);
    expect(caveats[4]).toMatch(/^ext ran 4 of 12 conditions/);
    expect(compareCaveats([entry({}), entry({ id: 'f' })], b)).toEqual([]);
  });
});

describe('loops in words', () => {
  it('gives the verdict against the first interval, with the digits that make it true, and what a new loop costs', () => {
    const gens = [
      { generation: 0, map50_95: 0.794, ci95: [0.752, 0.831] },
      { generation: 3, map50_95: 0.857, ci95: [0.824, 0.89] },
    ] as BenchLoop['generations'];
    expect(loopVerdictText({ verdict: 'improved', generations: gens })).toMatch(/generation 3 \(0\.857\) is above .* \(0\.752–0\.831\)/);
    expect(loopVerdictText({ verdict: 'held', generations: [gens[0]!, { ...gens[1]!, generation: 1, map50_95: 0.8 }] })).toMatch(/generation 1 \(0\.800\) is inside/);
    expect(loopVerdictText({ verdict: null, generations: gens })).toBeNull();
    // The real loop: 0.8661 above 0.8660 prints with the digits that make it true.
    const real = [
      { generation: 0, map50_95: 0.857, ci95: [0.8476, 0.866] },
      { generation: 2, map50_95: 0.8661, ci95: [0.85, 0.88] },
    ] as BenchLoop['generations'];
    expect(loopVerdictText({ verdict: 'improved', generations: real })).toMatch(/\(0\.8661\) is above .* \(0\.8476–0\.8660\)/);
    expect(loopPlanText({ generations: 3, epochs: 30, poolImages: 4200, teacherEntry: false })).toMatch(/^3 generations: 3 training runs of 30 epochs on up to 4,200 images, 4 benchmark entries$/);
    expect(loopPlanText({ generations: 1, epochs: 1, poolImages: null, teacherEntry: true })).toMatch(/^1 generation: 1 training run of 1 epoch, 1 benchmark entry$/);
  });

  it('lets a model teach only with the benchmark’s classes, same names in the same order', () => {
    const b = { classes: ['bolt', 'nut', 'washer'] };
    expect(teacherEligibility({ classes: ['Bolt', 'nut', 'washer'] }, b)).toEqual({ ok: true });
    for (const classes of [['nut', 'bolt', 'washer'], ['bolt', 'nut'], ['bolt', 'nut', 'washer', 'pin'], ['cat']]) {
      expect(teacherEligibility({ classes }, b), classes.join()).toMatchObject({ ok: false, why: expect.any(String) });
    }
    expect(teacherEligibility({ classes: ['bolt', 'nut'] }, b)).toMatchObject({ why: expect.stringMatching(/2 of these 3/) });
  });
});
