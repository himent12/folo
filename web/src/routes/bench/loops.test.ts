import { describe, expect, it } from 'vitest';
import type { BenchGeneration, BenchLabelPoint, BenchLoop } from '$lib/api/types';
import benchDoc from '../../../../docs/BENCH.md?raw';
import {
  forecastText,
  intervalReading,
  labelForecast,
  labelsLine,
  loopErrorField,
  loopStatus,
  loopSteps,
  lowerConfidence,
  mapWithInterval,
  plusMinus,
  SPEED_HELP,
} from './words';

// A loop's page and list, read off what the loop says (docs/API.md, Self-improvement loops), and New loop's forecast
// of a teacher's labels (docs/BENCH.md: a teacher that keeps boxes on fewer than 10 pool images ends the loop).

const gen = (generation: number, o: Partial<BenchGeneration> = {}): BenchGeneration => ({
  generation,
  labels: null,
  run_id: null,
  model_id: null,
  model_name: null,
  entry_id: null,
  score: null,
  map50_95: null,
  ci95: null,
  ...o,
});
const LABELS = { threshold: 0.32, images: 1920, labeled_images: 1810, boxes: 8306, excluded: 0, dataset_id: 'ds_1' };
const GRADED = { score: 85.8, map50_95: 0.857, ci95: [0.848, 0.866] as [number, number] };
const teacher = gen(0, { model_id: 'm_t', model_name: 'shapes-teacher-best', entry_id: 'be_0' });
const loop = (phase: BenchLoop['phase'], generations: BenchGeneration[], status: BenchLoop['status'] = 'running') => ({ status, phase, generations });
const states = (l: ReturnType<typeof loop>) => loopSteps(l)?.steps.map((s) => `${s.name}: ${s.state}`);

describe("a running loop's steps", () => {
  it('walks the real server’s sequence for 2 generations, nothing done before it has happened', () => {
    // Generation 0: one step, the benchmark grading the first teacher; nothing is labeled or trained for it.
    const g0 = loop('benchmarking', [gen(0, { model_id: 'm_t', model_name: 'shapes-teacher-best' })]);
    expect(states(g0)).toEqual(['Benchmark grades the teacher: now']);
    expect(loopStatus(g0).label).toBe('Benchmarking the teacher');
    expect(states(loop('benchmarking', [teacher]))).toEqual(['Benchmark grades the teacher: now']);
    expect(loopSteps(loop('benchmarking', [teacher]))!.generation).toBe(0);

    // The teacher graded, generation 1 pushed: its labeling is now, even in the moment the phase still says
    // benchmarking (the server pushes the generation, then sets the phase).
    const t = { ...teacher, ...GRADED };
    for (const phase of ['benchmarking', 'labeling'] as const) {
      const l = loop(phase, [t, gen(1)]);
      expect(states(l)).toEqual(['Teacher labels the pool: now', 'Student trains: next', 'Benchmark grades the student: next']);
      expect(loopStatus(l).label).toBe('Labeling for gen 1');
    }
    // Labels in: training, before and after its run exists.
    for (const g of [gen(1, { labels: LABELS }), gen(1, { labels: LABELS, run_id: 'run_1' })]) {
      const l = loop('training', [t, g]);
      expect(states(l)).toEqual(['Teacher labels the pool: done', 'Student trains: now', 'Benchmark grades the student: next']);
      expect(loopStatus(l).label).toBe('Training gen 1');
    }
    // Its model in: the benchmark grades the student, before and after its entry starts.
    const trained = { labels: LABELS, run_id: 'run_1', model_id: 'm_1', model_name: 'shapes self-training gen 1-best' };
    for (const g of [gen(1, trained), gen(1, { ...trained, entry_id: 'be_1' })]) {
      const l = loop('benchmarking', [t, g]);
      expect(states(l)).toEqual(['Teacher labels the pool: done', 'Student trains: done', 'Benchmark grades the student: now']);
      expect(loopStatus(l).label).toBe('Benchmarking gen 1');
    }
    // Generation 2, the same three steps.
    const g1 = gen(1, { ...trained, entry_id: 'be_1', ...GRADED });
    expect(loopStatus(loop('labeling', [t, g1, gen(2)])).label).toBe('Labeling for gen 2');
    expect(loopStatus(loop('training', [t, g1, gen(2, { labels: LABELS })])).label).toBe('Training gen 2');
    expect(loopStatus(loop('benchmarking', [t, g1, gen(2, { labels: LABELS, model_id: 'm_2', model_name: 'x' })])).label).toBe('Benchmarking gen 2');
    // Done: no steps, the word of how it ended.
    expect(loopSteps(loop(null, [t, g1], 'done'))).toBeNull();
    expect(loopStatus(loop(null, [t, g1], 'done')).label).toBe('Done');
    expect(loopStatus(loop(null, [t, gen(1)], 'failed')).label).toBe('Failed');
    // A stopping loop still shows where it was.
    expect(states(loop('training', [t, gen(1, { labels: LABELS })], 'stopping'))).toEqual(['Teacher labels the pool: done', 'Student trains: now', 'Benchmark grades the student: next']);
    // Without its generations (a list that has only the status), the phase.
    expect(loopStatus({ status: 'running', phase: 'training' }).label).toBe('Training');
  });
});

describe("a generation's labels", () => {
  it('say what the teacher kept in one line, and when it is thin', () => {
    expect(labelsLine({ threshold: 0.31, images: 1920, labeled_images: 48, boxes: 243, excluded: 12 })).toEqual({
      text: 'kept 243 boxes on 48 of 1,920 pool images from confidence 0.31; 12 left out as benchmark images',
      thin: true,
    });
    expect(labelsLine(LABELS)).toEqual({ text: 'kept 8,306 boxes on 1,810 of 1,920 pool images from confidence 0.32', thin: false });
    // The real starved generation of an older server: 14 boxes on 13 images.
    expect(labelsLine({ threshold: 0.78, images: 1920, labeled_images: 13, boxes: 14, excluded: 1 })).toEqual({
      text: 'kept 14 boxes on 13 of 1,920 pool images from confidence 0.78; 1 left out as benchmark image',
      thin: true,
    });
    expect(labelsLine({ threshold: 0.5, images: 100, labeled_images: 10, boxes: 1, excluded: 0 }).text).toBe('kept 1 box on 10 of 100 pool images from confidence 0.50');
  });
});

/** A teacher's self-labeling curve: recall falls as `(1 - t)^b`, precision rises to 0.9 at `at` and on past it. */
function curve(rmax: number, b: number, at: number, truths = 10000): BenchLabelPoint[] {
  return Array.from({ length: 101 }, (_, i) => {
    const t = i / 100;
    const recall = rmax * (1 - t) ** b;
    const precision = t >= at ? 0.92 + 0.07 * (t - at) : 0.3 + 0.6 * (t / at);
    const labels = Math.round((truths * recall) / precision);
    return { confidence: t, precision: labels ? precision : null, recall: labels ? recall : 0, labels };
  });
}

describe("New loop's forecast of the teacher's labels", () => {
  // The real weak teacher: 90% right only from 0.78, where it finds a sliver of the objects.
  const weak = curve(0.6, 5, 0.78);
  const weakT = weak.find((p) => p.precision !== null && p.precision >= 0.9)!;
  const weakSelf = { threshold: weakT.confidence, readiness: 'not_ready' as const, curve: weak };
  // A good one: 29% of objects at 0.55.
  const good = curve(0.75, 1.6, 0.55);
  const goodT = good.find((p) => p.precision !== null && p.precision >= 0.9)!;
  const goodSelf = { threshold: goodT.confidence, readiness: 'partial' as const, curve: good };

  it('reads a good teacher at its own threshold: nothing to warn of', () => {
    const f = labelForecast(goodSelf, 300, 1920, null);
    expect(f).toMatchObject({ threshold: goodT.confidence, from: 'entry', starved: false, stops: false });
    expect(f.recall).toBeCloseTo(goodT.recall, 9);
    expect(f.images).toBeGreaterThan(1000);
    const w = forecastText(f, 1920);
    expect(w.title).toBeNull();
    expect(w.text).toMatch(new RegExp(`^At its own threshold, ${goodT.confidence.toFixed(2)}, 9\\d% of its labels are right and they cover \\d+% of all objects \\(partly ready\\): about [\\d,]+ of the pool’s 1,920 images would get a box\\.$`));
  });

  it('warns plainly of a not-ready teacher, and of one under a tenth of the objects', () => {
    const f = labelForecast(weakSelf, 300, 1920, null);
    expect(f.starved).toBe(true);
    expect(f.recall).toBeLessThan(0.01);
    const w = forecastText(f, 1920);
    expect(w.title).toMatch(/^Too few labels/);
    expect(w.text).toMatch(/^At its own threshold, 0\.78, 9\d% of its labels are right but they cover 0\.\d% of all objects \(not ready\)/);
    expect(w.remedy).toBe('A lower confidence keeps more of its detections, fewer of them right.');
    // Under 10% of the objects is too few, whatever the verdict says; 10% is not.
    const at = (recall: number) => labelForecast({ threshold: 0.5, readiness: 'partial', curve: [{ confidence: 0.5, precision: 0.92, recall, labels: Math.round(recall * 3000) }] }, 300, 1920, null);
    expect(at(0.09).starved).toBe(true);
    expect(at(0.1).starved).toBe(false);
  });

  it('says when the loop would stop at labeling: fewer than 10 pool images with a box', () => {
    // Its labels on the benchmark: 2 on 300 images; on a pool of 800 that is about 5 images.
    const f = labelForecast({ threshold: 0.78, readiness: 'not_ready', curve: [{ confidence: 0.78, precision: 1, recall: 0.002, labels: 2 }] }, 300, 800, null);
    expect(f.images).toBe(5);
    expect(f.stops).toBe(true);
    expect(forecastText(f, 800).title).toBe('Too few labels: the loop would stop at labeling');
    expect(forecastText(f, 800).text).toMatch(/A loop stops when fewer than 10 pool images get a box\.$/);
    // Without a pool size, it cannot count images: starved, but not said to stop.
    expect(labelForecast({ threshold: 0.78, readiness: 'not_ready', curve: [{ confidence: 0.78, precision: 1, recall: 0.002, labels: 2 }] }, 300, null, null).stops).toBe(false);
  });

  it('reads the cost of a lower confidence off the curve, live, and takes the default when the entry found none', () => {
    const low = labelForecast(weakSelf, 300, 1920, 0.3);
    expect(low.from).toBe('chosen');
    expect(low.precision).toBeCloseTo(weak[30]!.precision!, 9);
    expect(low.recall).toBeCloseTo(weak[30]!.recall, 9);
    expect(low.starved).toBe(false);
    expect(forecastText(low, 1920).text).toMatch(/^From 0\.30, \d+% of its labels are right and they cover \d+% of all objects: about [\d,]+ of the pool’s 1,920 images would get a box\. That is under the 90% right a label set to train on needs: the student learns from its wrong labels too\.$/);
    // Lower still, more found and fewer right.
    const lower = labelForecast(weakSelf, 300, 1920, 0.1);
    expect(lower.recall).toBeGreaterThan(low.recall);
    expect(lower.precision!).toBeLessThan(low.precision!);
    // Its entry found no threshold and none is set: a loop never guesses one, it stops before labeling, and that is said.
    const never = labelForecast({ threshold: null, readiness: 'not_ready', curve: weak.map((p) => ({ ...p, precision: p.precision === null ? null : Math.min(0.8, p.precision) })) }, 300, 1920, null);
    expect(never).toMatchObject({ from: 'none', threshold: null, starved: true, stops: true });
    const w = forecastText(never, 1920);
    expect(w.title).toBe('Its entry found no threshold: choose a confidence');
    expect(w.text).toMatch(/a loop never guesses one: without a confidence set here it stops before labeling\.$/);
    expect(JSON.stringify(w)).not.toMatch(/0\.5/);
    // With one chosen, read off the curve as any other.
    expect(labelForecast({ threshold: null, readiness: 'not_ready', curve: weak }, 300, 1920, 0.3)).toMatchObject({ from: 'chosen', threshold: 0.3 });
    // A confidence where it keeps nothing.
    const none = labelForecast({ threshold: 0.9, readiness: 'not_ready', curve: [{ confidence: 0.99, precision: null, recall: 0, labels: 0 }] }, 300, 1920, 0.99);
    expect(forecastText(none, 1920).text).toMatch(/^From 0\.99, it keeps no detection there: about 0 of the pool’s 1,920 images would get a box\./);
  });

  it('proposes its best-F1 confidence as the lower one, else a step of 0.2 down', () => {
    expect(lowerConfidence(0.78, 0.19)).toBe(0.19);
    expect(lowerConfidence(0.4, 0.55)).toBe(0.2);
    expect(lowerConfidence(0.1, null)).toBe(0.05);
  });
});

describe('a loop the server refuses to start', () => {
  it('puts its error at the field it names, else on the form', () => {
    expect(loopErrorField({ code: 'invalid_config', message: 'imgsz must be at least 64 for training, got 32' }, false)).toBe('imgsz');
    expect(loopErrorField({ code: 'invalid_config', message: 'epochs must be at least 1' }, false)).toBe('epochs');
    expect(loopErrorField({ code: 'model_not_found', message: 'No model with id m_x (student.weights).' }, true)).toBe('weights');
    expect(loopErrorField({ code: 'model_not_found', message: 'No model with id m_x.' }, false)).toBeNull();
    expect(loopErrorField({ code: 'device_not_found', message: 'No device gpu:3 on this machine.' }, false)).toBeNull();
    expect(loopErrorField({ code: 'invalid_config', message: 'batch must be at least 1' }, false)).toBeNull();
  });
});

describe('an interval in words', () => {
  it('prints "±" only when the interval is symmetric as printed, else the interval', () => {
    expect(plusMinus(0.641, [0.623, 0.659])).toBe('± 0.018');
    expect(mapWithInterval(0.641, [0.623, 0.659])).toBe('mAP50-95 0.641 ± 0.018');
    // The real model entry: 0.354 in 0.337–0.402 is not 0.354 ± 0.033.
    expect(plusMinus(0.354, [0.337, 0.402])).toBeNull();
    expect(mapWithInterval(0.354, [0.337, 0.402])).toBe('mAP50-95 0.354 (0.337–0.402)');
    expect(mapWithInterval(0.3531, [0.2644, 0.4449])).toBe('mAP50-95 0.353 (0.264–0.445)');
    // Symmetric to the printed precision, whatever the fourth decimal.
    expect(plusMinus(0.6414, [0.6233, 0.6594])).toBe('± 0.018');
    expect(mapWithInterval(0.5, null)).toBe('mAP50-95 0.500');
  });

  it("reads the interval in docs/BENCH.md's words", () => {
    const s = intervalReading([0.337, 0.402], 300);
    expect(s).toBe('Resampling these 300 images puts it between 0.337 and 0.402 (95%). Entries whose intervals overlap are not clearly different on this benchmark; more images make the interval narrower.');
    expect(benchDoc).toMatch(/Two entries whose intervals overlap are not clearly different on this benchmark; more\s+images make the interval narrower\./);
    expect(benchDoc).toMatch(/bootstrap over images/);
  });

  it('calls a model’s speed a throughput, never a latency', () => {
    expect(SPEED_HELP).toMatch(/throughput/);
    expect(SPEED_HELP).toMatch(/first batch/);
    expect(SPEED_HELP).not.toMatch(/latency/i);
  });
});
