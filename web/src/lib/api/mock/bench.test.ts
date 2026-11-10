import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { BenchEntry, BenchEventMap, BenchResult } from '../types';
import { AXES, AXIS_WEIGHT, rankEntries } from '../../bench/rules';
import { conditionRetention, expectedCalibrationError, histogramMeanIou, labelOperatingPoint, localizationAxis, readiness, retention, robustnessAxis, scoreTotal } from '../../bench/score';
import { conditionBoxes, distanceScale } from './benchConditions';
import { designPlan } from './benchDesign';
import { MockApi } from './index';
import benchDoc from '../../../../../docs/BENCH.md?raw';

const NOW = Date.parse('2026-11-09T12:00:00Z');

beforeEach(() => {
  vi.useFakeTimers({ now: NOW });
});
afterEach(() => vi.useRealTimers());

/** Holds a result to docs/BENCH.md's definitions, number by number. */
function expectConsistent(r: BenchResult, benchImages: number, planConditions: number) {
  const a = r.accuracy;
  // Accuracy: the class mean, inside its interval; mAP50 above it, mAP75 between.
  const mean = a.per_class.reduce((s, c) => s + c.map50_95, 0) / a.per_class.length;
  expect(a.map50_95).toBeCloseTo(mean, 3);
  expect(a.ci95[0]).toBeLessThanOrEqual(a.map50_95);
  expect(a.ci95[1]).toBeGreaterThanOrEqual(a.map50_95);
  expect(a.map50).toBeGreaterThanOrEqual(a.map50_95);
  expect(a.map75).toBeLessThanOrEqual(a.map50);
  // Robustness: retention = min(1, map / clean), a condition the mean of its severities, the axis their weighted mean.
  if (r.robustness) {
    for (const c of r.robustness.conditions) {
      for (const p of c.points) expect(p.retention).toBeCloseTo(retention(p.map50_95, a.map50_95), 3);
      expect(c.retention).toBeCloseTo(conditionRetention(c.points), 3);
      expect(c.points.map((p) => p.severity)).toEqual([...c.points.map((p) => p.severity)].sort((x, y) => x - y));
    }
    expect(r.robustness.retention).toBeCloseTo(robustnessAxis(r.robustness.conditions)!, 3);
  }
  // Calibration: the ECE is what its bins say; the first bin is empty (detections enter from 0.1).
  if (r.calibration) {
    expect(r.calibration.bins).toHaveLength(10);
    expect(r.calibration.bins[0]!.count).toBe(0);
    expect(r.calibration.ece).toBeCloseTo(expectedCalibrationError(r.calibration.bins), 3);
    expect(r.calibration.detections).toBe(r.calibration.bins.reduce((s, b) => s + b.count, 0));
  }
  // Localization: the mean of the histogram, its count the matches.
  expect(r.localization.histogram).toHaveLength(10);
  expect(r.localization.histogram.reduce((s, n) => s + n, 0)).toBe(r.localization.matches);
  if (r.localization.mean_iou !== null) expect(r.localization.mean_iou).toBeCloseTo(histogramMeanIou(r.localization.histogram)!, 3);
  // Self-labeling: the threshold is where the curve first reaches the target; the axis is the recall there.
  const op = labelOperatingPoint(r.self_labeling.curve);
  expect(r.self_labeling.threshold).toBe(op.threshold);
  expect(r.self_labeling.recall).toBe(op.point?.recall ?? 0);
  expect(r.self_labeling.readiness).toBe(readiness(r.self_labeling.recall));
  if (r.entry.scored) expect(r.self_labeling.curve).toHaveLength(101);
  else expect(r.self_labeling.curve).toHaveLength(1);
  // The score: five axes in order with the catalog's weights; the total their weighted mean over the measured ones.
  const s = r.entry.score!;
  expect(s.axes.map((x) => x.axis)).toEqual([...AXES]);
  for (const x of s.axes) expect(x.weight).toBe(AXIS_WEIGHT[x.axis]);
  expect(s.axes[0]!.value).toBe(a.map50_95);
  expect(s.axes[1]!.value).toBe(r.robustness ? r.robustness.retention : null);
  expect(s.axes[2]!.value).toBeCloseTo(r.self_labeling.recall, 4);
  expect(s.axes[3]!.value).toBe(r.calibration ? Math.round((1 - r.calibration.ece) * 1e4) / 1e4 : null);
  expect(s.axes[4]!.value).toBeCloseTo(localizationAxis(r.localization.mean_iou), 3);
  expect(s.total).toBeCloseTo(scoreTotal(s.axes), 1);
  const partial = s.axes.some((x) => x.value === null && (x.axis !== 'robustness' || planConditions > 0)) || r.entry.images < benchImages || (!!r.robustness && r.robustness.conditions.length < planConditions);
  expect(s.partial).toBe(partial);
  expect(r.entry.map50_95).toBe(a.map50_95);
}

describe('MockApi vision benchmarks: catalog and seeds', () => {
  it('has the twelve conditions in six groups, the default plan and the five axes', async () => {
    const c = await new MockApi('normal', 0).benchCatalog();
    expect(c.conditions.map((x) => x.id)).toEqual(['low_light', 'overexposure', 'low_contrast', 'defocus', 'motion_blur', 'sensor_noise', 'haze', 'color_cast', 'jpeg', 'low_resolution', 'distance', 'occlusion']);
    expect(new Set(c.conditions.map((x) => x.group))).toEqual(new Set(['light', 'blur', 'noise', 'medium', 'digital', 'geometry']));
    expect(c.severities).toBe(5);
    expect(c.default_plan.conditions.every((x) => x.severities.join() === '1,3,5' && x.weight === 1)).toBe(true);
    expect(c.axes.map((a) => [a.axis, a.weight])).toEqual([
      ['accuracy', 0.35],
      ['robustness', 0.25],
      ['self_labeling', 0.2],
      ['calibration', 0.1],
      ['localization', 0.1],
    ]);
  });

  it('seeds four benchmarks, newest first, that tell the story', async () => {
    const api = new MockApi('normal', 0);
    const list = await api.benches();
    expect(list.map((b) => b.name)).toEqual(['Sorting tray, night shift', 'Fasteners val', 'Pond fish', 'Shapes val']);
    const [fresh, fast, pond, shapes] = list;
    expect(fresh!.entries).toBe(0);
    expect(fresh!.best).toBeNull();
    expect(fresh!.images).toBe(200);
    expect(fast!.images).toBe(360);
    expect(fast!.source).toEqual({ kind: 'dataset', dataset_id: 'ds_fasteners', split: 'val' });
    expect(pond!.source).toEqual({ kind: 'label_job', job_id: 'lb_8d02' });
    expect(pond!.plan.designed_by).toEqual({ agent: 'claude', model: 'sonnet' });
    expect(pond!.plan.rationale).toBeTruthy();
    expect(pond!.plan.conditions.every((c) => c.why)).toBe(true);
    // The pond's designed plan weights the water's conditions most.
    const w = Object.fromEntries(pond!.plan.conditions.map((c) => [c.condition, c.weight]));
    expect(w.haze).toBe(3);
    expect(w.color_cast).toBe(3);
    expect(w.jpeg).toBeUndefined();
    for (const b of list) {
      expect(b.boxes).toBe(b.class_boxes.reduce((a, n) => a + n, 0));
      expect(b.fingerprint).toMatch(/^[0-9a-f]{16}$/);
    }
    expect(shapes!.best?.name).toBeTruthy();
  });

  it('has an entry of every kind and state on Fasteners val, ranked as the server ranks them', async () => {
    const api = new MockApi('normal', 0);
    const entries = await api.benchEntries('bn_3c90');
    expect(rankEntries(entries).map((e) => e.id)).toEqual(entries.map((e) => e.id));
    const by = (n: string) => entries.find((e) => e.name === n)!;
    const done = entries.filter((e) => e.status === 'done');
    expect(done.map((e) => e.score!.total)).toEqual([...done.map((e) => e.score!.total)].sort((a, b) => b - a));
    expect(done[0]!.name).toBe('fasteners-s-v1');
    expect(by('yolo26n').contestant.kind).toBe('external');
    const claude = by('Claude Code (sonnet)');
    expect(claude).toMatchObject({ images: 48, conditions: false, scored: false, status: 'done' });
    expect(claude.score!.partial).toBe(true);
    expect(claude.usage!.calls).toBe(12);
    expect(claude.usage!.cost_usd).toBeGreaterThan(0);
    expect(by('Codex (gpt-6.1-sol)').usage!.cost_usd).toBeNull();
    expect(by('fasteners-mini-n').missing_classes).toEqual(['washer', 'screw', 'spring', 'rivet', 'pin', 'bearing']);
    const failed = by('Claude Code (opus)');
    expect(failed.status).toBe('failed');
    expect(failed.error).toMatch(/not logged in/i);
    expect(failed.hint).toBeTruthy();
    const running = entries.find((e) => e.status === 'running')!;
    expect(running.progress.done).toBeGreaterThan(0);
    expect(running.progress.done).toBeLessThan(running.progress.total);
    expect(running.progress.total).toBe(360 * 37);
    expect(running.progress.condition).toBe('haze');
    expect(running.score).toBeNull();
    // Model entries report their speed once done, agents what their calls used; neither the other.
    expect(by('fasteners-s-v1').speed?.ms_per_image).toBeGreaterThan(0);
    expect(claude.speed).toBeNull();
  });

  it('generates results consistent with the definitions, for every finished entry', async () => {
    const api = new MockApi('normal', 0);
    for (const b of await api.benches()) {
      for (const e of await api.benchEntries(b.id)) {
        if (e.status !== 'done') continue;
        expectConsistent(await api.benchResult(b.id, e.id), b.images, b.plan.conditions.length);
      }
    }
  });

  it('marks an external entry that left conditions out partial, and its robustness is over the others', async () => {
    const api = new MockApi('normal', 0);
    const r = await api.benchResult('bn_7a41', 'be_p203');
    expect(r.robustness!.conditions.map((c) => c.condition)).toEqual(['low_light', 'low_contrast', 'haze', 'color_cast']);
    expect(r.calibration).toBeNull();
    expect(r.entry.score!.partial).toBe(true);
  });

  it('answers per-image results in both sorts, the worst images first', async () => {
    const api = new MockApi('normal', 0);
    const order = await api.benchEntryImages('bn_3c90', 'be_f103', { limit: 20 });
    expect(order.total).toBe(360);
    expect(order.images.map((i) => i.image.id).slice(0, 3)).toEqual(['1', '2', '3']);
    const worst = await api.benchEntryImages('bn_3c90', 'be_f103', { sort: 'worst', limit: 360 });
    const bad = worst.images.map((i) => i.missed + i.wrong);
    expect(bad).toEqual([...bad].sort((a, b) => b - a));
    for (const i of worst.images) {
      expect(i.missed).toBeLessThanOrEqual(i.image.boxes.length);
      expect(i.wrong).toBeLessThanOrEqual(i.detections.length);
    }
    await expect(api.benchEntryImages('bn_3c90', 'be_f108')).rejects.toMatchObject({ status: 409, code: 'bench_entry_not_done' });
  });

  it('has none in the empty scenario, and answers 501 on a server without them', async () => {
    expect(await new MockApi('empty', 0).benches()).toEqual([]);
    const off = new MockApi('no-bench', 0);
    expect((await off.system()).features).not.toContain('vision_bench');
    await expect(off.benches()).rejects.toMatchObject({ status: 501, code: 'not_implemented' });
    expect((await new MockApi('normal', 0).system()).features).toContain('vision_bench');
  });
});

describe('MockApi vision benchmarks: images and conditions', () => {
  it('serves the truth under a condition: distance moves the boxes towards the centre, others keep them', async () => {
    const api = new MockApi('normal', 0);
    const plain = (await api.benchImages('bn_3c90', { limit: 5 })).images;
    const far = (await api.benchImages('bn_3c90', { limit: 5, condition: 'distance', severity: 5 })).images;
    const hazy = (await api.benchImages('bn_3c90', { limit: 5, condition: 'haze', severity: 5 })).images;
    expect(hazy.map((i) => i.boxes)).toEqual(plain.map((i) => i.boxes));
    const k = distanceScale(5);
    const im = plain.find((i) => i.boxes.length)!;
    const b = im.boxes[0]!;
    const moved = far.find((i) => i.id === im.id)!.boxes[0]!;
    expect(moved.x2 - moved.x1).toBeCloseTo((b.x2 - b.x1) * k, 0);
    expect(conditionBoxes(im.boxes, im.width, im.height, 'distance', 1)[0]!.x2 - conditionBoxes(im.boxes, im.width, im.height, 'distance', 1)[0]!.x1).toBeGreaterThan(moved.x2 - moved.x1);
    await expect(api.benchImages('bn_3c90', { condition: 'haze' })).rejects.toMatchObject({ status: 400, code: 'invalid_query' });
    await expect(api.benchImages('bn_3c90', { condition: 'rain', severity: 2 })).rejects.toMatchObject({ status: 400 });
  });
});

describe('MockApi vision benchmarks: photos of every shape', () => {
  const inside = (b: { x1: number; y1: number; x2: number; y2: number }, w: number, h: number) => b.x1 >= 0 && b.y1 >= 0 && b.x2 <= w && b.y2 <= h && b.x2 > b.x1 && b.y2 > b.y1;

  it("serves Pond fish's phone photos in several shapes, truth and detections inside each, also under distance", async () => {
    const api = new MockApi('normal', 0);
    const { images } = await api.benchImages('bn_7a41', { limit: 1000 });
    const shapes = new Map<string, number>();
    for (const im of images) shapes.set(`${im.width}x${im.height}`, (shapes.get(`${im.width}x${im.height}`) ?? 0) + 1);
    // Landscape and portrait, wide and a panorama; 4:3 the commonest.
    expect([...shapes.keys()]).toEqual(expect.arrayContaining(['960x720', '720x960', '1280x720', '1600x520']));
    expect([...shapes.entries()].sort((a, b) => b[1] - a[1])[0]![0]).toBe('960x720');
    expect(images.some((i) => i.height > i.width)).toBe(true);
    for (const im of images) for (const b of im.boxes) expect(inside(b, im.width, im.height), `${im.file} ${im.width}x${im.height}`).toBe(true);
    // Under distance the scene shrinks about its own centre: every box stays inside its own image, whatever its shape.
    const far = (await api.benchImages('bn_7a41', { limit: 1000, condition: 'distance', severity: 5 })).images;
    for (const im of far) {
      const plain = images.find((i) => i.id === im.id)!;
      expect([im.width, im.height]).toEqual([plain.width, plain.height]);
      for (const b of im.boxes) expect(inside(b, im.width, im.height)).toBe(true);
      const c = (x: { x1: number; x2: number; y1: number; y2: number }) => [(x.x1 + x.x2) / 2, (x.y1 + x.y2) / 2];
      if (plain.boxes[0]) {
        // Towards the image's centre, in both directions alike.
        const [px, py] = c(plain.boxes[0]);
        const [fx, fy] = c(im.boxes[0]!);
        expect(Math.abs(fx! - im.width / 2)).toBeLessThanOrEqual(Math.abs(px! - im.width / 2) + 0.1);
        expect(Math.abs(fy! - im.height / 2)).toBeLessThanOrEqual(Math.abs(py! - im.height / 2) + 0.1);
      }
    }
    // An entry's images carry their own sizes; its detections fall on them.
    const page = await api.benchEntryImages('bn_7a41', 'be_p201', { sort: 'worst', limit: 100 });
    expect(new Set(page.images.map((i) => i.image.width / i.image.height)).size).toBeGreaterThan(2);
    for (const i of page.images) for (const d of i.detections) expect(d.x2 > 0 && d.y2 > 0 && d.x1 < i.image.width && d.y1 < i.image.height).toBe(true);
  });

  it('keeps every other benchmark’s images the size they were', async () => {
    const api = new MockApi('normal', 0);
    for (const [id, w, h] of [['bn_3c90', 1024, 768], ['bn_5e12', 640, 640]] as const) {
      const { images } = await api.benchImages(id, { limit: 50 });
      expect(images.every((i) => i.width === w && i.height === h)).toBe(true);
    }
  });
});

describe('MockApi vision benchmarks: making and changing them', () => {
  it('makes one from a dataset split with the default plan, sampled with its seed', async () => {
    const api = new MockApi('normal', 0);
    const b = await api.createBench({ name: 'hard-100', source: { kind: 'dataset', dataset_id: 'ds_shapes_hard', split: 'val' }, max_images: 100 });
    expect(b).toMatchObject({ images: 100, entries: 0, source_name: 'shapes-hard', best: null });
    expect(b.plan.conditions).toHaveLength(12);
    expect((await api.benches())[0]!.id).toBe(b.id);
    const again = await api.createBench({ name: 'hard-100', source: { kind: 'dataset', dataset_id: 'ds_shapes_hard', split: 'val' }, max_images: 100 });
    expect(again.fingerprint).toBe(b.fingerprint);
  });

  it('makes one from a labeling job, of its labeled images only', async () => {
    const api = new MockApi('normal', 0);
    const b = await api.createBench({ name: 'review', source: { kind: 'label_job', job_id: 'lb_51d0' } });
    expect(b.images).toBe(36);
    expect(b.classes).toEqual(['bolt', 'nut', 'washer', 'screw']);
    await expect(api.createBench({ name: 'none', source: { kind: 'label_job', job_id: 'lb_7c21' } })).rejects.toMatchObject({ status: 400, code: 'invalid_dataset' });
    await expect(api.createBench({ name: 'x', source: { kind: 'dataset', dataset_id: 'ds_nope' } })).rejects.toMatchObject({ status: 404, code: 'dataset_not_found' });
    await expect(api.createBench({ name: 'x', source: { kind: 'dataset', dataset_id: 'ds_fasteners_mini', split: 'val' } })).rejects.toMatchObject({ code: 'invalid_dataset' });
    await expect(api.createBench({ name: ' ', source: { kind: 'dataset', dataset_id: 'ds_shapes' } })).rejects.toMatchObject({ code: 'invalid_body' });
  });

  it('changes a plan while there are no entries, refuses one after (bench_frozen), and checks it', async () => {
    const api = new MockApi('normal', 0);
    const plan = { conditions: [{ condition: 'low_light', severities: [2, 4], weight: 2, why: null }], rationale: null, designed_by: null };
    const b = await api.updateBench('bn_9f03', { plan });
    expect(b.plan.conditions).toEqual(plan.conditions);
    await expect(api.updateBench('bn_9f03', { plan: { conditions: [{ condition: 'low_light', severities: [4, 2], weight: 1 }] } })).rejects.toMatchObject({ code: 'invalid_body' });
    await expect(api.updateBench('bn_3c90', { plan })).rejects.toMatchObject({ status: 409, code: 'bench_frozen' });
    // A name and a description can always change.
    expect((await api.updateBench('bn_3c90', { name: 'Fasteners, val split' })).name).toBe('Fasteners, val split');
  });

  it('refuses to delete a benchmark while an entry of it runs (bench_active)', async () => {
    const api = new MockApi('normal', 0);
    await expect(api.deleteBench('bn_3c90')).rejects.toMatchObject({ status: 409, code: 'bench_active' });
    await api.deleteBench('bn_9f03');
    await expect(api.bench('bn_9f03')).rejects.toMatchObject({ status: 404 });
  });

  it('designs a plan from the description after a delay, and answers 501 for an agent that is not installed', async () => {
    const api = new MockApi('normal', 0);
    const d = await api.designBench('bn_9f03', { agent: 'claude', note: '' });
    const ids = d.plan.conditions.map((c) => c.condition);
    expect(ids).toEqual(expect.arrayContaining(['low_light', 'sensor_noise', 'motion_blur']));
    expect(d.plan.designed_by).toEqual({ agent: 'claude', model: 'haiku' });
    expect(d.usage.calls).toBe(1);
    // Not saved.
    expect((await api.bench('bn_9f03')).plan.designed_by).toBeNull();
    await expect(api.designBench('bn_9f03', { agent: 'codex' })).rejects.toMatchObject({ status: 501, code: 'not_supported' });
    await expect(new MockApi('bench-agent-fails', 0).designBench('bn_9f03', { agent: 'claude' })).rejects.toMatchObject({ status: 502, code: 'agent_failed' });
  });

  it('reads the words of a description: water, night, far', () => {
    const ids = (t: string) => designPlan(t, 'claude', null).conditions.map((c) => c.condition);
    expect(ids('a koi pond')).toEqual(expect.arrayContaining(['haze', 'color_cast']));
    expect(ids('a drone, far above the field')).toEqual(expect.arrayContaining(['distance', 'low_resolution']));
    expect(ids('nothing to say')).toHaveLength(5);
  });
});

describe('MockApi vision benchmarks: entries run', () => {
  it('queues a model entry behind the running one, then runs it through its conditions to a result', async () => {
    const api = new MockApi('normal', 0);
    const seen: BenchEventMap['entry'][] = [];
    const close = api.benchEvents('bn_3c90', { entry: (e) => seen.push(e) });
    const e = await api.startBenchEntry('bn_3c90', { contestant: { kind: 'model', model_id: 'm_fast_n' }, max_images: 40 });
    expect(e.status).toBe('queued');
    expect(e.progress.total).toBe(40 * 37);
    // The running entry ends about 70 s after load; this one then takes its turn.
    await vi.advanceTimersByTimeAsync(75_000);
    const mid = await api.benchEntry('bn_3c90', e.id);
    expect(['running', 'done']).toContain(mid.status);
    await vi.advanceTimersByTimeAsync(30_000);
    const done = await api.benchEntry('bn_3c90', e.id);
    expect(done.status).toBe('done');
    expect(done.score!.partial).toBe(true);
    expect(seen.some((x) => x.id === e.id && x.status === 'running' && x.progress.condition)).toBe(true);
    close();
  });

  it('starts an agent entry with the agent defaults: 48 images, unchanged only, no confidences', async () => {
    const api = new MockApi('normal', 0);
    const e = await api.startBenchEntry('bn_9f03', { contestant: { kind: 'agent', agent: 'claude' } });
    // Named as the server names it: the agent, its model in parentheses.
    expect(e).toMatchObject({ images: 48, conditions: false, scored: false, name: 'Claude Code (haiku)' });
    await vi.advanceTimersByTimeAsync(20_000);
    const done = await api.benchEntry('bn_9f03', e.id);
    expect(done.status).toBe('done');
    expect(done.usage!.calls).toBe(12);
    await expect(api.startBenchEntry('bn_9f03', { contestant: { kind: 'agent', agent: 'codex' } })).rejects.toMatchObject({ status: 501 });
  });

  it('refuses a model with none of the classes (bench_class_mismatch), and lists the ones a model lacks', async () => {
    const api = new MockApi('normal', 0);
    await expect(api.startBenchEntry('bn_3c90', { contestant: { kind: 'model', model_id: 'm_coco_s' } })).rejects.toMatchObject({ status: 400, code: 'bench_class_mismatch' });
  });

  it('scores external predictions in the request, and names the first wrong record', async () => {
    const api = new MockApi('normal', 0);
    const images = (await api.benchImages('bn_9f03', { limit: 1000 })).images;
    const predictions = images.map((i) => ({ image: i.id, boxes: i.boxes.map((b) => ({ ...b, score: 0.9 })) }));
    const e = await api.startBenchEntry('bn_9f03', { name: 'perfect', contestant: { kind: 'external', predictions } });
    expect(e.status).toBe('done');
    expect(e.score!.partial).toBe(true);
    expect(e.map50_95).toBeGreaterThan(0.5);
    await expect(api.startBenchEntry('bn_9f03', { name: 'gap', contestant: { kind: 'external', predictions: predictions.slice(1) } })).rejects.toMatchObject({
      code: 'invalid_body',
      message: expect.stringMatching(/no record without a condition/),
    });
    await expect(api.startBenchEntry('bn_9f03', { contestant: { kind: 'external', predictions } })).rejects.toMatchObject({ code: 'invalid_body' });
  });

  it('takes an external entry in parts: open, parts all or nothing, finish, the errors of the API', async () => {
    const api = new MockApi('normal', 0);
    const b = await api.bench('bn_9f03');
    const images = (await api.benchImages('bn_9f03', { limit: 1000 })).images;
    const rec = (condition: string | null, severity: number | null) =>
      images.map((i) => ({ image: i.id, ...(condition ? { condition, severity } : {}), boxes: i.boxes.map((x) => ({ ...x, score: 0.8 })) }));
    const passes = b.plan.conditions.reduce((n, c) => n + c.severities.length, 0);
    // Open, with its first part: running, progress the records received of images × (1 + the plan's severities).
    const e = await api.startBenchEntry('bn_9f03', { name: 'parts', contestant: { kind: 'external', open: true, predictions: rec(null, null).slice(0, 50) } });
    expect(e).toMatchObject({ status: 'running', progress: { done: 50, total: 200 * (1 + passes), condition: null } });
    expect((await api.bench('bn_9f03')).active).toBe(true);
    // A part: all or nothing. One that repeats a record received is refused and adds nothing.
    await expect(api.sendBenchPredictions('bn_9f03', e.id, rec(null, null).slice(40, 60))).rejects.toMatchObject({ status: 400, code: 'invalid_body', message: expect.stringMatching(/answered twice/) });
    expect((await api.benchEntry('bn_9f03', e.id)).progress.done).toBe(50);
    expect((await api.sendBenchPredictions('bn_9f03', e.id, rec(null, null).slice(50))).progress.done).toBe(200);
    const haze = b.plan.conditions.find((c) => c.condition === 'haze')!;
    // A condition in part: the finish names what is missing and leaves it open.
    await api.sendBenchPredictions('bn_9f03', e.id, rec('haze', haze.severities[0]!));
    await expect(api.finishBenchEntry('bn_9f03', e.id)).rejects.toMatchObject({ status: 400, message: expect.stringMatching(/Haze is answered in part/) });
    expect((await api.benchEntry('bn_9f03', e.id)).status).toBe('running');
    for (const s of haze.severities.slice(1)) await api.sendBenchPredictions('bn_9f03', e.id, rec('haze', s));
    const done = await api.finishBenchEntry('bn_9f03', e.id);
    expect(done).toMatchObject({ status: 'done', conditions: true, conditions_run: 1 });
    expect(done.score!.partial).toBe(true);
    // Only an open external entry takes parts.
    await expect(api.sendBenchPredictions('bn_9f03', e.id, [])).rejects.toMatchObject({ status: 409, code: 'bench_entry_not_open' });
    await expect(api.finishBenchEntry('bn_3c90', 'be_f101')).rejects.toMatchObject({ status: 409, code: 'bench_entry_not_open' });
    // Stopping an open entry discards it.
    const gone = await api.startBenchEntry('bn_9f03', { name: 'gone', contestant: { kind: 'external', open: true } });
    const removed: string[] = [];
    const close = api.benchEvents('bn_9f03', { removed: ({ id }) => removed.push(id) });
    await api.stopBenchEntry('bn_9f03', gone.id);
    expect(removed).toEqual([gone.id]);
    expect((await api.benchEntries('bn_9f03')).some((x) => x.id === gone.id)).toBe(false);
    expect((await api.bench('bn_9f03')).active).toBe(false);
    close();
  });

  it('stops and deletes, with the conflicts of the API', async () => {
    const api = new MockApi('normal', 0);
    await expect(api.deleteBenchEntry('bn_3c90', 'be_f108')).rejects.toMatchObject({ status: 409, code: 'bench_entry_active' });
    const stopping = await api.stopBenchEntry('bn_3c90', 'be_f108');
    expect(stopping.status).toBe('stopping');
    await vi.advanceTimersByTimeAsync(1500);
    expect((await api.benchEntry('bn_3c90', 'be_f108')).status).toBe('stopped');
    await expect(api.stopBenchEntry('bn_3c90', 'be_f108')).rejects.toMatchObject({ status: 409, code: 'bench_entry_not_active' });
    await expect(api.benchResult('bn_3c90', 'be_f108')).rejects.toMatchObject({ status: 409, code: 'bench_entry_not_done' });
    const removed: string[] = [];
    const close = api.benchEvents('bn_3c90', { removed: ({ id }) => removed.push(id) });
    await api.deleteBenchEntry('bn_3c90', 'be_f108');
    expect(removed).toEqual(['be_f108']);
    expect((await api.benchEntries('bn_3c90')).some((e) => e.id === 'be_f108')).toBe(false);
    close();
  });
});

describe('MockApi vision benchmarks: loops', () => {
  it('has an improved loop of three generations, a held one and one that starved, their numbers the entries\' own', async () => {
    const api = new MockApi('normal', 0);
    const loops = await api.benchLoops('bn_5e12');
    expect(loops.map((l) => [l.status, l.verdict, l.generations.length])).toEqual([
      ['failed', null, 2],
      ['done', 'held', 2],
      ['done', 'improved', 4],
    ]);
    const entries = new Map((await api.benchEntries('bn_5e12')).map((e) => [e.id, e]));
    for (const l of loops.filter((x) => x.status === 'done')) {
      const first = l.generations[0]!;
      const last = l.generations.at(-1)!;
      expect(first.labels).toBeNull();
      for (const g of l.generations.slice(1)) {
        const k = g.labels!;
        expect(k.threshold).toBeGreaterThan(0);
        // The student's dataset is the pool images the teacher kept a box on: fewer than it looked at, never more
        // than its boxes.
        expect(k.labeled_images).toBeLessThan(k.images);
        expect(k.labeled_images).toBeLessThanOrEqual(k.boxes);
        expect(k.labeled_images).toBeGreaterThanOrEqual(10);
      }
      for (const g of l.generations) expect(entries.get(g.entry_id!)!.map50_95).toBe(g.map50_95);
      const verdict = last.map50_95! > first.ci95![1] ? 'improved' : last.map50_95! < first.ci95![0] ? 'regressed' : 'held';
      expect(l.verdict).toBe(verdict);
    }
  });

  it('names loops and students as the server does', async () => {
    const api = new MockApi('normal', 0);
    const loops = await api.benchLoops('bn_5e12');
    const improved = loops.find((l) => l.verdict === 'improved')!;
    expect(improved.name).toBe('shapes-n-best loop');
    expect(improved.generations.map((g) => g.model_name)).toEqual(['shapes-n-best', 'shapes-n-best loop gen 1-best', 'shapes-n-best loop gen 2-best', 'shapes-n-best loop gen 3-best']);
    // A new loop without a name: `<teacher> loop`; its student `<loop> gen 1-best`, and so its entry.
    const l = await api.startBenchLoop('bn_5e12', { teacher: 'm_shapes_n', pool: { kind: 'dataset', dataset_id: 'ds_shapes', split: 'train' } });
    expect(l.name).toBe('shapes-n-best loop');
    for (let t = 0; t < 120 && (await api.benchLoop('bn_5e12', l.id)).status === 'running'; t++) await vi.advanceTimersByTimeAsync(1000);
    const done = await api.benchLoop('bn_5e12', l.id);
    expect(done.generations[1]!.model_name).toBe('shapes-n-best loop gen 1-best');
    expect((await api.benchEntry('bn_5e12', done.generations[1]!.entry_id!)).name).toBe('shapes-n-best loop gen 1-best');
  });

  it('fails a loop whose teacher keeps boxes on fewer than 10 pool images, as the server words it', async () => {
    const api = new MockApi('normal', 0);
    const starved = (await api.benchLoops('bn_5e12')).find((l) => l.status === 'failed')!;
    expect(starved).toMatchObject({ id: 'bl_9e40', name: 'shapes-n-early loop', phase: null, verdict: null });
    // Generation 0 graded; generation 1 started and left with nothing in it: no labels, no run, no model.
    expect(starved.generations[0]!.map50_95).not.toBeNull();
    expect(starved.generations[1]).toMatchObject({ generation: 1, labels: null, run_id: null, model_id: null });
    const m = /^the teacher 'shapes-n-early' kept (\d+) box(?:es)? on (\d+) of the pool's 800 images at confidence (0\.\d+): too few to train a student on \(at least 10 images\)$/.exec(starved.error!);
    expect(m, starved.error!).not.toBeNull();
    expect(Number(m![2])).toBeLessThan(10);
    expect(Number(m![2])).toBeLessThanOrEqual(Number(m![1]));
    expect(starved.hint).toMatch(/lower min_confidence/);
    // The teacher's entry: not ready, its threshold the one the loop kept labels from.
    const r = await api.benchResult('bn_5e12', starved.generations[0]!.entry_id!);
    expect(r.self_labeling.readiness).toBe('not_ready');
    expect(r.self_labeling.recall).toBeLessThan(0.01);
    expect(Number(m![3])).toBeCloseTo(r.self_labeling.threshold!, 2);
    // Starting a loop from that teacher again fails the same way, at labeling.
    const l = await api.startBenchLoop('bn_5e12', { teacher: 'm_shapes_e5', pool: { kind: 'dataset', dataset_id: 'ds_shapes', split: 'train' } });
    for (let t = 0; t < 60 && (await api.benchLoop('bn_5e12', l.id)).status === 'running'; t++) await vi.advanceTimersByTimeAsync(1000);
    const again = await api.benchLoop('bn_5e12', l.id);
    expect(again.status).toBe('failed');
    expect(again.error).toMatch(/too few to train a student on/);
    expect(again.generations.at(-1)!.labels).toBeNull();
    // A lower confidence keeps enough: the same teacher from 0.05 labels its pool, the images it kept nothing on left out.
    const low = await api.startBenchLoop('bn_5e12', { teacher: 'm_shapes_e5', min_confidence: 0.05, pool: { kind: 'dataset', dataset_id: 'ds_shapes', split: 'train' } });
    for (let t = 0; t < 20 && !(await api.benchLoop('bn_5e12', low.id)).generations[1]?.labels; t++) await vi.advanceTimersByTimeAsync(1000);
    const k = (await api.benchLoop('bn_5e12', low.id)).generations[1]!.labels!;
    expect(k.threshold).toBe(0.05);
    expect(k.labeled_images).toBeGreaterThanOrEqual(10);
    expect(k.labeled_images).toBeLessThan(k.images);
  });

  it('runs a new loop through its phases, telling each on the stream', async () => {
    const api = new MockApi('normal', 0);
    const phases: string[] = [];
    const close = api.benchEvents('bn_5e12', { loop: (l) => phases.push(`${l.generations.length}:${l.phase ?? l.status}`) });
    const l = await api.startBenchLoop('bn_5e12', { teacher: 'm_shapes_n', pool: { kind: 'dataset', dataset_id: 'ds_shapes', split: 'train' }, generations: 1 });
    expect(l.status).toBe('running');
    // The teacher has a finished, complete entry on this fingerprint: it is reused.
    expect(l.generations[0]!.entry_id).toBe('be_5a10');
    for (let t = 0; t < 120 && (await api.benchLoop('bn_5e12', l.id)).status === 'running'; t++) await vi.advanceTimersByTimeAsync(1000);
    const done = await api.benchLoop('bn_5e12', l.id);
    expect(done.status).toBe('done');
    expect(done.verdict).not.toBeNull();
    expect(done.generations[1]!.run_id).toMatch(/^run_/);
    expect(phases.join(' ')).toMatch(/labeling.*training.*benchmarking.*done/);
    await expect(api.startBenchLoop('bn_5e12', { teacher: 'm_fast_n', pool: { kind: 'dataset', dataset_id: 'ds_shapes' } })).rejects.toMatchObject({ code: 'invalid_body' });
    await expect(api.startBenchLoop('bn_5e12', { teacher: 'm_shapes_n', pool: { kind: 'dataset', dataset_id: 'ds_shapes', split: 'val' } })).rejects.toMatchObject({ code: 'invalid_dataset' });
    close();
  });
});

describe('MockApi vision benchmarks: the contract as the engine fills it', () => {
  it("names and words the conditions as docs/BENCH.md's table does", async () => {
    const table = benchDoc.slice(benchDoc.indexOf('<!-- conditions:begin'), benchDoc.indexOf('<!-- conditions:end'));
    const rows = table
      .split('\n')
      .filter((l) => l.startsWith('| `'))
      .map((l) => l.split('|').map((c) => c.trim()))
      .map((c) => ({ id: c[1]!.replace(/`/g, ''), name: c[2]!, group: c[3]!, description: c[4]! }));
    expect(rows).toHaveLength(12);
    const cat = await new MockApi('normal', 0).benchCatalog();
    expect(cat.conditions).toEqual(rows);
  });

  it('says which benchmarks are active and whether the leader is partial', async () => {
    const api = new MockApi('normal', 0);
    for (const b of await api.benches()) {
      const entries = await api.benchEntries(b.id);
      const loops = await api.benchLoops(b.id);
      const busy = entries.some((e) => ['queued', 'running', 'stopping'].includes(e.status)) || loops.some((l) => l.status === 'running' || l.status === 'stopping');
      expect(b.active).toBe(busy);
      const lead = entries.find((e) => e.status === 'done');
      if (lead) expect(b.best).toEqual({ entry_id: lead.id, name: lead.name, score: lead.score!.total, partial: lead.score!.partial });
      else expect(b.best).toBeNull();
    }
    const by = new Map((await api.benches()).map((b) => [b.name, b]));
    expect(by.get('Fasteners val')!.active).toBe(true);
    expect(by.get('Shapes val')!.active).toBe(false);
    // Deleting answers 409 exactly while it is active.
    await expect(api.deleteBench(by.get('Fasteners val')!.id)).rejects.toMatchObject({ status: 409, code: 'bench_active' });
  });

  it("repeats each finished entry's interval and the conditions it ran; nothing before it is done", async () => {
    const api = new MockApi('normal', 0);
    for (const b of await api.benches()) {
      for (const e of await api.benchEntries(b.id)) {
        if (e.status !== 'done') {
          expect(e.ci95).toBeNull();
          expect(e.conditions_run).toBe(0);
          continue;
        }
        const r = await api.benchResult(b.id, e.id);
        expect(e.ci95).toEqual(r.accuracy.ci95);
        expect(e.conditions_run).toBe(r.robustness?.conditions.length ?? 0);
        expect(e.conditions_run).toBeLessThanOrEqual(b.plan.conditions.length);
      }
    }
    const pond = await api.benchEntries('bn_7a41');
    expect(pond.find((e) => e.id === 'be_p203')!.conditions_run).toBe(4);
  });

  it('reports all detections (t = 0.00) when no threshold reaches 90% precision', async () => {
    const api = new MockApi('normal', 0);
    const r = await api.benchResult('bn_7a41', 'be_p204');
    const s = r.self_labeling;
    expect(r.entry.scored).toBe(true);
    expect(s.curve.every((p) => p.precision === null || p.precision < 0.9)).toBe(true);
    expect(s.threshold).toBeNull();
    expect(s.recall).toBe(0);
    expect(s.readiness).toBe('not_ready');
    const all = s.curve[0]!;
    expect(all.confidence).toBe(0);
    expect(s.precision).toBe(all.precision);
    expect(s.labels_per_image).toBeCloseTo(all.labels / r.entry.images, 2);
    expect(r.entry.score!.axes[2]!.value).toBe(0);
  });

  it("gives a contestant without confidences one operating point, at confidence 0", async () => {
    const api = new MockApi('normal', 0);
    for (const [b, e] of [['bn_3c90', 'be_f104'], ['bn_7a41', 'be_p203']] as const) {
      const r = await api.benchResult(b, e);
      expect(r.self_labeling.curve).toHaveLength(1);
      expect(r.self_labeling.curve[0]!.confidence).toBe(0);
      expect(r.self_labeling.threshold).toBeNull();
      expect(r.calibration).toBeNull();
    }
  });

  it('keeps a plan without conditions from making its entries partial', async () => {
    const api = new MockApi('normal', 0);
    await api.updateBench('bn_9f03', { plan: { conditions: [], rationale: null, designed_by: null } });
    const images = (await api.benchImages('bn_9f03', { limit: 1000 })).images;
    const predictions = images.map((i) => ({ image: i.id, boxes: i.boxes.map((b) => ({ ...b, score: 0.9 })) }));
    const e = await api.startBenchEntry('bn_9f03', { name: 'all', contestant: { kind: 'external', predictions } });
    expect(e.score!.axes[1]!.value).toBeNull();
    expect(e.score!.partial).toBe(false);
    expect(e.conditions_run).toBe(0);
  });

  it('has a loop running as the demo opens, its first student training on its teacher’s labels', async () => {
    const api = new MockApi('normal', 0);
    const l = await api.benchLoop('bn_3c90', 'bl_f3a0');
    expect(l).toMatchObject({ status: 'running', phase: 'training', verdict: null });
    expect(l.generations).toHaveLength(2);
    expect(l.generations[0]!.entry_id).toBe('be_f103');
    expect(l.generations[1]!.labels!.threshold).toBeGreaterThan(0);
    expect(l.generations[1]!.run_id).toMatch(/^run_/);
    expect(l.generations[1]!.model_id).toBeNull();
  });
});

describe('MockApi vision benchmarks: a loop at generation 0 (`bench-loop-gen0`)', () => {
  it('benchmarks its first teacher, its entry waiting for the GPU, then labels for generation 1', async () => {
    const api = new MockApi('bench-loop-gen0', 0);
    const l = await api.benchLoop('bn_3c90', 'bl_g0a1');
    expect(l).toMatchObject({ name: 'fasteners-n-f16 loop', status: 'running', phase: 'benchmarking' });
    expect(l.generations).toHaveLength(1);
    expect(l.generations[0]).toMatchObject({ generation: 0, labels: null, run_id: null, model_id: 'm_fast_n_f16', score: null });
    const e = await api.benchEntry('bn_3c90', l.generations[0]!.entry_id!);
    expect(e).toMatchObject({ name: 'fasteners-n-f16', status: 'queued' });
    // Not in the normal demo.
    await expect(new MockApi('normal', 0).benchLoop('bn_3c90', 'bl_g0a1')).rejects.toMatchObject({ status: 404 });
    // Once its entry is graded, generation 1 starts with the teacher labeling the pool.
    for (let t = 0; t < 200 && (await api.benchLoop('bn_3c90', 'bl_g0a1')).generations.length < 2; t++) await vi.advanceTimersByTimeAsync(1000);
    const next = await api.benchLoop('bn_3c90', 'bl_g0a1');
    expect(next.generations[0]!.map50_95).not.toBeNull();
    expect(next.phase).toBe('labeling');
    expect(next.generations[1]).toMatchObject({ generation: 1, labels: null, run_id: null });
  });
});

describe('rankEntries', () => {
  it('puts finished entries by score first, then the rest newest first', () => {
    const e = (id: string, status: BenchEntry['status'], total: number | null, created: string) => ({ id, status, created, score: total === null ? null : { total, partial: false, axes: [] } }) as unknown as BenchEntry;
    const out = rankEntries([e('a', 'running', null, '2026-01-02'), e('b', 'done', 50, '2026-01-01'), e('c', 'done', 70, '2026-01-01'), e('d', 'failed', null, '2026-01-03')]);
    expect(out.map((x) => x.id)).toEqual(['c', 'b', 'd', 'a']);
  });
});
