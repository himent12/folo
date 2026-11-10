// The mock's benchmark results: everything an entry measures (docs/BENCH.md, Definitions), generated from a seed and a
// contestant's skill, then derived with the definitions themselves ($lib/bench/score), so every number agrees with the
// others the way a server's do: retention is min(1, mAP under / mAP unchanged), robustness their weighted mean, the
// ECE what its bins say, the self-labeling threshold where its curve first reaches 90% precision, and the score the
// weighted mean of the axes that were measured. `benchResults.test.ts` holds them to that.

import type {
  BenchAccuracy,
  BenchCalibration,
  BenchCalibrationBin,
  BenchConditionResult,
  BenchDetection,
  BenchImage,
  BenchLabelPoint,
  BenchLocalization,
  BenchPlan,
  BenchRobustness,
  BenchScore,
  BenchSelfLabeling,
  Box,
} from '../types';
import {
  axisScores,
  conditionRetention,
  expectedCalibrationError,
  f1,
  histogramMeanIou,
  labelOperatingPoint,
  localizationAxis,
  readiness,
  retention,
  robustnessAxis,
  scoreTotal,
} from '../../bench/score';
import { TARGET_PRECISION } from '../../bench/rules';
import { predictBoxes } from './boxes';
import { HARDNESS } from './benchCatalog';
import { iou } from './datasets';
import { hash, rng } from './rng';

/** How good a contestant is, before the benchmark measures it. */
export interface Skill {
  /** mAP@0.5:0.95 on the unchanged images, were it to know every class. */
  map: number;
  /** How fast it loses accuracy under the conditions: 1 is a typical detector, 0.7 a robust one. */
  fragility: number;
  /** Per condition, a factor on top (an agent shrugs off a colour cast, a small model hates distance). */
  fragile?: Partial<Record<string, number>>;
  /** The expected calibration error its confidences come to. */
  ece: number;
  /** Mean IoU of its correct detections. */
  iou: number;
  /** The share of all objects it finds at 90% precision; 0 when it never gets there. */
  selfRecall: number;
  /**
   * How steeply the objects it labels fall off as the confidence rises (the exponent of its recall curve; 1.6 when
   * absent). A steep one with a tiny `selfRecall` reaches 90% right only at a high confidence where it keeps almost
   * nothing: a teacher that starves a loop of labels.
   */
  labelCurve?: number;
  /**
   * The shape of its reliability: `mixed` is underconfident where most detections are (low scores) and overconfident in
   * a few thin top bins, so a reading that quoted the furthest bin of either side would contradict its own tone; `none`
   * gives confidences that never reach 0.1 (calibration not measured, the entry partial for that alone).
   */
  calibration?: 'mixed' | 'none';
  /** Without confidences: its one operating point. */
  point?: { precision: number; recall: number };
}

/** What a result is of: the benchmark, the entry and the truth of the images it ran. */
export interface ResultInput {
  benchId: string;
  entryId: string;
  classes: number;
  plan: BenchPlan;
  /** The benchmark's images, and those the entry ran. */
  benchImages: number;
  images: readonly Pick<BenchImage, 'width' | 'height' | 'boxes'>[];
  /** Whether it ran the plan's conditions, and which (an external contestant may answer a part; null: all of them). */
  conditions: boolean;
  conditionsRun?: readonly string[] | null;
  scored: boolean;
  /** Class indices it does not know: AP 0. */
  missing: ReadonlySet<number>;
  skill: Skill;
}

export interface GeneratedResult {
  accuracy: BenchAccuracy;
  robustness: BenchRobustness | null;
  calibration: BenchCalibration | null;
  localization: BenchLocalization;
  self_labeling: BenchSelfLabeling;
  score: BenchScore;
}

const r4 = (x: number) => Math.round(x * 1e4) / 1e4;
const r2 = (x: number) => Math.round(x * 100) / 100;
const clamp = (x: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, x));
const mean = (xs: readonly number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

/** Areas of COCO's size ranges. */
const SMALL = 32 * 32;
const LARGE = 96 * 96;

export function generateResult(input: ResultInput): GeneratedResult {
  const r = rng(hash(`result:${input.benchId}:${input.entryId}`));
  const jitter = (amp: number) => (r() - 0.5) * 2 * amp;
  const truths = Array.from({ length: input.classes }, () => 0);
  let small = 0;
  let medium = 0;
  let large = 0;
  for (const im of input.images)
    for (const b of im.boxes) {
      truths[b.class] = (truths[b.class] ?? 0) + 1;
      const a = (b.x2 - b.x1) * (b.y2 - b.y1);
      if (a < SMALL) small++;
      else if (a > LARGE) large++;
      else medium++;
    }
  const T = truths.reduce((a, b) => a + b, 0);
  const s = input.skill;

  // ---------- accuracy: per-class AP around the skill, 0 on the classes it does not know; the mean is the axis ----------
  const present = truths.map((n, c) => (n > 0 ? c : -1)).filter((c) => c >= 0);
  const known = present.filter((c) => !input.missing.has(c));
  const spread = known.map(() => jitter(0.16));
  const shift = mean(spread);
  const ap = new Map<number, number>();
  known.forEach((c, i) => ap.set(c, r4(clamp(s.map * (1 + (spread[i]! - shift)), 0.01, 0.97))));
  const per_class = present.map((c) => {
    const a = ap.get(c) ?? 0;
    return { class: c, truths: truths[c]!, map50_95: a, map50: a > 0 ? r4(clamp(a * 1.36 + 0.07, 0, 0.995)) : 0 };
  });
  const map50_95 = r4(mean(per_class.map((p) => p.map50_95)));
  const map50 = r4(mean(per_class.map((p) => p.map50)));
  const map75 = r4(clamp(mean(per_class.map((p) => (p.map50_95 > 0 ? p.map50_95 * 1.07 - 0.015 : 0))), 0, map50));
  const share = present.length ? known.length / present.length : 0;
  const precision = r4(clamp((0.46 + map50_95 * 0.72) * (0.5 + 0.5 * share)));
  const recall = r4(clamp((0.34 + map50_95 * 0.82) * share));
  // A 95% interval of a bootstrap over images: about as wide as the binomial error of the images it ran, its
  // resamples centred on the value (docs/BENCH.md, Accuracy). Their percentiles still fall unevenly about it, more so
  // with few images or rare classes, so some intervals are lopsided and some symmetric as printed (the ± rule).
  const hw = 1.2 * Math.sqrt((map50_95 * (1 - map50_95)) / Math.max(1, input.images.length)) + 0.004;
  const skew = r() < 0.4 ? 1 : 0.75 + 0.5 * r();
  const ci95: [number, number] = [r4(clamp(map50_95 - hw * skew)), r4(clamp(map50_95 + hw / skew))];
  const accuracy: BenchAccuracy = {
    map50_95,
    ci95,
    map50,
    map75,
    precision,
    recall,
    f1: r4(f1(precision, recall)),
    best_conf: input.scored ? r2(0.24 + r() * 0.24) : 1,
    map_small: small ? r4(clamp(map50_95 * 0.58 + jitter(0.02))) : null,
    map_medium: medium ? r4(clamp(map50_95 * 1.02 + jitter(0.02))) : null,
    map_large: large ? r4(clamp(map50_95 * 1.16 + jitter(0.02))) : null,
    per_class,
  };

  // ---------- robustness: each severity's mAP a share of the unchanged one, retention as defined ----------
  let robustness: BenchRobustness | null = null;
  const ran = input.conditions ? input.plan.conditions.filter((c) => !input.conditionsRun || input.conditionsRun.includes(c.condition)) : [];
  if (ran.length) {
    const conditions: BenchConditionResult[] = ran.map((c) => {
      const hard = (HARDNESS[c.condition] ?? 0.4) * s.fragility * (s.fragile?.[c.condition] ?? 1);
      const points = c.severities.map((sev) => {
        const keep = clamp(1 - hard * (sev / 5) ** 1.35 + jitter(0.015), 0.02, 1.03);
        const m = r4(map50_95 * keep);
        return { severity: sev, map50_95: m, map50: r4(clamp(map50 * keep ** 0.85, 0, 0.995)), retention: r4(retention(m, map50_95)) };
      });
      return { condition: c.condition, weight: c.weight, retention: r4(conditionRetention(points)), points };
    });
    robustness = { retention: r4(robustnessAxis(conditions) ?? 0), conditions };
  }

  // ---------- calibration: nine filled bins (detections from 0.1 up), precision off by what makes the ECE ----------
  let calibration: BenchCalibration | null = null;
  const detections = Math.round(T * 1.7 + input.images.length * 0.6);
  if (input.scored && detections > 0 && s.calibration === 'mixed') {
    // Most detections at low scores, right more often than they say; a thin top, right far less often.
    const SHAPE = [0, 0.34, 0.26, 0.17, 0.1, 0.06, 0.035, 0.02, 0.01, 0.005];
    const bins: BenchCalibrationBin[] = SHAPE.map((w, b) => {
      const count = b ? Math.max(3, Math.round(detections * w)) : 0;
      const conf = r4(b / 10 + 0.05 + jitter(0.01));
      const off = b <= 4 ? 0.07 + jitter(0.01) : -(0.12 + 0.05 * (b - 5));
      return { lo: b / 10, hi: (b + 1) / 10, count, confidence: count ? conf : null, precision: count ? r4(clamp(conf + off, 0.01, 0.995)) : null };
    });
    calibration = { ece: r4(expectedCalibrationError(bins)), detections: bins.reduce((a, x) => a + x.count, 0), bins };
  } else if (input.scored && detections > 0 && s.calibration !== 'none') {
    const SHAPE = [0, 0.21, 0.15, 0.11, 0.09, 0.08, 0.08, 0.08, 0.1, 0.1];
    const raw = SHAPE.map((w, b) => ({ b, n: Math.round(detections * w), conf: b ? r4(b / 10 + 0.05 + jitter(0.015)) : 0 }));
    // Over- or under-confident in proportion to how far a bin is from 0.35: scaled so the ECE comes to the skill's.
    const gaps = raw.map((x) => (x.n ? x.conf - 0.35 : 0));
    const n = raw.reduce((a, x) => a + x.n, 0);
    const k = s.ece / Math.max(1e-6, raw.reduce((a, x, i) => a + (x.n / n) * Math.abs(gaps[i]!), 0));
    const bins: BenchCalibrationBin[] = raw.map((x, i) => ({
      lo: x.b / 10,
      hi: (x.b + 1) / 10,
      count: x.n,
      confidence: x.n ? x.conf : null,
      precision: x.n ? r4(clamp(x.conf - k * gaps[i]! + jitter(0.006), 0.01, 0.995)) : null,
    }));
    calibration = { ece: r4(expectedCalibrationError(bins)), detections: bins.reduce((a, x) => a + x.count, 0), bins };
  }

  // ---------- localization: an IoU histogram around the skill; the mean is what it says ----------
  const matches = Math.round(T * clamp(recall * 1.05));
  const centres = Array.from({ length: 10 }, (_, i) => 0.525 + i * 0.05);
  const wts = centres.map((c) => Math.exp(-(((c - (s.iou + 0.015)) / 0.09) ** 2)));
  const wsum = wts.reduce((a, b) => a + b, 0);
  const histogram = wts.map((w) => Math.floor((matches * w) / wsum));
  const rest = matches - histogram.reduce((a, b) => a + b, 0);
  const peak = wts.indexOf(Math.max(...wts));
  histogram[peak] = histogram[peak]! + rest;
  const meanIou = histogramMeanIou(histogram);
  const localization: BenchLocalization = { mean_iou: meanIou === null ? null : r4(meanIou), matches, histogram };

  // ---------- self-labeling: precision and recall of the labels kept at each confidence ----------
  let curve: BenchLabelPoint[];
  if (!input.scored) {
    const p = s.point ?? { precision: 0.9, recall: 0.5 };
    const rec = r4(clamp(p.recall * share));
    // One operating point, kept from confidence 0: every detection it gave (they all count as score 1).
    curve = [{ confidence: 0, precision: r4(p.precision), recall: rec, labels: Math.round((T * rec) / Math.max(0.01, p.precision)) }];
  } else {
    const rmax = clamp(0.52 + map50_95 * 0.72, 0, 0.97) * share;
    const b = s.labelCurve ?? 1.6;
    const target = s.selfRecall * share;
    const pmin = 0.3;
    const pmax = target > 0 ? 0.985 : 0.86;
    const t0 = target > 0 && target < rmax ? 1 - (target / rmax) ** (1 / b) : 0;
    const q = (TARGET_PRECISION + 0.0006 - pmin) / (pmax - pmin);
    const width = 0.08;
    const c = target > 0 ? t0 - width * Math.log(q / (1 - q)) : 0.55;
    curve = Array.from({ length: 101 }, (_, i) => {
      const t = i / 100;
      const rec = r4(rmax * (1 - t) ** b);
      const prec = pmin + (pmax - pmin) / (1 + Math.exp(-(t - c) / width));
      const labels = rec > 0 ? Math.round((T * rec) / prec) : 0;
      return { confidence: t, precision: labels > 0 ? r4(prec) : null, recall: labels > 0 ? rec : 0, labels };
    });
  }
  const op = labelOperatingPoint(curve);
  const selfRecall = op.point?.recall ?? 0;
  // At t*, the labels kept there; when no threshold reaches the target (and for a contestant without confidences that
  // misses it), every detection it gave: the curve's first point, t = 0.00 (as the engine reports it).
  const at = op.point ?? curve[0]!;
  const mean_iou =
    matches > 0 && localization.mean_iou !== null && at.labels > 0
      ? r4(clamp(localization.mean_iou + (op.point ? 0.012 : -0.008), 0.5, 1))
      : null;
  const self_labeling: BenchSelfLabeling = {
    target_precision: TARGET_PRECISION,
    threshold: op.threshold,
    precision: at.precision ?? 0,
    recall: selfRecall,
    mean_iou,
    labels_per_image: r2(at.labels / Math.max(1, input.images.length)),
    labels: at.labels,
    readiness: readiness(selfRecall),
    curve,
  };

  // ---------- the score: the measured axes' weighted mean ----------
  const axes = axisScores({
    accuracy: map50_95,
    robustness: robustness ? robustness.retention : null,
    self_labeling: r4(selfRecall),
    calibration: calibration ? r4(1 - calibration.ece) : null,
    localization: r4(localizationAxis(localization.mean_iou)),
  });
  const planned = input.plan.conditions.length > 0;
  const partial =
    (planned && !robustness) ||
    !calibration ||
    input.images.length < input.benchImages ||
    (!!robustness && robustness.conditions.length < input.plan.conditions.length);
  return { accuracy, robustness, calibration, localization, self_labeling, score: { total: r2(scoreTotal(axes)), partial, axes } };
}

/**
 * One image's detections at or above `bestConf` (highest first) and what they missed and got wrong, matched as
 * docs/BENCH.md says: per class, in descending confidence, to the unmatched true box of highest IoU from 0.5.
 */
export function imageDetections(
  image: Pick<BenchImage, 'width' | 'height' | 'boxes'>,
  o: { quality: number; seed: number; classes: number; missing: ReadonlySet<number>; bestConf: number; scored: boolean },
): { detections: BenchDetection[]; missed: number; wrong: number } {
  const known = image.boxes.filter((b) => !o.missing.has(b.class));
  const raw = predictBoxes(known, image.width, image.height, o.quality, o.seed, o.classes).filter((b) => !o.missing.has(b.class));
  // A contestant without confidences answers with what it is sure of (its own cut); every score is then 1.
  const detections: BenchDetection[] = raw
    .filter((b) => b.score >= (o.scored ? o.bestConf : 0.32))
    .map(({ class: c, x1, y1, x2, y2, score }) => ({ class: c, x1, y1, x2, y2, score: o.scored ? score : 1 }))
    // Highest confidence first, as the server sends them.
    .sort((a, b) => b.score - a.score);
  const used = new Set<Box>();
  let right = 0;
  for (const d of detections) {
    let hit: Box | null = null;
    let bestIou = 0.5;
    for (const t of image.boxes) {
      if (t.class !== d.class || used.has(t)) continue;
      const v = iou(t, d);
      if (v >= bestIou) {
        bestIou = v;
        hit = t;
      }
    }
    if (hit) {
      used.add(hit);
      right++;
    }
  }
  return { detections, missed: image.boxes.length - used.size, wrong: detections.length - right };
}
