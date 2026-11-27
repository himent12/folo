// Operating point and server-calibrated threshold helpers for Model Detail (`#/models/<id>`):
// - `buildModelOperatingPoint(ev)` reads ONLY `/eval`'s own measured `best_f1_conf` (never substituting
//   unrelated checkpoint `model.metrics.best_f1_conf` or fabricating per-class cutoffs from PR curves).
// - `formatServerCalibration(cal, classes)` formats the exact `folo_vision::ThresholdCalibration` returned by
//   `POST /api/val` (0.001-grid per-class sweep with evaluator-provided `truths` and `preds`) for both
//   `thresholds.json` download (`folo predict --thresholds` / Playground import) and Playground handoff.
import type { Evaluation } from '$lib/api/evaluation';
import type { ThresholdCalibration } from '$lib/api/types';
import { downloadText, safeFilename, toJson } from '$lib/tableExport';
import { roundConf, type PerClassConfMap } from '../playground/perClassConf';

export const THRESHOLD_CALIBRATION_FEATURE = 'threshold_calibration';

export interface ModelOperatingSummary {
  globalConf: number;
  macroF1: number;
  classCount: number;
}

export interface ModelClassThreshold {
  class: number;
  name: string;
  best_conf: number;
  best_f1: number;
  precision: number;
  recall: number;
  truths: number;
  preds: number;
}

/**
 * Matches `folo_vision::ThresholdCalibration` (with human-readable `name` per class as written by
 * `folo val --save-thresholds`), ready for `folo predict --thresholds` and Playground JSON import.
 */
export interface ModelThresholdCalibration {
  global: {
    best_conf: number;
    best_f1: number;
    precision: number;
    recall: number;
  };
  macro_f1: number;
  per_class: ModelClassThreshold[];
}

export interface ServerModelCalibration {
  /** Exact server-computed `ThresholdCalibration` payload (with class `name` added per entry). */
  payload: ModelThresholdCalibration;
  /** Per-class confidence map (`classId -> best_conf`) ready for Playground handoff. */
  perClassConf: PerClassConfMap;
}

function isUnitFinite(v: number | null | undefined): v is number {
  return typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1;
}

function roundGrid(v: number): number {
  const clamped = Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
  return Math.round(clamped * 1000) / 1000;
}

function roundMetric(v: number): number {
  const clamped = Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
  return Math.round(clamped * 10_000) / 10_000;
}

function f1Of(p: number, r: number): number {
  const denom = p + r;
  return denom > 0 ? (2 * p * r) / denom : 0;
}

/**
 * Summarizes the evaluator's measured global operating point from `GET /api/models/{id}/eval`.
 * Requires a valid finite `ev.bestF1Conf` reported for this exact evaluation (returns `null` when
 * `ev.bestF1Conf` is absent/null or no classes with ground truth were evaluated).
 */
export function buildModelOperatingPoint(ev: Evaluation): ModelOperatingSummary | null {
  if (!isUnitFinite(ev.bestF1Conf)) return null;
  const byClass = new Map<number, (typeof ev.perClass)[number]>();
  for (const c of ev.perClass) {
    if (!Number.isInteger(c.class) || c.class < 0 || !Number.isFinite(c.instances) || c.instances <= 0) continue;
    if (!byClass.has(c.class)) byClass.set(c.class, c);
  }
  const evaluated = [...byClass.values()];
  if (evaluated.length === 0) return null;

  const clamp01 = (v: number) => (Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0);
  const f1Sum = evaluated.reduce((s, c) => s + f1Of(clamp01(c.precision), clamp01(c.recall)), 0);
  return {
    globalConf: roundConf(roundGrid(ev.bestF1Conf)),
    macroF1: roundMetric(f1Sum / evaluated.length),
    classCount: evaluated.length,
  };
}

/**
 * Formats a server-computed `ThresholdCalibration` from `POST /api/val` for `thresholds.json` download
 * (adding human-readable `name` per class as `folo val --save-thresholds` does) and Playground handoff.
 * Returns `null` if the server calibration has no calibrated classes or fails finite `[0, 1]` checks.
 */
export function formatServerCalibration(
  cal: ThresholdCalibration | undefined | null,
  classes: readonly string[],
): ServerModelCalibration | null {
  if (!cal || !isUnitFinite(cal.global?.best_conf)) return null;
  const byClass = new Map<number, (typeof cal.per_class)[number]>();
  for (const c of cal.per_class ?? []) {
    if (!Number.isInteger(c.class) || c.class < 0 || !isUnitFinite(c.best_conf)) continue;
    if (!byClass.has(c.class)) byClass.set(c.class, c);
  }
  const sorted = [...byClass.values()].sort((a, b) => a.class - b.class);
  if (sorted.length === 0) return null;

  const perClassConf: PerClassConfMap = {};
  const perClass: ModelClassThreshold[] = sorted.map((c) => {
    const conf = roundConf(c.best_conf);
    perClassConf[c.class] = conf;
    return {
      class: c.class,
      name: c.name ?? classes[c.class] ?? `class ${c.class}`,
      best_conf: conf,
      best_f1: roundMetric(c.best_f1),
      precision: roundMetric(c.precision),
      recall: roundMetric(c.recall),
      truths: Math.max(0, Math.round(Number.isFinite(c.truths) ? c.truths : 0)),
      preds: Math.max(0, Math.round(Number.isFinite(c.preds) ? c.preds : 0)),
    };
  });

  return {
    payload: {
      global: {
        best_conf: roundConf(cal.global.best_conf),
        best_f1: roundMetric(cal.global.best_f1),
        precision: roundMetric(cal.global.precision),
        recall: roundMetric(cal.global.recall),
      },
      macro_f1: roundMetric(cal.macro_f1),
      per_class: perClass,
    },
    perClassConf,
  };
}

/** Downloads `calibration.payload` as `<model-name>-thresholds.json` ready for `folo predict --thresholds` or Playground. */
export function downloadModelThresholds(modelName: string, calibration: ServerModelCalibration): boolean {
  const filename = safeFilename(`${modelName || 'model'}-thresholds`, 'json');
  return downloadText(toJson(calibration.payload), filename, 'application/json;charset=utf-8', 'model-thresholds');
}
