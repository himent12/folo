// Pure measured operating-point threshold builder for Model Detail (`#/models/<id>`): exports the evaluator's
// measured `best_f1_conf` and per-class precision/recall/F1 at that operating point in `folo_vision::ThresholdCalibration`
// JSON format (compatible with both `folo predict --thresholds` and Playground's `parseThresholdsJson`), without
// fabricating unmeasured per-class thresholds from the COCO PR-curve envelope.
import type { Evaluation } from '$lib/api/evaluation';
import type { Model } from '$lib/api/types';
import { downloadText, safeFilename, toJson } from '$lib/tableExport';
import { roundConf, type PerClassConfMap } from '../playground/perClassConf';

export interface ModelOperatingPoint {
  best_conf: number;
  best_f1: number;
  precision: number;
  recall: number;
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
  global: ModelOperatingPoint;
  macro_f1: number;
  per_class: ModelClassThreshold[];
}

export interface ComputedModelCalibration {
  /** JSON payload matching `folo_vision::ThresholdCalibration` at the evaluator's measured operating point. */
  payload: ModelThresholdCalibration;
  /** Per-class overrides for Playground handoff (empty unless distinct per-class thresholds were measured). */
  perClassConf: PerClassConfMap;
  /** True when per-class thresholds came from an actual per-class sweep rather than the global operating point. */
  hasMeasuredPerClass: boolean;
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
 * Builds a `ThresholdCalibration` payload from the evaluator's measured global operating point (`best_f1_conf`)
 * and per-class precision/recall measured at that exact confidence (`GET /api/models/{id}/eval`).
 *
 * Returns `null` when no classes with ground-truth instances were evaluated or when the server did not report
 * a measured `best_f1_conf` (never fabricates confidence thresholds from PR curves or confusion-matrix cutoffs).
 */
export function buildModelThresholds(
  model: Pick<Model, 'classes' | 'metrics'>,
  ev: Evaluation,
): ComputedModelCalibration | null {
  const measuredConf = isUnitFinite(ev.bestF1Conf)
    ? ev.bestF1Conf
    : isUnitFinite(model.metrics?.best_f1_conf)
      ? model.metrics.best_f1_conf
      : null;
  if (measuredConf === null) return null;

  // Deduplicate by canonical non-negative integer class ID, sorted ascending (required by `ThresholdCalibration::validate`).
  const byClass = new Map<number, (typeof ev.perClass)[number]>();
  for (const c of ev.perClass) {
    if (!Number.isInteger(c.class) || c.class < 0 || !Number.isFinite(c.instances) || c.instances <= 0) continue;
    if (!byClass.has(c.class)) byClass.set(c.class, c);
  }
  const evaluated = [...byClass.values()].sort((a, b) => a.class - b.class);
  if (evaluated.length === 0) return null;

  const globalConf = roundConf(roundGrid(measuredConf));
  const nEval = evaluated.length;
  const clamp01 = (v: number) => (Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0);

  const perClass: ModelClassThreshold[] = [];
  let pSum = 0;
  let rSum = 0;
  let f1Sum = 0;

  for (const c of evaluated) {
    const p = clamp01(c.precision);
    const r = clamp01(c.recall);
    const f1 = f1Of(p, r);
    const truths = Math.max(0, Math.round(c.instances));
    // `/eval` reports precision and recall at `best_f1_conf` (not the full 0.001-floor prediction count);
    // derive the prediction count at `best_f1_conf` from `tp / precision` instead of mixing in the separate
    // confusion-matrix cutoff (`confusion_conf = 0.25`).
    const tp = r * truths;
    const preds = p > 0 && tp > 0 ? Math.max(Math.round(tp), Math.round(tp / p)) : 0;

    pSum += p;
    rSum += r;
    f1Sum += f1;

    perClass.push({
      class: c.class,
      name: model.classes[c.class] ?? `class ${c.class}`,
      best_conf: globalConf,
      best_f1: roundMetric(f1),
      precision: roundMetric(p),
      recall: roundMetric(r),
      truths,
      preds,
    });
  }

  const globalPrecision = roundMetric(pSum / nEval);
  const globalRecall = roundMetric(rSum / nEval);
  const globalF1 = roundMetric(f1Sum / nEval);

  return {
    payload: {
      global: {
        best_conf: globalConf,
        best_f1: globalF1,
        precision: globalPrecision,
        recall: globalRecall,
      },
      macro_f1: globalF1,
      per_class: perClass,
    },
    perClassConf: {},
    hasMeasuredPerClass: false,
  };
}

/** Downloads `calibration.payload` as `<model-name>-thresholds.json` ready for `folo predict --thresholds` or Playground. */
export function downloadModelThresholds(modelName: string, calibration: ComputedModelCalibration): boolean {
  const filename = safeFilename(`${modelName || 'model'}-thresholds`, 'json');
  return downloadText(toJson(calibration.payload), filename, 'application/json;charset=utf-8', 'model-thresholds');
}
