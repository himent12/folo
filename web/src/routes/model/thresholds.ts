// Pure calibrated per-class F1 threshold builder for Model Detail (`#/models/<id>`): derives a
// `ThresholdCalibration` JSON payload compatible with both `folo predict --thresholds`
// (`crates/folo-vision/src/eval/thresholds.rs`) and Playground's `parseThresholdsJson`, plus a browser
// download helper and per-class confidence map for one-click Playground handoff.
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
  /** JSON payload matching `folo_vision::ThresholdCalibration`. */
  payload: ModelThresholdCalibration;
  /** Per-class confidence map (`classId -> best_conf`) ready for Playground handoff. */
  perClassConf: PerClassConfMap;
  /** Number of evaluated classes whose calibrated confidence differs from the global operating point. */
  distinctClassCount: number;
}

const DEFAULT_GLOBAL_CONF = 0.25;
const MIN_EST_CONF = 0.05;
const MAX_EST_CONF = 0.95;

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
 * Finds the best F1 point on a class's PR curve (`x = recall`, `y = precision`).
 * On equal F1, prefers higher precision / lower recall (conservative confidence threshold, matching
 * `folo-vision`'s tie-breaking rule where the highest confidence wins).
 */
function bestCurvePoint(
  curve: readonly { x: number; y: number }[],
): { recall: number; precision: number; f1: number; maxRecall: number } {
  let bestF1 = 0;
  let bestP = 0;
  let bestR = 0;
  let maxRecall = 0;
  for (const pt of curve) {
    const r = Math.max(0, Math.min(1, pt.x));
    const p = Math.max(0, Math.min(1, pt.y));
    if (p > 0 && r > maxRecall) maxRecall = r;
    const f1 = f1Of(p, r);
    if (f1 > bestF1 + 1e-6 || (Math.abs(f1 - bestF1) <= 1e-6 && f1 > 0 && (p > bestP + 1e-6 || r < bestR))) {
      bestF1 = f1;
      bestP = p;
      bestR = r;
    }
  }
  return { recall: bestR, precision: bestP, f1: bestF1, maxRecall };
}

/**
 * Estimates a class's F1-optimal confidence threshold from its PR curve peak recall `targetRecall`,
 * anchored at the evaluator's global operating point `(globalConf, globalRecall)` and (when reported)
 * the confusion matrix operating point `(matrixConf, matrixRecall)`.
 */
function estimateClassConf(
  targetRecall: number,
  globalConf: number,
  globalRecall: number,
  maxRecall: number,
  matrixAnchor: { conf: number; recall: number } | null,
): number {
  if (Math.abs(targetRecall - globalRecall) <= 0.005) {
    return roundGrid(globalConf);
  }
  // If we have a valid second operating-point anchor from the confusion matrix with monotonic recall(conf)
  // that brackets `targetRecall`, linearly interpolate directly between the two measured anchors.
  if (
    matrixAnchor &&
    Math.abs(matrixAnchor.conf - globalConf) >= 0.01 &&
    Math.abs(matrixAnchor.recall - globalRecall) >= 0.01
  ) {
    const hiConf = matrixAnchor.conf > globalConf ? matrixAnchor : { conf: globalConf, recall: globalRecall };
    const loConf = matrixAnchor.conf > globalConf ? { conf: globalConf, recall: globalRecall } : matrixAnchor;
    // Higher confidence must correspond to equal-or-lower recall to interpolate monotonically.
    if (loConf.recall > hiConf.recall && targetRecall <= loConf.recall && targetRecall >= hiConf.recall) {
      const t = (loConf.recall - targetRecall) / (loConf.recall - hiConf.recall);
      return roundGrid(loConf.conf + t * (hiConf.conf - loConf.conf));
    }
  }

  if (targetRecall < globalRecall && globalRecall > 0) {
    // Peak F1 requires higher precision / lower recall -> raise confidence above globalConf.
    const frac = (globalRecall - targetRecall) / globalRecall;
    const ceiling = Math.max(globalConf, MAX_EST_CONF);
    return roundGrid(globalConf + frac * (ceiling - globalConf));
  }
  if (targetRecall > globalRecall && maxRecall > globalRecall) {
    // Peak F1 requires higher recall -> lower confidence below globalConf.
    const frac = (targetRecall - globalRecall) / (maxRecall - globalRecall);
    const floor = Math.min(globalConf, MIN_EST_CONF);
    return roundGrid(globalConf - frac * (globalConf - floor));
  }
  return roundGrid(globalConf);
}

/**
 * Builds calibrated global and per-class F1 thresholds from a model's `/eval` payload.
 * Returns `null` when no classes with ground-truth instances were evaluated.
 */
export function buildModelThresholds(
  model: Pick<Model, 'classes' | 'metrics'>,
  ev: Evaluation,
): ComputedModelCalibration | null {
  const evaluated = ev.perClass
    .filter((c) => c.instances > 0)
    .sort((a, b) => a.class - b.class);
  if (evaluated.length === 0) return null;

  const rawGlobalConf =
    ev.bestF1Conf !== null && Number.isFinite(ev.bestF1Conf)
      ? ev.bestF1Conf
      : model.metrics?.best_f1_conf != null && Number.isFinite(model.metrics.best_f1_conf)
        ? model.metrics.best_f1_conf
        : ev.matrixAt?.conf ?? DEFAULT_GLOBAL_CONF;
  const globalConf = roundGrid(rawGlobalConf);

  const nEval = evaluated.length;
  const globalPrecision = roundMetric(evaluated.reduce((s, c) => s + c.precision, 0) / nEval);
  const globalRecall = roundMetric(evaluated.reduce((s, c) => s + c.recall, 0) / nEval);
  const globalF1 = roundMetric(evaluated.reduce((s, c) => s + f1Of(c.precision, c.recall), 0) / nEval);

  const matrixConf = ev.matrixAt && Number.isFinite(ev.matrixAt.conf) ? ev.matrixAt.conf : null;
  const perClass: ModelClassThreshold[] = [];
  const perClassConf: PerClassConfMap = {};
  let macroF1Sum = 0;
  let distinctClassCount = 0;

  for (const c of evaluated) {
    const curve = ev.curves[c.class] ?? [];
    const globalClassF1 = f1Of(c.precision, c.recall);
    const curveBest = bestCurvePoint(curve);
    const useCurve = curveBest.f1 > globalClassF1 + 1e-4;

    const bestF1 = useCurve ? curveBest.f1 : globalClassF1;
    const bestP = useCurve ? curveBest.precision : c.precision;
    const bestR = useCurve ? curveBest.recall : c.recall;

    // Total predictions assigned to class `c` in the confusion matrix (column sum over all rows including background).
    let preds = 0;
    for (let r = 0; r < ev.confusion.length; r++) {
      const cell = ev.confusion[r]?.[c.class] ?? 0;
      if (Number.isFinite(cell) && cell > 0) preds += Math.round(cell);
    }

    const matrixTp = ev.confusion[c.class]?.[c.class] ?? 0;
    const matrixAnchor =
      matrixConf !== null && c.instances > 0
        ? { conf: matrixConf, recall: Math.max(0, Math.min(1, matrixTp / c.instances)) }
        : null;

    const bestConf =
      useCurve && bestF1 > 0
        ? estimateClassConf(curveBest.recall, globalConf, c.recall, curveBest.maxRecall, matrixAnchor)
        : globalConf;

    const roundedConf = roundConf(bestConf);
    if (Math.abs(roundedConf - globalConf) > 1e-6) distinctClassCount++;
    perClassConf[c.class] = roundedConf;
    macroF1Sum += bestF1;

    perClass.push({
      class: c.class,
      name: model.classes[c.class] ?? `class ${c.class}`,
      best_conf: roundedConf,
      best_f1: roundMetric(bestF1),
      precision: roundMetric(bestP),
      recall: roundMetric(bestR),
      truths: Math.max(0, Math.round(c.instances)),
      preds,
    });
  }

  const macroF1 = roundMetric(Math.max(globalF1, macroF1Sum / nEval));

  return {
    payload: {
      global: {
        best_conf: globalConf,
        best_f1: globalF1,
        precision: globalPrecision,
        recall: globalRecall,
      },
      macro_f1: macroF1,
      per_class: perClass,
    },
    perClassConf,
    distinctClassCount,
  };
}

/** Downloads `calibration.payload` as `<model-name>-thresholds.json` ready for `folo predict --thresholds` or Playground. */
export function downloadModelThresholds(modelName: string, calibration: ComputedModelCalibration): boolean {
  const filename = safeFilename(`${modelName || 'model'}-thresholds`, 'json');
  return downloadText(toJson(calibration.payload), filename, 'application/json;charset=utf-8', 'model-thresholds');
}
