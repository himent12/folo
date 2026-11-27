// Per-class confidence threshold helpers for Playground: normalization, class-name remapping across models
// (Compare A/B and model switches), and JSON import/export compatible with `folo-vision`'s `ThresholdCalibration`
// (`{ global, per_class: [{ class, best_conf, ... }] }`) as well as simple `{ "0": 0.4 }` / `{ "person": 0.4 }` maps.

export type PerClassConfMap = Record<number, number>;

const CANONICAL_INT = /^(?:0|[1-9]\d*)$/;

/** Rounds a finite confidence in `[0, 1]` to at most 4 decimal places (preserving 0.001 calibration grid steps). */
export function roundConf(v: number): number {
  const clamped = Math.max(0, Math.min(1, v));
  return Math.round(clamped * 10_000) / 10_000;
}

/**
 * Formats a confidence threshold for badges, tooltips, and status copy: uses 2 fixed decimals for standard 0.01-step
 * values (`0.40`, `0.25`) while preserving finer 3-4 decimal calibration steps (`0.005`, `0.375`, `0.0025`).
 */
export function formatConfThreshold(v: number): string {
  const r = roundConf(Number.isFinite(v) ? v : 0);
  const two = Math.round(r * 100) / 100;
  if (Math.abs(r - two) <= 1e-6) return two.toFixed(2);
  return String(r);
}

/** Keeps only canonical non-negative integer class IDs with finite `[0, 1]` thresholds. */
export function normalizePerClassConf(map: Readonly<Record<number, number>> | null | undefined): PerClassConfMap {
  if (!map) return {};
  const out: PerClassConfMap = {};
  for (const [k, v] of Object.entries(map)) {
    if (!CANONICAL_INT.test(k)) continue;
    const id = Number(k);
    if (!Number.isSafeInteger(id) || id > 0xffff_ffff) continue;
    if (typeof v !== 'number' || !Number.isFinite(v) || v < 0 || v > 1) continue;
    out[id] = roundConf(v);
  }
  return out;
}

/** Number of active per-class confidence overrides. */
export function perClassConfCount(map: Readonly<Record<number, number>> | null | undefined): number {
  return map ? Object.keys(map).length : 0;
}

/** Whether at least one per-class confidence override is active. */
export function hasPerClassConf(map: Readonly<Record<number, number>> | null | undefined): boolean {
  return perClassConfCount(map) > 0;
}

/** Stable sorted key for reactive effects and change detection. */
export function perClassConfKey(map: Readonly<Record<number, number>> | null | undefined): string {
  if (!map) return '';
  const entries = Object.entries(map)
    .map(([k, v]) => [Number(k), v] as const)
    .filter(([k, v]) => Number.isInteger(k) && k >= 0 && Number.isFinite(v))
    .sort((a, b) => a[0] - b[0]);
  return entries.map(([k, v]) => `${k}:${v}`).join(',');
}

/**
 * Maps per-class confidence overrides from one model's class indices onto another model's class indices by class
 * name (used when switching models or applying overrides to Model B in Compare A/B).
 */
export function mapPerClassConfToClasses(
  map: Readonly<Record<number, number>> | null | undefined,
  fromClasses: readonly string[],
  toClasses: readonly string[],
): PerClassConfMap {
  if (!map) return {};
  const out: PerClassConfMap = {};
  const sameOrder =
    fromClasses.length === toClasses.length && fromClasses.every((name, idx) => name === toClasses[idx]);
  if (sameOrder || (fromClasses.length === 0 && toClasses.length === 0)) {
    return normalizePerClassConf(map);
  }
  const toByName = new Map<string, number>();
  toClasses.forEach((name, idx) => {
    if (!toByName.has(name)) toByName.set(name, idx);
  });
  for (const [k, v] of Object.entries(map)) {
    if (!CANONICAL_INT.test(k)) continue;
    const fromIdx = Number(k);
    if (!Number.isFinite(v) || v < 0 || v > 1) continue;
    const name = fromClasses[fromIdx];
    if (name !== undefined) {
      const targetIdx = toByName.get(name);
      if (targetIdx !== undefined) out[targetIdx] = roundConf(v);
    }
  }
  return out;
}

/** Returns a human-readable record keyed by class name (for JSON exports alongside numeric `per_class_conf`). */
export function perClassConfForExport(
  map: Readonly<Record<number, number>> | null | undefined,
): Record<string, number> | undefined {
  const norm = normalizePerClassConf(map);
  const keys = Object.keys(norm)
    .map(Number)
    .sort((a, b) => a - b);
  if (!keys.length) return undefined;
  const out: Record<string, number> = {};
  for (const id of keys) out[String(id)] = norm[id]!;
  return out;
}

export interface ThresholdCalibrationExport {
  /** Threshold-only export schema (round-trips cleanly in Playground; omits evaluation metrics like F1/precision/recall). */
  schema: 'folo.thresholds.v1';
  global: { best_conf: number };
  per_class: { class: number; name?: string; best_conf: number }[];
}

/** Exports global and per-class thresholds in threshold-only JSON shape (`folo.thresholds.v1`, round-tripping in Playground). */
export function exportThresholdsJson(
  globalConf: number,
  perClassConf: Readonly<Record<number, number>>,
  classes: readonly string[],
): ThresholdCalibrationExport {
  const norm = normalizePerClassConf(perClassConf);
  const per_class = Object.keys(norm)
    .map(Number)
    .sort((a, b) => a - b)
    .map((id) => ({
      class: id,
      ...(classes[id] !== undefined ? { name: classes[id] } : {}),
      best_conf: norm[id]!,
    }));
  return {
    schema: 'folo.thresholds.v1',
    global: { best_conf: roundConf(globalConf) },
    per_class,
  };
}

export type ParseThresholdsResult =
  | {
      ok: true;
      /** Global confidence threshold in `[0, 1]` when supplied by the JSON, else `null`. */
      globalConf: number | null;
      perClassConf: PerClassConfMap;
      /** True when parsed from a calibrated `ThresholdCalibration` object (`score >= conf` inclusive semantics). */
      confInclusive: boolean;
      appliedCount: number;
      warnings: string[];
    }
  | {
      ok: false;
      error: string;
    };

function isFiniteUnit(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1;
}

function resolveClassKey(rawKey: string, classes: readonly string[]): { id: number } | { unknown: string } | { ambiguous: string } {
  if (CANONICAL_INT.test(rawKey)) {
    const id = Number(rawKey);
    if (Number.isSafeInteger(id) && id <= 0xffff_ffff) return { id };
  }
  const exact = classes.indexOf(rawKey);
  if (exact >= 0) return { id: exact };
  const needle = rawKey.trim().toLowerCase();
  if (needle) {
    const matches: number[] = [];
    classes.forEach((name, idx) => {
      if (name.trim().toLowerCase() === needle) matches.push(idx);
    });
    if (matches.length === 1) return { id: matches[0]! };
    if (matches.length > 1) return { ambiguous: rawKey };
  }
  return { unknown: rawKey };
}

/**
 * Parses per-class threshold JSON in any of these formats:
 * 1. `folo-vision`'s `ThresholdCalibration` (`{ global: { best_conf }, per_class: [{ class, best_conf }] }`,
 *    or wrapped under `{ thresholds: { ... } }`) and Playground's `folo.thresholds.v1` export,
 * 2. A `folo val --json` record carrying top-level `best_f1_conf` (applies global threshold with a note when
 *    `per_class` contains evaluation metrics without per-class `best_conf`), OR
 * 3. A simple object map keyed by class ID (`{ "0": 0.4 }`) or class name (`{ "person": 0.4 }`).
 */
export function parseThresholdsJson(raw: string, classes: readonly string[]): ParseThresholdsResult {
  const trimmed = raw.trim();
  if (!trimmed) return { ok: false, error: 'Paste or upload a JSON threshold file.' };
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch (err) {
    return { ok: false, error: `Invalid JSON: ${err instanceof Error ? err.message : 'parse error'}` };
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { ok: false, error: 'Expected a JSON object (ThresholdCalibration or { "class": conf } map).' };
  }

  let root = parsed as Record<string, unknown>;
  // Unwrap `{ thresholds: { global, per_class } }` when present.
  if (
    root.thresholds &&
    typeof root.thresholds === 'object' &&
    !Array.isArray(root.thresholds) &&
    ('per_class' in (root.thresholds as Record<string, unknown>) || 'global' in (root.thresholds as Record<string, unknown>))
  ) {
    root = root.thresholds as Record<string, unknown>;
  }

  const warnings: string[] = [];

  // Format 1b: `folo val --json` record with top-level `best_f1_conf` and metrics-only `per_class` (no `best_conf`).
  const isValMetricsOnly =
    typeof root.best_f1_conf === 'number' &&
    !('global' in root) &&
    Array.isArray(root.per_class) &&
    root.per_class.every(
      (e) =>
        e &&
        typeof e === 'object' &&
        !('best_conf' in (e as Record<string, unknown>)) &&
        !('conf' in (e as Record<string, unknown>)),
    );
  if (isValMetricsOnly) {
    if (!isFiniteUnit(root.best_f1_conf)) {
      return { ok: false, error: `best_f1_conf must be a finite number in [0, 1], got ${String(root.best_f1_conf)}.` };
    }
    warnings.push('No per-class best_conf entries found; applied global best_f1_conf only.');
    return {
      ok: true,
      globalConf: roundConf(root.best_f1_conf),
      perClassConf: {},
      confInclusive: true,
      appliedCount: 0,
      warnings,
    };
  }

  // Format 1: ThresholdCalibration (`{ global, per_class }`)
  const isCalibrationShape =
    'per_class' in root ||
    ('global' in root && typeof root.global === 'object' && root.global !== null && !Array.isArray(root.global));
  if (isCalibrationShape) {
    let globalConf: number | null = null;
    if ('global' in root && root.global !== undefined && root.global !== null) {
      if (typeof root.global === 'number') {
        if (!isFiniteUnit(root.global)) {
          return { ok: false, error: `global threshold must be a finite number in [0, 1], got ${root.global}.` };
        }
        globalConf = roundConf(root.global);
      } else if (typeof root.global === 'object' && !Array.isArray(root.global)) {
        const g = root.global as Record<string, unknown>;
        const gv = 'best_conf' in g ? g.best_conf : g.conf;
        if (gv !== undefined && gv !== null) {
          if (!isFiniteUnit(gv)) {
            return { ok: false, error: `global.best_conf must be a finite number in [0, 1], got ${String(gv)}.` };
          }
          globalConf = roundConf(gv);
        }
      } else {
        return { ok: false, error: 'global must be an object with best_conf or a number in [0, 1].' };
      }
    }

    const perClassConf: PerClassConfMap = {};
    const rawPerClass = root.per_class;
    if (rawPerClass !== undefined && rawPerClass !== null) {
      if (!Array.isArray(rawPerClass)) {
        return { ok: false, error: 'per_class must be an array of { class, best_conf } objects.' };
      }
      const seen = new Set<number>();
      for (let i = 0; i < rawPerClass.length; i++) {
        const entry = rawPerClass[i];
        if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
          return { ok: false, error: `per_class[${i}] must be an object.` };
        }
        const rec = entry as Record<string, unknown>;
        let classId: number;
        if (typeof rec.class === 'number') {
          if (!Number.isInteger(rec.class) || rec.class < 0 || rec.class > 0xffff_ffff) {
            return { ok: false, error: `per_class[${i}].class must be a non-negative integer.` };
          }
          classId = rec.class;
        } else if (typeof rec.class === 'string') {
          const resolved = resolveClassKey(rec.class, classes);
          if ('unknown' in resolved) {
            warnings.push(`Skipped unknown class "${resolved.unknown}" in per_class[${i}].`);
            continue;
          }
          if ('ambiguous' in resolved) {
            return { ok: false, error: `Class name "${resolved.ambiguous}" is ambiguous.` };
          }
          classId = resolved.id;
        } else {
          return { ok: false, error: `per_class[${i}].class is required.` };
        }

        const rawVal = 'best_conf' in rec ? rec.best_conf : rec.conf;
        if (!isFiniteUnit(rawVal)) {
          return { ok: false, error: `per_class[${i}].best_conf must be a finite number in [0, 1].` };
        }
        if (seen.has(classId)) {
          return { ok: false, error: `Duplicate per_class entry for class ${classId}.` };
        }
        seen.add(classId);
        perClassConf[classId] = roundConf(rawVal);
      }
    }

    return {
      ok: true,
      globalConf,
      perClassConf,
      confInclusive: true,
      appliedCount: Object.keys(perClassConf).length,
      warnings,
    };
  }

  // Format 2: Simple map (`{ "0": 0.4, "person": 0.65 }`)
  const entries = Object.entries(root);
  const perClassConf: PerClassConfMap = {};
  const seen = new Set<number>();
  const unknown: string[] = [];

  for (const [k, v] of entries) {
    if (!isFiniteUnit(v)) {
      return { ok: false, error: `Threshold for "${k}" must be a finite number in [0, 1].` };
    }
    // Reject malformed numeric-looking keys like "-1", "+0", "01", "1.5" when they don't match a real class name.
    if (!classes.includes(k) && /^[+-]?\d+(?:\.\d+)?$/.test(k.trim()) && !CANONICAL_INT.test(k)) {
      return { ok: false, error: `Invalid class ID "${k}": must be a non-negative integer or class name.` };
    }
    const resolved = resolveClassKey(k, classes);
    if ('unknown' in resolved) {
      unknown.push(resolved.unknown);
      continue;
    }
    if ('ambiguous' in resolved) {
      return { ok: false, error: `Class name "${resolved.ambiguous}" matches multiple classes; use numeric class IDs.` };
    }
    if (seen.has(resolved.id)) {
      return { ok: false, error: `Duplicate threshold entry resolving to class ${resolved.id} ("${k}").` };
    }
    seen.add(resolved.id);
    perClassConf[resolved.id] = roundConf(v);
  }

  if (unknown.length > 0 && Object.keys(perClassConf).length === 0) {
    return { ok: false, error: `No matching classes found in this model (unknown: ${unknown.slice(0, 4).join(', ')}).` };
  }
  if (unknown.length > 0) {
    warnings.push(`Skipped unknown ${unknown.length === 1 ? 'class' : 'classes'}: ${unknown.slice(0, 4).join(', ')}${unknown.length > 4 ? '…' : ''}.`);
  }

  return {
    ok: true,
    globalConf: null,
    perClassConf,
    confInclusive: true,
    appliedCount: Object.keys(perClassConf).length,
    warnings,
  };
}
