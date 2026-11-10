// An external contestant's predictions (docs/API.md, Vision benchmarks → entries, `external`), checked in the browser
// before they are sent, with the server's rules: what is wrong is named at the first record that breaks a rule, as the
// server's `400 invalid_body` does, so a 60 MiB file is never uploaded to be refused.

import type { BenchPlan, BenchPrediction } from '../api/types';
import { MAX_SEVERITY } from './rules';

/** What the predictions hold, once they pass. */
export interface PredictionSummary {
  /** Unchanged images answered (every image of the benchmark, when the image list is known). */
  images: number;
  /** The plan's conditions answered, in plan order. */
  conditions: string[];
  /** The plan's conditions left out (robustness is then over the others, and the entry is partial). */
  missing: string[];
  /** Boxes over all records. */
  boxes: number;
  /** Some box carries a `score`: the model gives confidences. */
  scores: boolean;
}

export type PredictionCheck = { ok: true; predictions: BenchPrediction[]; summary: PredictionSummary } | { ok: false; error: string };

/** What the benchmark is, for the checks: its image ids (null: not loaded, so membership is not checked), classes and plan. */
export interface PredictionContext {
  imageIds: readonly string[] | null;
  classes: number;
  plan: BenchPlan;
  /** Every condition the catalog knows (a record under one outside the plan is refused as unplanned, not unknown). */
  catalog: readonly string[];
  /** Display names of conditions, for the messages. */
  names?: Record<string, string>;
}

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/** The records of a predictions file: `{ "predictions": [...] }` or a bare array. */
export function parsePredictions(text: string): { records: unknown[] } | { error: string } {
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch (e) {
    const why = e instanceof Error ? e.message.replace(/^JSON\.parse: /, '') : 'it does not parse';
    return { error: `The file is not JSON: ${why}.` };
  }
  if (Array.isArray(json)) return { records: json };
  if (isObj(json) && Array.isArray(json.predictions)) return { records: json.predictions };
  return { error: 'Expected an object with a "predictions" array, or an array of records.' };
}

/** Checks the records against the benchmark; the first problem found, or what they hold. */
export function checkPredictions(records: readonly unknown[], ctx: PredictionContext): PredictionCheck {
  const name = (id: string) => ctx.names?.[id] ?? id;
  const at = (i: number, msg: string): PredictionCheck => ({ ok: false, error: `Record ${i + 1}: ${msg}` });
  if (!records.length) return { ok: false, error: 'The file has no records.' };
  const known = ctx.imageIds ? new Set(ctx.imageIds) : null;
  const planned = new Map(ctx.plan.conditions.map((c) => [c.condition, c.severities]));
  const seen = new Set<string>();
  /** Per condition and severity, the images answered. */
  const under = new Map<string, Map<number, Set<string>>>();
  const clean = new Set<string>();
  const out: BenchPrediction[] = [];
  let boxes = 0;
  let scores = false;
  for (let i = 0; i < records.length; i++) {
    const r = records[i];
    if (!isObj(r)) return at(i, 'is not an object.');
    if (typeof r.image !== 'string' || !r.image) return at(i, '"image" must be an image id (a string).');
    if (known && !known.has(r.image)) return at(i, `the benchmark has no image "${r.image}".`);
    const cond = r.condition === undefined || r.condition === null ? null : r.condition;
    const sev = r.severity === undefined || r.severity === null ? null : r.severity;
    if (cond !== null && typeof cond !== 'string') return at(i, '"condition" must be a condition id or null.');
    if (cond === null && sev !== null) return at(i, '"severity" without a "condition".');
    if (cond !== null) {
      if (!ctx.catalog.includes(cond)) return at(i, `unknown condition "${cond}".`);
      if (!planned.has(cond)) return at(i, `${name(cond)} is not in this benchmark's plan.`);
      if (sev === null) return at(i, `${name(cond)} needs a "severity".`);
      if (typeof sev !== 'number' || !Number.isInteger(sev) || sev < 1 || sev > MAX_SEVERITY) return at(i, `"severity" must be a whole number from 1 to ${MAX_SEVERITY}.`);
      if (!planned.get(cond)!.includes(sev)) return at(i, `${name(cond)} runs at severities ${planned.get(cond)!.join(', ')}, not ${sev}.`);
    }
    const key = `${r.image}|${cond ?? ''}|${sev ?? ''}`;
    if (seen.has(key)) return at(i, `image "${r.image}"${cond ? ` under ${name(cond)} ${sev}` : ''} is answered twice.`);
    seen.add(key);
    if (!Array.isArray(r.boxes)) return at(i, '"boxes" must be an array (empty when nothing was found).');
    const bs: BenchPrediction['boxes'] = [];
    for (let k = 0; k < r.boxes.length; k++) {
      const b: unknown = r.boxes[k];
      if (!isObj(b)) return at(i, `box ${k + 1} is not an object.`);
      if (!isNum(b.class) || !Number.isInteger(b.class) || b.class < 0 || b.class >= ctx.classes)
        return at(i, `box ${k + 1}: "class" must be a class index from 0 to ${ctx.classes - 1}.`);
      if (![b.x1, b.y1, b.x2, b.y2].every(isNum)) return at(i, `box ${k + 1}: "x1", "y1", "x2" and "y2" must be numbers (pixels).`);
      if (b.score !== undefined) {
        if (!isNum(b.score) || b.score < 0 || b.score > 1) return at(i, `box ${k + 1}: "score" must be between 0 and 1.`);
        scores = true;
      }
      bs.push({ class: b.class, x1: b.x1 as number, y1: b.y1 as number, x2: b.x2 as number, y2: b.y2 as number, ...(b.score !== undefined ? { score: b.score as number } : {}) });
    }
    boxes += bs.length;
    if (cond === null) clean.add(r.image);
    else {
      const m = under.get(cond) ?? new Map<number, Set<string>>();
      under.set(cond, m);
      const s = m.get(sev as number) ?? new Set<string>();
      m.set(sev as number, s);
      s.add(r.image);
    }
    out.push({ image: r.image, ...(cond ? { condition: cond, severity: sev as number } : {}), boxes: bs });
  }
  // Every unchanged image answered for; a condition answered whole (every image, every planned severity) or not at all.
  const images = ctx.imageIds ?? [...clean];
  const missingImage = images.find((id) => !clean.has(id));
  if (missingImage !== undefined) {
    const n = images.filter((id) => !clean.has(id)).length;
    return { ok: false, error: `Image "${missingImage}" has no record without a condition${n > 1 ? ` (and ${n - 1} more)` : ''}: every unchanged image must be answered, with empty "boxes" when nothing was found.` };
  }
  for (const [cond, bySev] of under) {
    for (const s of planned.get(cond)!) {
      const got = bySev.get(s);
      const gap = images.find((id) => !got?.has(id));
      if (gap !== undefined)
        return { ok: false, error: `${name(cond)} is answered in part: image "${gap}" at severity ${s} is missing. A condition is answered for every image at every planned severity, or left out.` };
    }
  }
  const conditions = ctx.plan.conditions.map((c) => c.condition).filter((c) => under.has(c));
  return {
    ok: true,
    predictions: out,
    summary: { images: images.length, conditions, missing: ctx.plan.conditions.map((c) => c.condition).filter((c) => !under.has(c)), boxes, scores },
  };
}
