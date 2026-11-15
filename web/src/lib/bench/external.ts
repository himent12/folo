// An external contestant's predictions (docs/API.md, Vision benchmarks → entries, `external`), checked in the browser
// before they are sent, with the server's rules: what is wrong is named at the first record that breaks a rule, as the
// server's `400 invalid_body` does, so a 60 MiB file is never uploaded to be refused. A file whose request would be
// large goes as an open entry filled in parts (`partsOf`, `sendExternal`): one request is capped at 64 MiB, and a
// real run (300 images × 20 passes, up to 300 boxes an image) is far over that.

import { ApiError } from '../api/client';
import type { BenchEntry, BenchPlan, BenchPrediction, StartBenchEntry } from '../api/types';
import { MAX_SEVERITY } from './rules';

/** The most boxes one record may carry (the server's rule). */
export const MAX_BOXES_PER_RECORD = 1000;

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
  /** Whether the entry's scores count (the contestant gives confidences), as `contestant.scored`: default true. */
  scored?: boolean;
}

/** A record's (image, condition, severity): what may be answered once. */
export const predictionKey = (image: string, condition: string | null | undefined, severity: number | null | undefined) => `${image}|${condition ?? ''}|${severity ?? ''}`;

const num = (n: number) => n.toLocaleString('en-US');

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/** A value as the server holds it: a number as a 32-bit float, zero's sign kept; anything else as its JSON. */
const f32 = (v: unknown) => {
  if (typeof v !== 'number') return `${JSON.stringify(v)}`;
  const f = Math.fround(v);
  return Object.is(f, -0) ? '-0' : String(f);
};

/**
 * A record's boxes as the server compares a repeated record (crates/folo-server/src/bench/external.rs, `content`): in
 * order, each box's class, score and corners as 32-bit floats, a missing score as 1 (its default), and no score at all
 * when the entry's scores don't count. The order of a box's fields, and any other field, make no difference.
 */
export function boxesKey(boxes: unknown, scored = true): string {
  if (!Array.isArray(boxes)) return `${JSON.stringify(boxes)}`;
  const one = (b: unknown) => (isObj(b) ? [b.class, scored ? (b.score === undefined ? 1 : b.score) : 0, b.x1, b.y1, b.x2, b.y2].map(f32).join(',') : JSON.stringify(b));
  return boxes.map(one).join(';');
}

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

/**
 * Checks the records against the benchmark; the first problem found, or what they hold. The records that pass are
 * handed back as they are (no copy: a 200 MB file's records are held once). With `part`, they are one part of an open
 * entry: `seen` holds the (image, condition, severity) keys received before (a repeat is an error, as on the server)
 * and gets this part's added (pass a copy to keep a refused part from adding any), and the completeness rules wait
 * for the finish.
 */
export function checkPredictions(records: readonly unknown[], ctx: PredictionContext, o: { part?: boolean; seen?: Set<string> } = {}): PredictionCheck {
  const name = (id: string) => ctx.names?.[id] ?? id;
  const at = (i: number, msg: string): PredictionCheck => ({ ok: false, error: `Record ${i + 1}: ${msg}` });
  if (!records.length && !o.part) return { ok: false, error: 'The file has no records.' };
  const known = ctx.imageIds ? new Set(ctx.imageIds) : null;
  const planned = new Map(ctx.plan.conditions.map((c) => [c.condition, c.severities]));
  const seen = o.seen ?? new Set<string>();
  /** This call's records by key, as their boxes: what a repeat is compared with. */
  const here = new Map<string, string>();
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
    const key = predictionKey(r.image, cond as string | null, sev as number | null);
    // A record of these records repeated with the same boxes is taken once, as the server takes it (a repeat with other
    // boxes, or of one received in an earlier part, is refused). "The same" as the server reads them (`boxesKey`).
    const content = boxesKey(r.boxes, ctx.scored ?? true);
    if (here.get(key) === content) continue;
    here.set(key, content);
    if (seen.has(key)) return at(i, `image "${r.image}"${cond ? ` under ${name(cond)} ${sev}` : ''} is answered twice.`);
    seen.add(key);
    if (!Array.isArray(r.boxes)) return at(i, '"boxes" must be an array (empty when nothing was found).');
    if (r.boxes.length > MAX_BOXES_PER_RECORD) return at(i, `${num(r.boxes.length)} boxes: one record carries at most ${num(MAX_BOXES_PER_RECORD)}.`);
    for (let k = 0; k < r.boxes.length; k++) {
      const b: unknown = r.boxes[k];
      if (!isObj(b)) return at(i, `box ${k + 1} is not an object.`);
      if (!isNum(b.class) || !Number.isInteger(b.class) || b.class < 0 || b.class >= ctx.classes)
        return at(i, `box ${k + 1}: "class" must be a class index from 0 to ${ctx.classes - 1}.`);
      // Finite as the server reads them (32-bit floats): 1e39 is a number here and infinity there.
      if (![b.x1, b.y1, b.x2, b.y2].every((v) => isNum(v) && Number.isFinite(Math.fround(v)))) return at(i, `box ${k + 1}: "x1", "y1", "x2" and "y2" must be numbers (pixels).`);
      if (Math.fround(b.x2 as number) < Math.fround(b.x1 as number) || Math.fround(b.y2 as number) < Math.fround(b.y1 as number))
        return at(i, `box ${k + 1}: "x2" or "y2" is less than "x1" or "y1". A box is its corners in pixels, top-left (x1, y1) and bottom-right (x2, y2), not x, y, width, height.`);
      if (b.score !== undefined) {
        if (!isNum(b.score) || b.score < 0 || b.score > 1) return at(i, `box ${k + 1}: "score" must be between 0 and 1.`);
        scores = true;
      }
    }
    boxes += r.boxes.length;
    if (cond === null) clean.add(r.image);
    else {
      const m = under.get(cond) ?? new Map<number, Set<string>>();
      under.set(cond, m);
      const s = m.get(sev as number) ?? new Set<string>();
      m.set(sev as number, s);
      s.add(r.image);
    }
    // The record itself, checked: its fields are the ones the server reads.
    out.push(r as unknown as BenchPrediction);
  }
  if (o.part) return { ok: true, predictions: out, summary: { images: clean.size, conditions: [...under.keys()], missing: [], boxes, scores } };
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

// ---------- sending a large file in parts ----------

/** Over this much JSON, an external entry goes as an open entry in parts; each part stays under it too. */
export const PART_BYTES = 8 * 1024 * 1024;

/**
 * About how many bytes of JSON a record takes as sent: its image id and condition, and each box's six numbers (a
 * coordinate with its decimals, a score). An estimate on the high side, so a part never nears the server's 64 MiB.
 */
export function recordBytes(p: BenchPrediction): number {
  return 48 + p.image.length + (p.condition ? p.condition.length + 16 : 0) + p.boxes.length * 112;
}

/** The JSON a list of records would take, by `recordBytes`. */
export const estimateBytes = (ps: readonly BenchPrediction[]) => ps.reduce((n, p) => n + recordBytes(p), 32);

/**
 * The parts a large file is sent in: grouped by pass, the unchanged images first, then each condition of the plan at
 * each severity in plan order (any other condition after them), so a part that fails names one pass; a pass larger
 * than `maxBytes` is split further, in its records' order. No part is empty, and every record is in exactly one.
 */
export function partsOf(predictions: readonly BenchPrediction[], plan: BenchPlan, maxBytes = PART_BYTES): BenchPrediction[][] {
  const order = new Map<string, number>([['|', 0]]);
  for (const c of plan.conditions) for (const s of c.severities) order.set(`${c.condition}|${s}`, order.size);
  const passes = new Map<string, BenchPrediction[]>();
  for (const p of predictions) {
    const k = `${p.condition ?? ''}|${p.condition ? (p.severity ?? '') : ''}`;
    const list = passes.get(k) ?? [];
    passes.set(k, list);
    list.push(p);
  }
  const keys = [...passes.keys()].sort((a, b) => (order.get(a) ?? order.size) - (order.get(b) ?? order.size) || a.localeCompare(b));
  const out: BenchPrediction[][] = [];
  for (const k of keys) {
    let part: BenchPrediction[] = [];
    let size = 32;
    for (const p of passes.get(k)!) {
      const b = recordBytes(p);
      if (part.length && size + b > maxBytes) {
        out.push(part);
        part = [];
        size = 32;
      }
      part.push(p);
      size += b;
    }
    if (part.length) out.push(part);
  }
  return out;
}

/** What `sendExternal` asks of the API (`hub.api`'s methods, or fakes in a test). */
export interface ExternalSender {
  start: (body: StartBenchEntry) => Promise<BenchEntry>;
  send: (entryId: string, predictions: BenchPrediction[]) => Promise<BenchEntry>;
  finish: (entryId: string) => Promise<BenchEntry>;
  /** The entry as it is now (a finish whose answer was lost may have scored it). */
  get: (entryId: string) => Promise<BenchEntry>;
  stop: (entryId: string) => Promise<unknown>;
  remove: (entryId: string) => Promise<unknown>;
}

/** Where a send is: the records sent of all of them, and the open entry once it exists. */
export interface SendProgress {
  sent: number;
  total: number;
  entry: BenchEntry | null;
}

/**
 * A send that failed once its open entry existed: the error, and what it left on the server. `left` is null when the
 * entry was stopped and deleted (nothing is left), else a sentence saying exactly what is left and where.
 */
export class ExternalSendError extends Error {
  constructor(
    override readonly cause: unknown,
    readonly left: string | null,
  ) {
    super(cause instanceof Error ? cause.message : String(cause));
    this.name = cause instanceof Error ? cause.name : 'ExternalSendError';
  }
}

/** A failure worth one retry: no answer (the network, a timeout) or the server's own trouble, never a refusal. */
const retryable = (e: unknown) => !(e instanceof ApiError) || e.status === 0 || e.status >= 500;

/**
 * Sends an external entry: in one request when it is small (`maxBytes` of JSON or less), else as an open entry filled
 * in parts (./partsOf) and finished, telling `progress` after each part. A part whose answer was lost (the network, a
 * 5xx) is sent once more: the server takes a record it already has with the same boxes again (an idempotent retry); so
 * is the finish, and when the first one scored the entry, the entry is read and kept.
 * Any failure once the entry exists (a part refused, the finish refused, the network twice, `signal` aborted by the
 * person) stops the entry and deletes it, so nothing is left on the leaderboard; it throws an ExternalSendError that
 * says what is left when the stop or the delete failed too.
 */
export async function sendExternal(
  api: ExternalSender,
  o: { name: string; scored: boolean; predictions: BenchPrediction[]; plan: BenchPlan },
  progress: (p: SendProgress) => void = () => {},
  signal?: AbortSignal,
  maxBytes = PART_BYTES,
): Promise<BenchEntry> {
  const total = o.predictions.length;
  if (estimateBytes(o.predictions) <= maxBytes) {
    const entry = await api.start({ name: o.name, contestant: { kind: 'external', scored: o.scored, predictions: o.predictions } });
    progress({ sent: total, total, entry });
    return entry;
  }
  const parts = partsOf(o.predictions, o.plan, maxBytes);
  let entry = await api.start({ name: o.name, contestant: { kind: 'external', scored: o.scored, open: true, predictions: [] } });
  let sent = 0;
  progress({ sent, total, entry });
  const once = async <T>(call: () => Promise<T>): Promise<T> => {
    try {
      return await race(call(), signal);
    } catch (e) {
      if (signal?.aborted || !retryable(e)) throw e;
      return await race(call(), signal);
    }
  };
  try {
    for (const part of parts) {
      entry = await once(() => api.send(entry.id, part));
      sent += part.length;
      progress({ sent, total, entry });
    }
    let tries = 0;
    return await once(async () => {
      try {
        return await api.finish(entry.id);
      } catch (err) {
        // A finish sent again after its answer was lost finds the entry no longer open when the first one scored it.
        if (tries++ === 0 || !(err instanceof ApiError && err.code === 'bench_entry_not_open')) throw err;
        const now = await api.get(entry.id);
        if (now.status !== 'done') throw err;
        return now;
      }
    });
  } catch (e) {
    // Leave nothing behind: stop the open entry, then delete it; say exactly what is left if either fails.
    const name = `an unfinished entry “${o.name}”`;
    let left: string | null = null;
    try {
      await api.stop(entry.id);
      try {
        await api.remove(entry.id);
      } catch {
        left = `${name} (stopped) is still on the leaderboard: delete it there.`;
      }
    } catch {
      left = `${name} is still on the leaderboard: stop and delete it there.`;
    }
    throw new ExternalSendError(e, left);
  }
}

/** `p`, or the abort of `signal` if that comes first (a part in flight is not waited for once the person stops). */
function race<T>(p: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return p;
  const stopped = () => signal.reason ?? new DOMException('Stopped', 'AbortError');
  if (signal.aborted) return Promise.reject(stopped());
  return new Promise<T>((resolve, reject) => {
    const off = () => reject(stopped());
    signal.addEventListener('abort', off, { once: true });
    p.then(
      (v) => (signal.removeEventListener('abort', off), resolve(v)),
      (e) => (signal.removeEventListener('abort', off), reject(e)),
    );
  });
}
