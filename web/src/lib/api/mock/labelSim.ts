// Pure serialization, box-drafting simulation, usage accounting, and dataset materialization for mock agent labeling (`./label.ts`).
import { ApiError } from '../client';
import type { Box, LabelDatasetRequest, LabelImage, LabelJob, LabelUsage } from '../types';
import { inventBoxes, predictBoxes, round } from './boxes';
import { image, specs, type MockDatasetSpec } from './datasets';
import { iso, plural, PRICE, QUALITY, type Img, type Job, type Run } from './labelAgents';
import { hash } from './rng';

export function serializeImage(im: Img): LabelImage {
  const reviewed = im.reviewed && im.status === 'labeled';
  return { id: im.id, file: im.file, width: im.width, height: im.height, status: im.status, boxes: im.boxes, source: im.source, error: im.error, reviewed };
}

export function serializeJob(j: Job): LabelJob {
  // An image in an agent call counts as what it was before it (pending + labeled + failed == images).
  const counts = { images: j.images.length, pending: 0, labeled: 0, failed: 0, boxes: 0 };
  for (const im of j.images) {
    const s = im.status === 'labeling' ? (im.before?.status ?? 'pending') : im.status;
    if (s === 'labeled') {
      counts.labeled++;
      counts.boxes += im.boxes.length;
    } else if (s === 'failed') counts.failed++;
    else counts.pending++;
  }
  const r = j.run;
  return {
    id: j.id,
    name: j.name,
    classes: [...j.classes],
    instructions: j.instructions,
    created: iso(j.createdAt),
    status: !r || r.outcome ? 'idle' : r.stopAt !== null ? 'stopping' : 'running',
    counts,
    run: r
      ? {
          agent: r.agent,
          model: r.model,
          effort: r.effort,
          ...(r.fast ? { fast: true } : {}),
          started: iso(r.startedAt),
          finished: r.finishedAt === null ? null : iso(r.finishedAt),
          total: r.ids.length - r.left.size,
          done: r.done,
          failed: r.failed,
          outcome: r.outcome,
          error: r.error,
          ...(r.hint ? { hint: r.hint } : {}),
          usage: { ...r.usage },
        }
      : null,
    usage: { ...j.usage },
    dataset_id: j.datasetId,
  };
}

/** The agent's boxes for an image: its objects as a decent model finds them, in the job's classes. */
export function draftBoxes(j: Job, im: Img, r: Run): Box[] {
  const seed = hash(`${j.id}:${im.id}:${j.runs}`);
  const quality = Math.min(0.97, (r.model ? (QUALITY[r.model] ?? 0.85) : 0.85) + (r.effort === 'high' ? 0.04 : r.effort === 'medium' ? 0.02 : 0) + (r.note ? 0.03 : 0));
  let drawn;
  if ('spec' in im.from) {
    const s = im.from.spec;
    const truth = image(s, im.from.index).boxes.flatMap((b) => {
      const c = j.classes.findIndex((x) => x.toLowerCase() === (s.classes[b.class] ?? '').toLowerCase());
      return c < 0 ? [] : [{ ...b, class: c }];
    });
    drawn = predictBoxes(truth, im.width, im.height, quality, seed, j.classes.length).filter((b) => b.score >= 0.3);
  } else drawn = inventBoxes(im.width, im.height, seed, j.classes.length).filter((b) => b.score >= 0.5);
  return drawn.map(({ class: c, x1, y1, x2, y2 }) => ({ class: c, x1, y1, x2, y2 }));
}

/** What a CLI reports for one call: about 1000 tokens an image plus its fixed prompt (Codex's ~10k), cached after the first call. */
export function callUsage(j: Job, r: Run, k: number, images: number, boxes: number): LabelUsage {
  const fixed = r.agent === 'claude' ? 420 + Math.round(j.instructions.length / 4) : r.agent === 'antigravity' ? 1_200 + Math.round(j.instructions.length / 4) : 10_200;
  const input = fixed + images * 1012 + (hash(`${j.id}:${k}`) % 60);
  const cached = k === 0 ? 0 : Math.round(fixed * 0.86);
  const reasoning = { low: 0, medium: 260, high: 820 }[r.effort] * images;
  const output = 48 + boxes * 23 + reasoning;
  const price = r.agent === 'claude' ? PRICE[r.model ?? 'haiku'] : undefined;
  const cost = price ? round(((input - cached) * price[0] + cached * price[0] * 0.1 + output * price[1]) / 1e6, 6) : null;
  return { calls: 1, input_tokens: input, cached_input_tokens: cached, output_tokens: output, cost_usd: cost };
}

/** A run's figures as the server's last log line has them. */
export function runSummary(r: Run): string {
  const total = r.ids.length - r.left.size;
  const u = r.usage;
  return `${r.done} of ${total} ${plural(total, 'image')} labeled, ${r.failed} failed · ${u.calls} ${plural(u.calls, 'call')}, ${u.input_tokens} tokens in, ${u.output_tokens} out`;
}

/**
 * Builds the `MockDatasetSpec` for `POST …/dataset`: the job's labeled images split into train/val under a unique
 * slug name (`-2`, `-3` when taken), returning the spec and log message for the caller to record.
 */
export function buildDatasetSpec(
  j: Job,
  body: LabelDatasetRequest,
  taken: readonly string[],
  now: number,
  t0: number,
  onCreated: (dsId: string, logMessage: string) => void,
): MockDatasetSpec {
  const frac = body.val_fraction ?? 0.2;
  if (!(frac >= 0 && frac <= 0.5)) throw new ApiError(400, 'invalid_body', 'field `val_fraction`: between 0 and 0.5');
  const labeled = j.images.filter((i) => i.status === 'labeled');
  if (!labeled.length) throw new ApiError(400, 'invalid_dataset', `No image of ${j.name} is labeled yet.`, 'Label some images first.');
  const base = (body.name?.trim() || j.name).toLowerCase().replace(/[^a-z0-9-]+/g, '-').replace(/^-+|-+$/g, '') || 'labeled';
  let name = base;
  for (let k = 2; taken.includes(name); k++) name = `${base}-${k}`;
  const val = Math.min(labeled.length - 1, Math.round(labeled.length * frac));
  const first = labeled.find((i) => 'spec' in i.from)?.from;
  const look = first && 'spec' in first ? first.spec : (specs[1] as MockDatasetSpec);
  const counts = j.classes.map((_, c) => labeled.reduce((a, i) => a + i.boxes.filter((b) => b.class === c).length, 0));
  const boxes = counts.reduce((a, b) => a + b, 0);
  const dsId = `ds_${(hash(name + now) & 0xffff).toString(16).padStart(4, '0')}`;
  onCreated(dsId, `dataset '${name}' made: ${labeled.length - val} train and ${val} val images`);
  return {
    ...look,
    id: dsId,
    name,
    path: `D:/folo/datasets/${name}`,
    format: 'yolo',
    classes: [...j.classes],
    classWeights: counts.map((c) => Math.max(1, c)),
    classShape: j.classes.map((_, c) => look.classShape[c] ?? 'square'),
    splits: { train: labeled.length - val, val, test: 0 },
    meanBoxes: boxes / labeled.length,
    emptyRate: labeled.filter((i) => !i.boxes.length).length / labeled.length,
    brokenLabels: 0,
    createdAgoS: (t0 - now) / 1000,
    seed: hash(j.id),
  };
}
