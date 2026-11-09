// Mock agent labeling (docs/API.md, Agent labeling): labeling jobs over images of the mock's procedural datasets (or
// files uploaded in this page), and agent runs that label one batch every 1.5 s. A run is a function of its start
// time: whatever is due is applied lazily (`advance`) before every answer and on every tick of a subscribed stream,
// so the demo moves in real time and a paused clock (the screenshots) holds it still. The agent's boxes are the
// image's own objects as a decent model would find them (a few missed, shifted or confused), never exact.

import { ApiError, type LabelEventHandlers } from '../client';
import type {
  Box,
  CreateLabelJob,
  LabelAgent,
  LabelAgentId,
  LabelDatasetRequest,
  LabelEffort,
  LabelImage,
  LabelImagePage,
  LabelImagesAdded,
  LabelImagesQuery,
  LabelImageStatus,
  LabelJob,
  LabelOutcome,
  LabelUsage,
  LogLine,
  SkippedImage,
  StartLabeling,
  UpdateLabelJob,
} from '../types';
import { AGENT_NAME, classChangeError, classesError, INSTRUCTIONS_MAX, JOB_NAME_MAX, NOTE_MAX } from '../../label';
import { imageSize, inventBoxes, predictBoxes, round } from './boxes';
import { image, specs, type MockDatasetSpec } from './datasets';
import { renderImage } from './render';
import { hash } from './rng';

/** One agent call ends every this many ms. */
export const BATCH_MS = 1500;
const MIN = 60_000;
const HOUR = 60 * MIN;
/** A job's image limit (docs/API.md). */
const MAX_IMAGES = 10_000;
/** Over this many images added at once, the stream says `reload` instead of one `image` event each (docs/API.md). */
const BULK = 100;

/** Images per call when a run names none (docs/API.md: Claude Code 4, Codex 8). */
const DEFAULT_BATCH: Record<LabelAgentId, number> = { claude: 4, codex: 8 };
/** Claude Code's list prices per million tokens, input and output (Codex reports no cost). */
const PRICE: Record<string, [number, number]> = { sonnet: [3, 15], haiku: [1, 5], opus: [15, 75] };
/** How well each model finds the objects (the share a call gets right, roughly). */
const QUALITY: Record<string, number> = { sonnet: 0.86, haiku: 0.74, opus: 0.92 };

/**
 * The agents a server finds: Claude Code installed; Codex not installed (its hint says how), except in `all-agents`,
 * where both are.
 */
export function mockAgents(codex: boolean): LabelAgent[] {
  return [
    {
      id: 'claude',
      name: AGENT_NAME.claude,
      available: true,
      version: '2.1.302',
      models: [
        { id: 'sonnet', label: 'Sonnet' },
        { id: 'haiku', label: 'Haiku' },
        { id: 'opus', label: 'Opus' },
      ],
      default_model: 'sonnet',
      hint: null,
    },
    {
      id: 'codex',
      name: AGENT_NAME.codex,
      available: codex,
      version: codex ? '0.58.0' : null,
      models: [
        { id: 'gpt-5-codex', label: 'GPT-5 Codex' },
        { id: 'gpt-5', label: 'GPT-5' },
      ],
      default_model: null,
      hint: codex ? null : 'Install Codex (npm install -g @openai/codex) and sign in with `codex login`, or set FOLO_CODEX_BIN to its path.',
    },
  ];
}

interface Img {
  id: string;
  file: string;
  width: number;
  height: number;
  status: LabelImageStatus;
  boxes: Box[];
  source: 'agent' | 'manual' | null;
  error: string | null;
  /** Its pixels: an image of a mock dataset, or a file uploaded in this page. */
  from: { spec: MockDatasetSpec; index: number } | { blob: Blob; key: string };
  /** What it was before an agent call took it: a stopped call gives it back. */
  before: { status: LabelImageStatus; error: string | null } | null;
  /** Agent calls that have failed for it: the mock's agent fails an image once at most. */
  failures: number;
}

interface Run {
  agent: LabelAgentId;
  model: string | null;
  effort: LabelEffort;
  note: string;
  startedAt: number;
  finishedAt: number | null;
  ids: string[];
  batch: number;
  parallel: number;
  /** Calls answered so far. */
  applied: number;
  done: number;
  failed: number;
  outcome: LabelOutcome | null;
  error: string | null;
  hint: string | null;
  usage: LabelUsage;
  /** Set by a stop: when the calls in flight have been ended. */
  stopAt: number | null;
}

interface Job {
  id: string;
  name: string;
  classes: string[];
  instructions: string;
  createdAt: number;
  images: Img[];
  run: Run | null;
  usage: LabelUsage;
  datasetId: string | null;
  logs: LogLine[];
  nextImage: number;
}

const zeroUsage = (): LabelUsage => ({ calls: 0, input_tokens: 0, cached_input_tokens: 0, output_tokens: 0, cost_usd: null });

function addUsage(a: LabelUsage, b: LabelUsage) {
  a.calls += b.calls;
  a.input_tokens += b.input_tokens;
  a.cached_input_tokens += b.cached_input_tokens;
  a.output_tokens += b.output_tokens;
  if (b.cost_usd !== null) a.cost_usd = round((a.cost_usd ?? 0) + b.cost_usd, 6);
}

const iso = (ms: number) => new Date(ms).toISOString();
const n = (x: number) => x.toLocaleString('en-US');

export class MockLabeling {
  #jobs: Job[] = [];
  #subs = new Map<string, Set<LabelEventHandlers>>();
  #urls = new Map<string, Promise<string>>();
  #next = 1;

  constructor(
    readonly t0: number,
    seeded: boolean,
    readonly codex = false,
  ) {
    if (seeded) this.#seed();
  }

  // ---------- seeded jobs ----------

  /** Four jobs in four states: fresh, mid-run (it runs on in real time), fully labeled and reviewed, failed. */
  #seed() {
    const t0 = this.t0;
    const spec = (id: string) => specs.find((s) => s.id === id) as MockDatasetSpec;

    // Its last run ended early: Claude Code was not signed in on the server's machine.
    const failed = this.#make('lb_9e44', 'parts-bench', ['bolt', 'nut', 'washer'], '', t0 - 26 * HOUR);
    this.#addFrom(failed, spec('ds_fasteners'), 1500, 24, 'bench', t0 - 26 * HOUR);
    this.#run(failed, { agent: 'claude', model: 'haiku', effort: 'low', image_ids: failed.images.slice(0, 8).map((i) => i.id) }, t0 - 26 * HOUR + MIN);
    this.advance(failed, t0);
    this.#fail(
      failed,
      { agent: 'claude', model: 'sonnet', effort: 'low' },
      t0 - 25 * HOUR,
      'Not logged in · Please run /login',
      'Sign in to Claude Code on the server’s machine (run `claude` and follow /login), then start the run again.',
    );

    // Labeled, reviewed and corrected by hand: ready to become a dataset.
    const done = this.#make(
      'lb_51d0',
      'fasteners-review',
      ['bolt', 'nut', 'washer', 'screw'],
      'Label every bolt, nut, washer and screw, also partly hidden ones. A spring washer is a washer.',
      t0 - 5 * HOUR,
    );
    this.#addFrom(done, spec('ds_fasteners'), 600, 36, 'line', t0 - 5 * HOUR);
    this.#run(done, { agent: 'claude', model: 'haiku', effort: 'low' }, t0 - 3 * HOUR);
    this.advance(done, t0 - 3 * HOUR + 30 * MIN);
    this.#run(done, { agent: 'claude', model: 'sonnet', effort: 'medium', note: 'Washers under a nut count too.' }, t0 - 2 * HOUR);
    this.advance(done, t0);
    done.images.slice(4, 7).forEach((im, k) => {
      im.boxes = im.boxes.slice(0, Math.max(1, im.boxes.length - k));
      im.source = 'manual';
    });
    for (const im of done.images.slice(20, 22)) {
      im.boxes = [];
      im.source = 'manual';
    }

    // Labeling now: started 12 s before the page loaded, it finishes about 24 s after.
    const live = this.#make(
      'lb_3f9a',
      'shapes-sorting',
      ['circle', 'square', 'triangle', 'star'],
      'Label every circle, square, triangle and star, also partly hidden ones. Ignore hexagons and crosses.',
      t0 - 40 * MIN,
    );
    this.#addFrom(live, spec('ds_shapes_hard'), 0, 96, 'tray', t0 - 40 * MIN);
    this.#run(live, { agent: 'claude', model: 'sonnet', effort: 'low' }, t0 - 12_000);
    this.advance(live, t0);

    // Fresh: images added, nothing labeled yet.
    const fresh = this.#make(
      'lb_7c21',
      'warehouse-bins',
      ['bolt', 'nut', 'washer', 'screw'],
      'Label every bolt, nut, washer and screw, also partly hidden ones. Ignore the rivets on the conveyor.',
      t0 - 20 * MIN,
    );
    this.#addFrom(fresh, spec('ds_fasteners'), 900, 40, 'bin', t0 - 20 * MIN);
  }

  #make(id: string, name: string, classes: string[], instructions: string, at: number): Job {
    const job: Job = { id, name, classes, instructions, createdAt: at, images: [], run: null, usage: zeroUsage(), datasetId: null, logs: [], nextImage: 1 };
    this.#jobs.unshift(job);
    return job;
  }

  /** `count` images of `spec` from `first` on, named like a camera's files (`bin-0901.jpg`). */
  #addFrom(job: Job, spec: MockDatasetSpec, first: number, count: number, prefix: string, at: number) {
    for (let index = first; index < first + count; index++) {
      const d = image(spec, index);
      job.images.push(this.#img(job, `${prefix}-${String(index + 1).padStart(4, '0')}.jpg`, d.width, d.height, { spec, index }));
    }
    this.#log(job, at, 'info', `added ${count} images from D:/photos/${prefix}`);
  }

  #img(job: Job, file: string, width: number, height: number, from: Img['from']): Img {
    return { id: String(job.nextImage++), file, width, height, status: 'pending', boxes: [], source: null, error: null, from, before: null, failures: 0 };
  }

  // ---------- reading ----------

  #job(id: string): Job {
    const j = this.#jobs.find((x) => x.id === id);
    if (!j) throw new ApiError(404, 'label_job_not_found', `No labeling job with id ${id}.`, 'List them with GET /api/label/jobs.');
    return j;
  }

  #image(job: Job, imageId: string): Img {
    const im = job.images.find((i) => i.id === imageId);
    if (!im) throw new ApiError(404, 'label_image_not_found', `Job ${job.name} has no image ${imageId}.`);
    return im;
  }

  #toImage(im: Img): LabelImage {
    return { id: im.id, file: im.file, width: im.width, height: im.height, status: im.status, boxes: im.boxes, source: im.source, error: im.error };
  }

  #toJob(j: Job): LabelJob {
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
            started: iso(r.startedAt),
            finished: r.finishedAt === null ? null : iso(r.finishedAt),
            total: r.ids.length,
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

  #active(j: Job) {
    return !!j.run && !j.run.outcome;
  }

  jobs(now: number): LabelJob[] {
    for (const j of this.#jobs) this.advance(j, now);
    return [...this.#jobs].sort((a, b) => b.createdAt - a.createdAt).map((j) => this.#toJob(j));
  }

  job(id: string, now: number): LabelJob {
    const j = this.#job(id);
    this.advance(j, now);
    return this.#toJob(j);
  }

  images(id: string, q: LabelImagesQuery, now: number): LabelImagePage {
    const j = this.#job(id);
    this.advance(j, now);
    if (q.status && !['pending', 'labeling', 'labeled', 'failed'].includes(q.status))
      throw new ApiError(400, 'invalid_query', `unknown status \`${q.status}\`: pending, labeling, labeled or failed`);
    const all = q.status ? j.images.filter((i) => i.status === q.status) : j.images;
    const offset = Math.max(0, q.offset ?? 0);
    const limit = Math.max(0, Math.min(1000, q.limit ?? 100));
    return { total: all.length, images: all.slice(offset, offset + limit).map((i) => this.#toImage(i)) };
  }

  logs(id: string, limit: number, now: number): { lines: LogLine[] } {
    const j = this.#job(id);
    this.advance(j, now);
    return { lines: j.logs.slice(-Math.max(1, Math.min(5000, Math.floor(limit)))) };
  }

  /** The image's URL: a rendered dataset image, or the uploaded file itself. */
  imageUrl(jobId: string, imageId: string, thumb?: number): Promise<string> {
    const im = this.#image(this.#job(jobId), imageId);
    const from = im.from;
    const key = 'blob' in from ? from.key : `${from.spec.id}/${from.index}/${thumb ?? 0}`;
    let url = this.#urls.get(key);
    if (!url) {
      url = 'blob' in from ? Promise.resolve(URL.createObjectURL(from.blob)) : renderImage(from.spec, image(from.spec, from.index), from.index, thumb).then((b) => URL.createObjectURL(b));
      this.#urls.set(key, url);
    }
    return url;
  }

  // ---------- jobs ----------

  #checkJobFields(b: UpdateLabelJob) {
    if (b.name !== undefined && (!b.name.trim() || b.name.trim().length > JOB_NAME_MAX))
      throw new ApiError(400, 'invalid_body', `field \`name\`: must not be blank, at most ${JOB_NAME_MAX} characters`);
    const ce = b.classes ? classesError(b.classes) : null;
    if (ce) throw new ApiError(400, 'invalid_body', `field \`classes\`: ${ce}`);
    if (b.instructions !== undefined && b.instructions.length > INSTRUCTIONS_MAX)
      throw new ApiError(400, 'invalid_body', `field \`instructions\`: at most ${INSTRUCTIONS_MAX} characters`);
  }

  create(body: CreateLabelJob, now: number): LabelJob {
    this.#checkJobFields(body);
    // As on the server, a folder that cannot be found creates nothing.
    const folder = body.path ? this.#folder(body.path) : null;
    const id = `lb_${(hash(body.name + now + this.#next++) & 0xffff).toString(16).padStart(4, '0')}`;
    const job = this.#make(id, body.name.trim(), body.classes.map((c) => c.trim()), body.instructions ?? '', now);
    if (folder) this.#import(job, folder, body.path ?? '', now);
    return this.#toJob(job);
  }

  update(id: string, body: UpdateLabelJob, now: number): LabelJob {
    const j = this.#job(id);
    this.advance(j, now);
    if (this.#active(j)) throw this.#busy(j);
    this.#checkJobFields(body);
    if (body.classes) {
      const ce = classChangeError(j.classes, body.classes, j.images.some((i) => i.boxes.length > 0));
      if (ce) throw new ApiError(400, 'invalid_body', `field \`classes\`: ${ce}`);
    }
    if (body.name !== undefined) j.name = body.name.trim();
    if (body.classes) j.classes = body.classes.map((c) => c.trim());
    if (body.instructions !== undefined) j.instructions = body.instructions;
    const out = this.#toJob(j);
    this.#emit(j.id, (h) => h.job?.(out));
    return out;
  }

  delete(id: string, now: number) {
    const j = this.#job(id);
    this.advance(j, now);
    if (this.#active(j)) throw this.#busy(j);
    this.#jobs = this.#jobs.filter((x) => x !== j);
  }

  #busy(j: Job) {
    return new ApiError(409, 'label_job_active', `An agent is labeling images of ${j.name}.`, 'Wait for the run to finish, or stop it first.');
  }

  // ---------- images ----------

  /** A mock dataset's folder (its path, whatever the spelling): what the demo server can "read". */
  #folder(path: string): MockDatasetSpec {
    const key = (p: string) => p.trim().replace(/\\/g, '/').replace(/\/+$/, '').toLowerCase();
    const s = specs.find((d) => key(d.path) === key(path));
    if (!s) throw new ApiError(404, 'path_not_found', `No folder at ${path}.`, 'The demo server cannot read your disk: try one of its own folders, such as D:/data/shapes.');
    return s;
  }

  #import(job: Job, spec: MockDatasetSpec, path: string, now: number): LabelImagesAdded {
    const added: Img[] = [];
    const skipped: SkippedImage[] = [];
    const total = spec.splits.train + spec.splits.val + spec.splits.test;
    for (let index = 0; index < total; index++) {
      const d = image(spec, index);
      if (job.images.some((i) => 'spec' in i.from && i.from.spec.id === spec.id && i.from.index === index))
        skipped.push({ file: d.file, reason: 'already in this job' });
      else if (job.images.length >= MAX_IMAGES) skipped.push({ file: d.file, reason: `the job is full (${n(MAX_IMAGES)} images)` });
      else {
        const im = this.#img(job, this.#unique(job, d.file.split('/').pop() ?? d.file), d.width, d.height, { spec, index });
        job.images.push(im);
        added.push(im);
      }
    }
    return this.#added(job, added, skipped, `added ${n(added.length)} images from ${path}`, now);
  }

  importPath(id: string, path: string, now: number): LabelImagesAdded {
    const j = this.#job(id);
    this.advance(j, now);
    return this.#import(j, this.#folder(path), path, now);
  }

  async upload(id: string, files: File[], now: number): Promise<LabelImagesAdded> {
    const j = this.#job(id);
    const added: Img[] = [];
    const skipped: SkippedImage[] = [];
    for (const f of files) {
      const key = `upload:${f.name}:${f.size}:${f.lastModified}`;
      if (j.images.some((i) => 'blob' in i.from && i.from.key === key)) {
        skipped.push({ file: f.name, reason: 'an image with the same bytes is already in this job' });
        continue;
      }
      if (j.images.length >= MAX_IMAGES) {
        skipped.push({ file: f.name, reason: `the job is full (${n(MAX_IMAGES)} images)` });
        continue;
      }
      try {
        const { width, height } = await imageSize(f);
        const im = this.#img(j, this.#unique(j, f.name), width, height, { blob: f, key });
        j.images.push(im);
        added.push(im);
      } catch {
        skipped.push({ file: f.name, reason: 'not an image FOLO can read (JPEG, PNG, BMP or WebP)' });
      }
    }
    this.advance(j, now);
    return this.#added(j, added, skipped, `added ${n(added.length)} uploaded ${added.length === 1 ? 'image' : 'images'}`, now);
  }

  #added(job: Job, added: Img[], skipped: SkippedImage[], what: string, now: number): LabelImagesAdded {
    if (added.length) this.#log(job, now, 'info', skipped.length ? `${what}; ${n(skipped.length)} skipped` : what);
    const out = this.#toJob(job);
    if (added.length > BULK) this.#emit(job.id, (h) => h.reload?.({}));
    else for (const im of added) this.#emit(job.id, (h) => h.image?.(this.#toImage(im)));
    this.#emit(job.id, (h) => h.job?.(out));
    return { added: added.map((i) => this.#toImage(i)), skipped, job: out };
  }

  /** File names are unique within a job: a second `a.jpg` becomes `a-2.jpg`. */
  #unique(job: Job, file: string): string {
    const taken = new Set(job.images.map((i) => i.file.toLowerCase()));
    if (!taken.has(file.toLowerCase())) return file;
    const dot = file.lastIndexOf('.');
    const [stem, ext] = dot > 0 ? [file.slice(0, dot), file.slice(dot)] : [file, ''];
    for (let k = 2; ; k++) if (!taken.has(`${stem}-${k}${ext}`.toLowerCase())) return `${stem}-${k}${ext}`;
  }

  /** `409` while the image waits for, or is in, an agent call of the active run. */
  #checkFree(j: Job, im: Img) {
    const r = j.run;
    if (im.status === 'labeling' || (r && !r.outcome && r.ids.slice(r.applied * r.batch).includes(im.id)))
      throw new ApiError(409, 'label_job_active', `${im.file} is in the agent's run.`, 'Edit it once the agent has labeled it, or stop the run.');
  }

  updateImage(id: string, imageId: string, boxes: Box[], now: number): LabelImage {
    const j = this.#job(id);
    this.advance(j, now);
    const im = this.#image(j, imageId);
    this.#checkFree(j, im);
    if (boxes.length > 1000) throw new ApiError(400, 'invalid_body', 'field `boxes`: at most 1000');
    const clipped: Box[] = [];
    for (const b of boxes) {
      if (!Number.isInteger(b.class) || b.class < 0 || b.class >= j.classes.length)
        throw new ApiError(400, 'invalid_body', `field \`boxes\`: class ${b.class} is not one of the job's ${j.classes.length} classes`);
      if (![b.x1, b.y1, b.x2, b.y2].every(Number.isFinite)) throw new ApiError(400, 'invalid_body', 'field `boxes`: a coordinate is not a number');
      const c = {
        class: b.class,
        x1: Math.max(0, Math.min(im.width, Math.min(b.x1, b.x2))),
        y1: Math.max(0, Math.min(im.height, Math.min(b.y1, b.y2))),
        x2: Math.max(0, Math.min(im.width, Math.max(b.x1, b.x2))),
        y2: Math.max(0, Math.min(im.height, Math.max(b.y1, b.y2))),
      };
      if (c.x2 - c.x1 < 1 || c.y2 - c.y1 < 1) throw new ApiError(400, 'invalid_body', 'field `boxes`: a box has less than a pixel left inside the image');
      clipped.push(c);
    }
    Object.assign(im, { boxes: clipped, status: 'labeled', source: 'manual', error: null });
    this.#log(j, now, 'info', `${im.file}: ${clipped.length} ${clipped.length === 1 ? 'box' : 'boxes'} drawn by hand`);
    this.#emit(j.id, (h) => h.image?.(this.#toImage(im)));
    const out = this.#toJob(j);
    this.#emit(j.id, (h) => h.job?.(out));
    return this.#toImage(im);
  }

  deleteImage(id: string, imageId: string, now: number) {
    const j = this.#job(id);
    this.advance(j, now);
    const im = this.#image(j, imageId);
    this.#checkFree(j, im);
    j.images = j.images.filter((x) => x !== im);
    this.#emit(j.id, (h) => h.removed?.({ id: im.id }));
    const out = this.#toJob(j);
    this.#emit(j.id, (h) => h.job?.(out));
  }

  // ---------- runs ----------

  start(id: string, body: StartLabeling, now: number): LabelJob {
    const j = this.#job(id);
    this.advance(j, now);
    if (body.agent !== 'claude' && body.agent !== 'codex')
      throw new ApiError(422, 'invalid_body', `field \`agent\`: unknown variant \`${String(body.agent)}\`, expected \`claude\` or \`codex\``);
    const agent = mockAgents(this.codex).find((a) => a.id === body.agent);
    if (!agent?.available) throw new ApiError(501, 'not_supported', `${AGENT_NAME[body.agent]} is not installed on the server's machine.`, agent?.hint);
    if (body.model && !/^[A-Za-z0-9._:/[\]-]{1,100}$/.test(body.model))
      throw new ApiError(400, 'invalid_body', 'field `model`: letters, digits and `. _ : / - [ ]` only, at most 100');
    if ((body.note ?? '').length > NOTE_MAX) throw new ApiError(400, 'invalid_body', `field \`note\`: at most ${NOTE_MAX} characters`);
    if (this.#active(j)) throw new ApiError(409, 'label_job_active', `An agent is already labeling ${j.name}.`, 'Wait for the run to finish, or stop it first.');
    const unknown = body.image_ids?.find((x) => !j.images.some((i) => i.id === x));
    if (unknown) throw new ApiError(400, 'invalid_body', `field \`image_ids\`: job ${j.name} has no image ${unknown}`);
    this.#run(j, body, now);
    if (!j.run?.ids.length) {
      j.run = null;
      throw new ApiError(400, 'invalid_body', 'nothing to label: every image of the job is labeled', 'Start with `scope: all` to label them again.');
    }
    this.advance(j, now);
    return this.#toJob(j);
  }

  /** Starts a run at `at` (the images it takes: `image_ids`, else the scope's) and sends its first calls. */
  #run(j: Job, body: StartLabeling, at: number) {
    const ids = body.image_ids
      ? [...body.image_ids]
      : j.images.filter((i) => body.scope === 'all' || i.status === 'pending' || i.status === 'failed').map((i) => i.id);
    const r: Run = { ...this.#blankRun(body, at), ids };
    j.run = r;
    if (!ids.length) return;
    this.#log(
      j,
      at,
      'info',
      `run started: ${n(ids.length)} ${ids.length === 1 ? 'image' : 'images'} with ${AGENT_NAME[r.agent]} (${r.model ?? 'default model'}, ${r.effort} effort), ${r.batch} per call, ${r.parallel} at a time${r.note ? `; note: ${r.note}` : ''}`,
    );
    this.#send(j);
  }

  #blankRun(body: StartLabeling, at: number): Omit<Run, 'ids'> {
    return {
      agent: body.agent,
      model: body.model ?? (body.agent === 'claude' ? 'sonnet' : null),
      effort: body.effort ?? 'low',
      note: body.note ?? '',
      startedAt: at,
      finishedAt: null,
      batch: body.batch ?? DEFAULT_BATCH[body.agent],
      parallel: body.parallel ?? 3,
      applied: 0,
      done: 0,
      failed: 0,
      outcome: null,
      error: null,
      hint: null,
      usage: zeroUsage(),
      stopAt: null,
    };
  }

  /** A run that ended at its first call: the CLI refused (not signed in), so nothing was labeled. */
  #fail(j: Job, body: StartLabeling, at: number, error: string, hint: string) {
    this.#run(j, body, at);
    const r = j.run as Run;
    for (const im of j.images) if (im.status === 'labeling') this.#giveBack(im);
    Object.assign(r, { outcome: 'failed', finishedAt: at + 4000, error, hint });
    r.usage.calls = 1;
    j.usage.calls += 1;
    this.#log(j, at + 4000, 'error', `${AGENT_NAME[r.agent]} exited with an error: ${error}`);
    this.#log(j, at + 4000, 'error', `run failed after 1 call: ${n(r.ids.length)} images left as they were`);
  }

  stop(id: string, now: number): LabelJob {
    const j = this.#job(id);
    this.advance(j, now);
    const r = j.run;
    if (!r || r.outcome) throw new ApiError(409, 'label_job_not_active', `No agent is labeling ${j.name}.`);
    if (r.stopAt === null) {
      r.stopAt = now + 1000;
      this.#log(j, now, 'warn', 'stop requested: ending the calls in flight');
    }
    const out = this.#toJob(j);
    this.#emit(j.id, (h) => h.job?.(out));
    return out;
  }

  /** The images of call `k` of the run. */
  #batch(r: Run, k: number) {
    return r.ids.slice(k * r.batch, (k + 1) * r.batch);
  }

  /** Puts the calls that are due in flight: up to `parallel` of them after the ones answered. */
  #send(j: Job) {
    const r = j.run as Run;
    const calls = Math.ceil(r.ids.length / r.batch);
    for (let k = r.applied; k < Math.min(calls, r.applied + r.parallel); k++)
      for (const id of this.#batch(r, k)) {
        const im = j.images.find((i) => i.id === id);
        if (!im || im.status === 'labeling') continue;
        im.before = { status: im.status, error: im.error };
        im.status = 'labeling';
        this.#emit(j.id, (h) => h.image?.(this.#toImage(im)));
      }
  }

  #giveBack(im: Img) {
    if (im.status !== 'labeling') return;
    im.status = im.before?.status ?? 'pending';
    im.error = im.before?.error ?? null;
    im.before = null;
  }

  /** Applies whatever is due by `now`: answered calls, the next calls sent, a stop's end, the run's end. */
  advance(j: Job, now: number) {
    const r = j.run;
    if (!r || r.outcome) return;
    let changed = false;
    if (r.stopAt !== null) {
      if (now < r.stopAt) return;
      for (const im of j.images) if (im.status === 'labeling') {
        this.#giveBack(im);
        this.#emit(j.id, (h) => h.image?.(this.#toImage(im)));
      }
      this.#end(j, 'stopped', r.stopAt);
      return;
    }
    const calls = Math.ceil(r.ids.length / r.batch);
    const due = Math.min(calls, Math.floor((now - r.startedAt) / BATCH_MS));
    while (r.applied < due) {
      this.#answer(j, r.applied, r.startedAt + (r.applied + 1) * BATCH_MS);
      r.applied++;
      changed = true;
    }
    if (!changed) return;
    this.#send(j);
    if (r.applied >= calls) this.#end(j, 'completed', r.startedAt + calls * BATCH_MS);
    else {
      const out = this.#toJob(j);
      this.#emit(j.id, (h) => h.job?.(out));
    }
  }

  /** Call `k`'s answer at `at`: boxes for each of its images (an image the reply leaves out fails, once at most). */
  #answer(j: Job, k: number, at: number) {
    const r = j.run as Run;
    const ids = this.#batch(r, k);
    let boxes = 0;
    for (const id of ids) {
      const im = j.images.find((i) => i.id === id);
      if (!im) continue;
      im.before = null;
      if (im.failures === 0 && hash(`${j.id}:${im.id}`) % 13 === 0) {
        Object.assign(im, { status: 'failed', error: 'The reply did not mention this image.' });
        im.failures++;
        r.failed++;
        this.#log(j, at, 'warn', `${im.file}: not in the reply; marked failed`);
      } else {
        Object.assign(im, { status: 'labeled', boxes: this.#draft(j, im, r), source: 'agent', error: null });
        boxes += im.boxes.length;
        r.done++;
      }
      this.#emit(j.id, (h) => h.image?.(this.#toImage(im)));
    }
    const usage = this.#usage(j, r, k, ids.length, boxes);
    addUsage(r.usage, usage);
    addUsage(j.usage, usage);
    this.#log(
      j,
      at,
      'info',
      `call ${k + 1}: ${ids.length} ${ids.length === 1 ? 'image' : 'images'}, ${boxes} ${boxes === 1 ? 'box' : 'boxes'} · ${n(usage.input_tokens)} tokens in (${n(usage.cached_input_tokens)} cached), ${n(usage.output_tokens)} out`,
    );
    if (hash(`${j.id}:drop:${k}:${r.startedAt}`) % 6 === 0) this.#log(j, at, 'warn', `call ${k + 1}: 1 box dropped: its label is not one of the job's classes`);
  }

  /** The agent's boxes for an image: its objects as a decent model finds them, in the job's classes. */
  #draft(j: Job, im: Img, r: Run): Box[] {
    const seed = hash(`${j.id}:${im.id}:${r.startedAt}`);
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
  #usage(j: Job, r: Run, k: number, images: number, boxes: number): LabelUsage {
    const fixed = r.agent === 'claude' ? 420 + Math.round(j.instructions.length / 4) : 10_200;
    const input = fixed + images * 1012 + (hash(`${j.id}:${k}`) % 60);
    const cached = k === 0 ? 0 : Math.round(fixed * 0.86);
    const reasoning = { low: 0, medium: 260, high: 820 }[r.effort] * images;
    const output = 48 + boxes * 23 + reasoning;
    const price = r.agent === 'claude' ? PRICE[r.model ?? 'sonnet'] : undefined;
    const cost = price ? round(((input - cached) * price[0] + cached * price[0] * 0.1 + output * price[1]) / 1e6, 6) : null;
    return { calls: 1, input_tokens: input, cached_input_tokens: cached, output_tokens: output, cost_usd: cost };
  }

  #end(j: Job, outcome: LabelOutcome, at: number) {
    const r = j.run as Run;
    r.outcome = outcome;
    r.finishedAt = at;
    r.stopAt = null;
    const left = r.ids.length - r.done - r.failed;
    this.#log(
      j,
      at,
      outcome === 'completed' ? 'info' : 'warn',
      outcome === 'completed'
        ? `run completed: ${n(r.done)} labeled, ${n(r.failed)} failed, ${n(r.usage.calls)} calls`
        : `run stopped: ${n(r.done)} labeled, ${n(left)} ${left === 1 ? 'image' : 'images'} left as they were`,
    );
    const out = this.#toJob(j);
    this.#emit(j.id, (h) => h.job?.(out));
  }

  // ---------- datasets ----------

  /**
   * What `POST …/dataset` writes, as a mock dataset (the caller registers it): the job's labeled images in train and
   * val, under the job's name made a folder name (`-2`, `-3` when taken).
   */
  dataset(id: string, body: LabelDatasetRequest, taken: readonly string[], now: number): MockDatasetSpec {
    const j = this.#job(id);
    this.advance(j, now);
    if (this.#active(j)) throw this.#busy(j);
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
    j.datasetId = dsId;
    this.#log(j, now, 'info', `dataset ${name} written: ${n(labeled.length - val)} train, ${n(val)} val images, ${n(boxes)} boxes`);
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
      createdAgoS: (this.t0 - now) / 1000,
      seed: hash(j.id),
    };
  }

  // ---------- events ----------

  /** Live events of a job; the subscriber is told the job first. */
  subscribe(id: string, h: LabelEventHandlers): () => void {
    const subs = this.#subs.get(id) ?? new Set<LabelEventHandlers>();
    this.#subs.set(id, subs);
    subs.add(h);
    return () => subs.delete(h);
  }

  /** Applies what is due for a subscribed job (each stream's tick). */
  tick(id: string, now: number) {
    const j = this.#jobs.find((x) => x.id === id);
    if (j) this.advance(j, now);
  }

  #emit(id: string, f: (h: LabelEventHandlers) => void) {
    for (const h of this.#subs.get(id) ?? []) f(h);
  }

  #log(j: Job, at: number, level: LogLine['level'], message: string) {
    const line: LogLine = { t: iso(at), level, message };
    j.logs.push(line);
    if (j.logs.length > 5000) j.logs.splice(0, j.logs.length - 5000);
    this.#emit(j.id, (h) => h.log?.(line));
  }
}
