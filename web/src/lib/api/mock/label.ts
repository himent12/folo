// Mock agent labeling (docs/API.md, Agent labeling): labeling jobs over images of the mock's procedural datasets (or
// files uploaded in this page), and agent runs that label one batch every 1.5 s. A run is a function of its start
// time: whatever is due is applied lazily (`advance`) before every answer and on every tick of a subscribed stream,
// so the demo moves in real time and a paused clock (the screenshots) holds it still. The agent's boxes are the
// image's own objects as a decent model would find them (a few missed, shifted or confused), never exact.
// Agent catalog/validation live in `./labelAgents.ts`; serialization/simulation/dataset building in `./labelSim.ts`.

import { AGENT_NAME, boxRevision, classChangeError, classesError, folderPath, folderPathError, INSTRUCTIONS_MAX, JOB_NAME_MAX, NOTE_MAX } from '../../label';
import { ApiError, type LabelEventHandlers } from '../client';
import type {
  Box,
  CreateLabelJob,
  LabelDatasetRequest,
  LabelImage,
  LabelImagePage,
  LabelImagesAdded,
  LabelImagesQuery,
  LabelJob,
  LabelOutcome,
  LogLine,
  SkippedImage,
  StartLabeling,
  UpdateLabelJob,
} from '../types';
import { imageSize } from './boxes';
import { image, specs, type MockDatasetSpec } from './datasets';
import {
  addUsage,
  BATCH_MAX,
  BATCH_MS,
  BULK,
  checkMockModel,
  DEFAULT_BATCH,
  FULL,
  HOUR,
  inRange,
  iso,
  LEFT_OUT,
  MAX_IMAGES,
  MIN,
  mockAgents,
  PARALLEL_MAX,
  plural,
  resolveAgentRequest,
  zeroUsage,
  type Img,
  type Job,
  type Run,
} from './labelAgents';
import { buildDatasetSpec, callUsage, draftBoxes, runSummary, serializeImage, serializeJob } from './labelSim';
import { renderImage } from './render';
import { hash, hashBytes } from './rng';

export { BATCH_MS, DEFAULT_BATCH, checkMockModel, mockAgents, resolveAgentRequest } from './labelAgents';

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

  /** Four jobs in four states: fresh, mid-run (it runs on in real time), fully labeled and partly reviewed, failed. */
  #seed() {
    const t0 = this.t0;
    const spec = (id: string) => specs.find((s) => s.id === id) as MockDatasetSpec;

    // Its last run ended early: Claude Code was not signed in on the server's machine.
    const failed = this.#make('lb_9e44', 'parts-bench', ['bolt', 'nut', 'washer'], '', t0 - 26 * HOUR);
    this.#addFrom(failed, spec('ds_fasteners'), 1500, 24, 'bench');
    this.#run(failed, { agent: 'claude', image_ids: failed.images.slice(0, 8).map((i) => i.id) }, t0 - 26 * HOUR + MIN);
    this.advance(failed, t0);
    this.#fail(
      failed,
      { agent: 'claude' },
      t0 - 25 * HOUR,
      'Not logged in · Please run /login',
      'Sign in to Claude Code on the server’s machine (run `claude` and follow /login), then start the run again.',
    );

    // Labeled, partly reviewed and corrected by hand: ready to become a dataset.
    const done = this.#make(
      'lb_51d0',
      'fasteners-review',
      ['bolt', 'nut', 'washer', 'screw'],
      'Label every bolt, nut, washer and screw, also partly hidden ones. A spring washer is a washer.',
      t0 - 5 * HOUR,
    );
    this.#addFrom(done, spec('ds_fasteners'), 600, 36, 'line');
    this.#run(done, { agent: 'claude' }, t0 - 3 * HOUR);
    this.advance(done, t0 - 3 * HOUR + 30 * MIN);
    this.#run(done, { agent: 'claude', note: 'Washers under a nut count too.' }, t0 - 2 * HOUR);
    this.advance(done, t0);
    done.images.slice(4, 7).forEach((im, k) => {
      im.boxes = im.boxes.slice(0, Math.max(1, im.boxes.length - k));
      im.source = 'manual';
    });
    for (const im of done.images.slice(20, 22)) {
      im.boxes = [];
      im.source = 'manual';
    }
    // Reviewed as far as the 22nd image: the ones corrected by hand and the ones that looked right.
    for (const im of done.images.slice(0, 22)) im.reviewed = true;

    // Labeling now: started 12 s before the page loaded, it finishes about 24 s after.
    const live = this.#make(
      'lb_3f9a',
      'shapes-sorting',
      ['circle', 'square', 'triangle', 'star'],
      'Label every circle, square, triangle and star, also partly hidden ones. Ignore hexagons and crosses.',
      t0 - 40 * MIN,
    );
    this.#addFrom(live, spec('ds_shapes_hard'), 0, 96, 'tray');
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
    this.#addFrom(fresh, spec('ds_fasteners'), 900, 40, 'bin');
  }

  #make(id: string, name: string, classes: string[], instructions: string, at: number): Job {
    const job: Job = { id, name, classes, instructions, createdAt: at, images: [], run: null, usage: zeroUsage(), datasetId: null, logs: [], nextImage: 1, runs: 0 };
    this.#jobs.unshift(job);
    return job;
  }

  /** `count` images of `spec` from `first` on, named like a camera's files (`bin-0901.jpg`). */
  #addFrom(job: Job, spec: MockDatasetSpec, first: number, count: number, prefix: string) {
    for (let index = first; index < first + count; index++) {
      const d = image(spec, index);
      job.images.push(this.#img(job, `${prefix}-${String(index + 1).padStart(4, '0')}.jpg`, d.width, d.height, { spec, index }));
    }
  }

  #img(job: Job, file: string, width: number, height: number, from: Img['from']): Img {
    return { id: String(job.nextImage++), file, width, height, status: 'pending', boxes: [], source: null, error: null, reviewed: false, from, before: null, failures: 0 };
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

  #active(j: Job) {
    return !!j.run && !j.run.outcome;
  }

  jobs(now: number): LabelJob[] {
    for (const j of this.#jobs) this.advance(j, now);
    return [...this.#jobs].sort((a, b) => b.createdAt - a.createdAt).map(serializeJob);
  }

  job(id: string, now: number): LabelJob {
    const j = this.#job(id);
    this.advance(j, now);
    return serializeJob(j);
  }

  images(id: string, q: LabelImagesQuery, now: number): LabelImagePage {
    const j = this.#job(id);
    this.advance(j, now);
    if (q.status && !['pending', 'labeling', 'labeled', 'failed'].includes(q.status))
      throw new ApiError(400, 'invalid_query', `unknown status \`${q.status}\`: pending, labeling, labeled or failed`);
    const all = j.images.filter((i) => (!q.status || i.status === q.status) && (!q.ids || q.ids.includes(i.id)));
    const offset = Math.max(0, q.offset ?? 0);
    const limit = Math.max(0, Math.min(1000, q.limit ?? 100));
    return { total: all.length, images: all.slice(offset, offset + limit).map(serializeImage) };
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

  /**
   * A job's labeled images with their boxes and where their pixels come from: what a benchmark made from the job copies
   * (docs/API.md, Vision benchmarks: "a labeling job's `labeled` images").
   */
  labeled(id: string, now: number): { name: string; classes: string[]; images: { file: string; width: number; height: number; boxes: Box[]; from: Img['from'] }[] } {
    const j = this.#job(id);
    this.advance(j, now);
    const images = j.images.filter((i) => i.status === 'labeled').map((i) => ({ file: i.file, width: i.width, height: i.height, boxes: i.boxes.map((b) => ({ ...b })), from: i.from }));
    return { name: j.name, classes: [...j.classes], images };
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
    const folder = body.path && folderPath(body.path) ? this.#folder(body.path) : null;
    const id = `lb_${(hash(body.name + now + this.#next++) & 0xffff).toString(16).padStart(4, '0')}`;
    const job = this.#make(id, body.name.trim(), body.classes.map((c) => c.trim()), body.instructions ?? '', now);
    if (folder) this.#import(job, folder);
    return serializeJob(job);
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
    const out = serializeJob(j);
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
  #folder(raw: string): MockDatasetSpec {
    // As the server: the quotes and spaces around a pasted path are not part of it, and a relative path is refused.
    const path = folderPath(raw);
    if (folderPathError(path)) throw new ApiError(400, 'invalid_body', `path must be a full path: ${path} is relative`);
    const key = (p: string) => p.replace(/\\/g, '/').replace(/\/+$/, '').toLowerCase();
    const s = specs.find((d) => key(d.path) === key(path));
    if (!s) throw new ApiError(404, 'path_not_found', `No folder at ${path}.`, 'The demo server cannot read your disk: try one of its own folders, such as D:/data/shapes.');
    return s;
  }

  #import(job: Job, spec: MockDatasetSpec): LabelImagesAdded {
    const added: Img[] = [];
    const skipped: SkippedImage[] = [];
    const total = spec.splits.train + spec.splits.val + spec.splits.test;
    for (let index = 0; index < total; index++) {
      const d = image(spec, index);
      if (job.images.some((i) => 'spec' in i.from && i.from.spec.id === spec.id && i.from.index === index))
        skipped.push({ file: d.file, reason: 'the job already has this file' });
      else if (job.images.length >= MAX_IMAGES) skipped.push({ file: d.file, reason: FULL });
      else {
        const im = this.#img(job, this.#unique(job, d.file.split('/').pop() ?? d.file), d.width, d.height, { spec, index });
        job.images.push(im);
        added.push(im);
      }
    }
    return this.#added(job, added, skipped);
  }

  importPath(id: string, path: string, now: number): LabelImagesAdded {
    const j = this.#job(id);
    this.advance(j, now);
    return this.#import(j, this.#folder(path));
  }

  async upload(id: string, files: File[], now: number): Promise<LabelImagesAdded> {
    const j = this.#job(id);
    const added: Img[] = [];
    const skipped: SkippedImage[] = [];
    for (const f of files) {
      // As on the server, an upload is the same image as one the job has when its bytes are the same, whatever its
      // name; another file under a taken name is added under a name of its own.
      let size: { width: number; height: number };
      try {
        size = await imageSize(f);
      } catch {
        skipped.push({ file: f.name, reason: 'not a readable image' });
        continue;
      }
      const key = `upload:${hashBytes(new Uint8Array(await f.arrayBuffer()))}`;
      const same = j.images.find((i) => 'blob' in i.from && i.from.key === key);
      if (same) {
        skipped.push({ file: f.name, reason: `the job already has this image (${same.file})` });
        continue;
      }
      if (j.images.length >= MAX_IMAGES) {
        skipped.push({ file: f.name, reason: FULL });
        continue;
      }
      const im = this.#img(j, this.#unique(j, f.name), size.width, size.height, { blob: f, key });
      j.images.push(im);
      added.push(im);
    }
    this.advance(j, now);
    return this.#added(j, added, skipped);
  }

  #added(job: Job, added: Img[], skipped: SkippedImage[]): LabelImagesAdded {
    const out = serializeJob(job);
    if (added.length > BULK) this.#emit(job.id, (h) => h.reload?.({}));
    else for (const im of added) this.#emit(job.id, (h) => h.image?.(serializeImage(im)));
    this.#emit(job.id, (h) => h.job?.(out));
    return { added: added.map(serializeImage), skipped, job: out };
  }

  /** File names are unique within a job: a second `a.jpg` becomes `a-2.jpg`. */
  #unique(job: Job, file: string): string {
    const taken = new Set(job.images.map((i) => i.file.toLowerCase()));
    if (!taken.has(file.toLowerCase())) return file;
    const dot = file.lastIndexOf('.');
    const [stem, ext] = dot > 0 ? [file.slice(0, dot), file.slice(dot)] : [file, ''];
    for (let k = 2; ; k++) if (!taken.has(`${stem}-${k}${ext}`.toLowerCase())) return `${stem}-${k}${ext}`;
  }

  /** Takes an image for a change by hand: `409` while it is in an agent call; one that only waits for its call leaves the run. */
  #claim(j: Job, im: Img) {
    if (im.status === 'labeling')
      throw new ApiError(409, 'label_job_active', `${im.file} is with the agent right now.`, 'Change it once that call is done, or stop the run.');
    const r = j.run;
    if (r && !r.outcome && r.ids.slice(r.applied * r.batch).includes(im.id)) r.left.add(im.id);
  }

  /** `409 label_boxes_changed` unless the image's boxes have the revision `expect` (when given), as the server checks. */
  #expect(im: Img, expect: string | undefined) {
    const now = boxRevision(im.boxes);
    if (expect !== undefined && expect !== now)
      throw new ApiError(409, 'label_boxes_changed', `image '${im.id}' has other boxes than revision ${expect} (now ${now})`);
  }

  updateImage(id: string, imageId: string, boxes: Box[], now: number, expect?: string): LabelImage {
    const j = this.#job(id);
    this.advance(j, now);
    const im = this.#image(j, imageId);
    this.#expect(im, expect);
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
    this.#claim(j, im);
    Object.assign(im, { boxes: clipped, status: 'labeled', source: 'manual', error: null, reviewed: true });
    this.#emit(j.id, (h) => h.image?.(serializeImage(im)));
    const out = serializeJob(j);
    this.#emit(j.id, (h) => h.job?.(out));
    return serializeImage(im);
  }

  /** `PUT …/images/{id}` with `reviewed` alone: the labels checked (or not), the boxes as they are. */
  reviewImage(id: string, imageId: string, reviewed: boolean, now: number, expect?: string): LabelImage {
    const j = this.#job(id);
    this.advance(j, now);
    const im = this.#image(j, imageId);
    this.#expect(im, expect);
    if (im.status === 'labeling')
      throw new ApiError(409, 'label_job_active', `${im.file} is with the agent right now.`, 'Change it once that call is done, or stop the run.');
    if (im.status !== 'labeled')
      throw new ApiError(400, 'invalid_body', `image '${imageId}' has no labels to mark reviewed: save its boxes instead`);
    this.#claim(j, im);
    im.reviewed = reviewed;
    this.#emit(j.id, (h) => h.image?.(serializeImage(im)));
    return serializeImage(im);
  }

  deleteImage(id: string, imageId: string, now: number) {
    const j = this.#job(id);
    this.advance(j, now);
    const im = this.#image(j, imageId);
    this.#claim(j, im);
    j.images = j.images.filter((x) => x !== im);
    this.#emit(j.id, (h) => h.removed?.({ id: im.id }));
    const out = serializeJob(j);
    this.#emit(j.id, (h) => h.job?.(out));
  }

  // ---------- runs ----------

  start(id: string, body: StartLabeling, now: number): LabelJob {
    const j = this.#job(id);
    this.advance(j, now);
    if (body.agent !== 'claude' && body.agent !== 'codex' && body.agent !== 'antigravity')
      throw new ApiError(422, 'invalid_body', `field \`agent\`: unknown variant \`${String(body.agent)}\`, expected \`claude\`, \`codex\` or \`antigravity\``);
    // In the server's order: the body's fields, then the agent, then the job.
    if ([...(body.note ?? '')].length > NOTE_MAX) throw new ApiError(400, 'invalid_body', `note must be at most ${NOTE_MAX} characters`);
    const model = checkMockModel(body.model);
    if (!inRange(body.batch, BATCH_MAX)) throw new ApiError(400, 'invalid_body', `batch must be in 1..=${BATCH_MAX}`);
    if (!inRange(body.parallel, PARALLEL_MAX)) throw new ApiError(400, 'invalid_body', `parallel must be in 1..=${PARALLEL_MAX}`);
    if (body.fast) {
      // As the server: the model named (or the default one) must offer it.
      const offers = mockAgents(true).find((a) => a.id === body.agent);
      const why = !offers?.fast
        ? `${AGENT_NAME[body.agent]} has no faster service tier`
        : model && !offers.models.find((m) => m.id === model)?.fast
          ? `Codex's model catalog lists no faster service tier for ${model}`
          : null;
      if (why) throw new ApiError(400, 'invalid_body', `${why}: send "fast": false`);
    }
    const agent = mockAgents(this.codex).find((a) => a.id === body.agent);
    if (!agent?.available) throw new ApiError(501, 'not_supported', `${AGENT_NAME[body.agent]} is not installed on the server's machine.`, agent?.hint);
    const resolved = resolveAgentRequest(body.agent, model, body.effort, agent.default_model, agent.default_effort);
    if (this.#active(j)) throw new ApiError(409, 'label_job_active', `An agent is already labeling ${j.name}.`, 'Wait for the run to finish, or stop it first.');
    const unknown = body.image_ids?.find((x) => !j.images.some((i) => i.id === x));
    if (unknown) throw new ApiError(400, 'invalid_body', `image_ids: the job has no image '${unknown}'`);
    this.#run(
      j,
      { ...body, model: resolved.model, effort: resolved.effort, image_ids: body.image_ids ? [...new Set(body.image_ids)] : body.image_ids },
      now,
    );
    if (!j.run?.ids.length) {
      j.run = null;
      throw new ApiError(400, 'invalid_body', 'nothing to label: every image of the job is labeled', 'Start with `scope: all` to label them again.');
    }
    this.advance(j, now);
    return serializeJob(j);
  }

  /** Starts a run at `at` (the images it takes: `image_ids`, else the scope's) and sends its first calls. */
  #run(j: Job, body: StartLabeling, at: number) {
    const ids = body.image_ids
      ? [...body.image_ids]
      : j.images.filter((i) => body.scope === 'all' || i.status === 'pending' || i.status === 'failed').map((i) => i.id);
    const r: Run = { ...this.#blankRun(body, at), ids };
    j.run = r;
    j.runs++;
    if (!ids.length) return;
    const calls = Math.ceil(ids.length / r.batch);
    this.#log(
      j,
      at,
      'info',
      `${AGENT_NAME[r.agent]} (${r.model ?? 'its default model'}, ${r.effort} effort) starts on ${ids.length} ${plural(ids.length, 'image')}: ${calls} ${plural(calls, 'call')} of up to ${r.batch}`,
    );
    this.#send(j);
  }

  #blankRun(body: StartLabeling, at: number): Omit<Run, 'ids'> {
    const agent = mockAgents(true).find((a) => a.id === body.agent);
    return {
      agent: body.agent,
      model: body.model ?? agent?.default_model ?? null,
      effort: body.effort ?? agent?.default_effort ?? 'low',
      fast: !!body.fast,
      note: body.note ?? '',
      startedAt: at,
      finishedAt: null,
      left: new Set(),
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
    this.#log(j, at + 4000, 'error', `${AGENT_NAME[r.agent]} stopped: ${error} (${runSummary(r)})`);
  }

  stop(id: string, now: number): LabelJob {
    const j = this.#job(id);
    this.advance(j, now);
    const r = j.run;
    if (!r || r.outcome) throw new ApiError(409, 'label_job_not_active', `No agent is labeling ${j.name}.`);
    if (r.stopAt === null) {
      r.stopAt = now + 1000;
      this.#log(j, now, 'info', 'stop requested');
    }
    const out = serializeJob(j);
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
        if (!im || im.status === 'labeling' || r.left.has(id)) continue;
        im.before = { status: im.status, error: im.error };
        im.status = 'labeling';
        this.#emit(j.id, (h) => h.image?.(serializeImage(im)));
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
      for (const im of j.images)
        if (im.status === 'labeling') {
          this.#giveBack(im);
          this.#emit(j.id, (h) => h.image?.(serializeImage(im)));
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
      const out = serializeJob(j);
      this.#emit(j.id, (h) => h.job?.(out));
    }
  }

  /** Call `k`'s answer at `at`: boxes for each of its images (an image the reply leaves out fails, once at most). */
  #answer(j: Job, k: number, at: number) {
    const r = j.run as Run;
    const ids = this.#batch(r, k).filter((id) => !r.left.has(id));
    if (!ids.length) return;
    let boxes = 0;
    let labeled = 0;
    let leftOut = false;
    for (const id of ids) {
      const im = j.images.find((i) => i.id === id);
      if (!im) continue;
      im.before = null;
      if (im.failures === 0 && hash(`${j.id}:${im.id}`) % 13 === 0) {
        Object.assign(im, { status: 'failed', error: LEFT_OUT, reviewed: false });
        im.failures++;
        r.failed++;
        leftOut = true;
      } else {
        Object.assign(im, { status: 'labeled', boxes: draftBoxes(j, im, r), source: 'agent', error: null, reviewed: false });
        boxes += im.boxes.length;
        labeled++;
        r.done++;
      }
      this.#emit(j.id, (h) => h.image?.(serializeImage(im)));
    }
    const usage = callUsage(j, r, k, ids.length, boxes);
    addUsage(r.usage, usage);
    addUsage(j.usage, usage);
    // The server's lines for a call: what it labeled and used, then what it dropped and why images failed.
    this.#log(j, at, 'info', `call ${k + 1}: ${labeled} ${plural(labeled, 'image')} labeled, ${boxes} ${plural(boxes, 'box')} · ${usage.input_tokens} tokens in, ${usage.output_tokens} out`);
    if (hash(`${j.id}:drop:${k}:${j.runs}`) % 6 === 0) this.#log(j, at, 'warn', `call ${k + 1}: 1 box dropped (a label that is not a class, or a box that is not one)`);
    if (leftOut) this.#log(j, at, 'warn', `call ${k + 1}: ${LEFT_OUT}`);
  }

  #end(j: Job, outcome: LabelOutcome, at: number) {
    const r = j.run as Run;
    r.outcome = outcome;
    r.finishedAt = at;
    r.stopAt = null;
    const name = AGENT_NAME[r.agent];
    this.#log(j, at, 'info', outcome === 'completed' ? `${name} is done: ${runSummary(r)}` : `${name} was stopped: ${runSummary(r)}`);
    const out = serializeJob(j);
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
    return buildDatasetSpec(j, body, taken, now, this.t0, (dsId, logMessage) => {
      j.datasetId = dsId;
      this.#log(j, now, 'info', logMessage);
    });
  }

  // ---------- events ----------

  /** Tells the job's stream the job, as the server does once a dataset made from it is registered (`set_dataset`). */
  announce(id: string, now: number) {
    const out = this.job(id, now);
    this.#emit(id, (h) => h.job?.(out));
  }

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
