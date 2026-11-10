// Mock vision benchmarks (docs/API.md, Vision benchmarks; docs/BENCH.md defines every number): benchmarks over images
// of the mock's datasets, labeling jobs and a pond camera of its own (./benchSeed), entries of every kind and state,
// and self-improvement loops. As the labeling mock does, everything that runs is a function of when it started:
// whatever is due is applied lazily (`advance`) before every answer and on every tick of a subscribed stream, so the
// demo moves in real time and a paused clock (the screenshots) holds it still. Results come from ./benchResults,
// generated from a seed and derived with the definitions, so they agree with each other as a server's do.
// MockBench (here) holds the endpoints; its state and time are in ./benchState, its records and rules in ./benchRecords.

import { ApiError } from '../client';
import type {
  Bench,
  BenchDesign,
  BenchEntry,
  BenchEntryImagePage,
  BenchEntryImagesQuery,
  BenchImage,
  BenchImageOptions,
  BenchImagePage,
  BenchImagesQuery,
  BenchLoop,
  BenchPlan,
  BenchPrediction,
  BenchResult,
  BenchSource,
  Box,
  CreateBench,
  DesignBench,
  LabelUsage,
  Model,
  StartBenchEntry,
  StartBenchLoop,
  UpdateBench,
} from '../types';
import { AGENT_NAME } from '../../label';
import { boxesKey, checkPredictions, predictionKey } from '../../bench/external';
import { clonePlan, planError } from '../../bench/plan';
import {
  AGENT_IMAGES,
  BENCH_NAME_MAX,
  DESCRIPTION_MAX,
  DESIGN_NOTE_MAX,
  GENERATIONS_MAX,
  IMAGES_DEFAULT,
  IMAGES_MAX,
  IMAGES_PAGE_MAX,
  isEntryActive,
  MAX_SEVERITY,
  planFreezes,
  rankEntries,
} from '../../bench/rules';
import { f1 } from '../../bench/score';
import { catalog, CONDITIONS } from './benchCatalog';
import { conditionBoxes } from './benchConditions';
import { designPlan, designUsage } from './benchDesign';
import { renderCondition } from './benchRender';
import { imageDetections, type Skill } from './benchResults';
import { iou, image, total, type MockDatasetSpec } from './datasets';
import { renderImage } from './render';
import { hash, rng } from './rng';
import {
  MS_PER_PASS,
  MS_PER_IMAGE,
  PER_CALL,
  CALL_MS,
  STOP_MS,
  type BenchImg,
  type MBench,
  type MEntry,
  type MLoop,
  r4,
  fingerprintOf,
  agentEntryName,
  loopName,
  openBoxesMax,
  SEED_MAX,
  missingClasses,
} from './benchRecords';
import { MockBenchState } from './benchState';

export {
  fingerprintOf,
  agentEntryName,
  loopName,
  studentName,
  MIN_LABELED_IMAGES,
  poolLabels,
  starvedError,
  noThresholdError,
  OPEN_IDLE_MS,
  OPEN_BOXES_MAX,
  setOpenBoxesMax,
  SEED_MAX,
  labelsPerImage,
  missingClasses,
} from './benchRecords';
export type { BenchImg, MBench, MEntry, MLoop, BenchHost } from './benchRecords';

export class MockBench extends MockBenchState {
  // MockBenchState's private members these endpoints use (MockBenchState.shared), under their names there.
  get #benches() {
    return this.shared.benches();
  }
  set #benches(benches: MBench[]) {
    this.shared.setBenches(benches);
  }
  get #entries() {
    return this.shared.entries();
  }
  set #entries(entries: MEntry[]) {
    this.shared.setEntries(entries);
  }
  get #loops() {
    return this.shared.loops();
  }
  set #loops(loops: MLoop[]) {
    this.shared.setLoops(loops);
  }
  #bench = this.shared.bench;
  #entry = this.shared.entry;
  #loop = this.shared.loop;
  #entriesOf = this.shared.entriesOf;
  #status = this.shared.status;
  #passes = this.shared.passes;
  #finish = this.shared.finish;
  #advanceLoop = this.shared.advanceLoop;
  #emit = this.shared.emit;

  #urls = new Map<string, Promise<string>>();

  // ---------- benchmarks ----------

  #checkFields(body: { name?: string; description?: string }) {
    if (body.name !== undefined && (!body.name.trim() || body.name.trim().length > BENCH_NAME_MAX))
      throw new ApiError(400, 'invalid_body', `field \`name\`: must not be blank, at most ${BENCH_NAME_MAX} characters`);
    if (body.description !== undefined && body.description.length > DESCRIPTION_MAX)
      throw new ApiError(400, 'invalid_body', `field \`description\`: at most ${DESCRIPTION_MAX} characters`);
  }

  #checkPlan(plan: BenchPlan) {
    const e = planError(plan, CONDITIONS);
    if (e) throw new ApiError(400, 'invalid_body', `field \`plan\`: ${e}`);
  }

  /** The images of a source, with their truth, in the source's order. */
  sourceImages(source: BenchSource, now: number): { name: string; classes: string[]; images: Omit<BenchImg, 'id'>[] } {
    if (source.kind === 'dataset') {
      const spec = this.host.datasets().find((d) => d.id === source.dataset_id);
      if (!spec) throw new ApiError(404, 'dataset_not_found', `No dataset with id ${source.dataset_id}.`, 'List datasets with GET /api/datasets.');
      const split = source.split ?? 'val';
      const lo = split === 'train' ? 0 : split === 'val' ? spec.splits.train : spec.splits.train + spec.splits.val;
      const images: Omit<BenchImg, 'id'>[] = [];
      for (let i = lo; i < lo + spec.splits[split]; i++) {
        const d = image(spec, i);
        images.push({ file: d.file.split('/').pop() ?? d.file, width: d.width, height: d.height, boxes: d.boxes, from: { spec, index: i } });
      }
      return { name: spec.name, classes: [...spec.classes], images };
    }
    let job;
    try {
      job = this.host.labeling.labeled(source.job_id, now);
    } catch {
      throw new ApiError(404, 'label_job_not_found', `No labeling job with id ${source.job_id}.`, 'List them with GET /api/label/jobs.');
    }
    return { name: job.name, classes: job.classes, images: job.images.map((i) => ({ file: i.file, width: i.width, height: i.height, boxes: i.boxes, from: i.from })) };
  }

  create(body: CreateBench, now: number): Bench {
    this.#checkFields({ name: body.name, description: body.description });
    if (!body.name?.trim()) throw new ApiError(400, 'invalid_body', 'field `name`: must not be blank');
    const max = body.max_images ?? IMAGES_DEFAULT;
    if (!Number.isInteger(max) || max < 1 || max > IMAGES_MAX) throw new ApiError(400, 'invalid_body', `field \`max_images\`: 1 to ${IMAGES_MAX}`);
    if (body.plan) this.#checkPlan(body.plan);
    const src = this.sourceImages(body.source, now);
    const where = body.source.kind === 'dataset' ? `the ${body.source.split ?? 'val'} split of ${src.name}` : src.name;
    if (!src.images.length) throw new ApiError(400, 'invalid_dataset', `${where[0]!.toUpperCase()}${where.slice(1)} has no labeled image.`, body.source.kind === 'label_job' ? 'Label some of its images first.' : 'Pick a split that has images.');
    const seed = body.seed ?? 0;
    if (!Number.isSafeInteger(seed) || seed < 0) throw new ApiError(400, 'invalid_body', `field \`seed\`: a whole number from 0 to ${SEED_MAX}`);
    // A seeded shuffle, then the first `max_images`: the sample when the source has more, the order either way.
    const r = rng(hash(`bench:${seed}:${body.name}`));
    const order = src.images.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1));
      [order[i], order[j]] = [order[j]!, order[i]!];
    }
    const images = order.slice(0, max).map((k, n) => ({ ...src.images[k]!, id: String(n + 1) }));
    const b: MBench = {
      id: this.id('bn', body.name + now, 12),
      name: body.name.trim(),
      description: body.description ?? '',
      classes: src.classes,
      createdAt: now,
      source: body.source.kind === 'dataset' ? { kind: 'dataset', dataset_id: body.source.dataset_id, split: body.source.split ?? 'val' } : { ...body.source },
      sourceName: src.name,
      images,
      plan: clonePlan(body.plan ?? catalog().default_plan),
      seed,
    };
    this.addBench(b);
    return this.toBench(b);
  }

  update(id: string, body: UpdateBench, now: number): Bench {
    const b = this.#bench(id);
    this.advance(now);
    this.#checkFields(body);
    if (body.plan) {
      // Frozen only by an entry that finished or is under way: a failed or stopped one graded nothing to keep comparable.
      const n = this.#entriesOf(b).filter((e) => planFreezes(this.#status(e, now))).length;
      if (n) throw new ApiError(409, 'bench_frozen', `${b.name} has ${n} finished or running ${n === 1 ? 'entry' : 'entries'}: its plan is fixed so they stay comparable.`, 'Delete those entries, or make a new benchmark with the plan you want.');
      this.#checkPlan(body.plan);
    }
    if (body.name !== undefined) b.name = body.name.trim();
    if (body.description !== undefined) b.description = body.description;
    if (body.plan) b.plan = clonePlan(body.plan);
    const out = this.toBench(b);
    this.#emit(b.id, (h) => h.bench?.(out));
    return out;
  }

  delete(id: string, now: number) {
    const b = this.#bench(id);
    this.advance(now);
    if (this.#entriesOf(b).some((e) => isEntryActive(this.toEntry(e, now))) || this.#loops.some((l) => l.benchId === b.id && !l.end))
      throw new ApiError(409, 'bench_active', `An entry or a loop of ${b.name} is running.`, 'Stop it first, or wait for it to finish.');
    this.#benches = this.#benches.filter((x) => x !== b);
    this.#entries = this.#entries.filter((e) => e.benchId !== b.id);
    this.#loops = this.#loops.filter((l) => l.benchId !== b.id);
  }

  /** The agent's proposal: one call, answered `delay` ms later by the caller. */
  design(id: string, body: DesignBench, now: number): BenchDesign {
    const b = this.#bench(id);
    this.advance(now);
    if (body.agent !== 'claude' && body.agent !== 'codex')
      throw new ApiError(422, 'invalid_body', `field \`agent\`: unknown variant \`${String(body.agent)}\`, expected \`claude\` or \`codex\``);
    if ((body.note ?? '').length > DESIGN_NOTE_MAX) throw new ApiError(400, 'invalid_body', `note must be at most ${DESIGN_NOTE_MAX} characters`);
    const agent = this.host.agents().find((a) => a.id === body.agent);
    if (!agent?.available) throw new ApiError(501, 'not_supported', `${AGENT_NAME[body.agent]} is not installed on the server's machine.`, agent?.hint);
    const model = body.model?.trim() || agent.default_model;
    if (this.designFails)
      throw new ApiError(
        502,
        'agent_failed',
        `${AGENT_NAME[body.agent]} answered, but not with a plan: its reply had no \`conditions\` array.`,
        'Ask again; a larger model (Sonnet) follows the format more reliably.',
      );
    const text = `${b.description}\n${body.note ?? ''}`;
    return { plan: designPlan(text, body.agent, model), usage: designUsage(text, body.agent, model) };
  }

  // ---------- images ----------

  images(id: string, q: BenchImagesQuery, now: number): BenchImagePage {
    const b = this.#bench(id);
    this.advance(now);
    this.#checkCondition(q.condition, q.severity);
    const offset = Math.max(0, q.offset ?? 0);
    const limit = Math.max(0, Math.min(IMAGES_PAGE_MAX, q.limit ?? 100));
    return { total: b.images.length, images: b.images.slice(offset, offset + limit).map((im) => this.#toImage(im, q.condition, q.severity)) };
  }

  #checkCondition(condition: string | undefined, severity: number | undefined) {
    if (condition === undefined && severity === undefined) return;
    if (condition === undefined || severity === undefined) throw new ApiError(400, 'invalid_query', '`condition` and `severity` go together');
    if (!CONDITIONS.some((c) => c.id === condition)) throw new ApiError(400, 'invalid_query', `unknown condition \`${condition}\``);
    if (!Number.isInteger(severity) || severity < 1 || severity > MAX_SEVERITY) throw new ApiError(400, 'invalid_query', `severity must be in 1..=${MAX_SEVERITY}`);
  }

  #toImage(im: BenchImg, condition?: string, severity?: number): BenchImage {
    return { id: im.id, file: im.file, width: im.width, height: im.height, boxes: conditionBoxes(im.boxes, im.width, im.height, condition, severity) };
  }

  /** The image's URL: rendered, then changed by the condition, as a blob. */
  imageUrl(benchId: string, imageId: string, o: BenchImageOptions): Promise<string> {
    const b = this.#bench(benchId);
    this.#checkCondition(o.condition, o.severity);
    const im = b.images.find((x) => x.id === imageId);
    if (!im) return Promise.reject(new ApiError(404, 'bench_image_not_found', `${b.name} has no image ${imageId}.`));
    const key = `${b.id}/${im.id}/${o.condition ?? ''}/${o.severity ?? ''}/${o.thumb ?? 0}`;
    let url = this.#urls.get(key);
    if (!url) {
      const from = im.from;
      url = (async () => {
        const base = 'blob' in from ? from.blob : await renderImage(from.spec, image(from.spec, from.index), from.index, o.thumb);
        const out = o.condition ? await renderCondition(base, { condition: o.condition, severity: o.severity ?? 1, boxes: im.boxes, fullWidth: im.width, seed: `${b.seed}:${im.id}` }) : base;
        return URL.createObjectURL(out);
      })();
      this.#urls.set(key, url);
      if (this.#urls.size > 600) {
        const [oldKey, old] = this.#urls.entries().next().value as [string, Promise<string>];
        this.#urls.delete(oldKey);
        void old.then((u) => setTimeout(() => URL.revokeObjectURL(u), 30_000), () => {});
      }
    }
    return url;
  }

  // ---------- entries ----------

  entries(id: string, now: number): BenchEntry[] {
    const b = this.#bench(id);
    this.advance(now);
    return rankEntries(this.#entriesOf(b).map((e) => this.toEntry(e, now)));
  }

  entry(id: string, entryId: string, now: number): BenchEntry {
    const b = this.#bench(id);
    this.advance(now);
    return this.toEntry(this.#entry(b, entryId), now);
  }

  /** A model entry's skill on a benchmark: its own validation score where it was trained on the same data. */
  modelSkill(m: Model, salt: string): Skill {
    const r = rng(hash(`skill:${m.id}:${salt}`));
    const map = m.metrics?.map50_95 ?? 0.45;
    const big = m.arch === 'folo-n' ? 1 : m.arch === 'folo-s' ? 0.9 : 0.82;
    return { map: map * (0.97 + r() * 0.05), fragility: big * (0.95 + r() * 0.1), ece: 0.035 + r() * 0.04, iou: 0.76 + r() * 0.05, selfRecall: Math.min(0.9, 0.2 + map * 0.8) };
  }

  #startError(b: MBench, body: StartBenchEntry) {
    const k = body.contestant?.kind;
    if (k !== 'model' && k !== 'agent' && k !== 'external') throw new ApiError(422, 'invalid_body', `field \`contestant\`: unknown variant \`${String(k)}\`, expected \`model\`, \`agent\` or \`external\``);
    if (k === 'external') {
      if (!body.name?.trim()) throw new ApiError(400, 'invalid_body', 'field `name`: required for an external contestant');
      if (body.max_images != null || body.conditions != null) throw new ApiError(400, 'invalid_body', '`max_images` and `conditions` are not for an external contestant');
    }
    if (body.name != null && body.name.trim().length > BENCH_NAME_MAX) throw new ApiError(400, 'invalid_body', `field \`name\`: at most ${BENCH_NAME_MAX} characters`);
    if (body.max_images != null && (!Number.isInteger(body.max_images) || body.max_images < 1 || body.max_images > b.images.length))
      throw new ApiError(400, 'invalid_body', `field \`max_images\`: 1 to ${b.images.length}, the benchmark's images`);
  }

  start(id: string, body: StartBenchEntry, now: number): BenchEntry {
    const b = this.#bench(id);
    this.advance(now);
    this.#startError(b, body);
    const c = body.contestant;
    const fp = fingerprintOf(b);
    const base = { benchId: b.id, createdAt: now, fingerprint: fp, stopAt: null, failAfter: null, end: null, result: null, lastEmit: 0, conditionsRun: null };
    let e: MEntry;
    if (c.kind === 'model') {
      const m = this.host.models().find((x) => x.id === c.model_id);
      if (!m) throw new ApiError(404, 'model_not_found', `No model with id ${c.model_id}.`);
      const missing = missingClasses(b.classes, m.classes);
      if (missing.length === b.classes.length)
        throw new ApiError(400, 'bench_class_mismatch', `None of ${m.name}'s ${m.classes.length} classes is a class of ${b.name}.`, `The benchmark's classes are ${b.classes.join(', ')}; a model is matched to them by name.`);
      const images = body.max_images ?? b.images.length;
      const conditions = body.conditions ?? true;
      e = {
        ...base,
        id: this.id('be', m.id + now),
        name: body.name?.trim() || m.name,
        contestant: { kind: 'model', model_id: m.id, arch: m.arch, params: m.params, imgsz: m.imgsz },
        images,
        conditions,
        scored: true,
        missing,
        skill: this.modelSkill(m, b.id),
        slot: true,
        startedAt: null,
        duration: 0,
        usage: null,
        speed: { device: 'gpu:0', ms_per_image: MS_PER_IMAGE[m.arch] ?? 5 },
      };
      e.duration = images * this.#passes(e, b).length * (MS_PER_PASS[m.arch] ?? 5);
    } else if (c.kind === 'agent') {
      if (c.agent !== 'claude' && c.agent !== 'codex') throw new ApiError(422, 'invalid_body', `field \`agent\`: unknown variant \`${String(c.agent)}\``);
      const agent = this.host.agents().find((a) => a.id === c.agent);
      if (!agent?.available) throw new ApiError(501, 'not_supported', `${AGENT_NAME[c.agent]} is not installed on the server's machine.`, agent?.hint);
      const model = c.model?.trim() || agent.default_model;
      const effort = c.effort ?? agent.default_effort;
      const images = body.max_images ?? Math.min(AGENT_IMAGES, b.images.length);
      const conditions = body.conditions ?? false;
      e = {
        ...base,
        id: this.id('be', c.agent + now),
        name: body.name?.trim() || agentEntryName(c.agent, model),
        contestant: { kind: 'agent', agent: c.agent, model, effort },
        images,
        conditions,
        scored: false,
        missing: [],
        skill: agentSkill(c.agent, model, effort, `${b.id}:${now}`),
        slot: false,
        startedAt: now + 600,
        duration: 0,
        usage: null,
        speed: null,
      };
      const passes = this.#passes(e, b).length;
      e.duration = Math.ceil((images * passes) / PER_CALL[c.agent]!) * CALL_MS;
      e.usage = agentUsage(c.agent, model, effort, images * passes, b.description, `${b.id}:${e.id}`);
    } else if (c.open) {
      // Open: created running, filled in parts, scored by its finish; the predictions here are its first part.
      const keys = new Set<string>();
      const first = c.predictions ?? [];
      const check = checkPredictions(first, this.#ctx(b, c.scored), { part: true, seen: keys });
      if (!check.ok) throw new ApiError(400, 'invalid_body', check.error);
      e = {
        ...base,
        id: this.id('be', (body.name ?? '') + now),
        name: (body.name ?? '').trim(),
        contestant: { kind: 'external' },
        images: b.images.length,
        conditions: false,
        scored: c.scored ?? true,
        missing: [],
        skill: { map: 0, fragility: 1, ece: 0, iou: 0.6, selfRecall: 0 },
        slot: false,
        startedAt: now,
        duration: Infinity,
        usage: null,
        speed: null,
        open: { records: [...first], keys, byKey: new Map(first.map((r) => [predictionKey(r.image, r.condition, r.severity), boxesKey(r.boxes, c.scored ?? true)])), lastPart: now, boxes: first.reduce((n, r) => n + r.boxes.length, 0) },
      };
    } else {
      const check = checkPredictions(c.predictions ?? [], this.#ctx(b, c.scored));
      if (!check.ok) throw new ApiError(400, 'invalid_body', check.error);
      const scored = c.scored ?? true;
      e = {
        ...base,
        id: this.id('be', (body.name ?? '') + now),
        name: (body.name ?? '').trim(),
        contestant: { kind: 'external' },
        images: b.images.length,
        conditions: check.summary.conditions.length > 0,
        conditionsRun: check.summary.conditions.length < b.plan.conditions.length ? check.summary.conditions : null,
        scored,
        missing: [],
        skill: externalSkill(b, check.predictions, scored),
        slot: false,
        startedAt: now,
        duration: 0,
        usage: null,
        speed: null,
      };
      this.addEntry(e);
      this.#finish(e, now);
      const out = this.toEntry(e, now);
      this.#emit(b.id, (h) => h.entry?.(out));
      const bench = this.toBench(b);
      this.#emit(b.id, (h) => h.bench?.(bench));
      return out;
    }
    this.addEntry(e);
    this.advance(now);
    const out = this.toEntry(e, now);
    this.#emit(b.id, (h) => h.entry?.(out));
    const bench = this.toBench(b);
    this.#emit(b.id, (h) => h.bench?.(bench));
    return out;
  }

  /** What an external contestant's records are checked against. */
  #ctx(b: MBench, scored?: boolean) {
    return { imageIds: b.images.map((i) => i.id), classes: b.classes.length, plan: b.plan, catalog: CONDITIONS.map((x) => x.id), names: Object.fromEntries(CONDITIONS.map((x) => [x.id, x.name])), scored };
  }

  #open(b: MBench, entryId: string, now: number): MEntry & { open: NonNullable<MEntry['open']> } {
    const e = this.#entry(b, entryId);
    if (!e.open) throw new ApiError(409, 'bench_entry_not_open', `${e.name} is not an open external entry (it is ${this.#status(e, now)}).`, 'Create one with `contestant.open: true`.');
    return e as MEntry & { open: NonNullable<MEntry['open']> };
  }

  /**
   * A part of an open external entry, all of it or nothing: a record already received with the same boxes is taken
   * again and adds nothing (a part retried after a lost answer), one with other boxes is refused; the entry holds at
   * most OPEN_BOXES_MAX boxes.
   */
  sendPredictions(id: string, entryId: string, predictions: BenchPrediction[], now: number): BenchEntry {
    const b = this.#bench(id);
    this.advance(now);
    const e = this.#open(b, entryId, now);
    const fresh: BenchPrediction[] = [];
    for (const [i, r] of (predictions ?? []).entries()) {
      const k = r && typeof r.image === 'string' ? predictionKey(r.image, r.condition, r.severity) : null;
      const had = k !== null ? e.open.byKey.get(k) : undefined;
      if (had === undefined) fresh.push(r);
      else if (had !== boxesKey(r.boxes, e.scored))
        throw new ApiError(400, 'invalid_body', `Record ${i + 1}: image "${r.image}"${r.condition ? ` under ${r.condition} ${r.severity}` : ''} was received before with other boxes.`);
    }
    const keys = new Set(e.open.keys);
    const check = checkPredictions(fresh, this.#ctx(b, e.scored), { part: true, seen: keys });
    if (!check.ok) throw new ApiError(400, 'invalid_body', check.error);
    const boxes = e.open.boxes + fresh.reduce((n, r) => n + r.boxes.length, 0);
    if (boxes > openBoxesMax)
      throw new ApiError(
        400,
        'invalid_body',
        `this part would make the entry hold ${boxes} boxes in memory, more than its ${openBoxesMax}: send it pass by pass (the unchanged images, then each condition and severity), so each condition's severity is graded and let go of as soon as all its images are in`,
      );
    e.open.records.push(...fresh);
    for (const r of fresh) e.open.byKey.set(predictionKey(r.image, r.condition, r.severity), boxesKey(r.boxes, e.scored));
    e.open.keys = keys;
    e.open.boxes = boxes;
    e.open.lastPart = now;
    const out = this.toEntry(e, now);
    this.#emit(b.id, (h) => h.entry?.(out));
    return out;
  }

  /** Scores an open external entry; what is missing is named and it stays open. */
  finishEntry(id: string, entryId: string, now: number): BenchEntry {
    const b = this.#bench(id);
    this.advance(now);
    const e = this.#open(b, entryId, now);
    const check = checkPredictions(e.open.records, this.#ctx(b, e.scored));
    if (!check.ok) throw new ApiError(400, 'invalid_body', check.error);
    e.conditions = check.summary.conditions.length > 0;
    e.conditionsRun = check.summary.conditions.length < b.plan.conditions.length ? check.summary.conditions : null;
    e.skill = externalSkill(b, e.open.records, e.scored);
    (e as MEntry).open = null;
    this.#finish(e, now);
    const out = this.toEntry(e, now);
    this.#emit(b.id, (h) => h.entry?.(out));
    const bench = this.toBench(b);
    this.#emit(b.id, (h) => h.bench?.(bench));
    return out;
  }

  result(id: string, entryId: string, now: number): BenchResult {
    const b = this.#bench(id);
    this.advance(now);
    const e = this.#entry(b, entryId);
    if (e.end?.status !== 'done' || !e.result) throw this.#notDone(e, now);
    const r = e.result;
    return {
      entry: this.toEntry(e, now),
      accuracy: structuredClone(r.accuracy),
      robustness: r.robustness ? structuredClone(r.robustness) : null,
      calibration: r.calibration ? structuredClone(r.calibration) : null,
      localization: structuredClone(r.localization),
      self_labeling: structuredClone(r.self_labeling),
    };
  }

  #notDone(e: MEntry, now: number) {
    return new ApiError(409, 'bench_entry_not_done', `${e.name} is ${this.#status(e, now)}, not done.`, 'Results are there once the entry is done.');
  }

  entryImages(id: string, entryId: string, q: BenchEntryImagesQuery, now: number): BenchEntryImagePage {
    const b = this.#bench(id);
    this.advance(now);
    const e = this.#entry(b, entryId);
    if (e.end?.status !== 'done' || !e.result) throw this.#notDone(e, now);
    if (q.sort && q.sort !== 'order' && q.sort !== 'worst') throw new ApiError(400, 'invalid_query', `unknown sort \`${q.sort}\`: order or worst`);
    const missing = new Set(e.missing);
    const all = b.images.slice(0, e.images).map((im, k) => {
      const d = imageDetections(im, {
        quality: Math.min(0.97, e.skill.map + 0.24),
        seed: hash(`${e.id}:${im.id}`),
        classes: b.classes.length,
        missing,
        bestConf: e.result!.accuracy.best_conf,
        scored: e.scored,
      });
      return { k, page: { image: this.#toImage(im), ...d } };
    });
    if (q.sort === 'worst') all.sort((x, y) => y.page.missed + y.page.wrong - (x.page.missed + x.page.wrong) || x.k - y.k);
    const offset = Math.max(0, q.offset ?? 0);
    const limit = Math.max(0, Math.min(IMAGES_PAGE_MAX, q.limit ?? 100));
    return { total: all.length, images: all.slice(offset, offset + limit).map((x) => x.page) };
  }

  stopEntry(id: string, entryId: string, now: number): BenchEntry {
    const b = this.#bench(id);
    this.advance(now);
    const e = this.#entry(b, entryId);
    const status = this.#status(e, now);
    if (!isEntryActive({ status })) throw new ApiError(409, 'bench_entry_not_active', `${e.name} is ${status}: nothing to stop.`);
    if (e.open) {
      // An open external entry: what it received is discarded and it ends stopped, on the leaderboard until deleted.
      e.end = { status: 'stopped', at: now, error: null, hint: null };
      e.open = null;
      const out = this.toEntry(e, now);
      this.#emit(b.id, (h) => h.entry?.(out));
      const bench = this.toBench(b);
      this.#emit(b.id, (h) => h.bench?.(bench));
      return out;
    }
    if (status === 'queued') {
      // Never started: it ends at once.
      e.end = { status: 'stopped', at: now, error: null, hint: null };
    } else if (e.stopAt === null) e.stopAt = now + STOP_MS;
    const out = this.toEntry(e, now);
    this.#emit(b.id, (h) => h.entry?.(out));
    return out;
  }

  deleteEntry(id: string, entryId: string, now: number) {
    const b = this.#bench(id);
    this.advance(now);
    const e = this.#entry(b, entryId);
    if (isEntryActive({ status: this.#status(e, now) })) throw new ApiError(409, 'bench_entry_active', `${e.name} is running.`, 'Stop it first.');
    const loop = this.#loops.find((l) => !l.end && l.generations.some((g) => g.entry_id === e.id));
    if (loop) throw new ApiError(409, 'bench_entry_in_loop', `The loop ${loop.name} uses ${e.name}.`, 'Wait for the loop to end, or stop it.');
    this.#entries = this.#entries.filter((x) => x !== e);
    this.#emit(b.id, (h) => h.removed?.({ id: e.id }));
    const bench = this.toBench(b);
    this.#emit(b.id, (h) => h.bench?.(bench));
  }

  // ---------- loops ----------

  loops(id: string, now: number): BenchLoop[] {
    const b = this.#bench(id);
    this.advance(now);
    return this.#loops.filter((l) => l.benchId === b.id).sort((x, y) => y.createdAt - x.createdAt).map((l) => this.toLoop(l));
  }

  loop(id: string, loopId: string, now: number): BenchLoop {
    const b = this.#bench(id);
    this.advance(now);
    return this.toLoop(this.#loop(b, loopId));
  }

  /** `fixedId`: the seed's loop that runs as the demo opens has a stable id, so it can be linked (./benchSeed). */
  startLoop(id: string, body: StartBenchLoop, now: number, fixedId?: string): BenchLoop {
    const b = this.#bench(id);
    this.advance(now);
    const m = this.host.models().find((x) => x.id === body.teacher);
    if (!m) throw new ApiError(404, 'model_not_found', `No model with id ${body.teacher}.`);
    if (m.classes.length !== b.classes.length || m.classes.some((c, i) => c !== b.classes[i]))
      throw new ApiError(400, 'invalid_body', `${m.name}'s classes are not ${b.name}'s (the same names in the same order).`, `The benchmark's classes: ${b.classes.join(', ')}.`);
    const gens = body.generations ?? 1;
    if (!Number.isInteger(gens) || gens < 1 || gens > GENERATIONS_MAX) throw new ApiError(400, 'invalid_body', `field \`generations\`: 1 to ${GENERATIONS_MAX}`);
    if (body.min_confidence != null && !(body.min_confidence >= 0 && body.min_confidence <= 1)) throw new ApiError(400, 'invalid_body', 'field `min_confidence`: 0 to 1');
    // The students' settings, checked as POST /api/runs checks a run's, before anything is made.
    const w = body.student?.weights;
    if (w && !this.host.models().some((x) => x.id === w)) throw new ApiError(404, 'model_not_found', `No model with id ${w} (student.weights).`, 'List models with GET /api/models.');
    const dev = body.student?.device;
    if (dev && dev !== 'auto' && dev !== 'cpu' && dev !== 'gpu:0') throw new ApiError(404, 'device_not_found', `No device ${dev} on this machine.`, 'Devices: cpu, gpu:0.');
    if (body.student?.imgsz != null && body.student.imgsz < 64) throw new ApiError(400, 'invalid_config', `imgsz must be at least 64 for training, got ${body.student.imgsz}`);
    const seed = body.student?.seed ?? 0;
    if (!Number.isSafeInteger(seed) || seed < 0) throw new ApiError(400, 'invalid_body', `field \`student.seed\`: a whole number from 0 to ${SEED_MAX}`);
    const pool = this.sourceImages(body.pool.kind === 'dataset' ? { ...body.pool, split: body.pool.split ?? 'train' } : body.pool, now);
    const inBench = new Set(b.images.map((i) => ('spec' in i.from ? `${i.from.spec.id}/${i.from.index}` : i.from.key)));
    const excluded = pool.images.filter((i) => inBench.has('spec' in i.from ? `${i.from.spec.id}/${i.from.index}` : i.from.key)).length;
    if (pool.images.length - excluded <= 0) throw new ApiError(400, 'invalid_dataset', `Every image of the pool is in ${b.name}.`, 'Pick a pool the benchmark was not made from (another split, another job).');
    const config: StartBenchLoop = {
      name: body.name?.trim() || loopName(m.name),
      teacher: m.id,
      pool: body.pool.kind === 'dataset' ? { kind: 'dataset', dataset_id: body.pool.dataset_id, split: body.pool.split ?? 'train' } : { ...body.pool },
      generations: gens,
      min_confidence: body.min_confidence ?? null,
      student: { arch: body.student?.arch ?? m.arch, epochs: body.student?.epochs ?? 30, imgsz: body.student?.imgsz ?? m.imgsz, batch: body.student?.batch ?? null, device: body.student?.device ?? 'auto', weights: body.student?.weights ?? null, seed: body.student?.seed ?? 0 },
    };
    const r = rng(hash(`loop:${b.id}:${m.id}:${now}`));
    const l: MLoop = {
      id: fixedId ?? this.id('bl', m.id + now),
      benchId: b.id,
      name: config.name as string,
      createdAt: now,
      config,
      teacher: { id: m.id, name: m.name, arch: m.arch, imgsz: m.imgsz, skill: this.modelSkill(m, b.id) },
      student: config.student?.arch ?? m.arch,
      pool: { images: pool.images.length, excluded },
      generations: [],
      step: null,
      gains: Array.from({ length: gens }, (_, g) => 0.004 + r() * 0.02 - g * 0.002),
      stopAt: null,
      end: null,
      lastEmit: 0,
    };
    this.addLoop(l);
    this.#advanceLoop(l, b, now);
    const out = this.toLoop(l);
    this.#emit(b.id, (h) => h.loop?.(out));
    return out;
  }

  stopLoop(id: string, loopId: string, now: number): BenchLoop {
    const b = this.#bench(id);
    this.advance(now);
    const l = this.#loop(b, loopId);
    if (l.end) throw new ApiError(409, 'bench_loop_not_active', `${l.name} has ended.`);
    if (l.stopAt === null) {
      l.stopAt = now + STOP_MS;
      const e = l.step?.entryId ? this.#entries.find((x) => x.id === l.step!.entryId) : undefined;
      if (e && !e.end) this.stopEntry(b.id, e.id, now);
    }
    const out = this.toLoop(l);
    this.#emit(b.id, (h) => h.loop?.(out));
    return out;
  }

  deleteLoop(id: string, loopId: string, now: number) {
    const b = this.#bench(id);
    this.advance(now);
    const l = this.#loop(b, loopId);
    if (!l.end) throw new ApiError(409, 'bench_loop_active', `${l.name} is running.`, 'Stop it first.');
    this.#loops = this.#loops.filter((x) => x !== l);
    this.#emit(b.id, (h) => h.removed?.({ id: l.id }));
  }
}

/** What an agent finds, by agent and model: no confidences, high precision, recall by how capable the model is. */
export function agentSkill(agent: 'claude' | 'codex', model: string | null, effort: string, salt: string): Skill {
  const r = rng(hash(`agent:${salt}`));
  const q = agent === 'claude' ? ({ haiku: 0.62, sonnet: 0.72, opus: 0.78 } as Record<string, number>)[model ?? 'haiku'] ?? 0.66 : 0.6;
  const lift = effort === 'high' ? 0.04 : effort === 'medium' ? 0.02 : 0;
  return {
    map: 0.22 + (q + lift) * 0.22 + r() * 0.02,
    fragility: 0.55,
    fragile: { color_cast: 0.4, low_contrast: 0.6 },
    ece: 0,
    iou: 0.7 + r() * 0.03,
    selfRecall: 0,
    point: { precision: 0.88 + (q + lift) * 0.08, recall: q + lift - 0.04 + r() * 0.04 },
  };
}

/** What an agent's calls use for `passes` image passes, as agent labeling's calls do (./label). */
export function agentUsage(agent: 'claude' | 'codex', model: string | null, effort: string, passes: number, description: string, salt: string): LabelUsage {
  const per = PER_CALL[agent]!;
  const calls = Math.ceil(passes / per);
  const fixed = agent === 'claude' ? 420 + Math.round(description.length / 4) : 10_200;
  const input = calls * fixed + passes * 1012 + (hash(salt) % 400);
  const cached = Math.round((calls - 1) * fixed * 0.86);
  const reasoning = ({ low: 0, medium: 260, high: 820 } as Record<string, number>)[effort] ?? 0;
  const output = calls * 48 + passes * (5 * 23 + reasoning);
  const price = agent === 'claude' ? ({ sonnet: [3, 15], haiku: [1, 5], opus: [15, 75] } as Record<string, number[]>)[model ?? 'haiku'] : undefined;
  const cost = price ? Math.round((((input - cached) * price[0]! + cached * price[0]! * 0.1 + output * price[1]!) / 1e6) * 1e6) / 1e6 : null;
  return { calls, input_tokens: input, cached_input_tokens: cached, output_tokens: output, cost_usd: cost };
}

/** An external contestant's skill, from how its unchanged-image predictions match the truth (IoU 0.5, same class). */
function externalSkill(b: MBench, predictions: readonly BenchPrediction[], scored: boolean): Skill {
  let right = 0;
  let found = 0;
  let truths = 0;
  let ious = 0;
  const byImage = new Map(b.images.map((i) => [i.id, i]));
  for (const p of predictions) {
    if (p.condition) continue;
    const im = byImage.get(p.image);
    if (!im) continue;
    truths += im.boxes.length;
    const used = new Set<Box>();
    for (const d of [...p.boxes].sort((x, y) => (y.score ?? 1) - (x.score ?? 1))) {
      found++;
      let hit: Box | null = null;
      let best = 0.5;
      for (const t of im.boxes) {
        if (t.class !== d.class || used.has(t)) continue;
        const v = iou(t, d);
        if (v >= best) {
          best = v;
          hit = t;
        }
      }
      if (hit) {
        used.add(hit);
        right++;
        ious += best;
      }
    }
  }
  const precision = found ? right / found : 0;
  const recall = truths ? right / truths : 0;
  const f = f1(precision, recall);
  return {
    map: r4(Math.max(0.01, f * (scored ? 0.68 : 0.5))),
    fragility: 0.9,
    ece: 0.07,
    iou: right ? ious / right : 0.6,
    selfRecall: scored ? Math.max(0, recall - 0.15) : 0,
    point: { precision, recall },
  };
}

/** For ./benchSeed: an image list made from a source, as `create` makes it, sampled with a seed. */
export function sampleImages(images: Omit<BenchImg, 'id'>[], seed: number, salt: string, max: number): BenchImg[] {
  const r = rng(hash(`bench:${seed}:${salt}`));
  const order = images.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [order[i], order[j]] = [order[j]!, order[i]!];
  }
  return order.slice(0, max).map((k, n) => ({ ...images[k]!, id: String(n + 1) }));
}

/** A mock dataset's images as benchmark images (all of them), for a spec the benchmark owns (the pond camera). */
export function specImages(spec: MockDatasetSpec): Omit<BenchImg, 'id'>[] {
  return Array.from({ length: total(spec) }, (_, i) => {
    const d = image(spec, i);
    return { file: `${spec.name}-${String(i + 1).padStart(4, '0')}.jpg`, width: d.width, height: d.height, boxes: d.boxes, from: { spec, index: i } };
  });
}


