// The benchmark mock's state (./bench): its benchmarks, entries and loops, how they read as the API's records, and
// time: whatever is due is applied when asked (`advance`) and told on its benchmark's stream. MockBench (./bench)
// extends it with the endpoints.

import { ApiError, type BenchEventHandlers } from '../client';
import type { Bench, BenchCatalog, BenchEntry, BenchLoop, LabelUsage } from '../types';
import { clonePlan } from '../../bench/plan';
import { isEntryActive, rankEntries } from '../../bench/rules';
import { catalog } from './benchCatalog';
import { generateResult, type Skill } from './benchResults';
import { hash } from './rng';
import {
  MS_PER_PASS,
  MS_PER_IMAGE,
  PARAMS,
  LABEL_MS,
  TRAIN_MS,
  type MBench,
  type MEntry,
  type MLoop,
  type BenchHost,
  iso,
  hex,
  fingerprintOf,
  studentName,
  MIN_LABELED_IMAGES,
  poolLabels,
  starvedError,
  noThresholdError,
  OPEN_IDLE_MS,
  labelsPerImage,
} from './benchRecords';

/** The benchmark mock's records, which MockBenchState and MockBench (./bench) both read and change. */
export interface MockBenchLists {
  benches: MBench[];
  entries: MEntry[];
  loops: MLoop[];
}

export class MockBenchState {
  /** The benchmarks, entries and loops, shared with MockBench (./bench). */
  protected readonly state: MockBenchLists = { benches: [], entries: [], loops: [] };
  #subs = new Map<string, Set<BenchEventHandlers>>();
  #next = 1;
  /** Scenario `bench-agent-fails`: a design request's answer is not a plan. */
  designFails = false;

  constructor(
    readonly t0: number,
    readonly host: BenchHost,
  ) {}

  // ---------- state the seed (./benchSeed) and the API both build with ----------

  addBench(b: MBench) {
    this.state.benches.unshift(b);
    this.state.benches.sort((x, y) => y.createdAt - x.createdAt);
  }

  addEntry(e: MEntry) {
    this.state.entries.push(e);
  }

  addLoop(l: MLoop) {
    this.state.loops.unshift(l);
  }

  /** An id that is not taken: `prefix_` and `digits` hex digits (a new benchmark's are 12, as the server's are now). */
  id(prefix: string, salt: string, digits = 4): string {
    for (let k = 0; ; k++) {
      const n = this.#next++;
      const id =
        digits <= 4
          ? `${prefix}_${hex(hash(`${salt}:${n}:${k}`))}`
          : `${prefix}_${Array.from({ length: Math.ceil(digits / 8) }, (_, i) => hex(hash(`${salt}:${n}:${k}:${i}`), 8)).join('').slice(0, digits)}`;
      if (!this.state.benches.some((b) => b.id === id) && !this.state.entries.some((e) => e.id === id) && !this.state.loops.some((l) => l.id === id)) return id;
    }
  }

  // ---------- reading ----------

  protected bench(id: string): MBench {
    const b = this.state.benches.find((x) => x.id === id);
    if (!b) throw new ApiError(404, 'bench_not_found', `No benchmark with id ${id}.`, 'List them with GET /api/bench/benchmarks.');
    return b;
  }

  protected entryOf(b: MBench, id: string): MEntry {
    const e = this.state.entries.find((x) => x.id === id && x.benchId === b.id);
    if (!e) throw new ApiError(404, 'bench_entry_not_found', `${b.name} has no entry ${id}.`);
    return e;
  }

  protected loopOf(b: MBench, id: string): MLoop {
    const l = this.state.loops.find((x) => x.id === id && x.benchId === b.id);
    if (!l) throw new ApiError(404, 'bench_loop_not_found', `${b.name} has no loop ${id}.`);
    return l;
  }

  protected entriesOf(b: MBench) {
    return this.state.entries.filter((e) => e.benchId === b.id);
  }

  toBench(b: MBench): Bench {
    const entries = rankEntries(this.entriesOf(b).map((e) => this.toEntry(e)));
    // The leader as the server names it: done, on the benchmark's own fingerprint (one graded before the engine
    // changed ranks below every current entry, so it never leads).
    const fp = fingerprintOf(b);
    const top = entries.find((e) => e.status === 'done' && e.score && e.fingerprint === fp);
    const class_boxes = b.classes.map(() => 0);
    for (const im of b.images) for (const x of im.boxes) class_boxes[x.class] = (class_boxes[x.class] ?? 0) + 1;
    return {
      id: b.id,
      name: b.name,
      description: b.description,
      classes: [...b.classes],
      created: iso(b.createdAt),
      source: { ...b.source },
      source_name: b.sourceName,
      images: b.images.length,
      boxes: class_boxes.reduce((a, n) => a + n, 0),
      class_boxes,
      plan: clonePlan(b.plan),
      seed: b.seed,
      fingerprint: fp,
      entries: entries.length,
      best: top?.score ? { entry_id: top.id, name: top.name, score: top.score.total, partial: top.score.partial } : null,
      // Exactly when DELETE answers 409 bench_active: an entry queued, running or stopping, or a loop running.
      active: entries.some((e) => isEntryActive(e)) || this.state.loops.some((l) => l.benchId === b.id && !l.end),
    };
  }

  protected status(e: MEntry, now: number): BenchEntry['status'] {
    if (e.end) return e.end.status;
    if (e.startedAt === null || now < e.startedAt) return 'queued';
    return e.stopAt !== null ? 'stopping' : 'running';
  }

  /** The image passes of an entry: its images, then once per condition and severity of the plan. */
  protected passes(e: MEntry, b: MBench): { condition: string | null; severity: number | null }[] {
    const out: { condition: string | null; severity: number | null }[] = [{ condition: null, severity: null }];
    if (!e.conditions) return out;
    for (const c of b.plan.conditions) if (!e.conditionsRun || e.conditionsRun.includes(c.condition)) for (const s of c.severities) out.push({ condition: c.condition, severity: s });
    return out;
  }

  toEntry(e: MEntry, now = Date.now()): BenchEntry {
    const b = this.bench(e.benchId);
    const passes = this.passes(e, b);
    // An open external entry: the plan's every pass is what it may receive, the records received so far its progress.
    const total = e.open ? e.images * (1 + b.plan.conditions.reduce((n, c) => n + c.severities.length, 0)) : e.images * passes.length;
    const status = this.status(e, now);
    let done = 0;
    if (e.open) done = e.open.records.length;
    else if (e.end?.status === 'done') done = total;
    else if (e.startedAt !== null && now > e.startedAt) {
      const until = Math.min(now, e.end?.at ?? now, e.stopAt ?? now);
      done = Math.min(total, Math.floor(((until - e.startedAt) / e.duration) * total));
    }
    const pass = status === 'done' || e.open ? null : passes[Math.min(passes.length - 1, Math.floor(done / Math.max(1, e.images)))];
    const r = e.end?.status === 'done' ? e.result : null;
    return {
      id: e.id,
      bench_id: e.benchId,
      name: e.name,
      contestant: { ...e.contestant },
      status,
      created: iso(e.createdAt),
      finished: e.end ? iso(e.end.at) : null,
      progress: { done, total, condition: pass?.condition ?? null, severity: pass?.severity ?? null },
      images: e.images,
      conditions: e.conditions,
      scored: e.scored,
      fingerprint: e.fingerprint,
      missing_classes: e.missing.map((i) => b.classes[i] ?? String(i)),
      score: r ? r.score : null,
      map50_95: r ? r.accuracy.map50_95 : null,
      ci95: r ? [r.accuracy.ci95[0], r.accuracy.ci95[1]] : null,
      // The plan's conditions it ran at every planned severity: what its robustness is over.
      conditions_run: r ? (r.robustness?.conditions.length ?? 0) : 0,
      speed: r && e.speed ? { ...e.speed } : null,
      usage: e.usage ? this.#usageAt(e, done, total) : null,
      error: e.end?.error ?? null,
      ...(e.end?.hint ? { hint: e.end.hint } : {}),
    };
  }

  /** An agent's usage so far: the calls it has made, in proportion to the images answered. */
  #usageAt(e: MEntry, done: number, total: number): LabelUsage {
    const u = e.usage as LabelUsage;
    if (e.end?.status === 'done' || total === 0) return { ...u };
    const f = done / total;
    const calls = e.end?.status === 'failed' ? 1 : Math.round(u.calls * f);
    return {
      calls,
      input_tokens: Math.round(u.input_tokens * f),
      cached_input_tokens: Math.round(u.cached_input_tokens * f),
      output_tokens: Math.round(u.output_tokens * f),
      cost_usd: u.cost_usd === null ? null : Math.round(u.cost_usd * f * 1e6) / 1e6,
    };
  }

  toLoop(l: MLoop): BenchLoop {
    return {
      id: l.id,
      bench_id: l.benchId,
      name: l.name,
      status: l.end ? l.end.status : l.stopAt !== null ? 'stopping' : 'running',
      phase: l.end ? null : (l.step?.phase ?? 'benchmarking'),
      created: iso(l.createdAt),
      finished: l.end ? iso(l.end.at) : null,
      config: structuredClone(l.config),
      generations: l.generations.map((g) => ({ ...g, labels: g.labels ? { ...g.labels } : null, ci95: g.ci95 ? [g.ci95[0], g.ci95[1]] : null })),
      verdict: l.end?.status === 'done' ? this.#verdict(l) : null,
      error: l.end?.error ?? null,
      ...(l.end?.hint ? { hint: l.end.hint } : {}),
    };
  }

  #verdict(l: MLoop): BenchLoop['verdict'] {
    const first = l.generations[0];
    const last = l.generations.at(-1);
    if (!first?.ci95 || last?.map50_95 == null || l.generations.length < 2) return null;
    return last.map50_95 > first.ci95[1] ? 'improved' : last.map50_95 < first.ci95[0] ? 'regressed' : 'held';
  }

  catalog(): BenchCatalog {
    return catalog();
  }

  list(now: number): Bench[] {
    this.advance(now);
    return this.state.benches.map((b) => this.toBench(b));
  }

  get(id: string, now: number): Bench {
    const b = this.bench(id);
    this.advance(now);
    return this.toBench(b);
  }

  // ---------- time ----------

  /**
   * Applies what is due by `now`, in time order across every benchmark: model entries take the GPU one at a time (the
   * oldest waiting first), entries end, loops move through their phases; each change is told on its benchmark's
   * stream, and a running entry's progress at most twice a second.
   */
  advance(now: number) {
    // An open external entry that has had no part for OPEN_IDLE_MS fails (its sender is gone).
    for (const e of this.state.entries) {
      if (!e.open || now - e.open.lastPart < OPEN_IDLE_MS) continue;
      e.open = null;
      e.end = { status: 'failed', at: now, error: 'no predictions arrived for 30 minutes', hint: 'start it again (POST /api/bench/benchmarks/{id}/entries with contestant.open: true) and send its parts without long pauses' };
      this.#tell(e, now);
    }
    for (let guard = 0; guard < 200; guard++) {
      if (!this.#step(now)) break;
    }
    for (const e of this.state.entries) {
      if (e.end || e.startedAt === null || now < e.startedAt || now - e.lastEmit < 500) continue;
      e.lastEmit = now;
      const out = this.toEntry(e, now);
      this.emit(e.benchId, (h) => h.entry?.(out));
    }
  }

  /** One change that is due (the earliest), applied; false when nothing is. */
  #step(now: number): boolean {
    // The earliest end of an entry that runs.
    let first: { e: MEntry; at: number; kind: 'done' | 'failed' | 'stopped' } | null = null;
    for (const e of this.state.entries) {
      if (e.end || e.startedAt === null) continue;
      const candidates: { at: number; kind: 'done' | 'failed' | 'stopped' }[] = [{ at: e.startedAt + e.duration, kind: 'done' }];
      if (e.failAfter) candidates.push({ at: e.startedAt + e.failAfter.ms, kind: 'failed' });
      if (e.stopAt !== null) candidates.push({ at: e.stopAt, kind: 'stopped' });
      const c = candidates.sort((a, b) => a.at - b.at)[0]!;
      if (c.at <= now && (!first || c.at < first.at)) first = { e, ...c };
    }
    // The slot: a waiting model entry starts once no model entry runs.
    const busy = this.state.entries.some((e) => e.slot && !e.end && e.startedAt !== null);
    const waiting = this.state.entries.filter((e) => e.slot && !e.end && e.startedAt === null).sort((a, b) => a.createdAt - b.createdAt)[0];
    if (!busy && waiting && waiting.createdAt + 400 <= now) {
      const lastEnd = Math.max(0, ...this.state.entries.filter((e) => e.slot && e.end).map((e) => e.end!.at));
      waiting.startedAt = Math.max(waiting.createdAt + 400, lastEnd);
      this.#tell(waiting, now);
      return true;
    }
    // A loop's phase that is due before the next entry ends.
    const loopAt = this.state.loops.filter((l) => !l.end).map((l) => ({ l, at: this.#loopDue(l) })).filter((x) => x.at !== null && x.at <= now).sort((a, b) => a.at! - b.at!)[0];
    if (loopAt && (!first || loopAt.at! <= first.at)) {
      this.advanceLoop(loopAt.l, this.bench(loopAt.l.benchId), now);
      return true;
    }
    if (!first) return false;
    const { e, at, kind } = first;
    if (kind === 'done') this.finish(e, at);
    else if (kind === 'failed') e.end = { status: 'failed', at, error: e.failAfter!.error, hint: e.failAfter!.hint };
    else e.end = { status: 'stopped', at, error: null, hint: null };
    this.#tell(e, now);
    const b = this.state.benches.find((x) => x.id === e.benchId);
    if (b) {
      const bench = this.toBench(b);
      this.emit(b.id, (h) => h.bench?.(bench));
    }
    return true;
  }

  #tell(e: MEntry, now: number) {
    e.lastEmit = now;
    const out = this.toEntry(e, now);
    this.emit(e.benchId, (h) => h.entry?.(out));
  }

  /** Scores an entry that ran to its end at `at`. */
  protected finish(e: MEntry, at: number) {
    const b = this.bench(e.benchId);
    e.result = generateResult({
      benchId: b.id,
      entryId: e.id,
      classes: b.classes.length,
      plan: b.plan,
      benchImages: b.images.length,
      images: b.images.slice(0, e.images),
      conditions: e.conditions,
      conditionsRun: e.conditionsRun,
      scored: e.scored,
      missing: new Set(e.missing),
      skill: e.skill,
    });
    e.end = { status: 'done', at, error: null, hint: null };
  }

  /** When the loop's next phase is due, or null while it waits on an entry. */
  #loopDue(l: MLoop): number | null {
    if (l.stopAt !== null) return l.stopAt;
    const s = l.step;
    if (!s) return l.createdAt;
    if (s.phase === 'labeling') return s.at + LABEL_MS;
    if (s.phase === 'training') return s.at + TRAIN_MS;
    const e = s.entryId ? this.state.entries.find((x) => x.id === s.entryId) : undefined;
    return e?.end ? e.end.at : null;
  }

  /** Moves a loop on by one phase, if one is due. */
  protected advanceLoop(l: MLoop, b: MBench, now: number) {
    if (l.end) return;
    const tell = () => {
      const out = this.toLoop(l);
      this.emit(b.id, (h) => h.loop?.(out));
    };
    if (l.stopAt !== null && l.stopAt <= now) {
      l.end = { status: 'stopped', at: l.stopAt, error: null, hint: null };
      l.step = null;
      return tell();
    }
    const s = l.step;
    if (!s) {
      // Generation 0: the teacher's entry, an existing finished complete one on this fingerprint when there is one.
      const fp = fingerprintOf(b);
      const had = this.state.entries.find(
        (e) => e.benchId === b.id && e.contestant.kind === 'model' && e.contestant.model_id === l.teacher.id && e.end?.status === 'done' && e.fingerprint === fp && !e.result?.score.partial,
      );
      const entry = had ?? this.#loopEntry(l, b, l.teacher.name, l.teacher.id, l.teacher.arch, l.teacher.skill, l.createdAt);
      l.generations = [{ generation: 0, labels: null, run_id: null, model_id: l.teacher.id, model_name: l.teacher.name, entry_id: entry.id, score: null, map50_95: null, ci95: null }];
      l.step = { gen: 0, phase: 'benchmarking', at: l.createdAt, entryId: entry.id };
      this.#fillGeneration(l, 0, entry);
      if (had) this.#afterBenchmark(l, l.createdAt + 300);
      return tell();
    }
    const at = this.#loopDue(l);
    if (at === null || at > now) return;
    if (s.phase === 'benchmarking') {
      const e = this.state.entries.find((x) => x.id === s.entryId);
      if (e?.end?.status !== 'done') {
        l.end = { status: 'failed', at, error: `The benchmark entry of ${l.generations[s.gen]?.model_name ?? 'the model'} did not finish (${e?.end?.status ?? 'gone'}).`, hint: 'Start the loop again.' };
        l.step = null;
        return tell();
      }
      this.#fillGeneration(l, s.gen, e);
      this.#afterBenchmark(l, at);
    } else if (s.phase === 'labeling') {
      // The teacher's labels of the pool, from its entry's curve at the confidence (as the server keeps them): the
      // images it kept nothing on are left out, and with fewer than MIN_LABELED_IMAGES the loop fails here.
      const teacher = l.generations[s.gen - 1]!;
      const te = this.state.entries.find((x) => x.id === teacher.entry_id);
      const threshold = l.config.min_confidence ?? te?.result?.self_labeling.threshold ?? null;
      if (threshold === null) {
        // Checked when the generation starts (#afterBenchmark); here only if the entry changed under it.
        const why = noThresholdError(teacher.model_name ?? l.teacher.name);
        l.end = { status: 'failed', at, error: why.error, hint: why.hint };
        l.step = null;
        return tell();
      }
      const imgs = l.pool.images - l.pool.excluded;
      const per = te?.result ? labelsPerImage(te.result, te.images, threshold) : 0;
      const kept = poolLabels(imgs, per, `${l.id}:${s.gen}`);
      if (kept.labeled < MIN_LABELED_IMAGES) {
        const why = starvedError(teacher.model_name ?? l.teacher.name, kept.boxes, kept.labeled, imgs, threshold);
        l.end = { status: 'failed', at, error: why.error, hint: why.hint };
        l.step = null;
        return tell();
      }
      l.generations[s.gen]!.labels = { threshold, images: imgs, labeled_images: kept.labeled, boxes: kept.boxes, excluded: l.pool.excluded, dataset_id: `ds_${hex(hash(`${l.id}:${s.gen}:ds`))}` };
      l.generations[s.gen]!.run_id = `run_${hex(hash(`${l.id}:${s.gen}:run`))}`;
      l.step = { gen: s.gen, phase: 'training', at, entryId: null };
    } else {
      const g = l.generations[s.gen]!;
      g.model_id = `m_${hex(hash(`${l.id}:${s.gen}:model`))}`;
      g.model_name = studentName(l.name, s.gen);
      const prev = l.generations[s.gen - 1]!;
      const prevEntry = this.state.entries.find((x) => x.id === prev.entry_id);
      const skill: Skill = { ...(prevEntry?.skill ?? l.teacher.skill), map: (prev.map50_95 ?? l.teacher.skill.map) + (l.gains[s.gen - 1] ?? 0.01) };
      const e = this.#loopEntry(l, b, g.model_name, g.model_id, l.student, skill, at);
      g.entry_id = e.id;
      l.step = { gen: s.gen, phase: 'benchmarking', at, entryId: e.id };
    }
    tell();
  }

  /** Generation `gen` has its score: the next one's labeling, or the loop's end. */
  #afterBenchmark(l: MLoop, at: number) {
    const gens = l.config.generations ?? 1;
    const s = l.step!;
    if (s.gen >= gens) {
      l.end = { status: 'done', at, error: null, hint: null };
      l.step = null;
      return;
    }
    l.generations.push({ generation: s.gen + 1, labels: null, run_id: null, model_id: null, model_name: null, entry_id: null, score: null, map50_95: null, ci95: null });
    l.step = { gen: s.gen + 1, phase: 'labeling', at, entryId: null };
    // No confidence to keep labels from (none set, none found by the teacher's entry): it fails before labeling.
    const te = this.state.entries.find((x) => x.id === l.generations[s.gen]?.entry_id);
    if (l.config.min_confidence == null && te?.result && te.result.self_labeling.threshold === null) {
      const why = noThresholdError(l.generations[s.gen]?.model_name ?? l.teacher.name);
      l.end = { status: 'failed', at, error: why.error, hint: why.hint };
      l.step = null;
    }
  }

  #fillGeneration(l: MLoop, gen: number, e: MEntry) {
    const g = l.generations[gen]!;
    if (e.end?.status !== 'done' || !e.result) return;
    g.score = e.result.score.total;
    g.map50_95 = e.result.accuracy.map50_95;
    g.ci95 = [e.result.accuracy.ci95[0], e.result.accuracy.ci95[1]];
  }

  /** A loop's model entry: complete (every image, every condition), queued with the other model entries. */
  #loopEntry(l: MLoop, b: MBench, name: string, modelId: string, arch: string, skill: Skill, at: number): MEntry {
    const e: MEntry = {
      id: this.id('be', `${l.id}:${name}`),
      benchId: b.id,
      name,
      contestant: { kind: 'model', model_id: modelId, arch: arch as 'folo-n', params: PARAMS[arch] ?? 9_000_000, imgsz: l.teacher.imgsz },
      createdAt: at,
      images: b.images.length,
      conditions: true,
      scored: true,
      fingerprint: fingerprintOf(b),
      missing: [],
      skill,
      conditionsRun: null,
      slot: true,
      startedAt: null,
      duration: 0,
      stopAt: null,
      failAfter: null,
      end: null,
      result: null,
      usage: null,
      speed: { device: 'gpu:0', ms_per_image: MS_PER_IMAGE[arch] ?? 5 },
      lastEmit: 0,
    };
    e.duration = e.images * this.passes(e, b).length * (MS_PER_PASS[arch] ?? 5) * 0.4;
    this.addEntry(e);
    const out = this.toEntry(e, at);
    this.emit(b.id, (h) => h.entry?.(out));
    return e;
  }

  // ---------- events ----------

  subscribe(id: string, h: BenchEventHandlers): () => void {
    const subs = this.#subs.get(id) ?? new Set<BenchEventHandlers>();
    this.#subs.set(id, subs);
    subs.add(h);
    return () => subs.delete(h);
  }

  tick(id: string, now: number) {
    const b = this.state.benches.find((x) => x.id === id);
    if (b) this.advance(now);
  }

  protected emit(id: string, f: (h: BenchEventHandlers) => void) {
    for (const h of this.#subs.get(id) ?? []) f(h);
  }
}
