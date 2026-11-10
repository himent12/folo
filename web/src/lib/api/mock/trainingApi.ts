// The mock's Training runs and Cloud compute endpoints (docs/API.md): runs, the queue, their metrics, samples, logs and
// streams, and the cloud connections, offers and instances a run can train on. Extended by MockApi (./index).

import { ApiError, type RunEventHandlers } from '../client';
import type {
  CliCommand,
  CloudConnection,
  CloudInstance,
  CloudOffer,
  CloudOfferList,
  CloudProvider,
  ConnectCloud,
  LogLine,
  QueueRequest,
  Run,
  RunCloud,
  RunLogs,
  RunEstimate,
  RunMetrics,
  RunSamples,
  Sample,
  TrainConfig,
} from '../types';
import { predictBoxes } from './boxes';
import { image, specs, total, type MockDatasetSpec } from './datasets';
import { hash, rng } from './rng';
import {
  archInfo,
  epochAt,
  cloudFinish,
  cloudOf,
  epochLog,
  history,
  loopStart,
  optimizerSteps,
  progressAt,
  startLogs,
  stepAt,
  toRun,
  trainEnd,
  type RunProfile,
} from './runs';
import {
  cloudLog,
  instanceAt,
  PROVIDERS,
  REMOTE_OFFER,
  RUNPOD_OFFERS,
  secretHint,
  trainAt,
  type CloudTimeline,
} from './cloud';
import { applyQueueMove } from '../queue';
import { trainCommand } from '../../trainCli';
import { MockServer } from './server';
import { STALE_AFTER } from './scenario';
import { DEVICE, withResolved } from './runSeed';

/** folo-server's hint on `501 not_supported` (engine::engine_error), verbatim. */
const AMP_HINT =
  'set `amp: false` to train in f32, or restart this server with tensor cores (`folo serve --tensor-cores`, or ' +
  "FOLO_GPU_TENSOR_CORES=1) on a GPU that has them; GET /api/system's `capabilities.amp` says when amp is available";

export class MockTrainingApi extends MockServer {
  // MockServer's private members these endpoints use (MockServer.shared), under their names there.
  get #t0() {
    return this.shared.t0();
  }
  get #runs() {
    return this.shared.runs();
  }
  set #runs(runs: RunProfile[]) {
    this.shared.setRuns(runs);
  }
  get #nextId() {
    return this.shared.nextId();
  }
  set #nextId(id: number) {
    this.shared.setNextId(id);
  }
  get #queue() {
    return this.shared.queue();
  }
  set #queue(queue: string[]) {
    this.shared.setQueue(queue);
  }
  get #throughput() {
    return this.shared.throughput();
  }
  get #created() {
    return this.shared.created();
  }
  get #connections() {
    return this.shared.connections();
  }
  set #connections(connections: CloudConnection[]) {
    this.shared.setConnections(connections);
  }
  get #cloudOn() {
    return this.shared.cloudOn();
  }
  set #unplugged(unplugged: boolean) {
    this.shared.setUnplugged(unplugged);
  }
  set #ampRefused(refused: boolean) {
    this.shared.setAmpRefused(refused);
  }
  #call = this.shared.call;
  #schedule = this.shared.schedule;
  #spec = this.shared.spec;
  #noRoute = this.shared.noRoute;
  #gpus = this.shared.gpus;
  #capabilities = this.shared.capabilities;
  #models = this.shared.models;

  /** Live subscribers per run, for log lines of actions outside a run's timeline (queue moves). */
  #subs = new Map<string, Set<RunEventHandlers>>();
  /** Those log lines, kept for `/logs`. */
  #notes = new Map<string, LogLine[]>();

  /** A run snapshot with its place in the queue. */
  #toRun(p: RunProfile, now: number): Run {
    const i = this.#queue.indexOf(p.id);
    return { ...toRun(p, now), queue_position: i < 0 ? null : i + 1 };
  }

  /** Logs `message` for run `id`: kept for `/logs` and sent to its live subscribers. */
  #note(id: string, message: string) {
    const line: LogLine = { t: new Date().toISOString(), level: 'info', message };
    this.#notes.set(id, [...(this.#notes.get(id) ?? []), line]);
    for (const h of this.#subs.get(id) ?? []) h.log?.(line);
  }

  #profile(id: string): RunProfile {
    const p = this.#runs.find((r) => r.id === id);
    if (!p) throw new ApiError(404, 'run_not_found', `No run with id ${id}.`);
    return p;
  }

  // ---------- runs ----------

  runs(): Promise<Run[]> {
    return this.#call(() => {
      const now = Date.now();
      return this.#runs.map((p) => this.#toRun(p, now)).sort((a, b) => b.created.localeCompare(a.created));
    });
  }

  run(id: string): Promise<Run> {
    return this.#call(() => this.#toRun(this.#profile(id), Date.now()));
  }

  /**
   * What the server does to a request's weights before it trains (folo-server's create_run): they must exist, and a
   * fine-tune trains their architecture, as `folo train --weights` does.
   */
  #withWeights(c: TrainConfig): TrainConfig {
    if (!c.weights) return c;
    const m = this.#models().find((x) => x.id === c.weights);
    if (!m) throw new ApiError(404, 'model_not_found', `No model with id ${c.weights}.`);
    return { ...c, arch: m.arch };
  }

  /**
   * `501 not_supported` for an `amp` run this server cannot train (docs/API.md): refused before anything is queued,
   * with `capabilities.amp.reason` as the message, as the real server and folo-server's mock do.
   */
  #checkAmp(c: TrainConfig) {
    if (!c.amp) return;
    // `amp-refused`: the server was restarted without tensor cores since the page asked.
    if (this.scenario === 'amp-refused') this.#ampRefused = true;
    const caps = this.#capabilities();
    if (c.device !== 'cpu' && caps.amp.available) return;
    const reason =
      c.device === 'cpu'
        ? 'mixed-precision training (amp) needs a GPU with tensor cores (f16 cooperative matrices): the run is on the CPU; train without amp (f32), or on such a GPU'
        : (caps.amp.reason ?? 'mixed-precision training (amp) is not available on this server');
    throw new ApiError(501, 'not_supported', reason, AMP_HINT);
  }

  /**
   * `folo train` for `c` as the server writes it (`RunEstimate.cli`, `GET /api/runs/{id}/cli`, quoted for PowerShell):
   * published weights by their name, another model by its file under the data dir (made up: mock models have no
   * files, as on folo-server's mock).
   */
  #cli(c: TrainConfig): CliCommand {
    const ds = this.#spec(c.dataset_id);
    const m = c.weights ? this.#models().find((x) => x.id === c.weights) : undefined;
    const cmd = trainCommand(c, ds.path, m && m.source !== 'published' ? { ...m, path: `D:/folo/models/${m.name}.folo` } : m);
    return { cli: cmd.line, argv: ['folo', ...cmd.argv] };
  }

  runCli(id: string): Promise<CliCommand> {
    return this.#call(() => {
      if (this.scenario === 'no-throughput') throw this.#noRoute(`GET /api/runs/${id}/cli`);
      return this.#cli(this.#profile(id).config);
    }, 80);
  }

  createRun(body: TrainConfig): Promise<Run> {
    return this.#call(() => {
      const ds = this.#spec(body.dataset_id);
      const c = this.#withWeights(body);
      const cloud = this.#compute(c);
      if (cloud) return this.#createCloudRun(c, ds, cloud);
      // `gpu-unplugged`: the second adapter goes away as soon as a run asks for it.
      if (this.scenario === 'gpu-unplugged' && c.device === 'gpu:1') this.#unplugged = true;
      const gpu = /^gpu:(\d+)$/.exec(c.device);
      if (gpu && !this.#gpus().some((g) => g.ordinal === Number(gpu[1]))) throw this.#noDevice(c.device);
      // The real server's body error for a head it does not know (serde: `o2m` or `o2o`; missing means `o2m`).
      if (c.head !== undefined && c.head !== 'o2m' && c.head !== 'o2o')
        throw new ApiError(400, 'invalid_body', `field \`head\`: unknown variant \`${String(c.head)}\`, expected \`o2m\` or \`o2o\``);
      this.#checkAmp(c);
      const now = Date.now();
      const id = `run_${(hash(c.name + now + this.#nextId++) & 0xffff).toString(16).padStart(4, '0')}`;
      const sizeIdx = ['folo-n', 'folo-s', 'folo-m', 'folo-l', 'folo-x'].indexOf(c.arch);
      const p: RunProfile = withResolved({
        id,
        name: c.name,
        config: c,
        classes: ds.classes.length,
        classNames: [...ds.classes],
        classEase: ds.classes.map((_, i) => 0.85 + 0.3 * rng(hash(id) + i)()),
        stepsPerEpoch: optimizerSteps(ds.splits.train, c.batch),
        epochSeconds: 600 / Math.max(1, c.epochs),
        mapMax: 0.55 + sizeIdx * 0.04,
        tau: Math.max(4, c.epochs / 7),
        imgsPerS: [520, 320, 170, 130, 70][sizeIdx] ?? 200,
        createdAt: now,
        // Queued: the worker starts it once the training slot is free (#schedule).
        startedAt: null,
        modelId: `m_${id.slice(4)}`,
        device: c.device === 'cpu' ? 'cpu (AMD Ryzen 9 7950X)' : DEVICE,
      }, ds.splits.train);
      this.#runs.unshift(p);
      this.#created.add(p.id);
      this.#queue.push(p.id);
      this.#schedule(now);
      return this.#toRun(p, now);
    }, 300);
  }

  estimateRun(body: TrainConfig): Promise<RunEstimate> {
    return this.#call(() => {
      if (this.scenario === 'no-throughput') throw this.#noRoute('POST /api/runs/estimate');
      // The same checks as POST /api/runs; nothing is created.
      const ds = this.#spec(body.dataset_id);
      const gpu = /^gpu:(\d+)$/.exec(body.device);
      if (gpu && !this.#gpus().some((g) => g.ordinal === Number(gpu[1]))) throw this.#noDevice(body.device);
      const c = this.#withWeights(body);
      const cloud = this.#compute(c, false);
      const est = { ...this.#throughput.estimate(c, ds.splits.train, ds.splits.val, cloud?.offer.gpu), cli: this.#cli(c).cli };
      const price = cloud?.offer.price_per_hour;
      if (!cloud || price === null || price === undefined) return est;
      // docs/API.md, `RunEstimate.cost`: the price over the training time (booting and the upload come on top).
      const usd = (s: number) => Math.round(price * (s / 3600) * 100) / 100;
      return {
        ...est,
        cost: {
          price_per_hour: price,
          est_usd: usd(est.est_total_s),
          est_usd_low: usd(est.est_total_s_low),
          est_usd_high: usd(est.est_total_s_high),
          max_usd: Math.round(price * cloud.hours * 100) / 100,
        },
      };
    }, 120);
  }

  /**
   * A config's `compute`, checked as `POST /api/runs` checks it (docs/API.md, Cloud runs): the connection, its offer,
   * the time limit and `device: auto`. `refuse`: an offer with nothing available is `409 cloud_refused`, as launching
   * it would be (an estimate launches nothing). Null for a run on this machine, and on a server without the feature,
   * which ignores a field it does not know.
   */
  #compute(c: TrainConfig, refuse = true): { conn: CloudConnection; offer: CloudOffer; hours: number } | null {
    const want = c.compute;
    if (!want || !this.#cloudOn) return null;
    const conn = this.#connection(want.connection_id);
    if (conn.status === 'error') throw new ApiError(422, 'cloud_auth_failed', conn.error ?? `${conn.name} rejected the credentials`, 'Check the connection in Settings › Cloud.');
    const offer = this.#offersOf(conn).find((o) => o.id === want.offer_id);
    if (!offer) throw new ApiError(404, 'offer_not_found', `${conn.name} has no offer '${want.offer_id}' (any more)`, 'List its offers again: GET /api/cloud/connections/{id}/offers.');
    const hours = want.max_hours ?? 12;
    if (!(hours >= 0.25 && hours <= 720)) throw new ApiError(400, 'invalid_config', `\`compute.max_hours\` must be from 0.25 to 720 hours, not ${hours}`);
    if (c.device !== 'auto') throw new ApiError(400, 'invalid_config', `\`device\` must be \`auto\` for a cloud run (the worker picks its GPU), not \`${c.device}\``);
    if (refuse && offer.availability === 'none')
      throw new ApiError(409, 'cloud_refused', `${conn.name} has no ${offer.gpu} (${offer.detail ?? offer.id}) available right now`, 'Pick another GPU, or try again later.');
    return { conn, offer, hours };
  }

  /**
   * A cloud run: out of the training queue, through every phase (about 2 s each of provisioning, booting and the
   * upload), then training like any mock run, on the worker, then its weights collected and its instance released.
   */
  #createCloudRun(c: TrainConfig, ds: MockDatasetSpec, cloud: { conn: CloudConnection; offer: CloudOffer; hours: number }): Run {
    const now = Date.now();
    const id = `run_${(hash(c.name + now + this.#nextId++) & 0xffff).toString(16).padStart(4, '0')}`;
    const nonce = (hash(`${id}:nonce`) & 0xffff).toString(16).padStart(4, '0');
    const sizeIdx = ['folo-n', 'folo-s', 'folo-m', 'folo-l', 'folo-x'].indexOf(c.arch);
    const provisions = PROVIDERS.find((x) => x.id === cloud.conn.provider)?.provisions ?? false;
    const timeline: CloudTimeline = {
      connectionId: cloud.conn.id,
      provider: cloud.conn.provider,
      offer: cloud.offer,
      maxHours: cloud.hours,
      instanceId: provisions ? `ci_${nonce}` : null,
      instanceName: `folo-${id}-${nonce}`,
      providerRef: (hash(`${id}:pod`) >>> 0).toString(36).padStart(7, '0'),
      launchedAt: now,
      // A remote server is not provisioned: it is asked at once.
      provisionMs: provisions ? 2000 : 0,
      bootMs: 2000,
      syncMs: 2000,
      // The dataset as FOLO sends it (about 90 kB an image), and best.folo with last.folo (16 bytes a parameter each).
      uploadBytes: total(ds) * 90_000,
      weightsBytes: archInfo[c.arch].params * 32,
    };
    const p: RunProfile = withResolved(
      {
        id,
        name: c.name,
        config: { ...c, compute: { ...c.compute!, max_hours: cloud.hours } },
        classes: ds.classes.length,
        classNames: [...ds.classes],
        classEase: ds.classes.map((_, i) => 0.85 + 0.3 * rng(hash(id) + i)()),
        stepsPerEpoch: optimizerSteps(ds.splits.train, c.batch),
        epochSeconds: 600 / Math.max(1, c.epochs),
        mapMax: 0.55 + sizeIdx * 0.04,
        tau: Math.max(4, c.epochs / 7),
        imgsPerS: [520, 320, 170, 130, 70][sizeIdx] ?? 200,
        createdAt: now,
        startedAt: trainAt(timeline),
        modelId: `m_${id.slice(4)}`,
        device: `NVIDIA GeForce ${cloud.offer.gpu} (vulkan)`,
        cloud: timeline,
      },
      ds.splits.train,
    );
    this.#runs.unshift(p);
    this.#created.add(p.id);
    return this.#toRun(p, now);
  }

  /** `404 device_not_found` in the real server's words. */
  #noDevice(device: string): ApiError {
    const gpus = this.#gpus();
    return new ApiError(
      404,
      'device_not_found',
      `device ${device} is not available: this machine has no such GPU; FOLO can use ${['cpu', ...gpus.map((g) => `gpu:${g.ordinal}`)].join(', ')}`,
      `this machine has ${gpus.map((g) => `gpu:${g.ordinal} (${g.name}, ${g.backend})`).join(', ')}, cpu; use device \`auto\` (the GPU when there is one) or \`cpu\``,
    );
  }

  /** Like `folo serve --mock`: resuming needs real checkpoints. */
  resumeRun(id: string): Promise<Run> {
    return this.#call(() => {
      this.#profile(id);
      throw new ApiError(501, 'not_implemented', 'Resuming runs needs a real server.', 'Start folo serve without --mock to resume from checkpoints.');
    }, 200);
  }

  stopRun(id: string): Promise<Run> {
    return this.#call(() => {
      const p = this.#profile(id);
      const now = Date.now();
      const { done, status } = progressAt(p, now);
      if (status !== 'running' && status !== 'queued')
        throw new ApiError(409, 'run_not_running', `Run ${p.name} is ${status}; only queued or running runs can be stopped.`);
      if (p.cloud) {
        // The worker's run stops, its checkpoint is collected and the instance released (before training, released at
        // once); the run says `stopping` until then (docs/CLOUD.md, Stopping).
        p.cloud.endAt = Math.min(now, trainEnd(p));
        const end = cloudFinish({ ...p, cloud: p.cloud });
        if (status === 'queued') {
          p.startedAt = null;
          p.neverStarted = true;
        }
        p.terminal = { status: 'stopped', atEpoch: done, at: end, stoppingUntil: end };
        return this.#toRun(p, now);
      }
      // A run that waits leaves the queue and stops at once; its trainer was never built.
      const waiting = status === 'queued';
      this.#queue = this.#queue.filter((x) => x !== p.id);
      const startedAt = loopStart(p, now);
      p.startedAt = startedAt;
      p.loop = false;
      p.neverStarted = waiting;
      p.terminal = waiting ? { status: 'stopped', atEpoch: 0, at: now } : { status: 'stopped', atEpoch: done, at: now + 1200, stoppingUntil: now + 1200 };
      return this.#toRun(p, now);
    });
  }

  deleteRun(id: string): Promise<void> {
    return this.#call(() => {
      const p = this.#profile(id);
      const status = progressAt(p, Date.now()).status;
      if (status === 'queued' || status === 'running' || status === 'stopping')
        throw new ApiError(409, 'run_active', `Run ${p.name} is ${status}.`, 'Stop it first.');
      this.#runs = this.#runs.filter((r) => r.id !== id);
    });
  }

  /** POST /api/runs/{id}/queue, with the real server's errors and log lines (docs/API.md). */
  moveRun(id: string, req: QueueRequest): Promise<Run> {
    return this.#call(() => {
      const p = this.#profile(id);
      const action = 'action' in req;
      const position = 'position' in req;
      if (action === position || (position && !(Number.isInteger(req.position) && req.position >= 1)))
        throw new ApiError(400, 'invalid_body', 'give exactly one of `action` and `position` (a place from 1)');
      const now = Date.now();
      const from = this.#queue.indexOf(id);
      const order = applyQueueMove(this.#queue, id, req);
      if (!order) {
        const status = progressAt(p, now).status;
        throw new ApiError(
          409,
          'run_not_queued',
          `run '${id}' ${status === 'queued' ? 'is starting' : `is ${status}`}, not waiting in the queue`,
          'only runs waiting in the queue (with a queue_position) can be moved',
        );
      }
      this.#queue = order;
      const start = action && req.action === 'start';
      // `start` on a free slot: the worker takes it now.
      if (start) this.#schedule(now);
      const to = this.#queue.indexOf(id);
      const place = `${to + 1} of ${this.#queue.length}`;
      this.#note(
        id,
        start
          ? to < 0
            ? 'start requested: starting now'
            : `start requested: moved to the front (${place}); it starts when the active run ends`
          : from === to
            ? `queue position unchanged (${place})`
            : `moved in the queue to position ${place}`,
      );
      return this.#toRun(p, now);
    }, 160);
  }

  runMetrics(id: string, _signal?: AbortSignal, every = 4): Promise<RunMetrics> {
    return this.#call(() => {
      const p = this.#profile(id);
      const now = Date.now();
      // A looping run's history restarts with each loop.
      if (p.loop && p.startedAt !== null) {
        const shifted = { ...p, startedAt: loopStart(p, now), loop: false };
        return history(shifted, now, every);
      }
      return history(p, now, every);
    }, 220);
  }

  runSamples(id: string, epoch?: number): Promise<RunSamples> {
    return this.#call(() => {
      const p = this.#profile(id);
      const done = Math.floor(progressAt(p, Date.now()).done);
      // `epoch` is 1-based like the API; `e` indexes the run model (0-based).
      const e = Math.max(0, Math.min(done - 1, (epoch ?? done) - 1));
      const ds = this.#spec(p.config.dataset_id);
      const quality = done > 0 ? epochAt(p, e).val.map50_95 : 0;
      const samples: Sample[] = [];
      // A fixed set of up to 16 validation images, spread evenly over the split; a dataset with no validation images
      // validates on its training split (docs/API.md, Split fallback), so its samples come from there.
      const fromTrain = ds.splits.val === 0;
      const pool = fromTrain ? ds.splits.train : ds.splits.val;
      const first = fromTrain ? 0 : ds.splits.train;
      const n = Math.min(16, pool);
      for (let i = 0; i < n; i++) {
        const index = first + Math.floor((i * pool) / n);
        const img = image(ds, index);
        samples.push({
          image_id: img.id,
          dataset_id: ds.id,
          width: img.width,
          height: img.height,
          truth: img.boxes,
          pred: predictBoxes(img.boxes, img.width, img.height, quality, hash(p.id) + i * 977 + e * 13, ds.classes.length)
            .filter((b) => b.score >= 0.25)
            .map((b) => ({ ...b, name: ds.classes[b.class] ?? `class ${b.class}` })),
        });
      }
      return { epoch: e + 1, samples };
    }, 260);
  }

  runEvents(id: string, h: RunEventHandlers): () => void {
    if (this.scenario === 'loading' || this.scenario === 'error') {
      if (this.scenario === 'error') setTimeout(() => h.connection?.('reconnecting'), 500);
      return () => {};
    }
    let p: RunProfile;
    try {
      p = this.#profile(id);
    } catch {
      return () => {};
    }
    const ds = specs.find((s) => s.id === p.config.dataset_id);
    const subs = this.#subs.get(id) ?? new Set<RunEventHandlers>();
    this.#subs.set(id, subs);
    subs.add(h);
    // The run's place in the queue and how many wait, as last reported (the snapshot carries the first one).
    const place = () => {
      const i = this.#queue.indexOf(id);
      return i < 0 ? null : { position: i + 1, waiting: this.#queue.length };
    };
    let lastPlace = JSON.stringify(place());
    let last = progressAt(p, Date.now());
    let lastStart = loopStart(p, Date.now());
    let lastStep = Math.floor(last.done * p.stepsPerEpoch);
    let lastEpoch = Math.floor(last.done);
    // A cloud run's state, sent when it changes: a phase or what it names at once, progress at most once a second and
    // the cost about once a minute (docs/API.md, SSE `cloud`).
    const cloudKey = (c: RunCloud) => JSON.stringify([c.phase, c.instance_id, c.deadline, c.synced]);
    let lastCloud = p.cloud ? cloudOf({ ...p, cloud: p.cloud }, Date.now()) : null;
    let cloudSent = Date.now();
    const t0 = setTimeout(() => {
      h.connection?.('open');
      h.status?.({ status: last.status, error: p.terminal?.error ?? null });
    }, 30);
    let dropped = false;
    const timer = setInterval(() => {
      const now = Date.now();
      // `stale`: the server stops, its stream with it (EventSource then reports each failed retry).
      if (this.scenario === 'stale' && now - this.#t0 > STALE_AFTER) {
        if (!dropped) h.connection?.('reconnecting');
        dropped = true;
        return;
      }
      this.#schedule(now);
      // A `queue` event whenever the place or the count changed (docs/API.md: state, not changes); before `running`.
      const q = place();
      if (JSON.stringify(q) !== lastPlace) {
        lastPlace = JSON.stringify(q);
        if (q) h.queue?.(q);
      }
      const cur = progressAt(p, now);
      const start = loopStart(p, now);
      if (p.loop && p.startedAt !== null && start !== lastStart) {
        // The demo loop restarted: behave like a fresh run.
        lastStart = start;
        lastStep = -1;
        lastEpoch = 0;
        h.status?.({ status: 'running', error: null });
        for (const l of startLogs(p, start, ds?.name ?? p.config.dataset_id, ds?.splits.train ?? 0)) h.log?.(l);
        h.reconnect?.();
      }
      if (cur.status === 'running') {
        const step = Math.floor(cur.done * p.stepsPerEpoch);
        if (step !== lastStep) {
          lastStep = step;
          h.step?.(stepAt(p, step));
        }
      }
      // Flush finished epochs before any terminal status, so the last one is never skipped.
      if (cur.status !== 'queued') {
        const ep = Math.floor(cur.done);
        while (lastEpoch < ep) {
          const e = epochAt(p, lastEpoch);
          h.epoch?.(e);
          for (const l of epochLog(p, e, start)) h.log?.(l);
          lastEpoch++;
        }
      }
      if (p.cloud && lastCloud) {
        const c = cloudOf({ ...p, cloud: p.cloud }, now);
        const moved = c.progress?.done_bytes !== lastCloud.progress?.done_bytes && now - cloudSent >= 1000;
        const billed = c.cost_usd !== lastCloud.cost_usd && now - cloudSent >= 60_000;
        if (cloudKey(c) !== cloudKey(lastCloud) || moved || billed) {
          lastCloud = c;
          cloudSent = now;
          h.cloud?.(c);
        }
      }
      if (cur.status !== last.status) h.status?.({ status: cur.status, error: p.terminal?.error ?? null, ...(p.terminal?.errorCode ? { error_code: p.terminal.errorCode } : {}) });
      if (cur.status === 'stopped' && last.status !== 'stopped')
        h.log?.({ t: new Date(now).toISOString(), level: 'warn', message: `stopped by user at epoch ${Math.floor(cur.done) + 1}; last checkpoint kept` });
      last = cur;
    }, 100);
    return () => {
      clearTimeout(t0);
      clearInterval(timer);
      subs.delete(h);
    };
  }

  runLogs(id: string, limit = 500): Promise<RunLogs> {
    return this.#call(() => {
      this.#profile(id);
      return { lines: this.#logHistory(id).slice(-Math.max(1, Math.min(5000, Math.floor(limit)))) };
    }, 120);
  }

  /** Log lines so far for a run, reconstructed from its profile and the clock. */
  #logHistory(id: string): LogLine[] {
    const p = this.#runs.find((r) => r.id === id);
    const notes = this.#notes.get(id) ?? [];
    if (!p) return [...notes];
    const now = Date.now();
    const ds = specs.find((s) => s.id === p.config.dataset_id);
    const cloud = p.cloud ? cloudLog(p.cloud, trainEnd(p), now, ds?.name ?? p.config.dataset_id) : [];
    if (p.startedAt === null || now < p.startedAt) return [...cloud, ...notes].sort((a, b) => a.t.localeCompare(b.t));
    const start = loopStart(p, now);
    const lines = startLogs(p, start, ds?.name ?? '', ds?.splits.train ?? 0);
    const done = Math.floor(progressAt(p, now).done);
    for (let e = 0; e < done; e++) lines.push(...epochLog(p, epochAt(p, e), start));
    if (p.terminal?.status === 'failed') lines.push({ t: new Date(start + done * p.epochSeconds * 1000).toISOString(), level: 'error', message: p.terminal.error ?? 'failed' });
    return [...cloud, ...lines, ...notes].sort((a, b) => a.t.localeCompare(b.t));
  }

  // ---------- cloud ----------

  /** A server without the `cloud` feature (`cloud-off`, older ones): `/api/cloud` is an unknown route. */
  #cloudRoute(path: string) {
    if (!this.#cloudOn) throw this.#noRoute(path);
  }

  #connection(id: string): CloudConnection {
    const c = this.#connections.find((x) => x.id === id);
    if (!c) throw new ApiError(404, 'connection_not_found', `no cloud connection '${id}'`, 'GET /api/cloud/connections lists them.');
    return c;
  }

  #offersOf(c: CloudConnection): CloudOffer[] {
    return c.provider === 'runpod' ? RUNPOD_OFFERS : [REMOTE_OFFER];
  }

  /** Every instance the mock's cloud runs launched, as at `now`: the live ones first, then the newest. */
  #instances(now: number): { instance: CloudInstance; run: RunProfile }[] {
    const out: { instance: CloudInstance; run: RunProfile }[] = [];
    for (const p of this.#runs) {
      const instance = p.cloud ? instanceAt(p.cloud, p.id, trainEnd(p), now) : null;
      if (instance) out.push({ instance, run: p });
    }
    const alive = (i: CloudInstance) => i.status !== 'terminated' && i.status !== 'lost';
    return out.sort((a, b) => Number(alive(b.instance)) - Number(alive(a.instance)) || b.instance.created.localeCompare(a.instance.created));
  }

  cloudProviders(): Promise<CloudProvider[]> {
    return this.#call(() => {
      this.#cloudRoute('GET /api/cloud/providers');
      return structuredClone(PROVIDERS);
    }, 60);
  }

  cloudConnections(): Promise<CloudConnection[]> {
    return this.#call(() => {
      this.#cloudRoute('GET /api/cloud/connections');
      return structuredClone(this.#connections);
    }, 80);
  }

  /**
   * `POST /api/cloud/connections`: any values but a secret containing `bad` (`422 cloud_auth_failed`), as on
   * folo-server's mock. The secrets are kept as their hint only; the provider takes a moment to answer.
   */
  connectCloud(body: ConnectCloud): Promise<CloudConnection> {
    return this.#call(() => {
      this.#cloudRoute('POST /api/cloud/connections');
      const provider = PROVIDERS.find((x) => x.id === body.provider);
      if (!provider) throw new ApiError(404, 'provider_not_found', `no cloud provider '${body.provider}'`, 'GET /api/cloud/providers lists them.');
      const value = (key: string) => (body.values[key] ?? '').trim();
      for (const f of provider.fields) if (f.required && !value(f.key)) throw new ApiError(400, 'invalid_body', `\`${f.key}\` is required to connect ${provider.name}`);
      const url = provider.id === 'remote' ? value('url') : '';
      if (provider.id === 'remote' && !/^https?:\/\/[^\s/]+/i.test(url)) throw new ApiError(400, 'invalid_body', '`url` must be an http:// or https:// URL');
      const host = url.replace(/^https?:\/\//i, '').replace(/[/?#].*$/, '');
      if (provider.fields.some((f) => f.secret && value(f.key).includes('bad')))
        throw new ApiError(
          422,
          'cloud_auth_failed',
          provider.id === 'remote' ? `the server at ${host} rejected the token` : `${provider.name} rejected this API key`,
          provider.id === 'remote' ? 'Use the token the server was started with (`folo serve --token`).' : 'Create a key with read and write access to pods: RunPod console → Settings → API Keys.',
        );
      const taken = new Set(this.#connections.map((c) => c.name));
      let name = body.name?.trim() || provider.name;
      for (let k = 2; !body.name?.trim() && taken.has(name); k++) name = `${provider.name} ${k}`;
      const local = /^(localhost|127\.|\[::1\]|10\.|192\.168\.)/i.test(host);
      const now = Date.now();
      const conn: CloudConnection = {
        id: `cc_${(hash(`${name}:${now}:${this.#nextId++}`) & 0xffff).toString(16).padStart(4, '0')}`,
        provider: provider.id,
        name,
        created: new Date(now).toISOString(),
        status: 'ok',
        account: provider.id === 'remote' ? host : 'you@example.com',
        balance_usd: provider.id === 'remote' ? null : 42.17,
        error: null,
        note:
          provider.id === 'remote' && /^http:/i.test(url) && !local
            ? 'Reached over http://: the token and the data travel unencrypted. Use https:// across networks you do not control.'
            : null,
        checked: new Date(now).toISOString(),
        values: Object.fromEntries(
          provider.fields.filter((f) => value(f.key)).map((f) => [f.key, f.secret ? secretHint(value(f.key)) : value(f.key)]),
        ),
      };
      this.#connections.push(conn);
      return structuredClone(conn);
    }, 1200);
  }

  /** `POST …/check`: the stored credentials again; the revoked key of `cloud-trouble` still fails (as `200`). */
  checkCloudConnection(id: string): Promise<CloudConnection> {
    return this.#call(() => {
      this.#cloudRoute(`POST /api/cloud/connections/${id}/check`);
      const c = this.#connection(id);
      c.checked = new Date().toISOString();
      return structuredClone(c);
    }, 900);
  }

  disconnectCloud(id: string): Promise<void> {
    return this.#call(() => {
      this.#cloudRoute(`DELETE /api/cloud/connections/${id}`);
      const c = this.#connection(id);
      const now = Date.now();
      const live = this.#instances(now).find(({ instance }) => instance.connection_id === id && instance.status !== 'terminated' && instance.status !== 'lost');
      const active = this.#runs.find((p) => p.cloud?.connectionId === id && ['queued', 'running', 'stopping'].includes(progressAt(p, now).status));
      if (live || active)
        throw new ApiError(
          409,
          'connection_in_use',
          live ? `${c.name} has a live instance, ${live.instance.name}` : `${c.name} has an active run, ${active!.name}`,
          'Terminate its instances, or wait for its runs to end, then disconnect it.',
        );
      this.#connections = this.#connections.filter((x) => x.id !== id);
    }, 300);
  }

  cloudOffers(id: string): Promise<CloudOfferList> {
    return this.#call(() => {
      this.#cloudRoute(`GET /api/cloud/connections/${id}/offers`);
      const c = this.#connection(id);
      if (c.status === 'error') throw new ApiError(422, 'cloud_auth_failed', c.error ?? `${c.name} rejected the credentials`, 'Check the connection in Settings › Cloud.');
      return { offers: structuredClone(this.#offersOf(c)), fetched: new Date().toISOString() };
    }, 400);
  }

  cloudInstances(): Promise<CloudInstance[]> {
    return this.#call(() => {
      this.#cloudRoute('GET /api/cloud/instances');
      return this.#instances(Date.now()).map((x) => x.instance);
    }, 100);
  }

  /**
   * `DELETE /api/cloud/instances/{id}`: terminating, confirmed a moment later; a run still using it fails with
   * `cloud_instance_lost` (its copied checkpoints stay). One already gone answers its record, and one whose termination
   * keeps failing (`cloud-trouble`) stays `terminating` with its error.
   */
  terminateCloudInstance(id: string): Promise<CloudInstance> {
    return this.#call(() => {
      this.#cloudRoute(`DELETE /api/cloud/instances/${id}`);
      const now = Date.now();
      const hit = this.#instances(now).find((x) => x.instance.id === id);
      if (!hit) throw new ApiError(404, 'instance_not_found', `no cloud instance '${id}'`, 'GET /api/cloud/instances lists them.');
      const { instance, run: p } = hit;
      if (instance.status === 'terminated' || instance.status === 'lost' || instance.error || !p.cloud) return instance;
      const { done, status } = progressAt(p, now);
      if (['queued', 'running', 'stopping'].includes(status)) {
        p.terminal = {
          status: 'failed',
          atEpoch: done,
          error: `the instance ${instance.name} was terminated before the run ended; the checkpoints copied so far are kept`,
          errorCode: 'cloud_instance_lost',
          at: now,
        };
        if (status === 'queued') {
          p.startedAt = null;
          p.neverStarted = true;
        }
      }
      p.cloud.endAt = Math.min(now, p.cloud.endAt ?? now);
      p.cloud.lost = true;
      return instanceAt(p.cloud, p.id, trainEnd(p), now) ?? instance;
    }, 500);
  }
}
