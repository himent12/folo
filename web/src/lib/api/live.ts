// A run kept live (LiveRun, below). Its steps, epochs and log lines are also exposed as
// append logs (`stepsLog`, `epochsLog`, `logsLog`, ./appendLog) beside the arrays (`steps`, `epochs`, `logs`, the
// logs' own lists): a chart can process only the steps added since it last looked, after `version` bumped.
//
// A run kept live: follows its SSE stream and loads its snapshot and metric history. Steps arrive up to ~10/s;
// they are appended to plain arrays and published once per animation frame through `version`, so charts redraw at
// most once a frame no matter how fast events come.
//
// Ordering: the stream replays nothing, so we subscribe *before* fetching history and merge the two by step /
// epoch number. A history response never removes newer events, and a snapshot never overrides a status or a queue
// place that an event reported after the request was sent.
//
// Resume: a stopped / failed run that becomes active again continues from its last finished epoch and repeats the
// steps after it (on the server, a repeated step replaces everything from it on; folo-server read_metrics). The view
// rewinds to the last finished epoch and enters a replay: steps and epochs streamed since the resume certainly belong
// to the new attempt, so they are kept and re-applied over every history response (which may still hold the
// interrupted attempt until replacements are written), and history is re-requested until the replay is over.
// Outside a replay the stream is monotonic (a late or duplicate delivery never removes anything). History and
// snapshots requested before a rewind are dropped (generation counter). A finished run's history is final: cached
// steps beyond it (an attempt this client saw but the server replaced) are dropped.

import { effect, ReactiveSet, signal, untrack } from 'fw';
import { AppendLog } from './appendLog';
import { ApiError } from './client';
import { hub } from './index';
import type { Epoch, LogLine, Run, RunCloud, RunLogs, Step } from './types';

/** Log lines a live view keeps (the latest). */
export const MAX_LOGS = 2000;
const ACTIVE = ['queued', 'running', 'stopping'];
/**
 * How long the stream may be down before the view stops calling itself live. EventSource retries a dropped connection
 * on its own within a few seconds; a blip that short is not worth a "Last known" state.
 */
export const STREAM_GRACE_MS = 5000;

/** Union of two lists sorted by `key`, preferring `b` (newer) on equal keys. Exported for tests. */
export function mergeBy<T>(a: T[], b: T[], key: (x: T) => number): T[] {
  if (!b.length) return a;
  if (!a.length) return b;
  const out: T[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length || j < b.length) {
    const x = a[i];
    const y = b[j];
    if (y === undefined || (x !== undefined && key(x) < key(y))) {
      out.push(x as T);
      i++;
    } else if (x === undefined || key(y) < key(x)) {
      out.push(y);
      j++;
    } else {
      out.push(y);
      i++;
      j++;
    }
  }
  return out;
}

/**
 * Steps per epoch is fixed for a run once the trainer has counted it (0 before): a snapshot taken earlier, even one
 * applied later, never takes a learned figure back.
 */
/** The run's length from an SSE status / step event: taken when known (steps per epoch is 0 until the trainer is built). */
function withLength(run: Run, stepsPerEpoch: number | undefined, epochs: number | undefined): Run {
  const spe = stepsPerEpoch !== undefined && stepsPerEpoch > 0 && stepsPerEpoch !== run.steps_per_epoch ? stepsPerEpoch : null;
  const ep = epochs !== undefined && epochs > 0 && epochs !== run.epochs ? epochs : null;
  return spe === null && ep === null ? run : { ...run, ...(spe !== null ? { steps_per_epoch: spe } : {}), ...(ep !== null ? { epochs: ep } : {}) };
}

export function keepStepsPerEpoch(next: Run, cur: Run | null | undefined): Run {
  return !(next.steps_per_epoch > 0) && cur?.id === next.id && cur.steps_per_epoch > 0 ? { ...next, steps_per_epoch: cur.steps_per_epoch } : next;
}

/** The server's history rule for one new step: it replaces every step from it on, and the epochs from its epoch on. */
function replaceFrom(steps: Step[], epochs: Epoch[], s: Step): [Step[], Epoch[]] {
  return [[...steps.filter((x) => x.step < s.step), s], epochs.filter((e) => e.epoch < s.epoch)];
}

/** Two log histories as one, oldest first, each line once (by time and text), the latest 2000. */
export function mergeLogs(a: LogLine[], b: LogLine[]): LogLine[] {
  const seen = new Set(a.map((l) => `${l.t}|${l.message}`));
  const out = [...a, ...b.filter((l) => !seen.has(`${l.t}|${l.message}`))];
  out.sort((x, y) => x.t.localeCompare(y.t));
  return out.slice(-MAX_LOGS);
}

/** Every mounted view's stream (Overview's live card, the run page), so the shell can read the one already open. */
const streams = new ReactiveSet<LiveRun>();

/**
 * The run as a mounted view's stream currently has it, while that stream is up (loaded, not `lost`): the sidebar's live
 * card reads it in place of the polled list, so it never says "Last known" beside a hero that is still live, nor
 * trails it by a poll (DESIGN_DIRECTION §2.8: two clocks, one reading per source). Undefined when no view streams `id`.
 */
export function streamedRun(id: string | undefined): Run | undefined {
  if (!id) return undefined;
  for (const s of streams) if (s.run?.id === id && s.lost === null && !s.loading) return s.run;
  return undefined;
}

export class LiveRun {
  #run = signal<Run | undefined>(undefined);
  #error = signal<unknown>(undefined);
  #loading = signal(true);
  #connection = signal<'idle' | 'open' | 'reconnecting'>('idle');
  #lost = signal<number | null>(null);
  #waiting = signal<number | null>(null);
  #version = signal(0);

  get run(): Run | undefined {
    return this.#run.get();
  }
  set run(v: Run | undefined) {
    this.#run.set(v);
  }
  get error(): unknown {
    return this.#error.get();
  }
  set error(v: unknown) {
    this.#error.set(v);
  }
  get loading(): boolean {
    return this.#loading.get();
  }
  set loading(v: boolean) {
    this.#loading.set(v);
  }
  get connection(): 'idle' | 'open' | 'reconnecting' {
    return this.#connection.get();
  }
  set connection(v: 'idle' | 'open' | 'reconnecting') {
    this.#connection.set(v);
  }
  /**
   * When the stream was last heard from (ms since the epoch), set once it has been down for `STREAM_GRACE_MS`; null
   * while it is open (or has not opened yet). What the view shows is then the last known state, as of this time: the
   * page says so and stops its live marks (DESIGN_DIRECTION §2.8, freshness).
   */
  get lost(): number | null {
    return this.#lost.get();
  }
  set lost(v: number | null) {
    this.#lost.set(v);
  }
  /** How many runs wait in the training queue, this one included, from its latest `queue` event (null: none yet). */
  get waiting(): number | null {
    return this.#waiting.get();
  }
  set waiting(v: number | null) {
    this.#waiting.set(v);
  }
  /** Bumps when steps/epochs/logs change (at most once per frame). */
  get version(): number {
    return this.#version.get();
  }
  set version(v: number) {
    this.#version.set(v);
  }

  /** The steps, oldest first, as an append log: what was added since a reader's cursor (read after `version`). */
  readonly stepsLog = new AppendLog<Step>();
  /** The finished epochs, oldest first, as an append log. */
  readonly epochsLog = new AppendLog<Epoch>();
  /** The log lines (the latest 2000), oldest first, as an append log (the cap trims its front: `dropped`). */
  readonly logsLog = new AppendLog<LogLine>();

  /** The steps, oldest first (the append log's list: the same array while it only grows). */
  get steps(): Step[] {
    return this.stepsLog.items;
  }
  set steps(v: Step[]) {
    this.stepsLog.replace(v);
  }
  get epochs(): Epoch[] {
    return this.epochsLog.items;
  }
  set epochs(v: Epoch[]) {
    this.epochsLog.replace(v);
  }
  get logs(): LogLine[] {
    return this.logsLog.items;
  }
  /**
   * Lines of the run's log older than the ones held (`logs`, the latest `MAX_LOGS`): what the server counted past them
   * when the history loaded, and the lines the cap dropped since. The log panel says how many it does not show.
   */
  logsEarlier = 0;
  /** Log lines the stream has delivered, ever: a history load counts the ones that arrived while it was in flight. */
  #streamedLogs = 0;
  /** The latest of them in the order they arrived (the held log is sorted by time and merged with history). */
  #streamedTail: LogLine[] = [];
  set logs(v: LogLine[]) {
    this.logsLog.replace(v);
  }
  #raf = 0;
  #retry = signal(0);
  /** The last time anything arrived from the stream or a history load (plain: steps arrive up to ~10/s). */
  #heard = 0;
  /** Increments with every status event, so a snapshot can tell whether one arrived while it was in flight. */
  #statusSeq = 0;
  #lastStatus: Pick<Run, 'status' | 'error'> | null = null;
  /** The same for `queue` events: the latest place outlives snapshots requested before it (and a run not loaded yet). */
  #queueSeq = 0;
  #lastQueue: number | null = null;
  /** The same for `cloud` events: a cloud run's latest state outlives snapshots requested before it. */
  #cloudSeq = 0;
  #lastCloud: RunCloud | null = null;
  /** Bumps on every rewind (resume): history requested before it describes an earlier attempt and is dropped. */
  #gen = 0;
  /** While a resumed run repeats steps it had already run: the old last step, and what the new attempt streamed. */
  #replay: { until: number; steps: Step[]; epochs: Epoch[] } | null = null;

  constructor(id: () => string | null) {
    effect(() => {
      untrack(() => streams.add(this));
      return () => void streams.delete(this);
    });
    effect(() => {
      const runId = id();
      if (hub.mode === 'connecting' || !runId) return;
      void hub.epoch;
      void this.#retry.get();
      const api = hub.api;
      let closed = false;
      this.#lastStatus = null;
      this.#lastQueue = null;
      this.#lastCloud = null;
      this.#replay = null;
      let lostTimer: ReturnType<typeof setTimeout> | undefined;
      this.#heard = Date.now();
      untrack(() => {
        this.lost = null;
        this.waiting = null;
        this.loading = true;
        this.error = undefined;
        this.run = undefined;
        this.steps = [];
        this.epochs = [];
        this.logs = [];
        this.logsEarlier = 0;
        this.#streamedTail = [];
        this.connection = 'idle';
        this.#bump();
      });

      // Snapshots can resolve out of order (history loads, terminal refreshes): a snapshot applies only if no newer
      // request has been applied by the time it is used.
      let snapIssued = 0;
      let snapApplied = 0;
      const snapshot = (): Promise<{ run: Run; claim: () => boolean }> => {
        const my = ++snapIssued;
        return api.run(runId).then((run) => ({
          run,
          claim: () => {
            if (my < snapApplied) return false;
            snapApplied = my;
            return true;
          },
        }));
      };

      // History responses can arrive out of order too: one older than the last applied is dropped.
      let loadIssued = 0;
      let loadApplied = 0;
      const load = async () => {
        const seq = this.#statusSeq;
        const queueSeq = this.#queueSeq;
        const cloudSeq = this.#cloudSeq;
        const gen = this.#gen;
        const my = ++loadIssued;
        // Older servers have no /logs; the live stream still fills the view.
        const asked = this.#streamedLogs;
        const logsLoad = api.runLogs(runId, MAX_LOGS).catch((): RunLogs => ({ lines: [] }));
        const held = (l: RunLogs) => {
          // The server's count is as of its answer: the lines streamed in since it was asked for (the last ones held,
          // in the order they arrived, whatever their times: a cloud worker's lines keep theirs) and not in its answer
          // are on top of it. Without a count (an older server, a failed request) the one known stays.
          const tail = this.#streamedTail;
          const since = Math.min(tail.length, this.#streamedLogs - asked);
          const seen = new Set(l.lines.map((x) => `${x.t}|${x.message}`));
          const newer = tail.slice(tail.length - since).filter((x) => !seen.has(`${x.t}|${x.message}`)).length;
          this.logs = mergeLogs(l.lines, this.logs);
          if (l.total !== undefined) this.logsEarlier = Math.max(0, l.total + newer - this.logs.length);
        };
        const metricsLoad = api.runMetrics(runId);
        // The log is a request of its own: it shows while the metrics failed (their panel says so, with Retry).
        metricsLoad.catch(() =>
          logsLoad.then((l) => {
            if (closed || my < loadApplied || gen !== this.#gen) return;
            held(l);
            this.#bump();
          }),
        );
        const [snap, metrics, logs] = await Promise.all([snapshot(), metricsLoad, logsLoad]);
        if (closed) return;
        if (my < loadApplied) return;
        // Requested before a resume rewound the view (or before its replay ended): it may hold the interrupted
        // attempt. Ask again.
        if (gen !== this.#gen) return sync();
        loadApplied = my;
        // Final only if the server said so in this same response (its snapshot), not because the run ended since.
        const final = !ACTIVE.includes(snap.run.status);
        this.#absorb(snap.claim() ? snap.run : (this.run ?? snap.run), metrics.steps, metrics.epochs, seq, queueSeq, cloudSeq, final);
        // Still replaying a resumed run: ask again shortly, until its new steps have replaced the old ones.
        if (this.#replay && this.run && ACTIVE.includes(this.run.status)) {
          clearTimeout(replayTimer);
          replayTimer = setTimeout(() => sync(), 3000);
        }
        held(logs);
        this.loading = false;
        // A history that failed before (the panels said so) is in now.
        this.error = undefined;
        // The history just applied is current: no longer "last known" (DESIGN_DIRECTION §2.8).
        this.#heard = Date.now();
        // Still down (a retry of the load got through): the view is last known as of now.
        if (this.lost !== null) this.lost = this.connection === 'reconnecting' ? this.#heard : null;
        this.#bump();
      };

      // Re-synchronising history (on open / reconnect) must not silently give up: a gap would stay forever.
      let syncTimer: ReturnType<typeof setTimeout> | undefined;
      let replayTimer: ReturnType<typeof setTimeout> | undefined;
      let speAsked = false;
      let resolvedAsked = false;
      // The run was deleted (in another tab, the Runs list, the CLI): the page says so, as a reload would ("Run not
      // found"), instead of "Reconnecting" over the last known run forever while the stream retries a 404 (stress
      // s3-live-4; LiveBench does the same for a benchmark). Nothing retries after; the page's Retry starts over.
      let unsub = (): void => {};
      const gone = (e: unknown): boolean => {
        if (closed || !(e instanceof ApiError && e.status === 404)) return false;
        closed = true;
        clearTimeout(lostTimer);
        clearTimeout(syncTimer);
        clearTimeout(replayTimer);
        unsub();
        this.run = undefined;
        this.steps = [];
        this.epochs = [];
        this.logs = [];
        this.logsEarlier = 0;
        this.#streamedTail = [];
        this.lost = null;
        this.waiting = null;
        this.connection = 'idle';
        this.loading = false;
        this.error = e;
        this.#bump();
        return true;
      };
      // A dropped stream may mean the run is gone (the stream answers 404 then, and EventSource gives no status): one
      // snapshot at a time says so; any other answer leaves the reconnect to the stream.
      let checking = false;
      const check = () => {
        if (checking || closed) return;
        checking = true;
        api.run(runId).then(
          () => void (checking = false),
          (e) => {
            checking = false;
            gone(e);
          },
        );
      };

      const sync = (attempt = 0) => {
        clearTimeout(syncTimer);
        load().catch((e) => {
          if (closed || gone(e)) return;
          syncTimer = setTimeout(() => sync(attempt + 1), Math.min(15_000, 1000 * 2 ** attempt));
        });
      };

      // Subscribe first: events that arrive while history loads are kept and merged, not lost.
      unsub = api.runEvents(runId, {
        connection: (s) => {
          if (closed) return;
          // EventSource construction does not mean the subscription is live: events emitted between our first
          // history load and the stream opening would be lost, so load (and merge) again once it opens.
          if (s === 'open' && this.connection === 'idle') sync();
          this.connection = s;
          if (s === 'reconnecting') check();
          if (s === 'open') {
            clearTimeout(lostTimer);
            lostTimer = undefined;
            // A stream that was lost is open again, but what the view holds is still from before the outage: it stays
            // "last known" until the reconnect's history load applies or the stream reports progress (a step, a
            // status), not merely because the connection came back.
            if (this.lost === null) this.#heard = Date.now();
          } else if (s === 'reconnecting' && this.lost === null && lostTimer === undefined) {
            // Timed from the first error: EventSource reports one per failed retry, which must not restart the grace.
            const heard = this.#heard;
            lostTimer = setTimeout(() => {
              lostTimer = undefined;
              if (!closed && this.connection === 'reconnecting') this.lost = heard;
            }, STREAM_GRACE_MS);
          }
        },
        status: ({ status, error, steps_per_epoch, epochs }) => {
          if (closed) return;
          this.#heard = Date.now();
          this.lost = null;
          this.#statusSeq++;
          this.#lastStatus = { status, error };
          // Finished → active again: the run was resumed (here or elsewhere).
          if (this.run && !ACTIVE.includes(this.run.status) && ACTIVE.includes(status)) {
            this.#rewind();
            sync();
          }
          if (this.run) this.run = withLength({ ...this.run, status, error }, steps_per_epoch, epochs);
          // Only a waiting run has a place in the queue.
          if (this.run && status !== 'queued' && this.run.queue_position != null) this.run = { ...this.run, queue_position: null };
          // The trainer resolved the config by the time it reports `running` (docs/API.md): ask once for `resolved`.
          if (status === 'running' && this.run && this.run.resolved == null && !resolvedAsked) {
            resolvedAsked = true;
            api.run(runId).then(
              (r) => {
                if (closed || !this.run || !r.resolved) return void (resolvedAsked = false);
                this.run = { ...this.run, resolved: r.resolved };
              },
              () => (resolvedAsked = false),
            );
          }
          if (!ACTIVE.includes(status)) {
            // The run ended: the server's final history settles it. Requests issued before may hold a partial
            // history (or, during a replay, the interrupted attempt), even when a snapshot already said it ended
            // (/run and /metrics are read independently): invalidate them and fetch the final one, every time.
            this.#replay = null;
            this.#gen++;
            sync();
            // Terminal: refresh the snapshot for finished time, best and model id (even if history is still loading).
            const seq = this.#statusSeq;
            const gen = this.#gen;
            const cloudSeq = this.#cloudSeq;
            snapshot().then((snap) => {
              if (closed || seq !== this.#statusSeq || gen !== this.#gen || !snap.claim()) return;
              this.run = this.#clouded(keepStepsPerEpoch(snap.run, this.run), cloudSeq);
            }, () => {});
          }
        },
        step: (s) => {
          if (closed) return;
          this.#heard = Date.now();
          this.lost = null;
          const last = this.steps[this.steps.length - 1];
          if (this.#replay) {
            // The new attempt's step: it replaces whatever is held from it on (the server's rule).
            this.#replay.steps.push(s);
            [this.steps, this.epochs] = replaceFrom(this.steps, this.epochs, s);
            if (s.step > this.#replay.until) {
              // Past the interrupted attempt: the replay is over; one last reconcile with the server (requests issued
              // during the replay are invalidated, they may still hold the interrupted attempt).
              this.#replay = null;
              this.#gen++;
              sync();
            }
          } else {
            // Ordinary stream: monotonic. A late or duplicate delivery never removes newer data.
            if (last && s.step <= last.step) return;
            this.steps.push(s);
          }
          if (this.run) this.run = withLength({ ...this.run, step: s.step, epoch: s.epoch, eta_s: s.eta_s }, s.steps_per_epoch, s.epochs);
          // A real run reports 0 steps per epoch until its trainer has loaded the data, i.e. in the snapshot taken
          // at start. Current servers put the run's length on every step (above); for older ones, whose steps lack
          // it, ask once more (only that field is taken; progress needs it).
          if (this.run?.steps_per_epoch === 0 && s.steps_per_epoch === undefined && !speAsked) {
            speAsked = true;
            api.run(runId).then(
              (r) => {
                if (closed || !this.run || r.steps_per_epoch <= 0) return void (speAsked = false);
                this.run = { ...this.run, steps_per_epoch: r.steps_per_epoch };
                this.#bump();
              },
              () => (speAsked = false),
            );
          }
          this.#bump();
        },
        epoch: (e) => {
          if (closed) return;
          this.#heard = Date.now();
          const last = this.epochs[this.epochs.length - 1];
          if (this.#replay) {
            this.#replay.epochs.push(e);
            this.epochs = [...this.epochs.filter((x) => x.epoch < e.epoch), e];
          } else {
            if (last && e.epoch <= last.epoch) return;
            this.epochs.push(e);
          }
          const best = this.run?.best;
          if (this.run && (!best || e.val.map50_95 > best.map50_95))
            this.run = { ...this.run, best: { epoch: e.epoch, map50_95: e.val.map50_95 } };
          this.#bump();
        },
        queue: ({ position, waiting }) => {
          if (closed) return;
          this.#heard = Date.now();
          // State, not a change: the latest event is the run's place (docs/API.md, SSE `queue`). It is kept for the
          // snapshots still in flight (#absorb), which may hold an older place or arrive before any run is loaded.
          this.#queueSeq++;
          this.#lastQueue = position;
          this.waiting = waiting;
          if (this.run && this.run.status === 'queued') this.run = { ...this.run, queue_position: position };
        },
        cloud: (c) => {
          if (closed) return;
          this.#heard = Date.now();
          // State, not a change (docs/API.md, SSE `cloud`): the latest event is the run's `cloud`. It is kept for the
          // snapshots still in flight (#absorb), which may hold an older phase or arrive before any run is loaded.
          this.#cloudSeq++;
          this.#lastCloud = c;
          if (this.run) this.run = { ...this.run, cloud: c };
        },
        log: (l) => {
          if (closed) return;
          this.#heard = Date.now();
          this.#streamedLogs++;
          this.#streamedTail.push(l);
          if (this.#streamedTail.length > MAX_LOGS) this.#streamedTail.shift();
          this.logs.push(l);
          if (this.logs.length > MAX_LOGS) {
            this.logsEarlier += this.logs.length - MAX_LOGS;
            this.logsLog.trimFront(this.logs.length - MAX_LOGS);
          }
          this.#bump();
        },
        reconnect: () => {
          // The stream replays nothing: refetch history and merge it with what arrived meanwhile.
          sync();
        },
      });

      load().catch((e) => {
        if (closed) return;
        this.error = e;
        this.loading = false;
      });

      return () => {
        closed = true;
        clearTimeout(lostTimer);
        clearTimeout(syncTimer);
        clearTimeout(replayTimer);
        unsub();
        cancelAnimationFrame(this.#raf);
        this.#raf = 0;
      };
    });
  }

  /** Reloads from scratch (after a load error). */
  retry() {
    this.#retry.set((n) => n + 1);
  }

  /**
   * Resumes the run through `call` (POST /api/runs with resume_run_id). The worker can start before the response
   * arrives: a status event received meanwhile is newer than the response and wins, and the history is rewound once
   * whichever of the two comes first.
   */
  async resume(call: () => Promise<Run>): Promise<Run> {
    const seq = this.#statusSeq;
    const queueSeq = this.#queueSeq;
    const cloudSeq = this.#cloudSeq;
    const r = await call();
    if (this.#statusSeq === seq && this.run && !ACTIVE.includes(this.run.status)) this.#rewind();
    // Progress comes from the rewind and the stream (the response may still carry the interrupted step count); the
    // status from the response unless an event already reported a newer one.
    const cur = this.run;
    const status = this.#statusSeq === seq ? { status: r.status, error: r.error } : (this.#lastStatus ?? { status: r.status, error: r.error });
    this.run = this.#clouded(this.#placed({ ...r, ...(cur ? { step: cur.step, epoch: cur.epoch, best: cur.best } : {}), ...status }, queueSeq), cloudSeq);
    return r;
  }

  /**
   * Drops the steps after the last finished epoch (1-based; a resumed run repeats them), moves the run there and
   * starts a replay.
   */
  #rewind() {
    this.#gen++;
    this.#replay = { until: this.steps[this.steps.length - 1]?.step ?? 0, steps: [], epochs: [] };
    const done = this.epochs[this.epochs.length - 1]?.epoch ?? 0;
    this.steps = this.steps.filter((s) => s.epoch <= done);
    const kept = this.steps[this.steps.length - 1]?.step ?? 0;
    if (this.run) this.run = { ...this.run, step: kept, epoch: done };
    this.#bump();
  }

  /** Merges a snapshot + history into what the stream has delivered so far, never moving backwards. */
  #absorb(run: Run, steps: Step[], epochs: Epoch[], statusSeqAtStart: number, queueSeqAtStart: number, cloudSeqAtStart: number, final: boolean) {
    // During a replay the new attempt's streamed steps / epochs are certain: re-apply them over the history.
    if (this.#replay) {
      for (const r of this.#replay.steps) [steps, epochs] = replaceFrom(steps, epochs, r);
      for (const r of this.#replay.epochs) epochs = [...epochs.filter((x) => x.epoch < r.epoch), r];
    }
    // The history is the server's word up to its last step / epoch. Beyond it, a live run keeps what the stream
    // delivered (newer than the file); a finished run's history is final, so anything beyond it is obsolete.
    const live = !final;
    const hs = steps[steps.length - 1]?.step ?? 0;
    const he = epochs[epochs.length - 1]?.epoch ?? 0;
    this.steps = [...steps, ...(live ? this.steps.filter((s) => s.step > hs) : [])];
    this.epochs = [...epochs, ...(live ? this.epochs.filter((e) => e.epoch > he) : [])];
    let next = run;
    const last = this.steps[this.steps.length - 1];
    if (last && last.step > next.step && ['running', 'stopping'].includes(next.status))
      next = { ...next, step: last.step, epoch: last.epoch, eta_s: last.eta_s };
    next = keepStepsPerEpoch(next, this.run);
    // During a replay the snapshot may still count the interrupted attempt: progress follows the steps held.
    if (this.#replay && live) next = { ...next, step: last?.step ?? 0, epoch: last?.epoch ?? this.epochs[this.epochs.length - 1]?.epoch ?? 0 };
    for (const e of this.epochs)
      if (!next.best || e.val.map50_95 > next.best.map50_95) next = { ...next, best: { epoch: e.epoch, map50_95: e.val.map50_95 } };
    // A status event that arrived after this request was sent is newer than the snapshot.
    if (this.#statusSeq !== statusSeqAtStart && this.#lastStatus) next = { ...next, ...this.#lastStatus };
    this.run = this.#clouded(this.#placed(next, queueSeqAtStart), cloudSeqAtStart);
  }

  /**
   * A snapshot's queue place, corrected: a `queue` event that arrived after the request was sent is newer, and only a
   * waiting run has a place.
   */
  #placed(run: Run, queueSeqAtStart: number): Run {
    if (run.status !== 'queued') return run.queue_position != null ? { ...run, queue_position: null } : run;
    if (this.#queueSeq !== queueSeqAtStart && this.#lastQueue !== null) return { ...run, queue_position: this.#lastQueue };
    return run;
  }

  /** A snapshot's cloud state, corrected: a `cloud` event that arrived after the request was sent is newer. */
  #clouded(run: Run, cloudSeqAtStart: number): Run {
    return this.#cloudSeq !== cloudSeqAtStart && this.#lastCloud ? { ...run, cloud: this.#lastCloud } : run;
  }

  #bump() {
    if (typeof requestAnimationFrame === 'undefined') {
      this.#version.set((v) => v + 1);
      return;
    }
    if (this.#raf) return;
    this.#raf = requestAnimationFrame(() => {
      this.#raf = 0;
      this.#version.set((v) => v + 1);
    });
  }
}
