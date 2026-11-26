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

import { batch, effect, ReactiveSet, signal, untrack } from 'fw';
import { AppendLog } from './appendLog';
import { ApiError } from './client';
import { hub } from './index';
import { heldResults, type HeldResults } from './progress';
import type { Epoch, LogLine, Run, RunCloud, RunLogs, RunMetrics, RunStatus, Step } from './types';

/** Log lines a live view keeps (the latest). */
export const MAX_LOGS = 2000;
/** Trimming chunk size for log buffers: avoids an O(MAX_LOGS) splice/shift on every incoming line. */
export const LOG_TRIM_BATCH = 200;
const ACTIVE = ['queued', 'running', 'stopping'];
/**
 * How long the stream may be down before the view stops calling itself live. The stream (./sse) reopens a dropped
 * connection on its own within a few seconds; a blip that short is not worth a "Last known" state.
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

/**
 * The current status of `id` from a live mounted stream while it is up (loaded, not `lost`). Reads only the stream's
 * id/status signals (never step-level progress fields), so callers that only need status consistency (Overview's header
 * and slot, the sidebar's Runs badge) do not recompute on 10 Hz step ticks.
 */
export function streamedStatus(id: string | undefined): RunStatus | undefined {
  if (!id) return undefined;
  for (const s of streams) if (s.runId === id && s.lost === null && !s.loading) return s.status;
  return undefined;
}

export class LiveRun {
  #run = signal<Run | undefined>(undefined);
  #runId = signal<string | undefined>(undefined);
  #status = signal<RunStatus | undefined>(undefined);
  #error = signal<unknown>(undefined);
  #loading = signal(true);
  #connection = signal<'idle' | 'open' | 'reconnecting'>('idle');
  #lost = signal<number | null>(null);
  #server = signal<Run | undefined>(undefined);
  #short = signal(0);
  #waiting = signal<number | null>(null);
  #version = signal(0);

  get run(): Run | undefined {
    return this.#run.get();
  }
  set run(v: Run | undefined) {
    this.#run.set(v);
    this.#runId.set(v?.id);
    this.#status.set(v?.status);
    this.#notifyEnded();
  }
  get runId(): string | undefined {
    return this.#runId.get();
  }
  get status(): RunStatus | undefined {
    return this.#status.get();
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
    if (!v) this.#notifyEnded();
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
    if (v === null) this.#notifyEnded();
  }
  /**
   * The run as the server's latest applied snapshot has it, uncorrected by events: once it says the run ended, how many
   * epochs the run validated (progress `epochsValidated`). A view holding fewer has the rest on their way (issue #327).
   */
  get server(): Run | undefined {
    return this.#server.get();
  }
  /** Complete history answers in a row that came back short of what the ended snapshot says the run validated. */
  get shortAnswers(): number {
    return this.#short.get();
  }
  /** What the view holds of an ended run's results (progress `heldResults`): pending, final, or a short history. */
  get results(): HeldResults | undefined {
    void this.#version.get();
    const s = this.#server.get() ?? this.run;
    return s && heldResults(s, this.epochs[this.epochs.length - 1]?.epoch ?? 0, this.#short.get());
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
  /** Coalesced step progress from SSE `step` events, applied to `#run` on the next `#bump()` animation frame. */
  #pendingStep: { step: number; epoch: number; eta_s: number | null; steps_per_epoch?: number; epochs?: number } | null = null;
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
  /** Bumps only on a rewind: the run's attempt (each resume starts one), for a stop's late answer. */
  #attempt = 0;
  /** Bumps whenever the bound run view starts or tears down, so late resume/stop answers never touch another run. */
  #viewSeq = 0;
  /** While a resumed run repeats steps it had already run: the old last step, and what the new attempt streamed. */
  #replay: { until: number; steps: Step[]; epochs: Epoch[] } | null = null;
  #onEnded: ((run: Run | null, terminal: { status: RunStatus; error: string | null }) => void) | undefined;
  #currentId: string | null = null;
  #endedId: string | null = null;

  #notifyEnded(terminal?: { status: RunStatus; error: string | null }) {
    if (!this.#onEnded) return;
    const r = this.#run.peek();
    const id = r?.id ?? this.#currentId;
    if (!id) return;
    if (!terminal) {
      if (r && ACTIVE.includes(r.status)) {
        if (this.#endedId === id) this.#endedId = null;
        return;
      }
      if (!r || this.#loading.peek() || this.#lost.peek() !== null) return;
      if (this.#endedId === id) return;
      this.#endedId = id;
      untrack(() => this.#onEnded?.(r, { status: r.status, error: r.error }));
      return;
    }
    if (ACTIVE.includes(terminal.status)) {
      if (this.#endedId === id) this.#endedId = null;
      return;
    }
    if (this.#endedId === id) return;
    this.#endedId = id;
    const endedRun = r ? { ...r, status: terminal.status, error: terminal.error, queue_position: null } : null;
    untrack(() => this.#onEnded?.(endedRun, terminal));
  }

  constructor(id: () => string | null, opts?: { logs?: boolean; onEnded?: (run: Run | null, terminal: { status: RunStatus; error: string | null }) => void }) {
    const withLogs = opts?.logs ?? true;
    this.#onEnded = opts?.onEnded;
    effect(() => {
      untrack(() => streams.add(this));
      return () => void streams.delete(this);
    });
    effect(() => {
      const runId = id();
      if (!runId) {
        this.#viewSeq++;
        this.#currentId = null;
        this.#endedId = null;
        this.#lastStatus = null;
        this.#lastQueue = null;
        this.#lastCloud = null;
        this.#replay = null;
        if (this.#raf) cancelAnimationFrame(this.#raf);
        this.#raf = 0;
        this.#pendingStep = null;
        untrack(() => {
          this.lost = null;
          this.waiting = null;
          this.#server.set(undefined);
          this.#short.set(0);
          this.loading = false;
          this.error = undefined;
          this.run = undefined;
          this.steps = [];
          this.epochs = [];
          this.logs = [];
          this.logsEarlier = 0;
          this.#streamedTail = [];
          this.connection = 'idle';
          this.#publish();
        });
        return;
      }
      if (hub.mode === 'connecting') return;
      void hub.epoch;
      void this.#retry.get();
      this.#viewSeq++;
      this.#currentId = runId;
      this.#endedId = null;
      const api = hub.api;
      let closed = false;
      // Every request this view makes ends with it (and with the run's deletion): nothing of it answers later.
      const view = new AbortController();
      this.#lastStatus = null;
      this.#lastQueue = null;
      this.#lastCloud = null;
      this.#replay = null;
      if (this.#raf) cancelAnimationFrame(this.#raf);
      this.#raf = 0;
      this.#pendingStep = null;
      let lostTimer: ReturnType<typeof setTimeout> | undefined;
      this.#heard = Date.now();
      untrack(() => {
        this.lost = null;
        this.waiting = null;
        this.#server.set(undefined);
        this.#short.set(0);
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
        return api.run(runId, view.signal).then((run) => ({
          run,
          claim: () => {
            if (my < snapApplied) return false;
            snapApplied = my;
            this.#server.set(run);
            return true;
          },
        }));
      };

      // History responses can arrive out of order too: one older than the last applied is dropped. A failure applies
      // only to the latest request (an older one's is moot: the latest decides), and resolves false otherwise. Resolves
      // true when the run ended and the server says it validated epochs the view does not hold yet: ask again.
      let loadIssued = 0;
      let loadApplied = 0;
      let logsApplied = 0;
      let sawActive = false;
      let postOpenSync = false;
      let postOpenSnap: Promise<{ run: Run; claim: () => boolean }> | null = null;
      const load = async (onSnap?: (snap: { run: Run; claim: () => boolean }) => void): Promise<boolean> => {
        const seq = this.#statusSeq;
        const queueSeq = this.#queueSeq;
        const cloudSeq = this.#cloudSeq;
        const gen = this.#gen;
        const my = ++loadIssued;
        // The log is a request of its own, applied when it comes: the metrics never wait for it, and it shows while
        // they failed (their panel says so, with Retry). Older servers have no /logs; the stream still fills the view.
        const asked = this.#streamedLogs;
        const held = (l: RunLogs) => {
          // The server's count is as of its answer: the lines streamed in since it was asked for (the last ones held,
          // in the order they arrived, whatever their times: a cloud worker's lines keep theirs) and not in its answer
          // are on top of it. Without a count (an older server, a failed request) the one known stays.
          const tail = this.#streamedTail;
          const since = Math.min(tail.length, this.#streamedLogs - asked);
          const seen = new Set(l.lines.map((x) => `${x.t}|${x.message}`));
          const newer = tail.slice(tail.length - since).filter((x) => !seen.has(`${x.t}|${x.message}`)).length;
          const before = this.logs.length;
          this.logs = mergeLogs(l.lines, this.logs);
          if (l.total !== undefined) this.logsEarlier = Math.max(0, l.total + newer - this.logs.length);
          else if (before > this.logs.length) this.logsEarlier += before - this.logs.length;
        };
        if (withLogs) {
          api.runLogs(runId, MAX_LOGS, view.signal).then(
            (l) => {
              if (closed || my < logsApplied || gen !== this.#gen) return;
              logsApplied = my;
              held(l);
              this.#bump();
            },
            () => {},
          );
        }
        let snap: { run: Run; claim: () => boolean }, metrics: RunMetrics;
        const snapP = snapshot();
        if (onSnap) snapP.then(onSnap, () => {});
        if (postOpenSync) postOpenSnap = snapP;
        try {
          [snap, metrics] = await Promise.all([snapP, api.runMetrics(runId, view.signal)]);
        } catch (e) {
          if (my !== loadIssued) return false;
          throw e;
        } finally {
          if (postOpenSnap === snapP) postOpenSnap = null;
        }
        if (closed || my < loadApplied) return false;
        // Requested before a resume rewound the view (or before its replay ended): it may hold the interrupted
        // attempt. Ask again.
        if (gen !== this.#gen) return (sync(), false);
        loadApplied = my;
        // Final only if the server said so in this same response (its snapshot), not because the run ended since, and
        // only if it holds every epoch that snapshot says the run validated: /run and /metrics are read independently,
        // so the history may predate the last validation of a run the snapshot already calls ended.
        // Answered short of an ended snapshot `SHORT_ANSWERS` times in a row, the history is all there is: final too.
        const answered = metrics.epochs[metrics.epochs.length - 1]?.epoch ?? 0;
        const ended = !ACTIVE.includes(snap.run.status);
        if (!ended) {
          sawActive = true;
          postOpenSync = false;
        }
        this.#short.set((n) => (ended && heldResults(snap.run, answered, 0).pending ? n + 1 : 0));
        const final = ended && !heldResults(snap.run, answered, this.#short.get()).pending;
        this.#absorb(snap.claim() ? snap.run : (this.run ?? snap.run), metrics.steps, metrics.epochs, seq, queueSeq, cloudSeq, final);
        // Still replaying a resumed run: ask again shortly, until its new steps have replaced the old ones.
        if (this.#replay && this.run && ACTIVE.includes(this.run.status)) {
          clearTimeout(replayTimer);
          replayTimer = setTimeout(() => sync(), 3000);
        }
        this.loading = false;
        // A history that failed before (the panels said so) is in now.
        this.error = undefined;
        // The history just applied is current: no longer "last known" (DESIGN_DIRECTION §2.8).
        this.#heard = Date.now();
        // Still down (a retry of the load got through): the view is last known as of now.
        if (this.lost !== null) this.lost = this.connection === 'reconnecting' ? this.#heard : null;
        // Published with the run it came with, not a frame later: a finished run never shows without its epochs.
        this.#publish();
        return this.#pending();
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
        this.#viewSeq++;
        this.#endedId = null;
        view.abort();
        clearTimeout(lostTimer);
        clearTimeout(syncTimer);
        clearTimeout(replayTimer);
        unsub();
        this.#pendingStep = null;
        this.run = undefined;
        this.steps = [];
        this.epochs = [];
        this.logs = [];
        this.logsEarlier = 0;
        this.#streamedTail = [];
        this.lost = null;
        this.waiting = null;
        this.#server.set(undefined);
        this.#short.set(0);
        this.connection = 'idle';
        this.loading = false;
        this.error = e;
        this.#publish();
        return true;
      };
      // A dropped stream may mean the run is gone (the stream answers 404 then, and EventSource gives no status): one
      // snapshot at a time says so; any other answer leaves the reconnect to the stream.
      let checking = false;
      const check = () => {
        if (checking || closed) return;
        checking = true;
        api.run(runId, view.signal).then(
          () => void (checking = false),
          (e) => {
            checking = false;
            gone(e);
          },
        );
      };

      const sync = (attempt = 0, onSnap?: (snap: { run: Run; claim: () => boolean }) => void) => {
        clearTimeout(syncTimer);
        // `owed`: only if, by then, the epochs are still on their way (a load in between may have settled it).
        const later = (owed = false) => {
          clearTimeout(syncTimer);
          syncTimer = setTimeout(() => void ((!owed || this.#pending()) && sync(attempt + 1)), Math.min(15_000, 1000 * 2 ** attempt));
        };
        load(onSnap).then(
          // The server says the run validated epochs this history did not hold yet (read before they landed): again.
          (owed) => void (owed && !closed && later(true)),
          (e) => {
            if (closed || gone(e)) return;
            // Nothing loaded yet, or the ended run's validated epochs are still to come: the panels say so, with Retry,
            // as for a failed first load, rather than wait in silence (the retries go on).
            if (this.loading || this.#pending()) {
              this.error = e;
              this.loading = false;
            }
            later();
          },
        );
      };

      // Subscribe first: events that arrive while history loads are kept and merged, not lost.
      unsub = api.runEvents(runId, {
        connection: (s) => {
          if (closed) return;
          // EventSource construction does not mean the subscription is live: events emitted between our first
          // history load and the stream opening would be lost, so load (and merge) again once it opens — unless an
          // already-terminal run's final history has settled (if it resumed meanwhile, the stream's initial `status`
          // event reports an active status and triggers `#rewind()` + `sync()`).
          const cur = this.#run.peek();
          const settledTerminal = !sawActive && !this.#replay && !!cur && !ACTIVE.includes(cur.status) && !untrack(() => this.#pending()) && loadApplied > 0;
          if (s === 'open' && this.#connection.peek() === 'idle' && !settledTerminal) {
            postOpenSync = true;
            sync();
          }
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
          if (ACTIVE.includes(status)) {
            sawActive = true;
            postOpenSync = false;
            postOpenSnap = null;
          }
          // Finished → active again: the run was resumed (here or elsewhere).
          if (this.run && !ACTIVE.includes(this.run.status) && ACTIVE.includes(status)) {
            this.#rewind();
            sync();
          }
          if (this.run) {
            const next = withLength({ ...this.#takePendingStep(this.run), status, error }, steps_per_epoch, epochs);
            // Only a waiting run has a place in the queue.
            this.run = status !== 'queued' && next.queue_position != null ? { ...next, queue_position: null } : next;
          } else {
            // SSE status arrived before the initial `/run` + `/metrics` GETs finished: notify terminal immediately or clear `#endedId` on resume.
            this.#notifyEnded({ status, error });
          }
          // The trainer resolved the config by the time it reports `running` (docs/API.md): ask once for `resolved`.
          if (status === 'running' && this.run && this.run.resolved == null && !resolvedAsked) {
            resolvedAsked = true;
            api.run(runId, view.signal).then(
              (r) => {
                if (closed || !this.run || !r.resolved) return void (resolvedAsked = false);
                this.run = { ...this.#takePendingStep(this.run), resolved: r.resolved };
              },
              () => (resolvedAsked = false),
            );
          }
          if (!ACTIVE.includes(status)) {
            const applySnap = (seq: number, gen: number, cloudSeq: number) => (snap: { run: Run; claim: () => boolean }) => {
              if (closed || seq !== this.#statusSeq || gen !== this.#gen || !snap.claim()) return;
              this.#pendingStep = null;
              this.run = this.#clouded(keepStepsPerEpoch(snap.run, this.run), cloudSeq);
            };
            const cur = this.#run.peek();
            const settledTerminal =
              !sawActive &&
              !this.#replay &&
              !!cur &&
              !ACTIVE.includes(cur.status) &&
              cur.status === status &&
              cur.error === error &&
              !untrack(() => this.#pending()) &&
              loadApplied > 0;
            if (settledTerminal) {
              postOpenSync = false;
              postOpenSnap = null;
              return;
            }
            if (!sawActive && !this.#replay && postOpenSync) {
              // The stream's initial terminal `status` arrived right after `open` / `reconnect` already started a
              // post-subscribe load: skip a duplicate `/run` + `/metrics` + `/logs`, while still applying its snapshot
              // even if `/metrics` fails.
              postOpenSync = false;
              const inFlight = postOpenSnap;
              postOpenSnap = null;
              if (inFlight) inFlight.then(applySnap(this.#statusSeq, this.#gen, this.#cloudSeq), () => {});
              return;
            }
            // The run ended: the server's final history settles it. Requests issued before may hold a partial
            // history (or, during a replay, the interrupted attempt), even when a snapshot already said it ended
            // (/run and /metrics are read independently): invalidate them and fetch the final one, every time.
            this.#replay = null;
            this.#gen++;
            // The epochs the stream delivered are published with the end, not a frame later.
            this.#publish();
            // Terminal: refresh the snapshot for finished time, best and model id (even if history is still loading),
            // reusing the single `snapshot()` request issued by `sync()`.
            sync(0, applySnap(this.#statusSeq, this.#gen, this.#cloudSeq));
          }
        },
        step: (s) => {
          if (closed) return;
          sawActive = true;
          postOpenSync = false;
          postOpenSnap = null;
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
          const prev = this.#pendingStep;
          this.#pendingStep = {
            step: s.step,
            epoch: s.epoch,
            eta_s: s.eta_s,
            steps_per_epoch: s.steps_per_epoch !== undefined && s.steps_per_epoch > 0 ? s.steps_per_epoch : prev?.steps_per_epoch,
            epochs: s.epochs !== undefined && s.epochs > 0 ? s.epochs : prev?.epochs,
          };
          // A real run reports 0 steps per epoch until its trainer has loaded the data, i.e. in the snapshot taken
          // at start. Current servers put the run's length on every step (above); for older ones, whose steps lack
          // it, ask once more (only that field is taken; progress needs it).
          const curSpe = this.#pendingStep.steps_per_epoch ?? this.run?.steps_per_epoch;
          if (this.run && curSpe === 0 && s.steps_per_epoch === undefined && !speAsked) {
            speAsked = true;
            api.run(runId, view.signal).then(
              (r) => {
                if (closed || !this.run || r.steps_per_epoch <= 0) return void (speAsked = false);
                this.run = { ...this.#takePendingStep(this.run), steps_per_epoch: r.steps_per_epoch };
                this.#bump();
              },
              () => (speAsked = false),
            );
          }
          this.#bump();
        },
        epoch: (e) => {
          if (closed) return;
          sawActive = true;
          postOpenSync = false;
          postOpenSnap = null;
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
            this.run = { ...this.#takePendingStep(this.run), best: { epoch: e.epoch, map50_95: e.val.map50_95 } };
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
          if (this.run && this.run.status === 'queued') this.run = { ...this.#takePendingStep(this.run), queue_position: position };
        },
        cloud: (c) => {
          if (closed) return;
          this.#heard = Date.now();
          // State, not a change (docs/API.md, SSE `cloud`): the latest event is the run's `cloud`. It is kept for the
          // snapshots still in flight (#absorb), which may hold an older phase or arrive before any run is loaded.
          this.#cloudSeq++;
          this.#lastCloud = c;
          if (this.run) this.run = { ...this.#takePendingStep(this.run), cloud: c };
        },
        log: (l) => {
          if (closed) return;
          this.#heard = Date.now();
          if (!withLogs) return;
          this.#streamedLogs++;
          this.#streamedTail.push(l);
          if (this.#streamedTail.length >= MAX_LOGS + LOG_TRIM_BATCH) {
            this.#streamedTail.splice(0, this.#streamedTail.length - MAX_LOGS);
          }
          this.logs.push(l);
          this.logsEarlier += this.logsLog.trimTo(MAX_LOGS, LOG_TRIM_BATCH);
          this.#bump();
        },
        reconnect: () => {
          // The stream replays nothing: refetch history and merge it with what arrived meanwhile.
          if (!closed) {
            postOpenSync = !sawActive && !this.#replay;
            sync();
          }
        },
      });

      load().then(
        (owed) => void (owed && !closed && (syncTimer = setTimeout(() => void (this.#pending() && sync(1)), 1000))),
        (e) => {
          if (closed) return;
          this.error = e;
          this.loading = false;
        },
      );

      return () => {
        closed = true;
        this.#viewSeq++;
        this.#endedId = null;
        view.abort();
        clearTimeout(lostTimer);
        clearTimeout(syncTimer);
        clearTimeout(replayTimer);
        unsub();
        if (this.#raf) cancelAnimationFrame(this.#raf);
        this.#raf = 0;
        this.#pendingStep = null;
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
    const view = this.#viewSeq;
    const seq = this.#statusSeq;
    const queueSeq = this.#queueSeq;
    const cloudSeq = this.#cloudSeq;
    const r = await call();
    if (this.#viewSeq !== view) return r;
    this.#server.set(r);
    if (this.#statusSeq === seq && this.run && !ACTIVE.includes(this.run.status)) this.#rewind();
    // Progress comes from the rewind and the stream (the response may still carry the interrupted step count); the
    // status from the response unless an event already reported a newer one.
    const cur = this.run ? this.#takePendingStep(this.run) : undefined;
    const status = this.#statusSeq === seq ? { status: r.status, error: r.error } : (this.#lastStatus ?? { status: r.status, error: r.error });
    this.run = this.#clouded(this.#placed({ ...r, ...(cur ? { step: cur.step, epoch: cur.epoch, best: cur.best } : {}), ...status }, queueSeq), cloudSeq);
    return r;
  }

  /**
   * Stops the run through `call` (POST /api/runs/{id}/stop, or a resumed run's removal from the queue). The answer is
   * the run as the server had it once it took the stop. It applies only over the state it answered: if the view heard
   * anything about the run's status meanwhile (a status event, a snapshot, a resume), the stream is newer than the
   * answer or about to be (the server sends every change of status as an event, the stop's own `stopping` too), and
   * the view keeps what it has: the run may have ended, or been resumed elsewhere after it stopped. Returns the run as
   * the view now has it.
   */
  async stop(call: () => Promise<Run | null>): Promise<Run | null> {
    const view = this.#viewSeq;
    const seq = this.#statusSeq;
    const attempt = this.#attempt;
    const before = this.run?.status;
    const r = await call();
    if (this.#viewSeq !== view) return r;
    if (!r) return null;
    const cur = this.run ? this.#takePendingStep(this.run) : undefined;
    if (cur && (seq !== this.#statusSeq || attempt !== this.#attempt || cur.status !== before)) {
      if (cur !== this.run) this.run = cur;
      return cur;
    }
    this.run = keepStepsPerEpoch(r, cur);
    return this.run ?? r;
  }

  /**
   * Drops the steps after the last finished epoch (1-based; a resumed run repeats them), moves the run there and
   * starts a replay.
   */
  #rewind() {
    this.#gen++;
    this.#attempt++;
    this.#pendingStep = null;
    this.#replay = { until: this.steps[this.steps.length - 1]?.step ?? 0, steps: [], epochs: [] };
    const done = this.epochs[this.epochs.length - 1]?.epoch ?? 0;
    this.steps = this.steps.filter((s) => s.epoch <= done);
    const kept = this.steps[this.steps.length - 1]?.step ?? 0;
    if (this.run) this.run = { ...this.run, step: kept, epoch: done };
    this.#bump();
  }

  /** The run ended and the epochs it validated are still on their way to this view (progress `heldResults`). */
  #pending(): boolean {
    return !!this.run && !ACTIVE.includes(this.run.status) && !!untrack(() => this.results?.pending);
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
    if (live && this.#pendingStep) next = withLength(next, this.#pendingStep.steps_per_epoch, this.#pendingStep.epochs);
    this.#pendingStep = null;
    next = keepStepsPerEpoch(next, this.run);
    // During a replay the snapshot may still count the interrupted attempt: progress follows the steps held.
    if (this.#replay && live) next = { ...next, step: last?.step ?? 0, epoch: last?.epoch ?? this.epochs[this.epochs.length - 1]?.epoch ?? 0 };
    for (const e of this.epochs)
      if (!next.best || e.val.map50_95 > next.best.map50_95) next = { ...next, best: { epoch: e.epoch, map50_95: e.val.map50_95 } };
    // A status event that arrived after this request was sent is newer than the snapshot.
    if (this.#statusSeq !== statusSeqAtStart && this.#lastStatus) next = { ...next, ...this.#lastStatus };
    this.run = this.#clouded(this.#placed(next, queueSeqAtStart), cloudSeqAtStart);
  }

  /** Captures the current SSE sequence counters before issuing an out-of-band queue/detail request. */
  queueToken(): { viewSeq: number; statusSeq: number; queueSeq: number } {
    return { viewSeq: this.#viewSeq, statusSeq: this.#statusSeq, queueSeq: this.#queueSeq };
  }

  /**
   * Applies a detail snapshot fetched for queue reconciliation (`queueState`), respecting SSE status and queue
   * events that arrived after the request started so an older HTTP response never moves the run backwards.
   */
  applyQueueSnapshot(
    id: string,
    r: Run,
    atStart: { viewSeq: number; statusSeq: number; queueSeq: number },
  ) {
    if (this.#viewSeq !== atStart.viewSeq) return;
    const cur = this.#run.peek();
    if (!cur || cur.id !== id) return;
    // A stream that has since said the run left the queue is newer, and a run that left it never goes back.
    if (r.status === 'queued' && cur.status !== 'queued') return;
    let next: Run = {
      ...cur,
      status: r.status,
      started: r.started,
      queue_position: r.queue_position ?? null,
      resolved: r.resolved ?? cur.resolved,
    };
    if (this.#statusSeq !== atStart.statusSeq && this.#lastStatus) next = { ...next, ...this.#lastStatus };
    this.run = this.#placed(next, atStart.queueSeq);
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

  /** Consumes any coalesced step progress and returns `run` updated with it. */
  #takePendingStep(run: Run): Run {
    const s = this.#pendingStep;
    if (!s) return run;
    this.#pendingStep = null;
    return withLength({ ...run, step: s.step, epoch: s.epoch, eta_s: s.eta_s }, s.steps_per_epoch, s.epochs);
  }

  /** Applies coalesced step progress and bumps `version` in a single reactive batch. */
  #flush() {
    this.#raf = 0;
    batch(() => {
      if (this.#pendingStep) {
        const cur = this.#run.peek();
        if (cur) this.#run.set(this.#takePendingStep(cur));
      }
      this.#version.set((v) => v + 1);
    });
  }

  /** Publishes now rather than at the next frame (`#bump`): for what must be read together with the run's status. */
  #publish() {
    if (this.#raf) cancelAnimationFrame(this.#raf);
    this.#flush();
  }

  #bump() {
    if (typeof requestAnimationFrame === 'undefined') {
      this.#flush();
      return;
    }
    if (this.#raf) return;
    this.#raf = requestAnimationFrame(() => this.#flush());
  }
}
