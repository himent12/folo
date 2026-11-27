// Derives spot preemption and automatic checkpoint-resume status for cloud runs from `RunCloud`,
// `Run` outcome fields, and the run's log lines emitted by `crates/folo-server/src/cloud/executor/`.
// Distinguishes live automatic spot recovery from permanent failures (failed replacement worker,
// exhausted retry budget, or missing synced checkpoint), resets across manual-resume boundaries,
// and memoizes results structurally so 10 Hz training ticks never churn DOM nodes.
import type { AppendLog, LogCursor } from '$lib/api/appendLog';
import type { LogLine, Run, RunCloud } from '$lib/api/types';

export type SpotRecoveryKind = 'recovering' | 'resumed' | 'failed' | 'exhausted' | 'no_checkpoint' | 'stopped';

export interface SpotRecovery {
  kind: SpotRecoveryKind;
  /** Number of automatic preemption resume attempts observed in the current execution segment. */
  attempts: number;
  /** Retry budget reported by the server (`attempt N/M`), or null when not reported. */
  maxAttempts: number | null;
  /** Checkpoint epoch resumed from (`epoch E`), or null when not reported. */
  resumeEpoch: number | null;
  /** Backoff delay in seconds if the latest attempt reported one. */
  backoffSeconds: number | null;
  /** Short badge label (e.g. "Spot auto-resume 1/3", "Auto-resumed (1×)", "Spot recovery failed"). */
  badge: string;
  /** Human-readable status copy distinguishing automatic spot recovery from permanent failure. */
  summary: string;
}

export type SpotRecoveryRun = Pick<Run, 'status' | 'error' | 'error_code'> & { id?: string };

interface ScanAcc {
  sawRecovery: boolean;
  /** True while a preemption recovery attempt is in flight and no replacement worker has taken the run yet. */
  pendingAttempt: boolean;
  /** Number of preemption recovery attempts where a replacement worker actually took the run. */
  completedRecoveries: number;
  attempts: number;
  maxAttempts: number | null;
  resumeEpoch: number | null;
  backoffSeconds: number | null;
  exhaustedMax: number | null;
  noCheckpoint: boolean;
  stoppedDuringRecovery: boolean;
  /** True when the log stream crossed a manual-resume boundary (`POST /api/runs` with `resume_run_id`). */
  sawManualReset: boolean;
}

const MANUAL_RESUME_RE = /^(?:resuming from weights\/last\.folo|no epoch finished before; starting over)$/i;
const AUTO_RESUME_RE =
  /auto-resuming from synced checkpoint[^\n]*?\(epoch (\d+),\s*attempt (\d+)\/(\d+)(?: after ([\d.]+)s backoff)?\)/i;
const RELEASE_PREEMPTED_RE = /releasing the preempted instance before resuming on a new worker/i;
const RESTART_RECOVERY_RE = /server restarted during preemption recovery; continuing auto-resume/i;
const WORKER_TOOK_RUN_RE = /the worker took the run as \S+/i;
const EXHAUSTED_RE = /automatic checkpoint resume exhausted all (\d+) attempt/i;
const NO_CHECKPOINT_RE =
  /no (?:synced checkpoint \(weights\/last\.folo\)|incomplete weights\/last\.folo) is available to (?:auto-)?resume from|has no incomplete training state to auto-resume from/i;
const STOPPED_RECOVERY_RE =
  /stop(?:ped| requested) during preemption recovery|ended the run (?:during preemption recovery|after worker preemption)/i;

function emptyAcc(): ScanAcc {
  return {
    sawRecovery: false,
    pendingAttempt: false,
    completedRecoveries: 0,
    attempts: 0,
    maxAttempts: null,
    resumeEpoch: null,
    backoffSeconds: null,
    exhaustedMax: null,
    noCheckpoint: false,
    stoppedDuringRecovery: false,
    sawManualReset: false,
  };
}

function scanMessage(acc: ScanAcc, message: string): void {
  const trimmed = message.trim();
  if (MANUAL_RESUME_RE.test(trimmed)) {
    // Manual resume (`resume_run_id`) resets the server's preemption state (`Private::default()`, `preemptions = 0`).
    Object.assign(acc, emptyAcc(), { sawManualReset: true });
    return;
  }
  const auto = AUTO_RESUME_RE.exec(message);
  if (auto) {
    acc.sawRecovery = true;
    acc.pendingAttempt = true;
    const ep = Number(auto[1]);
    const att = Number(auto[2]);
    const max = Number(auto[3]);
    const backoff = auto[4] !== undefined ? Number(auto[4]) : null;
    if (Number.isFinite(ep)) acc.resumeEpoch = ep;
    if (Number.isFinite(att)) acc.attempts = Math.max(acc.attempts, att);
    if (Number.isFinite(max)) acc.maxAttempts = max;
    acc.backoffSeconds = backoff !== null && Number.isFinite(backoff) ? backoff : null;
    return;
  }
  if (RELEASE_PREEMPTED_RE.test(message) || RESTART_RECOVERY_RE.test(message)) {
    acc.sawRecovery = true;
    if (!acc.pendingAttempt) {
      acc.pendingAttempt = true;
      acc.attempts = Math.max(acc.attempts, acc.completedRecoveries + 1);
    }
    return;
  }
  if (WORKER_TOOK_RUN_RE.test(message)) {
    if (acc.pendingAttempt) {
      acc.pendingAttempt = false;
      acc.completedRecoveries += 1;
    }
    return;
  }
  const ex = EXHAUSTED_RE.exec(message);
  if (ex) {
    const max = Number(ex[1]);
    if (Number.isFinite(max)) acc.exhaustedMax = max;
  }
  if (NO_CHECKPOINT_RE.test(message)) {
    acc.noCheckpoint = true;
  }
  if (STOPPED_RECOVERY_RE.test(message)) {
    acc.stoppedDuringRecovery = true;
  }
}

function recoveringStepText(phase: string): string {
  switch (phase) {
    case 'booting':
      return 'booting a replacement worker';
    case 'syncing':
      return 'syncing the dataset and checkpoint to a replacement worker';
    case 'provisioning':
      return 'provisioning a replacement worker';
    default:
      return 'releasing the preempted worker and provisioning a replacement';
  }
}

function sameRecovery(a: SpotRecovery | null, b: SpotRecovery | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    a.kind === b.kind &&
    a.attempts === b.attempts &&
    a.maxAttempts === b.maxAttempts &&
    a.resumeEpoch === b.resumeEpoch &&
    a.backoffSeconds === b.backoffSeconds &&
    a.badge === b.badge &&
    a.summary === b.summary
  );
}

function resolveRecovery(
  acc: ScanAcc,
  cloud: Pick<RunCloud, 'phase'>,
  run?: SpotRecoveryRun,
): SpotRecovery | null {
  // Also inspect run.error in case logs were truncated or not yet loaded on a failed/stopped run.
  let exhaustedMax = acc.exhaustedMax;
  let noCheckpoint = acc.noCheckpoint;
  let stoppedDuringRecovery = acc.stoppedDuringRecovery;
  if (run?.error) {
    const ex = EXHAUSTED_RE.exec(run.error);
    if (ex && Number.isFinite(Number(ex[1]))) exhaustedMax = Number(ex[1]);
    if (NO_CHECKPOINT_RE.test(run.error)) noCheckpoint = true;
    if (STOPPED_RECOVERY_RE.test(run.error)) stoppedDuringRecovery = true;
  }

  const attempts = exhaustedMax !== null ? Math.max(acc.attempts, exhaustedMax) : acc.attempts;
  const maxAttempts = exhaustedMax ?? acc.maxAttempts;
  const resumeEpoch = acc.resumeEpoch;
  const backoffSeconds = acc.backoffSeconds;

  if (run?.status === 'failed') {
    const isLostErr = run.error_code === 'cloud_instance_lost' || run.error_code == null;
    if (isLostErr && exhaustedMax !== null) {
      const unit = exhaustedMax === 1 ? 'attempt' : 'attempts';
      return {
        kind: 'exhausted',
        attempts,
        maxAttempts,
        resumeEpoch,
        backoffSeconds,
        badge: 'Auto-resume exhausted',
        summary: `Spot worker preempted; automatic checkpoint resume exhausted all ${exhaustedMax} ${unit}. Permanent failure — resume manually to continue from the last synced checkpoint.`,
      };
    }
    if (isLostErr && noCheckpoint) {
      return {
        kind: 'no_checkpoint',
        attempts,
        maxAttempts,
        resumeEpoch,
        backoffSeconds,
        badge: 'Preempted · no checkpoint',
        summary:
          'Spot worker was preempted with no resumable synced checkpoint (weights/last.folo), so automatic recovery could not run. Permanent failure — start the run again to retry.',
      };
    }
    if (acc.pendingAttempt) {
      const attText = maxAttempts !== null ? `${attempts} of ${maxAttempts}` : `${attempts}`;
      const priorNote =
        acc.completedRecoveries > 0
          ? ` (after ${acc.completedRecoveries} earlier ${acc.completedRecoveries === 1 ? 'recovery' : 'recoveries'})`
          : '';
      return {
        kind: 'failed',
        attempts,
        maxAttempts,
        resumeEpoch,
        backoffSeconds,
        badge: 'Spot recovery failed',
        summary: `Spot recovery attempt ${attText} failed before a replacement worker resumed training${priorNote}. Permanent failure — resume manually to continue from the last synced checkpoint.`,
      };
    }
    if (acc.completedRecoveries > 0) {
      const n = acc.completedRecoveries;
      const count = n === 1 ? '1 spot preemption' : `${n} spot preemptions`;
      return {
        kind: 'resumed',
        attempts: n,
        maxAttempts,
        resumeEpoch,
        backoffSeconds,
        badge: n === 1 ? 'Auto-resumed (1×)' : `Auto-resumed (${n}×)`,
        summary: `Automatically recovered from ${count}${resumeEpoch !== null ? ` (last resumed from epoch ${resumeEpoch})` : ''} before the run ended.`,
      };
    }
    return null;
  }

  if (run?.status === 'stopped') {
    if (stoppedDuringRecovery || acc.pendingAttempt) {
      const attText =
        attempts > 0 ? ` (attempt ${attempts}${maxAttempts !== null ? ` of ${maxAttempts}` : ''})` : '';
      return {
        kind: 'stopped',
        attempts,
        maxAttempts,
        resumeEpoch,
        backoffSeconds,
        badge: 'Stopped during spot recovery',
        summary: `Stopped while auto-resuming from spot preemption${attText}. Synced checkpoint weights/last.folo was kept.`,
      };
    }
    if (acc.completedRecoveries > 0) {
      const n = acc.completedRecoveries;
      const count = n === 1 ? '1 spot preemption' : `${n} spot preemptions`;
      return {
        kind: 'resumed',
        attempts: n,
        maxAttempts,
        resumeEpoch,
        backoffSeconds,
        badge: n === 1 ? 'Auto-resumed (1×)' : `Auto-resumed (${n}×)`,
        summary: `Recovered automatically from ${count}${resumeEpoch !== null ? ` (resumed from epoch ${resumeEpoch})` : ''} before the run stopped.`,
      };
    }
    return null;
  }

  if (!acc.sawRecovery) return null;

  const active =
    run?.status === undefined ||
    run.status === 'queued' ||
    run.status === 'running' ||
    run.status === 'stopping';
  if (active && acc.pendingAttempt) {
    const attemptBadge =
      maxAttempts !== null ? `Spot auto-resume ${attempts}/${maxAttempts}` : `Spot auto-resume #${attempts}`;
    const attemptDetail =
      maxAttempts !== null ? `attempt ${attempts} of ${maxAttempts}` : `attempt ${attempts}`;
    const epochNote = resumeEpoch !== null ? ` from epoch ${resumeEpoch}` : '';
    return {
      kind: 'recovering',
      attempts,
      maxAttempts,
      resumeEpoch,
      backoffSeconds,
      badge: attemptBadge,
      summary: `Spot worker preempted — automatically ${recoveringStepText(cloud.phase)} to resume${epochNote} (${attemptDetail}). Automatic recovery in progress, not a permanent failure.`,
    };
  }

  const completed = acc.completedRecoveries > 0 ? acc.completedRecoveries : attempts;
  if (completed <= 0) return null;
  const count = completed === 1 ? '1 spot preemption' : `${completed} spot preemptions`;
  return {
    kind: 'resumed',
    attempts: completed,
    maxAttempts,
    resumeEpoch,
    backoffSeconds,
    badge: completed === 1 ? 'Auto-resumed (1×)' : `Auto-resumed (${completed}×)`,
    summary: `Recovered automatically from ${count}${resumeEpoch !== null ? ` (resumed from epoch ${resumeEpoch})` : ''}.`,
  };
}

/** Pure helper deriving spot preemption recovery status from a cloud run's state and log lines. */
export function parseSpotRecovery(
  cloud: Pick<RunCloud, 'phase'>,
  run: SpotRecoveryRun | undefined,
  logs: readonly LogLine[],
): SpotRecovery | null {
  const acc = emptyAcc();
  for (const line of logs) scanMessage(acc, line.message);
  return resolveRecovery(acc, cloud, run);
}

/**
 * Incremental scanner over `LiveRun.logsLog` (mirroring `WarningsLog`) so 10 Hz training ticks
 * never re-scan the entire run log or churn downstream DOM nodes when recovery status is unchanged.
 */
export class SpotRecoveryTracker {
  #cursor: LogCursor | null = null;
  #acc: ScanAcc = emptyAcc();
  #last: SpotRecovery | null = null;
  #runId: string | null = null;

  sync(
    log: AppendLog<LogLine>,
    cloud: Pick<RunCloud, 'phase'> | undefined,
    run: SpotRecoveryRun | undefined,
  ): SpotRecovery | null {
    const runId = run?.id ?? null;
    const sameRun = runId !== null && this.#runId === runId;
    if (runId !== null && !sameRun) {
      this.#runId = runId;
      this.#acc = emptyAcc();
      this.#cursor = null;
      this.#last = null;
    }

    const r = log.read(this.#cursor);
    this.#cursor = r.cursor;
    if (r.reset) {
      const fresh = emptyAcc();
      for (const line of r.added) scanMessage(fresh, line.message);
      const freshHasSignal =
        fresh.sawManualReset ||
        fresh.sawRecovery ||
        fresh.exhaustedMax !== null ||
        fresh.noCheckpoint ||
        fresh.stoppedDuringRecovery;
      if (!sameRun || r.added.length === 0 || freshHasSignal) {
        this.#acc = fresh;
      } else {
        // Reconnect snapshot replaced the log tail after older preemption lines aged out of the 2,000-line
        // window for the same run: preserve already-observed recovery history and scan only new lines.
        for (const line of r.added) scanMessage(this.#acc, line.message);
      }
    } else {
      for (const line of r.added) {
        scanMessage(this.#acc, line.message);
      }
    }

    const next = cloud ? resolveRecovery(this.#acc, cloud, run) : null;
    if (sameRecovery(this.#last, next)) return this.#last;
    this.#last = next;
    return next;
  }
}
