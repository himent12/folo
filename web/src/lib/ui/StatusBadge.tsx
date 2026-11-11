/**
 * A run's status as its mark and word. Two forms, one rule (DESIGN_DIRECTION 6): the tinted pill is the title form,
 * beside the name of the one run a screen is about (the run page's header, Overview's live card), at most one per
 * screen; everywhere else (tables, lists, Compare, meta lines) it is `plain`, dot + word.
 * It takes the run itself (`run`), not its bare status, so a cloud run in its hand-over
 * says its phase in the neutral look of a waiting run wherever a status is shown; anything else with a run's states
 * (a labeling job) passes its `state`, the status with its own word.
 */
import { computed, css } from 'fw';
import type { Run, RunStatus } from '$lib/api/types';
import { inHandover, phaseLabel } from '$lib/cloud';

/** A status's one word, in sentence case: never "queued". */
const STATUS_LABEL: Record<RunStatus, string> = {
  queued: 'Queued',
  running: 'Running',
  stopping: 'Stopping',
  stopped: 'Stopped',
  completed: 'Completed',
  failed: 'Failed',
};

/** What a run's status is read from: the status and, for a cloud run, its hand-over (`Run.cloud`). */
export type RunState = Pick<Run, 'status' | 'cloud'>;

/**
 * A run's status in one word, wherever it is written (the badge, lists, meta lines, the palette): a cloud run in its
 * hand-over is its phase ("Provisioning", "Uploading data"), since it waits in no queue.
 */
export const statusWord = (run: RunState): string => (inHandover(run) ? phaseLabel(run.cloud.phase) : STATUS_LABEL[run.status]);

/**
 * Something that is not a run but has a run's states (a labeling job): the run status whose mark it shares and its
 * own word ("Labeling", "Idle"). The word is required, so a run's bare status never gets here: a run passes `run`.
 */
export interface OwnStatus {
  status: RunStatus;
  label: string;
}

export type StatusBadgeProps = ({ run: RunState; state?: never } | { state: OwnStatus; run?: never }) & {
  size?: 'sm' | 'md';
  /** Dot and word without the pill, for table cells. */
  plain?: boolean;
  /** The dot alone, where the word is written next to it (still announced to assistive tech). */
  dotOnly?: boolean;
  /**
   * The status is no longer current (its source went quiet): the whole words to show instead, "Last known: Running ·
   * 2m ago" (`lib/api/freshness`). The pill turns neutral and the heartbeat stops: nothing here says "now" any more.
   */
  lastKnown?: string;
};

export default function StatusBadge(props: StatusBadgeProps) {
  const status = computed(() => (props.run ? props.run.status : props.state!.status));
  const word = computed(() => props.lastKnown ?? (props.run ? statusWord(props.run) : props.state!.label));

  // Each status has its own mark as well as its hue (running and completed are both green-ish): a ringed dot while
  // running, a check when completed, a cross when failed, hollow while queued, a square when stopped.
  return (
    <span
      class={`badge ${status.get()} ${props.size ?? 'md'}`}
      class:plain={props.plain ?? false}
      class:alone={props.dotOnly ?? false}
      class:held={!!props.lastKnown}
    >
      {status.get() === 'completed' ? (
        <svg class="mark" viewBox="0 0 8 8" aria-hidden="true">
          <path d="M1.2 4.3 3.2 6.2 6.8 1.9" />
        </svg>
      ) : status.get() === 'failed' ? (
        <svg class="mark" viewBox="0 0 8 8" aria-hidden="true">
          <path d="M1.8 1.8 6.2 6.2M6.2 1.8 1.8 6.2" />
        </svg>
      ) : (
        <span class="dot" aria-hidden="true"></span>
      )}
      {props.dotOnly ? <span class="sr-only">{word.get()}</span> : word.get()}
    </span>
  );
}

css`
  .badge {
    --c: var(--text-tertiary);
    --bg: var(--bg-active);
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 20px;
    padding: 0 8px 0 6px;
    border-radius: var(--radius-full);
    background: var(--bg);
    color: var(--c);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    line-height: 1;
    white-space: nowrap;
  }
  .sm {
    height: 18px;
    padding: 0 6px 0 4px;
  }
  /* Every mark sits in the same 8 px slot, so names after it line up whatever the status. */
  .dot {
    width: 6px;
    height: 6px;
    margin: 0 1px;
    border-radius: 50%;
    background: currentColor;
    flex-shrink: 0;
  }
  .running {
    --c: var(--accent-text);
    --bg: var(--accent-muted);
  }
  .mark {
    width: 8px;
    height: 8px;
    flex-shrink: 0;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.6;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .plain .mark {
    stroke: var(--c);
  }
  /* Running: a dot in a ring (it reads "live" even when the pulse is off for reduced motion). */
  .running .dot {
    width: 4px;
    height: 4px;
    margin: 0 2px;
    background: var(--c);
    outline: 1px solid var(--c);
    outline-offset: 1px;
    --pulse: var(--c);
    animation: live-pulse 2.4s var(--ease-out) infinite;
  }
  .completed {
    --c: var(--success-text);
    --bg: var(--success-muted);
  }
  .failed {
    --c: var(--danger-text);
    --bg: var(--danger-muted);
  }
  .stopping {
    --c: var(--warning-text);
    --bg: var(--warning-muted);
  }
  /* Waiting is not a signal: neutral grey, with a hollow dot. */
  .queued {
    --c: var(--text-secondary);
    --bg: var(--bg-active);
  }
  .queued .dot {
    background: none;
    box-shadow: inset 0 0 0 1.5px currentColor;
  }
  .stopped .dot {
    border-radius: 1.5px;
  }
  /* In tables: the hue is on the dot only, the word stays in ink. The badge takes the cell's own line box (and sits at
     its top), so the word shares the row's baseline instead of a flex box's synthesized one a couple of px lower. A word
     of two (a cloud run's "Uploading data") may take two lines where its column is tight (Runs on a tablet), as the
     row's name does, rather than widen the table past its panel. */
  .plain {
    align-items: flex-start;
    white-space: normal;
    height: auto;
    padding: 0;
    line-height: inherit;
    vertical-align: top;
    background: none;
    color: var(--text-secondary);
    font-size: inherit;
    font-weight: var(--weight-regular);
  }
  .plain .dot {
    background: var(--c);
  }
  /* Beside its word, the mark sits on the first line's centre, also when a word of two wraps; alone, its row centres it. */
  .plain:not(.alone) .dot {
    margin-top: calc((1lh - 6px) / 2);
  }
  .plain:not(.alone) .mark {
    margin-top: calc((1lh - 8px) / 2);
  }
  .plain.running:not(.alone) .dot {
    margin-top: calc((1lh - 4px) / 2);
  }
  .plain.queued .dot {
    background: none;
    box-shadow: inset 0 0 0 1.5px var(--c);
  }
  .plain.running {
    color: var(--text-primary);
  }
  /* A last known status: neutral, still. Its mark keeps its shape (the ringed dot still says which status it was), but
     neither lime nor a heartbeat claims it is current. */
  .held {
    --c: var(--text-secondary);
    --bg: var(--bg-active);
  }
  .held .dot {
    animation: none;
  }
  /* The live heartbeat (\`live-pulse\`, base.css): one ring every 2.4 s; off under reduced motion, where the ringed dot
     and the word still say "running". */
  @media (prefers-reduced-motion: reduce) {
    .running .dot {
      animation: none;
    }
  }
`;
