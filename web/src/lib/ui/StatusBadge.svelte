<script lang="ts">
  import type { RunStatus } from '$lib/api/types';

  let {
    status,
    size = 'md',
    plain = false,
    dotOnly = false,
  }: {
    status: RunStatus;
    size?: 'sm' | 'md';
    /** Dot and word without the pill, for table cells. */
    plain?: boolean;
    /** The dot alone, where the word is written next to it (still announced to assistive tech). */
    dotOnly?: boolean;
  } = $props();

  const labels: Record<RunStatus, string> = {
    queued: 'Queued',
    running: 'Running',
    stopping: 'Stopping',
    stopped: 'Stopped',
    completed: 'Completed',
    failed: 'Failed',
  };
</script>

<span class="badge {status} {size}" class:plain>
  <!-- Each status has its own mark as well as its hue (running and completed are both green-ish): a ringed dot
       while running, a check when completed, a cross when failed, hollow while queued, a square when stopped. -->
  {#if status === 'completed'}
    <svg class="mark" viewBox="0 0 8 8" aria-hidden="true"><path d="M1.2 4.3 3.2 6.2 6.8 1.9" /></svg>
  {:else if status === 'failed'}
    <svg class="mark" viewBox="0 0 8 8" aria-hidden="true"><path d="M1.8 1.8 6.2 6.2M6.2 1.8 1.8 6.2" /></svg>
  {:else}
    <span class="dot" aria-hidden="true"></span>
  {/if}{#if dotOnly}<span class="sr-only">{labels[status]}</span>{:else}{labels[status]}{/if}
</span>

<style>
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
    animation: pulse 1.6s var(--ease-out) infinite;
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
  /* In tables: the hue is on the dot only, the word stays in ink. */
  .plain {
    height: auto;
    padding: 0;
    background: none;
    color: var(--text-secondary);
    font-size: inherit;
    font-weight: var(--weight-regular);
  }
  .plain .dot {
    background: var(--c);
  }
  .plain.queued .dot {
    background: none;
    box-shadow: inset 0 0 0 1.5px var(--c);
  }
  .plain.running {
    color: var(--text-primary);
  }
  @keyframes pulse {
    0% {
      box-shadow: 0 0 0 0 color-mix(in srgb, var(--c) 65%, transparent);
    }
    100% {
      box-shadow: 0 0 0 5px transparent;
    }
  }
</style>
