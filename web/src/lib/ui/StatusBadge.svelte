<script lang="ts">
  import type { RunStatus } from '$lib/api/types';

  let { status, size = 'md' }: { status: RunStatus; size?: 'sm' | 'md' } = $props();

  const labels: Record<RunStatus, string> = {
    queued: 'Queued',
    running: 'Running',
    stopping: 'Stopping',
    stopped: 'Stopped',
    completed: 'Completed',
    failed: 'Failed',
  };
</script>

<span class="badge {status} {size}">
  <span class="dot" aria-hidden="true"></span>{labels[status]}
</span>

<style>
  .badge {
    --c: var(--text-tertiary);
    --bg: var(--bg-active);
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 20px;
    padding: 0 8px 0 7px;
    border-radius: var(--radius-full);
    background: var(--bg);
    color: var(--c);
    font-size: var(--type-xs-size);
    font-weight: var(--weight-medium);
    line-height: 1;
    white-space: nowrap;
  }
  .sm {
    height: 18px;
    padding: 0 6px 0 5px;
  }
  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    flex-shrink: 0;
  }
  .running {
    --c: var(--accent-text);
    --bg: var(--accent-muted);
  }
  .running .dot {
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
  @keyframes pulse {
    0% {
      box-shadow: 0 0 0 0 color-mix(in srgb, currentColor 65%, transparent);
    }
    100% {
      box-shadow: 0 0 0 5px transparent;
    }
  }
</style>
