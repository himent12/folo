<script lang="ts">
  /**
   * A finished run's outcome as one line under the header (status mark, what happened, the next action); it wraps
   * rather than grows into a card. A failure keeps its cause in full, since it decides what to do next, except out of
   * GPU memory: that is said plainly, with the smaller settings to retry with as actions and the driver's report one
   * step away.
   */
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import Play from '@lucide/svelte/icons/play';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import ScanSearch from '@lucide/svelte/icons/scan-search';
  import Square from '@lucide/svelte/icons/square';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import type { Run } from '$lib/api/types';
  import { deviceName, duration, metric } from '$lib/format';
  import { isOom, oomDetail, oomFixes, retryPath } from '$lib/oom';
  import { href } from '$lib/router.svelte';
  import Button from '$lib/ui/Button.svelte';

  let {
    run,
    elapsed,
    resuming = false,
    onresume,
  }: { run: Run; elapsed: number | null; resuming?: boolean; onresume: () => void } = $props();

  // Out of GPU memory: the same config fails again, so the ways out are smaller settings (a new run), most likely first.
  const oom = $derived(run.status === 'failed' && isOom(run.error));
  const fixes = $derived(oom ? oomFixes(run.config) : []);
  const driver = $derived(oom ? oomDetail(run.error) : null);
</script>

{#if run.status === 'failed' && oom}
  <!-- What happened in plain words, the fixes as actions under it (the likeliest one primary), and the driver's
       report one step away. -->
  <div class="callout danger stack" role="alert">
    <TriangleAlert size={16} aria-hidden="true" />
    <p class="c-text">
      <span class="c-title">Ran out of GPU memory</span>
      <span class="c-msg">At epoch {run.epoch}, batch {run.config.batch} at {run.config.imgsz} px with <span class="mono">{run.config.arch}</span> needed more memory than {run.device ? `the ${deviceName(run.device)}` : 'the GPU'} has.</span>
    </p>
    <div class="c-acts">
      {#each fixes as f, i (f.kind)}
        <Button size="sm" variant={i === 0 ? 'primary' : 'secondary'} icon={i === 0 ? RotateCcw : undefined} href={href(retryPath(run.id, f))}>{f.label}</Button>
      {/each}
      {#if !fixes.length}<Button size="sm" icon={RotateCcw} href={href(retryPath(run.id))}>New run like this</Button>{/if}
    </div>
    {#if driver}
      <details class="c-detail">
        <summary>Driver report</summary>
        <code class="mono">{driver}</code>
      </details>
    {/if}
  </div>
{:else if run.status === 'failed'}
  <div class="callout danger" role="alert">
    <TriangleAlert size={16} aria-hidden="true" />
    <p class="c-text"><span class="c-title">Failed at epoch {run.epoch}</span> <span class="c-msg">{run.error}</span></p>
    <div class="c-acts">
      <!-- Anything but out of memory can continue from the last checkpoint. -->
      <Button size="sm" icon={Play} loading={resuming} onclick={onresume}>Resume</Button>
      <Button size="sm" icon={RotateCcw} href={href(retryPath(run.id))}>New run like this</Button>
    </div>
  </div>
{:else if run.status === 'completed'}
  <div class="callout success" role="status">
    <CircleCheck size={16} aria-hidden="true" />
    <p class="c-text">
      <span class="c-title">Completed in {duration(elapsed)}</span>
      <span class="c-msg brief">Best mAP50-95 <span class="num">{metric(run.best?.map50_95)}</span> at epoch {run.best?.epoch ?? 0}, saved as a model.</span>
    </p>
    {#if run.model_id}<div class="c-acts"><Button size="sm" icon={ScanSearch} href={href(`/playground?model=${run.model_id}`)}>Try in Playground</Button></div>{/if}
  </div>
{:else if run.status === 'stopped'}
  <div class="callout neutral" role="status">
    <Square size={16} aria-hidden="true" />
    <p class="c-text">
      <span class="c-title">Stopped at epoch {run.epoch}</span>
      <span class="c-msg">The last checkpoint was kept{run.model_id ? ' and saved as a model' : ''}; Resume continues from it.</span>
    </p>
    <div class="c-acts"><Button size="sm" icon={Play} loading={resuming} onclick={onresume}>Resume</Button></div>
  </div>
{/if}

<style>
  .callout {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-2) var(--space-3);
    min-height: 44px;
    padding: var(--space-2) var(--space-2) var(--space-2) var(--space-4);
    background: var(--bg-panel);
    border: 1px solid var(--line-default);
    border-radius: var(--radius-lg);
  }
  /* The status mark carries the hue; the text stays in ink, so it reads at AA on the panel in both themes. */
  .callout > :global(svg) {
    flex-shrink: 0;
  }
  .danger > :global(svg) {
    color: var(--danger-text);
  }
  .success > :global(svg) {
    color: var(--success-text);
  }
  .neutral > :global(svg) {
    color: var(--text-tertiary);
  }
  .c-text {
    min-width: 0;
    color: var(--text-secondary);
  }
  .c-title {
    margin-right: var(--space-2);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  .c-msg .num {
    color: var(--text-primary);
  }
  .c-acts {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: var(--space-2);
  }
  /* Several actions: they go under the text instead of squeezing it, the driver's report under them. */
  .callout.stack {
    grid-template-columns: auto minmax(0, 1fr);
    align-items: start;
    padding-block: var(--space-3);
  }
  .callout.stack > :global(svg) {
    margin-top: 2px;
  }
  .stack .c-acts,
  .c-detail {
    grid-column: 2;
    justify-content: flex-start;
  }
  .c-detail {
    min-width: 0;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .c-detail summary {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    width: max-content;
    cursor: pointer;
    list-style: none;
  }
  .c-detail summary::-webkit-details-marker {
    display: none;
  }
  /* The system disclosure chevron (as in ErrorState), not the native triangle. */
  .c-detail summary::before {
    content: '';
    width: 5px;
    height: 5px;
    margin: 0 2px;
    border-right: 1.5px solid currentColor;
    border-bottom: 1.5px solid currentColor;
    transform: rotate(-45deg);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .c-detail[open] summary::before {
    transform: rotate(45deg);
  }
  .c-detail summary:hover {
    color: var(--text-secondary);
  }
  .c-detail code {
    display: block;
    margin-top: var(--space-1);
    overflow-wrap: anywhere;
    color: var(--text-secondary);
  }
  @media (pointer: coarse) {
    .c-detail summary {
      min-height: 44px;
    }
  }
  @media (max-width: 767px) {
    /* Phones: the mark and the title share the first line; a cause wraps below it, the actions under the text. */
    .callout {
      padding-left: var(--space-3);
    }
    .callout:has(.c-msg:not(.brief)) {
      grid-template-columns: auto minmax(0, 1fr);
      align-items: start;
      padding-block: var(--space-3);
    }
    .callout:has(.c-msg:not(.brief)) > :global(svg) {
      margin-top: 2px;
    }
    .callout:has(.c-msg:not(.brief)) .c-acts {
      grid-column: 2;
      justify-content: flex-start;
    }
    .c-msg {
      display: block;
      margin-top: 2px;
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
    }
    /* Success is a one-line summary with its action; the best epoch is in the tiles below. */
    .c-msg.brief {
      display: none;
    }
  }
</style>
