<script lang="ts">
  /** A retry of a run that ran out of GPU memory: what changed since, and whether the forecast says it now fits. */
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import type { Run } from '$lib/api/types';
  import { bytes } from '$lib/format';
  import { prefs } from '$lib/prefs.svelte';
  import { href } from '$lib/router.svelte';
  import { roughBytes } from './summary';

  let {
    oomRun,
    oomChanges,
    vramFit,
    vram,
    gpuMem,
    vramLevel,
  }: {
    oomRun: Run;
    oomChanges: { key: 'batch' | 'imgsz' | 'model' | 'device'; was: string; now: string }[];
    vramFit: 'fits' | 'no' | 'unknown' | null;
    vram: number;
    gpuMem: number | null;
    vramLevel: 'danger' | 'warning' | null;
  } = $props();

  /** A sentence that starts with a change: "Batch 16, 480 px" (an architecture or a device keeps its own case). */
  const capital = (t: string) => t.replace(/^batch/, 'Batch');
</script>

<!-- Why the form is not a plain copy: the run it retries ran out of memory, and what was changed for that. -->
<p class="oom-note" role="status">
  <TriangleAlert size={16} aria-hidden="true" />
  <span>
    <a class="link" href={href(`/runs/${oomRun.id}`)}>{oomRun.name}</a> ran out of GPU memory at
    {#if oomChanges.length}{oomChanges.map((x) => x.was).join(', ')}.
      <!-- "Now …" claims a fix only when the forecast above says it fits; otherwise it says the change is not
           enough, and the rail's one primary ("Use batch 8") is the change that is. A GPU that does not report
           its memory gets the figure and says the fit cannot be checked, never a fix it cannot vouch for. -->
      {#if vramFit === 'no'}{capital(oomChanges.map((x) => x.now).join(', '))} still won't fit: {roughBytes(vram)} of {bytes(gpuMem, prefs.units)}.{:else if vramFit === 'fits'}Now <span class="nowrap">{oomChanges.map((x) => x.now).join(', ')} · {roughBytes(vram)}</span>{vramLevel === 'warning' ? `, close to this GPU's ${bytes(gpuMem, prefs.units)}` : ''}.{:else if vramFit === 'unknown'}Now <span class="nowrap">{oomChanges.map((x) => x.now).join(', ')} · {roughBytes(vram)}</span> · this GPU doesn't report its memory, so the fit can't be checked.{:else}Now {oomChanges.map((x) => x.now).join(', ')}.{/if}{:else}batch {oomRun.config.batch}, {oomRun.config.imgsz} px. These are the same settings: lower the batch or image size, or pick a smaller model.{/if}
  </span>
</p>

<style>
  /* The retry note spans the form and the summary; the mark carries the hue, the text stays in ink. */
  .oom-note {
    grid-column: 1 / -1;
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    color: var(--text-secondary);
  }
  .oom-note > :global(svg) {
    flex-shrink: 0;
    margin-top: 2px;
    color: var(--warning-text);
  }
  .oom-note .link {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .oom-note .link:hover {
    text-decoration-color: var(--text-secondary);
  }
</style>
