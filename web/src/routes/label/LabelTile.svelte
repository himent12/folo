<script lang="ts">
  /**
   * One image of a labeling job in the grid, in each of its states: waiting (the image alone), with the agent (a live
   * mark), labeled (its boxes, or "No objects" at rest, so a background image never reads as a tile that failed to
   * load), failed (dimmed, with the word). An image corrected by hand says "Edited".
   */
  import CircleAlert from '@lucide/svelte/icons/circle-alert';
  import Pencil from '@lucide/svelte/icons/pencil';
  import type { LabelImage } from '$lib/api/types';
  import DatasetImage from '$lib/detect/DatasetImage.svelte';
  import type { Density } from '$lib/ui/TileGrid.svelte';

  let { img, jobId, classes, boxes, density }: { img: LabelImage; jobId: string; classes: readonly string[]; boxes: boolean; density: Density } = $props();

  const labeled = $derived(img.status === 'labeled');
</script>

<span class="tile" class:failed={img.status === 'failed'}>
  <DatasetImage {jobId} imageId={img.id} width={img.width} height={img.height} thumb={256} boxes={boxes && labeled ? img.boxes : []} {classes} compact fit="contain" alt="" />
  <span class="marks">
    {#if img.status === 'labeling'}
      <span class="mark live"><span class="dot" aria-hidden="true"></span>Labeling</span>
    {:else if img.status === 'failed'}
      <span class="mark"><CircleAlert size={12} strokeWidth={2} class="bad" aria-hidden="true" />Failed</span>
    {:else if labeled && img.boxes.length === 0}
      <span class="mark">No objects</span>
    {/if}
    {#if img.source === 'manual' && img.status !== 'labeling'}
      <span class="mark"><Pencil size={12} strokeWidth={2} aria-hidden="true" />Edited</span>
    {/if}
  </span>
  {#if density !== 's'}<span class="file"><span class="mono truncate">{img.file}</span></span>{/if}
</span>

<style>
  .tile {
    position: absolute;
    inset: 0;
  }
  /* A failed image steps back, and says so in its own frame. */
  .failed :global(.dsimg) {
    opacity: 0.55;
  }
  .marks,
  .file {
    position: absolute;
    left: 4px;
    right: 4px;
    display: flex;
    gap: var(--space-1);
    pointer-events: none;
  }
  .marks {
    top: 4px;
    flex-wrap: wrap;
  }
  .file {
    bottom: 4px;
    opacity: 0;
    transition: opacity var(--dur-fast) var(--ease-standard);
  }
  :global(.cell:hover) .file,
  :global(.cell:focus-visible) .file {
    opacity: 1;
  }
  .mark,
  .file > span {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    height: 18px;
    padding: 0 4px;
    border-radius: var(--radius-xs);
    background: var(--bg-scrim);
    color: var(--text-primary);
    font-size: var(--type-micro-size); /* small: image chip (a thumbnail's state) */
    font-weight: var(--weight-medium);
    backdrop-filter: blur(4px);
    white-space: nowrap;
  }
  .file > span {
    display: block;
    line-height: 18px;
    font-weight: var(--weight-regular);
  }
  :global([data-theme='light']) .mark,
  :global([data-theme='light']) .file > span {
    background: color-mix(in srgb, var(--bg-overlay) 88%, transparent);
  }
  .mark :global(.bad) {
    color: var(--danger-text);
  }
  /* The agent has the image now: the live dot, with the heartbeat that lasts exactly as long as the call. */
  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-solid);
    --pulse: var(--accent-line);
    animation: live-pulse 2.4s var(--ease-out) infinite;
  }
  :global([data-theme='light']) .dot {
    background: var(--accent-line);
  }
  @media (prefers-reduced-motion: reduce) {
    .dot {
      animation: none;
    }
  }
</style>
