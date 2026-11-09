<script lang="ts">
  /**
   * A dataset's thumbnails in a virtualised grid (ui/TileGrid): image pages load as their rows come into view. The
   * cells take the dataset's dominant image shape.
   */
  import DatasetImage from '$lib/detect/DatasetImage.svelte';
  import TileGrid, { dominantAspect, type Density } from '$lib/ui/TileGrid.svelte';
  import type { ImagePager } from './pager.svelte';

  let {
    pager,
    classes,
    density,
    boxes,
    only,
    onopen,
  }: {
    pager: ImagePager;
    classes: readonly string[];
    density: Density;
    boxes: boolean;
    only: ReadonlySet<number> | null;
    onopen: (index: number, id: string) => void;
  } = $props();

  /** The cells take the shape of the dataset's first 64 images (4:3 until they load). */
  const aspect = $derived(dominantAspect(Array.from({ length: Math.min(pager.total ?? 0, 64) }, (_, i) => pager.get(i))));

  const stem = (file: string) => file.split('/').pop() ?? file;

  function label(i: number) {
    const img = pager.get(i);
    return img ? `${stem(img.file)}, ${img.split}, ${img.boxes.length} ${img.boxes.length === 1 ? 'box' : 'boxes'}` : 'Loading image';
  }

  function open(i: number) {
    const img = pager.get(i);
    if (img) onopen(i, img.id);
  }
</script>

<TileGrid total={pager.total ?? 0} {density} {aspect} {label} onopen={open} onrange={(lo, hi) => pager.ensure(lo, hi)} toolbar=".ds-toolbar">
  {#snippet tile(i)}
    {@const img = pager.get(i)}
    {#if img}
      <span class="tile" class:bare={img.boxes.length === 0}>
        <DatasetImage
          datasetId={pager.datasetId}
          imageId={img.id}
          width={img.width}
          height={img.height}
          thumb={256}
          boxes={boxes ? img.boxes : []}
          {classes}
          {only}
          compact
          fit="contain"
          alt=""
        />
        {#if density !== 's'}
          <span class="meta">
            <span class="file mono truncate">{stem(img.file)}</span>
            <span class="n num">{img.boxes.length}</span>
          </span>
        {/if}
      </span>
    {:else}
      <span class="ph" class:still={!!pager.error}></span>
    {/if}
  {/snippet}
</TileGrid>

<style>
  .tile {
    position: absolute;
    inset: 0;
  }
  .meta {
    position: absolute;
    left: 4px;
    right: 4px;
    bottom: 4px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-1);
    pointer-events: none;
    opacity: 0;
    transform: translateY(2px);
    transition:
      opacity var(--dur-fast) var(--ease-standard),
      transform var(--dur-fast) var(--ease-standard);
  }
  .meta > span {
    height: 18px;
    display: inline-flex;
    align-items: center;
    padding: 0 4px;
    border-radius: var(--radius-xs);
    background: var(--bg-scrim);
    color: var(--text-primary);
    font-size: var(--type-micro-size); /* small: image chip (the thumbnail's box count) */
    backdrop-filter: blur(4px);
  }
  :global([data-theme='light']) .meta > span {
    background: color-mix(in srgb, var(--bg-overlay) 88%, transparent);
    color: var(--text-primary);
  }
  :global(.cell:hover) .meta,
  :global(.cell:focus-visible) .meta {
    opacity: 1;
    transform: none;
  }
  /* An image with no boxes (a background example) keeps its "0" at rest, so a plain surface never reads as a tile that
     failed to load; its name still waits for hover or focus. */
  .tile.bare .meta {
    opacity: 1;
    transform: none;
  }
  :global(.cell:not(:hover, :focus-visible)) .tile.bare .meta .file {
    visibility: hidden;
  }
  .ph {
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent, var(--bg-hover), transparent);
    background-size: 200% 100%;
    /* Loading motion is bounded (DESIGN_DIRECTION §2.9): a few sweeps while the page of images is requested, none once
       a request has failed (the grid says so), so a cell that never fills never shimmers forever. */
    animation: shimmer 1.4s linear 6;
  }
  .ph.still {
    animation: none;
  }
  @keyframes shimmer {
    from {
      background-position: 100% 0;
    }
    to {
      background-position: -100% 0;
    }
  }
</style>
