<script lang="ts">
  /** Pick an image from a registered dataset: dataset + split, then a page of thumbnails. */
  import ChevronLeft from '@lucide/svelte/icons/chevron-left';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import { untrack } from 'svelte';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Dataset, DatasetImage as Img, Split } from '$lib/api/types';
  import DatasetImage from '$lib/detect/DatasetImage.svelte';
  import { num } from '$lib/format';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import Images from '@lucide/svelte/icons/images';

  let {
    open = $bindable(false),
    onpick,
    onclose,
    prefer = [],
  }: {
    open?: boolean;
    onpick: (d: Dataset, img: Img) => void;
    onclose?: () => void;
    /** Classes of the selected model: a dataset with the same classes is shown first. */
    prefer?: readonly string[];
  } = $props();

  const PAGE = 24;
  const isOpen = $derived(open);
  const datasets = new Resource((s) => (isOpen ? hub.api.datasets(s) : Promise.resolve(null)));

  let chosen = $state<string | null>(null);
  let split = $state<Split>('val');
  let offset = $state(0);

  const list = $derived(datasets.data ?? []);
  const preferKey = $derived(prefer.join('|'));
  const ds = $derived(list.find((d) => d.id === chosen) ?? list.find((d) => d.classes.join('|') === preferKey) ?? list[0] ?? null);
  const dsId = $derived(ds?.id ?? null);
  const splitN = $derived(ds ? ds.splits[split] : 0);

  // When the dataset changes, start at its first page and on a split it actually has (val, else train, else test).
  $effect(() => {
    const d = ds;
    if (!d) return;
    untrack(() => {
      offset = 0;
      if (d.splits[split] === 0) split = (['val', 'train', 'test'] as Split[]).find((x) => d.splits[x] > 0) ?? 'train';
    });
  });

  const images = new Resource((s) => {
    const id = dsId;
    const sp = split;
    const off = offset;
    if (!id || !isOpen) return Promise.resolve(null);
    return hub.api.datasetImages(id, { split: sp, offset: off, limit: PAGE }, s);
  });

  function pick(img: Img) {
    if (!ds) return;
    onpick(ds, img);
    open = false;
  }

  const splits = $derived(
    (['val', 'train', 'test'] as Split[])
      .filter((s) => !ds || ds.splits[s] > 0)
      .map((s) => ({ value: s, label: s, meta: ds ? num(ds.splits[s]) : undefined })),
  );
</script>

<Dialog bind:open title="Pick an image" description="Run the model on an image from a registered dataset." size="lg" {onclose}>
  {#if datasets.error && !datasets.data}
    <ErrorState error={datasets.error} retry={() => datasets.refresh()} compact />
  {:else if datasets.data && list.length === 0}
    <EmptyState compact icon={Images} title="No datasets registered" description="Register a dataset first, or drop an image file instead." />
  {:else}
    <div class="bar">
      <div class="ds-select">
        <Select
          label="Dataset"
          value={ds?.id ?? ''}
          options={list.map((d) => ({ value: d.id, label: d.name }))}
          disabled={!list.length}
          onchange={(v) => {
            chosen = v;
            offset = 0;
          }}
        />
      </div>
      <Segmented label="Split" size="sm" bind:value={split} options={splits} onchange={() => (offset = 0)} />
    </div>

    <!-- Errors show even with an older page retained, so a stale page of another dataset is never pickable. -->
    {#if images.error}
      <ErrorState error={images.error} retry={() => images.refresh()} compact />
    {:else}
      <div class="grid" aria-busy={images.loading}>
        {#if !images.data || images.loading}
          {#each { length: 12 } as _, i (i)}<Skeleton h="auto" r="var(--radius-sm)" class="thumb-sk" />{/each}
        {:else}
          {#each images.data.images as img (img.id)}
            <button type="button" class="thumb" onclick={() => pick(img)} aria-label="Use {img.file}" title={img.file}>
              <DatasetImage
                datasetId={ds?.id ?? ''}
                imageId={img.id}
                width={img.width}
                height={img.height}
                thumb={192}
                classes={ds?.classes ?? []}
                fit="cover"
                compact
              />
            </button>
          {/each}
        {/if}
      </div>
    {/if}
  {/if}

  {#snippet footer()}
    <span class="range num">
      {#if splitN}{num(offset + 1)}–{num(Math.min(splitN, offset + PAGE))} of {num(splitN)}{/if}
    </span>
    <Button size="sm" variant="ghost" icon={ChevronLeft} aria-label="Previous page" disabled={offset === 0} onclick={() => (offset = Math.max(0, offset - PAGE))} />
    <Button size="sm" variant="ghost" icon={ChevronRight} aria-label="Next page" disabled={offset + PAGE >= splitN} onclick={() => (offset += PAGE)} />
  {/snippet}
</Dialog>

<style>
  .bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    margin-bottom: var(--space-3);
    flex-wrap: wrap;
  }
  .ds-select {
    width: 220px;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: var(--space-2);
  }
  .grid :global(.thumb-sk) {
    aspect-ratio: 1;
  }
  .thumb {
    position: relative;
    aspect-ratio: 1;
    border-radius: var(--radius-sm);
    overflow: hidden;
    box-shadow: 0 0 0 1px var(--line-subtle);
    transition: box-shadow var(--dur-fast) var(--ease-standard);
  }
  .thumb:hover {
    box-shadow: 0 0 0 1px var(--text-tertiary);
  }
  .thumb:focus-visible {
    outline-offset: 2px;
  }
  .range {
    margin-right: auto;
    align-self: center;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  @media (max-width: 767px) {
    .grid {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .ds-select {
      width: 100%;
    }
  }
</style>
