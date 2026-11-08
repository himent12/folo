<script lang="ts">
  /**
   * Validation samples: the same 16 images every epoch, predictions (solid) over ground truth (dashed). Follows the
   * latest epoch while the run is live; pick any earlier epoch to see how detections improved.
   */
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Sample } from '$lib/api/types';
  import DatasetImage from '$lib/detect/DatasetImage.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import Slider from '$lib/ui/Slider.svelte';

  let { runId, classes, epochsDone }: { runId: string; classes: string[]; epochsDone: number } = $props();

  let pick = $state<string>('latest');
  let show = $state<'both' | 'pred' | 'truth'>('both');
  let conf = $state(0.25);
  let open = $state<Sample | null>(null);
  let dialogOpen = $state(false);

  // Memoise the props the fetcher reads: the parent passes values derived from a run object that changes every
  // step, and an unmemoised prop read would refetch on each one.
  const rid = $derived(runId);
  const done = $derived(epochsDone);
  const epoch = $derived(pick === 'latest' ? done - 1 : Number(pick));
  const samples = new Resource((s) => (done > 0 ? hub.api.runSamples(rid, epoch, s) : Promise.resolve(null)));

  const options = $derived([
    { value: 'latest', label: `Latest (epoch ${Math.max(1, epochsDone)})` },
    ...Array.from({ length: epochsDone }, (_, i) => epochsDone - 1 - i)
      .filter((e) => e === 0 || (e + 1) % 5 === 0)
      .map((e) => ({ value: String(e), label: `Epoch ${e + 1}` })),
  ]);

  const preds = (s: Sample) => (show === 'truth' ? [] : s.pred.filter((p) => p.score >= conf));
  const truth = (s: Sample) => (show === 'pred' ? [] : s.truth);
  const hits = (s: Sample) => s.pred.filter((p) => p.score >= conf).length;
</script>

<Panel title="Validation samples" meta={samples.data ? `epoch ${samples.data.epoch + 1}` : undefined} id="samples">
  {#snippet actions()}
    <div class="ctl">
      <Segmented
        size="sm"
        label="Show boxes"
        bind:value={show}
        options={[
          { value: 'both', label: 'Both' },
          { value: 'pred', label: 'Predicted' },
          { value: 'truth', label: 'Truth' },
        ]}
      />
      <Select size="sm" label="Epoch" bind:value={pick} {options} />
    </div>
  {/snippet}
  <div class="bar">
    <div class="legend" aria-hidden="true">
      <span><i class="solid"></i>prediction</span>
      <span><i class="dashed"></i>ground truth</span>
      <span class="count-key num">7/8 = predictions / true boxes</span>
    </div>
    <div class="conf"><Slider label="Min confidence" bind:value={conf} min={0.05} max={0.95} step={0.05} /></div>
  </div>
  {#if epochsDone === 0}
    <p class="none">Samples appear after the first validation pass.</p>
  {:else if samples.error && !samples.data}
    <ErrorState error={samples.error} retry={() => samples.refresh()} compact />
  {:else if !samples.data}
    <div class="grid">{#each Array(16) as _, i (i)}<Skeleton h="auto" class="cell-sk" r="var(--radius-md)" />{/each}</div>
  {:else}
    <div class="grid">
      {#each samples.data.samples as s (s.image_id)}
        <button
          class="cell"
          style:aspect-ratio="{s.width} / {s.height}"
          onclick={() => ((open = s), (dialogOpen = true))}
          aria-label="Open sample {s.image_id}: {hits(s)} predictions, {s.truth.length} ground-truth boxes"
        >
          <DatasetImage
            datasetId={s.dataset_id}
            imageId={s.image_id}
            width={s.width}
            height={s.height}
            thumb={384}
            boxes={preds(s)}
            truth={truth(s)}
            {classes}
            compact
          />
          <span class="count num">{hits(s)}<span class="faint">/{s.truth.length}</span></span>
        </button>
      {/each}
    </div>
  {/if}
</Panel>

<Dialog bind:open={dialogOpen} title={open ? `Sample ${open.image_id}` : 'Sample'} description={open ? `${hits(open)} predictions ≥ ${conf.toFixed(2)} · ${open.truth.length} ground-truth boxes` : undefined} size="lg">
  {#if open}
    <div class="big" style:aspect-ratio="{open.width} / {open.height}">
      <DatasetImage datasetId={open.dataset_id} imageId={open.image_id} width={open.width} height={open.height} boxes={preds(open)} truth={truth(open)} {classes} />
    </div>
  {/if}
</Dialog>

<style>
  .ctl {
    display: flex;
    gap: var(--space-2);
    align-items: center;
  }
  .bar {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: var(--space-4);
    margin-bottom: var(--space-3);
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-4);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .legend span {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }
  .legend i {
    width: 14px;
    height: 10px;
    border-radius: 1px;
  }
  .solid {
    border: 2px solid var(--text-secondary);
  }
  .dashed {
    border: 1.5px dashed var(--text-secondary);
  }
  .conf {
    width: 220px;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(8, minmax(0, 1fr));
    gap: var(--space-2);
  }
  .grid :global(.cell-sk) {
    aspect-ratio: 4 / 3;
  }
  .cell {
    position: relative;
    display: block;
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: 0 0 0 1px var(--line-subtle);
    transition: box-shadow var(--dur-fast) var(--ease-standard);
  }
  .cell:hover {
    box-shadow: 0 0 0 1px var(--line-strong);
  }
  .cell:focus-visible {
    outline-offset: 2px;
  }
  .count {
    position: absolute;
    right: 4px;
    bottom: 4px;
    padding: 1px 5px;
    border-radius: var(--radius-xs);
    background: #000000b3;
    color: #fff;
    font-size: var(--type-micro-size);
    line-height: 14px;
  }
  .count .faint {
    color: #ffffff99;
  }
  .none {
    padding: var(--space-6) 0;
    text-align: center;
    color: var(--text-tertiary);
  }
  .big {
    width: 100%;
    border-radius: var(--radius-md);
    overflow: hidden;
  }
  @media (max-width: 1199px) {
    .grid {
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }
  }
  @media (max-width: 767px) {
    .grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .bar {
      flex-direction: column;
      align-items: stretch;
    }
    .conf {
      width: 100%;
    }
    .ctl :global(.seg) {
      display: none;
    }
  }
</style>
