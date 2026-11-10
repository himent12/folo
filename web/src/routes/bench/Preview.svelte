<script lang="ts">
  /**
   * What a condition does to an image of this benchmark: the image unchanged and under the condition, split by a
   * divider the person drags (or moves with the arrow keys), with the true boxes on both sides (a geometric condition
   * moves them: `distance`'s boxes shrink with the scene). Picks the condition, its severity (1–5) and the image. The
   * changed image is the one the server serves contestants (the mock draws an approximation).
   */
  import ChevronLeft from '@lucide/svelte/icons/chevron-left';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import Scan from '@lucide/svelte/icons/scan';
  import { hub } from '$lib/api/index.svelte';
  import type { Bench, BenchCatalog, BenchImage } from '$lib/api/types';
  import { GROUP_NAME, SEVERITIES } from '$lib/bench/rules';
  import DetectionView from '$lib/detect/DetectionView.svelte';
  import { num } from '$lib/format';
  import Button from '$lib/ui/Button.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import ToggleButton from '$lib/ui/ToggleButton.svelte';

  let {
    bench,
    catalog,
    condition = $bindable(),
    severity = $bindable(3),
  }: {
    bench: Bench;
    catalog: BenchCatalog;
    condition: string;
    severity?: number;
  } = $props();

  const uid = $props.id();
  /** The images people step through: the benchmark's first ones (a seeded shuffle, so a fair look). */
  const PAGE = 24;
  let index = $state(0);
  let images = $state.raw<BenchImage[]>([]);
  let total = $state(0);
  let changedBoxes = $state.raw<BenchImage['boxes']>([]);
  let plainUrl = $state<string | null>(null);
  let changedUrl = $state<string | null>(null);
  let error = $state.raw<unknown>(undefined);
  let boxes = $state(true);
  /** Where the divider is, 0–100 from the left: unchanged on its left, the condition on its right. */
  let split = $state(50);

  // The first images, once per benchmark.
  $effect(() => {
    const id = bench.id;
    if (hub.mode === 'connecting') return;
    void hub.epoch;
    let gone = false;
    error = undefined;
    hub.api.benchImages(id, { limit: PAGE }).then(
      (p) => {
        if (gone) return;
        images = p.images;
        total = p.total;
        index = Math.min(index, Math.max(0, p.images.length - 1));
      },
      (e) => !gone && (error = e),
    );
    return () => (gone = true);
  });

  const image = $derived(images[index]);

  // The image unchanged, and under the condition with its truth (a geometric condition moves the boxes).
  $effect(() => {
    const im = image;
    if (!im) return;
    let gone = false;
    plainUrl = null;
    void hub.api.benchImageUrl(bench.id, im.id).then((u) => !gone && (plainUrl = u), (e) => !gone && (error = e));
    return () => (gone = true);
  });
  $effect(() => {
    const im = image;
    const c = condition;
    const s = severity;
    if (!im) return;
    let gone = false;
    changedUrl = null;
    void hub.api.benchImageUrl(bench.id, im.id, { condition: c, severity: s }).then((u) => !gone && (changedUrl = u), (e) => !gone && (error = e));
    // The truth under it: only a geometric condition moves it; the others keep the image's own.
    if (c === 'distance') {
      void hub.api.benchImages(bench.id, { offset: index, limit: 1, condition: c, severity: s }).then((p) => !gone && (changedBoxes = p.images[0]?.boxes ?? im.boxes), () => {});
    } else changedBoxes = im.boxes;
    return () => (gone = true);
  });

  const options = $derived(catalog.conditions.map((c) => ({ value: c.id, label: c.name, group: GROUP_NAME[c.group] })));
  const name = $derived(catalog.conditions.find((c) => c.id === condition)?.name ?? condition);
  const desc = $derived(catalog.conditions.find((c) => c.id === condition)?.description ?? '');
  const ratio = $derived(image ? `${image.width} / ${image.height}` : '4 / 3');
</script>

<Panel title="Preview" meta={image ? `image ${num(index + 1)} of ${num(total)}` : undefined} pad="none">
  {#snippet actions()}
    <ToggleButton pressed={boxes} label="Boxes" icon={Scan} onchange={(v) => (boxes = v)} />
  {/snippet}
  <div class="pv">
    <div class="controls">
      <div class="cond">
        <label class="sr-only" for="{uid}-cond">Condition</label>
        <Select id="{uid}-cond" bind:value={condition} {options} label="Condition to preview" />
      </div>
      <Segmented label="Severity" size="sm" bind:value={() => String(severity), (v) => (severity = Number(v))} options={SEVERITIES.map((s) => ({ value: String(s), label: String(s) }))} />
    </div>
    {#if error && !image}
      <ErrorState {error} title="Could not load the images" compact />
    {:else}
      <div class="stage" style:aspect-ratio={ratio} style:--ar={image ? image.width / image.height : 4 / 3}>
        {#if image && plainUrl && changedUrl}
          <div class="side plain" style:clip-path="inset(0 {100 - split}% 0 0)">
            <DetectionView src={plainUrl} width={image.width} height={image.height} truth={boxes ? image.boxes : []} classes={bench.classes} labels={false} alt="{image.file}, unchanged" />
          </div>
          <div class="side changed" style:clip-path="inset(0 0 0 {split}%)">
            <DetectionView src={changedUrl} width={image.width} height={image.height} truth={boxes ? changedBoxes : []} classes={bench.classes} labels={false} alt="{image.file} under {name}, severity {severity}" />
          </div>
          <span class="divider" style:left="{split}%" aria-hidden="true"><span class="knob"></span></span>
          <span class="tag l" aria-hidden="true">Unchanged</span>
          <span class="tag r" aria-hidden="true">{name} {severity}</span>
          <input
            class="drag"
            type="range"
            min="0"
            max="100"
            step="1"
            bind:value={split}
            aria-label="Divider between the unchanged image and {name}, severity {severity}"
            aria-valuetext="{split}% unchanged"
          />
        {:else}
          <Skeleton h="100%" r="0" image />
        {/if}
      </div>
    {/if}
    <div class="foot">
      <p class="what">{desc}</p>
      <div class="step">
        <Button variant="ghost" size="sm" icon={ChevronLeft} aria-label="Previous image" disabled={index === 0} onclick={() => (index = Math.max(0, index - 1))} />
        <span class="file truncate" title={image?.file}>{image?.file ?? ''}</span>
        <Button variant="ghost" size="sm" icon={ChevronRight} aria-label="Next image" disabled={index >= images.length - 1} onclick={() => (index = Math.min(images.length - 1, index + 1))} />
      </div>
    </div>
  </div>
</Panel>

<style>
  .pv {
    display: grid;
    gap: var(--space-3);
    padding: 0 var(--space-4) var(--space-4);
  }
  /* The condition takes what the severities leave; where that is too little for its name (a phone's 16 px touch
     text), the severities wrap under it. */
  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
  .cond {
    flex: 1 1 176px;
    min-width: 0;
  }
  .stage {
    position: relative;
    width: 100%;
    border-radius: var(--radius-md);
    overflow: hidden;
    background: var(--bg-inset);
  }
  /* Below 1024 px the preview spans the column up to half the viewport's height, so the plan starts in the first
     screen (as the Playground's canvas does on a tablet): a tall or square image takes the width that height allows,
     centred, never letterboxed in a frame wider than it. */
  @media (max-width: 1023px) {
    .stage {
      width: min(100%, calc(50dvh * var(--ar)));
      margin-inline: auto;
    }
  }
  .side {
    position: absolute;
    inset: 0;
    display: grid;
  }
  .side :global(.view) {
    width: 100%;
    height: 100%;
  }
  /* The divider: a hairline in the image's own light, with a grip at its centre. */
  .divider {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 2px;
    margin-left: -1px;
    background: #ffffffd9;
    box-shadow: 0 0 0 1px #00000040;
    pointer-events: none;
  }
  .knob {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 16px;
    height: 24px;
    margin: -12px 0 0 -8px;
    border-radius: 8px;
    background: #ffffffe6;
    box-shadow: 0 1px 4px #00000059;
  }
  .tag {
    position: absolute;
    top: var(--space-2);
    padding: 2px 6px;
    border-radius: var(--radius-sm);
    background: #000000a6;
    color: #ffffff;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    pointer-events: none;
  }
  .tag.l {
    left: var(--space-2);
  }
  .tag.r {
    right: var(--space-2);
  }
  /* The range input is the divider's control: invisible over the image, full height, so a drag anywhere moves it. */
  .drag {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    margin: 0;
    opacity: 0;
    cursor: ew-resize;
  }
  .stage:has(.drag:focus-visible) {
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: var(--size-focus-offset);
  }
  .foot {
    display: grid;
    gap: var(--space-2);
  }
  .what {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    min-height: 32px;
  }
  .step {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-2);
  }
  .file {
    text-align: center;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
</style>
