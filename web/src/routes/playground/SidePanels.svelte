<script lang="ts">
  /**
   * The rail beside the image: Thresholds (with what is drawn on the image), Classes and Timing. One column it
   * interleaves with the image's panels; on tablets it is a two-column block under the image, and phones with an
   * image fold it away (their sheet and disclosures hold the same).
   */
  import type { Snippet } from 'svelte';
  import type { Prediction } from '$lib/api/types';
  import Panel from '$lib/ui/Panel.svelte';
  import Slider from '$lib/ui/Slider.svelte';
  import type { Source } from './source';

  let {
    source,
    result,
    counts,
    pred,
    where,
    conf = $bindable(),
    resetButton,
    secondarySliders,
    displayToggles,
    showAllButton,
    classList,
    timingBody,
  }: {
    source: Source | null;
    result: { boxes: Prediction['boxes']; ms: number } | null;
    counts: [number, number][];
    pred: Prediction | null;
    where: string;
    conf: number;
    resetButton: Snippet;
    secondarySliders: Snippet<[boolean]>;
    displayToggles: Snippet;
    showAllButton: Snippet;
    classList: Snippet;
    timingBody: Snippet;
  } = $props();
</script>

<!-- Phones with an image: Thresholds open as a sheet from the image's footer, the rest folds under the detections. -->
<div class="aside" class:tucked={!!source}>
  <div class="thresholds">
    <Panel title="Thresholds" divided>
      {#snippet actions()}{@render resetButton()}{/snippet}
      <div class="sliders">
        <Slider label="Confidence" bind:value={conf} min={0.01} max={0.95} step={0.01} hint="Hide boxes scored below this." />
        {@render secondarySliders(false)}
      </div>
      <!-- What is drawn on the image: here rather than on the image's toolbar, which is only as wide as the image
           (on phones the same toggles sit in the Thresholds sheet). -->
      {#if source}<div class="show" role="group" aria-label="Show on the image">{@render displayToggles()}</div>{/if}
    </Panel>
  </div>

  <div class="classes">
    <Panel title="Classes" meta={result ? `${counts.length} found` : undefined} divided>
      {#snippet actions()}{@render showAllButton()}{/snippet}
      {@render classList()}
    </Panel>
  </div>

  <div class="timing">
    <Panel title="Timing" meta={pred ? where : undefined} divided>
      {@render timingBody()}
    </Panel>
  </div>
</div>

<style>
  .aside {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: var(--space-4);
    min-width: 0;
  }
  .show {
    display: flex;
    flex-wrap: wrap;
    gap: 2px;
    margin-top: var(--space-4);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
  }
  .sliders {
    display: grid;
    gap: var(--space-5);
  }
  /* Threshold values are numbers, not code: sans with tabular numerals. */
  .sliders :global(output.val) {
    font-family: var(--font-sans);
    font-variant-numeric: tabular-nums;
  }
  @media (max-width: 1023px) {
    /* One column: interleave the panels so the controls sit right under the image. */
    .aside {
      display: contents;
    }
    .thresholds {
      order: 2;
    }
    .classes {
      order: 4;
    }
    .timing {
      order: 5;
    }
  }
  /* Tablets: the rail is a two-column block under the image, Thresholds beside Classes over Timing, then the
     detections full width. */
  @media (min-width: 768px) and (max-width: 1023px) {
    .aside {
      order: 2;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      grid-template-rows: auto 1fr;
      gap: var(--space-4);
    }
    .thresholds {
      grid-row: span 2;
    }
    .thresholds {
      display: grid;
    }
    /* Content-sized: Timing sits right under Classes and is as tall as what it holds (one line before a run), never
       stretched to Thresholds' height. */
    .timing {
      align-self: start;
    }
    .aside.tucked .thresholds {
      display: none;
    }
    .aside.tucked {
      grid-template-rows: auto;
    }
    /* Side by side, the two panels end together (DESIGN_DIRECTION 2). */
    .aside.tucked .classes,
    .aside.tucked .timing {
      display: grid;
      align-self: stretch;
    }
  }
  @media (max-width: 767px) {
    .aside.tucked {
      display: none;
    }
  }
</style>
