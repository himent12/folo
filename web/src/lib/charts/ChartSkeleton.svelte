<script lang="ts">
  /**
   * A line chart while its data loads: the real frame (legend row, gridlines, baseline axis, tick gutters) with only
   * the unknown parts as placeholders, so nothing moves when the chart arrives.
   */
  import Skeleton from '$lib/ui/Skeleton.svelte';

  let {
    height = 200,
    legend = true,
    legendRows = 1,
  }: {
    height?: number;
    legend?: boolean;
    /** Rows the loaded legend takes (it wraps on narrow cards). */
    legendRows?: number;
  } = $props();
</script>

<div class="cs" aria-hidden="true">
  {#if legend}
    <div class="legend" style:height="{legendRows * 16 + (legendRows - 1) * 4}px"><Skeleton w="48px" h="12px" /><Skeleton w="96px" h="12px" /><Skeleton w="80px" h="12px" /></div>
  {/if}
  <div class="plot" style:height="{height}px">
    <div class="ticks">{#each [0, 1, 2, 3, 4] as i (i)}<Skeleton w="20px" h="10px" />{/each}</div>
    <div class="grid">{#each [0, 1, 2, 3] as i (i)}<span></span>{/each}<span class="axis"></span></div>
    <div class="xt">{#each [0, 1, 2, 3, 4, 5] as i (i)}<Skeleton w="16px" h="10px" />{/each}</div>
  </div>
</div>

<style>
  .cs {
    display: grid;
    gap: var(--space-2);
    min-width: 0;
  }
  .legend {
    display: flex;
    gap: var(--space-4);
    height: 16px;
    align-items: center;
  }
  .plot {
    display: grid;
    grid-template-columns: 32px minmax(0, 1fr);
    grid-template-rows: minmax(0, 1fr) 20px;
    column-gap: var(--space-2);
  }
  .ticks {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: flex-end;
    padding: 4px 0 0;
  }
  .grid {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding-top: 8px;
  }
  .grid span {
    height: 1px;
    background: var(--chart-grid);
  }
  .grid .axis {
    background: var(--chart-axis);
  }
  .xt {
    grid-column: 2;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }
</style>
