<script lang="ts">
  /** A stacked bar of class shares in class colours, with 1 px surface gaps between segments. */
  import { classInk } from '$lib/design/classInk.svelte';

  let { counts, classes, height = 6 }: { counts: number[]; classes: readonly string[]; height?: number } = $props();
  const total = $derived(counts.reduce((a, b) => a + b, 0));
  const label = $derived(
    total ? classes.map((c, i) => `${c} ${(((counts[i] ?? 0) / total) * 100).toFixed(0)}%`).join(', ') : 'no boxes',
  );
</script>

<div class="strip" style:height="{height}px" role="img" aria-label="Class balance: {label}">
  {#each counts as c, i (i)}
    {#if c > 0}<span style:flex-grow={c} style:background={classInk(i)} title="{classes[i]}: {c.toLocaleString('en-US')}"></span>{/if}
  {/each}
</div>

<style>
  .strip {
    display: flex;
    gap: 1px;
    width: 100%;
    border-radius: 999px;
    overflow: hidden;
    background: var(--bg-active);
  }
  span {
    flex-basis: 0;
    min-width: 2px;
  }
</style>
