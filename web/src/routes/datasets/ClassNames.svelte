<script lang="ts">
  /** Class names on one line, cut by whole names with "+n more" for the rest (the full list is in the title). */
  import { fitCount } from './fit';

  let { names }: { names: readonly string[] } = $props();

  const SEP = ', ';
  let el = $state<HTMLSpanElement>();
  let width = $state(0);
  // Bumped when web fonts finish loading, so widths are measured in Geist rather than the fallback.
  let fontsReady = $state(0);
  $effect(() => {
    document.fonts?.ready.then(() => fontsReady++);
  });

  let ctx: CanvasRenderingContext2D | null = null;
  const shown = $derived.by(() => {
    void fontsReady;
    if (!el || width <= 0) return names.length;
    ctx ??= document.createElement('canvas').getContext('2d');
    if (!ctx) return names.length;
    const cs = getComputedStyle(el);
    ctx.font = `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
    const w = (s: string) => ctx!.measureText(s).width;
    // A px of slack for sub-pixel rounding between canvas and layout.
    return fitCount(names.map(w), w(SEP), (rest) => w(`+${rest} more`) + 6, width - 1);
  });
  const rest = $derived(names.length - shown);
</script>

<span class="names" bind:this={el} bind:clientWidth={width} title={rest ? names.join(SEP) : undefined}>
  <span class="list">{names.slice(0, shown).join(SEP)}</span>
  {#if rest > 0}<span class="more">+{rest} more</span>{/if}
</span>

<style>
  .names {
    display: flex;
    align-items: baseline;
    gap: 6px;
    min-width: 0;
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .list {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .more {
    flex-shrink: 0;
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
  }
</style>
