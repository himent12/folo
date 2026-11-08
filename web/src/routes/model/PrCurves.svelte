<script lang="ts">
  /**
   * Precision-recall curves per class with a toggle list, plus the mean over all classes drawn emphasised. Phones get
   * the plotted curves as chips above the plot and the full class list in an expandable section below it.
   */
  import type { Evaluation } from '$lib/api/evaluation';
  import LineChart, { type Series } from '$lib/charts/LineChart.svelte';
  import { classInk } from '$lib/design/classInk.svelte';
  import { classColor } from '$lib/design/tokens';
  import { metric } from '$lib/format';
  import Input from '$lib/ui/Input.svelte';
  import Search from '@lucide/svelte/icons/search';
  import X from '@lucide/svelte/icons/x';
  import { matchesClass, trimPrTail } from './pr';

  let { classes, ev, map50 }: { classes: readonly string[]; ev: Evaluation; map50: number | null } = $props();

  // Curves are 2 px strokes: the darker light-theme ink. Legend swatches use the palette hue itself.
  const lineColor = classInk;
  // Phones: a shorter plot, chips above it and the class list folded away below.
  let narrow = $state(typeof matchMedia !== 'undefined' && matchMedia('(max-width: 767px)').matches);
  $effect(() => {
    const mq = matchMedia('(max-width: 767px)');
    const on = () => (narrow = mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  });

  // Few classes: a leaderboard by AP. Many: by frequency, matching the default selection.
  const order = $derived(
    ev.perClass.map((c) => c).sort((a, b) => (classes.length <= 10 ? b.ap50 - a.ap50 : b.instances - a.instances)),
  );

  // Phones: the mean plus the first class of the list. Few classes: all of them. Many (COCO): the eight most frequent.
  const defaults = $derived.by(() => {
    if (narrow) return new Set(order.slice(0, 1).map((c) => c.class));
    if (classes.length <= 10) return new Set(classes.map((_, i) => i));
    return new Set(order.slice(0, 8).map((c) => c.class));
  });
  let picked = $state<Set<number> | null>(null);
  const shown = $derived(picked ?? defaults);

  function toggle(i: number) {
    const next = new Set(shown);
    if (next.has(i)) next.delete(i);
    else next.add(i);
    picked = next;
  }

  // Curves end at their last recall point with precision > 0; the zero padding up to recall 1 is not data.
  const series = $derived<Series[]>([
    ...[...shown].sort((a, b) => a - b).map((i) => ({
      id: `c${i}`,
      label: classes[i] ?? `class ${i}`,
      color: lineColor(i),
      points: trimPrTail(ev.curves[i] ?? []),
    })),
    { id: 'mean', label: 'all classes', color: 'var(--text-primary)', points: trimPrTail(ev.mean), emphasis: true },
  ]);

  const meanAp = $derived(map50 ?? ev.perClass.reduce((a, c) => a + c.ap50, 0) / Math.max(1, ev.perClass.length));
  // Many classes: a filter to find one in the list.
  const many = $derived(classes.length > 10);
  let query = $state('');
  const listed = $derived(order.filter((c) => matchesClass(classes[c.class] ?? '', query)));
  const plotted = $derived(order.filter((c) => shown.has(c.class)));

  // A long list fades out at the bottom (so no row is cut through mid-glyph) until it is expanded or scrolled to its end.
  let expanded = $state(false);
  let atEnd = $state(false);
  function onListScroll(e: Event) {
    const el = e.currentTarget as HTMLElement;
    atEnd = el.scrollTop + el.clientHeight >= el.scrollHeight - 2;
  }
</script>

{#snippet side()}
  <div class="side" class:expanded>
    <div class="side-tools">
      <span class="count" aria-live="polite"><span class="num">{shown.size}</span> of <span class="num">{classes.length}</span> classes</span>
      <span class="side-actions" role="group" aria-label="Classes on the chart">
        <button type="button" onclick={() => (picked = new Set(classes.map((_, i) => i)))}>Plot all</button>
        <button type="button" onclick={() => (picked = new Set())}>None</button>
      </span>
    </div>
    {#if many}
      <div class="filter">
        <Input bind:value={query} icon={Search} size="sm" type="search" placeholder="Find a class" aria-label="Find a class" />
      </div>
    {/if}
    <div class="cols" aria-hidden="true">
      <span>Class</span>
      <span>AP50</span>
    </div>
    {#if !narrow}
      <div class="mean">
        <span class="mean-swatch" aria-hidden="true"></span>
        <span class="name">All classes</span>
        <span class="ap num">{metric(meanAp)}</span>
      </div>
    {/if}
    <ul class="list" class:fade={many && !expanded && !atEnd} aria-label="Classes shown on the chart, with AP50" onscroll={onListScroll}>
      {#each listed as c (c.class)}
        {@const on = shown.has(c.class)}
        <li>
          <button type="button" class:on aria-pressed={on} onclick={() => toggle(c.class)}>
            <span class="swatch" style:--c={classColor(c.class)} aria-hidden="true"></span>
            <span class="name truncate">{classes[c.class]}</span>
            <span class="ap num">{metric(c.ap50)}</span>
          </button>
        </li>
      {:else}
        <li class="none">No class matches “{query.trim()}”</li>
      {/each}
    </ul>
    {#if many && listed.length > 10}
      <button type="button" class="expand" aria-expanded={expanded} onclick={() => (expanded = !expanded)}>
        {expanded ? 'Show fewer' : `Show all ${listed.length}`}
      </button>
    {/if}
  </div>
{/snippet}

<div class="wrap">
  {#if narrow}
    <ul class="chips" aria-label="Curves on the chart">
      <li class="chip mean-chip">
        <span class="mean-swatch" aria-hidden="true"></span>
        All classes <span class="num">{metric(meanAp)}</span>
      </li>
      {#each plotted as c (c.class)}
        <li>
          <button type="button" class="chip" aria-label="Remove {classes[c.class]} from the chart" onclick={() => toggle(c.class)}>
            <span class="swatch on-sw" style:--c={classColor(c.class)} aria-hidden="true"></span>
            {classes[c.class]} <span class="num">{metric(c.ap50)}</span>
            <X size={12} strokeWidth={2} aria-hidden="true" />
          </button>
        </li>
      {/each}
    </ul>
  {/if}
  <div class="pr">
    <div class="plot">
      <span class="axis-y" aria-hidden="true">Precision</span>
      <LineChart
        {series}
        height={narrow ? 240 : 330}
        xDomain={[0, 1]}
        yDomain={[0, 1]}
        xFormat={(x) => x.toFixed(1)}
        yFormat={(y) => y.toFixed(1)}
        valueFormat={(y) => y.toFixed(3)}
        xTitle={(x) => `recall ${x.toFixed(2)}`}
        xTicks={6}
        yTicks={5}
        legend={false}
        label="Precision-recall curves at IoU 0.5 for {shown.size} classes and the mean over all classes"
      />
      <span class="axis-x" aria-hidden="true">Recall</span>
    </div>

    {#if narrow}
      <details class="picker">
        <summary>Choose classes <span class="faint num">{shown.size} of {classes.length} plotted</span></summary>
        {@render side()}
      </details>
    {:else}
      {@render side()}
    {/if}
  </div>
  {#if order.length > 1}
    {@const top = order.reduce((a, c) => (c.ap50 > a.ap50 ? c : a))}
    {@const low = order.reduce((a, c) => (c.ap50 < a.ap50 ? c : a))}
    <p class="summary">
      <span>Best <b>{classes[top.class]}</b> <span class="num">{metric(top.ap50)}</span></span>
      <span>Weakest <b>{classes[low.class]}</b> <span class="num">{metric(low.ap50)}</span></span>
      <span>Spread <span class="num">{metric(top.ap50 - low.ap50)}</span></span>
    </p>
  {/if}
</div>

<style>
  .wrap {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .summary {
    margin-top: auto;
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-6);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .summary b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .summary .num {
    color: var(--text-secondary);
  }
  .pr {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 196px;
    gap: var(--space-4);
    align-items: start;
  }
  .plot {
    position: relative;
    padding-left: 14px;
    min-width: 0;
  }
  .axis-y {
    position: absolute;
    left: -2px;
    top: 148px;
    transform: rotate(-90deg);
    transform-origin: center;
    font-size: var(--type-micro-size);
    letter-spacing: var(--type-micro-tracking);
    text-transform: uppercase;
    color: var(--text-tertiary);
    width: 14px;
    white-space: nowrap;
    display: flex;
    justify-content: center;
  }
  .axis-x {
    display: block;
    text-align: center;
    margin-top: 2px;
    font-size: var(--type-micro-size);
    letter-spacing: var(--type-micro-tracking);
    text-transform: uppercase;
    color: var(--text-tertiary);
  }
  .side {
    display: flex;
    flex-direction: column;
    min-width: 0;
    max-height: 350px;
  }
  .side.expanded {
    max-height: none;
  }
  .side-tools {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    min-height: 24px;
    padding-left: var(--space-2);
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .count {
    white-space: nowrap;
  }
  .count .num {
    color: var(--text-secondary);
  }
  .side-actions {
    display: flex;
    gap: 2px;
  }
  .filter {
    padding: var(--space-2) 0 var(--space-1);
  }
  .cols {
    display: flex;
    justify-content: space-between;
    padding: var(--space-2) var(--space-2) var(--space-1);
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .none {
    padding: var(--space-2);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .side-actions button,
  .expand {
    height: 20px;
    padding: 0 6px;
    border-radius: var(--radius-sm);
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .side-actions button:hover,
  .expand:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }
  .expand {
    height: 28px;
    margin-top: var(--space-1);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
  }
  .mean,
  .list button {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 100%;
    height: 28px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
  }
  .mean {
    border-bottom: 1px solid var(--line-subtle);
    border-radius: 0;
    margin-bottom: 2px;
  }
  .mean .name {
    font-weight: var(--weight-medium);
  }
  .list {
    overflow-y: auto;
    min-height: 0;
  }
  .list.fade {
    mask-image: linear-gradient(to bottom, #000 calc(100% - 24px), transparent);
  }
  .list button {
    color: var(--text-tertiary);
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .list button:hover {
    background: var(--bg-hover);
  }
  .list button.on {
    color: var(--text-primary);
  }
  .list button:focus-visible {
    outline-offset: -2px;
  }
  /* Legend swatches carry the class's palette hue (edged on light surfaces); a class off the chart is the same swatch
     at 40 %. */
  .swatch {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
    background: var(--c);
    opacity: 0.4;
  }
  .on .swatch,
  .on-sw {
    opacity: 1;
  }
  :global([data-theme='light']) .swatch {
    box-shadow: inset 0 0 0 1px #0a0e1433;
  }
  .mean-swatch {
    width: 12px;
    height: 3px;
    border-radius: 2px;
    flex-shrink: 0;
    background: var(--text-primary);
  }
  .name {
    flex: 1;
    min-width: 0;
    text-align: left;
  }
  .ap {
    font-size: var(--type-xs-size);
    color: var(--text-secondary);
  }
  .on .ap {
    color: var(--text-primary);
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1-5);
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1-5);
    height: var(--control-md);
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    border: 1px solid var(--line-default);
    font-size: var(--type-sm-size);
    color: var(--text-primary);
    white-space: nowrap;
  }
  .chip .num {
    color: var(--text-secondary);
  }
  .chip :global(svg) {
    color: var(--text-tertiary);
  }
  .mean-chip {
    font-weight: var(--weight-medium);
    border-color: transparent;
    background: var(--bg-active);
  }
  .picker {
    border-top: 1px solid var(--line-subtle);
  }
  .picker summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-height: 44px;
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    cursor: pointer;
  }
  .picker summary .faint {
    font-weight: var(--weight-regular);
    font-size: var(--type-xs-size);
  }
  @media (pointer: coarse) {
    .list button,
    .chip,
    .expand,
    .side-actions button {
      height: 44px;
    }
    .side-actions button {
      padding: 0 var(--space-3);
    }
  }
  @media (max-width: 767px) {
    .pr {
      grid-template-columns: minmax(0, 1fr);
    }
    .side {
      max-height: 360px;
    }
  }
</style>
