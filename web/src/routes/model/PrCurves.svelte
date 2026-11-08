<script lang="ts">
  /** Precision-recall curves per class with a toggle list, plus the mean over all classes drawn emphasised. */
  import type { Evaluation } from '$lib/api/evaluation';
  import LineChart, { type Series } from '$lib/charts/LineChart.svelte';
  import { classInk } from '$lib/design/classInk.svelte';
  import { metric } from '$lib/format';
  import Input from '$lib/ui/Input.svelte';
  import Search from '@lucide/svelte/icons/search';
  import { matchesClass, trimPrTail } from './pr';

  let { classes, ev, map50 }: { classes: readonly string[]; ev: Evaluation; map50: number | null } = $props();

  const lineColor = classInk;
  // Phones: a shorter plot so the class list is closer.
  let narrow = $state(typeof matchMedia !== 'undefined' && matchMedia('(max-width: 767px)').matches);
  $effect(() => {
    const mq = matchMedia('(max-width: 767px)');
    const on = () => (narrow = mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  });

  // Few classes: show them all. Many (COCO): start with the eight most frequent.
  const defaults = $derived.by(() => {
    if (classes.length <= 10) return new Set(classes.map((_, i) => i));
    const byCount = ev.perClass.map((c) => c).sort((a, b) => b.instances - a.instances);
    return new Set(byCount.slice(0, 8).map((c) => c.class));
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
  // Few classes: a leaderboard by AP. Many: by frequency, matching the default selection.
  const order = $derived(
    ev.perClass.map((c) => c).sort((a, b) => (classes.length <= 10 ? b.ap50 - a.ap50 : b.instances - a.instances)),
  );
  // Many classes: a filter to find one in the list.
  const many = $derived(classes.length > 10);
  let query = $state('');
  const listed = $derived(order.filter((c) => matchesClass(classes[c.class] ?? '', query)));
</script>

<div class="wrap">
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

    <div class="side">
      <div class="side-tools">
        <span class="count" aria-live="polite"><span class="num">{shown.size}</span> of <span class="num">{classes.length}</span> classes</span>
        <span class="side-actions" role="group" aria-label="Show classes">
          <button type="button" onclick={() => (picked = new Set(classes.map((_, i) => i)))}>Show all</button>
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
      <div class="mean">
        <span class="swatch mean-swatch" aria-hidden="true"></span>
        <span class="name">all classes</span>
        <span class="ap num">{metric(meanAp)}</span>
      </div>
      <ul class="list" aria-label="Classes shown on the chart, with AP50">
        {#each listed as c (c.class)}
          {@const on = shown.has(c.class)}
          <li>
            <button type="button" class:on aria-pressed={on} onclick={() => toggle(c.class)}>
              <span class="swatch" style:--c={lineColor(c.class)} aria-hidden="true"></span>
              <span class="name truncate">{classes[c.class]}</span>
              <span class="ap num">{metric(c.ap50)}</span>
            </button>
          </li>
        {:else}
          <li class="none">No class matches “{query.trim()}”</li>
        {/each}
      </ul>
    </div>
  </div>
  {#if order.length > 1}
    {@const top = order.reduce((a, c) => (c.ap50 > a.ap50 ? c : a))}
    {@const low = order.reduce((a, c) => (c.ap50 < a.ap50 ? c : a))}
    <p class="summary">
      <span>Best <b>{classes[top.class]}</b> <span class="mono">{metric(top.ap50)}</span></span>
      <span>Weakest <b>{classes[low.class]}</b> <span class="mono">{metric(low.ap50)}</span></span>
      <span>Spread <span class="mono">{metric(top.ap50 - low.ap50)}</span></span>
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
  .summary .mono {
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
  .side-actions button {
    height: 20px;
    padding: 0 6px;
    border-radius: var(--radius-sm);
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .side-actions button:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
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
  .swatch {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
    box-shadow: inset 0 0 0 1.5px var(--c);
  }
  .on .swatch {
    background: var(--c);
  }
  .mean-swatch {
    height: 3px;
    border-radius: 2px;
    background: var(--text-primary);
    box-shadow: none;
  }
  .name {
    flex: 1;
    min-width: 0;
    text-align: left;
  }
  .ap {
    font-family: var(--font-mono);
    font-size: var(--type-xs-size);
    color: var(--text-secondary);
  }
  .on .ap {
    color: var(--text-primary);
  }
  @media (pointer: coarse) {
    .list button {
      height: 44px;
    }
    .side-actions button {
      height: 36px;
      padding: 0 var(--space-3);
    }
  }
  @media (max-width: 767px) {
    .pr {
      grid-template-columns: minmax(0, 1fr);
    }
    .side {
      max-height: 320px;
    }
  }
</style>
