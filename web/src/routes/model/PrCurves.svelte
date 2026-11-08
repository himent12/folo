<script lang="ts">
  /** Precision-recall curves per class with a toggle list, plus the mean over all classes drawn emphasised. */
  import type { Evaluation } from '$lib/api/mock/evaluation';
  import LineChart, { type Series } from '$lib/charts/LineChart.svelte';
  import { classColor } from '$lib/design/tokens';
  import { metric } from '$lib/format';

  let { classes, ev, map50 }: { classes: readonly string[]; ev: Evaluation; map50: number | null } = $props();

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

  const series = $derived<Series[]>([
    ...[...shown]
      .sort((a, b) => a - b)
      .map((i) => ({ id: `c${i}`, label: classes[i] ?? `class ${i}`, color: classColor(i), points: ev.curves[i] ?? [] })),
    { id: 'mean', label: 'all classes', color: 'var(--text-primary)', points: ev.mean, emphasis: true },
  ]);

  const meanAp = $derived(map50 ?? ev.perClass.reduce((a, c) => a + c.ap50, 0) / Math.max(1, ev.perClass.length));
  // Few classes: a leaderboard by AP. Many: by frequency, matching the default selection.
  const order = $derived(
    ev.perClass.map((c) => c).sort((a, b) => (classes.length <= 10 ? b.ap50 - a.ap50 : b.instances - a.instances)),
  );
</script>

<div class="wrap">
  <div class="pr">
    <div class="plot">
      <span class="axis-y" aria-hidden="true">Precision</span>
      <LineChart
        {series}
        height={330}
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
      <div class="side-head">
        <span class="overline">Classes</span>
        <span class="side-actions">
          <button type="button" onclick={() => (picked = new Set(classes.map((_, i) => i)))}>All</button>
          <button type="button" onclick={() => (picked = new Set())}>None</button>
        </span>
      </div>
      <div class="mean">
        <span class="swatch mean-swatch" aria-hidden="true"></span>
        <span class="name">all classes</span>
        <span class="ap num">{metric(meanAp)}</span>
      </div>
      <ul class="list" aria-label="Classes shown on the chart">
        {#each order as c (c.class)}
          {@const on = shown.has(c.class)}
          <li>
            <button type="button" class:on aria-pressed={on} onclick={() => toggle(c.class)}>
              <span class="swatch" style:--c={classColor(c.class)} aria-hidden="true"></span>
              <span class="name truncate">{classes[c.class]}</span>
              <span class="ap num">{metric(c.ap50)}</span>
            </button>
          </li>
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
  .side-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 24px;
    padding: 0 var(--space-2);
  }
  .side-actions {
    display: flex;
    gap: 2px;
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
  @media (max-width: 767px) {
    .pr {
      grid-template-columns: minmax(0, 1fr);
    }
    .side {
      max-height: 240px;
    }
  }
</style>
