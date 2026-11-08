<script lang="ts">
  /**
   * Precision-recall curves with the per-class table as their legend. Every class is drawn as a thin tertiary line and
   * the mean over all classes in ink, so 80 classes read as one shape instead of spaghetti; hovering a table row
   * gives its curve the class hue, clicking keeps it highlighted. The legend row above the plot is the readout.
   */
  import type { Evaluation } from '$lib/api/evaluation';
  import LineChart, { type Series } from '$lib/charts/LineChart.svelte';
  import { classInk, classStroke } from '$lib/design/classInk.svelte';
  import { metric, num } from '$lib/format';
  import Input from '$lib/ui/Input.svelte';
  import Search from '@lucide/svelte/icons/search';
  import { matchesClass, trimPrTail } from './pr';

  let { classes, ev, map50 }: { classes: readonly string[]; ev: Evaluation; map50: number | null } = $props();

  // Phones get a shorter plot so the table (the legend) starts on the first screen.
  let narrow = $state(typeof matchMedia !== 'undefined' && matchMedia('(max-width: 767px)').matches);
  $effect(() => {
    const mq = matchMedia('(max-width: 767px)');
    const on = () => (narrow = mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  });

  // Few classes: a leaderboard by AP. Many: by frequency.
  const many = $derived(classes.length > 10);
  const order = $derived(ev.perClass.map((c) => c).sort((a, b) => (many ? b.instances - a.instances : b.ap50 - a.ap50)));

  let hovered = $state<number | null>(null);
  let selected = $state<number[]>([]);
  const toggle = (i: number) => (selected = selected.includes(i) ? selected.filter((x) => x !== i) : [...selected, i]);
  /** Classes drawn in their hue, in legend order: the selection, then the hovered row. */
  const lit = $derived(hovered !== null && !selected.includes(hovered) ? [...selected, hovered] : selected);

  const name = (i: number) => classes[i] ?? `class ${i}`;
  // Curves end at their last recall point with precision > 0; the zero padding up to recall 1 is not data.
  const series = $derived<Series[]>([
    ...classes
      .map((_, i) => i)
      .filter((i) => !lit.includes(i))
      .map((i) => ({ id: `c${i}`, label: name(i), color: 'var(--text-tertiary)', points: trimPrTail(ev.curves[i] ?? []) })),
    { id: 'mean', label: 'All classes', color: 'var(--text-primary)', points: trimPrTail(ev.mean), emphasis: true },
    ...lit.map((i) => ({ id: `c${i}`, label: name(i), color: classStroke(i), points: trimPrTail(ev.curves[i] ?? []), emphasis: true })),
  ]);

  const meanAp = $derived(map50 ?? ev.perClass.reduce((a, c) => a + c.ap50, 0) / Math.max(1, ev.perClass.length));
  const apOf = $derived(new Map(ev.perClass.map((c) => [c.class, c.ap50])));

  // Large class sets: a filter and 16 rows until the reader asks for all of them.
  const ROWS_MAX = 16;
  let query = $state('');
  let all = $state(false);
  const listed = $derived(order.filter((c) => matchesClass(name(c.class), query)));
  const rows = $derived(all || query ? listed : listed.slice(0, ROWS_MAX));
</script>

<div class="pr">
  <div class="legend" aria-live="polite">
    <span class="item"><span class="line mean" aria-hidden="true"></span>All classes <span class="num">{metric(meanAp)}</span></span>
    {#each lit as i (i)}
      <span class="item"><span class="line" style:background={classStroke(i)} aria-hidden="true"></span>{name(i)} <span class="num">{metric(apOf.get(i))}</span></span>
    {/each}
    {#if selected.length}
      <button type="button" class="clear" onclick={() => (selected = [])}>Clear</button>
    {:else if hovered === null}
      <span class="hint">Point at a class below to trace its curve</span>
    {/if}
  </div>
  <div class="plot">
    <LineChart
      {series}
      height={narrow ? 220 : 300}
      xDomain={[0, 1]}
      yDomain={[0, 1]}
      xFormat={(x) => x.toFixed(1)}
      yFormat={(y) => y.toFixed(1)}
      valueFormat={(y) => y.toFixed(3)}
      xTicks={6}
      yTicks={5}
      legend={false}
      label="Precision-recall curves at IoU 0.5 for {classes.length} classes and the mean over all classes"
    />
  </div>

  {#if many}
    <div class="filter">
      <Input bind:value={query} icon={Search} size="sm" type="search" placeholder="Find a class" aria-label="Find a class" />
    </div>
  {/if}
  <div class="scroll">
    <table class="table dense">
      <thead>
        <tr>
          <th scope="col">Class</th>
          <th scope="col" class="r">AP50</th>
          <th scope="col" class="r">AP50-95</th>
          <th scope="col" class="r hide-sm">Precision</th>
          <th scope="col" class="r hide-sm">Recall</th>
          <th scope="col" class="r">Instances</th>
        </tr>
      </thead>
      <tbody>
        {#each rows as c (c.class)}
          {@const on = selected.includes(c.class)}
          <tr
            class="link"
            class:on
            class:hot={hovered === c.class}
            onpointerenter={() => (hovered = c.class)}
            onpointerleave={() => (hovered = null)}
            onclick={() => toggle(c.class)}
          >
            <td>
              <button type="button" class="cls" aria-pressed={on} onclick={(e) => (e.stopPropagation(), toggle(c.class))} onfocus={() => (hovered = c.class)} onblur={() => (hovered = null)}>
                <span class="sw" style:background={classInk(c.class)} aria-hidden="true"></span>
                <span class="truncate">{name(c.class)}</span>
              </button>
            </td>
            <td class="r">{metric(c.ap50)}</td>
            <td class="r muted">{metric(c.ap50_95)}</td>
            <td class="r hide-sm muted">{metric(c.precision)}</td>
            <td class="r hide-sm muted">{metric(c.recall)}</td>
            <td class="r muted">{num(c.instances)}</td>
          </tr>
        {:else}
          <tr><td colspan="6" class="none">No class matches “{query.trim()}”</td></tr>
        {/each}
      </tbody>
    </table>
  </div>
  {#if !query && listed.length > ROWS_MAX}
    <button type="button" class="more" aria-expanded={all} onclick={() => (all = !all)}>
      {all ? `Show the first ${ROWS_MAX}` : `Show all ${listed.length}`}
    </button>
  {/if}
  <p class="foot">Evaluated on {num(ev.images)} validation images at IoU 0.5; instances are ground-truth boxes per class.</p>
</div>

<style>
  .pr {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .legend {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-4);
    min-height: 24px;
    padding: 0 var(--space-4);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .item {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    white-space: nowrap;
  }
  .item .num {
    color: var(--text-primary);
  }
  .line {
    width: 12px;
    height: 2px;
    border-radius: 1px;
    flex-shrink: 0;
  }
  .line.mean {
    background: var(--text-primary);
  }
  .hint {
    color: var(--text-tertiary);
  }
  .clear {
    height: 20px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
  }
  .clear:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }
  /* Class curves are 1 px hairlines; the mean and the highlighted classes 2 px. */
  .plot {
    padding: var(--space-2) var(--space-4) var(--space-4);
  }
  .plot :global(path.line) {
    stroke-width: 1;
  }
  .plot :global(path.line.emphasis) {
    stroke-width: 2;
  }
  .filter {
    width: 240px;
    max-width: calc(100% - 2 * var(--space-4));
    margin: 0 var(--space-4) var(--space-2);
  }
  .scroll {
    overflow-x: auto;
    border-top: 1px solid var(--line-subtle);
  }
  /* Selected classes: a 2 px accent bar on the row's leading edge. */
  tr.hot {
    background: var(--bg-hover);
  }
  tr.on {
    background: var(--bg-selected);
  }
  tr.on td:first-child {
    box-shadow: inset 2px 0 0 var(--accent-line);
  }
  .cls {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    max-width: 200px;
    text-align: left;
  }
  .cls:focus-visible {
    outline-offset: 2px;
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  .none {
    text-align: center;
    color: var(--text-tertiary);
  }
  .more {
    display: block;
    width: 100%;
    height: var(--control-lg);
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .more:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .more:focus-visible {
    outline-offset: -2px;
  }
  .foot {
    margin-top: auto;
    padding: var(--space-3) var(--space-4);
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  @media (pointer: coarse) {
    .more {
      height: 44px;
    }
  }
  @media (max-width: 767px) {
    .hide-sm {
      display: none;
    }
    .legend,
    .plot {
      padding-inline: var(--space-3);
    }
  }
</style>
