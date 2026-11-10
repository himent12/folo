<script lang="ts">
  /**
   * SEAM (the loop's detail, built next): one self-improvement loop opened under its row on the Loops tab. For now its
   * generations as a compact table: who each was, the labels its teacher made, its score and mAP@0.5:0.95 with the
   * 95% interval, and a link to its entry. The next version replaces the body (a generations chart against the first
   * teacher's interval, the labels and runs of each step, Stop and Delete) and keeps these props.
   */
  import type { Bench, BenchEntry, BenchLoop } from '$lib/api/types';
  import { metric, num } from '$lib/format';
  import { href } from '$lib/router.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import NoValue from '$lib/ui/NoValue.svelte';
  import { scoreText, VERDICT } from './words';

  let {
    bench,
    loop,
    entries,
  }: {
    bench: Bench;
    loop: BenchLoop;
    /** The benchmark's entries, to link each generation's own. */
    entries: readonly BenchEntry[];
  } = $props();

  const known = $derived(new Set(entries.map((e) => e.id)));
</script>

<div class="detail">
  {#if loop.verdict}<p class="verdict"><b>{VERDICT[loop.verdict].word}.</b> {VERDICT[loop.verdict].means}</p>{/if}
  {#if loop.error}<p class="err">{loop.error}{#if loop.hint}{' '}{loop.hint}{/if}</p>{/if}
  <table class="table dense gens" aria-label="Generations of {loop.name}">
    <thead>
      <tr>
        <th class="lead">Generation</th>
        <th>Model</th>
        <th class="hide-md">Labels it trained on</th>
        <th class="r hide-sm" style:width="72px">Score</th>
        <th class="r map">mAP50-95<span class="ci-head"> · 95% interval</span></th>
      </tr>
    </thead>
    <tbody>
      {#each loop.generations as g (g.generation)}
        <tr>
          <td class="lead num">{g.generation === 0 ? 'Teacher' : `G${g.generation}`}</td>
          <td>
            {#if g.entry_id && known.has(g.entry_id)}
              <a href={href(`/benchmarks/${bench.id}/entries/${g.entry_id}`)}>{g.model_name ?? 'its entry'}</a>
            {:else}{g.model_name ?? '…'}{/if}
          </td>
          <td class="hide-md muted num">
            {#if g.labels}<Meta line><span>{num(g.labels.boxes)} boxes on {num(g.labels.labeled_images)} images</span><span>from confidence {g.labels.threshold.toFixed(2)}</span></Meta>{:else if g.generation === 0}<span class="faint">True labels</span>{:else}<NoValue label="not yet" />{/if}
          </td>
          <td class="r num hide-sm">{#if g.score !== null}{scoreText(g.score)}{:else}<NoValue label="not yet" />{/if}</td>
          <td class="r num">{#if g.map50_95 !== null}{metric(g.map50_95)}{#if g.ci95}<span class="ci">{metric(g.ci95[0])}–{metric(g.ci95[1])}</span>{/if}{:else}<NoValue label="not yet" />{/if}</td>
        </tr>
      {/each}
    </tbody>
  </table>
</div>

<style>
  .detail {
    display: grid;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4) var(--space-4);
    white-space: normal;
  }
  .verdict,
  .err {
    color: var(--text-secondary);
  }
  .verdict b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .detail {
    min-width: 0;
  }
  .gens {
    border: 1px solid var(--line-subtle);
    border-radius: var(--radius-md);
    border-collapse: separate;
    border-spacing: 0;
  }
  /* Dense rows, whatever the table it opens in (the loops' table is tall). */
  .gens td {
    height: var(--size-row);
  }
  .gens th.map {
    width: 176px;
  }
  .ci {
    margin-left: var(--space-2);
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  /* Phones: the score and mAP alone; the interval is on the entry's own page. */
  @media (max-width: 767px) {
    .ci,
    .ci-head {
      display: none;
    }
    .gens th.map {
      width: auto;
    }
  }
</style>
