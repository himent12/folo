<script lang="ts">
  /**
   * What a score is made of, one row per axis in the score's order: its name, what it asks, a bar of its value on 0–1
   * in neutral ink, the value, and its share of this score (its weight over the measured axes' weights). An axis not
   * measured says so in words in place of its bar and takes no share (docs/BENCH.md, The score).
   */
  import type { BenchScore } from '$lib/api/types';
  import { AXIS_ASKS, AXIS_NAME } from '$lib/bench/rules';
  import { axisShares } from '$lib/bench/score';
  import { num, pct } from '$lib/format';

  let { score }: { score: BenchScore } = $props();
  const shares = $derived(axisShares(score.axes));
</script>

<ul class="axes" aria-label="The score's axes">
  {#each score.axes as a (a.axis)}
    <li class:none={a.value === null}>
      <span class="a-name">{AXIS_NAME[a.axis]}<span class="a-asks">{AXIS_ASKS[a.axis]}</span></span>
      {#if a.value === null}
        <span class="a-not">Not measured</span>
        <span class="a-v faint">—</span>
        <span class="a-share faint num">0%</span>
      {:else}
        <span class="a-bar" aria-hidden="true"><span class="a-fill" style:width="{Math.max(0, Math.min(1, a.value)) * 100}%"></span></span>
        <span class="a-v num">{num(a.value, 2)}</span>
        <span class="a-share num" title="Its share of this score">{pct(shares[a.axis], 0)}</span>
      {/if}
    </li>
  {/each}
</ul>

<style>
  .axes {
    display: grid;
  }
  li {
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(96px, 1fr) 40px 40px;
    align-items: center;
    gap: var(--space-3);
    min-height: 44px;
    padding: var(--space-2) 0;
    border-top: 1px solid var(--line-subtle);
  }
  li:first-child {
    border-top: 0;
  }
  .a-name {
    display: grid;
    font-weight: var(--weight-medium);
  }
  .a-asks {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  li.none .a-name {
    color: var(--text-secondary);
  }
  .a-bar {
    height: 6px;
    border-radius: 3px;
    background: var(--bg-active);
    overflow: hidden;
  }
  .a-fill {
    display: block;
    height: 100%;
    background: var(--text-secondary);
  }
  .a-not {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .a-v,
  .a-share {
    text-align: right;
  }
  .a-share {
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  /* Phones: the axis and what it asks over its bar, value and share. */
  @media (max-width: 767px) {
    li {
      grid-template-columns: minmax(0, 1fr) 40px 40px;
      row-gap: var(--space-2);
    }
    .a-name {
      grid-column: 1 / -1;
    }
  }
</style>
