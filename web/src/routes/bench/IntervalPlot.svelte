<script lang="ts">
  /**
   * A value and its 95% interval as a dot and whiskers on a scale the whole leaderboard shares (./leaderboard
   * `intervalDomain`), so two rows whose intervals overlap are seen to overlap. `band` draws another interval behind
   * it (the leader's) across the plot's full height: in a table whose rows each draw it at the row's height, the bands
   * join into one column every row is read against. Decorative to the eye, named for assistive tech.
   */
  import { metric } from '$lib/format';
  import { along } from './leaderboard';

  let {
    value,
    ci,
    domain,
    band = null,
    height = 20,
    label = 'mAP50-95',
  }: {
    value: number;
    ci: readonly [number, number];
    domain: readonly [number, number];
    band?: readonly [number, number] | null;
    /** The plot's height in px (the dot and whiskers sit on its middle). */
    height?: number;
    label?: string;
  } = $props();

  const pct = (v: number) => `${(along(domain, v) * 100).toFixed(2)}%`;
  const mid = $derived(height / 2);
</script>

<svg class="plot" {height} role="img" aria-label="{label} {metric(value)}, 95% interval {metric(ci[0])} to {metric(ci[1])}">
  {#if band}<rect class="band" x={pct(band[0])} width="{((along(domain, band[1]) - along(domain, band[0])) * 100).toFixed(2)}%" y="0" {height} />{/if}
  <line class="axis" x1="0" x2="100%" y1={mid} y2={mid} />
  <line class="whisker" x1={pct(ci[0])} x2={pct(ci[1])} y1={mid} y2={mid} />
  <line class="cap" x1={pct(ci[0])} x2={pct(ci[0])} y1={mid - 4} y2={mid + 4} />
  <line class="cap" x1={pct(ci[1])} x2={pct(ci[1])} y1={mid - 4} y2={mid + 4} />
  <circle class="dot" cx={pct(value)} cy={mid} r="3.5" />
</svg>

<style>
  .plot {
    display: block;
    width: 100%;
    overflow: visible;
  }
  .band {
    fill: var(--bg-active);
  }
  .axis {
    stroke: var(--chart-grid);
    stroke-width: 1;
  }
  .whisker,
  .cap {
    stroke: var(--text-tertiary);
    stroke-width: 1.5;
    stroke-linecap: round;
  }
  .dot {
    fill: var(--text-primary);
  }
</style>
