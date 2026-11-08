<script lang="ts">
  /** Where the time of one prediction went: preprocess / inference / postprocess as a stacked bar. */
  import type { Timing } from '$lib/api/types';
  import { ms, num } from '$lib/format';

  let { timing, device }: { timing: Timing; device: string } = $props();

  const parts = $derived([
    { key: 'pre', label: 'Preprocess', v: timing.preprocess_ms, color: 'var(--chart-2)' },
    { key: 'inf', label: 'Inference', v: timing.inference_ms, color: 'var(--chart-1)' },
    { key: 'post', label: 'Postprocess', v: timing.postprocess_ms, color: 'var(--chart-4)' },
  ]);
  const sum = $derived(parts.reduce((a, p) => a + p.v, 0) || 1);
  const total = $derived(timing.total_ms || sum);
  const [whole, unit] = $derived(ms(total).split(' '));
</script>

<div class="timing">
  <div class="head">
    <span class="total num">{whole}<span class="unit">{unit}</span></span>
    <span class="rate num">{num(1000 / total, 0)} img/s</span>
  </div>
  <div
    class="bar"
    role="img"
    aria-label="Preprocess {ms(timing.preprocess_ms)}, inference {ms(timing.inference_ms)}, postprocess {ms(timing.postprocess_ms)}"
  >
    {#each parts as p (p.key)}
      <span style:flex-grow={p.v / sum} style:background={p.color}></span>
    {/each}
  </div>
  <ul class="legend">
    {#each parts as p (p.key)}
      <li>
        <span class="sw" style:background={p.color} aria-hidden="true"></span>
        <span class="name">{p.label}</span>
        <span class="v num">{ms(p.v)}</span>
        <span class="pct num">{Math.round((p.v / sum) * 100)}%</span>
      </li>
    {/each}
  </ul>
  <p class="device"><span class="faint">Device</span> <span class="mono">{device}</span></p>
</div>

<style>
  .timing {
    display: grid;
    gap: var(--space-3);
  }
  .head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
  }
  .total {
    font-size: var(--type-2xl-size);
    line-height: var(--type-2xl-line);
    letter-spacing: var(--type-2xl-tracking);
    font-weight: var(--weight-semibold);
  }
  .unit {
    margin-left: 3px;
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    letter-spacing: 0;
  }
  .rate {
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .bar {
    display: flex;
    gap: 2px;
    height: 8px;
    border-radius: var(--radius-full);
    overflow: hidden;
    background: var(--bg-active);
  }
  .bar span {
    flex-basis: 0;
    min-width: 3px;
  }
  .legend {
    display: grid;
    gap: 2px;
  }
  .legend li {
    display: grid;
    grid-template-columns: auto 1fr auto 36px;
    align-items: center;
    gap: var(--space-2);
    height: 22px;
    font-size: var(--type-sm-size);
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
  }
  .name {
    color: var(--text-secondary);
  }
  .v {
    font-family: var(--font-mono);
    font-size: var(--type-xs-size);
  }
  .pct {
    text-align: right;
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .device {
    display: flex;
    justify-content: space-between;
    gap: var(--space-2);
    padding-top: var(--space-2);
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
  }
</style>
