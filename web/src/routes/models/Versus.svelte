<script lang="ts">
  /**
   * How fast the engine is, against the incumbent of the same size on the same machine: every row of
   * docs/BENCHMARKS.md's summary where both have a number, including the ones FOLO trails. Values in ink; the lead is a
   * word ("1.66× faster"), never a colour, and a row where FOLO is behind says so plainly.
   */
  import { advantage, BENCH_SOURCE, versusYolo26n, type Versus } from '$lib/benchmarks';
  import { num } from '$lib/format';
  import Panel from '$lib/ui/Panel.svelte';

  const fmt = (v: Versus, x: number) => `${num(x, v.unit === 'ms' ? (x < 10 ? 2 : 1) : 0)} ${v.unit}`;
  /** "1.66× faster", "17% less memory", "8% slower": the lead in the metric's own terms. */
  function lead(v: Versus): string {
    const a = advantage(v);
    const ahead = a >= 1;
    if (v.unit === 'MiB') return `${Math.round(Math.abs(1 - v.folo / v.yolo) * 100)}% ${ahead ? 'less' : 'more'} memory`;
    if (!ahead) return `${Math.round((1 / a - 1) * 100)}% slower`;
    return a >= 1.2 ? `${a.toFixed(2)}× faster` : `${Math.round((a - 1) * 100)}% faster`;
  }
</script>

<Panel title="Engine speed" meta="folo-n vs YOLO26n" pad="none" id="engine-speed">
  <table class="table vs">
    <thead>
      <tr>
        <th>Measured</th>
        <th class="r">folo-n</th>
        <th class="r">YOLO26n</th>
        <th class="r lead-h">FOLO</th>
      </tr>
    </thead>
    <tbody>
      {#each versusYolo26n as v (v.metric)}
        {@const ahead = advantage(v) >= 1}
        <tr class:behind={!ahead}>
          <td>
            <span class="m">{v.metric}</span>
            {#if v.note}<span class="note">{v.note}</span>{/if}
          </td>
          <td class="r num v">{fmt(v, v.folo)}</td>
          <td class="r num v yolo">{fmt(v, v.yolo)}</td>
          <td class="r lead">{lead(v)}</td>
        </tr>
      {/each}
    </tbody>
  </table>
  <!-- The conditions every row shares, said once. -->
  <p class="src">
    Both on the same RTX 4080 SUPER (and i7-14700KF for the CPU rows), 640 × 640 input. folo-n has untrained
    weights and YOLO26n its COCO-pretrained ones, so almost no FOLO candidates pass the confidence threshold and its
    end-to-end times include less NMS work than a trained model's (about 0.1–0.3 ms per image). Times and memory:
    lower is better; images per second: higher. Source: {BENCH_SOURCE.split(' · ')[0]}.
  </p>
</Panel>

<style>
  .vs td {
    vertical-align: top;
  }
  .m {
    display: block;
    font-weight: var(--weight-medium);
  }
  .note {
    display: block;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .r {
    text-align: right;
  }
  .v {
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }
  .yolo {
    color: var(--text-secondary);
  }
  .lead {
    white-space: nowrap;
    font-weight: var(--weight-medium);
  }
  .behind .lead {
    font-weight: var(--weight-regular);
    color: var(--text-secondary);
  }
  .src {
    padding: var(--space-3) var(--space-4);
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  /* Phones: the metric on its own line, then the three values in a row. */
  @media (max-width: 767px) {
    .vs thead {
      display: none;
    }
    .vs tr {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: var(--space-1) var(--space-3);
      padding: var(--space-3) var(--space-4);
      border-bottom: 1px solid var(--line-subtle);
    }
    .vs td {
      padding: 0;
      border: 0;
    }
    .vs td:first-child {
      grid-column: 1 / -1;
    }
    .vs .r {
      text-align: left;
    }
    .vs td.v::before {
      display: block;
      font-size: var(--type-xs-size);
      color: var(--text-tertiary);
      font-weight: var(--weight-regular);
    }
    .vs td.v:not(.yolo)::before {
      content: 'folo-n';
    }
    .vs td.yolo::before {
      content: 'YOLO26n';
    }
    .vs td.lead::before {
      content: 'FOLO';
      display: block;
      font-size: var(--type-xs-size);
      color: var(--text-tertiary);
      font-weight: var(--weight-regular);
    }
  }
</style>
