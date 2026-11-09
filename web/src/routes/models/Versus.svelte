<script lang="ts">
  /**
   * How fast the engine is, against the incumbent of the same size on the same machine: every row of
   * docs/BENCHMARKS.md's summary where both have a number, including the ones FOLO trails. Values in ink; the lead is a
   * word ("1.66× faster", "17% less memory"), never a colour, and a row where FOLO is behind says so plainly. Only rows
   * measured under the same conditions state a winner: a row whose engines ran at different precision or did different
   * work keeps both measurements, and its conclusion is "—" over the specific difference ("f32 vs TensorFloat-32"),
   * with one footnote under the table saying what "—" means. Units sit in the row's header, never in the cells. Each row
   * keeps one short qualifier, and the method and caveats sit in a "How this was measured" fold.
   * Phones stack each row (the metric, then its three values, each with its engine named in the markup); the header
   * row stays for assistive tech, every value names its metric and column through `headers`, and explicit cell roles
   * keep the table's semantics through the grid layout.
   */
  import { tick } from 'svelte';
  import { advantage, BENCH_SOURCE, comparable, unlike, versusYolo26n, type Versus } from '$lib/benchmarks';
  import { num } from '$lib/format';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import Panel from '$lib/ui/Panel.svelte';

  const fmt = (v: Versus, x: number) => num(x, v.unit === 'ms' ? (x < 10 ? 2 : 1) : 0);
  /** One format per kind: speed as a ratio ("1.66× faster", "1.09× slower"), memory as a share ("17% less memory"). */
  function lead(v: Versus): string {
    const a = advantage(v);
    const ahead = a >= 1;
    if (v.unit === 'MiB') return `${Math.round(Math.abs(1 - v.folo / v.yolo) * 100)}% ${ahead ? 'less' : 'more'} memory`;
    return ahead ? `${a.toFixed(2)}× faster` : `${(1 / a).toFixed(2)}× slower`;
  }
  const caveats = versusYolo26n.filter((v) => v.caveat);

  // What keeps a row from being like for like: one list, lib/benchmarks.ts `unlike` (Models' summary counts it too).
  const mixed = versusYolo26n.filter((v) => !comparable(v)).length;

  let how = $state<HTMLDetailsElement>();
  /** "See how this was measured": opens the fold and takes the reader there. */
  async function showHow() {
    if (!how) return;
    how.open = true;
    await tick();
    const s = how.querySelector('summary');
    s?.focus({ preventScroll: true });
    s?.scrollIntoView({ block: 'nearest', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  }
</script>

<Panel title="Engine speed" pad="none" id="engine-speed">
  <p class="lede">folo-n vs YOLO26n on the same RTX 4080 SUPER, 640 px.</p>
  <table class="table vs">
    <thead>
      <tr>
        <th scope="col" role="columnheader" id="vs-h-m">Measured</th>
        <th scope="col" role="columnheader" class="r" id="vs-h-folo">folo-n</th>
        <th scope="col" role="columnheader" class="r" id="vs-h-yolo">YOLO26n</th>
        <th scope="col" role="columnheader" class="r lead-h" id="vs-h-lead">folo-n vs YOLO26n</th>
      </tr>
    </thead>
    <tbody>
      {#each versusYolo26n as v, i (v.metric)}
        {@const ahead = advantage(v) >= 1}
        {@const diff = unlike[v.metric]}
        <tr class:behind={!diff && !ahead} class:unlike={!!diff}>
          <th scope="row" role="rowheader" id="vs-r-{i}" headers="vs-h-m">
            <span class="m">{v.metric}<span class="unit">, {v.unit}</span></span>
            {#if v.note}<span class="note">{v.note}</span>{/if}
          </th>
          <!-- The engine's name over each value is for the stacked phone rows; screen readers get it from `headers`. -->
          <td role="cell" class="r num v" headers="vs-r-{i} vs-h-folo"><span class="cl" aria-hidden="true">folo-n</span>{fmt(v, v.folo)}</td>
          <td role="cell" class="r num v yolo" headers="vs-r-{i} vs-h-yolo"><span class="cl" aria-hidden="true">YOLO26n</span>{fmt(v, v.yolo)}</td>
          <td role="cell" class="r lead" headers="vs-r-{i} vs-h-lead"
            ><span class="cl" aria-hidden="true">folo-n vs YOLO26n</span>{#if diff}<span class="dc" aria-label="No winner: different conditions">—</span><span class="why">{diff.differs}</span>{:else}{lead(v)}{/if}</td
          >
        </tr>
      {/each}
    </tbody>
  </table>
  <!-- What "—" means, said once for every row that has it. -->
  {#if mixed}
    <div class="foot">
      <p><span aria-hidden="true">—&nbsp;</span>{mixed} of {versusYolo26n.length} rows ran under different conditions, so they name no winner;
        <button type="button" class="see" onclick={showHow}>see how this was measured</button>.</p>
    </div>
  {/if}
  <!-- How the rows were made and what qualifies them, one step away. -->
  <details class="how" bind:this={how}>
    <summary><ChevronDown size={14} aria-hidden="true" />How this was measured</summary>
    <div class="how-body">
      <p>
        One RTX 4080 SUPER (and an i7-14700KF for the CPU rows), 640 × 640 input. folo-n has untrained weights and YOLO26n
        its COCO-pretrained ones, so almost no FOLO candidates pass the confidence threshold and its end-to-end times
        include less NMS work than a trained model's (about 0.1–0.3 ms per image).
      </p>
      <p>Times and memory: lower is better. Images per second: higher is better.</p>
      <p>
        <span class="cv-m">Rows marked —:</span> the engines ran at different numeric precision (f32, TensorFloat-32, f16),
        did different work, or ran under different load. Those rows keep both measurements and name the difference, but
        no winner: the ratio would mix the engine with its conditions.
      </p>
      {#if caveats.length}
        <ul>
          {#each caveats as v (v.metric)}<li><span class="cv-m">{v.metric}:</span> {v.caveat}</li>{/each}
        </ul>
      {/if}
      <p>Source: {BENCH_SOURCE.split(' · ')[0]}.</p>
    </div>
  </details>
</Panel>

<style>
  /* Two-line rows (metric + qualifier): 8 px above and below, values on the metric's line. The row header looks like
     the table system's first cell (13 / 500 primary), not a column header. */
  .vs tbody :is(td, th) {
    vertical-align: top;
    padding-block: var(--space-2);
  }
  .vs tbody th {
    height: auto;
    text-align: left;
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
    color: var(--text-primary);
    white-space: normal;
    border-bottom: 1px solid var(--line-subtle);
  }
  .vs tbody tr:last-child th {
    border-bottom: 0;
  }
  .cl {
    display: none;
  }
  .m {
    display: block;
    font-weight: var(--weight-medium);
  }
  .lede {
    padding: 0 var(--space-4) var(--space-3);
    color: var(--text-secondary);
  }
  /* The footnote for "—", under the rows it explains, one step quieter. */
  .foot p {
    max-width: 60ch;
  }
  .foot {
    padding: var(--space-3) var(--space-4);
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .unit {
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .see {
    color: var(--text-secondary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .see:hover {
    color: var(--text-primary);
    text-decoration-color: currentColor;
  }
  @media (pointer: coarse) {
    .see {
      display: inline-block;
      padding-block: var(--space-3);
      margin-block: calc(var(--space-3) * -1);
    }
  }
  /* One 12 px tertiary line per row (qualifiers are ≤ 60 characters, tested). */
  .note {
    display: block;
    font-weight: var(--weight-regular);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .vs tbody th {
    max-width: 0;
    width: 50%;
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
  /* No winner: the conclusion in secondary ink, over the specific difference on the qualifier's 12 px line. */
  .unlike .lead {
    font-weight: var(--weight-regular);
  }
  .dc {
    display: block;
    color: var(--text-secondary);
  }
  .why {
    display: block;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .how {
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .how summary {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    min-height: 40px;
    padding: 0 var(--space-4);
    list-style: none;
    cursor: pointer;
    color: var(--text-secondary);
    border-radius: 0 0 calc(var(--radius-lg) - 1px) calc(var(--radius-lg) - 1px);
  }
  .how summary::-webkit-details-marker {
    display: none;
  }
  .how summary:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }
  .how summary:focus-visible {
    outline-offset: -2px;
  }
  .how summary :global(svg) {
    color: var(--text-tertiary);
    transform: rotate(-90deg);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .how[open] summary :global(svg) {
    transform: none;
  }
  .how-body {
    display: grid;
    gap: var(--space-2);
    max-width: calc(60ch + var(--space-4) * 2 + 18px);
    padding: 0 var(--space-4) var(--space-4) calc(var(--space-4) + 18px);
    color: var(--text-secondary);
  }
  .how-body ul {
    display: grid;
    gap: var(--space-1);
  }
  .cv-m {
    color: var(--text-primary);
  }
  @media (pointer: coarse) {
    .how summary {
      min-height: 44px;
    }
  }
  /* Phones: the metric on its own line, then the three values in a row. */
  @media (max-width: 767px) {
    /* The header row is read, not seen: each value shows its engine above it instead. */
    .vs thead {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip-path: inset(50%);
      white-space: nowrap;
    }
    .vs tbody tr {
      display: grid;
      /* One column template for every row, so folo-n, YOLO26n and the lead sit on the same three edges all the way
         down (a content-sized lead column would move them row by row); the lead's column is the widest. */
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1.5fr);
      gap: var(--space-1) var(--space-3);
      padding: var(--space-3) var(--space-4);
      border-bottom: 1px solid var(--line-subtle);
    }
    .vs tbody tr:last-child {
      border-bottom: 0;
    }
    /* Rows grow with their lines (the table's fixed row height would let a note run into the divider). */
    .vs tbody :is(td, th) {
      height: auto;
      padding: 0;
      border: 0;
    }
    .vs tbody th {
      grid-column: 1 / -1;
      max-width: none;
      width: auto;
    }
    .vs .r {
      text-align: left;
    }
    .vs .lead {
      white-space: normal;
    }
    .cl {
      display: block;
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-tertiary);
      font-weight: var(--weight-regular);
    }
  }
</style>
