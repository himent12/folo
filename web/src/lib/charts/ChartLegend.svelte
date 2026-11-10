<script lang="ts">
  /** A line chart's legend, its readout: each series' swatch, name and value (at the cursor, else its latest). */
  import type { Pt } from './downsample';
  import { latest, type Series } from './lineSeries';
  import SeriesSwatch from './SeriesSwatch.svelte';

  let {
    hintText,
    hover,
    compact,
    legendNote,
    keyed,
    fmtValue,
  }: {
    hintText: string | null;
    hover: { rows: { s: Series; p: Pt; at: string | null }[]; title: string } | null;
    /** A narrow plot: the cursor's x reads out on the axis, not in the note. */
    compact: boolean;
    legendNote: string | undefined;
    keyed: Series[];
    fmtValue: (y: number) => string;
  } = $props();
</script>

<ul class="legend" aria-hidden="true">
  <!-- The note says which value the legend shows: the cursor's x, or at rest `legendNote` (narrow plots too, where
       the cursor's x reads out on the axis instead). -->
  {#if hintText}<li class="note hint">{hintText}</li>
  {:else if hover && !compact}<li class="note at">{hover.title}</li>
  {:else if legendNote && !hover}<li class="note">{legendNote}</li>{/if}
  {#each keyed as s (s.id)}
    {@const v = hover ? hover.rows.find((r) => r.s.id === s.id)?.p.y : latest(s)}
    <li>
      <SeriesSwatch {s} />
      <span class="name">{s.label}</span>
      <span class="val">{v === null || v === undefined ? '—' : fmtValue(v)}</span>
    </li>
  {/each}
</ul>

<style>
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-4);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  /* Items wrap whole: a value never ends up on a line of its own. */
  .legend li {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }
  .legend .note {
    /* Fixed width so the readout does not shift the legend as the cursor moves (phones wrap instead). */
    min-width: min(6.5em, 100%);
    color: var(--text-tertiary);
    font-variant-numeric: tabular-nums;
  }
  .legend .note.at {
    color: var(--text-secondary);
  }
  /* A reading hint takes the note's place for as long as it applies, on narrow plots too. */
  .legend .note.hint {
    min-width: 0;
    color: var(--text-secondary);
  }
  .name {
    color: var(--text-secondary);
  }
  .val {
    color: var(--text-primary);
    font-variant-numeric: tabular-nums;
  }
</style>
