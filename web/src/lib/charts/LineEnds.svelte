<svelte:options namespace="svg" />

<script lang="ts">
  /**
   * What sits at the lines' ends, inside the chart's SVG: each line's end mark (a dot, or how the run ended), the
   * direct labels with their leaders, and the best markers (the hollow ring, its pulse and its focus brackets).
   */
  import type { Placed } from './endLabels';
  import { MORE_ID } from './endLabels';
  import { endMarkInk, LIVE_GROW } from './endMarks';
  import { paintOrder } from './layers';
  import { inkOf, swatchDash, type Marker, type Series } from './lineSeries';

  let {
    endDots,
    keyed,
    receding,
    highlight,
    contextual,
    mergedInto,
    lifted,
    x,
    y,
    pulseBase,
    focusBase,
    showEnds,
    ends,
    markers,
    merges,
    xExtent,
    width,
    margin,
  }: {
    endDots: boolean;
    keyed: Series[];
    receding: boolean;
    highlight: string | null;
    contextual: (s: Series) => boolean;
    mergedInto: (id: string) => Marker | null;
    lifted: <T>(items: T[], id: (t: T) => string) => T[];
    x: (v: number) => number;
    y: (v: number) => number;
    pulseBase: number;
    focusBase: number;
    showEnds: boolean;
    ends: (Placed & { key: Series | null; tx: number; color: string; text: string; lead: { x1: number; y1: number; x2: number; y2: number } | null })[];
    markers: Marker[];
    merges: (string | null)[];
    xExtent: [number, number];
    width: number;
    margin: { top: number; right: number; bottom: number; left: number };
  } = $props();
</script>

<!-- The run just completed: lime viewfinder corners (the product's mark) close in on the best ring and fade. -->
{#snippet brackets(h: number)}
  {@const k = 3.5}
  <path class="focus" d="M{-h} {-h + k}V{-h}H{-h + k}M{h - k} {-h}H{h}V{-h + k}M{h} {h - k}V{h}H{h - k}M{-h + k} {h}H{-h}V{h - k}" />
{/snippet}

{#if endDots}
  {#each lifted(paintOrder(keyed, (s) => s), (s) => s.id) as s (s.id)}
    {@const p = s.points[s.points.length - 1]}
    {#if p && Number.isFinite(p.y)}
      {@const r = receding && s.id === highlight ? 4 : 3}
      {@const ink = contextual(s) ? 'var(--chart-context)' : inkOf(s)}
      {@const best = mergedInto(s.id)}
      {#if best}
        <!-- The best is the line's end: its ink ring around the end mark, one mark (./endMarks). -->
        {@const rr = (s.end === 'live' ? r + LIVE_GROW : r) + 3}
        <g class="marker" data-merged={s.id} transform="translate({x(p.x)},{y(p.y)})">
          {#if best.pulse && best.pulse > pulseBase}
            {#key best.pulse}<circle class="pulse" r={rr} />{/key}
          {/if}
          <circle class="ring" r={rr} />
          {#if best.focus && best.focus > focusBase}
            {#key best.focus}{@render brackets(rr + 4)}{/key}
          {/if}
        </g>
      {/if}
      {#if s.end === 'failed'}
        <path class="end-x" d="M{x(p.x) - r} {y(p.y) - r}l{2 * r} {2 * r}m0 {-2 * r}l{-2 * r} {2 * r}" style:stroke={endMarkInk(s.end, ink)} />
      {:else if s.end === 'stopped'}
        <rect class="end-dot" x={x(p.x) - r} y={y(p.y) - r} width={2 * r} height={2 * r} style:fill={ink} />
      {:else if s.end === 'live'}
        <!-- Still training: a filled lime dot with the heartbeat, StatusBadge's running mark. -->
        <circle class="end-beat" cx={x(p.x)} cy={y(p.y)} r={r + LIVE_GROW} />
        <circle class="end-dot end-live" cx={x(p.x)} cy={y(p.y)} r={r + LIVE_GROW} style:fill={endMarkInk(s.end, ink)} />
      {:else}
        <circle class="end-dot" cx={x(p.x)} cy={y(p.y)} {r} style:fill={ink} />
      {/if}
    {/if}
  {/each}
{/if}

{#if showEnds}
  {#each ends as e (e.id)}
    <!-- A label moved off its line's end (to keep clear of lines and labels) is tied to its marker by a leader. -->
    {#if e.lead}<line class="leader" x1={e.lead.x1} y1={e.lead.y1} x2={e.lead.x2} y2={e.lead.y2} style:stroke={e.color} />{/if}
  {/each}
  {#each ends as e (e.id)}
    {#if e.key}
      <!-- Keyed: the label could not sit beside its line's end, so it starts with the line's own swatch. -->
      <line class="end-key" x1={e.x + 1} x2={e.x + 13} y1={e.y} y2={e.y} style:stroke={receding && e.id !== highlight ? 'var(--chart-context)' : inkOf(e.key)} style:stroke-dasharray={swatchDash(e.key)} />
    {/if}
    <text class="end-label" class:hl={e.id === highlight} class:more={e.id === MORE_ID} x={e.tx} y={e.y} dy="0.32em" style:fill={e.text}>{e.label}</text>
  {/each}
{/if}
{#each markers as m, i (m.label + m.x)}
  {#if m.x >= xExtent[0] && m.x <= xExtent[1] && merges[i] == null}
    <g class="marker" transform="translate({x(m.x)},{y(m.y)})">
      {#if m.pulse && m.pulse > pulseBase}
        {#key m.pulse}<circle class="pulse" r="4.5" />{/key}
      {/if}
      <circle class="ring" r="4.5" />
      {#if m.focus && m.focus > focusBase}
        {#key m.focus}{@render brackets(8.5)}{/key}
      {/if}
      {#if !m.quiet}
        <text
          y="-10"
          text-anchor={x(m.x) > width - margin.right - 40 ? 'end' : x(m.x) < margin.left + 40 ? 'start' : 'middle'}
          dx={x(m.x) > width - margin.right - 40 ? 6 : x(m.x) < margin.left + 40 ? -6 : 0}>{m.label}</text
        >
      {/if}
    </g>
  {/if}
{/each}

<style>
  .end-dot {
    stroke: var(--bg-panel);
    stroke-width: 1.5;
  }
  /* A failed run's line ends in a cross, over a halo in the panel colour like the dots. */
  .end-x {
    fill: none;
    stroke-width: 2;
    stroke-linecap: round;
    filter: drop-shadow(0 0 1px var(--bg-panel));
  }
  /* Still training: a filled lime dot (style:fill) whose heartbeat, a ring every 2.4 s, is StatusBadge's `live-pulse`
     drawn in SVG; off under reduced motion, where the dot still says "now". */
  .end-beat {
    fill: none;
    stroke: var(--accent-line);
    stroke-width: 1.5;
    opacity: 0;
    transform-box: fill-box;
    transform-origin: center;
    animation: end-beat 2.4s var(--ease-out) infinite;
  }
  @keyframes end-beat {
    0% {
      opacity: 0.55;
      transform: scale(1);
    }
    33%,
    100% {
      opacity: 0;
      transform: scale(2.2);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .end-beat {
      display: none;
    }
  }
  .end-label {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    paint-order: stroke;
    stroke: var(--bg-panel);
    stroke-width: 3px;
  }
  .end-label.hl {
    font-weight: var(--weight-semibold);
  }
  /* The column's "+N more": not a line's name, so the quiet ink and the regular weight. */
  .end-label.more {
    fill: var(--text-tertiary);
    font-weight: var(--weight-regular);
  }
  /* A keyed label's swatch: the legend's (2 px, round caps). */
  .end-key {
    stroke-width: 2;
    stroke-linecap: round;
  }
  .leader {
    stroke-width: 1;
    stroke-linecap: round;
  }
  /* The best epoch: a hollow ink ring, 1.5 px, the header's "○ best" glyph (./endMarks). On its own it is filled with
     the panel, so the line does not run through it; around an end mark it is only the ring. */
  .marker .ring {
    fill: none;
    stroke: var(--text-primary);
    stroke-width: 1.5;
  }
  .marker:not([data-merged]) .ring {
    fill: var(--bg-panel);
  }
  /* A new best: a ring grows out of the marker and fades, once, over the chart duration. */
  .marker .pulse {
    fill: none;
    stroke: var(--text-primary);
    stroke-width: 1.5;
    opacity: 0;
    transform-box: fill-box;
    transform-origin: center;
    animation: best-pulse var(--dur-chart) var(--ease-out);
  }
  @keyframes best-pulse {
    from {
      opacity: 0.9;
      transform: scale(1);
    }
    to {
      opacity: 0;
      transform: scale(2.4);
    }
  }
  /* The run completed: four lime corners close in on the best ring over 200 ms, hold, then fade (1 s in all). */
  .marker .focus {
    fill: none;
    stroke: var(--accent-line);
    stroke-width: 1.5;
    stroke-linecap: square;
    opacity: 0;
    transform-box: fill-box;
    transform-origin: center;
    animation: best-focus 1s var(--ease-out);
  }
  @keyframes best-focus {
    0% {
      opacity: 0;
      transform: scale(1.8);
    }
    20%,
    70% {
      opacity: 1;
      transform: scale(1);
    }
    100% {
      opacity: 0;
      transform: scale(1);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .marker .pulse,
    .marker .focus {
      display: none;
    }
  }
  .marker text {
    fill: var(--text-secondary);
    font-size: var(--type-sm-size);
    font-variant-numeric: tabular-nums;
    paint-order: stroke;
    stroke: var(--bg-panel);
    stroke-width: 3px;
  }
</style>
