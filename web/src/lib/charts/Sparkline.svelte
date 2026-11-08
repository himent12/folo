<script lang="ts">
  /** A tiny trend line for tables and tiles. Decorative: always pair it with the number it summarises. */
  import { scaleLinear } from 'd3-scale';
  import { area, line } from 'd3-shape';

  let {
    values,
    width = 80,
    height = 24,
    color = 'var(--chart-1)',
    domain,
    fill = true,
    /** Total length the series will reach (e.g. epochs), so a live run fills left to right. */
    length,
  }: {
    values: number[];
    width?: number;
    height?: number;
    color?: string;
    domain?: [number, number];
    fill?: boolean;
    length?: number;
  } = $props();

  const geo = $derived.by(() => {
    const vs = values.filter((v) => Number.isFinite(v));
    if (vs.length < 2) return null;
    const lo = domain?.[0] ?? Math.min(...vs);
    const hi = domain?.[1] ?? Math.max(...vs);
    const n = Math.max(length ?? values.length, values.length) - 1;
    const x = scaleLinear().domain([0, Math.max(1, n)]).range([1, width - 3]);
    const y = scaleLinear().domain([lo, hi === lo ? lo + 1 : hi]).range([height - 2, 2]);
    const pts = values.map((v, i) => [x(i), y(v)] as [number, number]);
    const last = pts[pts.length - 1]!;
    return {
      d: line()(pts) ?? '',
      a: area()
        .y0(height)
        .y1((p) => p[1])(pts) ?? '',
      last,
    };
  });
</script>

<svg class="spark" {width} {height} viewBox="0 0 {width} {height}" aria-hidden="true">
  {#if geo}
    {#if fill}<path class="a" d={geo.a} style:fill={color} />{/if}
    <path class="l" d={geo.d} style:stroke={color} />
    <circle cx={geo.last[0]} cy={geo.last[1]} r="2" style:fill={color} />
  {:else}
    <line x1="1" x2={width - 1} y1={height / 2} y2={height / 2} class="none" />
  {/if}
</svg>

<style>
  .spark {
    display: block;
    overflow: visible;
    flex-shrink: 0;
  }
  .l {
    fill: none;
    stroke-width: 1.5;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
  .a {
    opacity: 0.12;
  }
  .none {
    stroke: var(--line-default);
    stroke-dasharray: 2 3;
  }
</style>
