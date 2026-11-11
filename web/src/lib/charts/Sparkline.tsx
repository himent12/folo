// fw port of Sparkline.svelte.
/** A tiny trend line for tables and tiles. Decorative: always pair it with the number it summarises. */
import { scaleLinear } from 'd3-scale';
import { area, line } from 'd3-shape';
import { computed, css, defaults } from 'fw';

export interface SparklineProps {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
  domain?: [number, number];
  fill?: boolean;
  /** Total length the series will reach (e.g. epochs), so a live run fills left to right. */
  length?: number;
}

export default function Sparkline(props: SparklineProps) {
  const p = defaults(props, { width: 80, height: 24, color: 'var(--chart-1)', fill: true });

  const geo = computed(() => {
    const values = p.values;
    const { width, height, domain, length } = p;
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
      a:
        area()
          .y0(height)
          .y1((pt) => pt[1])(pts) ?? '',
      last,
    };
  });

  return (
    <svg class="spark" width={p.width} height={p.height} viewBox={`0 0 ${p.width} ${p.height}`} aria-hidden="true">
      {geo.get() ? (
        <>
          {p.fill ? <path class="a" d={geo.get()!.a} style:fill={p.color} /> : null}
          <path class="l" d={geo.get()!.d} style:stroke={p.color} />
          <circle cx={geo.get()!.last[0]} cy={geo.get()!.last[1]} r="2" style:fill={p.color} />
        </>
      ) : (
        <line x1="1" x2={p.width - 1} y1={p.height / 2} y2={p.height / 2} class="none" />
      )}
    </svg>
  );
}

css`
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
`;
