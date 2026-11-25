/** A tiny trend line for tables and tiles. Decorative: always pair it with the number it summarises. */
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
  /** The end dot's colour, when it says more than the line (a live run's lime "now"); else the line's. */
  endColor?: string;
}

const fmt = (n: number) => Number(n.toFixed(2));

/** Direct SVG polyline and closed area paths for a sparkline (avoids pulling d3-scale/d3-shape for linear mapping). */
export function sparklinePaths(values: readonly number[], width: number, height: number, domain?: [number, number], length?: number) {
  let min = Infinity;
  let max = -Infinity;
  let finiteCount = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (!Number.isFinite(v)) continue;
    finiteCount++;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (finiteCount < 2) return null;
  const lo = domain?.[0] ?? min;
  const hi = domain?.[1] ?? max;
  const xSpan = Math.max(1, Math.max(length ?? values.length, values.length) - 1);
  const ySpan = hi === lo ? 1 : hi - lo;
  const xRange = width - 4;
  const yRange = height - 4;
  const xOf = (i: number) => fmt(1 + (i / xSpan) * xRange);
  const yOf = (v: number) => fmt(height - 2 - ((v - lo) / ySpan) * yRange);

  const lineParts: string[] = [];
  const areaParts: string[] = [];
  let run: [number, number][] = [];
  let last: [number, number] = [0, 0];

  const flushRun = () => {
    if (!run.length) return;
    lineParts.push(`M${run[0]![0]},${run[0]![1]}${run.slice(1).map(([px, py]) => `L${px},${py}`).join('')}`);
    if (run.length >= 2) {
      const firstX = run[0]![0];
      const lastX = run[run.length - 1]![0];
      areaParts.push(`M${firstX},${run[0]![1]}${run.slice(1).map(([px, py]) => `L${px},${py}`).join('')}L${lastX},${height}L${firstX},${height}Z`);
    }
    run = [];
  };

  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (!Number.isFinite(v)) {
      flushRun();
      continue;
    }
    const pt: [number, number] = [xOf(i), yOf(v)];
    run.push(pt);
    last = pt;
  }
  flushRun();

  return { d: lineParts.join(''), a: areaParts.join(''), last };
}

export default function Sparkline(props: SparklineProps) {
  const p = defaults(props, { width: 80, height: 24, color: 'var(--chart-1)', fill: true });

  const geo = computed(() => sparklinePaths(p.values, p.width, p.height, p.domain, p.length));

  return (
    <svg class="spark" width={p.width} height={p.height} viewBox={`0 0 ${p.width} ${p.height}`} aria-hidden="true">
      {geo.get() ? (
        <>
          {p.fill && geo.get()!.a ? <path class="a" d={geo.get()!.a} style:fill={p.color} /> : null}
          <path class="l" d={geo.get()!.d} style:stroke={p.color} />
          <circle cx={geo.get()!.last[0]} cy={geo.get()!.last[1]} r="2" style:fill={p.endColor ?? p.color} />
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
    /* The theme's chart.area_opacity, relative to its mode's default: the default themes draw 0.12 as before. */
    opacity: calc(var(--chart-area-opacity) / var(--chart-area-opacity-default) * 0.12);
  }
  .none {
    stroke: var(--line-default);
    stroke-dasharray: 2 3;
  }
`;
