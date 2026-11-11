/**
 * A value and its 95% interval as a dot and whiskers on a scale the whole leaderboard shares (./leaderboard
 * `intervalDomain`), so two rows whose intervals overlap are seen to overlap. `band` draws another interval behind
 * it (the leader's) across the plot's full height: in a table whose rows each draw it at the row's height, the bands
 * join into one column every row is read against. Decorative to the eye, named for assistive tech.
 */
import { css, defaults } from 'fw';
import { metric } from '$lib/format';
import { along } from './leaderboard';

export interface IntervalPlotProps {
  value: number;
  ci: readonly [number, number];
  domain: readonly [number, number];
  band?: readonly [number, number] | null;
  /** The plot's height in px (the dot and whiskers sit on its middle). */
  height?: number;
  label?: string;
  /** The interval is below the scale (the leaderboard's focus, ./leaderboard intervalScale): a mark at the left edge. */
  below?: boolean;
  /** The entry's own ink on Compare (`var(--chart-N)`); the leaderboard draws every row in ink. */
  color?: string | null;
}

export default function IntervalPlot(props: IntervalPlotProps) {
  const p = defaults(props, { band: null, height: 20, label: 'mAP50-95', below: false, color: null });
  const pct = (v: number) => `${(along(p.domain, v) * 100).toFixed(2)}%`;
  const mid = () => p.height / 2;

  return (
    <svg
      class="plot"
      height={p.height}
      style={{ '--ink': p.color ?? undefined }}
      role="img"
      aria-label={`${p.label} ${metric(p.value)}, 95% interval ${metric(p.ci[0])} to ${metric(p.ci[1])}`}
    >
      {p.band ? (
        <rect
          class="band"
          x={pct(p.band[0])}
          width={`${((along(p.domain, p.band[1]) - along(p.domain, p.band[0])) * 100).toFixed(2)}%`}
          y="0"
          height={p.height}
        />
      ) : null}
      <line class="axis" x1="0" x2="100%" y1={mid()} y2={mid()} />
      {p.below ? (
        /* Off the scale to the left: an arrowhead on the edge, pointing where it is. */
        <path class="off" d={`M0 ${mid()} l7 -4.5 v9 z`} />
      ) : (
        <>
          <line class="whisker" x1={pct(p.ci[0])} x2={pct(p.ci[1])} y1={mid()} y2={mid()} />
          <line class="cap" x1={pct(p.ci[0])} x2={pct(p.ci[0])} y1={mid() - 4} y2={mid() + 4} />
          <line class="cap" x1={pct(p.ci[1])} x2={pct(p.ci[1])} y1={mid() - 4} y2={mid() + 4} />
          <circle class="dot" cx={pct(p.value)} cy={mid()} r="3.5" />
        </>
      )}
    </svg>
  );
}

css`
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
    stroke: var(--ink, var(--text-tertiary));
    stroke-width: 1.5;
    stroke-linecap: round;
  }
  .dot {
    fill: var(--ink, var(--text-primary));
    stroke: var(--bg-panel);
    stroke-width: 1.5;
  }
  .off {
    fill: var(--text-tertiary);
  }
`;
