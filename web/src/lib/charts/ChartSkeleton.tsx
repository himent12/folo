// fw port of ChartSkeleton.svelte.
/**
 * A line chart while its data loads: the real frame (legend row, gridlines, baseline axis, tick gutters) with only
 * the unknown parts as placeholders, so nothing moves when the chart arrives.
 */
import { css, defaults, Index } from 'fw';
import Skeleton from '$lib/ui/Skeleton';

export interface ChartSkeletonProps {
  height?: number;
  legend?: boolean;
  /** Rows the loaded legend takes (it wraps on narrow cards). */
  legendRows?: number;
}

export default function ChartSkeleton(props: ChartSkeletonProps) {
  const p = defaults(props, { height: 200, legend: true, legendRows: 1 });
  return (
    <div class="cs" aria-hidden="true">
      {p.legend ? (
        <div class="legend" style:height={`${p.legendRows * 16 + (p.legendRows - 1) * 4}px`}>
          <Skeleton w="48px" h="12px" />
          <Skeleton w="96px" h="12px" />
          <Skeleton w="80px" h="12px" />
        </div>
      ) : null}
      <div class="plot" style:height={`${p.height}px`}>
        <div class="ticks">
          <Index each={[0, 1, 2, 3, 4]}>{() => <Skeleton w="20px" h="10px" />}</Index>
        </div>
        <div class="grid">
          <Index each={[0, 1, 2, 3]}>{() => <span></span>}</Index>
          <span class="axis"></span>
        </div>
        <div class="xt">
          <Index each={[0, 1, 2, 3, 4, 5]}>{() => <Skeleton w="16px" h="10px" />}</Index>
        </div>
      </div>
    </div>
  );
}

css`
  .cs {
    display: grid;
    gap: var(--space-2);
    min-width: 0;
  }
  .legend {
    display: flex;
    gap: var(--space-4);
    height: 16px;
    align-items: center;
  }
  .plot {
    display: grid;
    grid-template-columns: 32px minmax(0, 1fr);
    grid-template-rows: minmax(0, 1fr) 20px;
    column-gap: var(--space-2);
  }
  .ticks {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: flex-end;
    padding: 4px 0 0;
  }
  .grid {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding-top: 8px;
  }
  .grid span {
    height: 1px;
    background: var(--chart-grid);
  }
  .grid .axis {
    background: var(--chart-axis);
  }
  .xt {
    grid-column: 2;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }
`;
