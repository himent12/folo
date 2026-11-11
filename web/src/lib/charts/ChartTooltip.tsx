// fw port of ChartTooltip.svelte.
/** A line chart's floating tooltip at the cursor: the x title, then each series' swatch, name and value. */
import { bindable, css, For } from 'fw';
import type { Pt } from './downsample';
import type { Series } from './lineSeries';
import SeriesSwatch from './SeriesSwatch';

export interface ChartTooltipProps {
  hover: {
    rows: { s: Series; p: Pt; at: string | null }[];
    px: number;
    title: string;
  };
  /** Too close to the plot's right edge: the tooltip sits left of the cursor. */
  tipLeft: boolean;
  tipWidth?: number;
  fmtValue: (y: number) => string;
}

export default function ChartTooltip(props: ChartTooltipProps) {
  const tipWidth = bindable(props, 'tipWidth');
  return (
    <div class="tip" class:left={props.tipLeft} style={{ '--x': `${props.hover.px}px` }} role="presentation" bind:clientWidth={tipWidth}>
      <p class="tip-title">{props.hover.title}</p>
      <For each={props.hover.rows} key={(r) => r.s.id}>
        {(r) => (
          <p class="tip-row">
            <SeriesSwatch s={r().s} />
            <span class="name">
              {r().s.label}
              {r().at ? <span class="at">{r().at}</span> : null}
            </span>
            <span class="val">{props.fmtValue(r().p.y)}</span>
          </p>
        )}
      </For>
    </div>
  );
}

css`
  .name {
    color: var(--text-secondary);
  }
  .val {
    color: var(--text-primary);
    font-variant-numeric: tabular-nums;
  }
  .tip {
    position: absolute;
    top: 0;
    left: var(--x);
    transform: translateX(12px);
    min-width: 140px;
    padding: 6px 8px;
    border-radius: var(--radius-md);
    background: var(--bg-overlay);
    box-shadow: var(--shadow-md);
    pointer-events: none;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    z-index: 2;
    white-space: nowrap;
  }
  .tip.left {
    transform: translateX(calc(-100% - 12px));
  }
  .tip-title {
    color: var(--text-tertiary);
    margin-bottom: 2px;
    font-variant-numeric: tabular-nums;
  }
  .tip-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .tip-row .at {
    margin-left: 6px;
    color: var(--text-tertiary);
    font-variant-numeric: tabular-nums;
    font-size: var(--type-sm-size);
  }
  .tip-row .name {
    flex: 1;
  }
  .tip-row .val {
    padding-left: var(--space-3);
  }
`;
