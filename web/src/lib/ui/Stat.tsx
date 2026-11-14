/**
 * A labelled metric, the KPI strip's tile (design/UNIFIED.md §2, KPI strip): its label in 12 px text.secondary, the
 * value in 24 / 600 (`lg`: the page's one 32 px value) with its unit one type step smaller in text.tertiary, and one
 * caption line in 12 px text.secondary. A tile is not a button. `sm` is the compact form for tight rails.
 */
import { css, defaults } from 'fw';
import Skeleton from './Skeleton';

export interface StatProps {
  label: string;
  value: string;
  unit?: string;
  sub?: JSX.Element | string;
  size?: 'sm' | 'md' | 'lg';
  /** The label is real; the value (and sub, when given) are placeholders in their exact line boxes. */
  loading?: boolean;
}

export default function Stat(props: StatProps) {
  const p = defaults(props, { size: 'md', loading: false });
  return (
    <div class={`stat ${p.size}`}>
      <span class="label">{p.label}</span>
      {p.loading ? (
        <>
          <span class="value ph">
            <Skeleton w={p.size === 'lg' ? '96px' : '64px'} h={p.size === 'lg' ? '28px' : p.size === 'sm' ? '12px' : '16px'} />
          </span>
          {p.sub ? (
            <span class="sub ph">
              <Skeleton w="72px" h="10px" />
            </span>
          ) : null}
        </>
      ) : (
        <>
          {/* A value is a metric, always in ink: status hue goes on a mark beside its words, never on the number (§2.6). */}
          <span class="value">
            {p.value}
            {p.unit ? <span class="unit">{p.unit}</span> : null}
          </span>
          {p.sub ? <span class="sub">{p.sub}</span> : null}
        </>
      )}
    </div>
  );
}

css`
  /* Rows pack to the top: a stat without a sub line, stretched beside one with it, keeps its value on the shared
     baseline instead of spreading its rows over the extra height. */
  .stat {
    display: grid;
    align-content: start;
    gap: 2px;
    min-width: 0;
  }
  .label {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    white-space: nowrap;
  }
  /* Inline flow, not flex: a unit or a superscript never changes the line box, so every stat's value sits on the same
     baseline whatever it contains. */
  .value {
    display: block;
    /* A fixed row: glyphs from a fallback font (superscripts like ⁻³) cannot grow the line box. */
    height: var(--type-2xl-line);
    font-size: var(--type-2xl-size);
    line-height: var(--type-2xl-line);
    letter-spacing: var(--type-2xl-tracking);
    font-weight: var(--weight-semibold);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .sm .value {
    height: var(--type-md-line);
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    letter-spacing: var(--type-md-tracking);
    font-weight: var(--weight-medium);
  }
  .lg .value {
    height: var(--type-3xl-line);
    font-size: var(--type-3xl-size);
    line-height: var(--type-3xl-line);
    letter-spacing: var(--type-3xl-tracking);
    font-weight: var(--weight-semibold);
  }
  .ph {
    display: flex;
    align-items: center;
  }
  .sub.ph {
    height: var(--type-sm-line);
  }
  .unit {
    /* line-height 1 keeps a smaller unit from growing the line box, so values with and without units share a baseline. */
    line-height: 1;
    margin-left: 4px;
    font-size: var(--type-xl-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    letter-spacing: 0;
  }
  .lg .unit {
    font-size: var(--type-2xl-size);
  }
  .sm .unit {
    font-size: var(--type-sm-size);
  }
  .sub {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
`;
