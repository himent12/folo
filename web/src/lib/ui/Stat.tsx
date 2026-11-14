/**
 * A labelled metric, the KPI strip's tile (design/UNIFIED.md §2, KPI strip): its label in 12 px text.secondary, the
 * value in 24 / 600 (`lg`: the page's one 32 px value) with its unit and the secondary part of the value ("img/s",
 * "/ 100", a duration's "m" and "s") in 12 px text.tertiary on the value's baseline, and one caption line in 12 px
 * text.secondary. A tile is not a button. `sm` is the compact form for tight rails. `Figure` is the value alone, for a
 * strip that lays out its own tiles (Model's).
 */
import { css, defaults, Index } from 'fw';
import Skeleton from './Skeleton';

type Part = { t: string; k: 'n' | 'u' | 'slash' };
/** A duration as the format helpers write it: "42s", "7m", "12m 02s", "1h 04m" (a leading "~" kept). */
const DURATION = /^~?\d+[hms]( \d+[hms])?$/;

/**
 * A KPI value's parts, as Desktop sets them (apps/desktop `ui::metric`): the figures at the value's size; a duration's
 * units small ("12" "m" " 02" "s"); a pair's slash between its two full-size values ("0.849" "/" "0.762").
 */
export function figureParts(value: string): Part[] {
  const pair = value.split(' / ');
  if (pair.length === 2) return [{ t: pair[0]!, k: 'n' }, { t: ' / ', k: 'slash' }, { t: pair[1]!, k: 'n' }];
  if (!DURATION.test(value)) return [{ t: value, k: 'n' }];
  const out: Part[] = [];
  for (const [, n, u] of value.matchAll(/([^hms]+)([hms])/g)) out.push({ t: n!, k: 'n' }, { t: u!, k: 'u' });
  return out;
}

/** A KPI value with its unit (design/UNIFIED.md §2, KPI strip): see `figureParts`; `unit` follows, small. */
export function Figure(props: { value: string; unit?: string }) {
  return (
    <>
      <Index each={figureParts(props.value)}>{(p) => <>{p().k === 'n' ? p().t : <span class={p().k === 'u' ? 'unit dur' : 'slash'}>{p().t}</span>}</>}</Index>
      {props.unit ? <span class="unit">{props.unit}</span> : null}
    </>
  );
}

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
            <Figure value={p.value} unit={p.unit} />
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
  /* A unit and a duration's "m" / "s": 12 px tertiary on the value's baseline, at every value size (Desktop's
     KPI_UNIT). line-height 1 keeps the smaller text from growing the line box, so values with and without units
     share a baseline. */
  .unit {
    line-height: 1;
    margin-left: 4px;
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    letter-spacing: 0;
  }
  .unit.dur {
    margin-left: 1px;
  }
  /* A pair's slash: between two full-size values, a step down and in tertiary, a space each side (Desktop's metric_pair:
     4 px), so the pair reads and copies as "0.849 / 0.762". */
  .slash {
    line-height: 1;
    white-space: pre;
    font-size: var(--type-lg-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    letter-spacing: 0;
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
