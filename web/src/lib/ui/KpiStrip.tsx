/**
 * The KPI strip (design/UNIFIED.md §2): one panel cut into tiles by hairlines, each tile a `ui/Stat` (or anything of
 * that shape). The tiles share the row equally; `min` is the narrowest a tile may get before the strip wraps to more
 * rows (an intrinsic layout: the strip's own width decides, not the viewport's). A parent that needs a fixed count
 * (Overview's 2 × 2 block) sets `--kpi-columns` on an ancestor. `flush`: a region of a larger panel rather than a
 * panel of its own, without its outer edge, filling its container's height with rows of one height.
 */
import { css } from 'fw';

export interface KpiStripProps {
  children: JSX.Element;
  /** The narrowest tile before the strip wraps (default 160 px). */
  min?: string;
  label?: string;
  /** No outer edge: the strip is a region of the panel around it, its edges that panel's hairlines. */
  flush?: boolean;
}

export default function KpiStrip(props: KpiStripProps) {
  return (
    <div class="kpi-strip" class:flush={props.flush} role="group" aria-label={props.label} style={{ '--kpi-min': props.min ?? '160px' }}>
      {props.children}
    </div>
  );
}

css`
  /* The hairlines are the strip's own fill showing through 1 px gaps, clipped to the padding box so the translucent
     edge does not composite over it (DESIGN_DIRECTION §2.1). */
  .kpi-strip {
    display: grid;
    grid-template-columns: var(--kpi-columns, repeat(auto-fit, minmax(min(100%, var(--kpi-min)), 1fr)));
    gap: 1px;
    background: var(--line-subtle);
    background-clip: padding-box;
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }
  .kpi-strip.flush {
    border: 0;
    border-radius: 0;
    grid-auto-rows: 1fr;
  }
  .kpi-strip > :global(*) {
    padding: var(--space-3) var(--space-4);
    background: var(--bg-panel);
  }
`;
