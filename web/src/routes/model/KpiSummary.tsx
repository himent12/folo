/**
 * The model's four headline metrics: one strip of numbers in ink, or (`grid`, phones) a 2 × 2 summary that folds
 * to one line; the ⓘ after the first label opens what they mean.
 */
import ChevronDown from '$icons/chevron-down';
import { bindable, css, defaults, For } from 'fw';
import type { Model } from '$lib/api/types';
import type { Evaluation } from '$lib/api/evaluation';
import { evaluatedOn, metricsLabel, type EvalSource } from '$lib/evaluated';
import { atBestF1, conf, metric } from '$lib/format';
import InfoButton from '$lib/ui/InfoButton';

export interface KpiSummaryProps {
  m: Model;
  source: EvalSource;
  ev: Evaluation | null;
  head: 'o2m' | 'o2o';
  /** Phones: the 2 × 2 grid, inside the sections. */
  grid?: boolean;
  kpiFolded?: boolean;
  kpiOpen?: boolean;
  helpOpen?: boolean;
}

const kpis = (m: Model) => [
  { label: 'mAP50-95', value: m.metrics?.map50_95, sub: 'IoU 0.50:0.95' },
  { label: 'mAP50', value: m.metrics?.map50, sub: 'IoU 0.50' },
  // The stored metrics' own threshold (the epoch that wrote the weights), never /eval's: one source per strip.
  { label: 'Precision', value: m.metrics?.precision, sub: atBestF1(m.metrics?.best_f1_conf) },
  { label: 'Recall', value: m.metrics?.recall, sub: atBestF1(m.metrics?.best_f1_conf) },
];

export default function KpiSummary(props: KpiSummaryProps) {
  const p = defaults(props, { grid: false, kpiFolded: false });
  const kpiOpen = bindable(props, 'kpiOpen');
  const helpOpen = bindable(props, 'helpOpen');

  // What the four numbers mean: one place in every layout, after the strip's first label (the strip and the phone's
  // 2 × 2 grid alike), so it never lands on whichever cell happens to be last or top-right.
  const kHelp = () => <InfoButton label="What these metrics mean" controls="metric-help" bind:open={helpOpen} class="k-help" />;
  const metricHelp = () => (
    <dl class="k-body" id="metric-help">
      <div>
        <dt>AP and mAP</dt>
        <dd>AP is the area under one class's precision–recall curve. mAP is the mean of AP over all {p.m.classes.length} classes; per-class tables show AP.</dd>
      </div>
      <div>
        <dt>50 and 50-95</dt>
        <dd>The overlap (IoU) a box needs with a true box to count as found: 0.50, or AP averaged over IoU 0.50 to 0.95 in 0.05 steps.</dd>
      </div>
      <div>
        <dt>Precision, recall</dt>
        <dd>
          Measured at the confidence that gives the best F1 (precision and recall balanced, averaged over classes, at IoU 0.50): the model's best trade-off between missed objects and false boxes.{' '}
          {p.m.metrics?.best_f1_conf != null ? (
            <>Here conf {conf(p.m.metrics.best_f1_conf)}, from the epoch that wrote these weights.</>
          ) : (
            <>This model's file does not say which threshold that was.</>
          )}
        </dd>
      </div>
      <div>
        <dt>Evaluated on</dt>
        <dd>
          {evaluatedOn(p.source, p.ev ? p.ev.images : null)}, at {p.m.imgsz} px, {p.head === 'o2o' ? 'end to end (the one-to-one head, no NMS)' : 'with NMS IoU 0.7'}.
        </dd>
      </div>
    </dl>
  );

  return (
    <>
      {p.grid ? (
        // Phones: the summary heads its section, one line that unfolds the grid.
        <section class="kpi-grid num" class:folded={p.kpiFolded} aria-label={metricsLabel(p.source)}>
          {p.kpiFolded ? (
            <button type="button" class="kpi-line" aria-expanded={kpiOpen.get()} aria-controls="kpi-cells" onclick={() => kpiOpen.set(!kpiOpen.get())}>
              <span class="kl-name">mAP50-95</span>
              <span class="kl-value">{metric(p.m.metrics?.map50_95)}</span>
              <span class="kl-verb">All metrics <ChevronDown size={14} aria-hidden="true" /></span>
            </button>
          ) : null}
          {!p.kpiFolded || kpiOpen.get() ? (
            <>
              <dl class="kg" id="kpi-cells">
                <For each={kpis(p.m)} key={(k) => k.label}>
                  {(k, i) => (
                    <div class="kg-cell" class:hero={i() === 0}>
                      <dt>
                        {k().label}
                        {i() === 0 ? kHelp() : null}
                      </dt>
                      <dd>{metric(k().value)}</dd>
                    </div>
                  )}
                </For>
              </dl>
              {helpOpen.get() ? metricHelp() : null}
            </>
          ) : null}
        </section>
      ) : (
        <section class="kpis" aria-label={metricsLabel(p.source)}>
          <For each={kpis(p.m)} key={(k) => k.label}>
            {(k, i) => (
              <div class="kpi" class:hero={i() === 0}>
                <span class="k-label">
                  <span class="truncate">{k().label}</span>
                  {i() === 0 ? kHelp() : null}
                </span>
                <span class="k-value num">{metric(k().value)}</span>
                <span class="k-sub">{k().sub}</span>
              </div>
            )}
          </For>
          {helpOpen.get() ? metricHelp() : null}
        </section>
      )}
    </>
  );
}

css`
  /* 1 px gaps over a hairline-coloured background draw the dividers in any column count. */
  .kpis {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1px;
    background: var(--line-subtle);
    background-clip: padding-box;
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }
  .kpi {
    display: grid;
    grid-template-rows: 16px 40px 16px;
    gap: 6px;
    align-content: start;
    padding: var(--space-4);
    background: var(--bg-panel);
    min-width: 0;
  }
  .k-label,
  .k-sub {
    font-size: var(--type-sm-size);
    line-height: 16px;
    color: var(--text-tertiary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  /* The caption is read ("at conf 0.39 (best F1)"): secondary, like every KPI caption (ui/Stat, Run's tabs). */
  .k-sub {
    color: var(--text-secondary);
  }
  .kpis {
    position: relative;
  }
  .k-value {
    align-self: end;
    font-size: var(--type-2xl-size);
    line-height: 32px;
    letter-spacing: var(--type-2xl-tracking);
    font-weight: var(--weight-semibold);
  }
  .hero .k-label {
    color: var(--text-secondary);
  }
  /* The help toggle follows the first label, on its line: a 20 px box (the row is 16 px tall), with a 44 px target on
     touch screens. */
  .k-label {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    min-width: 0;
    overflow: visible;
  }
  /* The 24 px ⓘ sits in a 16 px label line without deepening it. */
  dt :global(.k-help),
  .k-label :global(.k-help) {
    margin: -4px 0;
  }
  .k-body {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-3) var(--space-8);
    padding: var(--space-4);
    background: var(--bg-panel);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .k-body > div {
    display: grid;
    grid-template-columns: 112px minmax(0, 1fr);
    align-content: start; /* a short definition beside a long one keeps its term on its first line */
    gap: var(--space-3);
    max-width: calc(112px + var(--space-3) + var(--measure)); /* the term column, then the definition at the measure */
  }
  .k-body dt {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .k-body dd {
    text-align: left;
    color: var(--text-secondary);
  }
  .hero .k-value {
    font-size: var(--type-3xl-size);
    line-height: 40px;
    letter-spacing: var(--type-3xl-tracking);
  }
  /* Phones, the head of the Curves section (folded to one line in the others): a compact 2 × 2 summary in the strip's
     frame, mAP50-95 first. Each cell is its label over
     its value, the cells parted by hairlines (1 px gaps over the line colour); the ⓘ sits in the top-right corner, and
     what the numbers mean opens under the grid. */
  .kpi-grid {
    position: relative;
    display: grid;
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
    overflow: hidden;
  }
  .kg {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1px;
    background: var(--line-subtle);
  }
  .kg-cell {
    display: grid;
    grid-template-rows: 16px 32px;
    gap: 2px;
    /* The phone panels' one 16 px inset, so labels line up with the sections' titles and bodies below. */
    padding: var(--space-2) var(--space-4);
    background: var(--bg-panel);
    min-width: 0;
  }
  .kg dt {
    font-size: var(--type-sm-size);
    line-height: 16px;
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .kg .hero dt {
    color: var(--text-secondary);
  }
  /* The desktop strip's hierarchy in two steps down: the headline at the KPI step (2xl), the others at lg, set 6 px
     lower in the same 32 px row so the two values in a row share a baseline. */
  .kg dd {
    text-align: left;
    padding-top: 6px;
    font-size: var(--type-lg-size);
    line-height: 26px;
    letter-spacing: var(--type-lg-tracking);
    font-weight: var(--weight-semibold);
  }
  .kg .hero dd {
    padding-top: 0;
    font-size: var(--type-2xl-size);
    line-height: var(--type-2xl-line);
    letter-spacing: var(--type-2xl-tracking);
  }
  .kg dt {
    display: flex;
    align-items: center;
    gap: var(--space-1);
  }
  .kpi-grid .k-body {
    border-top: 1px solid var(--line-subtle);
  }
  /* Folded (every phone section but Curves): one 48 px line in the grid's frame, the headline's label and value on
     the left and the verb that unfolds the four cells on the right. */
  .kpi-line {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
    width: 100%;
    min-height: 48px;
    padding: var(--space-3) var(--space-4);
    text-align: left;
  }
  .kpi-line:focus-visible {
    outline-offset: -2px;
  }
  .kl-name {
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .kl-value {
    font-size: var(--type-lg-size);
    line-height: 24px;
    letter-spacing: var(--type-lg-tracking);
    font-weight: var(--weight-semibold);
  }
  .kl-verb {
    display: inline-flex;
    align-self: center;
    align-items: center;
    gap: var(--space-1);
    margin-left: auto;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .kl-verb :global(svg) {
    color: var(--text-tertiary);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .kpi-line[aria-expanded='true'] .kl-verb :global(svg) {
    transform: rotate(180deg);
  }
  .kpi-line + .kg {
    border-top: 1px solid var(--line-subtle);
  }
  dt {
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  dd {
    min-width: 0;
    text-align: right;
    overflow-wrap: anywhere;
  }
  /* The definitions keep their two columns down to the tablet tier (a panel at 768–1279 px uses its width,
     DESIGN_DIRECTION §2.2); on a tablet each term sits over its definition, so a definition keeps the column's width
     instead of a 200 px strip beside a 112 px term. Phones: one column. */
  @media (max-width: 1023px) {
    .k-body > div {
      grid-template-columns: minmax(0, 1fr);
      gap: 2px;
      max-width: var(--measure);
    }
  }
  @media (max-width: 767px) {
    .k-body {
      grid-template-columns: minmax(0, 1fr);
      padding: var(--space-3);
    }
    .k-body > div {
      grid-template-columns: minmax(0, 1fr);
      gap: 2px;
    }
  }
`;
