/**
 * The model's headline numbers: one KPI strip of five tiles (mAP50-95 at twice the width and one size up, mAP50,
 * Precision / Recall, Parameters, File size), or (`grid`, phones) a 2 × 2 summary of the four metrics that folds to
 * one line; the ⓘ after the first label opens what the metrics mean.
 */
import ChevronDown from '$icons/chevron-down';
import { bindable, css, defaults, For } from 'fw';
import type { Model } from '$lib/api/types';
import type { Evaluation } from '$lib/api/evaluation';
import { evaluatedOn, metricsLabel, sourceShort, type EvalSource } from '$lib/evaluated';
import { atBestF1, bytes, compact, conf, metric, num } from '$lib/format';
import { prefs } from '$lib/prefs';
import InfoButton from '$lib/ui/InfoButton';
import KpiStrip from '$lib/ui/KpiStrip';

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

/** "2.62M" → ["2.62", "M"], "5.2 MiB" → ["5.2", "MiB"]: the figure, and its unit one step smaller. */
function splitUnit(s: string): [string, string] {
  const m = /^(.*?)\s?([kMB]|[KMGT]?i?B|[kMGT]B)$/.exec(s);
  return m ? [m[1]!, m[2]!] : [s, ''];
}

// Phones' 2 × 2 grid: the four metrics.
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
  // An imported or published file's stored metrics: what they were measured on is not recorded. The page's /eval
  // provenance (the curves) is not theirs, so it is never given as theirs (models F15: the list said "split unknown"
  // while this strip said "validation split of shapes").
  const fromFile = () => !p.m.run_id;
  const label = () => (fromFile() ? 'Metrics stored with the file' : metricsLabel(p.source));
  // The strip (§2 KPI strip): the headline names what it was measured on; the operating point, the compute and the
  // file's format are the other tiles' captions.
  const tiles = () => {
    const m = p.m;
    const mm = m.metrics;
    // In Desktop's few words ("fasteners · val"), so the caption holds in the tablet's narrower tile.
    const src = p.source;
    const where = fromFile() ? null : [src.datasetName, src.split].filter(Boolean).join(' · ') || null;
    const [pv, pu] = splitUnit(compact(m.params));
    const [bv, bu] = splitUnit(bytes(m.bytes, prefs.units));
    const dtype = m.precision === 'f32' || m.precision === 'f16' ? `${m.precision} weights` : null;
    return [
      { key: 'map', label: 'mAP50-95', value: metric(mm?.map50_95), sub: where ? `IoU 0.50–0.95 · ${where}` : 'IoU 0.50–0.95' },
      { key: 'map50', label: 'mAP50', value: metric(mm?.map50), sub: 'IoU 0.50' },
      // The stored metrics' own threshold (the epoch that wrote the weights), never /eval's: one source per strip.
      { key: 'pr', label: 'Precision / Recall', pair: [metric(mm?.precision), metric(mm?.recall)] as const, sub: atBestF1(mm?.best_f1_conf) },
      { key: 'params', label: 'Parameters', value: pv, unit: pu, sub: `${num(m.flops_g, 1)} GFLOPs` },
      { key: 'size', label: 'File size', value: bv, unit: bu, sub: ['.folo', dtype].filter(Boolean).join(' · ') },
    ] as { key: string; label: string; value?: string; unit?: string; pair?: readonly [string, string]; sub: string }[];
  };
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
          {fromFile() ? (
            <>
              Not recorded: these numbers came with the file, and it does not say which dataset or split they are on (so the
              Models list does not rank them).
              {sourceShort(p.source) ? ` The curves and the matrix are this server's own evaluation, on ${sourceShort(p.source)}.` : ''}
            </>
          ) : (
            <>
              {evaluatedOn(p.source, p.ev ? p.ev.images : null)}, at {p.m.imgsz} px, {p.head === 'o2o' ? 'end to end (the one-to-one head, no NMS)' : 'with NMS IoU 0.7'}.
            </>
          )}
        </dd>
      </div>
    </dl>
  );

  return (
    <>
      {p.grid ? (
        // Phones: the summary heads its section, one line that unfolds the grid.
        <section class="kpi-grid num" class:folded={p.kpiFolded} aria-label={label()}>
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
        // The Run page's strip (ui/KpiStrip: its frame, hairlines and tile padding), on this page's columns: the headline
        // twice as wide, its 32 px value on the same baseline as the others' 24 px ones.
        <section class="kpis" aria-label={label()}>
          <KpiStrip>
            <For each={tiles()} key={(k) => k.key}>
              {(k, i) => (
                <div class={`kpi ${k().key}`} class:hero={i() === 0}>
                  <span class="k-label">
                    <span class="truncate">{k().label}</span>
                    {i() === 0 ? kHelp() : null}
                  </span>
                  <span class="k-value num">
                    {k().pair ? (
                      <>
                        {k().pair![0]}
                        <span class="unit"> / </span>
                        {k().pair![1]}
                      </>
                    ) : (
                      <>
                        {k().value}
                        {k().unit ? <span class="unit"> {k().unit}</span> : null}
                      </>
                    )}
                  </span>
                  <span class="k-sub" title={k().sub}>
                    {k().sub}
                  </span>
                </div>
              )}
            </For>
            {helpOpen.get() ? metricHelp() : null}
          </KpiStrip>
        </section>
      )}
    </>
  );
}

css`
  /* Five tiles: the headline twice as wide as a plain one, the precision / recall pair a little wider (two values). */
  .kpis {
    --kpi-columns: minmax(0, 2fr) minmax(0, 1fr) minmax(0, 1.5fr) minmax(0, 1fr) minmax(0, 1fr);
  }
  .kpi {
    display: grid;
    grid-template-rows: 16px 40px 16px;
    gap: 6px;
    align-content: start;
    min-width: 0;
  }
  .k-label,
  .k-sub {
    font-size: var(--type-sm-size);
    line-height: 16px;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
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
  /* Units and the pair's slash one step smaller, in tertiary (README, Type). */
  .k-value .unit {
    font-size: var(--type-xl-size);
    letter-spacing: var(--type-xl-tracking);
    color: var(--text-tertiary);
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
    /* Tablets: the three metrics on one row, the model's size under them (a 680 px strip holds no five tiles). */
    .kpis {
      --kpi-columns: repeat(6, minmax(0, 1fr));
    }
    .kpi {
      grid-column: span 2;
    }
    .kpi.params,
    .kpi.size {
      grid-column: span 3;
    }
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
