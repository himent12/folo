// Compare runs (`#/runs/compare?ids=a,b`): up to four runs' curves on one plot, and a table of the runs side by side.
// The parts are in ./compare/.
import GitCompare from '$icons/git-compare-arrows';
import { css, onDocument } from 'fw';
import { isTyping } from '$lib/keys';
import { href } from '$lib/router';
import { setCrumbs } from '$lib/shell/crumbs';
import Page from '$lib/shell/Page';
import { pageKeys } from '$lib/shell/pageCommands';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import InfoButton from '$lib/ui/InfoButton';
import InfoRegion from '$lib/ui/InfoRegion';
import { provideInfoSlot } from '$lib/ui/infoSlot';
import { keyPrefs } from '$lib/ui/keyPrefs';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import AddRun from './compare/AddRun';
import ComparePlot, { MissingRunsNotice, PhoneAbout } from './compare/ComparePlot';
import CompareTable from './compare/CompareTable';
import { DATA_CAVEAT, MAX, METRIC_OPTIONS, PROGRESS_HOW, SLOT_PATTERN, X_OPTIONS, X_PHONE, Y_OPTIONS } from './compare/config';
import ManageSheet from './compare/ManageSheet';
import RunChips from './compare/RunChips';
import { compareState } from './compare/state';

export { COMPARE_POLL_MS, startFrom } from './compare/config';

export default function RunCompare() {
  setCrumbs({ label: 'Runs', path: '/runs' }, { label: 'Compare' });

  const s = compareState();
  // Phones: the axis ⓘ opens in place under the control row (ui/infoSlot).
  const axisInfo = provideInfoSlot('cmp-axis-info');

  // 1–4 pin or unpin a run chip across the plot (matching BenchCompare); B rotates baseline in CompareTable.
  pageKeys((k) => {
    if (document.querySelector('[aria-expanded="true"][aria-haspopup], [role="menu"]')) return false;
    const list = s.chosen.peek();
    if (list.length < 2) return false;
    if (k >= '1' && k <= '4') {
      const target = list[Number(k) - 1];
      if (!target) return false;
      s.pinned.set((p) => (p === target.id ? null : target.id));
      return true;
    }
    return false;
  });

  // Escape unpins an active chip before global Escape returns to Runs.
  onDocument('keydown', (e) => {
    if (e.key !== 'Escape' || e.defaultPrevented || e.isComposing || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey || isTyping(e)) return;
    if (!keyPrefs.singleKeys || document.querySelector('dialog[open], [aria-expanded="true"][aria-haspopup], [role="menu"], #folo-tooltip[data-show], .info-pop.open')) return;
    const p = s.pinned.peek();
    if (p !== null && s.chosen.peek().some((x) => x.id === p)) {
      e.preventDefault();
      s.pinned.set(null);
    }
  });

  const lede = () => <span>Overlay the curves of up to four runs. Colours follow the run, not its rank.</span>;

  const adder = () => <AddRun candidates={s.candidates.get()} ids={s.ids.get()} setIds={s.setIds} />;

  const chips = () => (
    <RunChips
      hint={s.hint.get()}
      readoutAt={s.readoutAt.get()}
      bind:managing={s.managing}
      chosen={s.chosen.get()}
      valueOf={s.valueOf}
      startOf={s.startOf}
      bind:pinned={s.pinned}
      bind:hovered={s.hovered}
      fill={s.fill}
      SLOT_PATTERN={SLOT_PATTERN}
      slotOf={s.slotOf}
      ids={s.ids.get()}
      setIds={s.setIds}
      narrow={s.narrow()}
      MAX={MAX}
      candidates={s.candidates.get()}
      adder={adder}
      metricErrorOf={(id) => s.metricErrors.get()[id]}
      retryingOf={(id) => !!s.retryingMetric.get()[id]}
      onRetryMetric={(id) => void s.fetchRunMetrics(id)}
    />
  );

  const plot = () => (
    <ComparePlot
      runs={s.runs}
      metrics={s.metrics}
      failedRuns={s.failedRuns.get()}
      failedWithCache={s.failedWithCache.get()}
      retryingMetric={s.retryingMetric.get()}
      onRetryMetric={(id) => void s.fetchRunMetrics(id)}
      metricId={s.metricId.get()}
      metricDef={s.metricDef.get()}
      series={s.series.get()}
      height={s.height.get()}
      x={s.x.get()}
      xDomain={s.xDomain.get()}
      fit={s.fit.get()}
      yCaption={s.yCaption.get()}
      bind:cursor={s.cursor}
      bind:hint={s.hint}
      highlight={s.highlight.get()}
    />
  );

  // Phones lead with the plot: the explanation sits in the disclosure under the legend.
  return (
    <Page title="Compare runs" sub={s.narrow() ? undefined : lede}>
      {s.runs.error && !s.runs.data ? (
        // The page's states sit in the chart's frame at the chart's height (DESIGN_DIRECTION 8), the shape the page
        // will have, never a frameless block in the corner of an empty canvas.
        <section class="compare c-state" aria-label="Curves" style:min-height={`${s.height.get()}px`}>
          <ErrorState error={s.runs.error} retry={() => s.runs.refresh()} compact />
        </section>
      ) : s.ids.get().length === 0 ? (
        <section class="compare c-state" aria-label="Curves" style:min-height={`${s.height.get()}px`}>
          <MissingRunsNotice dropped={s.dropped.get()} onDismiss={() => s.dropped.set([])} />
          <EmptyState
            icon={GitCompare}
            title="Pick runs to compare"
            description="Add two to four runs, or select them on the Runs page."
            // One action: pick runs here when there are any, else go and make some.
            actions={() => <>{s.candidates.get().length ? adder() : <Button href={href('/runs')}>Go to Runs</Button>}</>}
          />
        </section>
      ) : (
        <>
          <section class="compare" aria-label="Curves">
            <MissingRunsNotice dropped={s.dropped.get()} onDismiss={() => s.dropped.set([])} />
            {s.narrow() ? (
              // Phones: one control row, the runs, the one caveat that changes how to read it right above the plot,
              // then the plot. Opening, removing and adding runs sit in a sheet (Edit); the longer explanation sits in
              // the disclosure.
              <>
                <header class="c-row">
                  <span class="c-metric">
                    <Select label="Metric" bind:value={s.metricId} options={METRIC_OPTIONS} />
                  </span>
                  {/* The x axis and its ⓘ move together: on the narrowest phones both wrap under the metric. */}
                  <span class="c-xi">
                    <span class="c-x">
                      <Select label="X axis" bind:value={s.xBind} options={X_PHONE} />
                    </span>
                    {s.x.get() === 'progress' ? (
                      <span class="i-slot">
                        <InfoButton chart label="About the progress axis" text={s.progressWhy.get()} how={PROGRESS_HOW} />
                      </span>
                    ) : null}
                  </span>
                </header>
                {/* The ⓘ opens here, under the controls, never as a popover over the legend and plot (ui/infoSlot). */}
                <InfoRegion slot={axisInfo} />
                {chips()}
                {!s.sameData.get() && s.chosen.get().length > 1 ? <p class="caveat">{DATA_CAVEAT}</p> : null}
                {plot()}
                <PhoneAbout floorNote={s.floorNote.get()} fit={s.fit.get()} yAxis={s.yAxis.get()} setY={s.setY} x={s.x.get()} />
                <ManageSheet
                  bind:managing={s.managing}
                  chosen={s.chosen.get()}
                  startOf={s.startOf}
                  fill={s.fill}
                  SLOT_PATTERN={SLOT_PATTERN}
                  slotOf={s.slotOf}
                  ids={s.ids.get()}
                  setIds={s.setIds}
                  MAX={MAX}
                  candidates={s.candidates.get()}
                  adder={adder}
                />
              </>
            ) : (
              <>
                {chips()}
                {/* One line, and only when it changes how to read the plot; the axis names itself (and why). */}
                {!s.sameData.get() && s.chosen.get().length > 1 ? <p class="caveat">{DATA_CAVEAT}</p> : null}
                <header class="c-head">
                  <span class="c-what">
                    <Segmented label="Metric" bind:value={s.metricId} options={METRIC_OPTIONS} />
                    {/* The header qualifier: what the x axis is, and where a fitted y axis starts ("y from 0.40"). */}
                    <span class="qual">{s.floorNote.get() ? `${s.xQual.get()} · ${s.floorNote.get()}` : s.xQual.get()}</span>
                  </span>
                  <span class="c-ctl">
                    <Select size="sm" label="Y axis" value={s.yAxis.get()} onchange={s.setY} options={Y_OPTIONS} />
                    <Segmented size="sm" label="X axis" value={s.x.get()} onchange={s.setX} options={X_OPTIONS} />
                  </span>
                </header>
                {plot()}
              </>
            )}
          </section>
          {s.chosen.get().length ? (
            <CompareTable
              chosen={s.chosen.get()}
              ids={s.ids.get()}
              narrow={s.narrow()}
              datasets={s.datasets}
              metrics={s.metrics}
              sameData={s.sameData.get()}
              startOf={s.startOf}
              bind:showSame={s.showSame}
              bind:allRuns={s.allRuns}
              bind:otherAt={s.otherAt}
              bind:baselineId={s.baselineId}
            />
          ) : null}
        </>
      )}
    </Page>
  );
}

css`
  .compare {
    display: grid;
    gap: var(--space-3);
    padding: var(--space-4);
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    min-width: 0;
  }
  /* Nothing to draw yet (no runs picked, or the runs could not load): the chart panel's frame at the plot's height,
     the message centred where the curves will be. */
  .compare.c-state {
    place-content: center;
    justify-items: center;
  }
  /* The one caveat that changes how to read the plot, always in view, on one line above it. */
  .caveat {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  @media (max-width: 767px) {
    /* Compact groups 8 px apart (controls, legend, caveat, plot), so the plot starts within ~200 px of the controls. */
    .compare {
      gap: var(--space-2);
      padding: var(--space-3);
    }
  }
  /* The axis ⓘ's region sits on the frame's own edges; the section's gap spaces it. */
  .compare {
    --info-inset: 0;
    --info-after: 0;
  }
  /* Phones: metric picker and x axis share one row above the legend. The metric takes what is left (at least 140 px,
     so "mAP50-95" shows whole at 16 px touch text); the x axis is as wide as its longest word; the ⓘ has a real 44 px
     slot, its touch target inside it rather than over the select beside it. On the narrowest phones (320 px) the x axis
     and its ⓘ wrap to a second line rather than overflow. */
  .c-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
  .c-metric {
    flex: 1 1 140px;
    min-width: 140px;
    display: flex;
  }
  .c-metric > :global(.select) {
    flex: 1;
  }
  .c-xi {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .c-x {
    display: flex;
  }
  .i-slot {
    flex: 0 0 44px;
    display: grid;
    place-items: center;
    height: 44px;
  }
  /* From 768 px up one layout whatever the qualifier's length: the metric tabs left, the x-axis control right on the
     tabs' row; a long qualifier ("by progress, % of planned epochs · plans differ in length") wraps under the tabs,
     never pushing the control onto a row of its own. Phones have their own control row (.c-row). */
  .c-head {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: start;
    gap: var(--space-3);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
  }
  /* The chart's controls, right: the y axis's choice, then the x axis's. */
  .c-ctl {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .c-what {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-3);
    min-width: 0;
  }
  .c-what .qual {
    min-height: var(--control-md);
    display: inline-flex;
    align-items: center;
  }
  /* The chart's header qualifier, as on every chart: tertiary, 12 px. */
  .qual {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
`;
