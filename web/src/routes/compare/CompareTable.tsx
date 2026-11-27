/**
 * The compared runs side by side: status, best score, precision and recall, then the settings that differ (each read
 * against the baseline run), the identical ones folded. Phones show the baseline and one other run at a time.
 */
import ArrowLeftRight from '$icons/arrow-left-right';
import Check from '$icons/check';
import ChevronLeft from '$icons/chevron-left';
import ChevronRight from '$icons/chevron-right';
import Copy from '$icons/copy';
import Download from '$icons/download';
import { bindable, computed, css, effect, For, MediaQuery, signal } from 'fw';
import { bestMark, bestWord } from '$lib/bestMark';
import type { Resource } from '$lib/api/resource';
import type { Dataset, Run, RunMetrics } from '$lib/api/types';
import { conf, count, metric } from '$lib/format';
import { settingText, type Setting } from '../run/resolved';
import { pageKeys } from '$lib/shell/pageCommands';
import Button from '$lib/ui/Button';
import Panel from '$lib/ui/Panel';
import Note from '$lib/ui/Note';
import StatusBadge from '$lib/ui/StatusBadge';
import { media } from '$lib/design/tokens';
import Ident from '$lib/ui/Ident';
import Meta from '$lib/ui/Meta';
import { buildCompareSettingRows, type CompareSettingRow as Row } from './export';

export interface CompareTableProps {
  chosen: Run[];
  ids: string[];
  narrow: boolean;
  datasets: Resource<Dataset[]>;
  metrics: Resource<Map<string, RunMetrics>>;
  /** Whether every run trains on one dataset: scores compare only on the same data. */
  sameData: boolean;
  startOf: (r: Run) => string | null;
  /** The identical settings unfolded. */
  showSame?: boolean;
  /** Phones: the whole table instead of two runs at a time. */
  allRuns?: boolean;
  /** Phones: which of the other runs sits beside the baseline. */
  otherAt?: number;
  /** The baseline run's id; null for the first run. */
  baselineId?: string | null;
  /** Copies the run comparison report (results and settings) as Markdown. */
  onCopyMarkdown?: () => void;
  /** Downloads the run comparison report as CSV. */
  onDownloadCsv?: () => void;
  /** True briefly after Copy Markdown succeeds. */
  copiedMd?: boolean;
}

export default function CompareTable(props: CompareTableProps) {
  const showSame = bindable(props, 'showSame', false);
  const allRuns = bindable(props, 'allRuns', false);
  const otherAt = bindable(props, 'otherAt', 0);
  const baselineId = bindable(props, 'baselineId', null);

  const dsName = (id: string) => props.datasets.data?.find((d) => d.id === id)?.name ?? props.chosen.find((r) => r.config.dataset_id === id)?.dataset?.name ?? id;
  /**
   * The best mAP50-95 of each dataset the runs share (lib/bestMark, as the Models list ranks within a group): "best", or
   * "joint best" on every run of an exact tie, and "on <dataset>" when the comparison spans datasets, since scores
   * compare only on the same data. A dataset with one run here has no best. Values that differ only past three
   * decimals are shown with as many as tell them apart.
   */
  const bestOf = computed(() => {
    const chosen = props.chosen;
    const marks = new Map<string, { word: string; decimals: number }>();
    for (const id of new Set(chosen.map((r) => r.config.dataset_id))) {
      const group = chosen.filter((r) => r.config.dataset_id === id);
      const m = bestMark(group.map((r) => r.best?.map50_95));
      for (const r of group) if (m.decimals > 3) marks.set(r.id, { word: '', decimals: m.decimals });
      for (const i of m.best) marks.set(group[i]!.id, { word: props.sameData ? bestWord(m) : `${bestWord(m)} on ${dsName(id)}`, decimals: m.decimals });
    }
    return marks;
  });
  const bestScore = (r: Run) => (r.best && bestOf.get().get(r.id) ? r.best.map50_95.toFixed(bestOf.get().get(r.id)!.decimals) : metric(r.best?.map50_95));

  // A cell is a Setting (../run/resolved): an architecture or a checkpoint through ui/Ident, as on every other screen
  // (it sets a name in sans and a path or id in mono), prose ("From scratch") and named optimizers ("AdamW") in sans. Where the server resolved a
  // run's config (`Run.resolved`), the optimizer, LR and batch are what it trains with, the request a note beside them.
  const rows = computed((): Row[] => buildCompareSettingRows(props.datasets.data, props.startOf));
  const differs = (f: (r: Run) => Setting) => new Set(props.chosen.map((r) => settingText(f(r)))).size > 1;
  /** Precision and recall of the epoch Best names (the run's model), with the threshold they were read at. */
  const bestVal = (r: Run) => (r.best ? props.metrics.data?.get(r.id)?.epochs.find((e) => e.epoch === r.best?.epoch)?.val : undefined);
  // What differs is the comparison; identical settings fold into one row that can be opened.
  const diffRows = computed(() => (props.chosen.length > 1 ? rows.get().filter(([, f]) => differs(f)) : rows.get()));
  const sameRows = computed(() => (props.chosen.length > 1 ? rows.get().filter(([, f]) => !differs(f)) : []));
  const tableEl = signal<HTMLElement>();
  const tableMore = signal(false);
  /** Run columns (1-based) fully in view while the table scrolls sideways; null when every run fits. */
  const tableRange = signal<{ first: number; last: number } | null>(null);
  function measureTable() {
    const el = tableEl.peek();
    tableMore.set(!!el && el.scrollLeft + el.clientWidth < el.scrollWidth - 1);
    if (!el || el.scrollWidth <= el.clientWidth + 1) {
      tableRange.set(null);
      return;
    }
    // In view = at least 80 % of the column between the pinned setting column and the right edge.
    const ths = [...el.querySelectorAll<HTMLElement>('thead th')];
    const left = el.scrollLeft + (ths[0]?.offsetWidth ?? 0);
    const right = el.scrollLeft + el.clientWidth;
    let first = 0;
    let last = 0;
    ths.slice(1).forEach((th, i) => {
      if (th.offsetLeft + th.offsetWidth * 0.2 >= left - 1 && th.offsetLeft + th.offsetWidth * 0.8 <= right) {
        first ||= i + 1;
        last = i + 1;
      }
    });
    // An equal range writes nothing: a scroll within the same columns leaves the line above the table alone.
    const prev = tableRange.peek();
    if (first && prev && prev.first === first && prev.last === last) return;
    tableRange.set(first ? { first, last } : null);
  }
  const scrollTableEnd = () => {
    const el = tableEl.peek();
    el?.scrollTo({ left: el.scrollWidth, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  };
  effect(() => {
    const el = tableEl.get();
    if (!el) return;
    void props.chosen.length;
    measureTable();
    const ro = new ResizeObserver(measureTable);
    ro.observe(el);
    return () => ro.disconnect();
  });

  /** The side-by-side table's setting column (104 px two at a time on phones), and the narrowest a run column gets before
   *  the table scrolls. Set on the header row, the one row a fixed table takes its widths from, so the columns hold the
   *  same widths from the first render whatever the pager does (DESIGN_DIRECTION §2.7). */
  const SETTING_COL = 152;
  const PAIR_SETTING_COL = 104;
  const RUN_COL_MIN = 136;
  /** A few runs across a wide panel (two or three, from 1024 px): the setting column and the run columns share the width
   *  equally, so the table uses its panel (DESIGN_DIRECTION §2.2) and two runs' values sit a third of the panel apart,
   *  never packed at the left over an empty right half, nor 500 px apart beside a 152 px setting column. */
  const laptopUp = new MediaQuery(media.laptopUp, true);

  const pairMode = computed(() => props.narrow && props.chosen.length > 2 && !allRuns.get());
  const shared = computed(() => !pairMode.get() && laptopUp.current && props.chosen.length <= 3);
  const base = computed(() => props.chosen.find((r) => r.id === baselineId.get()) ?? props.chosen[0]);
  const others = computed(() => props.chosen.filter((r) => r !== base.get()));
  /** Makes `r` the baseline; the run it replaces takes the other column, so the pair on screen stays the same two. */
  function setBaseline(r: Run) {
    const prev = base.peek();
    baselineId.set(r.id);
    otherAt.set(Math.max(0, props.chosen.filter((x) => x !== r).indexOf(prev!)));
  }
  const other = computed(() => others.get()[Math.min(otherAt.get(), others.get().length - 1)]);
  // B: the next run becomes the baseline (in the chips' order, wrapping), the keyboard's Swap (shell/shortcuts lists it).
  pageKeys((k) => {
    if (document.querySelector('[aria-expanded="true"][aria-haspopup], [role="menu"]')) return false;
    const chosen = props.chosen;
    const b = base.peek();
    if (k.toLowerCase() !== 'b' || chosen.length < 2 || !b) return false;
    setBaseline(chosen[(chosen.indexOf(b) + 1) % chosen.length]!);
    return true;
  });
  const shown = computed(() => {
    const b = base.get();
    const o = other.get();
    return pairMode.get() && b && o ? [b, o] : props.chosen;
  });
  /** The phone pager's height: the pair's sticky head row holds just under it. */
  const pairH = signal(0);
  const stepOther = (d: number) => {
    const n = others.peek().length;
    otherAt.set((Math.min(otherAt.peek(), n - 1) + d + n) % n);
  };
  /** Whether a run's value differs from the baseline's: marked (a dot and the weight); an equal one is tertiary. */
  const offBase = (r: Run, f: (r: Run) => Setting) => {
    const b = base.get();
    return r !== b && !!b && settingText(f(r)) !== settingText(f(b));
  };

  const cell = (v: () => Setting) => (
    <>
      {v().ident ? (
        <Ident value={v().value} />
      ) : (
        <span class="val" class:whole={v().value.length <= 14 && !/\s/.test(v().value)} data-fact>
          {v().value}
        </span>
      )}
      {v().note ? <Note text={v().note!} fact /> : null}
      {v().about ? <span class="k-sub v-about">{v().about}</span> : null}
    </>
  );

  const rangeText = (t: { first: number; last: number }) => (t.first === t.last ? `Run ${t.first}` : `Runs ${t.first}–${t.last}`);

  const showExport = computed(() => props.chosen.length >= 2 && !!(props.onCopyMarkdown || props.onDownloadCsv));

  return (
    <Panel
      title="Side by side"
      meta={sameRows.get().length ? `${count(diffRows.get().length, 'setting')} ${diffRows.get().length === 1 ? 'differs' : 'differ'}` : undefined}
      pad="none"
      id="cmp-table"
      actions={
        showExport.get() ? (
          <span class="cmp-export-actions" data-run-compare-export-actions>
            {props.onCopyMarkdown ? (
              <Button size="sm" variant="ghost" icon={props.copiedMd ? Check : Copy} data-run-compare-copy-md onclick={props.onCopyMarkdown}>
                {props.copiedMd ? 'Copied' : 'Copy Markdown'}
              </Button>
            ) : null}
            {props.onDownloadCsv ? (
              <Button size="sm" variant="ghost" icon={Download} data-run-compare-download-csv onclick={props.onDownloadCsv}>
                Download CSV
              </Button>
            ) : null}
          </span>
        ) : undefined
      }
    >
      {pairMode.get() && other.get() ? (
        // Phones: the baseline and one run beside it; the pager steps through the others. The pager and the pair's
        // head row (both names, Baseline, Swap) hold under the top bar while the table is in view, so the names
        // never scroll away from the settings being read.
        <div class="pair num" bind:offsetHeight={pairH}>
          <button type="button" class="pg" aria-label="Previous run" onclick={() => stepOther(-1)}>
            <ChevronLeft size={16} aria-hidden="true" />
          </button>
          <span class="pair-what" aria-live="polite">
            Run {props.chosen.indexOf(other.get()!) + 1} of {props.chosen.length}
            <span class="faint">{' '}vs baseline</span>
          </span>
          <button type="button" class="pg" aria-label="Next run" onclick={() => stepOther(1)}>
            <ChevronRight size={16} aria-hidden="true" />
          </button>
          <button type="button" class="more" onclick={() => allRuns.set(true)}>
            All {props.chosen.length} runs
          </button>
        </div>
      ) : props.narrow && allRuns.get() && props.chosen.length > 2 ? (
        <p class="t-range num">
          {tableRange.get() ? (
            <>
              {rangeText(tableRange.get()!)} of {props.chosen.length}
              <span class="sep" aria-hidden="true">
                ·
              </span>
            </>
          ) : null}
          <button type="button" class="more" onclick={() => allRuns.set(false)}>
            Two at a time
          </button>
        </p>
      ) : tableRange.get() ? (
        // What the sideways scroll hides: the runs in view, and a way to the others.
        <p class="t-range num">
          {rangeText(tableRange.get()!)} of {props.chosen.length}
          {tableRange.get()!.last < props.chosen.length ? (
            <>
              <span class="sep" aria-hidden="true">
                ·
              </span>
              <button type="button" class="more" onclick={scrollTableEnd}>
                Scroll for {props.chosen.length - tableRange.get()!.last} more →
              </button>
            </>
          ) : null}
        </p>
      ) : null}
      <div
        class="scroll"
        class:more={tableMore.get()}
        class:pairing={pairMode.get() && !!other.get()}
        style={{ '--pair-h': `${pairH.get()}px` }}
        ref={tableEl}
        onscroll={measureTable}
      >
        {/* Fixed layout: every run column the same width, so each header's key sits over its own values. The widths
            live on the header row (no colgroup, whose keyed columns could outlive a page of runs). */}
        <table class="table fixed" class:two={pairMode.get()} style:min-width={pairMode.get() ? undefined : `${SETTING_COL + props.chosen.length * RUN_COL_MIN}px`}>
          <thead>
            <tr>
              <th style:width={shared.get() ? undefined : `${pairMode.get() ? PAIR_SETTING_COL : SETTING_COL}px`}></th>
              <For each={shown.get()} key={(r) => r.id}>
                {(r) => (
                  <th>
                    <span class="th">
                      <span class="th-name" class:truncate={!pairMode.get()} title={pairMode.get() ? undefined : r().name}>
                        {r().name}
                      </span>
                      {props.chosen.length > 1 ? (
                        r() === base.get() ? (
                          <span class="base-tag">Baseline</span>
                        ) : (
                          <button type="button" class="set-base" aria-label={`Make ${r().name} the baseline`} title="Make this run the baseline" onclick={() => setBaseline(r())}>
                            <ArrowLeftRight size={12} aria-hidden="true" />
                            Swap
                          </button>
                        )
                      ) : null}
                    </span>
                  </th>
                )}
              </For>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="k">Status</td>
              <For each={shown.get()} key={(r) => r.id}>
                {(r) => (
                  <td>
                    <StatusBadge run={r()} plain />
                  </td>
                )}
              </For>
            </tr>
            <tr>
              {/* The best epoch's score, which the chips' latest value need not be: the row says so. */}
              <td class="k">
                Best mAP50-95<span class="k-sub">any epoch</span>
              </td>
              <For each={shown.get()} key={(r) => r.id}>
                {(r) => {
                  const win = () => bestOf.get().get(r().id)?.word;
                  return (
                    <td class="num" class:winner={!!win()}>
                      {bestScore(r())}
                      {win() ? (
                        <>
                          {' '}
                          <span class="tag" data-fact>
                            {win()}
                          </span>
                        </>
                      ) : null}
                    </td>
                  );
                }}
              </For>
            </tr>
            <tr>
              <td class="k">Precision · recall</td>
              <For each={shown.get()} key={(r) => r.id}>
                {(r) => {
                  const v = () => bestVal(r());
                  return (
                    <td class="num">
                      {v() ? (
                        <>
                          {metric(v()!.precision)} · {metric(v()!.recall)}
                          <span class="pr-at">
                            {/* Nothing detected: said in words, as the run page says it, never "conf 1.00 · ep 1". */}
                            {v()!.precision === 0 && v()!.recall === 0 ? (
                              'no detections'
                            ) : (
                              <Meta>
                                <span>{v()!.best_f1_conf != null ? `conf ${conf(v()!.best_f1_conf)}` : 'best-F1 conf'}</span>
                                <span>ep {r().best?.epoch}</span>
                              </Meta>
                            )}
                          </span>
                        </>
                      ) : (
                        '—'
                      )}
                    </td>
                  );
                }}
              </For>
            </tr>
            <For each={diffRows.get()} key={(row) => row[0]}>
              {(row) => (
                <tr>
                  <td class="k">{row()[0]}</td>
                  <For each={shown.get()} key={(r) => r.id}>
                    {(r) => {
                      // Read against the baseline: a value that differs is marked (a dot and the weight), an equal one
                      // recedes to tertiary, so "11 settings differ" shows where they differ.
                      const off = () => offBase(r(), row()[1]);
                      const value = () => row()[1](r());
                      return (
                        <td class="num cfg" class:diff={off()} class:same={r() !== base.get() && !off()} class:base={r() === base.get()}>
                          {off() ? (
                            <span class="chg-row">
                              <span class="chg" aria-hidden="true"></span>
                              <span>
                                {cell(value)}
                                <span class="sr-only">, differs from the baseline</span>
                              </span>
                            </span>
                          ) : (
                            cell(value)
                          )}
                        </td>
                      );
                    }}
                  </For>
                </tr>
              )}
            </For>
            {sameRows.get().length ? (
              <>
                <tr class="fold">
                  <td colspan={shown.get().length + 1}>
                    <button aria-expanded={showSame.get()} onclick={() => showSame.set(!showSame.peek())}>
                      {showSame.get() ? 'Hide' : 'Show'} {sameRows.get().length} identical {sameRows.get().length === 1 ? 'setting' : 'settings'}
                    </button>
                  </td>
                </tr>
                {showSame.get() ? (
                  <For each={sameRows.get()} key={(row) => row[0]}>
                    {(row) => (
                      <tr>
                        <td class="k">{row()[0]}</td>
                        <For each={shown.get()} key={(r) => r.id}>
                          {(r) => <td class="num cfg">{cell(() => row()[1](r()))}</td>}
                        </For>
                      </tr>
                    )}
                  </For>
                ) : null}
              </>
            ) : null}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

css`
  .cmp-export-actions {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
  .t-range {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    min-height: 24px;
    padding: var(--space-2) var(--space-4) 0;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .t-range .sep {
    margin-inline: var(--space-2);
  }
  .t-range .more {
    color: var(--text-secondary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .t-range .more:hover {
    color: var(--text-primary);
  }
  /* Phones, two at a time: what is shown, the pager (44 px targets), and the way to the whole table. */
  .pair {
    display: flex;
    align-items: center;
    /* The ‹ chevron's glyph on the content edge; its 44 px target reaches into the padding. */
    padding: 0 var(--space-4) 0 calc(var(--space-4) - 14px);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .pair-what {
    text-align: center;
  }
  /* Two at a time, the pair fits the width: no sideways scroller (which would also trap the sticky head), and the
     pager and the head row stick under the top bar while the table is in view. */
  .pair {
    position: sticky;
    top: var(--size-topbar);
    z-index: var(--z-sticky);
    background: var(--bg-panel);
  }
  .scroll.pairing {
    overflow: visible;
  }
  .scroll.pairing :global(thead th) {
    position: sticky;
    top: calc(var(--size-topbar) + var(--pair-h, 0px));
    z-index: var(--z-sticky);
    background: var(--bg-panel);
    box-shadow: inset 0 -1px 0 var(--line-subtle);
  }
  .pair .more {
    margin-left: auto;
  }
  .pair .pg {
    flex-shrink: 0;
    display: grid;
    place-items: center;
    width: 44px;
    height: 44px;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
  }
  .pair .pg:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }
  .pair .more,
  .t-range .more {
    min-height: 44px;
  }
  .pair .more {
    color: var(--text-secondary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .pair .more:hover {
    color: var(--text-primary);
  }
  /* Nothing truncates: labels and run names wrap (a name at its hyphens, a word only when it is longer than the
     column), and the two headers start on one line. */
  table.two th,
  table.two td {
    white-space: normal;
    overflow-wrap: break-word;
    vertical-align: top;
  }
  table.two th {
    height: auto;
    padding-block: var(--space-2);
  }
  .base-tag {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  /* A changed value: the dot hangs in the cell's padding, on the value's first line, so changed and unchanged values
     share one left edge and the dot never wraps away from its value. */
  .chg-row {
    display: block;
    position: relative;
  }
  .chg {
    position: absolute;
    left: -10px; /* grid-ok: a 6 px dot 4 px before the text */
    top: 8px;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-primary);
  }
  td.same {
    color: var(--text-tertiary);
  }
  /* The baseline is the reference the other column is read against: plain body text. */
  td.cfg.base {
    color: var(--text-primary);
  }
  @media (max-width: 767px) {
    /* The side-by-side table scrolls sideways on phones: setting names stay pinned, and the right edge fades only
       while more runs lie beyond it. */
    .scroll.more {
      mask-image: linear-gradient(90deg, #000 calc(100% - 32px), transparent);
    }
    .scroll :global(th:first-child),
    .scroll :global(td.k) {
      position: sticky;
      left: 0;
      z-index: 1;
      background: var(--bg-panel);
    }
  }
  .fold td {
    padding-top: 0;
    padding-bottom: 0;
  }
  .fold button {
    height: 32px;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .fold button:hover {
    color: var(--text-primary);
  }
  .scroll {
    overflow-x: auto;
  }
  table.fixed {
    table-layout: fixed;
    width: 100%;
  }
  .fixed th,
  .fixed td {
    overflow: hidden;
    text-overflow: ellipsis;
  }
  /* A setting is a fact: the request beside its value moves, whole, to a line of its own rather than lose its end;
     the value itself ("360 steps") stays whole. */
  .fixed td.cfg {
    white-space: normal;
    overflow-wrap: break-word;
  }
  /* A short value of one word ("0.000833", "AdamW") stays whole; a value in words ("360 steps", "last 10 epochs",
     "Mixed precision (f16)") wraps at its spaces in a narrow column (a 320 px phone), never cut. So does the request
     beside a value ("(requested 0.01)"), which elsewhere keeps to one line. */
  .val.whole {
    white-space: nowrap;
  }
  .fixed td.cfg :global(.ui-note) {
    white-space: normal;
  }
  /* A run's head: its name over its role (Baseline, or Swap), each on a line of its own. */
  .th {
    display: grid;
    justify-items: start;
    color: var(--text-primary);
    font-size: var(--type-sm-size);
  }
  .th-name {
    max-width: 100%;
  }
  .fixed thead th {
    height: auto;
    padding-block: var(--space-2);
    vertical-align: top;
  }
  /* The row's label over what it measures ("any epoch"): 12 px tertiary. */
  .k-sub {
    display: block;
    font-size: var(--type-sm-size);
  }
  /* What a short value means ("Auto": MuSGD for every run), under the value it explains. */
  .v-about {
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .k {
    color: var(--text-tertiary);
  }
  td.diff {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  /* The best run's value: ink at 600 and a tertiary "best", never a lime value or chip (DESIGN_DIRECTION §2.6: status
     hue on the mark, not the metric; the Models list marks its best rows the same way). */
  .winner {
    color: var(--text-primary);
    font-weight: var(--weight-semibold);
  }
  /* A fact: in a narrow column the word moves, whole, under its value, never cut ("best on fast…" at 820). */
  .fixed td.winner {
    white-space: normal;
  }
  /* A phone's pair columns are narrower than "best on shapes-hard": there it wraps at its spaces. */
  .two .tag {
    white-space: normal;
  }
  .tag {
    display: inline-block;
    white-space: nowrap;
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
  }
  /* Where P · R were read. A setting's request beside its value is ui/Note. */
  .pr-at {
    display: block;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  /* The pair's other column: "Swap" makes it the baseline, the old baseline taking its place (the baseline never
     follows the legend's emphasis). One short word, so it stays on one line in the narrow column. */
  .set-base {
    justify-self: start;
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    white-space: nowrap;
    position: relative;
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-secondary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .set-base:hover {
    color: var(--text-primary);
  }
  /* A 44 px target around the 16 px line. */
  .set-base::after {
    content: '';
    position: absolute;
    inset: -14px -8px; /* grid-ok: a 44 px target around the 16 px line */
  }
`;
