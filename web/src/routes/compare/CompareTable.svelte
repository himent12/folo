<script lang="ts">
  /**
   * The compared runs side by side: status, best score, precision and recall, then the settings that differ (each read
   * against the baseline run), the identical ones folded. Phones show the baseline and one other run at a time.
   */
  import { MediaQuery } from 'svelte/reactivity';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import ChevronLeft from '@lucide/svelte/icons/chevron-left';
  import ArrowLeftRight from '@lucide/svelte/icons/arrow-left-right';
  import { bestMark, bestWord } from '$lib/bestMark';
  import type { Resource } from '$lib/api/resource.svelte';
  import type { Dataset, Run, RunMetrics } from '$lib/api/types';
  import { conf, deviceName, duration, metric } from '$lib/format';
  import { FROM_SCRATCH } from '$lib/lineage';
  import { batchSetting, effectiveBatchSetting, closeMosaicSetting, headSetting, lrSetting, optimizerSetting, precisionSetting, settingText, warmupSetting, type Setting } from '../run/resolved';
  import { pageKeys } from '$lib/shell/pageCommands.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Note from '$lib/ui/Note.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';
  import { media } from '$lib/design/tokens';
  import Ident from '$lib/ui/Ident.svelte';
  import Meta from '$lib/ui/Meta.svelte';

  let {
    chosen,
    ids,
    narrow,
    datasets,
    metrics,
    sameData,
    startOf,
    showSame = $bindable(),
    allRuns = $bindable(),
    otherAt = $bindable(),
    baselineId = $bindable(),
  }: {
    chosen: Run[];
    ids: string[];
    narrow: boolean;
    datasets: Resource<Dataset[]>;
    metrics: Resource<Map<string, RunMetrics>>;
    /** Whether every run trains on one dataset: scores compare only on the same data. */
    sameData: boolean;
    startOf: (r: Run) => string | null;
    /** The identical settings unfolded. */
    showSame: boolean;
    /** Phones: the whole table instead of two runs at a time. */
    allRuns: boolean;
    /** Phones: which of the other runs sits beside the baseline. */
    otherAt: number;
    /** The baseline run's id; null for the first run. */
    baselineId: string | null;
  } = $props();

  const dsName = (id: string) => datasets.data?.find((d) => d.id === id)?.name ?? id;
  /**
   * The best mAP50-95 of each dataset the runs share (lib/bestMark, as the Models list ranks within a group): "best", or
   * "joint best" on every run of an exact tie, and "on <dataset>" when the comparison spans datasets, since scores
   * compare only on the same data. A dataset with one run here has no best. Values that differ only past three
   * decimals are shown with as many as tell them apart.
   */
  const bestOf = $derived.by(() => {
    const marks = new Map<string, { word: string; decimals: number }>();
    for (const id of new Set(chosen.map((r) => r.config.dataset_id))) {
      const group = chosen.filter((r) => r.config.dataset_id === id);
      const m = bestMark(group.map((r) => r.best?.map50_95));
      for (const r of group) if (m.decimals > 3) marks.set(r.id, { word: '', decimals: m.decimals });
      for (const i of m.best) marks.set(group[i]!.id, { word: sameData ? bestWord(m) : `${bestWord(m)} on ${dsName(id)}`, decimals: m.decimals });
    }
    return marks;
  });
  const bestScore = (r: Run) => (r.best && bestOf.get(r.id) ? r.best.map50_95.toFixed(bestOf.get(r.id)!.decimals) : metric(r.best?.map50_95));

  // A cell is a Setting (../run/resolved): identifiers (an architecture, a checkpoint) in mono like the chips and every
  // other screen (ui/Ident), prose ("From scratch") and named optimizers ("AdamW") in sans. Where the server resolved a
  // run's config (`Run.resolved`), the optimizer, LR and batch are what it trains with, the request a note beside them.
  const plain = (value: string): Setting => ({ value });
  const ident = (value: string): Setting => ({ value, ident: true });
  const rows = $derived<[string, (r: Run) => Setting][]>([
    ['Dataset', (r) => plain(datasets.data?.find((d) => d.id === r.config.dataset_id)?.name ?? r.config.dataset_id)],
    ['Model', (r) => ident(r.config.arch)],
    ['Start from', (r) => (startOf(r) ? ident(startOf(r) ?? '') : plain(FROM_SCRATCH))],
    ['Epochs', (r) => plain(`${r.epoch}/${r.epochs}`)],
    ['Batch', (r) => batchSetting(r.config, r.resolved)],
    ['Effective batch', (r) => (r.resolved ? effectiveBatchSetting(r.resolved) : plain('—'))],
    ['Image size', (r) => plain(`${r.config.imgsz} px`)],
    ['Optimizer', (r) => optimizerSetting(r.config, r.resolved)],
    ['LR', (r) => lrSetting(r.config, r.resolved)],
    ['Warmup', (r) => (r.resolved ? warmupSetting(r.resolved) : plain('—'))],
    ['Mosaic / mixup', (r) => plain(`${r.config.augment.mosaic} / ${r.config.augment.mixup}`)],
    ['Close mosaic', (r) => closeMosaicSetting(r)],
    ['Precision', (r) => precisionSetting(r.config)],
    ['Head', (r) => headSetting(r.config)],
    ['Device', (r) => plain(deviceName(r.device))],
    ['Duration', (r) => plain(r.started ? duration(((r.finished ? Date.parse(r.finished) : Date.now()) - Date.parse(r.started)) / 1000) : '—')],
  ]);
  const differs = (f: (r: Run) => Setting) => new Set(chosen.map((r) => settingText(f(r)))).size > 1;
  /** Precision and recall of the epoch Best names (the run's model), with the threshold they were read at. */
  const bestVal = (r: Run) => (r.best ? metrics.data?.get(r.id)?.epochs.find((e) => e.epoch === r.best?.epoch)?.val : undefined);
  // What differs is the comparison; identical settings fold into one row that can be opened.
  const diffRows = $derived(chosen.length > 1 ? rows.filter(([, f]) => differs(f)) : rows);
  const sameRows = $derived(chosen.length > 1 ? rows.filter(([, f]) => !differs(f)) : []);
  let tableEl = $state<HTMLElement>();
  let tableMore = $state(false);
  /** Run columns (1-based) fully in view while the table scrolls sideways; null when every run fits. */
  let tableRange = $state<{ first: number; last: number } | null>(null);
  function measureTable() {
    const el = tableEl;
    tableMore = !!el && el.scrollLeft + el.clientWidth < el.scrollWidth - 1;
    if (!el || el.scrollWidth <= el.clientWidth + 1) {
      tableRange = null;
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
    tableRange = first ? { first, last } : null;
  }
  const scrollTableEnd = () =>
    tableEl?.scrollTo({ left: tableEl.scrollWidth, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  $effect(() => {
    const el = tableEl;
    if (!el) return;
    void chosen.length;
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

  const pairMode = $derived(narrow && chosen.length > 2 && !allRuns);
  const shared = $derived(!pairMode && laptopUp.current && chosen.length <= 3);
  const base = $derived(chosen.find((r) => r.id === baselineId) ?? chosen[0]);
  const others = $derived(chosen.filter((r) => r !== base));
  /** Makes `r` the baseline; the run it replaces takes the other column, so the pair on screen stays the same two. */
  function setBaseline(r: Run) {
    const prev = base;
    baselineId = r.id;
    otherAt = Math.max(0, chosen.filter((x) => x !== r).indexOf(prev!));
  }
  const other = $derived(others[Math.min(otherAt, others.length - 1)]);
  // B: the next run becomes the baseline (in the chips' order, wrapping), the keyboard's Swap (shell/shortcuts lists it).
  pageKeys((k) => {
    if (k.toLowerCase() !== 'b' || chosen.length < 2 || !base) return false;
    setBaseline(chosen[(chosen.indexOf(base) + 1) % chosen.length]!);
    return true;
  });
  const shown = $derived(pairMode && base && other ? [base, other] : chosen);
  /** The phone pager's height: the pair's sticky head row holds just under it. */
  let pairH = $state(0);
  const stepOther = (d: number) => (otherAt = (Math.min(otherAt, others.length - 1) + d + others.length) % others.length);
  /** Whether a run's value differs from the baseline's: marked (a dot and the weight); an equal one is tertiary. */
  const offBase = (r: Run, f: (r: Run) => Setting) => r !== base && !!base && settingText(f(r)) !== settingText(f(base));
</script>

{#snippet cell(v: Setting)}{#if v.ident}<Ident value={v.value} />{:else}<span class="val" class:whole={v.value.length <= 14 && !/\s/.test(v.value)} data-fact>{v.value}</span>{/if}{#if v.note}<Note text={v.note} fact />{/if}{#if v.about}<span class="k-sub v-about">{v.about}</span>{/if}{/snippet}


<Panel title="Side by side" meta={sameRows.length ? `${diffRows.length} settings differ` : undefined} pad="none" id="cmp-table">
  {#if pairMode && other}
    <!-- Phones: the baseline and one run beside it; the pager steps through the others. The pager and the pair's
         head row (both names, Baseline, Swap) hold under the top bar while the table is in view, so the names
         never scroll away from the settings being read. -->
    <div class="pair num" bind:offsetHeight={pairH}>
      <button type="button" class="pg" aria-label="Previous run" onclick={() => stepOther(-1)}><ChevronLeft size={16} aria-hidden="true" /></button>
      <span class="pair-what" aria-live="polite">Run {chosen.indexOf(other) + 1} of {chosen.length}<span class="faint">{' '}vs baseline</span></span>
      <button type="button" class="pg" aria-label="Next run" onclick={() => stepOther(1)}><ChevronRight size={16} aria-hidden="true" /></button>
      <button type="button" class="more" onclick={() => (allRuns = true)}>All {chosen.length} runs</button>
    </div>
  {:else if narrow && allRuns && chosen.length > 2}
    <p class="t-range num">
      {#if tableRange}{tableRange.first === tableRange.last ? `Run ${tableRange.first}` : `Runs ${tableRange.first}–${tableRange.last}`} of {chosen.length}<span
          class="sep"
          aria-hidden="true">·</span
        >{/if}<button type="button" class="more" onclick={() => (allRuns = false)}>Two at a time</button>
    </p>
  {:else if tableRange}
    <!-- What the sideways scroll hides: the runs in view, and a way to the others. -->
    <p class="t-range num">
      {tableRange.first === tableRange.last ? `Run ${tableRange.first}` : `Runs ${tableRange.first}–${tableRange.last}`} of {chosen.length}{#if tableRange.last < chosen.length}<span class="sep" aria-hidden="true">·</span><button
          type="button"
          class="more"
          onclick={scrollTableEnd}>Scroll for {chosen.length - tableRange.last} more →</button
        >{/if}
    </p>
  {/if}
  <div class="scroll" class:more={tableMore} class:pairing={pairMode && !!other} style:--pair-h="{pairH}px" bind:this={tableEl} onscroll={measureTable}>
    <!-- Fixed layout: every run column the same width, so each header's key sits over its own values. The widths
         live on the header row (no colgroup, whose keyed columns could outlive a page of runs), so the setting
         column is fixed and the run columns share the rest equally from the first render (two or three runs
         from 1024 px: the setting column takes an equal share too, `shared`). -->
    <table class="table fixed" class:two={pairMode} style:min-width={pairMode ? undefined : `${SETTING_COL + chosen.length * RUN_COL_MIN}px`}>
      <thead>
        <tr>
          <th style:width={shared ? undefined : `${pairMode ? PAIR_SETTING_COL : SETTING_COL}px`}></th>
          {#each shown as r (r.id)}
            <!-- The key in a column of its own beside the name, so a name that wraps (at its hyphens) keeps its
                 left edge and never squeezes the key. -->
            <th>
              <!-- The run's name over its role: the baseline every other column is read against, or Swap to make
                   it the baseline. No key here: the chips above are the legend, and a swatch at the column's
                   edge could not line up with its chip's. -->
              <span class="th"
                ><span class="th-name" class:truncate={!pairMode} title={pairMode ? undefined : r.name}>{r.name}</span
                >{#if chosen.length > 1}{#if r === base}<span class="base-tag">Baseline</span>{:else}<button
                      type="button"
                      class="set-base"
                      aria-label="Make {r.name} the baseline"
                      title="Make this run the baseline"
                      onclick={() => setBaseline(r)}><ArrowLeftRight size={12} aria-hidden="true" />Swap</button
                    >{/if}{/if}</span
              >
            </th>
          {/each}
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="k">Status</td>
          {#each shown as r (r.id)}<td><StatusBadge run={r} plain /></td>{/each}
        </tr>
        <tr>
          <!-- The best epoch's score, which the chips' latest value need not be (a stopped run's 0.595 · stopped
               against its best 0.606): the row says so. -->
          <td class="k">Best mAP50-95<span class="k-sub">any epoch</span></td>
          {#each shown as r (r.id)}
            {@const win = bestOf.get(r.id)?.word}
            <td class="num" class:winner={!!win}>{bestScore(r)}{#if win}{' '}<span class="tag" data-fact>{win}</span>{/if}</td>
          {/each}
        </tr>
        <tr>
          <td class="k">Precision · recall</td>
          {#each shown as r (r.id)}
            {@const v = bestVal(r)}
            <td class="num">
              {#if v}{metric(v.precision)} · {metric(v.recall)}<span class="pr-at"
                  ><Meta><span>{v.best_f1_conf != null ? `conf ${conf(v.best_f1_conf)}` : 'best-F1 conf'}</span><span>ep {r.best?.epoch}</span></Meta></span
                >{:else}—{/if}
            </td>
          {/each}
        </tr>
        {#each diffRows as [k, f] (k)}
          <tr>
            <td class="k">{k}</td>
            {#each shown as r (r.id)}
              <!-- Read against the baseline: a value that differs is marked (a dot and the weight), an equal one
                   recedes to tertiary, so "11 settings differ" shows where they differ. -->
              {@const off = offBase(r, f)}
              <td class="num cfg" class:diff={off} class:same={r !== base && !off}
                class:base={r === base}
                >{#if off}<span class="chg-row"><span class="chg" aria-hidden="true"></span><span
                      >{@render cell(f(r))}<span class="sr-only">, differs from the baseline</span></span
                    ></span
                  >{:else}{@render cell(f(r))}{/if}</td
              >
            {/each}
            </tr>
        {/each}
        {#if sameRows.length}
          <tr class="fold">
            <td colspan={shown.length + 1}>
              <button aria-expanded={showSame} onclick={() => (showSame = !showSame)}>
                {showSame ? 'Hide' : 'Show'} {sameRows.length} identical {sameRows.length === 1 ? 'setting' : 'settings'}
              </button>
            </td>
          </tr>
          {#if showSame}
            {#each sameRows as [k, f] (k)}
              <tr>
                <td class="k">{k}</td>
                {#each shown as r (r.id)}<td class="num cfg">{@render cell(f(r))}</td>{/each}
                    </tr>
            {/each}
          {/if}
        {/if}
      </tbody>
    </table>
  </div>
</Panel>

<style>
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
  /* What a short value means ("Auto": AdamW for short runs, else SGD), under the value it explains. */
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
</style>
