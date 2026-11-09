<script lang="ts" module>
  import { origin } from '$lib/lineage';

  /**
   * The weights a run started from, by name: a model id resolves through the models list, a path shows its file name,
   * none is null ("From scratch"). Fine-tuned vs from scratch often explains most of a gap between two runs.
   */
  export function startFrom(weights: string | null, models: readonly { id: string; name: string }[] | undefined): string | null {
    const o = origin(weights, models ?? []);
    return o.kind === 'scratch' ? null : o.label;
  }
</script>

<script lang="ts">
  import { MediaQuery } from 'svelte/reactivity';
  import ChartSkeleton from '$lib/charts/ChartSkeleton.svelte';
  import { untrack } from 'svelte';
  import GitCompare from '@lucide/svelte/icons/git-compare-arrows';
  import X from '@lucide/svelte/icons/x';
  import ArrowUpRight from '@lucide/svelte/icons/arrow-up-right';
  import Plus from '@lucide/svelte/icons/plus';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Epoch, Run, RunMetrics } from '$lib/api/types';
  import LineChart, { type Series } from '$lib/charts/LineChart.svelte';
  import { niceDomain, unitDomain } from '$lib/charts/domain';
  import { nearest } from '$lib/charts/downsample';
  import { defaultAxis, runEnd, runX } from '$lib/charts/runAxis';
  import { conf, deviceName, duration, metric } from '$lib/format';
  import { batchSetting, lrSetting, optimizerSetting, settingText, warmupSetting, type Setting } from './run/resolved';
  import { href, router } from '$lib/router.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Page from '$lib/shell/Page.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';
  import { media } from '$lib/design/tokens';
  import Ident from '$lib/ui/Ident.svelte';

  setCrumbs({ label: 'Runs', path: '/runs' }, { label: 'Compare' });

  const MAX = 4;
  const ids = $derived((router.route.query.get('ids') ?? '').split(',').filter(Boolean).slice(0, MAX));
  const runs = new Resource((s) => hub.api.runs(s));
  const datasets = new Resource((s) => hub.api.datasets(s));
  // Names for runs fine-tuned from a model ("Start from").
  const models = new Resource((s) => hub.api.models(s));
  const metrics = new Resource(async (s) => {
    const list = ids;
    const out = await Promise.all(list.map((id) => hub.api.runMetrics(id, s).then((m) => [id, m] as const)));
    return new Map<string, RunMetrics>(out);
  });

  const phoneQuery = new MediaQuery(media.phone, false);
  const narrow = $derived(phoneQuery.current);
  let cursor = $state<number | null>(null);

  const chosen = $derived(ids.map((id) => runs.data?.find((r) => r.id === id)).filter((r): r is Run => !!r));
  // Runs planned for very different lengths (30 vs 150 epochs) would crush the short ones into the start of an epoch
  // axis: then the axis defaults to "By progress" (each run over its planned epochs, ../lib/charts/runAxis), and the axis
  // says why. Every run is divided by its plan, so a run that failed or stopped early ends where it ended.
  let xChoice = $state<Axis | null>(null);
  const x = $derived<Axis>(xChoice ?? defaultAxis(chosen));
  const xAuto = $derived(xChoice === null && x === 'progress');
  const X_OPTIONS: { value: Axis; label: string }[] = [
    { value: 'epoch', label: 'By epoch' },
    { value: 'progress', label: 'By progress' },
  ];
  // The normalised axis names itself; chosen for you, it also says why.
  const xLabel = $derived(x === 'progress' ? (xAuto ? '% of planned epochs · plans differ in length' : '% of planned epochs') : undefined);
  let hint = $state<string | null>(null);
  // Colours follow the run, not its position: removing one run never repaints the others.
  let slots = $state<Record<string, number>>({});
  $effect(() => {
    const cur = ids;
    untrack(() => {
      const next: Record<string, number> = {};
      for (const id of cur) if (slots[id] !== undefined) next[id] = slots[id];
      for (const id of cur) {
        if (next[id] !== undefined) continue;
        let k = 0;
        while (Object.values(next).includes(k)) k++;
        next[id] = k;
      }
      slots = next;
    });
  });
  type Axis = 'epoch' | 'progress';
  // Run slots: sky and amber first (the most distinct pair, for the common two-run comparison), then magenta and teal.
  // No four hues stay apart for every colour-vision type, so slots 3 and 4 also get a line pattern.
  const RUN_SLOTS = [1, 2, 3, 4];
  const SLOT_PATTERN = [undefined, undefined, 'dash', 'dot'] as const;
  const slotOf = (id: string) => slots[id] ?? Math.max(0, ids.indexOf(id));
  const color = (id: string) => `var(--chart-${RUN_SLOTS[slotOf(id)] ?? 1})`;

  function setIds(next: string[]) {
    router.go(`/runs/compare${next.length ? `?ids=${next.join(',')}` : ''}`, { replace: true });
  }

  const curve = (f: (e: Epoch) => number) =>
    chosen.map((r, i): Series => {
      const eps = metrics.data?.get(r.id)?.epochs ?? [];
      return {
        id: r.id,
        label: r.name,
        color: color(r.id),
        pattern: SLOT_PATTERN[slotOf(r.id)],
        // Each line ends in its run's status mark (live, stopped, failed), so a line that stops short says why.
        end: runEnd(r.status) ?? undefined,
        points: eps.map((e) => ({ x: runX(e.epoch, r, x), y: f(e) })),
      };
    });

  type MetricId = 'm95' | 'm50' | 'loss' | 'rec';
  const METRICS: { id: MetricId; label: string; f: (e: Epoch) => number; unit?: boolean }[] = [
    { id: 'm95', label: 'mAP50-95', f: (e) => e.val.map50_95, unit: true },
    { id: 'm50', label: 'mAP50', f: (e) => e.val.map50, unit: true },
    { id: 'loss', label: 'Train loss', f: (e) => e.train_loss.total },
    { id: 'rec', label: 'Recall', f: (e) => e.val.recall, unit: true },
  ];
  // One chart, one metric at a time: four small charts repeated the same legend four times.
  let metricId = $state<MetricId>('m95');
  const metricDef = $derived(METRICS.find((m) => m.id === metricId)!);
  const series = $derived(curve(metricDef.f));
  // The run page's main-chart size; phones a shorter plot.
  const height = $derived(narrow ? 240 : 320);
  // Fitted to the runs (./domain): four runs between 0.45 and 0.62 spread over 0–0.7, not bunched in a 0–1 axis.
  const fit = $derived.by(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of series)
      for (const p of s.points) {
        if (!Number.isFinite(p.y)) continue;
        if (p.y < lo) lo = p.y;
        if (p.y > hi) hi = p.y;
      }
    if (!Number.isFinite(lo)) return undefined;
    // LineChart's plot sits inside 8 px top and 22 px bottom margins (36 under an axis name).
    return metricDef.unit ? unitDomain(hi, height - (xLabel ? 44 : 30)) : niceDomain(lo, hi);
  });
  // The chips are the legend: each shows its run's value under the cursor, or its latest value at rest.
  function valueOf(id: string): number | null {
    const pts = series.find((s) => s.id === id)?.points ?? [];
    const first = pts[0];
    const last = pts[pts.length - 1];
    if (!first || !last) return null;
    if (cursor === null) return last.y;
    if (cursor < first.x - 0.5 || cursor > last.x + 0.5) return null;
    return pts[nearest(pts, cursor)]?.y ?? null;
  }
  // Hover, focus or tap a run chip to bring its line forward: 3 px in its hue over a halo, painted last with a bold end
  // label; the others become context curves, 1 px in chart.context (>= 3:1, never faded), while their chips keep their
  // hue. Unpinning restores every hue.
  let hovered = $state<string | null>(null);
  let pinned = $state<string | null>(null);
  // Only runs still in the comparison count (a removed run must not leave every line faded).
  const highlight = $derived([hovered, pinned].find((x) => x !== null && ids.includes(x)) ?? null);
  // At rest the chips read each run's latest value: "Latest · ep 37" when the runs share that epoch, else per run.
  const lastEps = $derived(new Set(chosen.map((r) => metrics.data?.get(r.id)?.epochs.at(-1)?.epoch ?? null)));
  const latestNote = $derived.by(() => {
    const [only] = lastEps;
    return lastEps.size === 1 && only !== null && only !== undefined ? `Latest · ep ${only}` : "Latest · each run's last epoch";
  });
  // Recall is read at each epoch's own best-F1 confidence (docs/API.md, Operating point): the readout says so.
  const readoutAt = $derived(
    (cursor === null ? latestNote : x === 'epoch' ? `Epoch ${Math.round(cursor)}` : `${Math.round(cursor)}% through`) +
      (metricId === 'rec' ? ' · at best-F1 conf' : ''),
  );
  const xDomain = $derived<[number, number]>(x === 'epoch' ? [0, Math.max(1, ...chosen.map((r) => r.epochs))] : [0, 100]);

  // mAP is only comparable on the same data: no winner across datasets.
  const sameData = $derived(new Set(chosen.map((r) => r.config.dataset_id)).size <= 1);
  const bestIdx = $derived(chosen.reduce((b, r, i) => ((r.best?.map50_95 ?? -1) > (chosen[b]?.best?.map50_95 ?? -1) ? i : b), 0));

  const startOf = (r: Run) => startFrom(r.config.weights, models.data);
  const startLine = (r: Run) => {
    const from = startOf(r);
    return from ? `Fine-tuned from ${from}` : 'From scratch';
  };
  const setX = (v: Axis) => (xChoice = v);

  // A cell is a Setting (./run/resolved): identifiers (an architecture, a checkpoint, a named optimizer) in mono like
  // the chips and every other screen (ui/Ident), prose ("From scratch", "auto") in sans. Where the server resolved a
  // run's config (`Run.resolved`), the optimizer, LR and batch are what it trains with, the request a note beside them.
  const plain = (value: string): Setting => ({ value });
  const ident = (value: string): Setting => ({ value, ident: true });
  const rows = $derived<[string, (r: Run) => Setting][]>([
    ['Dataset', (r) => plain(datasets.data?.find((d) => d.id === r.config.dataset_id)?.name ?? r.config.dataset_id)],
    ['Model', (r) => ident(r.config.arch)],
    ['Start from', (r) => (startOf(r) ? ident(startOf(r) ?? '') : plain('From scratch'))],
    ['Epochs', (r) => plain(`${r.epoch}/${r.epochs}`)],
    ['Batch', (r) => batchSetting(r.config, r.resolved)],
    ['Image size', (r) => plain(`${r.config.imgsz} px`)],
    ['Optimizer', (r) => optimizerSetting(r.config, r.resolved)],
    ['LR', (r) => lrSetting(r.config, r.resolved)],
    ['Warmup', (r) => (r.resolved ? warmupSetting(r.resolved) : plain('—'))],
    ['Mosaic / mixup', (r) => plain(`${r.config.augment.mosaic} / ${r.config.augment.mixup}`)],
    ['Close mosaic', (r) => plain(r.config.augment.close_mosaic > 0 ? `last ${r.config.augment.close_mosaic} epochs` : 'never')],
    ['Device', (r) => plain(deviceName(r.device))],
    ['Duration', (r) => plain(r.started ? duration(((r.finished ? Date.parse(r.finished) : Date.now()) - Date.parse(r.started)) / 1000) : '—')],
  ]);
  const differs = (f: (r: Run) => Setting) => new Set(chosen.map((r) => settingText(f(r)))).size > 1;
  /** Precision and recall of the epoch Best names (the run's model), with the threshold they were read at. */
  const bestVal = (r: Run) => (r.best ? metrics.data?.get(r.id)?.epochs.find((e) => e.epoch === r.best?.epoch)?.val : undefined);
  // What differs is the comparison; identical settings fold into one row that can be opened.
  const diffRows = $derived(chosen.length > 1 ? rows.filter(([, f]) => differs(f)) : rows);
  const sameRows = $derived(chosen.length > 1 ? rows.filter(([, f]) => !differs(f)) : []);
  let showSame = $state(false);
  let managing = $state(false);
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

  /** The side-by-side table's setting column, and the narrowest a run column gets before the table scrolls. */
  const SETTING_COL = 152;
  const RUN_COL_MIN = 136;

  const candidates = $derived((runs.data ?? []).filter((r) => !ids.includes(r.id) && r.status !== 'queued'));
</script>

{#snippet lede()}<span>Overlay the curves of up to four runs. Colours follow the run, not its rank.</span>{/snippet}

<!-- Phones lead with the plot: the explanation sits in the disclosure under the legend. -->
<Page title="Compare runs" sub={narrow ? undefined : lede}>
  {#if runs.error && !runs.data}
    <ErrorState error={runs.error} retry={() => runs.refresh()} />
  {:else if ids.length === 0}
    <!-- Nothing to compare yet: the frameless empty state on the page's content edge, not a card stretched across an
         empty canvas (DESIGN_DIRECTION 8). -->
    <EmptyState icon={GitCompare} title="Pick runs to compare" description="Add two to four runs, or select them on the Runs page.">
      {#snippet actions()}
        <!-- One action: pick runs here when there are any, else go and make some. -->
        {#if candidates.length}{@render adder()}{:else}<Button href={href('/runs')}>Go to Runs</Button>{/if}
      {/snippet}
    </EmptyState>

  {:else}
    <section class="compare" aria-label="Curves">
      {#if narrow}
        <!-- Phones: one control row, the runs (a compact legend whose values follow a tap or scrub on the plot), the
             one caveat that changes how to read it, then the plot. The longer explanation sits in the disclosure. -->
        <header class="c-row">
          <Select label="Metric" bind:value={metricId} options={METRICS.map((m) => ({ value: m.id, label: m.label }))} />
          <Segmented size="sm" label="X axis" value={x} onchange={setX} options={X_OPTIONS} />
        </header>
        {@render chips()}
        {#if !sameData && chosen.length > 1}<p class="caveat">Different datasets: scores are not directly comparable</p>{/if}
        {@render plot()}
        <details class="about">
          <summary>
            <ChevronRight size={14} aria-hidden="true" />
            <span>How to read this</span>
          </summary>
          <div class="about-body">
            {#if !sameData && chosen.length > 1}<p>The runs use different datasets: compare the shape of the curves, not their values.</p>{/if}
            {#if x === 'progress'}<p>By progress puts each run over its planned epochs, so a short plan and a long one compare by shape.</p>{/if}
            <p>Each line ends in its run's status: a ringed dot while it trains, a square where it was stopped, a cross
              where it failed.</p>
            <p>Colours follow the run, not its rank. Tap a run to keep its line in front; Edit shows how each run
              started (fine-tuned, or from scratch).</p>
            <p>Tap or drag across the chart to read every run at one point; tap it again or anywhere else to let go.</p>
          </div>
        </details>
      {:else}
        {@render chips()}
        <!-- One line, and only when it changes how to read the plot; the axis names itself (and why it was chosen). -->
        {#if !sameData && chosen.length > 1}<p class="caveat">Different datasets: compare the shape of the curves, not their values.</p>{/if}
        <header class="c-head">
          <Segmented label="Metric" bind:value={metricId} options={METRICS.map((m) => ({ value: m.id, label: m.label }))} />
          <Segmented size="sm" label="X axis" value={x} onchange={setX} options={X_OPTIONS} />
        </header>
        {@render plot()}
      {/if}
    </section>

    {#if chosen.length}
      <Panel title="Side by side" meta={sameRows.length ? `${diffRows.length} settings differ` : undefined} pad="none" id="cmp-table">
        {#if tableRange}
          <!-- What the sideways scroll hides: the runs in view, and a way to the others. -->
          <p class="t-range num">
            {tableRange.first === tableRange.last ? `Run ${tableRange.first}` : `Runs ${tableRange.first}–${tableRange.last}`} of {chosen.length}{#if tableRange.last < chosen.length}<span class="sep" aria-hidden="true">·</span><button
                type="button"
                class="more"
                onclick={scrollTableEnd}>Scroll for {chosen.length - tableRange.last} more →</button
              >{/if}
          </p>
        {/if}
        <div class="scroll" class:more={tableMore} bind:this={tableEl} onscroll={measureTable}>
          <!-- Fixed layout: every run column the same width, so each header's key sits over its own values. -->
          <table class="table fixed" style:min-width="{SETTING_COL + chosen.length * RUN_COL_MIN}px">
            <colgroup>
              <col style:width="{SETTING_COL}px" />
              {#each chosen as r (r.id)}<col />{/each}
            </colgroup>
            <thead>
              <tr>
                <th></th>
                {#each chosen as r (r.id)}
                  <th><span class="th" style:--c={color(r.id)}><i data-pattern={SLOT_PATTERN[slotOf(r.id)]}></i>{r.name}</span></th>
                {/each}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td class="k">Status</td>
                {#each chosen as r (r.id)}<td><StatusBadge status={r.status} plain /></td>{/each}
              </tr>
              <tr>
                <td class="k">Best mAP50-95</td>
                {#each chosen as r, i (r.id)}
                  <td class="num" class:winner={sameData && i === bestIdx && chosen.length > 1}>{metric(r.best?.map50_95)}{#if sameData && i === bestIdx && chosen.length > 1}<span class="tag">best</span>{/if}</td>
                {/each}
              </tr>
              <tr>
                <td class="k">Precision · recall</td>
                {#each chosen as r (r.id)}
                  {@const v = bestVal(r)}
                  <td class="num">
                    {#if v}{metric(v.precision)} · {metric(v.recall)}<span class="pr-at">{v.best_f1_conf != null ? `conf ${conf(v.best_f1_conf)}` : 'best-F1 conf'} · ep {r.best?.epoch}</span>{:else}—{/if}
                  </td>
                {/each}
              </tr>
              {#each diffRows as [k, f] (k)}
                <tr>
                  <td class="k">{k}</td>
                  {#each chosen as r (r.id)}<td class="num cfg diff">{@render cell(f(r))}</td>{/each}
                </tr>
              {/each}
              {#if sameRows.length}
                <tr class="fold">
                  <td colspan={chosen.length + 1}>
                    <button aria-expanded={showSame} onclick={() => (showSame = !showSame)}>
                      {showSame ? 'Hide' : 'Show'} {sameRows.length} identical {sameRows.length === 1 ? 'setting' : 'settings'}
                    </button>
                  </td>
                </tr>
                {#if showSame}
                  {#each sameRows as [k, f] (k)}
                    <tr>
                      <td class="k">{k}</td>
                      {#each chosen as r (r.id)}<td class="num cfg">{@render cell(f(r))}</td>{/each}
                    </tr>
                  {/each}
                {/if}
              {/if}
            </tbody>
          </table>
        </div>
      </Panel>
    {/if}
  {/if}
</Page>

{#snippet cell(v: Setting)}{#if v.ident}<Ident value={v.value} />{:else}{v.value}{/if}{#if v.note}<span class="note">{v.note}</span>{/if}{/snippet}

{#snippet chips()}
  <!-- The run chips are the legend: colour key, value under the cursor, link and remove. -->
  <div class="chips" class:managing role="group" aria-label="Runs being compared">
    <div class="chips-head">
      <!-- The readout row: where the values are read, or how to read them while the plot has keyboard focus or on a
           first touch. -->
      <span class="at num" class:hint={!!hint} aria-hidden="true">{hint ?? readoutAt}</span>
      <!-- Phones: opening and removing runs sit behind one toggle, so the legend stays compact. -->
      <button type="button" class="manage" aria-pressed={managing} onclick={() => (managing = !managing)}>{managing ? 'Done' : 'Edit'}</button>
    </div>
    {#each chosen as r (r.id)}
      {@const v = valueOf(r.id)}
      {@const from = startOf(r)}
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <span
        class="chip"
        class:pinned={pinned === r.id}
        style:--c={color(r.id)}
        onpointerenter={() => (hovered = r.id)}
        onpointerleave={() => (hovered = null)}
        onfocusin={() => (hovered = r.id)}
        onfocusout={() => (hovered = null)}
      >
        <!-- The legend entry (key and name) is the toggle: it keeps the run's line in front. -->
        <button
          type="button"
          class="key"
          aria-pressed={pinned === r.id}
          title={pinned === r.id ? 'Let the line fall back' : 'Keep this line in front'}
          onclick={() => (pinned = pinned === r.id ? null : r.id)}
        ><i aria-hidden="true" data-pattern={SLOT_PATTERN[slotOf(r.id)]}></i><span class="nm">{r.name}</span></button>
        <span class="v num">{v === null ? '—' : metric(v)}</span>
        <!-- How the run started: fine-tuned vs from scratch often explains a gap between two curves. It lies over the
             toggle's lower half (clicks pass through to it) and runs on under the value. Phones show it only in Edit,
             on a full-width line of its own. -->
        <span class="from" class:tuned={!!from} title={startLine(r)}>{#if from}Fine-tuned from <Ident value={from} class="src" />{:else}From scratch{/if}</span>
        <span class="acts">
          <a class="open" href={href(`/runs/${r.id}`)} aria-label="Open {r.name}" title="Open run"><ArrowUpRight size={12} /></a>
          <button class="rm" aria-label="Remove {r.name}" onclick={() => setIds(ids.filter((id) => id !== r.id))}><X size={12} /></button>
        </span>
      </span>
    {/each}
    {#if ids.length < MAX && candidates.length}{@render adder()}{/if}
  </div>
{/snippet}

{#snippet plot()}
  {#if metrics.error}
    <ErrorState error={metrics.error} compact retry={() => metrics.refresh()} title="Could not load these runs' metrics" />
  {:else if !metrics.data || !runs.data}
    <ChartSkeleton {height} legend={false} />
  {:else}
    <LineChart
      {series}
      {height}
      {xDomain}
      xInteger={x === 'epoch'}
      yDomain={fit?.domain}
      yTickValues={fit?.ticks}
      curve="monotone"
      valueFormat={(v) => metric(v)}
      xFormat={(v) => (x === 'epoch' ? String(Math.round(v)) : `${Math.round(v)}%`)}
      xTitle={(v) => (x === 'epoch' ? `Epoch ${Math.round(v)}` : `${v.toFixed(0)}% through`)}
      {xLabel}
      bind:cursor
      bind:hint
      legend={false}
      {highlight}
      recede
      endDots
      endLabels
      label="{metricDef.label} for the compared runs"
    />
  {/if}
{/snippet}

{#snippet adder()}
  <!-- The face is a secondary Button's; the native picker lies over all of it, so any point of it opens the list. -->
  <label class="add">
    <Plus size={14} strokeWidth={1.75} aria-hidden="true" />
    <span aria-hidden="true">Add run…</span>
    <select
      aria-label="Add a run to the comparison"
      onchange={(e) => {
        const v = (e.currentTarget as HTMLSelectElement).value;
        if (v) setIds([...ids, v]);
      }}
    >
      <option value="">Add run…</option>
      {#each candidates as c (c.id)}<option value={c.id}>{c.name} · {c.status}</option>{/each}
    </select>
  </label>
{/snippet}

<style>
  .compare {
    display: grid;
    gap: var(--space-3);
    padding: var(--space-4);
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    min-width: 0;
  }
  /* The legend: one card per run (key, name, value under the cursor; how it started below), side by side. */
  .chips {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(232px, 1fr));
    gap: var(--space-2);
    align-items: stretch;
  }
  /* Laptops and tablets: a 2 × 2 grid, so four runs never leave one card orphaned on a row of its own. */
  @media (min-width: 768px) and (max-width: 1279px) {
    .chips {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
  .chips-head {
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .at {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .chip {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto auto;
    grid-template-rows: auto auto;
    column-gap: var(--space-1);
    min-width: 0;
    padding: var(--space-1) var(--space-1) var(--space-1) var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-panel);
    border: 1px solid var(--line-default);
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      background-color var(--dur-fast) var(--ease-standard);
  }
  .chip:hover {
    border-color: var(--line-strong);
  }
  /* The run kept in front: the one selection rule (bg.active and a 2 px accent bar). */
  .chip.pinned {
    background: var(--bg-active);
  }
  .chip.pinned::before {
    content: '';
    position: absolute;
    left: -1px;
    top: -1px;
    bottom: -1px;
    width: 2px;
    border-radius: var(--radius-md) 0 0 var(--radius-md);
    background: var(--accent-line);
  }
  /* The legend entry is the toggle: key and name, over both lines of the card. */
  .key {
    grid-column: 1;
    grid-row: 1 / 3;
    position: relative;
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
    padding: var(--space-1);
    margin-left: calc(-1 * var(--space-1));
    border-radius: var(--radius-sm);
    min-width: 0;
    text-align: left;
  }
  .key:hover {
    background: var(--bg-hover);
  }
  /* On the name's centre line (20 px line). */
  .key i {
    margin-top: 8px;
  }
  .nm {
    font-weight: var(--weight-medium);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .chip .v {
    grid-column: 2;
    grid-row: 1;
    align-self: start;
    padding: var(--space-1) var(--space-1) 0 0;
    min-width: 3.2em;
    color: var(--text-primary);
    font-weight: var(--weight-medium);
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
  /* Indented under the name, past the key (16 px swatch + 8 px gap); clicks fall through to the toggle. */
  .from {
    grid-column: 1 / 3;
    grid-row: 2;
    align-self: end;
    position: relative;
    min-width: 0;
    padding: 0 var(--space-1) var(--space-1) var(--space-6);
    pointer-events: none;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .from.tuned {
    color: var(--text-secondary);
  }
  .from :global(.ident) {
    font-size: var(--type-sm-size);
    white-space: nowrap;
  }
  .acts {
    grid-column: 3;
    grid-row: 1 / 3;
    display: grid;
    align-content: center;
  }
  .open,
  .chip .rm {
    display: grid;
    place-items: center;
    width: 24px;
    height: 24px;
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
  }
  .open:hover,
  .chip .rm:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  @media (pointer: coarse) {
    .chip .rm {
      position: relative;
    }
    .chip .rm::after {
      content: '';
      position: absolute;
      inset: -8px;
    }
  }
  .manage {
    display: none;
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
  .at.hint {
    color: var(--text-secondary);
  }
  /* The one caveat that changes how to read the plot, always in view, on one line above it. */
  .caveat {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  @media (max-width: 767px) {
    /* Phones: a compact legend above the plot (44 px rows of key, name, value), so every run is named and the plot with
       its axis still sits in the first screen. Edit adds open / remove and, on a full-width line of its own, how each
       run started. */
    .chips {
      grid-template-columns: minmax(0, 1fr);
      gap: 0;
    }
    .chips-head {
      margin-top: calc(-1 * var(--space-2));
    }
    .manage {
      display: inline-flex;
      align-items: center;
      height: 44px;
      padding-left: var(--space-3);
      font-size: var(--type-sm-size);
      color: var(--text-secondary);
    }
    /* Rows bleed into the panel padding, so the selection bar sits at the panel's inner edge. The value is centred
       on the row; the start line stays under the name. */
    .chips .chip {
      margin-inline: calc(-1 * var(--space-2));
      padding: 0 var(--space-2);
      border-width: 0 0 1px;
      border-color: var(--line-subtle);
      border-radius: 0;
      background: none;
    }
    .chips .chip:first-of-type {
      border-top-width: 1px;
    }
    .chips .chip.pinned {
      background: var(--bg-active);
    }
    .chips .chip.pinned::before {
      top: 0;
      bottom: 0;
      left: 0;
      border-radius: 0;
    }
    .chips .chip .key {
      grid-row: 1;
      align-items: center;
      min-height: 44px;
      margin-left: 0;
      padding: 0;
    }
    .chips .chip .key:hover {
      background: none;
    }
    .chips .chip .key i {
      margin-top: 0;
    }
    .chips .chip .v {
      grid-row: 1;
      align-self: center;
      padding-top: 0;
    }
    /* At rest: two name / value pairs per row, each pair a 44 px toggle; Edit goes back to one run per row, with open,
       remove and how each run started. Legend text is 12 px; the value keeps the body size. */
    .chips:not(.managing) {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      column-gap: var(--space-3);
    }
    .chips:not(.managing) .chip {
      margin-inline: 0;
      padding: 0 0 0 var(--space-2);
    }
    .chips:not(.managing) .chip:nth-of-type(2) {
      border-top-width: 1px;
    }
    .chips:not(.managing) .chip .key {
      gap: 6px;
    }
    .chips:not(.managing) .chip .key i {
      width: 12px;
    }
    .chips:not(.managing) .nm {
      font-size: var(--type-sm-size);
    }
    .chips:not(.managing) .chip .v {
      min-width: 2.8em;
      padding-right: 0;
    }
    /* Edit: the start line under the whole row, past the key; the identifier keeps whole and ellipsizes (title has it). */
    .chips .chip .from {
      grid-column: 1 / -1;
      margin-top: calc(-1 * var(--space-2));
      padding: 0 0 var(--space-2) var(--space-6);
    }
    .chips:not(.managing) .chip .from,
    .chips:not(.managing) .chip .acts {
      display: none;
    }
    .chips.managing .chip .acts {
      grid-row: 1;
      grid-template-columns: 44px 44px;
    }
    .chips .chip .open,
    .chips .chip .rm {
      width: 44px;
      height: 44px;
    }
    .chips .chip .rm::after {
      content: none;
    }
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
    .compare {
      padding: var(--space-3);
    }
  }
  /* Phones: metric picker and x axis share one row above the legend. */
  .c-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .c-row > :global(.select) {
    flex: 1;
  }
  .about {
    margin-top: calc(-1 * var(--space-2));
    border-top: 1px solid var(--line-subtle);
  }
  .about summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-height: 44px;
    list-style: none;
    cursor: pointer;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .about summary::-webkit-details-marker {
    display: none;
  }
  .about summary :global(svg) {
    flex-shrink: 0;
    color: var(--text-tertiary);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .about[open] summary :global(svg) {
    transform: rotate(90deg);
  }
  .about summary:focus-visible {
    outline-offset: -2px;
  }
  .about-body {
    display: grid;
    gap: var(--space-2);
    padding-bottom: var(--space-1);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .c-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    flex-wrap: wrap;
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
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
  /* Adding a run: a native picker (the platform list is the most accessible) dressed as the standard secondary Button
     (ui/Button): raised fill, default edge, 28 px, the icon and a medium label. */
  .add {
    position: relative;
    display: inline-flex;
    align-items: center;
    align-self: center;
    justify-self: start;
    gap: var(--space-1-5);
    height: var(--control-md);
    padding: 0 var(--space-3);
    border-radius: var(--radius-md);
    border: 1px solid var(--line-default);
    background: var(--bg-raised);
    box-shadow: var(--shadow-sm);
    color: var(--text-primary);
    font-weight: var(--weight-medium);
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      border-color var(--dur-fast) var(--ease-standard);
  }
  .add:hover {
    border-color: var(--line-strong);
    background: color-mix(in srgb, var(--bg-raised), var(--text-primary) 4%);
  }
  .add select {
    position: absolute;
    inset: 0;
    width: 100%;
    opacity: 0;
    cursor: pointer;
  }
  @media (pointer: coarse) {
    .add {
      min-height: 44px;
    }
  }
  .add:has(select:focus-visible) {
    outline: 2px solid var(--focus);
    outline-offset: 2px;
  }
  .add select:focus-visible {
    outline: none;
  }
  .add option {
    background: var(--bg-overlay);
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
  .th {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--text-primary);
    font-size: var(--type-sm-size);
  }
  /* Same patterns as the lines, so a run's key matches its curve without relying on colour. */
  .chip i,
  .th i {
    flex-shrink: 0;
    width: 16px;
    height: 3px;
    border-radius: 2px;
    background: var(--c);
  }
  i[data-pattern='dash'] {
    background: repeating-linear-gradient(90deg, var(--c) 0 5px, transparent 5px 7px);
    border-radius: 0;
  }
  i[data-pattern='dot'] {
    background: repeating-linear-gradient(90deg, var(--c) 0 2px, transparent 2px 4px);
    border-radius: 0;
  }
  .k {
    color: var(--text-tertiary);
  }
  td.diff {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  td.cfg:not(.diff) {
    color: var(--text-tertiary);
  }
  .winner {
    color: var(--accent-text);
    font-weight: var(--weight-medium);
  }
  .tag {
    margin-left: 6px;
    padding: 1px 6px;
    border-radius: var(--radius-xs);
    background: var(--accent-muted);
    font-size: var(--type-sm-size);
  }
  /* The request beside what the run resolved it to ("(auto)", "(requested 0.01)"), and where P · R were read. */
  .note {
    margin-left: var(--space-1);
    /* It wraps as a whole, never inside "(requested 0.01)". */
    white-space: nowrap;
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .pr-at {
    display: block;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
</style>
