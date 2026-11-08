<script lang="ts">
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
  import { nearest } from '$lib/charts/downsample';
  import { deviceName, duration, metric } from '$lib/format';
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

  setCrumbs({ label: 'Runs', path: '/runs' }, { label: 'Compare' });

  const MAX = 4;
  const ids = $derived((router.route.query.get('ids') ?? '').split(',').filter(Boolean).slice(0, MAX));
  const runs = new Resource((s) => hub.api.runs(s));
  const datasets = new Resource((s) => hub.api.datasets(s));
  const metrics = new Resource(async (s) => {
    const list = ids;
    const out = await Promise.all(list.map((id) => hub.api.runMetrics(id, s).then((m) => [id, m] as const)));
    return new Map<string, RunMetrics>(out);
  });

  let x = $state<'epoch' | 'progress'>('epoch');
  let narrow = $state(typeof matchMedia !== 'undefined' && matchMedia('(max-width: 767px)').matches);
  $effect(() => {
    const mq = matchMedia('(max-width: 767px)');
    const on = () => (narrow = mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  });
  let cursor = $state<number | null>(null);

  const chosen = $derived(ids.map((id) => runs.data?.find((r) => r.id === id)).filter((r): r is Run => !!r));
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
        points: eps.map((e) => ({ x: x === 'epoch' ? e.epoch : (e.epoch / r.epochs) * 100, y: f(e) })),
      };
    });

  type MetricId = 'm95' | 'm50' | 'loss' | 'rec';
  const METRICS: { id: MetricId; label: string; f: (e: Epoch) => number; y?: [number, number] }[] = [
    { id: 'm95', label: 'mAP50-95', f: (e) => e.val.map50_95, y: [0, 1] },
    { id: 'm50', label: 'mAP50', f: (e) => e.val.map50, y: [0, 1] },
    { id: 'loss', label: 'Train loss', f: (e) => e.train_loss.total },
    { id: 'rec', label: 'Recall', f: (e) => e.val.recall, y: [0, 1] },
  ];
  // One chart, one metric at a time: four small charts repeated the same legend four times.
  let metricId = $state<MetricId>('m95');
  const metricDef = $derived(METRICS.find((m) => m.id === metricId)!);
  const series = $derived(curve(metricDef.f));
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
  // Hover, focus or tap a run chip to bring its line forward (the others fade).
  let hovered = $state<string | null>(null);
  let pinned = $state<string | null>(null);
  // Only runs still in the comparison count (a removed run must not leave every line faded).
  const highlight = $derived([hovered, pinned].find((x) => x !== null && ids.includes(x)) ?? null);
  const readoutAt = $derived(cursor === null ? 'Latest' : x === 'epoch' ? `Epoch ${Math.round(cursor)}` : `${Math.round(cursor)}% through`);
  const xDomain = $derived<[number, number]>(x === 'epoch' ? [0, Math.max(1, ...chosen.map((r) => r.epochs))] : [0, 100]);

  // mAP is only comparable on the same data: no winner across datasets.
  const sameData = $derived(new Set(chosen.map((r) => r.config.dataset_id)).size <= 1);
  const bestIdx = $derived(chosen.reduce((b, r, i) => ((r.best?.map50_95 ?? -1) > (chosen[b]?.best?.map50_95 ?? -1) ? i : b), 0));

  const rows = $derived<[string, (r: Run) => string][]>([
    ['Dataset', (r) => datasets.data?.find((d) => d.id === r.config.dataset_id)?.name ?? r.config.dataset_id],
    ['Model', (r) => r.config.arch],
    ['Epochs', (r) => `${r.epoch}/${r.epochs}`],
    ['Batch', (r) => String(r.config.batch)],
    ['Image size', (r) => `${r.config.imgsz} px`],
    ['Optimizer', (r) => r.config.optimizer],
    ['LR', (r) => String(r.config.lr0)],
    ['Mosaic / mixup', (r) => `${r.config.augment.mosaic} / ${r.config.augment.mixup}`],
    ['Device', (r) => deviceName(r.device)],
    ['Duration', (r) => (r.started ? duration(((r.finished ? Date.parse(r.finished) : Date.now()) - Date.parse(r.started)) / 1000) : '—')],
  ]);
  const differs = (f: (r: Run) => string) => new Set(chosen.map(f)).size > 1;
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

  const candidates = $derived((runs.data ?? []).filter((r) => !ids.includes(r.id) && r.status !== 'queued'));
</script>

{#snippet lede()}<span>Overlay the curves of up to four runs. Colours follow the run, not its rank.</span>{/snippet}

<!-- Phones lead with the plot: the explanation sits in the disclosure under the legend. -->
<Page title="Compare runs" sub={narrow ? undefined : lede}>
  {#if runs.error && !runs.data}
    <ErrorState error={runs.error} retry={() => runs.refresh()} />
  {:else if ids.length === 0}
    <Panel>
      <EmptyState icon={GitCompare} title="Pick runs to compare" description="Add two to four runs, or select them on the Runs page.">
        {#snippet actions()}
          <!-- One action: pick runs here when there are any, else go and make some. -->
          {#if candidates.length}{@render adder()}{:else}<Button href={href('/runs')}>Go to Runs</Button>{/if}
        {/snippet}
      </EmptyState>
    </Panel>
  {:else}
    <section class="compare" aria-label="Curves">
      {#if narrow}
        <!-- Phones: one control row, the plot, then the legend (its readout follows a tap or scrub on the plot). -->
        <header class="c-row">
          <Select label="Metric" bind:value={metricId} options={METRICS.map((m) => ({ value: m.id, label: m.label }))} />
          <Segmented size="sm" label="X axis" bind:value={x} options={[{ value: 'epoch', label: 'By epoch' }, { value: 'progress', label: '% of run' }]} />
        </header>
        {@render plot()}
        {@render chips()}
        <details class="about">
          <summary>
            <ChevronRight size={14} aria-hidden="true" />
            <span>{!sameData && chosen.length > 1 ? 'Different datasets: how to read this' : 'How to read this'}</span>
          </summary>
          <div class="about-body">
            {#if !sameData && chosen.length > 1}<p>The runs use different datasets: compare the shape of the curves, not their values.</p>{/if}
            <p>Colours follow the run, not its rank. Tap a run to keep its line in front.</p>
            <p>Tap or drag across the chart to read every run at one point; tap it again or anywhere else to let go.</p>
          </div>
        </details>
      {:else}
        {@render chips()}
        <p class="note">
          Press a run to keep its line in front.{#if !sameData && chosen.length > 1}{' '}Different datasets: compare the shape of the
            curves, not their values.{/if}
        </p>
        <header class="c-head">
          <Segmented label="Metric" bind:value={metricId} options={METRICS.map((m) => ({ value: m.id, label: m.label }))} />
          <Segmented size="sm" label="X axis" bind:value={x} options={[{ value: 'epoch', label: 'By epoch' }, { value: 'progress', label: '% of run' }]} />
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
          <table class="table">
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
              {#each diffRows as [k, f] (k)}
                <tr>
                  <td class="k">{k}</td>
                  {#each chosen as r (r.id)}<td class="num cfg diff">{f(r)}</td>{/each}
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
                      {#each chosen as r (r.id)}<td class="num cfg">{f(r)}</td>{/each}
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

{#snippet chips()}
  <!-- The run chips are the legend: colour key, value under the cursor, link and remove. -->
  <div class="chips" class:managing role="group" aria-label="Runs being compared">
    <span class="at num" aria-hidden="true">{readoutAt}</span>
    <!-- Phones: opening and removing runs sit behind one toggle, so the legend stays compact. -->
    <button type="button" class="manage" aria-pressed={managing} onclick={() => (managing = !managing)}>{managing ? 'Done' : 'Edit'}</button>
    {#each chosen as r (r.id)}
      {@const v = valueOf(r.id)}
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
        <a class="open" href={href(`/runs/${r.id}`)} aria-label="Open {r.name}" title="Open run"><ArrowUpRight size={12} /></a>
        <button class="rm" aria-label="Remove {r.name}" onclick={() => setIds(ids.filter((id) => id !== r.id))}><X size={12} /></button>
      </span>
    {/each}
    {#if ids.length < MAX && candidates.length}{@render adder()}{/if}
  </div>
{/snippet}

{#snippet plot()}
  {#if metrics.error}
    <ErrorState error={metrics.error} compact retry={() => metrics.refresh()} title="Could not load these runs' metrics" />
  {:else if !metrics.data || !runs.data}
    <ChartSkeleton height={narrow ? 240 : 340} legend={false} />
  {:else}
    <LineChart
      {series}
      height={narrow ? 240 : 340}
      {xDomain}
      xInteger={x === 'epoch'}
      yDomain={metricDef.y}
      curve="monotone"
      valueFormat={(v) => metric(v)}
      xFormat={(v) => (x === 'epoch' ? String(Math.round(v)) : `${Math.round(v)}%`)}
      xTitle={(v) => (x === 'epoch' ? `Epoch ${Math.round(v)}` : `${v.toFixed(0)}% through`)}
      bind:cursor
      legend={false}
      {highlight}
      endDots
      endLabels
      label="{metricDef.label} for the compared runs"
    />
  {/if}
{/snippet}

{#snippet adder()}
  <label class="add">
    <Plus size={14} />
    <span class="sr-only">Add a run</span>
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
    border: 1px solid var(--line-default);
    border-radius: var(--radius-lg);
    min-width: 0;
  }
  :global([data-theme='dark']) .compare {
    border-color: var(--line-subtle);
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    align-items: center;
  }
  .at {
    min-width: 6.5em;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .chip {
    transition: border-color var(--dur-fast) var(--ease-standard);
  }
  .key {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    height: 24px;
    padding: 0 var(--space-1);
    margin-left: calc(-1 * var(--space-1));
    border-radius: var(--radius-sm);
    font-weight: var(--weight-medium);
    min-width: 0;
  }
  .nm {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .open {
    display: grid;
    place-items: center;
    width: 20px;
    height: 20px;
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
  }
  .open:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .key:hover {
    background: var(--bg-hover);
  }
  .key[aria-pressed='true'] {
    background: var(--bg-active);
  }
  @media (pointer: coarse) {
    .key::after {
      content: '';
      position: absolute;
      inset: -12px;
    }
  }
  .chip:hover,
  .chip.pinned {
    border-color: var(--line-strong);
  }
  .chip .v {
    min-width: 3.2em;
    color: var(--text-primary);
    font-weight: var(--weight-medium);
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
  .note {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
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
  @media (max-width: 767px) {
    /* Phones: the legend is one column of full-width rows (key, name, value), so every run's whole name shows;
       open / remove appear with Edit. */
    .chips {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 0 var(--space-3);
    }
    .chips .chip {
      grid-column: 1 / -1;
    }
    .chips .at {
      grid-column: 1;
      align-self: center;
    }
    .manage {
      display: inline-flex;
      grid-column: 2;
      justify-self: end;
      align-items: center;
      height: 44px;
      font-size: var(--type-sm-size);
      color: var(--text-secondary);
    }
    .chips .chip {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: center;
      border-radius: 0;
      border-width: 0 0 1px;
      background: none;
      padding: 0;
      min-height: 44px;
    }
    .chips:not(.managing) .chip .open,
    .chips:not(.managing) .chip .rm {
      display: none;
    }
    .chips.managing {
      grid-template-columns: minmax(0, 1fr);
    }
    .chips.managing .chip {
      grid-template-columns: minmax(0, 1fr) auto 44px 44px;
      min-height: 48px;
    }
    .chips.managing .manage {
      grid-column: 1;
      grid-row: 1;
    }
    .chips .chip .key {
      min-height: 44px;
      height: auto;
      margin-left: 0;
      padding: var(--space-1) 0;
    }
    /* A name too long for the row wraps instead of ending in an ellipsis: compared runs often differ only in the
       suffix (shapes-hard-s / -m), which must stay visible. */
    .chips .chip .key {
      text-align: left;
    }
    .chips .chip .nm {
      white-space: normal;
      overflow-wrap: anywhere;
      display: -webkit-box;
      -webkit-box-orient: vertical;
      -webkit-line-clamp: 2;
      line-clamp: 2;
      line-height: var(--type-sm-line);
    }
    .chips .chip .open,
    .chips .chip .rm {
      width: 44px;
      height: 44px;
    }
    .chips .chip .rm::after,
    .chips .chip .key::after {
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
    .at {
      flex-basis: 100%;
    }
  }
  /* Phones: metric picker and x axis share one row above the plot. */
  .c-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .c-row > :global(.select) {
    flex: 1;
  }
  /* Directly under the legend rows, whose last hairline closes the legend. */
  .about {
    margin-top: calc(-1 * var(--space-2));
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
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    height: var(--control-md);
    padding: 0 4px 0 8px;
    border-radius: var(--radius-full);
    background: var(--bg-panel);
    border: 1px solid var(--line-default);
    font-weight: var(--weight-medium);
  }
  .chip i {
    width: 10px;
    height: 3px;
    border-radius: 2px;
    background: var(--c);
  }
  .chip a:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
  }
  .chip .rm {
    display: grid;
    place-items: center;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    color: var(--text-tertiary);
  }
  @media (pointer: coarse) {
    .chip .rm {
      position: relative;
    }
    .chip .rm::after {
      content: '';
      position: absolute;
      inset: -12px;
    }
  }
  .chip .rm:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .add {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: var(--control-md);
    padding: 0 12px;
    border-radius: var(--radius-full);
    border: 1px solid var(--line-default);
    color: var(--text-secondary);
  }
  .add select {
    appearance: none;
    background: none;
    border: 0;
    color: inherit;
    cursor: pointer;
    padding-right: 4px;
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
  .th {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--text-primary);
    font-size: var(--type-sm-size);
  }
  .th i {
    width: 10px;
    height: 3px;
    border-radius: 2px;
    background: var(--c);
  }
  /* Same patterns as the lines, so a run's key matches its curve without relying on colour. */
  .chip i,
  .th i {
    width: 14px;
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
    font-size: var(--type-micro-size);
  }
</style>
