<script lang="ts">
  import { untrack } from 'svelte';
  import GitCompare from '@lucide/svelte/icons/git-compare-arrows';
  import X from '@lucide/svelte/icons/x';
  import Plus from '@lucide/svelte/icons/plus';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Epoch, Run, RunMetrics } from '$lib/api/types';
  import LineChart, { type Series } from '$lib/charts/LineChart.svelte';
  import { nearest } from '$lib/charts/downsample';
  import { duration, metric } from '$lib/format';
  import { href, router } from '$lib/router.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Page from '$lib/shell/Page.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
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
        points: eps.map((e) => ({ x: x === 'epoch' ? e.epoch + 1 : ((e.epoch + 1) / r.epochs) * 100, y: f(e) })),
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
    ['Image size', (r) => String(r.config.imgsz)],
    ['Optimizer', (r) => r.config.optimizer],
    ['LR', (r) => String(r.config.lr0)],
    ['Mosaic / mixup', (r) => `${r.config.augment.mosaic} / ${r.config.augment.mixup}`],
    ['Device', (r) => r.device.replace('NVIDIA GeForce ', '')],
    ['Duration', (r) => (r.started ? duration(((r.finished ? Date.parse(r.finished) : Date.now()) - Date.parse(r.started)) / 1000) : '—')],
  ]);
  const differs = (f: (r: Run) => string) => new Set(chosen.map(f)).size > 1;
  // What differs is the comparison; identical settings fold into one row that can be opened.
  const diffRows = $derived(chosen.length > 1 ? rows.filter(([, f]) => differs(f)) : rows);
  const sameRows = $derived(chosen.length > 1 ? rows.filter(([, f]) => !differs(f)) : []);
  let showSame = $state(false);

  const candidates = $derived((runs.data ?? []).filter((r) => !ids.includes(r.id) && r.status !== 'queued'));
</script>

<Page title="Compare runs">
  {#snippet sub()}<span>Overlay the curves of up to four runs. Colours follow the run, not its rank.</span>{/snippet}

  {#if runs.error && !runs.data}
    <ErrorState error={runs.error} retry={() => runs.refresh()} />
  {:else if ids.length === 0}
    <Panel>
      <EmptyState icon={GitCompare} title="Pick runs to compare" description="Add two to four runs, or select them on the Runs page.">
        {#snippet actions()}
          {#if candidates.length}{@render adder()}{/if}
          <Button href={href('/runs')}>Go to Runs</Button>
        {/snippet}
      </EmptyState>
    </Panel>
  {:else}
    <section class="compare" aria-label="Curves">
      <!-- The run chips are the legend: colour key, value under the cursor, link and remove. -->
      <div class="chips" role="group" aria-label="Runs being compared">
        <span class="at num" aria-hidden="true">{readoutAt}</span>
        {#each chosen as r (r.id)}
          {@const v = valueOf(r.id)}
          <span class="chip" style:--c={color(r.id)}>
            <i aria-hidden="true" data-pattern={SLOT_PATTERN[slotOf(r.id)]}></i>
            <a href={href(`/runs/${r.id}`)}>{r.name}</a>
            <span class="v num">{v === null ? '—' : v.toFixed(3)}</span>
            <button aria-label="Remove {r.name}" onclick={() => setIds(ids.filter((id) => id !== r.id))}><X size={12} /></button>
          </span>
        {/each}
        {#if ids.length < MAX && candidates.length}{@render adder()}{/if}
      </div>
      {#if !sameData && chosen.length > 1}
        <p class="note">Different datasets: compare the shape of the curves, not their values.</p>
      {/if}
      <header class="c-head">
        <Segmented label="Metric" bind:value={metricId} options={METRICS.map((m) => ({ value: m.id, label: m.label }))} />
        <Segmented size="sm" label="X axis" bind:value={x} options={[{ value: 'epoch', label: 'By epoch' }, { value: 'progress', label: '% of run' }]} />
      </header>
      {#if metrics.error}
        <ErrorState error={metrics.error} compact retry={() => metrics.refresh()} title="Could not load these runs' metrics" />
      {:else if !metrics.data || !runs.data}
        <Skeleton h="{narrow ? 240 : 340}px" r="var(--radius-md)" />
      {:else}
        <LineChart
          {series}
          height={narrow ? 240 : 340}
          {xDomain}
          yDomain={metricDef.y}
          curve="monotone"
          valueFormat={(v) => v.toFixed(3)}
          xFormat={(v) => (x === 'epoch' ? String(Math.round(v)) : `${Math.round(v)}%`)}
          xTitle={(v) => (x === 'epoch' ? `Epoch ${Math.round(v)}` : `${v.toFixed(0)}% through`)}
          xLabel={x === 'epoch' ? 'epoch' : undefined}
          bind:cursor
          legend={false}
          label="{metricDef.label} for the compared runs"
        />
      {/if}
    </section>

    {#if chosen.length}
      <Panel title="Side by side" meta={sameRows.length ? `${diffRows.length} settings differ` : undefined} pad="none" id="cmp-table">
        <div class="scroll">
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
                {#each chosen as r (r.id)}<td><StatusBadge status={r.status} size="sm" /></td>{/each}
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
  @media (max-width: 767px) {
    .compare {
      padding: var(--space-3);
    }
    .at {
      flex-basis: 100%;
    }
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
    padding: 0 4px 0 10px;
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
  .chip button {
    display: grid;
    place-items: center;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    color: var(--text-tertiary);
  }
  @media (pointer: coarse) {
    .chip button {
      position: relative;
    }
    .chip button::after {
      content: '';
      position: absolute;
      inset: -12px;
    }
  }
  .chip button:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .add {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: var(--control-md);
    padding: 0 10px;
    border-radius: var(--radius-full);
    border: 1px dashed var(--line-strong);
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
    padding: 1px 5px;
    border-radius: var(--radius-xs);
    background: var(--accent-muted);
    font-size: var(--type-micro-size);
  }
</style>
