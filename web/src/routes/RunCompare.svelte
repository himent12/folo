<script lang="ts">
  import GitCompare from '@lucide/svelte/icons/git-compare-arrows';
  import X from '@lucide/svelte/icons/x';
  import Plus from '@lucide/svelte/icons/plus';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Epoch, Run, RunMetrics } from '$lib/api/types';
  import LineChart, { type Series } from '$lib/charts/LineChart.svelte';
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
  let cursor = $state<number | null>(null);

  const chosen = $derived(ids.map((id) => runs.data?.find((r) => r.id === id)).filter((r): r is Run => !!r));
  const color = (i: number) => `var(--chart-${i + 1})`;

  function setIds(next: string[]) {
    router.go(`/runs/compare${next.length ? `?ids=${next.join(',')}` : ''}`, { replace: true });
  }

  const curve = (f: (e: Epoch) => number) =>
    chosen.map((r, i): Series => {
      const eps = metrics.data?.get(r.id)?.epochs ?? [];
      return {
        id: r.id,
        label: r.name,
        color: color(i),
        points: eps.map((e) => ({ x: x === 'epoch' ? e.epoch + 1 : ((e.epoch + 1) / r.epochs) * 100, y: f(e) })),
      };
    });

  const charts = $derived([
    { id: 'm95', title: 'mAP50-95', series: curve((e) => e.val.map50_95), y: [0, 1] as [number, number] },
    { id: 'm50', title: 'mAP50', series: curve((e) => e.val.map50), y: [0, 1] as [number, number] },
    { id: 'loss', title: 'Train loss', series: curve((e) => e.train_loss.total), y: undefined },
    { id: 'rec', title: 'Recall', series: curve((e) => e.val.recall), y: [0, 1] as [number, number] },
  ]);
  const xDomain = $derived<[number, number]>(x === 'epoch' ? [0, Math.max(1, ...chosen.map((r) => r.epochs))] : [0, 100]);

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

  const candidates = $derived((runs.data ?? []).filter((r) => !ids.includes(r.id) && r.status !== 'queued'));
</script>

<Page title="Compare runs">
  {#snippet sub()}<span>Overlay the curves of up to four runs. Colours follow the run, not its rank.</span>{/snippet}
  {#snippet actions()}
    <Segmented label="X axis" bind:value={x} options={[{ value: 'epoch', label: 'Epoch' }, { value: 'progress', label: '% of run' }]} />
  {/snippet}

  <div class="picker" role="group" aria-label="Runs being compared">
    {#each chosen as r, i (r.id)}
      <span class="chip" style:--c={color(i)}>
        <i aria-hidden="true"></i>
        <a href={href(`/runs/${r.id}`)}>{r.name}</a>
        <button aria-label="Remove {r.name}" onclick={() => setIds(ids.filter((id) => id !== r.id))}><X size={12} /></button>
      </span>
    {/each}
    {#if ids.length < MAX && candidates.length}
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
    {/if}
  </div>

  {#if runs.error && !runs.data}
    <ErrorState error={runs.error} retry={() => runs.refresh()} />
  {:else if ids.length === 0}
    <Panel>
      <EmptyState icon={GitCompare} title="Pick runs to compare" description="Add two to four runs above, or select them on the Runs page.">
        {#snippet actions()}<Button href={href('/runs')}>Go to Runs</Button>{/snippet}
      </EmptyState>
    </Panel>
  {:else}
    <div class="charts">
      {#each charts as c (c.id)}
        <Panel title={c.title} id="cmp-{c.id}">
          {#if metrics.error}
            <ErrorState error={metrics.error} compact retry={() => metrics.refresh()} title="Could not load these runs' metrics" />
          {:else if !metrics.data || !runs.data}
            <Skeleton h="200px" r="var(--radius-md)" />
          {:else}
            <LineChart
              series={c.series}
              height={200}
              {xDomain}
              yDomain={c.y}
              curve="monotone"
              yFormat={(v) => v.toFixed(1)}
              valueFormat={(v) => v.toFixed(3)}
              xFormat={(v) => (x === 'epoch' ? String(Math.round(v)) : `${Math.round(v)}%`)}
              xTitle={(v) => (x === 'epoch' ? `Epoch ${Math.round(v)}` : `${v.toFixed(0)}% through`)}
              bind:cursor
              label="{c.title} for the compared runs"
            />
          {/if}
        </Panel>
      {/each}
    </div>

    {#if chosen.length}
      <Panel title="Side by side" pad="none" id="cmp-table">
        <div class="scroll">
          <table class="table">
            <thead>
              <tr>
                <th></th>
                {#each chosen as r, i (r.id)}
                  <th><span class="th"><i style:background={color(i)}></i>{r.name}</span></th>
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
                  <td class="num" class:winner={i === bestIdx && chosen.length > 1}>{metric(r.best?.map50_95)}{#if i === bestIdx && chosen.length > 1}<span class="tag">best</span>{/if}</td>
                {/each}
              </tr>
              {#each rows as [k, f] (k)}
                {@const diff = differs(f)}
                <tr>
                  <td class="k">{k}{#if diff}<span class="dot" title="Differs between runs"></span>{/if}</td>
                  {#each chosen as r (r.id)}<td class="mono" class:diff>{f(r)}</td>{/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </Panel>
    {/if}
  {/if}
</Page>

<style>
  .picker {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    align-items: center;
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
  .charts {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-4);
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
  }
  .k {
    color: var(--text-tertiary);
  }
  .dot {
    display: inline-block;
    width: 5px;
    height: 5px;
    margin-left: 6px;
    border-radius: 50%;
    background: var(--warning-solid);
    vertical-align: middle;
  }
  td.diff {
    color: var(--text-primary);
  }
  td.mono:not(.diff) {
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
  @media (max-width: 1023px) {
    .charts {
      grid-template-columns: minmax(0, 1fr);
    }
  }
</style>
