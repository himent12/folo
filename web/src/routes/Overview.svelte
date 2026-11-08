<script lang="ts">
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import Activity from '@lucide/svelte/icons/activity';
  import Box from '@lucide/svelte/icons/box';
  import Images from '@lucide/svelte/icons/images';
  import Plus from '@lucide/svelte/icons/plus';
  import ScanSearch from '@lucide/svelte/icons/scan-search';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Dataset } from '$lib/api/types';
  import RunSpark from '$lib/charts/RunSpark.svelte';
  import ClassStrip from '$lib/detect/ClassStrip.svelte';
  import { ago, bytes, compact, metric } from '$lib/format';
  import { prefs } from '$lib/prefs.svelte';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Badge from '$lib/ui/Badge.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Progress from '$lib/ui/Progress.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';
  import LiveRunCard from './overview/LiveRunCard.svelte';
  import SystemCard from './overview/SystemCard.svelte';

  setCrumbs({ label: 'Overview' });

  const runs = new Resource((s) => hub.api.runs(s), { poll: 5000 });
  const models = new Resource((s) => hub.api.models(s));
  // The list endpoint has no stats; fetch each dataset for its class balance.
  const datasets = new Resource(async (s) => {
    const list = await hub.api.datasets(s);
    return Promise.all(list.map((d) => hub.api.dataset(d.id, s).catch(() => d)));
  });

  const live = $derived(runs.data?.find((r) => r.status === 'running' || r.status === 'stopping'));
  const recent = $derived((runs.data ?? []).filter((r) => r.id !== live?.id).slice(0, 6));
  const topModels = $derived(
    [...(models.data ?? [])].sort((a, b) => (b.metrics?.map50_95 ?? 0) - (a.metrics?.map50_95 ?? 0)).slice(0, 5),
  );
  const dsById = $derived(new Map<string, Dataset>((datasets.data ?? []).map((d) => [d.id, d])));
  const total = (d: Dataset) => d.splits.train + d.splits.val + d.splits.test;
</script>

<Page title="Overview">
  {#snippet sub()}
    <span>
      {#if hub.mode === 'live'}Connected to <span class="mono">folo serve {hub.version}</span>{:else if hub.mode === 'mock'}Showing built-in demo data{:else}Connecting…{/if}
    </span>
  {/snippet}
  {#snippet actions()}
    <Button icon={Images} href={href('/datasets')}>Datasets</Button>
    <Button icon={ScanSearch} href={href('/playground')}>Playground</Button>
  {/snippet}

  <div class="top">
    <div class="live">
      {#if runs.error && !runs.data}
        <Panel title="Training"><ErrorState error={runs.error} retry={() => runs.refresh()} compact /></Panel>
      {:else if !runs.data}
        <div class="live-sk"><Skeleton h="260px" r="var(--radius-lg)" /></div>
      {:else if live}
        {#key live.id}<LiveRunCard run={live} dataset={dsById.get(live.config.dataset_id)} />{/key}
      {:else}
        <Panel title="Training">
          <EmptyState
            icon={Activity}
            title="Nothing is training"
            description={runs.data.length ? 'Start a run to watch its loss, mAP and samples update live here.' : 'Train your first detector: pick a dataset, a model size and go.'}
          >
            {#snippet actions()}<Button variant="primary" icon={Plus} href={href('/runs/new')}>New run</Button>{/snippet}
          </EmptyState>
        </Panel>
      {/if}
    </div>
    <SystemCard />
  </div>

  <div class="mid">
    <Panel title="Recent runs" meta={runs.data ? String(runs.data.length) : undefined} pad="none" id="recent-runs">
      {#snippet actions()}<Button size="sm" variant="ghost" iconRight={ArrowRight} href={href('/runs')}>All runs</Button>{/snippet}
      {#if runs.error && !runs.data}
        <ErrorState error={runs.error} retry={() => runs.refresh()} />
      {:else if !runs.data}
        <div class="list-sk">{#each [0, 1, 2, 3] as i (i)}<Skeleton h="20px" />{/each}</div>
      {:else if recent.length === 0}
        <EmptyState compact icon={Activity} title="No runs yet" description="Runs you start appear here with their progress and best mAP." />
      {:else}
        <div class="scroll">
          <table class="table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Status</th>
                <th class="hide-sm">Progress</th>
                <th class="r">mAP50-95</th>
                <th class="hide-sm">Trend</th>
                <th class="r hide-sm">Created</th>
              </tr>
            </thead>
            <tbody>
              {#each recent as r (r.id)}
                <tr class="link" onclick={() => router.go(`/runs/${r.id}`)}>
                  <td>
                    <a class="run-name" href={href(`/runs/${r.id}`)} onclick={(e) => e.stopPropagation()}>{r.name}</a>
                    <span class="run-sub">{dsById.get(r.config.dataset_id)?.name ?? r.config.dataset_id} · {r.config.arch}</span>
                  </td>
                  <td><StatusBadge status={r.status} size="sm" /></td>
                  <td class="hide-sm">
                    <div class="prog">
                      <Progress value={r.epoch / r.epochs} size="xs" tone={r.status === 'failed' ? 'danger' : r.status === 'completed' ? 'success' : r.status === 'running' ? 'accent' : 'neutral'} label="{r.name} progress" />
                      <span class="num faint">{r.epoch}/{r.epochs}</span>
                    </div>
                  </td>
                  <td class="r num">{metric(r.best?.map50_95)}</td>
                  <td class="hide-sm"><RunSpark run={r} /></td>
                  <td class="r faint hide-sm">{ago(r.created)}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    </Panel>

    <Panel title="Best models" pad="none" id="best-models">
      {#snippet actions()}<Button size="sm" variant="ghost" iconRight={ArrowRight} href={href('/models')}>All models</Button>{/snippet}
      {#if models.error && !models.data}
        <ErrorState error={models.error} retry={() => models.refresh()} />
      {:else if !models.data}
        <div class="list-sk">{#each [0, 1, 2] as i (i)}<Skeleton h="20px" />{/each}</div>
      {:else if topModels.length === 0}
        <EmptyState compact icon={Box} title="No models yet" description="Finished runs save their best weights here." />
      {:else}
        <ul class="models">
          {#each topModels as m, i (m.id)}
            <li>
              <a href={href(`/models/${m.id}`)}>
                <span class="rank num">{i + 1}</span>
                <span class="m-text">
                  <span class="m-name truncate">{m.name}</span>
                  <span class="m-sub num">{m.arch} · {m.classes.length} classes · {bytes(m.bytes, prefs.units)}</span>
                </span>
                <span class="m-map">
                  <span class="num">{metric(m.metrics?.map50_95)}</span>
                  <span class="m-cap">mAP50-95</span>
                </span>
              </a>
            </li>
          {/each}
        </ul>
      {/if}
    </Panel>
  </div>

  <section class="datasets" aria-labelledby="ds-title">
    <div class="section-head">
      <h2 id="ds-title">Datasets</h2>
      <Button size="sm" variant="ghost" iconRight={ArrowRight} href={href('/datasets')}>All datasets</Button>
    </div>
    {#if datasets.error && !datasets.data}
      <Panel><ErrorState error={datasets.error} retry={() => datasets.refresh()} compact /></Panel>
    {:else if !datasets.data}
      <div class="ds-grid">{#each [0, 1, 2] as i (i)}<Skeleton h="118px" r="var(--radius-lg)" />{/each}</div>
    {:else if datasets.data.length === 0}
      <Panel>
        <EmptyState compact icon={Images} title="No datasets registered" description="Register a YOLO or COCO folder, or generate a synthetic shapes set to try FOLO in a minute.">
          {#snippet actions()}<Button icon={Plus} href={href('/datasets')}>Add dataset</Button>{/snippet}
        </EmptyState>
      </Panel>
    {:else}
      <div class="ds-grid">
        {#each datasets.data.slice(0, 3) as d (d.id)}
          <a class="ds" href={href(`/datasets/${d.id}`)}>
            <span class="ds-top">
              <span class="ds-name">{d.name}</span>
              <Badge tone="outline" mono>{d.format}</Badge>
            </span>
            <span class="ds-nums num">
              <span><b>{compact(total(d))}</b> images</span>
              <span><b>{d.classes.length}</b> classes</span>
              {#if d.stats}<span><b>{compact(d.stats.boxes)}</b> boxes</span>{/if}
            </span>
            {#if d.stats}<ClassStrip counts={d.stats.class_counts} classes={d.classes} />{/if}
            <span class="ds-classes truncate">{d.classes.join(', ')}</span>
          </a>
        {/each}
      </div>
    {/if}
  </section>
</Page>

<style>
  .top {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 340px;
    gap: var(--space-4);
    align-items: start;
  }
  .live,
  .live-sk {
    min-width: 0;
  }
  .mid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 340px;
    gap: var(--space-4);
    align-items: start;
  }
  .scroll {
    overflow-x: auto;
  }
  .list-sk {
    display: grid;
    gap: var(--space-3);
    padding: var(--space-4);
  }
  .run-name {
    display: block;
    font-weight: var(--weight-medium);
  }
  .run-name:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
  }
  .run-sub {
    display: block;
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
    line-height: var(--type-xs-line);
  }
  .table td {
    height: 48px;
  }
  .prog {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 120px;
  }
  .prog span {
    font-size: var(--type-xs-size);
    min-width: 46px;
    text-align: right;
  }
  .models {
    padding: 0 var(--space-2) var(--space-2);
  }
  .models a {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    height: 52px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-md);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .models a:hover {
    background: var(--bg-hover);
  }
  .rank {
    width: 20px;
    height: 20px;
    display: grid;
    place-items: center;
    border-radius: var(--radius-sm);
    background: var(--bg-active);
    font-size: var(--type-xs-size);
    color: var(--text-secondary);
    flex-shrink: 0;
  }
  .models li:first-child .rank {
    background: var(--accent-muted);
    color: var(--accent-text);
  }
  .m-text {
    flex: 1;
    min-width: 0;
    display: grid;
  }
  .m-name {
    font-weight: var(--weight-medium);
  }
  .m-sub {
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .m-map {
    display: grid;
    justify-items: end;
  }
  .m-map .num {
    font-weight: var(--weight-medium);
  }
  .m-cap {
    font-size: var(--type-micro-size);
    color: var(--text-tertiary);
  }
  .section-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: var(--space-3);
  }
  .section-head h2 {
    font-size: var(--type-base-size);
    font-weight: var(--weight-medium);
  }
  .ds-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-4);
  }
  .ds {
    display: grid;
    gap: var(--space-3);
    padding: var(--space-4);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
    border: 1px solid var(--line-subtle);
    transition: border-color var(--dur-fast) var(--ease-standard);
  }
  :global([data-theme='light']) .ds {
    border-color: var(--line-default);
  }
  .ds:hover {
    border-color: var(--line-strong);
  }
  .ds-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
  }
  .ds-name {
    font-weight: var(--weight-medium);
    font-size: var(--type-md-size);
  }
  .ds-nums {
    display: flex;
    gap: var(--space-4);
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .ds-nums b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .ds-classes {
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  @media (max-width: 1199px) {
    .top,
    .mid {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  @media (max-width: 1023px) {
    .ds-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
  @media (max-width: 767px) {
    .ds-grid {
      grid-template-columns: minmax(0, 1fr);
    }
    .hide-sm {
      display: none;
    }
  }
</style>
