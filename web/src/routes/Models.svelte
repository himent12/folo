<script lang="ts">
  import ArrowDown from '@lucide/svelte/icons/arrow-down';
  import ArrowUp from '@lucide/svelte/icons/arrow-up';
  import Box from '@lucide/svelte/icons/box';
  import Plus from '@lucide/svelte/icons/plus';
  import ScanSearch from '@lucide/svelte/icons/scan-search';
  import Search from '@lucide/svelte/icons/search';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Model, Run } from '$lib/api/types';
  import { ago, bytes, compact, date, metric, num } from '$lib/format';
  import { prefs } from '$lib/prefs.svelte';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Badge from '$lib/ui/Badge.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import MetricBar from './models/MetricBar.svelte';

  setCrumbs({ label: 'Models' });

  const models = new Resource((s) => hub.api.models(s));
  // Only for source-run names; the list still renders if this fails.
  const runs = new Resource((s) => hub.api.runs(s));

  type Key = 'name' | 'arch' | 'classes' | 'imgsz' | 'params' | 'flops' | 'bytes' | 'map50_95' | 'map50' | 'source' | 'created';
  interface Col {
    key: Key;
    label: string;
    num?: boolean;
    get: (m: Model) => number | string;
    hide?: 'md';
  }

  const runById = $derived(new Map<string, Run>((runs.data ?? []).map((r) => [r.id, r])));
  const sourceName = (m: Model) => (m.run_id ? (runById.get(m.run_id)?.name ?? m.run_id) : '');

  const cols: Col[] = [
    { key: 'name', label: 'Model', get: (m) => m.name.toLowerCase() },
    { key: 'arch', label: 'Arch', get: (m) => m.arch },
    { key: 'classes', label: 'Classes', num: true, get: (m) => m.classes.length },
    { key: 'imgsz', label: 'Input', num: true, get: (m) => m.imgsz, hide: 'md' },
    { key: 'params', label: 'Params', num: true, get: (m) => m.params },
    { key: 'flops', label: 'GFLOPs', num: true, get: (m) => m.flops_g, hide: 'md' },
    { key: 'bytes', label: 'Size', num: true, get: (m) => m.bytes },
    { key: 'map50_95', label: 'mAP50-95', num: true, get: (m) => m.metrics?.map50_95 ?? -1 },
    { key: 'map50', label: 'mAP50', num: true, get: (m) => m.metrics?.map50 ?? -1, hide: 'md' },
    { key: 'source', label: 'Source', get: (m) => sourceName(m) || '~', hide: 'md' },
    { key: 'created', label: 'Created', num: true, get: (m) => Date.parse(m.created) },
  ];

  let sortKey = $state<Key>('map50_95');
  let sortDir = $state<'asc' | 'desc'>('desc');
  let query = $state('');

  function sortBy(c: Col) {
    if (sortKey === c.key) sortDir = sortDir === 'asc' ? 'desc' : 'asc';
    else {
      sortKey = c.key;
      sortDir = c.num ? 'desc' : 'asc';
    }
  }

  const filtered = $derived.by(() => {
    const q = query.trim().toLowerCase();
    const list = (models.data ?? []).filter(
      (m) => !q || m.name.toLowerCase().includes(q) || m.arch.includes(q) || m.id.includes(q) || sourceName(m).toLowerCase().includes(q),
    );
    const col = cols.find((c) => c.key === sortKey) ?? cols[0]!;
    const sign = sortDir === 'asc' ? 1 : -1;
    return list.sort((a, b) => {
      const x = col.get(a);
      const y = col.get(b);
      const d = typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y));
      return d * sign || a.name.localeCompare(b.name);
    });
  });

  const best = $derived(
    (models.data ?? []).reduce<Model | null>((b, m) => ((m.metrics?.map50_95 ?? -1) > (b?.metrics?.map50_95 ?? -1) ? m : b), null),
  );
  const totalBytes = $derived((models.data ?? []).reduce((a, m) => a + m.bytes, 0));
  const ariaSort = (k: Key) => (sortKey === k ? (sortDir === 'asc' ? 'ascending' : 'descending') : undefined);
</script>

<Page title="Models">
  {#snippet sub()}
    {#if models.data && models.data.length}
      <span><span class="num">{models.data.length}</span> {models.data.length === 1 ? 'model' : 'models'}</span>
      <span class="dot" aria-hidden="true"></span>
      <span class="num">{bytes(totalBytes, prefs.units)}</span>
      {#if best?.metrics}
        <span class="dot" aria-hidden="true"></span>
        <span>best <a class="sub-link" href={href(`/models/${best.id}`)}>{best.name}</a> <span class="num strong">{metric(best.metrics.map50_95)}</span></span>
      {/if}
    {:else}
      <span>Trained weights and imported <span class="mono">.folo</span> files</span>
    {/if}
  {/snippet}
  {#snippet actions()}
    {#if models.data?.length}
      <div class="search">
        <Input bind:value={query} icon={Search} placeholder="Filter models" aria-label="Filter models" type="search" />
      </div>
    {/if}
    <span class="pg"><Button icon={ScanSearch} href={href('/playground')}>Playground</Button></span>
  {/snippet}

  {#if models.error && !models.data}
    <Panel><ErrorState error={models.error} retry={() => models.refresh()} title="Could not load models" compact /></Panel>
  {:else if !models.data}
    <Panel pad="none">
      <div class="sk" aria-busy="true" aria-label="Loading models">
        <Skeleton h="14px" w="40%" />
        {#each [0, 1, 2, 3, 4, 5] as i (i)}<Skeleton h="22px" />{/each}
      </div>
    </Panel>
  {:else if models.data.length === 0}
    <Panel>
      <EmptyState
        icon={Box}
        title="No models yet"
        description="A training run saves its best weights here when it finishes. You can also drop .folo files into the models directory of folo serve."
      >
        {#snippet actions()}
          <Button variant="primary" icon={Plus} href={href('/runs/new')}>New run</Button>
        {/snippet}
      </EmptyState>
    </Panel>
  {:else}
    <div class="desk">
      <Panel pad="none">
        <div class="scroll">
          <table class="table">
            <thead>
              <tr>
                {#each cols as c (c.key)}
                  <th class:r={c.num} class:hide-md={c.hide === 'md'} aria-sort={ariaSort(c.key)} scope="col">
                    <button type="button" class="sort" class:on={sortKey === c.key} onclick={() => sortBy(c)}>
                      <span>{c.label}</span>
                      {#if sortKey === c.key}
                        {#if sortDir === 'asc'}<ArrowUp size={12} strokeWidth={2} aria-hidden="true" />{:else}<ArrowDown size={12} strokeWidth={2} aria-hidden="true" />{/if}
                      {/if}
                    </button>
                  </th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each filtered as m (m.id)}
                <tr class="link" onclick={() => router.go(`/models/${m.id}`)}>
                  <td>
                    <a class="name" href={href(`/models/${m.id}`)} onclick={(e) => e.stopPropagation()}>{m.name}</a>
                    <span class="id mono">{m.id}</span>
                  </td>
                  <td><Badge tone="outline" mono>{m.arch}</Badge></td>
                  <td class="r" title={m.classes.slice(0, 12).join(', ') + (m.classes.length > 12 ? ', …' : '')}>{m.classes.length}</td>
                  <td class="r hide-md">{m.imgsz}<span class="unit">px</span></td>
                  <td class="r">{compact(m.params)}</td>
                  <td class="r hide-md">{num(m.flops_g, 1)}</td>
                  <td class="r">{bytes(m.bytes, prefs.units)}</td>
                  <td class="r"><MetricBar value={m.metrics?.map50_95} best={m.id === best?.id} /></td>
                  <td class="r hide-md"><MetricBar value={m.metrics?.map50} tone="quiet" /></td>
                  <td class="hide-md">
                    {#if m.run_id}
                      <a class="src" href={href(`/runs/${m.run_id}`)} onclick={(e) => e.stopPropagation()}>{sourceName(m)}</a>
                    {:else}
                      <span class="faint">imported</span>
                    {/if}
                  </td>
                  <td class="r faint" title={date(m.created)}>{ago(m.created)}</td>
                </tr>
              {:else}
                <tr><td colspan={cols.length} class="none">No models match “{query}”.</td></tr>
              {/each}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>

    <ul class="cards" aria-label="Models">
      {#each filtered as m (m.id)}
        <li>
          <a class="card" href={href(`/models/${m.id}`)}>
            <span class="c-top">
              <span class="c-name truncate">{m.name}</span>
              <Badge tone="outline" mono>{m.arch}</Badge>
            </span>
            <span class="c-metrics">
              <span class="c-map">
                <span class="c-val num">{metric(m.metrics?.map50_95)}</span>
                <span class="c-cap">mAP50-95</span>
              </span>
              <span class="c-map">
                <span class="c-val num quiet">{metric(m.metrics?.map50)}</span>
                <span class="c-cap">mAP50</span>
              </span>
              <span class="c-bar"><MetricBar value={m.metrics?.map50_95} best={m.id === best?.id} bare /></span>
            </span>
            <span class="c-meta num">
              {m.classes.length} classes · {m.imgsz}px · {compact(m.params)} params · {bytes(m.bytes, prefs.units)}
            </span>
            <span class="c-meta">{m.run_id ? `from ${sourceName(m)}` : 'imported'} · {ago(m.created)}</span>
          </a>
        </li>
      {:else}
        <li class="none">No models match “{query}”.</li>
      {/each}
    </ul>
  {/if}
</Page>

<style>
  .dot {
    width: 3px;
    height: 3px;
    border-radius: 50%;
    background: var(--text-tertiary);
  }
  .strong {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .sub-link {
    color: var(--text-primary);
  }
  .sub-link:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
  }
  .search {
    width: 220px;
  }
  .sk {
    display: grid;
    gap: var(--space-4);
    padding: var(--space-4);
  }
  .scroll {
    overflow-x: auto;
  }
  th {
    padding: 0 !important;
  }
  th .sort {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    width: 100%;
    height: 32px;
    padding: 0 var(--space-3);
    color: inherit;
    font-weight: inherit;
    white-space: nowrap;
    transition: color var(--dur-fast) var(--ease-standard);
  }
  th.r .sort {
    justify-content: flex-end;
  }
  th:first-child .sort {
    padding-left: var(--space-4);
  }
  th:last-child .sort {
    padding-right: var(--space-4);
  }
  th .sort:hover,
  th .sort.on {
    color: var(--text-primary);
  }
  th .sort:focus-visible {
    outline-offset: -2px;
  }
  .table td {
    height: 52px;
  }
  .name {
    display: block;
    font-weight: var(--weight-medium);
  }
  .name:hover,
  .src:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
  }
  .id {
    display: block;
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
    line-height: var(--type-xs-line);
  }
  .unit {
    margin-left: 1px;
    color: var(--text-tertiary);
    font-size: var(--type-xs-size);
  }
  .src {
    color: var(--text-secondary);
  }
  .none {
    text-align: center;
    color: var(--text-tertiary);
    height: 72px;
  }
  .cards {
    display: none;
  }
  @media (max-width: 1199px) {
    .hide-md {
      display: none;
    }
  }
  @media (max-width: 767px) {
    .desk {
      display: none;
    }
    .search {
      width: calc(100vw - 2 * var(--space-4));
    }
    .pg {
      display: none;
    }
    .cards {
      display: grid;
      gap: var(--space-2);
    }
    .card {
      display: grid;
      gap: var(--space-2);
      padding: var(--space-3) var(--space-4);
      border-radius: var(--radius-lg);
      background: var(--bg-panel);
      border: 1px solid var(--line-subtle);
    }
    :global([data-theme='light']) .card {
      border-color: var(--line-default);
    }
    .card:active {
      background: var(--bg-raised);
    }
    .c-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: var(--space-2);
    }
    .c-name {
      font-weight: var(--weight-medium);
      font-size: var(--type-md-size);
    }
    .c-metrics {
      display: flex;
      align-items: flex-end;
      gap: var(--space-5);
    }
    .c-map {
      display: grid;
    }
    .c-val {
      font-size: var(--type-lg-size);
      line-height: var(--type-lg-line);
      font-weight: var(--weight-medium);
    }
    .c-val.quiet {
      color: var(--text-secondary);
    }
    .c-cap {
      font-size: var(--type-micro-size);
      color: var(--text-tertiary);
    }
    .c-bar {
      flex: 1;
      padding-bottom: 6px;
    }
    .c-meta {
      font-size: var(--type-xs-size);
      color: var(--text-tertiary);
    }
    .none {
      padding: var(--space-6);
      text-align: center;
    }
  }
</style>
