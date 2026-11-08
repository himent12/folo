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
  import Select from '$lib/ui/Select.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import MetricBar from './models/MetricBar.svelte';

  setCrumbs({ label: 'Models' });

  const models = new Resource((s) => hub.api.models(s));
  // Only for source-run names and evaluation datasets; the list still renders if these fail.
  const runs = new Resource((s) => hub.api.runs(s));
  const datasets = new Resource((s) => hub.api.datasets(s));

  type Key = 'name' | 'dataset' | 'arch' | 'classes' | 'imgsz' | 'params' | 'flops' | 'bytes' | 'map50_95' | 'map50' | 'source' | 'created';
  interface Col {
    key: Key;
    label: string;
    num?: boolean;
    get: (m: Model) => number | string;
    hide?: 'md';
  }

  const runById = $derived(new Map<string, Run>((runs.data ?? []).map((r) => [r.id, r])));
  const sourceName = (m: Model) => (m.run_id ? (runById.get(m.run_id)?.name ?? m.run_id) : '');

  // mAP is only comparable on the same validation data: a model is evaluated on its run's dataset. Imported weights
  // carry no dataset, so they are never ranked against anything.
  const IMPORTED = '~imported';
  const UNKNOWN = '~unknown';
  const dsName = $derived(new Map((datasets.data ?? []).map((d) => [d.id, d.name])));
  const evalSet = (m: Model): { key: string; label: string } => {
    if (!m.run_id) return { key: IMPORTED, label: 'imported' };
    const dsId = runById.get(m.run_id)?.config.dataset_id;
    if (!dsId) return { key: UNKNOWN, label: '—' };
    return { key: dsId, label: dsName.get(dsId) ?? dsId };
  };

  const cols: Col[] = [
    { key: 'name', label: 'Model', get: (m) => m.name.toLowerCase() },
    // Imported weights sort after the named datasets.
    { key: 'dataset', label: 'Evaluation dataset', get: (m) => (m.run_id ? '0' : '1') + evalSet(m).label.toLowerCase() },
    { key: 'arch', label: 'Arch', get: (m) => m.arch },
    { key: 'classes', label: 'Classes', num: true, get: (m) => m.classes.length, hide: 'md' },
    { key: 'imgsz', label: 'Input', num: true, get: (m) => m.imgsz, hide: 'md' },
    { key: 'params', label: 'Params', num: true, get: (m) => m.params },
    { key: 'flops', label: 'GFLOPs', num: true, get: (m) => m.flops_g, hide: 'md' },
    { key: 'bytes', label: 'Size', num: true, get: (m) => m.bytes },
    { key: 'map50_95', label: 'mAP50-95', num: true, get: (m) => m.metrics?.map50_95 ?? -1 },
    { key: 'map50', label: 'mAP50', num: true, get: (m) => m.metrics?.map50 ?? -1, hide: 'md' },
    { key: 'source', label: 'Source', get: (m) => sourceName(m) || '~', hide: 'md' },
    { key: 'created', label: 'Created', num: true, get: (m) => Date.parse(m.created) },
  ];

  // Default: grouped by evaluation dataset, best first within each.
  let sortKey = $state<Key>('dataset');
  let sortDir = $state<'asc' | 'desc'>('asc');
  let query = $state('');
  let dsFilter = $state('all');

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
      (m) =>
        (dsFilter === 'all' || evalSet(m).key === dsFilter) &&
        (!q ||
          m.name.toLowerCase().includes(q) ||
          m.arch.includes(q) ||
          m.id.includes(q) ||
          sourceName(m).toLowerCase().includes(q) ||
          evalSet(m).label.toLowerCase().includes(q)),
    );
    const col = cols.find((c) => c.key === sortKey) ?? cols[0]!;
    const sign = sortDir === 'asc' ? 1 : -1;
    const map = (m: Model) => m.metrics?.map50_95 ?? -1;
    return list.sort((a, b) => {
      const x = col.get(a);
      const y = col.get(b);
      const d = typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y));
      return d * sign || (sortKey === 'dataset' ? map(b) - map(a) : 0) || a.name.localeCompare(b.name);
    });
  });

  /** The best model (mAP50-95) of each evaluation dataset that has at least two to compare. */
  const bestByDs = $derived.by(() => {
    const groups = new Map<string, Model[]>();
    for (const m of models.data ?? []) {
      const k = evalSet(m).key;
      if (k === IMPORTED || k === UNKNOWN || !m.metrics) continue;
      groups.set(k, [...(groups.get(k) ?? []), m]);
    }
    const out = new Map<string, Model>();
    for (const [k, list] of groups)
      if (list.length > 1) out.set(k, list.reduce((b, m) => ((m.metrics?.map50_95 ?? -1) > (b.metrics?.map50_95 ?? -1) ? m : b)));
    return out;
  });
  const isBest = (m: Model) => bestByDs.get(evalSet(m).key)?.id === m.id;
  const best = $derived(dsFilter === 'all' ? null : (bestByDs.get(dsFilter) ?? null));

  const dsOptions = $derived.by(() => {
    const keys = new Map<string, string>();
    for (const m of models.data ?? []) {
      const e = evalSet(m);
      if (e.key !== UNKNOWN) keys.set(e.key, e.label);
    }
    const named = [...keys].filter(([k]) => k !== IMPORTED).sort((a, b) => a[1].localeCompare(b[1]));
    return [
      { value: 'all', label: 'All datasets' },
      ...named.map(([value, label]) => ({ value, label })),
      ...(keys.has(IMPORTED) ? [{ value: IMPORTED, label: 'Imported' }] : []),
    ];
  });
  const dsCount = $derived(dsOptions.filter((o) => o.value !== 'all' && o.value !== IMPORTED).length);
  const totalBytes = $derived((models.data ?? []).reduce((a, m) => a + m.bytes, 0));
  // Loading placeholder widths per column, close to the real content so nothing jumps when rows arrive.
  const skWidth: Record<Key, string> = {
    name: '112px',
    dataset: '72px',
    arch: '48px',
    classes: '16px',
    imgsz: '40px',
    params: '36px',
    flops: '28px',
    bytes: '52px',
    map50_95: '88px',
    map50: '88px',
    source: '72px',
    created: '52px',
  };
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
        <span>best on {evalSet(best).label} <a class="sub-link" href={href(`/models/${best.id}`)}>{best.name}</a> <span class="num strong">{metric(best.metrics.map50_95)}</span></span>
      {:else if dsCount > 1}
        <span class="dot sub-extra" aria-hidden="true"></span>
        <span class="sub-extra"><span class="num">{dsCount}</span> evaluation datasets, ranked separately</span>
      {/if}
    {:else}
      <span>Trained weights and imported <span class="mono">.folo</span> files</span>
    {/if}
  {/snippet}
  {#snippet actions()}
    {#if models.data?.length}
      {#if dsOptions.length > 2}
        <div class="ds-filter"><Select label="Evaluation dataset" value={dsFilter} options={dsOptions} onchange={(v) => (dsFilter = v)} /></div>
      {/if}
      <div class="search">
        <Input bind:value={query} icon={Search} placeholder="Filter models" aria-label="Filter models" type="search" />
      </div>
    {:else if !models.data && !models.error}
      <div class="search"><Skeleton h="var(--control-md)" r="var(--radius-md)" /></div>
    {/if}
    <span class="pg"><Button icon={ScanSearch} href={href('/playground')}>Playground</Button></span>
  {/snippet}

  {#if models.error && !models.data}
    <ErrorState error={models.error} retry={() => models.refresh()} title="Could not load models" compact />
  {:else if !models.data}
    <!-- Mirrors the table (and the phone cards) row for row, so the page doesn't move when the list arrives. -->
    <div class="desk" aria-busy="true" aria-label="Loading models">
      <Panel pad="none">
        <div class="scroll">
          <table class="table sk-table" aria-hidden="true">
            <thead>
              <tr>
                {#each cols as c (c.key)}
                  <th class:r={c.num} class:hide-md={c.hide === 'md'}><span class="sort">{c.label}</span></th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each [0, 1, 2, 3, 4] as i (i)}
                <tr>
                  {#each cols as c (c.key)}
                    <td class:r={c.num} class:hide-md={c.hide === 'md'}>
                      {#if c.key === 'name'}
                        <span class="sk-two"><Skeleton h="12px" w={skWidth.name} /><Skeleton h="10px" w="56px" /></span>
                      {:else}
                        <Skeleton h={c.key === 'arch' ? '18px' : '12px'} w={skWidth[c.key]} />
                      {/if}
                    </td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
    <ul class="cards" aria-hidden="true">
      {#each [0, 1, 2] as i (i)}
        <li class="card sk-card">
          <span class="c-top"><Skeleton h="14px" w="140px" /><Skeleton h="18px" w="48px" /></span>
          <span class="c-metrics"><Skeleton h="22px" w="56px" /><Skeleton h="22px" w="56px" /></span>
          <Skeleton h="10px" w="70%" />
          <Skeleton h="10px" w="50%" />
        </li>
      {/each}
    </ul>
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
                  <td>
                    {#if evalSet(m).key === IMPORTED}
                      <span class="faint">imported</span>
                    {:else if evalSet(m).key === UNKNOWN}
                      <span class="faint">—</span>
                    {:else}
                      <a class="ds" href={href(`/datasets/${evalSet(m).key}`)} onclick={(e) => e.stopPropagation()}>{evalSet(m).label}</a>
                    {/if}
                  </td>
                  <td><Badge tone="outline" mono>{m.arch}</Badge></td>
                  <td class="r hide-md" title={m.classes.slice(0, 12).join(', ') + (m.classes.length > 12 ? ', …' : '')}>{m.classes.length}</td>
                  <td class="r hide-md">{m.imgsz}<span class="unit">px</span></td>
                  <td class="r">{compact(m.params)}</td>
                  <td class="r hide-md">{num(m.flops_g, 1)}</td>
                  <td class="r">{bytes(m.bytes, prefs.units)}</td>
                  <td class="r">
                    {#if isBest(m)}<span class="best-tag" title="Best mAP50-95 on {evalSet(m).label}">best</span>{/if}
                    <MetricBar value={m.metrics?.map50_95} best={isBest(m)} />
                  </td>
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
              <span class="c-bar"><MetricBar value={m.metrics?.map50_95} best={isBest(m)} bare /></span>
            </span>
            <span class="c-meta num">
              {m.classes.length} classes · {m.imgsz}px · {compact(m.params)} params · {bytes(m.bytes, prefs.units)}
            </span>
            <span class="c-meta">
              {#if m.run_id}on <b>{evalSet(m).label}</b>{#if isBest(m)}{' · '}<span class="c-best">best</span>{/if}{' · '}from {sourceName(m)}{:else}imported{/if} · {ago(m.created)}
            </span>
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
  .ds-filter {
    width: 200px;
  }
  .ds {
    color: var(--text-secondary);
  }
  .ds:hover {
    color: var(--text-primary);
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
  }
  .sk-table td.r :global(.sk) {
    margin-left: auto;
  }
  .sk-two {
    display: grid;
    gap: 6px;
  }
  .sk-table th .sort {
    cursor: default;
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
    .search,
    .ds-filter {
      width: calc(100vw - 2 * var(--space-4));
    }
    .sub-extra {
      display: none;
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
    .c-meta b {
      color: var(--text-secondary);
      font-weight: var(--weight-medium);
    }
    .c-best {
      color: var(--accent-text);
    }
    .none {
      padding: var(--space-6);
      text-align: center;
    }
  }
  .best-tag {
    margin-right: var(--space-2);
    padding: 1px 6px;
    border-radius: var(--radius-xs);
    background: var(--accent-muted);
    color: var(--accent-text);
    font-size: var(--type-micro-size);
    font-weight: var(--weight-medium);
    vertical-align: middle;
  }
</style>
