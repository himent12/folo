<script lang="ts">
  import ArrowDown from '@lucide/svelte/icons/arrow-down';
  import ArrowUp from '@lucide/svelte/icons/arrow-up';
  import Box from '@lucide/svelte/icons/box';
  import Plus from '@lucide/svelte/icons/plus';
  import Search from '@lucide/svelte/icons/search';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Model, Run } from '$lib/api/types';
  import { ago, bytes, compact, date, metric, num } from '$lib/format';
  import { prefs } from '$lib/prefs.svelte';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';

  setCrumbs({ label: 'Models' });

  const models = new Resource((s) => hub.api.models(s));
  // Only for source-run names and evaluation datasets; the list still renders if these fail.
  const runs = new Resource((s) => hub.api.runs(s));
  const datasets = new Resource((s) => hub.api.datasets(s));

  type Key = 'name' | 'arch' | 'classes' | 'imgsz' | 'params' | 'flops' | 'bytes' | 'map50_95' | 'map50' | 'source' | 'created';
  interface Col {
    key: Key;
    label: string;
    num?: boolean;
    get: (m: Model) => number | string;
    hide?: 'md' | 'lg';
  }

  const runById = $derived(new Map<string, Run>((runs.data ?? []).map((r) => [r.id, r])));
  const sourceName = (m: Model) => (m.run_id ? (runById.get(m.run_id)?.name ?? m.run_id) : '');

  // mAP is only comparable on the same validation data: a model is evaluated on its run's dataset, so the list is
  // grouped by it and each group is ranked on its own. Imported weights carry no dataset and are never ranked.
  const IMPORTED = '~imported';
  const UNKNOWN = '~unknown';
  const dsName = $derived(new Map((datasets.data ?? []).map((d) => [d.id, d.name])));
  const evalSet = (m: Model): { key: string; label: string } => {
    if (!m.run_id) return { key: IMPORTED, label: 'Imported' };
    const dsId = runById.get(m.run_id)?.config.dataset_id;
    if (!dsId) return { key: UNKNOWN, label: 'Unknown dataset' };
    return { key: dsId, label: dsName.get(dsId) ?? dsId };
  };

  const cols: Col[] = [
    { key: 'name', label: 'Model', get: (m) => m.name.toLowerCase() },
    { key: 'arch', label: 'Arch', get: (m) => m.arch },
    { key: 'classes', label: 'Classes', num: true, get: (m) => m.classes.length, hide: 'lg' },
    { key: 'imgsz', label: 'Input px', num: true, get: (m) => m.imgsz, hide: 'md' },
    { key: 'params', label: 'Params', num: true, get: (m) => m.params },
    { key: 'flops', label: 'GFLOPs', num: true, get: (m) => m.flops_g, hide: 'lg' },
    { key: 'bytes', label: 'Size', num: true, get: (m) => m.bytes, hide: 'md' },
    { key: 'map50_95', label: 'mAP50-95', num: true, get: (m) => m.metrics?.map50_95 ?? -1 },
    { key: 'map50', label: 'mAP50', num: true, get: (m) => m.metrics?.map50 ?? -1 },
    { key: 'source', label: 'Source run', get: (m) => sourceName(m) || '~', hide: 'md' },
    { key: 'created', label: 'Created', num: true, get: (m) => Date.parse(m.created) },
  ];

  // Default: best first within each dataset.
  let sortKey = $state<Key>('map50_95');
  let sortDir = $state<'asc' | 'desc'>('desc');
  let query = $state('');
  let dsFilter = $state('all');

  function sortBy(c: Col) {
    if (sortKey === c.key) sortDir = sortDir === 'asc' ? 'desc' : 'asc';
    else {
      sortKey = c.key;
      sortDir = c.num ? 'desc' : 'asc';
    }
  }

  const matches = (m: Model, q: string) =>
    !q || `${m.name} ${m.arch} ${m.id} ${sourceName(m)} ${evalSet(m).label}`.toLowerCase().includes(q);

  /** Groups in display order (named datasets A–Z, then imported, then unknown), each with its rows sorted. */
  const groups = $derived.by(() => {
    const q = query.trim().toLowerCase();
    const col = cols.find((c) => c.key === sortKey) ?? cols[0]!;
    const sign = sortDir === 'asc' ? 1 : -1;
    const by = new Map<string, { key: string; label: string; rows: Model[] }>();
    for (const m of models.data ?? []) {
      const e = evalSet(m);
      if ((dsFilter !== 'all' && e.key !== dsFilter) || !matches(m, q)) continue;
      if (!by.has(e.key)) by.set(e.key, { ...e, rows: [] });
      by.get(e.key)!.rows.push(m);
    }
    const rank = (k: string) => (k === IMPORTED ? 1 : k === UNKNOWN ? 2 : 0);
    const out = [...by.values()].sort((a, b) => rank(a.key) - rank(b.key) || a.label.localeCompare(b.label));
    for (const g of out)
      g.rows.sort((a, b) => {
        const x = col.get(a);
        const y = col.get(b);
        const d = typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y));
        return d * sign || a.name.localeCompare(b.name);
      });
    return out;
  });
  const shownCount = $derived(groups.reduce((a, g) => a + g.rows.length, 0));

  /** The best model (mAP50-95) of each evaluation dataset that has at least two to compare. */
  const bestByDs = $derived.by(() => {
    const by = new Map<string, Model[]>();
    for (const m of models.data ?? []) {
      const k = evalSet(m).key;
      if (k === IMPORTED || k === UNKNOWN || !m.metrics) continue;
      by.set(k, [...(by.get(k) ?? []), m]);
    }
    const out = new Map<string, string>();
    for (const [k, list] of by)
      if (list.length > 1) out.set(k, list.reduce((b, m) => ((m.metrics?.map50_95 ?? -1) > (b.metrics?.map50_95 ?? -1) ? m : b)).id);
    return out;
  });
  const isBest = (m: Model) => bestByDs.get(evalSet(m).key) === m.id;

  const dsOptions = $derived.by(() => {
    const keys = new Map<string, { label: string; n: number }>();
    for (const m of models.data ?? []) {
      const e = evalSet(m);
      const k = keys.get(e.key) ?? { label: e.label, n: 0 };
      k.n++;
      keys.set(e.key, k);
    }
    const named = [...keys].filter(([k]) => k !== IMPORTED && k !== UNKNOWN).sort((a, b) => a[1].label.localeCompare(b[1].label));
    const rest = [...keys].filter(([k]) => k === IMPORTED || k === UNKNOWN);
    return [
      { value: 'all', label: 'All', meta: String(models.data?.length ?? 0) },
      ...[...named, ...rest].map(([value, { label, n }]) => ({ value, label, meta: String(n) })),
    ];
  });
  const totalBytes = $derived((models.data ?? []).reduce((a, m) => a + m.bytes, 0));
  const ariaSort = (k: Key) => (sortKey === k ? (sortDir === 'asc' ? 'ascending' : 'descending') : undefined);
  const hideCls = (c: Col) => (c.hide ? `hide-${c.hide}` : '');
</script>

<Page title="Models">
  {#snippet sub()}
    {#if models.data && models.data.length}
      <span class="num">{models.data.length} {models.data.length === 1 ? 'model' : 'models'} · {bytes(totalBytes, prefs.units)} · ranked per evaluation dataset</span>
    {:else}
      <span>Trained weights and imported <span class="mono">.folo</span> files</span>
    {/if}
  {/snippet}

  {#if !models.error && (!models.data || models.data.length > 0)}
    <div class="toolbar">
      {#if models.data}
        <div class="seg-scroll"><Segmented label="Evaluation dataset" bind:value={dsFilter} options={dsOptions} /></div>
      {:else}
        <Skeleton w="320px" h="var(--control-md)" r="var(--radius-md)" />
      {/if}
      <div class="search"><Input bind:value={query} icon={Search} placeholder="Filter by name, arch or run" aria-label="Filter models" type="search" /></div>
    </div>
  {/if}

  <Panel pad="none">
    {#if models.error && !models.data}
      <ErrorState error={models.error} retry={() => models.refresh()} title="Could not load models" />
    {:else if models.data?.length === 0}
      <EmptyState
        icon={Box}
        title="No models yet"
        description="A training run saves its best weights here when it finishes. You can also drop .folo files into the models directory of folo serve."
      >
        {#snippet actions()}<Button variant="primary" icon={Plus} href={href('/runs/new')}>New run</Button>{/snippet}
      </EmptyState>
    {:else if models.data && shownCount === 0}
      <EmptyState compact icon={Search} title="No models match" description="Try another dataset or search term.">
        {#snippet actions()}<Button size="sm" onclick={() => ((dsFilter = 'all'), (query = ''))}>Clear filters</Button>{/snippet}
      </EmptyState>
    {:else}
      <div class="scroll">
        <table class="table models" aria-busy={!models.data} aria-label="Models by evaluation dataset">
          <thead>
            <tr>
              {#each cols as c (c.key)}
                <th class={hideCls(c)} class:r={c.num} aria-sort={ariaSort(c.key)} scope="col">
                  <button type="button" class="sort" class:on={sortKey === c.key} disabled={!models.data} onclick={() => sortBy(c)}>
                    <span>{c.label}</span>
                    {#if sortKey === c.key}
                      {#if sortDir === 'asc'}<ArrowUp size={12} strokeWidth={2} aria-hidden="true" />{:else}<ArrowDown size={12} strokeWidth={2} aria-hidden="true" />{/if}
                    {/if}
                  </button>
                </th>
              {/each}
            </tr>
          </thead>
          {#if !models.data}
            <tbody aria-hidden="true">
              <tr class="group"><th colspan={cols.length}><Skeleton w="96px" h="10px" /></th></tr>
              {#each [0, 1, 2, 3] as i (i)}
                <tr>
                  {#each cols as c (c.key)}
                    <td class={hideCls(c)} class:r={c.num}><Skeleton h="12px" w={c.key === 'name' ? '112px' : '40px'} /></td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          {:else}
            {#each groups as g (g.key)}
              <tbody>
                <tr class="group">
                  <th colspan={cols.length} scope="colgroup">
                    {#if g.key === IMPORTED || g.key === UNKNOWN}
                      <span>{g.label}</span>
                    {:else}
                      <a class="g-name" href={href(`/datasets/${g.key}`)}>{g.label}</a>
                    {/if}
                    <span class="faint num">{g.rows.length}</span>
                  </th>
                </tr>
                {#each g.rows as m (m.id)}
                  <tr class="link" onclick={() => router.go(`/models/${m.id}`)}>
                    <td><a class="name" href={href(`/models/${m.id}`)} title={m.id} onclick={(e) => e.stopPropagation()}>{m.name}</a></td>
                    <td class="mono faint">{m.arch}</td>
                    <td class="r hide-lg" title={m.classes.slice(0, 12).join(', ') + (m.classes.length > 12 ? ', …' : '')}>{m.classes.length}</td>
                    <td class="r hide-md">{m.imgsz}</td>
                    <td class="r">{compact(m.params)}</td>
                    <td class="r hide-lg">{num(m.flops_g, 1)}</td>
                    <td class="r hide-md">{bytes(m.bytes, prefs.units)}</td>
                    <td class="r map">
                      {#if isBest(m)}<span class="best" title="Best mAP50-95 on {g.label}">best</span>{/if}{metric(m.metrics?.map50_95)}
                    </td>
                    <td class="r muted">{metric(m.metrics?.map50)}</td>
                    <td class="hide-md">
                      {#if m.run_id}
                        <a class="src" href={href(`/runs/${m.run_id}`)} onclick={(e) => e.stopPropagation()}>{sourceName(m)}</a>
                      {:else}
                        <span class="faint">—</span>
                      {/if}
                    </td>
                    <td class="r faint" title={date(m.created)}>{ago(m.created)}</td>
                  </tr>
                {/each}
              </tbody>
            {/each}
          {/if}
        </table>
      </div>

      <!-- Phones: the same groups as two-line rows (name and one meta line, mAP on the right). -->
      <div class="rows">
        {#each groups as g (g.key)}
          <section aria-label={g.label}>
            <h2 class="m-group">{g.label} <span class="faint num">{g.rows.length}</span></h2>
            <ul>
              {#each g.rows as m (m.id)}
                <li>
                  <a href={href(`/models/${m.id}`)}>
                    <span class="m-main">
                      <span class="name truncate">{m.name}</span>
                      <span class="m-meta truncate"><span class="mono">{m.arch}</span> · <span class="num">{m.imgsz} px · {compact(m.params)} · {bytes(m.bytes, prefs.units)}</span></span>
                    </span>
                    <span class="m-side">
                      <span class="num map">{metric(m.metrics?.map50_95)}</span>
                      <span class="m-cap">{isBest(m) ? 'best · ' : ''}mAP50-95</span>
                    </span>
                  </a>
                </li>
              {/each}
            </ul>
          </section>
        {/each}
      </div>
    {/if}
  </Panel>
</Page>

<style>
  .toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    flex-wrap: wrap;
  }
  .search {
    width: 280px;
    max-width: 100%;
  }
  .seg-scroll {
    max-width: 100%;
    overflow-x: auto;
    scrollbar-width: none;
  }
  /* Phones: the dataset filters scroll sideways; the faded edge says there is more. */
  @media (max-width: 767px) {
    .seg-scroll {
      mask-image: linear-gradient(90deg, #000 calc(100% - 32px), transparent);
      padding-right: var(--space-6);
    }
  }
  .scroll {
    overflow-x: auto;
  }
  .models td {
    height: 40px;
  }
  thead th {
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
  th .sort:not(:disabled):hover,
  th .sort.on {
    color: var(--text-primary);
  }
  th .sort:disabled {
    cursor: default;
  }
  th .sort:focus-visible {
    outline-offset: -2px;
  }
  /* A 32 px section header per evaluation dataset: the groups are ranked separately. */
  .group th {
    height: 32px;
    padding: 0 var(--space-4);
    background: var(--bg-raised);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    text-align: left;
    border-bottom: 1px solid var(--line-subtle);
  }
  .group th .faint {
    margin-left: var(--space-2);
    font-weight: var(--weight-regular);
  }
  .g-name:hover,
  .name:hover,
  .src:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
  }
  /* The base table drops the last row's border; inside a group it still separates the next header. */
  .models tbody:not(:last-child) tr:last-child td {
    border-bottom: 1px solid var(--line-subtle);
  }
  .name {
    font-weight: var(--weight-medium);
  }
  .src {
    color: var(--text-secondary);
  }
  .map {
    font-weight: var(--weight-medium);
  }
  .best {
    margin-right: var(--space-2);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .rows {
    display: none;
  }
  @media (max-width: 1279px) {
    .hide-lg {
      display: none;
    }
  }
  @media (max-width: 1023px) {
    .hide-md {
      display: none;
    }
  }
  @media (max-width: 767px) {
    .scroll {
      display: none;
    }
    .rows {
      display: block;
    }
    .search {
      width: 100%;
    }
    .m-group {
      display: flex;
      align-items: center;
      gap: var(--space-2);
      height: 32px;
      padding: 0 var(--space-4);
      background: var(--bg-raised);
      border-bottom: 1px solid var(--line-subtle);
      font-size: var(--type-sm-size);
      font-weight: var(--weight-medium);
      color: var(--text-secondary);
    }
    section + section .m-group {
      border-top: 1px solid var(--line-subtle);
    }
    section:first-child .m-group {
      border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    }
    .m-group .faint {
      font-weight: var(--weight-regular);
    }
    li + li {
      border-top: 1px solid var(--line-subtle);
    }
    li a {
      display: flex;
      align-items: center;
      gap: var(--space-3);
      min-height: 56px;
      padding: var(--space-2) var(--space-4);
    }
    li a:active {
      background: var(--bg-hover);
    }
    .m-main {
      flex: 1;
      min-width: 0;
      display: grid;
    }
    .m-meta {
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-tertiary);
    }
    .m-side {
      display: grid;
      justify-items: end;
      flex-shrink: 0;
    }
    .m-cap {
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-tertiary);
    }
  }
</style>
