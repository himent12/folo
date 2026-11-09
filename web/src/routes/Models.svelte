<script lang="ts">
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
  import { ownCreateAction, setCrumbs } from '$lib/shell/crumbs.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Versus from './models/Versus.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import SortHeader from '$lib/ui/SortHeader.svelte';
  import TableState from '$lib/ui/TableState.svelte';
  import Ident from '$lib/ui/Ident.svelte';
  import { rowNav } from '$lib/ui/rowNav';
  import { unlike, versusYolo26n } from '$lib/benchmarks';
  import { origin } from '$lib/lineage';

  setCrumbs({ label: 'Models' });

  const models = new Resource((s) => hub.api.models(s));
  // No models yet: the empty state's New run is the page's create action, so the top bar's steps aside.
  ownCreateAction(() => models.data?.length === 0);
  /** Loading: placeholder dataset groups (models per group), so the list keeps its grouped rhythm when it arrives. */
  const SK_GROUPS = [2, 3, 2];
  // Only for source-run names and evaluation datasets; the list still renders if these fail.
  const runs = new Resource((s) => hub.api.runs(s));
  const datasets = new Resource((s) => hub.api.datasets(s));

  type Key = 'name' | 'arch' | 'classes' | 'imgsz' | 'params' | 'flops' | 'bytes' | 'map50_95' | 'map50' | 'source' | 'parent' | 'created';
  interface Col {
    key: Key;
    label: string;
    num?: boolean;
    get: (m: Model) => number | string;
    hide?: 'md' | 'lg';
  }

  const runById = $derived(new Map<string, Run>((runs.data ?? []).map((r) => [r.id, r])));
  const sourceName = (m: Model) => (m.run_id ? (runById.get(m.run_id)?.name ?? m.run_id) : '');
  const modelById = $derived(new Map<string, Model>((models.data ?? []).map((m) => [m.id, m])));
  /** The model a trained model was fine-tuned from (its run's starting weights); null for imported weights. */
  const parentOf = (m: Model) => {
    const run = m.run_id ? runById.get(m.run_id) : undefined;
    return run ? origin(run.config.weights, modelById) : null;
  };

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
    { key: 'parent', label: 'Started from', get: (m) => parentOf(m)?.label ?? '~', hide: 'lg' },
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
  const hideCls = (c: Col) => (c.hide ? `hide-${c.hide}` : '');
  /** Medium weight goes to the name and the score the list is ranked by; specs stay regular. */
  const strongScore = $derived<Key>(sortKey === 'map50' ? 'map50' : 'map50_95');
  /** One line for the collapsed engine benchmarks: what is measured and how many rows are not like for like (a win
   *  count would add up unlike conditions; each row's own qualifier says which way it goes). */
  const benchUnlike = versusYolo26n.filter((v) => unlike[v.metric]).length;
</script>

<Page title="Models">
  <!-- One fixed line: the counts live in the toolbar, so nothing here changes as the list loads. -->
  {#snippet sub()}<span>Trained weights and imported <span class="mono">.folo</span> files, ranked per evaluation dataset.</span>{/snippet}
  <!-- Phones: one line, so the list starts 16 px under it. -->
  {#snippet subPhone()}<span>Ranked per evaluation dataset.</span>{/snippet}

  {#if !models.error && (!models.data || models.data.length > 0)}
    <!-- The list toolbar (base.css): one row on laptops and up; tablets and phones put the filter field on a row of its own. -->
    <div class="list-toolbar">
      <div class="lt-filter">
      {#if models.data}
        <div class="seg" data-scrolls><Segmented label="Evaluation dataset" bind:value={dsFilter} options={dsOptions} /></div>
        <!-- Phones: every dataset in a native picker; a strip that scrolls sideways hides the last ones. -->
        <div class="ds-pick">
          <label for="models-ds">Dataset</label>
          <Select id="models-ds" bind:value={dsFilter} options={dsOptions.map((o) => ({ value: o.value, label: `${o.label} · ${o.meta}` }))} />
        </div>
      {:else}
        <!-- The same two controls at their loaded size: the dataset strip (a picker on phones). -->
        <div class="seg ctl-sk" style:width="320px"><Skeleton h="100%" r="var(--radius-md)" /></div>
        <div class="ds-pick" aria-hidden="true">
          <span class="ds-label">Dataset</span>
          <span class="ctl-sk"><Skeleton h="100%" r="var(--radius-md)" /></span>
        </div>
      {/if}
      </div>
        <span class="lt-count num">
          {#if models.data}{models.data.length} {models.data.length === 1 ? 'model' : 'models'} · {bytes(totalBytes, prefs.units)}{:else}<Skeleton inline w="104px" h="10px" />{/if}
        </span>
        <div class="lt-search"><Input bind:value={query} icon={Search} placeholder="Filter by name, arch or run" aria-label="Filter models" type="search" /></div>
    </div>
  {/if}

  <Panel pad="none">
    <div class="scroll" class:state={(!!models.data && shownCount === 0) || (!!models.error && !models.data)}>
      <table class="table models" aria-busy={!models.data} aria-label="Models by evaluation dataset" use:rowNav>
        <thead>
          <tr>
            {#each cols as c (c.key)}
              <SortHeader
                label={c.label}
                class={hideCls(c)}
                align={c.num ? 'right' : 'left'}
                dir={sortKey === c.key ? sortDir : null}
                disabled={!models.data}
                onsort={() => sortBy(c)}
              />
            {/each}
          </tr>
        </thead>
        {#if models.error && !models.data}
          <tbody><TableState cols={cols.length}><ErrorState error={models.error} retry={() => models.refresh()} title="Could not load models" /></TableState></tbody>
        {:else if models.data?.length === 0}
          <tbody>
            <TableState cols={cols.length}>
              <EmptyState icon={Box} title="No models yet" description="A training run saves its best weights here when it finishes.">
                <!-- The page's only action (its header has none), so primary. -->
                {#snippet actions()}<Button variant="primary" icon={Plus} href={href('/runs/new')}>New run</Button>{/snippet}
              </EmptyState>
            </TableState>
          </tbody>
        {:else if models.data && shownCount === 0}
          <tbody>
            <TableState cols={cols.length}>
              <EmptyState icon={Search} title="No models match" description="Try another dataset or search term.">
                {#snippet actions()}<Button size="sm" onclick={() => ((dsFilter = 'all'), (query = ''))}>Clear filters</Button>{/snippet}
              </EmptyState>
            </TableState>
          </tbody>
        {:else if !models.data}
          <!-- The loaded rhythm: dataset groups (a header row, then their models), with placeholder content. -->
          {#each SK_GROUPS as n, gi (gi)}
            <tbody aria-hidden="true">
              <tr class="group"><th colspan={cols.length}><Skeleton h="10px" w="96px" /></th></tr>
              {#each Array(n) as _, i (i)}
                <tr>
                  {#each cols as c (c.key)}
                    <td class={hideCls(c)} class:r={c.num}><Skeleton h="12px" w={c.key === 'name' ? '112px' : '40px'} /></td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          {/each}
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
                {@const parent = parentOf(m)}
                <tr class="link" data-row aria-label={m.name} onclick={() => router.go(`/models/${m.id}`)}>
                  <td><a class="name" href={href(`/models/${m.id}`)} title={m.id} onclick={(e) => e.stopPropagation()}>{m.name}</a></td>
                  <td><Ident value={m.arch} class="faint" /></td>
                  <td class="r spec hide-lg" title={m.classes.slice(0, 12).join(', ') + (m.classes.length > 12 ? ', …' : '')}>{m.classes.length}</td>
                  <td class="r spec hide-md">{m.imgsz}</td>
                  <td class="r spec">{compact(m.params)}</td>
                  <td class="r spec hide-lg">{num(m.flops_g, 1)}</td>
                  <td class="r spec hide-md">{bytes(m.bytes, prefs.units)}</td>
                  <td class="r score" class:strong={strongScore === 'map50_95'}>{metric(m.metrics?.map50_95)}</td>
                  <td class="r score" class:strong={strongScore === 'map50'}>{metric(m.metrics?.map50)}</td>
                  <td class="hide-md">
                    {#if m.run_id}
                      <a class="src" href={href(`/runs/${m.run_id}`)} onclick={(e) => e.stopPropagation()}>{sourceName(m)}</a>
                    {:else}
                      <span class="faint">—</span>
                    {/if}
                  </td>
                  <td class="hide-lg">
                    {#if parent?.kind === 'model' && modelById.has(parent.id)}
                      <Ident value={parent.label} class="src" href={href(`/models/${parent.id}`)} onclick={(e) => e.stopPropagation()} />
                    {:else if parent?.kind === 'file'}
                      <Ident value={parent.label} class="src" title={parent.path} />
                    {:else}
                      {#if parent && parent.kind !== 'scratch'}<Ident value={parent.label} class="faint" />{:else}<span class="faint">{parent ? 'From scratch' : '—'}</span>{/if}
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
      {#if !models.data && !models.error}
        {#each SK_GROUPS as n, gi (gi)}
          <section aria-hidden="true">
            <h2 class="m-group"><Skeleton h="10px" w="96px" /></h2>
            <ul>
              {#each Array(n) as _, i (i)}
                <li>
                  <span class="row-sk">
                    <span class="m-main"><span class="l-name"><Skeleton w="55%" h="14px" /></span><span class="l-meta"><Skeleton w="75%" h="10px" /></span></span>
                    <span class="m-side"><span class="l-name"><Skeleton w="40px" h="14px" /></span></span>
                  </span>
                </li>
              {/each}
            </ul>
          </section>
        {/each}
      {/if}
      {#each groups as g, gi (g.key)}
        <section aria-label={g.label}>
          <!-- The value column is named once, on the first group's header, not under every value. -->
          <h2 class="m-group">
            {g.label} <span class="faint num">{g.rows.length}</span>
            {#if gi === 0}<span class="m-col">mAP50-95</span>{/if}
          </h2>
          <ul>
            {#each g.rows as m (m.id)}
              <li>
                <a href={href(`/models/${m.id}`)}>
                  <span class="m-main">
                    <span class="name truncate">{m.name}</span>
                    <span class="m-meta truncate"><Ident value={m.arch} /> · <span class="num">{m.imgsz} px · {compact(m.params)} · {bytes(m.bytes, prefs.units)}</span></span>
                  </span>
                  <span class="m-side"><span class="num map"><span class="sr-only">mAP50-95 </span>{metric(m.metrics?.map50_95)}</span></span>
                </a>
              </li>
            {/each}
          </ul>
        </section>
      {/each}
    </div>
  </Panel>

  <!-- Only with models to look at. The engine comparison is reference: one line until it is asked for. -->
  {#if models.data?.length}
    <details class="bench">
      <summary>
        <span class="chev" aria-hidden="true"></span>
        <span class="b-title">Engine benchmarks</span>
        <span class="b-sum">{versusYolo26n.length} engine measurements · {benchUnlike} under different conditions</span>
      </summary>
      <div class="b-body"><Versus /></div>
    </details>
  {/if}
</Page>

<style>
  .seg {
    max-width: 100%;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .ds-pick {
    display: none;
    align-items: center;
    gap: var(--space-3);
    width: 100%;
  }
  .ds-pick label,
  .ds-label {
    flex-shrink: 0;
    color: var(--text-tertiary);
  }
  .ds-pick :global(.select),
  .ds-pick .ctl-sk {
    flex: 1;
  }
  .ctl-sk {
    display: block;
    max-width: 100%;
    height: var(--control-md);
  }
  @media (pointer: coarse) {
    .ctl-sk {
      height: 44px;
    }
  }
  .scroll {
    overflow-x: auto;
  }
  /* A 32 px section header per evaluation dataset: the groups are ranked separately. */
  /* Placeholders in number columns sit where the numbers will: at the right. */
  td.r :global(.sk) {
    margin-left: auto;
  }
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
  .g-name:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
  }
  /* The base table drops the last row's border; inside a group it still separates the next header. */
  .models tbody:not(:last-child) tr:last-child td {
    border-bottom: 1px solid var(--line-subtle);
  }
  /* Specs (and the score the list is not ranked by) are reference: regular and secondary, so the name and the ranked
     score lead the row. */
  .spec,
  .score {
    color: var(--text-secondary);
  }
  .score.strong {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .src,
  td :global(.ident.src) {
    color: var(--text-secondary);
  }
  /* The engine benchmarks: a collapsed row with a one-line summary; the comparison opens beneath it. */
  .bench {
    min-width: 0;
  }
  .bench summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-height: 40px;
    padding: var(--space-2) var(--space-4);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
    cursor: pointer;
    list-style: none;
    transition: border-color var(--dur-fast) var(--ease-standard);
  }
  .bench summary:hover {
    border-color: var(--line-strong);
  }
  .bench summary::-webkit-details-marker {
    display: none;
  }
  .bench summary:focus-visible {
    border-radius: var(--radius-lg);
  }
  .chev {
    flex-shrink: 0;
    width: 6px;
    height: 6px;
    margin: 0 var(--space-1) 0 2px;
    border-right: 1.5px solid var(--text-tertiary);
    border-bottom: 1.5px solid var(--text-tertiary);
    transform: rotate(-45deg);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .bench[open] .chev {
    transform: rotate(45deg);
  }
  .b-title {
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    font-weight: var(--weight-medium);
    white-space: nowrap;
  }
  .b-sum {
    min-width: 0;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  /* Open: the summary row becomes the head of one frame, the comparison panel its body. */
  .bench[open] summary {
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
  }
  /* Measurement names and their notes are sentences: they wrap (table cells default to one line). */
  .b-body :global(td:first-child) {
    white-space: normal;
  }
  .b-body > :global(.panel) {
    border-top: 0;
    border-top-left-radius: 0;
    border-top-right-radius: 0;
  }
  @media (pointer: coarse) {
    .bench summary {
      min-height: 44px;
    }
  }
  .map {
    font-weight: var(--weight-medium);
  }
  .rows {
    display: none;
  }
  @media (max-width: 767px) {
    /* Phones show the grouped rows below; the table frame stays only to hold an empty or error state. */
    .scroll:not(.state),
    .scroll.state thead {
      display: none;
    }
    /* The summary wraps under the title, on the title's edge (past the chevron). */
    .b-sum {
      flex-basis: 100%;
      padding-left: 20px;
    }
    .bench summary {
      flex-wrap: wrap;
      row-gap: 0;
    }
    .rows {
      display: block;
    }
    .seg {
      display: none;
    }
    .ds-pick {
      display: flex;
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
    li a,
    .row-sk {
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
    /* Skeleton lines in the line boxes of the name (20 px) and meta line (16 px). */
    .l-name,
    .l-meta {
      display: flex;
      align-items: center;
      height: 20px;
    }
    .l-meta {
      height: 16px;
    }
    .m-side .l-name {
      justify-content: flex-end;
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
    .m-col {
      margin-left: auto;
      font-weight: var(--weight-regular);
      color: var(--text-tertiary);
    }
  }
</style>
