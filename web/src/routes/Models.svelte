<script lang="ts">
  import Box from '@lucide/svelte/icons/box';
  import Plus from '@lucide/svelte/icons/plus';
  import Search from '@lucide/svelte/icons/search';
  import { hub } from '$lib/api/index.svelte';
  import { bestMark, bestWord } from '$lib/bestMark';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Dataset, Model, Run } from '$lib/api/types';
  import { ago, bytes, compact, date, metric, num, sizeIn, sizeUnit } from '$lib/format';
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
  import Meta from '$lib/ui/Meta.svelte';
  import { rowNav } from '$lib/ui/rowNav';
  import { MediaQuery } from 'svelte/reactivity';
  import { media } from '$lib/design/tokens';
  import { unlike, versusYolo26n } from '$lib/benchmarks';
  import { FROM_SCRATCH, origin, originTip } from '$lib/lineage';
  import { filterLabel, groupKey, groupLabel, rankKey, scoreContext, splitLabel, type ScoreContext } from '$lib/scoreSummary';
  import { loadThroughput, measuredLatency, measuredTraining } from '$lib/throughput';

  setCrumbs({ label: 'Models' });

  const models = new Resource((s) => hub.api.models(s));
  // No models yet: the empty state's New run is the page's create action, so the top bar's steps aside.
  ownCreateAction(() => models.data?.length === 0);
  /** Loading: placeholder dataset groups (models per group), so the list keeps its grouped rhythm when it arrives. */
  const SK_GROUPS = [2, 3, 2];
  // Only for source-run names and evaluation datasets; the list still renders if these fail.
  const runs = new Resource((s) => hub.api.runs(s));
  const datasets = new Resource((s) => hub.api.datasets(s));
  // This server's measured speed, for the engine benchmarks' "Measured on this server" lines (null on older servers).
  const throughput = new Resource((s) => loadThroughput(hub.api, s));
  const benchHere = $derived([measuredLatency(throughput.data, 'folo-n'), measuredTraining(throughput.data, 'folo-n')].filter(Boolean).length);

  type Key = 'name' | 'arch' | 'classes' | 'imgsz' | 'params' | 'flops' | 'bytes' | 'map50_95' | 'map50' | 'source' | 'parent' | 'created';
  interface Col {
    key: Key;
    label: string;
    num?: boolean;
    get: (m: Model) => number | string;
    hide?: 'md' | 'lg';
    /** Fixed width (header label, sort mark and padding fit in it); the name column takes what is left. */
    w?: string;
  }

  const runById = $derived(new Map<string, Run>((runs.data ?? []).map((r) => [r.id, r])));
  const sourceName = (m: Model) => (m.run_id ? (runById.get(m.run_id)?.name ?? m.run_id) : '');
  const modelById = $derived(new Map<string, Model>((models.data ?? []).map((m) => [m.id, m])));
  /** The model a trained model was fine-tuned from (its run's starting weights); null for imported weights. */
  const parentOf = (m: Model) => {
    const run = m.run_id ? runById.get(m.run_id) : undefined;
    return run ? origin(run.config.weights, modelById) : null;
  };

  // A score compares only with scores on the same dataset and split (lib/scoreSummary): a trained model is scored on
  // its run's dataset, on its validation split or, when the dataset has none, its training split. The list is grouped
  // by that, each group headed by what its scores are on ("Validation split of fasteners") and ranked on its own.
  // Published weights and imported files have no known split here: a group each, after the datasets, never ranked.
  // The filter picks a group of the list: an evaluation dataset, or a source with none ("Published weights",
  // "Imported"), so it is "Group", never "Dataset". Its accessible name says what it filters; the phone picker's keeps
  // its visible word first (WCAG 2.5.3, label in name).
  const GROUP_NAME = 'Filter by evaluation dataset or source';
  const GROUP_LABEL = `Group: ${GROUP_NAME.charAt(0).toLowerCase()}${GROUP_NAME.slice(1)}`;
  const dsById = $derived(new Map<string, Dataset>((datasets.data ?? []).map((d) => [d.id, d])));
  const evalSet = (m: Model) => {
    const ctx = scoreContext(m, runById, dsById);
    return { key: groupKey(ctx), label: groupLabel(ctx), filter: filterLabel(ctx), ctx, ranked: rankKey(ctx) !== null };
  };
  /** Groups that are not ranked (published, imported, unknown split): no "best", after the ranked ones, in this order. */
  const otherRank = (k: string) => (k === '~published' ? 1 : k === '~imported' ? 2 : k.startsWith('~') ? 3 : 0);
  const splitRank = (s: string | null) => (s === 'val' ? 0 : s === 'test' ? 1 : 2);
  const groupOrder = (a: { key: string; ctx: ScoreContext }, b: { key: string; ctx: ScoreContext }) =>
    otherRank(a.key) - otherRank(b.key) ||
    (a.ctx.datasetName ?? '').localeCompare(b.ctx.datasetName ?? '') ||
    splitRank(a.ctx.split) - splitRank(b.ctx.split);

  // Source run: a run's name ends in what tells it from its siblings ("shapes-hard-s" beside "shapes-hard-m"), so it is
  // never cut (DESIGN_DIRECTION §2.3). From 1280 px every column is in use and 112 px (88 of text) holds the usual
  // names, a longer one wrapping at its hyphens; between 1024 and 1279 Classes, GFLOPs and Started from step out and the
  // name column has room to spare, so Source run takes 136.
  const laptopOnly = new MediaQuery(`${media.laptopUp} and ${media.upToDesktop}`, false);
  // Sizes in one unit for the column, named in its header (prefs.units: MiB or MB).
  const cols: Col[] = $derived([
    { key: 'name', label: 'Model', get: (m) => m.name.toLowerCase() },
    { key: 'arch', label: 'Arch', get: (m) => m.arch, w: '72px' },
    { key: 'classes', label: 'Classes', num: true, get: (m) => m.classes.length, hide: 'lg', w: '84px' },
    { key: 'imgsz', label: 'Input px', num: true, get: (m) => m.imgsz, hide: 'md', w: '88px' },
    { key: 'params', label: 'Params', num: true, get: (m) => m.params, w: '80px' },
    { key: 'flops', label: 'GFLOPs', num: true, get: (m) => m.flops_g, hide: 'lg', w: '84px' },
    { key: 'bytes', label: `Size, ${sizeUnit(prefs.units)}`, num: true, get: (m) => m.bytes, hide: 'md', w: '80px' },
    { key: 'map50_95', label: 'mAP50-95', num: true, get: (m) => m.metrics?.map50_95 ?? -1, w: '96px' },
    { key: 'map50', label: 'mAP50', num: true, get: (m) => m.metrics?.map50 ?? -1, w: '80px' },
    { key: 'source', label: 'Source run', get: (m) => sourceName(m) || '~', hide: 'md', w: laptopOnly.current ? '136px' : '112px' },
    { key: 'parent', label: 'Started from', get: (m) => parentOf(m)?.label ?? '~', hide: 'lg', w: '120px' },
    { key: 'created', label: 'Created', num: true, get: (m) => Date.parse(m.created), w: '112px' },
  ]);

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
    !q || `${m.name} ${m.arch} ${m.id} ${sourceName(m)} ${evalSet(m).ctx.datasetName ?? ''}`.toLowerCase().includes(q);

  /** Groups in display order (datasets A–Z, validation first, then published, imported, unknown), rows sorted. */
  const groups = $derived.by(() => {
    const q = query.trim().toLowerCase();
    const col = cols.find((c) => c.key === sortKey) ?? cols[0]!;
    const sign = sortDir === 'asc' ? 1 : -1;
    const by = new Map<string, ReturnType<typeof evalSet> & { rows: Model[] }>();
    for (const m of models.data ?? []) {
      const e = evalSet(m);
      if ((dsFilter !== 'all' && e.key !== dsFilter) || !matches(m, q)) continue;
      if (!by.has(e.key)) by.set(e.key, { ...e, rows: [] });
      by.get(e.key)!.rows.push(m);
    }
    const out = [...by.values()].sort(groupOrder);
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
  /** Rows to sort, or their skeleton: the sort mark shows only then. */
  const hasRows = $derived(models.data ? shownCount > 0 : !models.error);
  /** Models to filter, or on their way: the toolbar is live (else it stays in place, disabled). */
  const hasData = $derived(models.data ? models.data.length > 0 : !models.error);

  /**
   * Each comparable group's best model (highest mAP50-95), marked with tertiary "best" text whatever the sort, "joint
   * best" on every model of an exact tie (lib/bestMark): within a group the scores compare, across groups they do not,
   * so no mark spans groups (and none on an unranked one). A group whose best differs from another score only past
   * three decimals shows its scores with as many as tell them apart.
   */
  const best = $derived.by(() => {
    const marks = new Map<string, string>();
    const decimals = new Map<string, number>();
    for (const g of groups) {
      if (!g.ranked || g.rows.length < 2) continue;
      const m = bestMark(g.rows.map((r) => r.metrics?.map50_95));
      for (const i of m.best) marks.set(g.rows[i]!.id, bestWord(m));
      if (m.decimals > 3) for (const r of g.rows) decimals.set(r.id, m.decimals);
    }
    return { marks, decimals };
  });
  /** A model's mAP50-95 as the list shows it: three decimals, more where its group needs them to tell the best apart. */
  const score = (m: Model) => (m.metrics && best.decimals.has(m.id) ? m.metrics.map50_95.toFixed(best.decimals.get(m.id)) : metric(m.metrics?.map50_95));


  const dsOptions = $derived.by(() => {
    const keys = new Map<string, ReturnType<typeof evalSet> & { n: number }>();
    for (const m of models.data ?? []) {
      const e = evalSet(m);
      const k = keys.get(e.key) ?? { ...e, n: 0 };
      k.n++;
      keys.set(e.key, k);
    }
    return [
      { value: 'all', label: 'All', meta: String(models.data?.length ?? 0) },
      ...[...keys.values()].sort(groupOrder).map((k) => ({ value: k.key, label: k.filter, meta: String(k.n) })),
    ];
  });
  // What the rows shown hold, said once at the list's end (base.css .list-foot).
  const shownBytes = $derived(groups.reduce((a, g) => a + g.rows.reduce((b, m) => b + m.bytes, 0), 0));
  const hideCls = (c: Col) => (c.hide ? `hide-${c.hide}` : '');
  /** Medium weight goes to the name and the score the list is ranked by; specs stay regular. */
  const strongScore = $derived<Key>(sortKey === 'map50' ? 'map50' : 'map50_95');
  /** One line for the collapsed engine benchmarks: what is measured and how many rows are not like for like (a win
   *  count would add up unlike conditions; each row's own qualifier says which way it goes). */
  const benchUnlike = versusYolo26n.filter((v) => unlike[v.metric]).length;
</script>

<!-- What a group's scores are on: "Validation split of fasteners" (the dataset linked), or an unranked group's name. -->
{#snippet groupHead(g: { label: string; ctx: ScoreContext; ranked: boolean; rows: Model[] })}
  {#if g.ranked && g.ctx.split && g.ctx.datasetId}
    <span>{splitLabel(g.ctx.split)} of <a class="g-name" href={href(`/datasets/${g.ctx.datasetId}`)}>{g.ctx.datasetName}</a></span>
  {:else}
    <span>{g.label}</span>
  {/if}
  <span class="faint num">{g.rows.length}</span>
  {#if !g.ranked}<span class="faint">not ranked</span>{/if}
{/snippet}

<Page title="Models">
  <!-- One fixed line: the counts live at the list's end, so nothing here changes as the list loads. -->
  {#snippet sub()}<span>Trained, published and imported weights, ranked within each evaluation dataset and split.</span>{/snippet}
  <!-- Phones: one line, so the list starts 16 px under it. -->
  {#snippet subPhone()}<span>Ranked per dataset and split.</span>{/snippet}

  <!-- The list toolbar (base.css): one row on laptops and up; tablets and phones put the filter field on a row of its
       own. It is always here, so the table never moves between states (DESIGN_DIRECTION §2.8): over an empty or failed
       list it is disabled, "All" alone and no counts. -->
  <div class="list-toolbar">
    <div class="lt-filter">
    {#if models.data?.length}
      <div class="seg" data-scrolls><Segmented label={GROUP_NAME} bind:value={dsFilter} options={dsOptions} /></div>
      <!-- Phones: every dataset in a native picker; a strip that scrolls sideways hides the last ones. -->
      <div class="ds-pick">
        <label for="models-ds">Group</label>
        <Select id="models-ds" label={GROUP_LABEL} bind:value={dsFilter} options={dsOptions.map((o) => ({ value: o.value, label: `${o.label} · ${o.meta}` }))} />
      </div>
    {:else if models.data || models.error}
      <div class="seg"><Segmented label={GROUP_NAME} value="all" disabled options={[{ value: 'all', label: 'All' }]} /></div>
      <div class="ds-pick">
        <label for="models-ds">Group</label>
        <Select id="models-ds" label={GROUP_LABEL} value="all" disabled options={[{ value: 'all', label: 'All' }]} />
      </div>
    {:else}
      <!-- The same two controls at their loaded size: the dataset strip (a picker on phones). -->
      <div class="seg ctl-sk" style:width="320px"><Skeleton h="100%" r="var(--radius-md)" /></div>
      <div class="ds-pick" aria-hidden="true">
        <span class="ds-label">Group</span>
        <span class="ctl-sk"><Skeleton h="100%" r="var(--radius-md)" /></span>
      </div>
    {/if}
    </div>
    <div class="lt-search"><Input bind:value={query} icon={Search} placeholder="Filter by name, arch or run" aria-label="Filter models" type="search" disabled={!hasData} /></div>
  </div>

  <Panel pad="none">
    <!-- With no rows (empty, no match, an error) the frame drops its column heads and holds the state alone. -->
    <div class="scroll" class:state={!hasRows}>
      <table class="table models" aria-busy={!models.data} aria-label="Models by evaluation dataset" use:rowNav>
        {#if hasRows}<thead>
          <tr>
            {#each cols as c (c.key)}
              <SortHeader
                label={c.label}
                class={hideCls(c)}
                align={c.num ? 'right' : 'left'}
                width={c.w}
                dir={sortKey === c.key ? sortDir : null}
                disabled={!models.data}
                onsort={() => sortBy(c)}
              />
            {/each}

          </tr>
        </thead>{/if}
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
                  {@render groupHead(g)}
                </th>
              </tr>
              {#each g.rows as m (m.id)}
                {@const parent = parentOf(m)}
                <tr class="link" data-row aria-label={m.name} onclick={() => router.go(`/models/${m.id}`)}>
                  <td><a class="name" href={href(`/models/${m.id}`)} title={m.about ?? m.id} onclick={(e) => e.stopPropagation()}>{m.name}</a></td>
                  <td><Ident value={m.arch} class="faint" /></td>
                  <td class="r spec hide-lg" title={m.classes.slice(0, 12).join(', ') + (m.classes.length > 12 ? ', …' : '')}>{m.classes.length}</td>
                  <td class="r spec hide-md">{m.imgsz}</td>
                  <td class="r spec">{compact(m.params)}</td>
                  <td class="r spec hide-lg">{num(m.flops_g, 1)}</td>
                  <td class="r spec hide-md">{sizeIn(m.bytes, prefs.units)}</td>
                  <td class="r score" class:strong={strongScore === 'map50_95'}>{#if best.marks.has(m.id)}<span class="best">{best.marks.get(m.id)}</span>{/if}{score(m)}</td>
                  <td class="r score" class:strong={strongScore === 'map50'}>{metric(m.metrics?.map50)}</td>
                  <td class="hide-md">
                    {#if m.run_id}
                      <a class="src" href={href(`/runs/${m.run_id}`)} title={sourceName(m)} onclick={(e) => e.stopPropagation()}>{sourceName(m)}</a>
                    {:else}
                      <span class="faint">—</span>
                    {/if}
                  </td>
                  <td class="hide-lg">
                    {#if parent?.kind === 'model' && modelById.has(parent.id)}
                      <Ident value={parent.label} class="src" title={originTip(parent)} href={href(`/models/${parent.id}`)} onclick={(e) => e.stopPropagation()} />
                    {:else if parent?.kind === 'file'}
                      <Ident value={parent.label} class="src" title={parent.path} />
                    {:else}
                      {#if parent && parent.kind !== 'scratch'}<Ident value={parent.label} class="faint" title={originTip(parent)} />{:else}<span class="faint">{parent ? FROM_SCRATCH : '—'}</span>{/if}
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
            <span class="g-head">{@render groupHead(g)}</span>
            {#if gi === 0}<span class="m-col">mAP50-95</span>{/if}
          </h2>
          <ul>
            {#each g.rows as m (m.id)}
              <li>
                <a href={href(`/models/${m.id}`)} title={m.about}>
                  <span class="m-main">
                    <span class="name truncate">{m.name}</span>
                    <span class="m-meta num"><Meta line><Ident value={m.arch} /><span>{m.imgsz} px</span><span>{compact(m.params)}</span><span>{bytes(m.bytes, prefs.units)}</span></Meta></span>
                  </span>
                  <span class="m-side"><span class="num map">{#if best.marks.has(m.id)}<span class="best">{best.marks.get(m.id)}</span>{/if}<span class="sr-only">mAP50-95 </span>{score(m)}</span></span>
                </a>
              </li>
            {/each}
          </ul>
        </section>
      {/each}
    </div>
  </Panel>
  {#if models.data && shownCount}
    <p class="list-foot num">
      <Meta>
        <span>{shownCount < models.data.length ? `${shownCount} of ${models.data.length}` : shownCount} {models.data.length === 1 ? 'model' : 'models'}</span>
        <span>{bytes(shownBytes, prefs.units)}</span>
      </Meta>
    </p>
  {/if}

  <!-- Only with models to look at. The engine comparison is reference: one line until it is asked for. -->
  {#if models.data?.length}
    <details class="bench">
      <summary>
        <span class="chev" aria-hidden="true"></span>
        <span class="b-title">Engine benchmarks</span>
        <span class="b-sum"
          >{versusYolo26n.length} reference measurements · {benchUnlike} under different conditions{#if benchHere}{' '}· {benchHere} measured on this server{/if}</span
        >
      </summary>
      <div class="b-body"><Versus table={throughput.data} /></div>
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
  /* The group's top model: a word in tertiary, never a colour (scores compare only within a dataset). */
  .best {
    margin-right: var(--space-2);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .src,
  td :global(.ident.src) {
    color: var(--text-secondary);
  }
  /* A run's name is never cut: its suffix is what tells it from its siblings ("shapes-hard-s" beside "shapes-hard-m",
     DESIGN_DIRECTION §2.3). The column holds the usual names on one line (the widths above); a longer one wraps after
     a hyphen and the row grows a line, never an ellipsis and never wider than its column. */
  a.src {
    display: inline-block;
    max-width: 100%;
    white-space: normal;
    overflow-wrap: break-word;
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
    /* Phones show the grouped rows below; the table frame stays only to hold an empty or error state (no column heads at any width). */
    .scroll:not(.state) {
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
    .g-head {
      min-width: 0;
    }
    .g-head .faint {
      margin-left: var(--space-2);
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
