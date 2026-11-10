<script lang="ts">
  import { MediaQuery } from 'svelte/reactivity';
  import Plus from '@lucide/svelte/icons/plus';
  import Search from '@lucide/svelte/icons/search';
  import Target from '@lucide/svelte/icons/target';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Bench } from '$lib/api/types';
  import { media } from '$lib/design/tokens';
  import { ago, date, num } from '$lib/format';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import PhoneSections from '$lib/shell/PhoneSections.svelte';
  import { ownCreateAction, setCrumbs } from '$lib/shell/crumbs.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Gate from '$lib/ui/Gate.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import NoValue from '$lib/ui/NoValue.svelte';
  import { isOffline, OFFLINE_REASON } from '$lib/ui/offline';
  import Panel from '$lib/ui/Panel.svelte';
  import { rowNav } from '$lib/ui/rowNav';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import SortHeader from '$lib/ui/SortHeader.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';
  import TableState from '$lib/ui/TableState.svelte';
  import { tooltip } from '$lib/ui/tooltip';
  import NewBenchDialog from './bench/NewBenchDialog.svelte';
  import { cannotBench } from './bench/support';
  import Unsupported from './bench/Unsupported.svelte';
  import { entriesText, entryStatus, imagesText, scoreText, sourceKind } from './bench/words';

  setCrumbs({ label: 'Benchmarks' });
  ownCreateAction();

  const system = new Resource((s) => hub.api.system(s));
  // Polled: an entry running somewhere changes a benchmark's leader and its count.
  const benches = new Resource((s) => hub.api.benches(s), { poll: 10_000 });
  const unsupported = $derived(cannotBench(system.data, benches.error));

  // `#/benchmarks?new=1` opens the dialog (the palette's "New benchmark").
  let creating = $state(router.route.query.get('new') === '1');

  type Key = 'name' | 'source' | 'images' | 'classes' | 'conditions' | 'entries' | 'leader' | 'created';
  interface Col {
    key: Key;
    label: string;
    num?: boolean;
    get: (b: Bench) => number | string;
    hide?: 'md' | 'lg';
    /** Fixed width on the header row loading and loaded share; the name takes what is left. */
    w?: string;
  }
  const cols: Col[] = [
    { key: 'name', label: 'Benchmark', get: (b) => b.name.toLowerCase() },
    { key: 'source', label: 'Images from', get: (b) => b.source_name.toLowerCase(), hide: 'md', w: '208px' },
    { key: 'images', label: 'Images', num: true, get: (b) => b.images, w: '88px' },
    { key: 'classes', label: 'Classes', num: true, get: (b) => b.classes.length, hide: 'lg', w: '88px' },
    { key: 'conditions', label: 'Conditions', num: true, get: (b) => b.plan.conditions.length, hide: 'lg', w: '104px' },
    { key: 'entries', label: 'Entries', num: true, get: (b) => b.entries, w: '88px' },
    { key: 'leader', label: 'Leader', get: (b) => b.best?.score ?? -1, w: '232px' },
    { key: 'created', label: 'Created', num: true, get: (b) => Date.parse(b.created), hide: 'md', w: '104px' },
  ];
  const COLS = cols.length;
  const hideCls = (c: Col) => (c.hide ? `hide-${c.hide}` : '');

  // Newest first, as the server sends them.
  let sortKey = $state<Key>('created');
  let sortDir = $state<'asc' | 'desc'>('desc');
  function sortBy(c: Col) {
    if (sortKey === c.key) sortDir = sortDir === 'asc' ? 'desc' : 'asc';
    else {
      sortKey = c.key;
      sortDir = c.num || c.key === 'leader' ? 'desc' : 'asc';
    }
  }

  let query = $state('');
  // Where the images came from, or what runs now (`Bench.active`: an entry or a loop queued or running).
  let filter = $state<'all' | 'dataset' | 'label_job' | 'active'>('all');
  const keep = (b: Bench, f: typeof filter) => f === 'all' || (f === 'active' ? b.active : b.source.kind === f);
  const count = (f: typeof filter) => (benches.data ?? []).filter((b) => keep(b, f)).length;
  const shown = $derived.by(() => {
    const q = query.trim().toLowerCase();
    const col = cols.find((c) => c.key === sortKey) ?? cols[0]!;
    const sign = sortDir === 'asc' ? 1 : -1;
    return (benches.data ?? [])
      .filter((b) => keep(b, filter) && (!q || `${b.name} ${b.source_name} ${b.classes.join(' ')} ${b.best?.name ?? ''}`.toLowerCase().includes(q)))
      .sort((a, b) => {
        const x = col.get(a);
        const y = col.get(b);
        const d = typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y));
        return d * sign || a.name.localeCompare(b.name);
      });
  });
  const totals = $derived({ images: shown.reduce((a, b) => a + b.images, 0), entries: shown.reduce((a, b) => a + b.entries, 0) });

  const offline = $derived(!benches.data && isOffline(benches.error));
  const hasData = $derived(benches.data ? benches.data.length > 0 : !benches.error);
  const hasRows = $derived(benches.data ? shown.length > 0 : !benches.error);
  const phone = new MediaQuery(media.phone, false);
  const open = (b: Bench) => router.go(`/benchmarks/${b.id}`);
</script>

<Page title="Benchmarks">
  {#snippet sub()}<span>Frozen sets of labeled images that grade any vision model, unchanged and under hard conditions.</span>{/snippet}
  {#snippet subPhone()}<span>Grade any model on your images.</span>{/snippet}
  {#snippet actions()}
    {#if !unsupported}
      <Gate reason={offline ? OFFLINE_REASON : null}>
        {#snippet children(why)}
          <Button variant="primary" icon={Plus} disabled={!!why} aria-describedby={why} onclick={() => (creating = true)}>{phone.current ? 'New' : 'New benchmark'}</Button>
        {/snippet}
      </Gate>
    {/if}
  {/snippet}

  <PhoneSections label="Models and benchmarks" items={[{ label: 'Models', path: '/models', current: false }, { label: 'Benchmarks', path: '/benchmarks', current: true }]} />
  {#if unsupported}
    <Unsupported />
  {:else}
    <div class="list-toolbar">
      <div class="lt-filter"><Segmented
        label="Filter benchmarks by where their images came from, or by what runs now"
        bind:value={filter}
        disabled={!hasData}
        options={[
          { value: 'all', label: 'All', meta: benches.data?.length ? String(count('all')) : undefined },
          { value: 'dataset', label: 'Datasets', meta: benches.data?.length ? String(count('dataset')) : undefined },
          { value: 'label_job', label: 'Labeling jobs', meta: benches.data?.length ? String(count('label_job')) : undefined },
          { value: 'active', label: 'Running', meta: benches.data?.length ? String(count('active')) : undefined },
        ]}
      /></div>
      <div class="lt-search"><Input bind:value={query} icon={Search} placeholder="Filter by name, source or class" aria-label="Filter benchmarks" type="search" disabled={!hasData} /></div>
    </div>

    <Panel pad="none">
      <div class="scroll" class:state={!hasRows}>
        <table class="table benches" aria-busy={!benches.data} aria-label="Benchmarks" use:rowNav>
          {#if hasRows}<thead>
            <tr>
              {#each cols as c (c.key)}
                <SortHeader label={c.label} class={hideCls(c)} align={c.num ? 'right' : 'left'} width={c.w} dir={sortKey === c.key ? sortDir : null} disabled={!benches.data} onsort={() => sortBy(c)} />
              {/each}
            </tr>
          </thead>{/if}
          <tbody>
            {#if benches.error && !benches.data}
              <TableState cols={COLS}><ErrorState error={benches.error} title="Could not load benchmarks" retry={() => benches.refresh()} /></TableState>
            {:else if benches.data?.length === 0}
              <TableState cols={COLS}>
                <EmptyState
                  icon={Target}
                  title="No benchmarks yet"
                  description="A benchmark freezes a set of your labeled images and grades any model on them: unchanged, and under haze, low light, blur or whatever your camera meets. Make one to see how a model holds up before you rely on it."
                >
                  {#snippet actions()}<Button icon={Plus} onclick={() => (creating = true)}>New benchmark</Button>{/snippet}
                </EmptyState>
              </TableState>
            {:else if benches.data && shown.length === 0}
              <TableState cols={COLS}>
                <EmptyState icon={Search} title="No benchmarks match" description="Try another source or search term.">
                  {#snippet actions()}<Button size="sm" onclick={() => ((query = ''), (filter = 'all'))}>Clear filters</Button>{/snippet}
                </EmptyState>
              </TableState>
            {:else if !benches.data}
              {#each [0, 1, 2] as i (i)}
                <tr aria-hidden="true">
                  {#each cols as c (c.key)}
                    <td class={hideCls(c)} class:r={c.num}><Skeleton h="12px" w={c.key === 'name' ? '140px' : c.key === 'leader' || c.key === 'source' ? '120px' : '32px'} /></td>
                  {/each}
                </tr>
              {/each}
            {:else}
              {#each shown as b (b.id)}
                <tr class="link" data-row aria-label={b.name} onclick={() => open(b)}>
                  <td class="name-cell">
                    <span class="nm-row">
                      <a class="name truncate" href={href(`/benchmarks/${b.id}`)} onclick={(e) => e.stopPropagation()} title={b.name}>{b.name}</a>
                      {#if b.active}<span class="run" use:tooltip={'An entry or a loop is running'}><StatusBadge plain dotOnly state={entryStatus({ status: 'running' })} /></span>{/if}
                    </span>
                  </td>
                  <td class="hide-md src"><Meta line><span>{b.source_name}</span><span>{sourceKind(b.source)}</span></Meta></td>
                  <td class="r num muted">{num(b.images)}</td>
                  <td class="r num muted hide-lg">{b.classes.length}</td>
                  <td class="r num muted hide-lg">{b.plan.conditions.length}</td>
                  <td class="r num muted">{num(b.entries)}</td>
                  <td class="leader">
                    {#if b.best}
                      <span class="lead-row"><span class="l-name truncate" title={b.best.name}>{b.best.name}</span><span class="l-score num">{#if b.best.partial}<span class="partial" use:tooltip={'The leader’s score is partial: it compares only loosely with complete entries'}>partial</span>{/if}{scoreText(b.best.score)}</span></span>
                    {:else}
                      <NoValue label={b.entries ? 'no finished entry yet' : 'no entries yet'} />
                    {/if}
                  </td>
                  <td class="r faint hide-md" title={date(b.created)}>{ago(b.created)}</td>
                </tr>
              {/each}
            {/if}
          </tbody>
        </table>
      </div>
      <!-- Phones: the name, then where its images came from and how many; the leader's score on the right. -->
      {#if shown.length}
        <ul class="rows" aria-label="Benchmarks">
          {#each shown as b (b.id)}
            <li>
              <a href={href(`/benchmarks/${b.id}`)}>
                <span class="m-main">
                  <span class="nm-row"><span class="name truncate">{b.name}</span>{#if b.active}<StatusBadge plain dotOnly state={entryStatus({ status: 'running' })} />{/if}</span>
                  <span class="m-meta num"><Meta><span>{b.source_name}</span><span>{imagesText(b.images)}</span><span>{entriesText(b.entries)}</span></Meta></span>
                </span>
                <span class="m-side">
                  {#if b.best}<span class="num m-score"><span class="sr-only">Leader's score </span>{#if b.best.partial}<span class="partial">partial</span>{/if}{scoreText(b.best.score)}</span><span class="m-who truncate">{b.best.name}</span>{:else}<NoValue label="no entries yet" />{/if}
                </span>
              </a>
            </li>
          {/each}
        </ul>
      {:else if !benches.data && !benches.error}
        <ul class="rows" aria-hidden="true">
          {#each [0, 1, 2] as i (i)}
            <li><span class="row-sk"><span class="m-main"><Skeleton w="55%" h="14px" /><Skeleton w="70%" h="10px" /></span><Skeleton w="40px" h="14px" /></span></li>
          {/each}
        </ul>
      {/if}
    </Panel>
    {#if benches.data && shown.length}
      <p class="list-foot num">
        <Meta>
          <span>{shown.length < benches.data.length ? `${shown.length} of ${benches.data.length}` : shown.length} {benches.data.length === 1 ? 'benchmark' : 'benchmarks'}</span>
          <span>{imagesText(totals.images)}</span>
          <span>{entriesText(totals.entries)}</span>
        </Meta>
      </p>
    {/if}
  {/if}
</Page>

<NewBenchDialog bind:open={creating} />

<style>
  .scroll {
    overflow-x: auto;
  }
  .name-cell {
    max-width: 0;
  }
  .name {
    display: block;
  }
  .src {
    color: var(--text-secondary);
  }
  /* The leader: its name (cut when long; its own page has it whole) and its score on the cell's right edge. */
  .leader {
    max-width: 0;
  }
  .lead-row {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
  }
  .l-name {
    flex: 1;
    min-width: 0;
    color: var(--text-secondary);
  }
  .l-score {
    font-weight: var(--weight-medium);
  }
  /* "partial" before a partial leader's score, a tertiary word (the leaderboard's mark). */
  .partial {
    margin-right: var(--space-2);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  /* The name, then the running mark when an entry or a loop of it runs. */
  .nm-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .run {
    flex-shrink: 0;
    display: inline-flex;
  }
  .rows {
    display: none;
  }
  @media (max-width: 767px) {
    .scroll:not(.state) {
      display: none;
    }
    .rows {
      display: block;
    }
    .rows li + li {
      border-top: 1px solid var(--line-subtle);
    }
    .rows .name {
      font-weight: var(--weight-medium);
    }
    .rows li > a,
    .row-sk {
      display: flex;
      align-items: center;
      gap: var(--space-3);
      min-height: 56px;
      padding: var(--space-2) var(--space-4);
    }
    .rows li > a:active {
      background: var(--bg-hover);
    }
    .m-main {
      flex: 1;
      min-width: 0;
      display: grid;
    }
    .row-sk .m-main {
      gap: var(--space-2);
    }
    .m-meta {
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-secondary);
    }
    .m-side {
      flex-shrink: 0;
      display: grid;
      justify-items: end;
      max-width: 40%;
    }
    .m-score {
      font-weight: var(--weight-medium);
    }
    .m-who {
      max-width: 100%;
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-tertiary);
    }
  }
</style>
