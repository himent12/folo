<script lang="ts">
  import Activity from '@lucide/svelte/icons/activity';
  import GitCompare from '@lucide/svelte/icons/git-compare-arrows';
  import Plus from '@lucide/svelte/icons/plus';
  import Search from '@lucide/svelte/icons/search';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Run, RunStatus } from '$lib/api/types';
  import RunSpark from '$lib/charts/RunSpark.svelte';
  import { ago, duration, metric } from '$lib/format';
  import { failureCause } from '$lib/oom';
  import { tooltip } from '$lib/ui/tooltip';
  import { href, router } from '$lib/router.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Page from '$lib/shell/Page.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Progress from '$lib/ui/Progress.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import { runProgress } from '$lib/api/progress';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';

  setCrumbs({ label: 'Runs' });

  type Filter = 'all' | 'active' | 'completed' | 'failed';
  const groups: Record<Filter, RunStatus[]> = {
    all: ['queued', 'running', 'stopping', 'stopped', 'completed', 'failed'],
    active: ['queued', 'running', 'stopping'],
    completed: ['completed', 'stopped'],
    failed: ['failed'],
  };

  const runs = new Resource((s) => hub.api.runs(s), { poll: 3000 });
  const datasets = new Resource((s) => hub.api.datasets(s));
  const dsName = $derived(new Map((datasets.data ?? []).map((d) => [d.id, d.name])));

  let filter = $state<Filter>((router.route.query.get('status') as Filter | null) ?? 'all');
  let query = $state('');
  let selected = $state<string[]>([]);
  /** Phones: checkboxes show in select mode (or once a run is selected), so rows keep their width otherwise. */
  let picking = $state(false);
  const choosing = $derived(picking || selected.length > 0);

  const count = (f: Filter) => (runs.data ?? []).filter((r) => groups[f].includes(r.status)).length;
  const shown = $derived(
    (runs.data ?? []).filter((r) => {
      if (!groups[filter].includes(r.status)) return false;
      const q = query.trim().toLowerCase();
      if (!q) return true;
      return `${r.name} ${r.config.arch} ${dsName.get(r.config.dataset_id) ?? ''} ${r.id}`.toLowerCase().includes(q);
    }),
  );

  function toggle(id: string) {
    selected = selected.includes(id) ? selected.filter((x) => x !== id) : selected.length >= 4 ? selected : [...selected, id];
  }

  function elapsed(r: Run): number | null {
    if (!r.started) return null;
    const end = r.finished ? Date.parse(r.finished) : Date.now();
    return (end - Date.parse(r.started)) / 1000;
  }

  // Progress bars stay neutral; only a live run's bar carries the accent ("now"). Status is the dot beside it.
  const STATUS: Record<Run['status'], string> = {
    queued: 'Queued',
    running: 'Running',
    stopping: 'Stopping',
    stopped: 'Stopped',
    completed: 'Completed',
    failed: 'Failed',
  };
  const tone = (r: Run) => (r.status === 'running' ? 'accent' : 'neutral');
</script>

<Page title="Runs">
  {#snippet sub()}<span>Training runs, newest first. Select up to four to compare their curves.</span>{/snippet}
  {#snippet actions()}
    {#if runs.data?.length !== 0}
      <span title={selected.length < 2 ? 'Select 2–4 runs to compare' : undefined}>
        <Button icon={GitCompare} disabled={selected.length < 2} href={href(`/runs/compare?ids=${selected.join(',')}`)}>
          Compare{#if selected.length}<span class="num faint">{selected.length}</span>{/if}
        </Button>
      </span>
      <Button variant="primary" icon={Plus} href={href('/runs/new')}>New run</Button>
    {/if}
  {/snippet}

  {#if !runs.data || runs.data.length > 0}<div class="toolbar">
    <Segmented
      label="Filter runs by status"
      bind:value={filter}
      options={[
        { value: 'all', label: 'All', meta: runs.data ? String(count('all')) : undefined },
        { value: 'active', label: 'Active', meta: runs.data ? String(count('active')) : undefined },
        { value: 'completed', label: 'Finished', meta: runs.data ? String(count('completed')) : undefined },
        { value: 'failed', label: 'Failed', meta: runs.data ? String(count('failed')) : undefined },
      ]}
    />
    <div class="search"><Input icon={Search} bind:value={query} placeholder="Filter by name, model or dataset" aria-label="Filter runs" /></div>
  </div>{/if}

  <Panel pad="none">
    {#if runs.error && !runs.data}
      <ErrorState error={runs.error} retry={() => runs.refresh()} title="Could not load runs" />
    {:else if !runs.data}
      <div class="sk">{#each [0, 1, 2, 3, 4] as i (i)}<Skeleton h="22px" />{/each}</div>
    {:else if runs.data.length === 0}
      <EmptyState icon={Activity} title="No runs yet" description="Start one to watch its loss, mAP and samples stream in live.">
        {#snippet actions()}<Button icon={Plus} href={href('/runs/new')}>New run</Button>{/snippet}
      </EmptyState>
    {:else if shown.length === 0}
      <EmptyState icon={Search} title="No runs match" description="Try another status filter or search term.">
        {#snippet actions()}<Button size="sm" onclick={() => ((filter = 'all'), (query = ''))}>Clear filters</Button>{/snippet}
      </EmptyState>
    {:else}
      <div class="scroll">
        <table class="table runs" class:selecting={selected.length > 0}>
          <thead>
            <tr>
              <th class="check"><span class="sr-only">Select for comparison</span></th>
              <th>Run</th>
              <th>Status</th>
              <th class="hide-md">Dataset</th>
              <th class="hide-md">Model</th>
              <th>Progress</th>
              <th class="r">Best mAP50-95</th>
              <th class="hide-lg">mAP trend</th>
              <th class="r hide-xl">Duration</th>
              <th class="r hide-sm">Created</th>
            </tr>
          </thead>
          <tbody>
            {#each shown as r (r.id)}
              {@const on = selected.includes(r.id)}
              <tr class="link" class:on onclick={() => router.go(`/runs/${r.id}`)}>
                <td class="check">
                  <input
                    type="checkbox"
                    checked={on}
                    disabled={!on && selected.length >= 4}
                    aria-label="Select {r.name} for comparison"
                    onclick={(e) => e.stopPropagation()}
                    onchange={() => toggle(r.id)}
                  />
                </td>
                <td>
                  <a class="name" href={href(`/runs/${r.id}`)} onclick={(e) => e.stopPropagation()}>{r.name}</a>
                  <span class="ctx">{dsName.get(r.config.dataset_id) ?? r.config.dataset_id} · <span class="mono">{r.config.arch}</span></span>
                </td>
                <td use:tooltip={failureCause(r)}><StatusBadge status={r.status} plain /></td>
                <td class="hide-md muted">{dsName.get(r.config.dataset_id) ?? r.config.dataset_id}</td>
                <td class="hide-md"><span class="mono muted">{r.config.arch}</span></td>
                <td>
                  <div class="prog">
                    <Progress value={r.status === 'running' ? runProgress(r) : r.epoch / r.epochs} size="xs" tone={tone(r)} label="{r.name} progress" />
                    <span class="num faint">{r.epoch}/{r.epochs}</span>
                  </div>
                </td>
                <td class="r num best">
                  {metric(r.best?.map50_95)}
                  {#if r.best}<span class="faint" title="Best epoch">ep {r.best.epoch}</span>{/if}
                </td>
                <td class="hide-lg"><RunSpark run={r} width={96} /></td>
                <td class="r num faint hide-xl">{r.status === 'queued' ? '—' : duration(elapsed(r))}</td>
                <td class="r faint hide-sm">{ago(r.created)}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
      <!-- Phones: stacked rows with the essentials (selection, name, status, score, progress). -->
      <div class="rows-head">
        <button type="button" class="pick" aria-pressed={choosing} onclick={() => (choosing ? ((picking = false), (selected = [])) : (picking = true))}
          >{choosing ? 'Cancel' : 'Select to compare'}</button
        >
        <span aria-hidden="true">Best mAP50-95</span>
      </div>
      <ul class="rows" class:choosing>
        {#each shown as r (r.id)}
          {@const on = selected.includes(r.id)}
          <li class:on>
            <label class="r-check">
              <input
                type="checkbox"
                checked={on}
                disabled={!on && selected.length >= 4}
                aria-label="Select {r.name} for comparison"
                onchange={() => toggle(r.id)}
              />
            </label>
            <a class="r-main" href={href(`/runs/${r.id}`)}>
              <span class="r-text">
                <span class="r-name"><StatusBadge status={r.status} plain dotOnly /><span class="name truncate">{r.name}</span></span>
                <span class="r-meta num truncate">{failureCause(r) ?? STATUS[r.status]} · {r.epoch}/{r.epochs} · {dsName.get(r.config.dataset_id) ?? r.config.dataset_id}</span>
              </span>
              <span class="r-end"><span class="r-val num">{metric(r.best?.map50_95)}</span><span class="r-age">{ago(r.created)}</span></span>
            </a>
            {#if r.status === 'running'}
              <span class="r-prog" aria-hidden="true" style:transform="scaleX({runProgress(r)})"></span>
            {/if}
          </li>
        {/each}
      </ul>
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
  .scroll {
    overflow-x: auto;
  }
  .sk {
    display: grid;
    gap: var(--space-4);
    padding: var(--space-4);
  }
  .runs td {
    height: 40px;
  }
  /* Selected for comparison: a 2 px accent bar on the row's leading edge. */
  /* The selection rule: neutral fill and a 2 px accent bar on the left. */
  .runs tr.on {
    background: var(--bg-active);
  }
  .runs tr.on td:first-child {
    box-shadow: inset 2px 0 0 var(--accent-line);
  }
  .check {
    width: 36px;
    padding-right: 0 !important;
  }
  input[type='checkbox'] {
    width: 14px;
    height: 14px;
  }
  @media (pointer: coarse) {
    input[type='checkbox'] {
      width: 20px;
      height: 20px;
    }
  }
  input[type='checkbox'] {
    accent-color: var(--accent-line);
    cursor: pointer;
  }
  .name {
    display: block;
    font-weight: var(--weight-medium);
  }
  .name:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
  }
  .prog {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 140px;
  }
  .prog span {
    font-size: var(--type-sm-size);
    min-width: 50px;
    text-align: right;
  }
  .best {
    font-weight: var(--weight-medium);
  }
  .best span {
    /* A fixed slot so 'ep 36' and 'ep 150' never shift the decimals. */
    display: inline-block;
    width: 6ch;
    text-align: left;
    font-weight: var(--weight-regular);
    font-size: var(--type-sm-size);
    margin-left: 6px;
  }
  .rows,
  .rows-head {
    display: none;
  }
  .ctx {
    display: none;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  @media (max-width: 1439px) {
    .hide-xl {
      display: none;
    }
  }
  @media (max-width: 1279px) {
    .hide-lg {
      display: none;
    }
  }
  @media (max-width: 1199px) {
    .hide-md {
      display: none;
    }
    .ctx {
      display: block;
    }
    .runs td {
      height: 52px;
    }
  }
  @media (max-width: 767px) {
    .scroll {
      display: none;
    }
    .rows {
      display: block;
    }
    .rows-head {
      display: flex;
      justify-content: space-between;
      gap: var(--space-3);
      height: 44px;
      align-items: center;
      padding: 0 var(--space-4);
      font-size: var(--type-sm-size);
      color: var(--text-tertiary);
      border-bottom: 1px solid var(--line-subtle);
    }
    /* Phones: 56 px rows (status and name, one meta line, the score on the right), like the desktop table's essentials. */
    .rows li {
      position: relative;
      display: flex;
      align-items: center;
      min-height: 56px;
      padding-right: var(--space-4);
      border-bottom: 1px solid var(--line-subtle);
    }
    /* Not choosing: no checkbox column, the row starts at the gutter. */
    .rows:not(.choosing) .r-check {
      display: none;
    }
    .rows:not(.choosing) li {
      padding-left: var(--space-4);
    }
    .pick {
      margin-right: auto;
      height: 44px;
      color: var(--text-secondary);
      font-size: var(--type-sm-size);
      text-decoration: underline;
      text-decoration-color: var(--line-strong);
      text-underline-offset: 3px;
    }
    .pick[aria-pressed='true'] {
      color: var(--text-primary);
    }
    .rows li:last-child {
      border-bottom: 0;
    }
    .rows li.on {
      background: var(--bg-active);
      box-shadow: inset 2px 0 0 var(--accent-line);
    }
    .r-check {
      display: grid;
      place-items: center;
      flex-shrink: 0;
      width: 44px;
      height: 56px;
    }
    .r-main {
      flex: 1;
      min-width: 0;
      display: flex;
      align-items: center;
      gap: var(--space-3);
      min-height: 56px;
    }
    .r-text {
      flex: 1;
      min-width: 0;
      display: grid;
      gap: 2px;
    }
    .r-name {
      display: flex;
      align-items: center;
      gap: var(--space-2);
      min-width: 0;
    }
    .r-meta {
      font-size: var(--type-sm-size);
      color: var(--text-tertiary);
    }
    .r-end {
      flex-shrink: 0;
      display: grid;
      justify-items: end;
      gap: 2px;
    }
    .r-val {
      font-weight: var(--weight-medium);
    }
    .r-age {
      font-size: var(--type-sm-size);
      color: var(--text-tertiary);
    }
    /* A live run shows its progress as a hairline along the row's bottom edge. */
    .r-prog {
      position: absolute;
      left: 0;
      right: 0;
      bottom: -1px;
      height: 2px;
      transform-origin: left;
      background: var(--accent-line);
    }
    .search {
      width: 100%;
    }
  }
</style>
