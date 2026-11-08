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
      <EmptyState icon={Activity} title="No runs yet" description="Train a detector on one of your datasets. Progress, curves and samples stream in live.">
        {#snippet actions()}<Button variant="primary" icon={Plus} href={href('/runs/new')}>New run</Button>{/snippet}
      </EmptyState>
    {:else if shown.length === 0}
      <EmptyState compact icon={Search} title="No runs match" description="Try another status filter or search term.">
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
              <th class="hide-lg">mAP per epoch</th>
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
                </td>
                <td><StatusBadge status={r.status} plain /></td>
                <td class="hide-md muted">{dsName.get(r.config.dataset_id) ?? r.config.dataset_id}</td>
                <td class="hide-md"><span class="mono muted">{r.config.arch}</span></td>
                <td>
                  <div class="prog">
                    <Progress value={r.status === 'running' ? r.step / Math.max(1, r.steps_per_epoch * r.epochs) : r.epoch / r.epochs} size="xs" tone={tone(r)} label="{r.name} progress" />
                    <span class="num faint">{r.epoch}/{r.epochs}</span>
                  </div>
                </td>
                <td class="r num best">
                  {metric(r.best?.map50_95)}
                  {#if r.best}<span class="faint" title="Best epoch">ep {r.best.epoch + 1}</span>{/if}
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
      <ul class="cards">
        {#each shown as r (r.id)}
          {@const on = selected.includes(r.id)}
          <li class:on>
            <label class="c-check">
              <input
                type="checkbox"
                checked={on}
                disabled={!on && selected.length >= 4}
                aria-label="Select {r.name} for comparison"
                onchange={() => toggle(r.id)}
              />
            </label>
            <a class="c-main" href={href(`/runs/${r.id}`)}>
              <span class="c-top">
                <span class="name">{r.name}</span>
                <span class="c-best num">
                  {metric(r.best?.map50_95)}
                  <span class="c-cap">{r.best ? `best mAP50-95 · ep ${r.best.epoch + 1}` : 'best mAP50-95'}</span>
                </span>
              </span>
              <span class="c-mid">
                <StatusBadge status={r.status} size="sm" />
                <span class="sub"><span>{dsName.get(r.config.dataset_id) ?? r.config.dataset_id}</span><span class="mono">{r.config.arch}</span></span>
              </span>
              <span class="c-prog">
                <Progress value={r.status === 'running' ? r.step / Math.max(1, r.steps_per_epoch * r.epochs) : r.epoch / r.epochs} size="xs" tone={tone(r)} label="{r.name} progress" />
                <span class="num faint">{r.epoch}/{r.epochs} · {ago(r.created)}</span>
              </span>
            </a>
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
  .runs tr.on {
    background: var(--bg-selected);
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
  /* Checkboxes appear on hover, on focus, or for every row once one is selected (Linear-style). */
  @media (hover: hover) {
    .runs:not(.selecting) tbody tr:not(:hover) .check input:not(:focus-visible):not(:checked) {
      opacity: 0;
    }
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
  .sub {
    display: flex;
    gap: var(--space-2);
    font-size: var(--type-xs-size);
    line-height: var(--type-xs-line);
    color: var(--text-tertiary);
  }
  .prog {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 140px;
  }
  .prog span {
    font-size: var(--type-xs-size);
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
    font-size: var(--type-xs-size);
    margin-left: 6px;
  }
  .cards {
    display: none;
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
  }
  @media (max-width: 767px) {
    .scroll {
      display: none;
    }
    .cards {
      display: block;
    }
    .cards li {
      display: flex;
      align-items: flex-start;
      gap: var(--space-3);
      padding: var(--space-3) var(--space-4);
      border-bottom: 1px solid var(--line-subtle);
    }
    .cards li:last-child {
      border-bottom: 0;
    }
    .cards li.on {
      background: var(--bg-selected);
    }
    .c-check {
      display: grid;
      place-items: center;
      width: 44px;
      height: 44px;
      margin: -12px 0 0 -14px;
    }
    .c-main {
      flex: 1;
      min-width: 0;
      display: grid;
      gap: 6px;
    }
    .c-top,
    .c-mid {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: var(--space-2);
      min-width: 0;
    }
    .c-mid {
      justify-content: flex-start;
    }
    .c-best {
      display: grid;
      justify-items: end;
      font-weight: var(--weight-medium);
    }
    .c-cap {
      font-size: var(--type-micro-size);
      line-height: var(--type-micro-line);
      font-weight: var(--weight-regular);
      color: var(--text-tertiary);
    }
    /* A fixed caption column, so every card's bar has the same length and progress compares down the list. */
    .c-prog {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 128px;
      align-items: center;
      gap: var(--space-3);
      font-size: var(--type-xs-size);
    }
    .c-prog > span {
      text-align: right;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .search {
      width: 100%;
    }
  }
</style>
