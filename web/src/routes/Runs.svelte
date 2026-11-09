<script lang="ts">
  import Activity from '@lucide/svelte/icons/activity';
  import GitCompare from '@lucide/svelte/icons/git-compare-arrows';
  import Plus from '@lucide/svelte/icons/plus';
  import Search from '@lucide/svelte/icons/search';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Model, Run, RunStatus } from '$lib/api/types';
  import RunSpark from '$lib/charts/RunSpark.svelte';
  import { ago, metric } from '$lib/format';
  import { failureCause } from '$lib/oom';
  import { tooltip } from '$lib/ui/tooltip';
  import { href, router } from '$lib/router.svelte';
  import { ownCreateAction, setCrumbs } from '$lib/shell/crumbs.svelte';
  import Page from '$lib/shell/Page.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Progress from '$lib/ui/Progress.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import { runProgress } from '$lib/api/progress';
  import { rowNav } from '$lib/ui/rowNav';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';
  import TableState from '$lib/ui/TableState.svelte';
  import Gate from '$lib/ui/Gate.svelte';
  import Ident from '$lib/ui/Ident.svelte';
  import Lineage from '$lib/ui/Lineage.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import { isOffline, OFFLINE_REASON } from '$lib/ui/offline';
  import { origin } from '$lib/lineage';
  import Menu from '$lib/ui/Menu.svelte';
  import { hasFeature, queuePlace, withQueueMove } from '$lib/api/queue';
  import type { QueueRequest } from '$lib/api/types';
  import { moveQueued, queueItems } from './run/queueActions';

  setCrumbs({ label: 'Runs' });
  // The header's create action is the one in the top-right corner (the top bar's New run steps aside).
  ownCreateAction();

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
  // Only to name the model a run was fine-tuned from; the list renders without it.
  const models = new Resource((s) => hub.api.models(s));
  const modelById = $derived(new Map<string, Model>((models.data ?? []).map((m) => [m.id, m])));
  /** Columns of the desktop table (for state rows that span it). */
  const COLS = 11;

  // Queue control (servers that list `queue_control`): a waiting run's row says its place and has a ⋯ menu with
  // Start next, Move up and Move down. The new order shows at once; a refusal puts the old one back.
  const system = new Resource((s) => hub.api.system(s));
  const queueCtl = $derived(hasFeature(system.data, 'queue_control'));
  const waiting = $derived((runs.data ?? []).filter((r) => r.status === 'queued' && r.queue_position != null).length);
  const training = $derived(runs.data?.find((r) => r.status === 'running' || r.status === 'stopping') ?? null);
  const placeOf = (r: Run) => (queueCtl && r.status === 'queued' ? (r.queue_position ?? null) : null);
  function move(r: Run, req: QueueRequest) {
    void moveQueued(
      r,
      req,
      () => {
        const list = runs.data;
        if (list) runs.set(withQueueMove(list, r.id, req));
        return () => {
          if (list) runs.set(list);
        };
      },
      () => void runs.refresh(),
      training,
    ).then((done) => {
      if (done) void runs.refresh();
    });
  }
  const rowItems = (r: Run) => {
    const at = placeOf(r);
    return at === null ? [] : queueItems(at, waiting, training, (req) => move(r, req));
  };

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
  // Nothing answers: the actions that need the server are disabled and say why (as on Datasets).
  const offline = $derived(!runs.data && isOffline(runs.error));
</script>

<Page title="Runs">
  {#snippet sub()}<span>Training runs, newest first. Select up to four to compare their curves.</span>{/snippet}
  <!-- Phones: one header row (this line and New run); Compare lives in the list's own header row, by the checkboxes. -->
  {#snippet subPhone()}<span>Training runs, newest first.</span>{/snippet}
  {#snippet actions()}
    {#if runs.data?.length !== 0}
      <span class="compare">
        <Gate reason={offline ? OFFLINE_REASON : selected.length < 2 ? 'Select 2–4 runs to compare' : null}>
          {#snippet children(why)}
            <Button variant={selected.length ? 'primary' : 'secondary'} icon={GitCompare} disabled={!!why} aria-describedby={why} href={href(`/runs/compare?ids=${selected.join(',')}`)}>
              Compare{#if selected.length}<span class="num count">{selected.length}</span>{/if}
            </Button>
          {/snippet}
        </Gate>
      </span>
      <Gate reason={offline ? OFFLINE_REASON : null}>
        {#snippet children(why)}
          <!-- While runs are selected, Compare is the page's one primary action and New run steps back. -->
          <Button variant={selected.length ? 'secondary' : 'primary'} icon={Plus} disabled={!!why} aria-describedby={why} href={href('/runs/new')}>New run</Button>
        {/snippet}
      </Gate>
    {/if}
  {/snippet}

  <!-- Filters only with runs to filter (or while they load): not over an empty or failed list, as on Models and Datasets. -->
  {#if runs.data ? runs.data.length > 0 : !runs.error}<div class="list-toolbar">
    <div class="lt-filter"><Segmented
      label="Filter runs by status"
      bind:value={filter}
      options={[
        { value: 'all', label: 'All', meta: runs.data ? String(count('all')) : undefined },
        { value: 'active', label: 'Active', meta: runs.data ? String(count('active')) : undefined },
        { value: 'completed', label: 'Finished', meta: runs.data ? String(count('completed')) : undefined },
        { value: 'failed', label: 'Failed', meta: runs.data ? String(count('failed')) : undefined },
      ]}
    /></div>
    <div class="lt-search"><Input icon={Search} bind:value={query} placeholder="Filter by name, model or dataset" aria-label="Filter runs" /></div>
  </div>{/if}

  <Panel pad="none">
    <!-- One table frame for every state: the header row stays, and an empty or failed list says so inside it. -->
    <div class="scroll" class:state={runs.data ? shown.length === 0 : !!runs.error}>
      <!-- Row checkboxes show on hover and focus, on selected rows, and on every row once one is selected (picking). -->
      <table
        class="table runs"
        class:picking={selected.length > 0}
        aria-label="Runs"
        aria-busy={!runs.data}
        use:rowNav={{ toggle: (row) => toggle(row.dataset.id ?? '') }}
      >
        <thead>
          <tr>
            <!-- One geometry in every state (DESIGN_DIRECTION 7): each column's width is fixed here, on the header row
                 that loading, empty, error and loaded share; the run's name takes what is left. -->
            <th class="check"><span class="sr-only">Select for comparison</span></th>
            <th>Run</th>
            <th style:width="112px">Status</th>
            <th class="hide-lg" style:width="112px">Dataset</th>
            <th class="hide-lg" style:width="80px">Model</th>
            <th class="hide-lg" style:width="128px">Started from</th>
            <th style:width="136px">Progress</th>
            <th class="r" style:width="128px">Best mAP50-95</th>
            <th class="hide-lg" style:width="120px">mAP trend</th>
            <th class="r hide-md" style:width="96px">Created</th>
            <th class="act" style:width="40px"><span class="sr-only">Actions</span></th>
          </tr>
        </thead>
        <tbody>
          {#if runs.error && !runs.data}
            <TableState cols={COLS}><ErrorState error={runs.error} retry={() => runs.refresh()} title="Could not load runs" /></TableState>
          {:else if !runs.data}
            {#each [0, 1, 2, 3, 4] as i (i)}
              <tr aria-hidden="true">
                <td class="check"></td>
                <td class="lead"><Skeleton w="112px" h="12px" /></td>
                <td><Skeleton w="72px" h="12px" /></td>
                <td class="hide-lg"><Skeleton w="64px" h="12px" /></td>
                <td class="hide-lg"><Skeleton w="48px" h="12px" /></td>
                <td class="hide-lg"><Skeleton w="88px" h="12px" /></td>
                <td><div class="prog"><Skeleton h="4px" /><span><Skeleton w="40px" h="10px" /></span></div></td>
                <td class="r"><Skeleton w="72px" h="12px" /></td>
                <td class="hide-lg"><Skeleton w="96px" h="12px" /></td>
                <td class="r hide-md"><Skeleton w="56px" h="12px" /></td>
                <td class="act"></td>
              </tr>
            {/each}
          {:else if runs.data.length === 0}
            <TableState cols={COLS}>
              <!-- The header steps its actions aside over an empty list, so this is the page's one primary. -->
              <EmptyState icon={Activity} title="No runs yet" description="Start one to watch its loss, mAP and samples stream in live.">
                {#snippet actions()}<Button variant="primary" icon={Plus} href={href('/runs/new')}>New run</Button>{/snippet}
              </EmptyState>
            </TableState>
          {:else if shown.length === 0}
            <TableState cols={COLS}>
              <EmptyState icon={Search} title="No runs match" description="Try another status filter or search term.">
                {#snippet actions()}<Button size="sm" onclick={() => ((filter = 'all'), (query = ''))}>Clear filters</Button>{/snippet}
              </EmptyState>
            </TableState>
          {:else}
            {#each shown as r (r.id)}
              {@const on = selected.includes(r.id)}
              {@const from = origin(r.config.weights, modelById)}
              <tr class="link" class:on data-row data-id={r.id} aria-label={r.name} onclick={() => router.go(`/runs/${r.id}`)}>
                <!-- The whole cell is the checkbox's target (44 px on touch); a click in it never opens the run. -->
                <td class="check" onclick={(e) => e.stopPropagation()}>
                  <label class="hit">
                    <input
                      type="checkbox"
                      checked={on}
                      disabled={!on && selected.length >= 4}
                      aria-label="Select {r.name} for comparison"
                      onchange={() => toggle(r.id)}
                    />
                  </label>
                </td>
                <td class="lead">
                  <a class="name" href={href(`/runs/${r.id}`)} onclick={(e) => e.stopPropagation()}>{r.name}</a>
                  <span class="sub ctx"><Meta line><span>{dsName.get(r.config.dataset_id) ?? r.config.dataset_id}</span><Ident value={r.config.arch} /><span>from <Lineage origin={from} link={false} scratch="scratch" /></span></Meta></span>
                </td>
                <td use:tooltip={failureCause(r)}><StatusBadge status={r.status} plain /></td>
                <td class="hide-lg muted">{dsName.get(r.config.dataset_id) ?? r.config.dataset_id}</td>
                <td class="hide-lg"><Ident value={r.config.arch} class="muted" /></td>
                <td class="hide-lg from">
                  {#if from.kind === 'model' && modelById.has(from.id)}
                    <Ident value={from.label} href={href(`/models/${from.id}`)} onclick={(e) => e.stopPropagation()} />
                  {:else if from.kind === 'file'}
                    <Ident value={from.label} title={from.path} />
                  {:else if from.kind === 'scratch'}
                    <span class="faint">From scratch</span>
                  {:else}
                    <Ident value={from.label} class="faint" />
                  {/if}
                </td>
                <td>
                  {#if placeOf(r) !== null}
                    <!-- A waiting run has no progress yet: its place in the queue instead. -->
                    <span class="prog place">{queuePlace(placeOf(r) ?? 0)}</span>
                  {:else}
                    <div class="prog">
                      <Progress value={r.status === 'running' ? runProgress(r) : r.epoch / r.epochs} size="xs" tone={tone(r)} label="{r.name} progress" />
                      <span class="num faint">{r.epoch}/{r.epochs}</span>
                    </div>
                  {/if}
                </td>
                <td class="r num best">
                  {metric(r.best?.map50_95)}
                  {#if r.best}<span class="faint" title="Best epoch">ep {r.best.epoch}</span>{/if}
                </td>
                <td class="hide-lg"><RunSpark run={r} width={96} /></td>
                <td class="r faint hide-md">{ago(r.created)}</td>
                <!-- A click in the cell never opens the run. -->
                <td class="act" onclick={(e) => e.stopPropagation()}>
                  {#if rowItems(r).length}<Menu label="Queue actions for {r.name}" items={rowItems(r)} variant="ghost" fixed />{/if}
                </td>
              </tr>
            {/each}
          {/if}
        </tbody>
      </table>
    </div>
    {#if shown.length}
      <!-- Phones: stacked rows with the essentials (selection, name, status, score, progress). -->
      <!-- Select mode: Compare (with its count) the primary once a run is selected, Cancel a ghost one, both 44 px targets
           on touch. -->
      <div class="rows-head">
        {#if choosing}
          <Button variant="ghost" onclick={() => ((picking = false), (selected = []))}>Cancel</Button>
          <Gate reason={offline ? OFFLINE_REASON : selected.length < 2 ? 'Select 2–4 runs to compare' : null}>
            {#snippet children(why)}
              <Button variant={selected.length ? 'primary' : 'secondary'} icon={GitCompare} disabled={!!why} aria-describedby={why} href={href(`/runs/compare?ids=${selected.join(',')}`)}>
                Compare{#if selected.length}<span class="num count">{selected.length}</span>{/if}
              </Button>
            {/snippet}
          </Gate>
        {:else}
          <Button variant="ghost" icon={GitCompare} onclick={() => (picking = true)}>Select to compare</Button>
        {/if}
        <span class="col" aria-hidden="true">Best mAP50-95</span>
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
                <!-- With the checkboxes in, the dataset steps out so the cause and progress keep the line. -->
                <span class="r-meta num truncate"><Meta line><span>{failureCause(r) ?? (placeOf(r) !== null ? queuePlace(placeOf(r) ?? 0) : STATUS[r.status])}</span><span>{r.epoch}/{r.epochs}</span>{#if !choosing}<span>{dsName.get(r.config.dataset_id) ?? r.config.dataset_id}</span>{/if}</Meta></span>
              </span>
              <span class="r-end"><span class="r-val num">{metric(r.best?.map50_95)}</span><span class="r-age">{ago(r.created)}</span></span>
            </a>
            {#if !choosing && rowItems(r).length}<span class="r-menu"><Menu label="Queue actions for {r.name}" items={rowItems(r)} variant="ghost" fixed /></span>{/if}
            {#if r.status === 'running'}
              <span class="r-prog" aria-hidden="true" style:transform="scaleX({runProgress(r)})"></span>
            {/if}
          </li>
        {/each}
      </ul>
    {:else if !runs.data && !runs.error}
      <!-- Phones, loading: the stacked rows' shape (name over one meta line, the score on the right). -->
      <ul class="rows sk" aria-hidden="true">
        {#each [0, 1, 2, 3, 4] as i (i)}
          <li>
            <span class="r-main">
              <span class="r-text"><Skeleton w="55%" h="14px" /><Skeleton w="75%" h="10px" /></span>
              <span class="r-end"><Skeleton w="40px" h="14px" /></span>
            </span>
          </li>
        {/each}
      </ul>
    {/if}
  </Panel>
</Page>

<style>
  /* Compare's count: part of the label, one weight lighter. */
  .count {
    font-weight: var(--weight-regular);
  }
  .compare {
    display: contents;
  }
  .scroll {
    overflow-x: auto;
  }
  /* The checkbox has a real 32 px column, there whether the boxes show or not, so revealing them never moves a thing:
     the box 12 px from the panel's edge, the name 14 px after it (its cell's leading padding 8). The whole cell is the
     box's target. */
  .runs .check {
    position: relative;
    /* A real 32 px column: never squeezed below it when the table is tight (tablet). */
    width: 32px;
    min-width: 32px;
    padding: 0;
  }

  .runs :is(th, td):nth-child(2) {
    padding-left: var(--space-2);
  }
  .hit {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    padding-left: var(--space-3);
    cursor: pointer;
  }
  input[type='checkbox'] {
    width: 14px;
    height: 14px;
    margin: 0;
    /* Neutral: a selected row's lime is its leading bar alone (the fill is ink, the check inverted). */
    accent-color: var(--text-primary);
    cursor: pointer;
  }
  .runs .check input {
    opacity: 0;
    transition: opacity var(--dur-fast) var(--ease-standard);
  }
  /* Shown on the row under the pointer or the keyboard (so X is discoverable), on selected rows, and on every row once
     any is selected. */
  .runs tbody tr:is(:hover, :focus-within, .on) .check input,
  .runs.picking .check input,
  .runs .check input:focus-visible {
    opacity: 1;
  }
  /* Touch: no hover to reveal them, so always shown, 20 px, in a 44 px wide cell of a 48 px row. */
  @media (pointer: coarse) {
    .runs .check {
      width: 44px;
    }
    .runs .check input {
      opacity: 1;
    }
    .hit {
      position: static;
      min-height: 48px;
    }

    input[type='checkbox'] {
      width: 20px;
      height: 20px;
    }
    .runs td {
      height: 48px;
    }
  }
  .name {
    display: block;
  }
  .from :global(.ident) {
    color: var(--text-secondary);
  }
  .prog {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    /* One width at every tier. */
    width: 112px;
  }
  .prog span {
    font-size: var(--type-sm-size);
    min-width: 50px;
    text-align: right;
  }
  .best {
    font-weight: var(--weight-medium);
  }
  .place {
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  /* The row menu's column: as wide as its ⋯ button, so rows without one keep the same edge. */
  .runs .act {
    width: 40px;
    padding: 0 var(--space-2) 0 0;
    text-align: right;
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
  }
  /* Below 1280 the Dataset, Model, Started from, and trend columns hide (.hide-lg, base.css). */
  @media (max-width: 1279px) {
    .ctx {
      display: block;
    }
    /* The name carries dataset, model and origin on a second line: the two-line row height. */
    .runs td {
      height: 48px;
    }
  }
  /* Below 1024 Created hides too (.hide-md), so the table fits from 768 px up, checkbox gutter included. */
  @media (max-width: 767px) {
    /* Phones show the stacked rows below; the table frame stays only to hold an empty or error state. */
    .scroll:not(.state),
    .scroll.state thead {
      display: none;
    }
    .rows {
      display: block;
    }
    .rows-head {
      display: flex;
      justify-content: space-between;
      gap: var(--space-2);
      height: 48px;
      align-items: center;
      padding: 0 var(--space-4) 0 var(--space-1);
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
    .compare {
      display: none;
    }
    .col {
      margin-left: auto;
    }
    .rows li:last-child {
      border-bottom: 0;
    }
    .rows li.on {
      background: var(--bg-active);
      box-shadow: inset 2px 0 0 var(--accent-line);
    }
    .r-menu {
      flex-shrink: 0;
      margin-left: var(--space-2);
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
    .r-name .name {
      font-weight: var(--weight-medium);
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
    .sk .r-text {
      gap: var(--space-2);
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
  }
</style>
