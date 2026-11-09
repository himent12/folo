<script lang="ts">
  import { MediaQuery } from 'svelte/reactivity';
  import Plus from '@lucide/svelte/icons/plus';
  import Search from '@lucide/svelte/icons/search';
  import Tags from '@lucide/svelte/icons/tags';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { LabelJob } from '$lib/api/types';
  import { media } from '$lib/design/tokens';
  import { ago, num } from '$lib/format';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
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
  import Progress from '$lib/ui/Progress.svelte';
  import { rowNav } from '$lib/ui/rowNav';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';
  import TableState from '$lib/ui/TableState.svelte';
  import JobDialog from './label/JobDialog.svelte';
  import { cannotLabel } from './label/support';
  import Unsupported from './label/Unsupported.svelte';
  import { jobStatus, runAgent } from './label/words';

  setCrumbs({ label: 'Datasets', path: '/datasets' }, { label: 'Label images' });
  ownCreateAction();

  const system = new Resource((s) => hub.api.system(s));
  // Polled: a job an agent is labeling moves on its own.
  const jobs = new Resource((s) => hub.api.labelJobs(s), { poll: 5000 });
  const agents = new Resource((s) => hub.api.labelAgents(s));
  const unsupported = $derived(cannotLabel(system.data, jobs.error));

  // `#/datasets/label?new=1` opens the dialog (the palette's "New labeling job").
  let creating = $state(router.route.query.get('new') === '1');

  let query = $state('');
  let filter = $state<'all' | 'active' | 'idle'>('all');
  const active = (j: LabelJob) => j.status !== 'idle';
  const count = (f: typeof filter) => (jobs.data ?? []).filter((j) => f === 'all' || (f === 'active') === active(j)).length;
  const shown = $derived.by(() => {
    const q = query.trim().toLowerCase();
    return (jobs.data ?? []).filter((j) => (filter === 'all' || (filter === 'active') === active(j)) && (!q || `${j.name} ${j.classes.join(' ')}`.toLowerCase().includes(q)));
  });
  const totals = $derived({ images: shown.reduce((a, j) => a + j.counts.images, 0), labeled: shown.reduce((a, j) => a + j.counts.labeled, 0) });
  /** Columns of the desktop table (for state rows that span it), the empty last one included. */
  const COLS = 7;

  const offline = $derived(!jobs.data && isOffline(jobs.error));
  const hasData = $derived(jobs.data ? jobs.data.length > 0 : !jobs.error);
  const hasRows = $derived(jobs.data ? shown.length > 0 : !jobs.error);
  const phone = new MediaQuery(media.phone, false);
  const open = (j: LabelJob) => router.go(`/datasets/label/${j.id}`);
</script>

<Page title="Label images">
  {#snippet sub()}<span>Unlabeled images, boxed by Claude Code or Codex, reviewed by you, then a dataset.</span>{/snippet}
  {#snippet subPhone()}<span>Agent-drawn boxes.</span>{/snippet}
  {#snippet actions()}
    {#if !unsupported}
      <Gate reason={offline ? OFFLINE_REASON : null}>
        {#snippet children(why)}
          <Button variant="primary" icon={Plus} disabled={!!why} aria-describedby={why} onclick={() => (creating = true)}>{phone.current ? 'New job' : 'New labeling job'}</Button>
        {/snippet}
      </Gate>
    {/if}
  {/snippet}

  {#if unsupported}
    <Unsupported />
  {:else}
    <div class="list-toolbar">
      <div class="lt-filter"><Segmented
        label="Filter labeling jobs by status"
        bind:value={filter}
        disabled={!hasData}
        options={[
          { value: 'all', label: 'All', meta: jobs.data?.length ? String(count('all')) : undefined },
          { value: 'active', label: 'Labeling', meta: jobs.data?.length ? String(count('active')) : undefined },
          { value: 'idle', label: 'Idle', meta: jobs.data?.length ? String(count('idle')) : undefined },
        ]}
      /></div>
      <div class="lt-search"><Input bind:value={query} icon={Search} placeholder="Filter by name or class" aria-label="Filter labeling jobs" type="search" disabled={!hasData} /></div>
    </div>

    <Panel pad="none">
      <div class="scroll" class:state={!hasRows}>
        <table class="table jobs" aria-busy={!jobs.data} aria-label="Labeling jobs" use:rowNav>
          {#if hasRows}<thead>
            <tr>
              <!-- One geometry in every state: the widths sit on the header row loading and loaded share; from 1280 the
                   name is bounded and the slack goes to the empty last column. -->
              <th scope="col" class="name-col">Name</th>
              <th scope="col" class="r" style:width="88px">Images</th>
              <th scope="col" style:width="176px">Labeled</th>
              <th scope="col" style:width="136px">Status</th>
              <th scope="col" class="hide-md" style:width="184px">Agent</th>
              <th scope="col" class="r hide-md" style:width="104px">Created</th>
              <th class="hide-lg rest" aria-hidden="true"></th>
            </tr>
          </thead>{/if}
          <tbody>
            {#if jobs.error && !jobs.data}
              <TableState cols={COLS}><ErrorState error={jobs.error} title="Could not load labeling jobs" retry={() => jobs.refresh()} /></TableState>
            {:else if jobs.data?.length === 0}
              <TableState cols={COLS}>
                <EmptyState icon={Tags} title="No labeling jobs yet" description="Add images, name the classes, and Claude Code or Codex draws the boxes for you to review.">
                  {#snippet actions()}<Button icon={Plus} onclick={() => (creating = true)}>New labeling job</Button>{/snippet}
                </EmptyState>
              </TableState>
            {:else if jobs.data && shown.length === 0}
              <TableState cols={COLS}>
                <EmptyState icon={Search} title="No labeling jobs match" description="Try another status or search term.">
                  {#snippet actions()}<Button size="sm" onclick={() => ((query = ''), (filter = 'all'))}>Clear filters</Button>{/snippet}
                </EmptyState>
              </TableState>
            {:else if !jobs.data}
              {#each [0, 1, 2] as i (i)}
                <tr aria-hidden="true">
                  <td><Skeleton w="120px" h="12px" /></td>
                  <td class="r"><Skeleton w="28px" h="12px" /></td>
                  <td><span class="lab"><Skeleton w="28px" h="12px" /><Skeleton h="4px" r="999px" /></span></td>
                  <td><Skeleton w="64px" h="12px" /></td>
                  <td class="hide-md"><Skeleton w="120px" h="12px" /></td>
                  <td class="r hide-md"><Skeleton w="44px" h="12px" /></td>
                  <td class="hide-lg rest"></td>
                </tr>
              {/each}
            {:else}
              {#each shown as j (j.id)}
                {@const st = jobStatus(j)}
                <tr class="link" data-row aria-label={j.name} onclick={() => open(j)}>
                  <td class="name-cell"><a class="name truncate" href={href(`/datasets/label/${j.id}`)} onclick={(e) => e.stopPropagation()} title={j.name}>{j.name}</a></td>
                  <td class="r num muted">{num(j.counts.images)}</td>
                  <td>
                    <span class="lab">
                      <span class="num">{num(j.counts.labeled)}</span>
                      <Progress value={j.counts.images ? j.counts.labeled / j.counts.images : 0} tone={active(j) ? 'accent' : 'neutral'} label="{j.counts.labeled} of {j.counts.images} images labeled" />
                    </span>
                  </td>
                  <td><StatusBadge plain status={st.status} label={st.label} /></td>
                  <td class="hide-md">
                    {#if j.run}<Meta line><span>{runAgent(j.run, agents.data)[0]}</span><span>{runAgent(j.run, agents.data)[1]}</span></Meta>{:else}<NoValue label="no run yet" />{/if}
                  </td>
                  <td class="r faint hide-md">{ago(j.created)}</td>
                  <td class="hide-lg rest" aria-hidden="true"></td>
                </tr>
              {/each}
            {/if}
          </tbody>
        </table>
      </div>
      <!-- Phones: the name, then how far labeling has got and the status; one line each, whole items. -->
      {#if shown.length}
        <ul class="rows" aria-label="Labeling jobs">
          {#each shown as j (j.id)}
            {@const st = jobStatus(j)}
            <li class="m-row">
              <a href={href(`/datasets/label/${j.id}`)}>
                <span class="m-main">
                  <span class="name truncate">{j.name}</span>
                  <span class="m-meta num"><Meta><span>{num(j.counts.labeled)} of {num(j.counts.images)} labeled</span><span>{ago(j.created)}</span></Meta></span>
                </span>
                <span class="m-side"><StatusBadge plain status={st.status} label={st.label} /></span>
              </a>
            </li>
          {/each}
        </ul>
      {:else if !jobs.data && !jobs.error}
        <ul class="rows" aria-hidden="true">
          {#each [0, 1, 2] as i (i)}
            <li><span class="row-sk"><span class="m-main"><Skeleton w="55%" h="14px" /><Skeleton w="45%" h="10px" /></span><Skeleton w="64px" h="12px" /></span></li>
          {/each}
        </ul>
      {/if}
    </Panel>
    {#if jobs.data && shown.length}
      <p class="list-foot num">
        <Meta>
          <span>{shown.length < jobs.data.length ? `${shown.length} of ${jobs.data.length}` : shown.length} {jobs.data.length === 1 ? 'job' : 'jobs'}</span>
          <span>{num(totals.images)} images</span>
          <span>{num(totals.labeled)} labeled</span>
        </Meta>
      </p>
    {/if}
  {/if}
</Page>

<JobDialog bind:open={creating} />

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
  @media (min-width: 1280px) {
    .name-col {
      width: 240px;
    }
  }
  .rest {
    padding: 0;
  }
  /* The count and its bar on one line: the bar takes what the count leaves. */
  .lab {
    display: grid;
    grid-template-columns: 40px minmax(0, 1fr);
    align-items: center;
    gap: var(--space-2);
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
    .m-row > a,
    .row-sk {
      display: flex;
      align-items: center;
      gap: var(--space-3);
      min-height: 56px;
      padding: var(--space-2) var(--space-4);
    }
    .m-row > a:active {
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
    }
  }
</style>
