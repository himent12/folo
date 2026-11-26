// The labeling jobs list (`#/datasets/label`), filtered by state (active, failed, idle), and the New labeling job
// dialog (`?new=1`). The parts are in ./label/.
import Plus from '$icons/plus';
import Search from '$icons/search';
import Tags from '$icons/tags';
import { computed, css, effect, For, MediaQuery, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { LabelJob } from '$lib/api/types';
import { LABEL_FEATURE } from '$lib/label';
import { media } from '$lib/design/tokens';
import { ago, count as counted, num } from '$lib/format';
import { href, router } from '$lib/router';
import Page from '$lib/shell/Page';
import { ownCreateAction, setCrumbs } from '$lib/shell/crumbs';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Gate from '$lib/ui/Gate';
import Input from '$lib/ui/Input';
import Meta from '$lib/ui/Meta';
import NoValue from '$lib/ui/NoValue';
import { isOffline, OFFLINE_REASON } from '$lib/ui/offline';
import Panel from '$lib/ui/Panel';
import Progress from '$lib/ui/Progress';
import { rowNav } from '$lib/ui/rowNav';
import Segmented from '$lib/ui/Segmented';
import Skeleton from '$lib/ui/Skeleton';
import StatusBadge from '$lib/ui/StatusBadge';
import TableState from '$lib/ui/TableState';
import JobDialog from './label/JobDialog';
import { cannotLabel } from './label/support';
import Unsupported from './label/Unsupported';
import { jobStatus, runAgent } from './label/words';

type Filter = 'all' | 'active' | 'failed' | 'idle';

/** Columns of the desktop table (for state rows that span it), the empty last one included. */
const COLS = 7;

/** The filter a job is listed under besides All: an agent at work, a last run that failed, else idle. */
const filterOf = (j: LabelJob): Exclude<Filter, 'all'> => (j.status !== 'idle' ? 'active' : jobStatus(j).status === 'failed' ? 'failed' : 'idle');

export default function LabelJobs() {
  setCrumbs({ label: 'Datasets', path: '/datasets' }, { label: 'Label images' });
  ownCreateAction();

  const system = new Resource((s) => hub.api.system(s));
  // Polled: a job an agent is labeling moves on its own.
  const jobs = new Resource((s) => hub.api.labelJobs(s), { poll: 5000 });
  const agents = new Resource((s) => hub.api.labelAgents(s));
  const unsupported = computed(() => cannotLabel(system.data, jobs.error));

  // `#/datasets/label?new=1` opens the dialog (the palette's "New labeling job").
  // Followed, not read once: the palette can ask while this page is open (pages are keyed by path, not query). The ask
  // leaves the URL at once, so the next one is a change again. Not on a server whose health said it cannot label
  // (`folo serve --mock`): the page says so instead of a form whose Create can only fail (#324).
  const creating = signal(false);
  const askedNew = computed(() => router.route.query.get('new') === '1');
  effect(() => {
    if (!askedNew.get()) return;
    untrack(() => {
      if (!hub.lacks(LABEL_FEATURE)) creating.set(true);
      router.go('/datasets/label', { replace: true });
    });
  });

  const query = signal('');
  const filter = signal<Filter>('all');
  const counts = computed(
    () => {
      let all = 0;
      let active = 0;
      let failed = 0;
      let idle = 0;
      for (const j of jobs.data ?? []) {
        all++;
        const k = filterOf(j);
        if (k === 'active') active++;
        else if (k === 'failed') failed++;
        else if (k === 'idle') idle++;
      }
      return { all, active, failed, idle };
    },
    { equals: (a, b) => a.all === b.all && a.active === b.active && a.failed === b.failed && a.idle === b.idle },
  );
  const count = (f: Filter) => counts.get()[f];
  const shown = computed(() => {
    const q = query.get().trim().toLowerCase();
    const f = filter.get();
    return (jobs.data ?? []).filter((j) => (f === 'all' || filterOf(j) === f) && (!q || `${j.name} ${j.classes.join(' ')}`.toLowerCase().includes(q)));
  });
  const totals = computed(() => ({
    images: shown.get().reduce((a, j) => a + j.counts.images, 0),
    labeled: shown.get().reduce((a, j) => a + j.counts.labeled, 0),
  }));

  const offline = computed(() => !jobs.data && isOffline(jobs.error));
  const hasData = computed(() => (jobs.data ? jobs.data.length > 0 : !jobs.error));
  const hasRows = computed(() => (jobs.data ? shown.get().length > 0 : !jobs.error));
  const phone = new MediaQuery(media.phone, false);
  const searchEl = signal<HTMLInputElement | undefined>(undefined);
  const clearSeq = signal(0);
  const filterStatus = signal('');
  const jobsReady = computed(() => jobs.data !== undefined);
  let hasFiltered = false;
  let lastClear = 0;
  effect(() => {
    const f = filter.get();
    const q = query.get().trim();
    const c = clearSeq.get();
    const ready = jobsReady.get();
    const active = f !== 'all' || q.length > 0;
    const cleared = c !== lastClear;
    lastClear = c;
    if (active || cleared) hasFiltered = true;
    if (!hasFiltered || !ready) return;
    const msg = untrack(() => {
      const all = jobs.data;
      if (!all) return '';
      const n = shown.get().length;
      const summary =
        n < all.length
          ? `Showing ${n} of ${all.length} ${all.length === 1 ? 'labeling job' : 'labeling jobs'}.`
          : `Showing all ${all.length} ${all.length === 1 ? 'labeling job' : 'labeling jobs'}.`;
      return cleared ? `Cleared labeling job filters. ${summary}` : summary;
    });
    filterStatus.set(msg);
  });
  function clearFilters() {
    clearSeq.set(clearSeq.peek() + 1);
    query.set('');
    filter.set('all');
    queueMicrotask(() => {
      const input = searchEl.peek();
      if (input && input.isConnected && !input.disabled) input.focus();
    });
  }
  const open = (j: LabelJob) => router.go(`/datasets/label/${j.id}`);

  return (
    <>
      <Page
        title="Label images"
        sub={() => <span>Unlabeled images, boxed by Claude Code, Codex or Antigravity, reviewed by you, then a dataset.</span>}
        subPhone={() => <span>Agent-drawn boxes.</span>}
        actions={() => (
          <>
            {!unsupported.get() ? (
              <Gate reason={offline.get() ? OFFLINE_REASON : null}>
                {(why) => (
                  <Button variant="primary" icon={Plus} disabled={!!why()} aria-describedby={why()} onclick={() => creating.set(true)}>
                    {phone.current ? 'New job' : 'New labeling job'}
                  </Button>
                )}
              </Gate>
            ) : null}
          </>
        )}
      >
        {unsupported.get() ? (
          <Unsupported />
        ) : (
          <>
            <div class="list-toolbar">
              <div class="lt-filter">
                <Segmented
                  label="Filter labeling jobs by status"
                  bind:value={filter}
                  disabled={!hasData.get()}
                  options={[
                    { value: 'all', label: 'All', meta: jobs.data?.length ? String(count('all')) : undefined },
                    { value: 'active', label: 'Labeling', meta: jobs.data?.length ? String(count('active')) : undefined },
                    { value: 'failed', label: 'Failed', meta: jobs.data?.length ? String(count('failed')) : undefined },
                    { value: 'idle', label: 'Idle', meta: jobs.data?.length ? String(count('idle')) : undefined },
                  ]}
                />
              </div>
              <div class="lt-search">
                <Input
                  bind:value={query}
                  bind:el={searchEl}
                  icon={Search}
                  placeholder="Filter by name or class"
                  aria-label="Filter labeling jobs"
                  type="search"
                  disabled={!hasData.get()}
                />
              </div>
            </div>

            <Panel pad="none">
              {phone.current && hasRows.get() ? (
                shown.get().length ? (
                  /* Phones: the name, then how far labeling has got and the status; one line each, whole items. */
                  <ul class="rows" aria-label="Labeling jobs">
                    <For each={shown.get()} key={(j) => j.id}>
                      {(j) => {
                        const st = () => jobStatus(j());
                        return (
                          <li class="m-row">
                            <a href={href(`/datasets/label/${j().id}`)}>
                              <span class="m-main">
                                <span class="name truncate">{j().name}</span>
                                <span class="m-meta num">
                                  <Meta>
                                    <span>
                                      {num(j().counts.labeled)} of {num(j().counts.images)} labeled
                                    </span>
                                    <span>{ago(j().created)}</span>
                                  </Meta>
                                </span>
                              </span>
                              <span class="m-side">
                                <StatusBadge plain state={st()} />
                              </span>
                            </a>
                          </li>
                        );
                      }}
                    </For>
                  </ul>
                ) : (
                  <ul class="rows" aria-hidden="true">
                    <For each={[0, 1, 2]}>
                      {() => (
                        <li>
                          <span class="row-sk">
                            <span class="m-main">
                              <Skeleton w="55%" h="14px" />
                              <Skeleton w="45%" h="10px" />
                            </span>
                            <Skeleton w="64px" h="12px" />
                          </span>
                        </li>
                      )}
                    </For>
                  </ul>
                )
              ) : (
                <div class="scroll" class:state={!hasRows.get()}>
                <table class="table jobs" aria-busy={!jobs.data && !jobs.error} aria-label="Labeling jobs" use:rowNav>
                  {hasRows.get() ? (
                    <thead>
                      <tr>
                        {/* One geometry in every state: the widths sit on the header row loading and loaded share; from 1280 the
                            name is bounded and the slack goes to the empty last column. */}
                        <th scope="col" class="name-col">
                          Name
                        </th>
                        <th scope="col" class="r" style:width="88px">
                          Images
                        </th>
                        <th scope="col" style:width="176px">
                          Labeled
                        </th>
                        <th scope="col" style:width="136px">
                          Status
                        </th>
                        <th scope="col" class="hide-md" style:width="184px">
                          Agent
                        </th>
                        <th scope="col" class="r hide-md" style:width="104px">
                          Created
                        </th>
                        <th class="hide-lg rest" aria-hidden="true"></th>
                      </tr>
                    </thead>
                  ) : null}
                  <tbody>
                    {jobs.error && !jobs.data ? (
                      <TableState cols={COLS}>
                        <ErrorState error={jobs.error} title="Could not load labeling jobs" retry={() => jobs.refresh()} />
                      </TableState>
                    ) : jobs.data?.length === 0 ? (
                      <TableState cols={COLS}>
                        <EmptyState
                          icon={Tags}
                          title="No labeling jobs yet"
                          description="Add images, name the classes, and Claude Code, Codex or Antigravity draws the boxes for you to review."
                          actions={() => (
                            <Button icon={Plus} onclick={() => creating.set(true)}>
                              New labeling job
                            </Button>
                          )}
                        />
                      </TableState>
                    ) : jobs.data && shown.get().length === 0 ? (
                      <TableState cols={COLS}>
                        <EmptyState
                          icon={Search}
                          title="No labeling jobs match"
                          description="Try another status or search term."
                          actions={() => (
                            <Button size="sm" onclick={clearFilters}>
                              Clear filters
                            </Button>
                          )}
                        />
                      </TableState>
                    ) : !jobs.data ? (
                      <For each={[0, 1, 2]}>
                        {() => (
                          <tr aria-hidden="true">
                            <td>
                              <Skeleton w="120px" h="12px" />
                            </td>
                            <td class="r">
                              <Skeleton w="28px" h="12px" />
                            </td>
                            <td>
                              <span class="lab">
                                <Skeleton w="28px" h="12px" />
                                <Skeleton h="4px" r="999px" />
                              </span>
                            </td>
                            <td>
                              <Skeleton w="64px" h="12px" />
                            </td>
                            <td class="hide-md">
                              <Skeleton w="120px" h="12px" />
                            </td>
                            <td class="r hide-md">
                              <Skeleton w="44px" h="12px" />
                            </td>
                            <td class="hide-lg rest"></td>
                          </tr>
                        )}
                      </For>
                    ) : (
                      <For each={shown.get()} key={(j) => j.id}>
                        {(j) => {
                          const st = () => jobStatus(j());
                          return (
                            <tr class="link" data-row data-id={j().id} aria-label={j().name} onclick={() => open(j())}>
                              <td class="name-cell">
                                <a class="name truncate" href={href(`/datasets/label/${j().id}`)} onclick={(e) => e.stopPropagation()} title={j().name}>
                                  {j().name}
                                </a>
                              </td>
                              <td class="r num muted">{num(j().counts.images)}</td>
                              <td>
                                <span class="lab">
                                  <span class="num">{num(j().counts.labeled)}</span>
                                  <Progress
                                    value={j().counts.images ? j().counts.labeled / j().counts.images : 0}
                                    tone={filterOf(j()) === 'active' ? 'accent' : 'neutral'}
                                    label={`${num(j().counts.labeled)} of ${counted(j().counts.images, 'image')} labeled`}
                                  />
                                </span>
                              </td>
                              <td>
                                <StatusBadge plain state={st()} />
                              </td>
                              <td class="hide-md agent-cell">
                                {j().run ? (
                                  <Meta line title={runAgent(j().run!, agents.data)[1]}>
                                    <span>{runAgent(j().run!, agents.data)[0]}</span>
                                    <span>{runAgent(j().run!, agents.data)[1]}</span>
                                  </Meta>
                                ) : (
                                  <NoValue label="no run yet" />
                                )}
                              </td>
                              <td class="r faint hide-md">{ago(j().created)}</td>
                              <td class="hide-lg rest" aria-hidden="true"></td>
                            </tr>
                          );
                        }}
                      </For>
                    )}
                  </tbody>
                </table>
              </div>
              )}
            </Panel>
            {jobs.data && shown.get().length ? (
              <p class="list-foot num">
                <Meta>
                  <span>
                    {shown.get().length < jobs.data!.length ? `${shown.get().length} of ${jobs.data!.length}` : shown.get().length}{' '}
                    {jobs.data!.length === 1 ? 'job' : 'jobs'}
                  </span>
                  <span>{counted(totals.get().images, 'image')}</span>
                  <span>{num(totals.get().labeled)} labeled</span>
                </Meta>
              </p>
            ) : null}
            <p class="sr-only" role="status" aria-live="polite" data-filter-status>
              {filterStatus.get()}
            </p>
          </>
        )}
      </Page>

      <JobDialog bind:open={creating} />
    </>
  );
}

css`
  .scroll {
    overflow-x: auto;
  }
  /* The name, and the agent's line with a long model id (a Bedrock profile's ARN), cut themselves inside their
     columns (whole in their tooltips) instead of widening the table. */
  .name-cell,
  .agent-cell {
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
`;
