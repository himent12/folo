import Activity from '$icons/activity';
import Box from '$icons/box';
import GitCompare from '$icons/git-compare-arrows';
import Plus from '$icons/plus';
import ScanSearch from '$icons/scan-search';
import Search from '$icons/search';
import Trash from '$icons/trash-2';
import X from '$icons/x';
import { computed, css, effect, For, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { modelsResource } from '$lib/api/models';
import { Resource } from '$lib/api/resource';
import type { Model, QueueRequest, Run } from '$lib/api/types';
import { runDatasetLabel } from '$lib/runDataset';
import RunSpark from '$lib/charts/RunSpark';
import { ago, duration, metric } from '$lib/format';
import { failureCause } from '$lib/oom';
import { tooltip } from '$lib/ui/tooltip';
import { href, router } from '$lib/router';
import { ownCreateAction, setCrumbs } from '$lib/shell/crumbs';
import Page from '$lib/shell/Page';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Input from '$lib/ui/Input';
import Panel from '$lib/ui/Panel';
import Progress from '$lib/ui/Progress';
import Segmented from '$lib/ui/Segmented';
import { runProgress, trainingSeconds } from '$lib/api/progress';
import { rowNav } from '$lib/ui/rowNav';
import Skeleton from '$lib/ui/Skeleton';
import StatusBadge from '$lib/ui/StatusBadge';
import TableState from '$lib/ui/TableState';
import Gate from '$lib/ui/Gate';
import { compareHint, compareReason } from './runs/compareHint';
import Ident from '$lib/ui/Ident';
import Lineage from '$lib/ui/Lineage';
import Meta from '$lib/ui/Meta';
import { isOffline, OFFLINE_REASON } from '$lib/ui/offline';
import { FROM_SCRATCH, origin } from '$lib/lineage';
import Menu, { type MenuItem } from '$lib/ui/Menu';
import SortHeader from '$lib/ui/SortHeader';
import { hasFeature, queuePlace, withQueueMove } from '$lib/api/queue';
import { moveQueued, queueItems, REMOVE_LABEL, removeDescription, removeQueued, removeTitle } from './run/queueActions';
import Dialog from '$lib/ui/Dialog';
import NoValue from '$lib/ui/NoValue';
import { DialogStatus } from '$lib/ui/feedback';
import { toast } from '$lib/ui/toast';
import DeleteRun from './run/DeleteRun';
import RunRows from './runs/RunRows';
import { groups, readState, sortRuns, writeState, type Filter, type SortKey } from './runs/listState';
import { exactTime, tickingNow } from './runs/when';

/** Columns of the desktop table (for state rows that span it). */
const COLS = 12;
const ACTIVE = ['queued', 'running', 'stopping'];

export default function Runs() {
  setCrumbs({ label: 'Runs' });
  // The header's create action is the one in the top-right corner (the top bar's New run steps aside).
  ownCreateAction();

  const runs = new Resource((s) => hub.api.runs(s), { poll: 3000 });
  const datasets = new Resource((s) => hub.api.datasets(s));
  const dsName = computed(() => new Map((datasets.data ?? []).map((d) => [d.id, d])));
  /** A run's dataset by name (lib/runDataset), the one it recorded while the datasets load; marked once it was removed. */
  const datasetName = (r: Run) => runDatasetLabel(r, datasets.data ? dsName.get() : undefined, !!datasets.error) ?? r.dataset?.name ?? '';
  // Only to name the model a run was fine-tuned from; the list renders without it.
  const models = modelsResource();
  const modelById = computed(() => new Map<string, Model>((models.data ?? []).map((m) => [m.id, m])));

  // Queue control (servers that list `queue_control`): a waiting run's row says its place and has a ⋯ menu with
  // Start next, Move up and Move down. The new order shows at once; a refusal puts the old one back.
  const system = new Resource((s) => hub.api.system(s));
  const queueCtl = computed(() => hasFeature(system.data, 'queue_control'));
  const waiting = computed(() => (runs.data ?? []).filter((r) => r.status === 'queued' && r.queue_position != null).length);
  const training = computed(() => runs.data?.find((r) => r.status === 'running' || r.status === 'stopping') ?? null);
  // A cloud run waits in no queue here (its hand-over is said by its status word), so it never has a place.
  const placeOf = (r: Run) => (queueCtl.get() && r.status === 'queued' && !r.cloud ? (r.queue_position ?? null) : null);
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
      training.get(),
    ).then((done) => {
      if (done) void runs.refresh();
    });
  }
  /**
   * A row's ⋯: a waiting run's queue actions; an ended run's Open model and Try in Playground (with a model) and Delete
   * run, the destructive one last, as on its own page. A training run has none (its page has Stop).
   */
  const rowItems = (r: Run): MenuItem[] => {
    const at = placeOf(r);
    if (at !== null) return queueItems(at, waiting.get(), training.get(), (req) => move(r, req), () => removing.set(r));
    if (ACTIVE.includes(r.status)) return [];
    const out: MenuItem[] = [];
    if (r.model_id) {
      out.push({ label: 'Open model', icon: Box, href: href(`/models/${r.model_id}`) });
      out.push({ label: 'Try in Playground', icon: ScanSearch, href: href(`/playground?model=${r.model_id}`) });
    }
    out.push({ label: 'Delete run', icon: Trash, danger: true, onclick: () => deleting.set(r) });
    return out;
  };
  /** The run Delete run asks about (the same confirm as on its page). */
  const deleting = signal<Run | null>(null);

  /** The waiting run Remove from queue asks about (the same confirm as on its own page). */
  const removing = signal<Run | null>(null);
  const removeBusy = signal(false);
  /** Its failure shows in the confirm, which stays open; the success toasts once it has closed. */
  const removeStatus = new DialogStatus();
  async function remove() {
    const r = removing.get();
    if (!r) return;
    removeBusy.set(true);
    try {
      await removeQueued(r);
      removing.set(null);
      // page-level: the confirm closed on the line above.
      toast.success(`Removed ${r.name} from the queue`);
      void runs.refresh();
    } catch (e) {
      removeStatus.error(e, `Could not remove ${r.name} from the queue`);
    } finally {
      removeBusy.set(false);
    }
  }

  // Filter, search, sort and the runs ticked for Compare live in the URL's query (./runs/listState), replaced as they
  // change, so Back from a run shows the list as it was left. A deep link's `?status=` is a filter only when it names
  // one (`?status=stopped` is a status, not a filter).
  const start = untrack(() => readState(router.route.query));
  const filter = signal<Filter>(start.filter);
  const query = signal(start.q);
  const sortKey = signal<SortKey>(start.sort);
  const sortDir = signal(start.dir);
  const selected = signal<string[]>(start.sel);
  // Queries compare in the one form writeState gives (a URL's own spelling may differ: `%2C` for the ids' commas).
  const norm = (q: URLSearchParams) => writeState(readState(q));
  let written = untrack(() => norm(router.route.query));
  effect(() => {
    const next = writeState({ filter: filter.get(), q: query.get(), sort: sortKey.get(), dir: sortDir.get(), sel: selected.get() });
    if (next === written || untrack(() => router.route.name) !== 'runs') return;
    written = next;
    router.go(`/runs${next ? `?${next}` : ''}`, { replace: true });
  });
  // A link to the list (the sidebar's Runs while here) brings its own state.
  effect(() => {
    const r = router.route;
    if (r.name !== 'runs' || norm(r.query) === written) return;
    const st = readState(r.query);
    written = norm(r.query);
    filter.set(st.filter);
    query.set(st.q);
    sortKey.set(st.sort);
    sortDir.set(st.dir);
    selected.set(st.sel);
  });
  // A ticked run that is gone (deleted elsewhere) leaves the selection.
  effect(() => {
    const ids = runs.data?.map((r) => r.id);
    if (!ids) return;
    const sel = untrack(() => selected.get());
    if (sel.some((id) => !ids.includes(id))) selected.set(sel.filter((id) => ids.includes(id)));
  });
  /** Click a sorted column again to turn it around; another column starts high first. */
  function sortBy(k: SortKey) {
    if (sortKey.peek() === k) sortDir.set(sortDir.peek() === 'asc' ? 'desc' : 'asc');
    else {
      sortKey.set(k);
      sortDir.set('desc');
    }
  }
  const dirOf = (k: SortKey) => (sortKey.get() === k ? sortDir.get() : null);
  /** The order the rows are in, as the page's subtitle says it. */
  const order = computed(() => {
    const up = sortDir.get() === 'asc';
    switch (sortKey.get()) {
      case 'created':
        return up ? 'oldest first' : 'newest first';
      case 'map':
        return up ? 'lowest best mAP50-95 first' : 'highest best mAP50-95 first';
      case 'progress':
        return up ? 'least progress first' : 'most progress first';
      case 'duration':
        return up ? 'shortest first' : 'longest first';
    }
  });
  // Relative times ("9m ago") and live durations move on without a reload.
  const now = tickingNow();
  /** Phones: checkboxes show in select mode (or once a run is selected), so rows keep their width otherwise. */
  const picking = signal(false);
  const choosing = computed(() => picking.get() || selected.get().length > 0);

  const count = (f: Filter) => (runs.data ?? []).filter((r) => groups[f].includes(r.status)).length;
  const shown = computed(() =>
    sortRuns(
      (runs.data ?? []).filter((r) => {
        if (!groups[filter.get()].includes(r.status)) return false;
        const q = query.get().trim().toLowerCase();
        if (!q) return true;
        return `${r.name} ${r.config.arch} ${datasetName(r)} ${r.id}`.toLowerCase().includes(q);
      }),
      sortKey.get(),
      sortDir.get(),
      now.get(),
    ),
  );

  /** Compare needs two runs: until then New run stays the page's one primary (DESIGN_DIRECTION 6). */
  const comparing = computed(() => selected.get().length >= 2);
  /** Phones: every row keeps the ⋯ slot while any row shown has a menu, so all rows share one right edge. */
  const menuSlot = computed(() => shown.get().some((r) => rowItems(r).length > 0));
  // What the rows shown hold, said once at the list's end (base.css .list-foot): how many, and their training time.
  const trained = computed(() => shown.get().reduce((a, r) => a + (trainingSeconds(r, now.get()) ?? 0), 0));

  function toggle(id: string) {
    const cur = selected.get();
    selected.set(cur.includes(id) ? cur.filter((x) => x !== id) : cur.length >= 4 ? cur : [...cur, id]);
  }

  // Progress bars stay neutral; only a live run's bar carries the accent ("now"). Status is the dot beside it.
  const tone = (r: Run) => (r.status === 'running' ? 'accent' : 'neutral');
  // Nothing answers: the actions that need the server are disabled and say why (as on Datasets).
  const offline = computed(() => !runs.data && isOffline(runs.error));
  /** Runs to filter, or on their way: the toolbar is live (else it stays in place, disabled). */
  const filterable = computed(() => (runs.data ? runs.data.length > 0 : !runs.error));
  /** Rows to head (or their skeleton): the column heads show; with none the frame holds the state alone. */
  const rowsOrLoading = computed(() => (runs.data ? shown.get().length > 0 : !runs.error));

  const compareButton = (why: () => string | undefined) => (
    <Button
      variant={comparing.get() ? 'primary' : 'secondary'}
      icon={GitCompare}
      disabled={!!why()}
      aria-describedby={why()}
      href={href(`/runs/compare?ids=${selected.get().join(',')}`)}
    >
      Compare{selected.get().length ? <span class="num count">{selected.get().length}</span> : null}
    </Button>
  );

  return (
    <>
      <Page
        title="Runs"
        sub={() => <span>Training runs, {order.get()}.</span>}
        subPhone={() => <span>Training runs, {order.get()}.</span>}
        actions={() => (
          <>
            {runs.data?.length !== 0 ? (
              <>
                <span class="compare">
                  {runs.data && !offline.get() ? (
                    <span class="cmp-hint" role="status" data-gate-why>
                      {compareHint(selected.get().length)}
                    </span>
                  ) : null}
                  <Gate reason={offline.get() ? OFFLINE_REASON : compareReason(selected.get().length)}>{compareButton}</Gate>
                </span>
                <Gate reason={offline.get() ? OFFLINE_REASON : null}>
                  {(why) => (
                    // Once Compare can run (2+ selected) it is the page's one primary and New run steps back; one
                    // selected run leaves Compare disabled, so New run stays primary.
                    <Button variant={comparing.get() ? 'secondary' : 'primary'} icon={Plus} disabled={!!why()} aria-describedby={why()} href={href('/runs/new')}>
                      New run
                    </Button>
                  )}
                </Gate>
              </>
            ) : null}
          </>
        )}
      >
        {/* The toolbar is always here, so the table never moves between states (DESIGN_DIRECTION §2.8): over an empty
            or failed list it is disabled and shows no counts. */}
        <div class="list-toolbar">
          <div class="lt-filter">
            <Segmented
              label="Filter runs by status"
              bind:value={filter}
              disabled={!filterable.get()}
              options={[
                { value: 'all', label: 'All', meta: runs.data?.length ? String(count('all')) : undefined },
                { value: 'active', label: 'Active', meta: runs.data?.length ? String(count('active')) : undefined },
                { value: 'completed', label: 'Finished', meta: runs.data?.length ? String(count('completed')) : undefined },
                { value: 'failed', label: 'Failed', meta: runs.data?.length ? String(count('failed')) : undefined },
              ]}
            />
          </div>{' '}
          {/* Tablets: Compare's prerequisite at the end of the filter's row, under the header's one row of actions. */}
          {runs.data?.length && !offline.get() ? (
            <span class="tool-hint" role="status" data-gate-why>
              {compareHint(selected.get().length)}
            </span>
          ) : null}{' '}
          <div class="lt-search">
            <Input icon={Search} bind:value={query} placeholder="Filter by name, model or dataset" aria-label="Filter runs" disabled={!filterable.get()} />
          </div>
        </div>{' '}
        <Panel pad="none">
          {/* One table frame for every state: with no rows (empty, no match, an error) it drops its column heads and
              says so inside it, so no header row of eleven columns sits over a one-line message. */}
          <div class="scroll" class:state={!rowsOrLoading.get()}>
            {/* Row checkboxes show on hover and focus, on selected rows, and on every row once one is selected (picking)
                or two runs are listed (pickable: the header asks to select 2–4, as the benchmark's leaderboard does). */}
            <table
              class="table runs"
              class:picking={selected.get().length > 0}
              class:pickable={shown.get().length >= 2}
              aria-label="Runs"
              aria-busy={!runs.data}
              use:rowNav={{ toggle: (row: HTMLElement) => toggle(row.dataset.id ?? '') }}
            >
              {rowsOrLoading.get() ? (
                <thead>
                  <tr>
                    {/* One geometry in every state (DESIGN_DIRECTION 7): each column's width is fixed here, on the
                        header row that loading and loaded share; the run's name takes what is left. */}
                    <th class="check">
                      <span class="sr-only">Select for comparison</span>
                    </th>
                    <th>Run</th>
                    <th style:width="128px">Status</th>
                    <th class="hide-xl" style:width="112px">
                      Dataset
                    </th>
                    <th class="hide-xl" style:width="80px">
                      Model
                    </th>
                    <th class="hide-xl" style:width="128px">
                      Started from
                    </th>
                    <SortHeader label="Progress" width="136px" dir={dirOf('progress')} onsort={() => sortBy('progress')} disabled={!runs.data} />
                    <SortHeader label="Best mAP50-95" align="right" width="136px" dir={dirOf('map')} onsort={() => sortBy('map')} disabled={!runs.data} />
                    <th class="hide-lg" style:width="120px">
                      mAP50-95 trend
                    </th>
                    <SortHeader label="Duration" align="right" width="104px" class="hide-lg" dir={dirOf('duration')} onsort={() => sortBy('duration')} disabled={!runs.data} />
                    <SortHeader label="Created" align="right" width="104px" class="hide-md" dir={dirOf('created')} onsort={() => sortBy('created')} disabled={!runs.data} />
                    <th class="act" style:width="40px">
                      <span class="sr-only">Actions</span>
                    </th>
                  </tr>
                </thead>
              ) : null}
              <tbody>
                {runs.error && !runs.data ? (
                  <TableState cols={COLS}>
                    <ErrorState error={runs.error} retry={() => runs.refresh()} title="Could not load runs" />
                  </TableState>
                ) : !runs.data ? (
                  <For each={[0, 1, 2, 3, 4]} key={(i) => i}>
                    {() => (
                      <tr aria-hidden="true">
                        <td class="check"></td>
                        <td class="lead">
                          <Skeleton w="112px" h="12px" />
                        </td>
                        <td>
                          <Skeleton w="72px" h="12px" />
                        </td>
                        <td class="hide-xl">
                          <Skeleton w="64px" h="12px" />
                        </td>
                        <td class="hide-xl">
                          <Skeleton w="48px" h="12px" />
                        </td>
                        <td class="hide-xl">
                          <Skeleton w="88px" h="12px" />
                        </td>
                        <td>
                          <div class="prog">
                            <Skeleton h="4px" />
                            <span>
                              <Skeleton w="40px" h="10px" />
                            </span>
                          </div>
                        </td>
                        <td class="r">
                          <Skeleton w="72px" h="12px" />
                        </td>
                        <td class="hide-lg">
                          <Skeleton w="96px" h="12px" />
                        </td>
                        <td class="r hide-lg">
                          <Skeleton w="48px" h="12px" />
                        </td>
                        <td class="r hide-md">
                          <Skeleton w="56px" h="12px" />
                        </td>
                        <td class="act"></td>
                      </tr>
                    )}
                  </For>
                ) : runs.data.length === 0 ? (
                  <TableState cols={COLS}>
                    {/* The header steps its actions aside over an empty list, so this is the page's one primary. */}
                    <EmptyState
                      icon={Activity}
                      title="No runs yet"
                      description="Start one to watch its loss, mAP and samples stream in live."
                      actions={() => (
                        <Button variant="primary" icon={Plus} href={href('/runs/new')}>
                          New run
                        </Button>
                      )}
                    />
                  </TableState>
                ) : shown.get().length === 0 ? (
                  <TableState cols={COLS}>
                    <EmptyState
                      icon={Search}
                      title="No runs match"
                      description="Try another status filter or search term."
                      actions={() => (
                        <Button size="sm" onclick={() => (filter.set('all'), query.set(''))}>
                          Clear filters
                        </Button>
                      )}
                    />
                  </TableState>
                ) : (
                  <For each={shown.get()} key={(r) => r.id}>
                    {(r) => {
                      const on = () => selected.get().includes(r().id);
                      const from = computed(() => origin(r().config.weights, modelById.get()));
                      const items = computed(() => rowItems(r()));
                      return (
                        <tr class="link" class:on={on()} data-row data-id={r().id} aria-label={r().name} onclick={() => router.go(`/runs/${r().id}`)}>
                          {/* The whole cell is the checkbox's target (44 px on touch); a click in it never opens the run. */}
                          <td class="check" onclick={(e) => e.stopPropagation()}>
                            <label class="hit">
                              <input
                                type="checkbox"
                                checked={on()}
                                disabled={!on() && selected.get().length >= 4}
                                aria-label={`Select ${r().name} for comparison`}
                                onchange={() => toggle(r().id)}
                              />
                            </label>
                          </td>
                          <td class="lead">
                            <a class="name" href={href(`/runs/${r().id}`)} title={r().name} onclick={(e) => e.stopPropagation()}>
                              {r().name}
                            </a>{' '}
                            {/* Whole items wrap: the name's column is what the fixed ones leave, and a fact is never cut. */}
                            <span class="sub ctx">
                              <Meta>
                                <span>{datasetName(r())}</span>
                                <Ident value={r().config.arch} />
                                <span>
                                  from <Lineage origin={from.get()} link={false} scratch="scratch" />
                                </span>
                              </Meta>
                            </span>
                          </td>
                          <td use:tooltip={failureCause(r())}>
                            <StatusBadge run={r()} plain />
                          </td>
                          <td class="hide-xl muted">{datasetName(r())}</td>
                          <td class="hide-xl">
                            <Ident value={r().config.arch} class="muted" />
                          </td>
                          <td class="hide-xl from">
                            {(() => {
                              const f = from.get();
                              return f.kind === 'model' && modelById.get().has(f.id) ? (
                                <Ident value={f.label} href={href(`/models/${f.id}`)} onclick={(e) => e.stopPropagation()} />
                              ) : f.kind === 'file' ? (
                                <Ident value={f.label} title={f.path} />
                              ) : f.kind === 'scratch' ? (
                                <span class="faint">{FROM_SCRATCH}</span>
                              ) : (
                                <Ident value={f.label} class="faint" />
                              );
                            })()}
                          </td>
                          <td>
                            {placeOf(r()) !== null ? (
                              // A waiting run has no progress yet: its place in the queue instead.
                              <span class="prog place">{queuePlace(placeOf(r()) ?? 0)}</span>
                            ) : (
                              <div class="prog">
                                <Progress value={runProgress(r())} size="xs" tone={tone(r())} label={`${r().name} progress`} />{' '}
                                <span class="num faint">
                                  {r().epoch}/{r().epochs}
                                </span>
                              </div>
                            )}
                          </td>
                          <td class="r num best">
                            {r().best ? (
                              <>
                                {metric(r().best!.map50_95)}{' '}
                                <span class="faint" title="Best epoch">
                                  ep {r().best!.epoch}
                                </span>
                              </>
                            ) : (
                              <NoValue />
                            )}
                          </td>
                          <td class="hide-lg">
                            <RunSpark run={r()} width={96} />
                          </td>
                          <td class="r num faint hide-lg">{trainingSeconds(r(), now.get()) === null ? <NoValue /> : duration(trainingSeconds(r(), now.get()))}</td>
                          <td class="r faint hide-md">
                            <time datetime={r().created} title={exactTime(r().created)}>
                              {ago(r().created, now.get())}
                            </time>
                          </td>
                          {/* A click in the cell never opens the run. */}
                          <td class="act" onclick={(e) => e.stopPropagation()}>
                            {items.get().length ? <Menu label={`${placeOf(r()) !== null ? 'Queue actions' : 'Actions'} for ${r().name}`} items={items.get()} variant="ghost" fixed /> : null}
                          </td>
                        </tr>
                      );
                    }}
                  </For>
                )}
              </tbody>
            </table>
          </div>{' '}
          <RunRows
            runs={shown.get()}
            loading={!runs.data && !runs.error}
            selected={selected.get()}
            toggle={toggle}
            choosing={choosing.get()}
            onpick={() => picking.set(true)}
            oncancel={() => (picking.set(false), selected.set([]))}
            compare={() => <Gate reason={offline.get() ? OFFLINE_REASON : compareReason(selected.get().length)}>{compareButton}</Gate>}
            hint={!offline.get() ? compareHint(selected.get().length) : null}
            menuSlot={menuSlot.get()}
            items={rowItems}
            placeOf={placeOf}
            datasetName={datasetName}
            now={now.get()}
          />
        </Panel>{' '}
        {runs.data && shown.get().length ? (
          <p class="list-foot num">
            <Meta>
              <span>
                {shown.get().length < runs.data.length ? `${shown.get().length} of ${runs.data.length}` : shown.get().length} {runs.data.length === 1 ? 'run' : 'runs'}
              </span>{' '}
              <span>{duration(trained.get())} of training</span>
            </Meta>
          </p>
        ) : null}
      </Page>{' '}
      <Dialog
        open={!!removing.get()}
        onclose={() => removing.set(null)}
        status={removeStatus}
        title={removing.get() ? removeTitle(removing.get()!.name) : ''}
        description={removing.get() ? removeDescription(removing.get()!) : undefined}
        size="sm"
        footer={() => (
          <>
            <Button onclick={() => removing.set(null)}>Keep it queued</Button>{' '}
            <Button variant="destructive" icon={X} loading={removeBusy.get()} onclick={remove}>
              {REMOVE_LABEL}
            </Button>
          </>
        )}
      />{' '}
      <DeleteRun
        run={deleting.get()}
        onclose={() => deleting.set(null)}
        ondeleted={(r) => {
          selected.set(selected.peek().filter((id) => id !== r.id));
          void runs.refresh();
        }}
      />
    </>
  );
}

css`
  /* Compare's count: part of the label, one weight lighter. */
  .count {
    font-weight: var(--weight-regular);
  }
  .compare {
    display: contents;
  }
  /* The prerequisite, beside the action it gates: 12 px secondary, one line. */
  .cmp-hint,
  .tool-hint,
  .rows-hint {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .cmp-hint {
    align-self: center;
    white-space: nowrap;
  }
  .rows-hint,
  .tool-hint {
    display: none;
  }
  /* Tablets: a page header's actions never wrap (shell/Page), so the hint leaves Compare's side for the toolbar. */
  @media (min-width: 768px) and (max-width: 1023px) {
    .cmp-hint {
      display: none;
    }
    .tool-hint {
      display: block;
      margin-left: auto;
      white-space: nowrap;
    }
  }
  /* The table's frame is the container the wide columns ask for room (below). */
  .scroll {
    overflow-x: auto;
    container-type: inline-size;
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
    /* Neutral: a selected row is marked in ink: its leading bar, the fill and the inverted check. */
    accent-color: var(--text-primary);
    cursor: pointer;
  }
  .runs .check input {
    opacity: 0;
    transition: opacity var(--dur-fast) var(--ease-standard);
  }
  /* Shown on the row under the pointer or the keyboard (so X is discoverable), on selected rows, and on every row once
     any is selected or two runs are listed to pick from. */
  .runs tbody tr:is(:hover, :focus-within, .on) .check input,
  .runs:is(.picking, .pickable) .check input,
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
  /* Fixed layout: the columns keep their widths and the run's name takes what is left, cut with an ellipsis (its full
     name in the tooltip), so one long name never pushes the other columns out of the frame. */
  .runs {
    table-layout: fixed;
  }
  .name {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
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
  .ctx {
    display: none;
  }
  /* In a frame under 1320 px the Dataset, Model and Started from columns hide (the name's second line has them): with
     Duration in, the page's widest frame (--size-content-max) would leave the name under 80 px. Below 1280 the trend
     and Duration hide too (.hide-lg, base.css), so the name keeps 160 px or more at every width. */
  @container (max-width: 1319px) {
    .runs :is(th, td).hide-xl {
      display: none;
    }
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
    /* Phones show the stacked rows below; the table frame stays only to hold an empty or error state (no column heads at any width). */
    .scroll:not(.state) {
      display: none;
    }
    .compare {
      display: none;
    }
  }
`;
