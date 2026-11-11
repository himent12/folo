import Activity from '$icons/activity';
import GitCompare from '$icons/git-compare-arrows';
import Plus from '$icons/plus';
import Search from '$icons/search';
import X from '$icons/x';
import { computed, css, For, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { Model, QueueRequest, Run, RunStatus } from '$lib/api/types';
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
import { runProgress } from '$lib/api/progress';
import { rowNav } from '$lib/ui/rowNav';
import Skeleton from '$lib/ui/Skeleton';
import StatusBadge, { statusWord } from '$lib/ui/StatusBadge';
import TableState from '$lib/ui/TableState';
import Gate from '$lib/ui/Gate';
import { compareHint, compareReason } from './runs/compareHint';
import Ident from '$lib/ui/Ident';
import Lineage from '$lib/ui/Lineage';
import Meta from '$lib/ui/Meta';
import { isOffline, OFFLINE_REASON } from '$lib/ui/offline';
import { FROM_SCRATCH, origin } from '$lib/lineage';
import Menu from '$lib/ui/Menu';
import { hasFeature, queuePlace, withQueueMove } from '$lib/api/queue';
import { moveQueued, queueItems, REMOVE_LABEL, removeDescription, removeQueued, removeTitle } from './run/queueActions';
import Dialog from '$lib/ui/Dialog';
import NoValue from '$lib/ui/NoValue';
import { DialogStatus } from '$lib/ui/feedback';
import { toast } from '$lib/ui/toast';

type Filter = 'all' | 'active' | 'completed' | 'failed';
const groups: Record<Filter, RunStatus[]> = {
  all: ['queued', 'running', 'stopping', 'stopped', 'completed', 'failed'],
  active: ['queued', 'running', 'stopping'],
  completed: ['completed', 'stopped'],
  failed: ['failed'],
};
/** Columns of the desktop table (for state rows that span it). */
const COLS = 11;

export default function Runs() {
  setCrumbs({ label: 'Runs' });
  // The header's create action is the one in the top-right corner (the top bar's New run steps aside).
  ownCreateAction();

  const runs = new Resource((s) => hub.api.runs(s), { poll: 3000 });
  const datasets = new Resource((s) => hub.api.datasets(s));
  const dsName = computed(() => new Map((datasets.data ?? []).map((d) => [d.id, d.name])));
  // Only to name the model a run was fine-tuned from; the list renders without it.
  const models = new Resource((s) => hub.api.models(s));
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
  const rowItems = (r: Run) => {
    const at = placeOf(r);
    return at === null ? [] : queueItems(at, waiting.get(), training.get(), (req) => move(r, req), () => removing.set(r));
  };

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

  // A deep link's `?status=` is a filter only when it names one (`?status=stopped` is a status, not a filter).
  const asked = untrack(() => router.route.query.get('status'));
  const filter = signal<Filter>(asked && Object.hasOwn(groups, asked) ? (asked as Filter) : 'all');
  const query = signal('');
  const selected = signal<string[]>([]);
  /** Phones: checkboxes show in select mode (or once a run is selected), so rows keep their width otherwise. */
  const picking = signal(false);
  const choosing = computed(() => picking.get() || selected.get().length > 0);

  const count = (f: Filter) => (runs.data ?? []).filter((r) => groups[f].includes(r.status)).length;
  const shown = computed(() =>
    (runs.data ?? []).filter((r) => {
      if (!groups[filter.get()].includes(r.status)) return false;
      const q = query.get().trim().toLowerCase();
      if (!q) return true;
      return `${r.name} ${r.config.arch} ${dsName.get().get(r.config.dataset_id) ?? ''} ${r.id}`.toLowerCase().includes(q);
    }),
  );

  /** Compare needs two runs: until then New run stays the page's one primary (DESIGN_DIRECTION 6). */
  const comparing = computed(() => selected.get().length >= 2);
  /** Phones: every row keeps the ⋯ slot while any row shown has a menu, so all rows share one right edge. */
  const menuSlot = computed(() => shown.get().some((r) => rowItems(r).length > 0));
  // What the rows shown hold, said once at the list's end (base.css .list-foot): how many, and their training time.
  const trained = computed(() =>
    shown
      .get()
      .reduce((a, r) => (r.started ? a + Math.max(0, (r.finished ? Date.parse(r.finished) : Date.now()) - Date.parse(r.started)) / 1000 : a), 0),
  );

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
        sub={() => <span>Training runs, newest first.</span>}
        subPhone={() => <span>Training runs, newest first.</span>}
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
            {/* Row checkboxes show on hover and focus, on selected rows, and on every row once one is selected (picking). */}
            <table
              class="table runs"
              class:picking={selected.get().length > 0}
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
                    <th class="hide-lg" style:width="112px">
                      Dataset
                    </th>
                    <th class="hide-lg" style:width="80px">
                      Model
                    </th>
                    <th class="hide-lg" style:width="128px">
                      Started from
                    </th>
                    <th style:width="136px">Progress</th>
                    <th class="r" style:width="128px">
                      Best mAP50-95
                    </th>
                    <th class="hide-lg" style:width="120px">
                      mAP50-95 trend
                    </th>
                    <th class="r hide-md" style:width="96px">
                      Created
                    </th>
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
                        <td class="hide-lg">
                          <Skeleton w="64px" h="12px" />
                        </td>
                        <td class="hide-lg">
                          <Skeleton w="48px" h="12px" />
                        </td>
                        <td class="hide-lg">
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
                            <a class="name" href={href(`/runs/${r().id}`)} onclick={(e) => e.stopPropagation()}>
                              {r().name}
                            </a>{' '}
                            <span class="sub ctx">
                              <Meta line>
                                <span>{dsName.get().get(r().config.dataset_id) ?? r().config.dataset_id}</span>
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
                          <td class="hide-lg muted">{dsName.get().get(r().config.dataset_id) ?? r().config.dataset_id}</td>
                          <td class="hide-lg">
                            <Ident value={r().config.arch} class="muted" />
                          </td>
                          <td class="hide-lg from">
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
                          <td class="r faint hide-md">{ago(r().created)}</td>
                          {/* A click in the cell never opens the run. */}
                          <td class="act" onclick={(e) => e.stopPropagation()}>
                            {items.get().length ? <Menu label={`Queue actions for ${r().name}`} items={items.get()} variant="ghost" fixed /> : null}
                          </td>
                        </tr>
                      );
                    }}
                  </For>
                )}
              </tbody>
            </table>
          </div>{' '}
          {shown.get().length ? (
            <>
              {/* Phones: stacked rows with the essentials (selection, name, status, score, progress). Select mode:
                  Compare (with its count) the primary once two runs are selected, Cancel a ghost one, both 44 px
                  targets on touch. */}
              <div class="rows-head" class:slot={menuSlot.get() && !choosing.get()}>
                {choosing.get() ? (
                  <>
                    <Button variant="ghost" onclick={() => (picking.set(false), selected.set([]))}>
                      Cancel
                    </Button>{' '}
                    <Gate reason={offline.get() ? OFFLINE_REASON : compareReason(selected.get().length)}>{compareButton}</Gate>
                  </>
                ) : (
                  <Button variant="ghost" icon={GitCompare} onclick={() => picking.set(true)}>
                    Select to compare
                  </Button>
                )}{' '}
                <span class="col" aria-hidden="true">
                  Best mAP50-95
                </span>
              </div>{' '}
              {choosing.get() && !offline.get() ? (
                <p class="rows-hint" role="status" data-gate-why>
                  {compareHint(selected.get().length)}
                </p>
              ) : null}{' '}
              <ul class="rows" class:choosing={choosing.get()}>
                <For each={shown.get()} key={(r) => r.id}>
                  {(r) => {
                    const on = () => selected.get().includes(r().id);
                    const cause = () => failureCause(r());
                    const items = computed(() => rowItems(r()));
                    return (
                      <li class:on={on()}>
                        <label class="r-check">
                          <input
                            type="checkbox"
                            checked={on()}
                            disabled={!on() && selected.get().length >= 4}
                            aria-label={`Select ${r().name} for comparison`}
                            onchange={() => toggle(r().id)}
                          />
                        </label>{' '}
                        <a class="r-main" href={href(`/runs/${r().id}`)}>
                          <span class="r-text">
                            <span class="r-name">
                              <StatusBadge run={r()} plain dotOnly />
                              <span class="name truncate" title={r().name}>
                                {r().name}
                              </span>
                            </span>{' '}
                            {/* A failure's cause is a line of its own (it is the news), with the epoch and the whole
                                dataset under it; otherwise the status leads that line. Lines wrap whole items, so no
                                fact is cut (DESIGN_DIRECTION §2.3). With the checkboxes in, the dataset steps out. */}
                            {cause() ? (
                              <span class="r-cause" data-fact>
                                {cause()}
                              </span>
                            ) : null}{' '}
                            <span class="r-meta num">
                              <Meta>
                                {!cause() ? <span>{placeOf(r()) !== null ? queuePlace(placeOf(r()) ?? 0) : statusWord(r())}</span> : null}
                                <span>
                                  {r().epoch}/{r().epochs}
                                </span>
                                {!choosing.get() ? <span>{dsName.get().get(r().config.dataset_id) ?? r().config.dataset_id}</span> : null}
                              </Meta>
                            </span>
                          </span>{' '}
                          <span class="r-end">
                            <span class="r-val num">{r().best ? metric(r().best!.map50_95) : <NoValue />}</span>
                            <span class="r-age">{ago(r().created)}</span>
                          </span>
                        </a>{' '}
                        {!choosing.get() && menuSlot.get() ? (
                          <span class="r-menu">
                            {items.get().length ? <Menu label={`Queue actions for ${r().name}`} items={items.get()} variant="ghost" fixed /> : null}
                          </span>
                        ) : null}{' '}
                        {r().status === 'running' ? <span class="r-prog" aria-hidden="true" style:transform={`scaleX(${runProgress(r())})`}></span> : null}
                      </li>
                    );
                  }}
                </For>
              </ul>
            </>
          ) : !runs.data && !runs.error ? (
            // Phones, loading: the stacked rows' shape (name over one meta line, the score on the right).
            <ul class="rows sk" aria-hidden="true">
              <For each={[0, 1, 2, 3, 4]} key={(i) => i}>
                {() => (
                  <li>
                    <span class="r-main">
                      <span class="r-text">
                        <Skeleton w="55%" h="14px" />
                        <Skeleton w="75%" h="10px" />
                      </span>{' '}
                      <span class="r-end">
                        <Skeleton w="40px" h="14px" />
                      </span>
                    </span>
                  </li>
                )}
              </For>
            </ul>
          ) : null}
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
    /* Neutral: a selected row is marked in ink: its leading bar, the fill and the inverted check. */
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
    /* Phones show the stacked rows below; the table frame stays only to hold an empty or error state (no column heads at any width). */
    .scroll:not(.state) {
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
    /* Phones: under Cancel and Compare, on the rows' edge, before the first row (one rule under both). */
    .rows-head:has(+ .rows-hint) {
      border-bottom: 0;
    }
    .rows-hint {
      display: block;
      padding: 0 var(--space-4) var(--space-3);
      border-bottom: 1px solid var(--line-subtle);
    }
    .col {
      margin-left: auto;
    }
    .rows li:last-child {
      border-bottom: 0;
    }
    .rows li.on {
      background: var(--bg-active);
      box-shadow: var(--select-bar);
    }
    /* The ⋯ slot: on every row while any row has a menu (empty where it has none), so values share one right edge. */
    .r-menu {
      flex-shrink: 0;
      display: flex;
      justify-content: flex-end;
      width: 32px;
      margin-left: var(--space-2);
    }
    .rows-head.slot {
      padding-right: calc(var(--space-4) + 40px);
    }
    .r-check {
      display: grid;
      place-items: center;
      flex-shrink: 0;
      width: 44px;
      height: 56px;
    }
    /* The checkbox itself is the 44 px target (DESIGN_DIRECTION §2.10), not only its label: the input is a 44 px box
       whose drawn mark is the 20 px square at its centre, ink when checked as everywhere (§2.6). */
    .r-check input[type='checkbox'] {
      -webkit-appearance: none;
      appearance: none;
      position: relative;
      width: 44px;
      height: 44px;
      border-radius: var(--radius-sm);
    }
    .r-check input[type='checkbox']::before {
      content: '';
      position: absolute;
      inset: 12px;
      border-radius: 4px;
      background: var(--bg-raised);
      box-shadow: inset 0 0 0 1.5px var(--line-control);
    }
    .r-check input[type='checkbox']:checked::before {
      background: var(--text-primary);
      box-shadow: none;
    }
    /* The check: a short L turned 45°, in the inverse ink on the filled box. */
    .r-check input[type='checkbox']:checked::after {
      content: '';
      position: absolute;
      left: 50%;
      top: 50%;
      width: 6px;
      height: 12px;
      border: solid var(--text-inverse);
      border-width: 0 2px 2px 0;
      transform: translate(-50%, -60%) rotate(45deg);
    }
    .r-check input[type='checkbox']:disabled {
      cursor: not-allowed;
    }
    /* Off (four runs picked, the most Compare takes): an outline only, the control edge at half strength, so it never
       reads as filled, and a filled square means checked (round 26: a dark inset fill read as checked). */
    .r-check input[type='checkbox']:disabled::before {
      background: transparent;
      box-shadow: inset 0 0 0 1.5px color-mix(in srgb, var(--line-control) 50%, transparent);
    }
    /* Focus rings the drawn square, not the invisible target around it. */
    .r-check input[type='checkbox']:focus-visible {
      outline: none;
    }
    .r-check input[type='checkbox']:focus-visible::before {
      outline: var(--size-focus-ring) solid var(--focus);
      outline-offset: var(--size-focus-offset);
    }
    /* 56 px at least; a row that grows a line (a failure's cause) keeps 8 px above and below its text. */
    .r-main {
      flex: 1;
      min-width: 0;
      display: flex;
      align-items: center;
      gap: var(--space-3);
      min-height: 56px;
      padding-block: var(--space-2);
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
    .r-meta,
    .r-cause {
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-tertiary);
    }
    /* What went wrong, in words, over the epoch and dataset: secondary ink, the status's dot already says failed. */
    .r-cause {
      color: var(--text-secondary);
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
`;
