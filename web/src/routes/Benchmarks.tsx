// The Benchmarks list (`#/benchmarks`): the server's vision benchmarks, filtered by source or by what is running, and
// the New benchmark dialog (`?new=1`). The parts are in ./bench/.
import Download from '$icons/download';
import Plus from '$icons/plus';
import Search from '$icons/search';
import Target from '$icons/target';
import { computed, css, effect, For, MediaQuery, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { Bench } from '$lib/api/types';
import { BENCH_FEATURE } from '$lib/bench/rules';
import { media } from '$lib/design/tokens';
import { ago, date, num } from '$lib/format';
import { href, router } from '$lib/router';
import Page from '$lib/shell/Page';
import PhoneSections from '$lib/shell/PhoneSections';
import { ownCreateAction, setCrumbs } from '$lib/shell/crumbs';
import { exportTableCsv, exportTableJson } from '$lib/tableExport';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Gate from '$lib/ui/Gate';
import Input from '$lib/ui/Input';
import Menu, { type MenuItem } from '$lib/ui/Menu';
import Meta from '$lib/ui/Meta';
import NoValue from '$lib/ui/NoValue';
import { isOffline, OFFLINE_REASON } from '$lib/ui/offline';
import Panel from '$lib/ui/Panel';
import { rowNav } from '$lib/ui/rowNav';
import Segmented from '$lib/ui/Segmented';
import Skeleton from '$lib/ui/Skeleton';
import SortHeader from '$lib/ui/SortHeader';
import StatusBadge from '$lib/ui/StatusBadge';
import TableState from '$lib/ui/TableState';
import { tooltip } from '$lib/ui/tooltip';
import { benchmarksExport } from './bench/export';
import NewBenchDialog from './bench/NewBenchDialog';
import { cannotBench } from './bench/support';
import Unsupported from './bench/Unsupported';
import { entriesText, entryStatus, imagesText, scoreText, sourceKind } from './bench/words';
import { tickingNow } from './runs/when';

type Key = 'name' | 'source' | 'images' | 'classes' | 'conditions' | 'entries' | 'leader' | 'created';
interface Col {
  key: Key;
  label: string;
  num?: boolean;
  /** Sort key value; `null` sorts last in both ascending and descending order. */
  get: (b: Bench) => number | string | null;
  hide?: 'md' | 'lg';
  /** Fixed width on the header row loading and loaded share; the name takes what is left. */
  w?: string;
}
const finite = (v: number | null | undefined): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const cols: Col[] = [
  { key: 'name', label: 'Benchmark', get: (b) => b.name.toLowerCase() },
  { key: 'source', label: 'Images from', get: (b) => b.source_name.toLowerCase(), hide: 'md', w: '208px' },
  { key: 'images', label: 'Images', num: true, get: (b) => b.images, w: '88px' },
  { key: 'classes', label: 'Classes', num: true, get: (b) => b.classes.length, hide: 'lg', w: '88px' },
  { key: 'conditions', label: 'Conditions', num: true, get: (b) => b.plan.conditions.length, hide: 'lg', w: '104px' },
  { key: 'entries', label: 'Entries', num: true, get: (b) => b.entries, w: '88px' },
  { key: 'leader', label: 'Leader', get: (b) => finite(b.best?.score), w: '256px' },
  { key: 'created', label: 'Created', num: true, get: (b) => finite(Date.parse(b.created)), hide: 'md', w: '104px' },
];
const COLS = cols.length;
const hideCls = (c: Col) => (c.hide ? `hide-${c.hide}` : '');

type Filter = 'all' | 'dataset' | 'label_job' | 'active';
const keep = (b: Bench, f: Filter) => f === 'all' || (f === 'active' ? b.active : b.source.kind === f);

export default function Benchmarks() {
  // Relative times ("made 3m ago") tick while the page is open (runs/when).
  const clock = tickingNow();
  setCrumbs({ label: 'Benchmarks' });
  ownCreateAction();

  const system = new Resource((s) => hub.api.system(s));
  const noBench = computed(() => hub.lacks(BENCH_FEATURE) || cannotBench(system.data));
  // Polled: an entry running somewhere changes a benchmark's leader and its count. Skip fetching/polling once health
  // or `/api/system` says the server lacks benchmarks (e.g. `folo serve --mock`, #309).
  const benches = new Resource((s) => (noBench.get() ? Promise.resolve<Bench[]>([]) : hub.api.benches(s)), { poll: 10_000 });
  const unsupported = computed(() => noBench.get() || cannotBench(system.data, benches.error));
  const unavailable = unsupported;

  // `#/benchmarks?new=1` opens the dialog (the palette's "New benchmark"), both on initial mount and when navigated
  // to while already on `/benchmarks`, then strips `?new=1` so reload/Back never reopens it. Suppressed and closed
  // whenever server health or capability checks say benchmarks are unavailable (`folo serve --mock`, #324).
  const creating = signal(false);
  effect(() => {
    if (router.route.name !== 'benchmarks' || router.route.query.get('new') !== '1') return;
    untrack(() => {
      if (!unavailable.peek()) creating.set(true);
      const q = new URLSearchParams(router.route.query);
      q.delete('new');
      const s = q.toString();
      router.go(`/benchmarks${s ? `?${s}` : ''}`, { replace: true });
    });
  });
  effect(() => {
    if (unavailable.get() && creating.peek()) creating.set(false);
  });

  // Newest first, as the server sends them.
  const sortKey = signal<Key>('created');
  const sortDir = signal<'asc' | 'desc'>('desc');
  function sortBy(c: Col) {
    if (sortKey.peek() === c.key) sortDir.set((d) => (d === 'asc' ? 'desc' : 'asc'));
    else {
      sortKey.set(c.key);
      sortDir.set(c.num || c.key === 'leader' ? 'desc' : 'asc');
    }
  }

  const query = signal('');
  // Where the images came from, or what runs now (`Bench.active`: an entry or a loop queued or running).
  const filter = signal<Filter>('all');
  const counts = computed(
    () => {
      let all = 0;
      let dataset = 0;
      let label_job = 0;
      let active = 0;
      for (const b of benches.data ?? []) {
        all++;
        if (b.source.kind === 'dataset') dataset++;
        else if (b.source.kind === 'label_job') label_job++;
        if (b.active) active++;
      }
      return { all, dataset, label_job, active };
    },
    { equals: (a, b) => a.all === b.all && a.dataset === b.dataset && a.label_job === b.label_job && a.active === b.active },
  );
  const count = (f: Filter) => counts.get()[f];
  const filtered = computed(() => {
    const q = query.get().trim().toLowerCase();
    const f = filter.get();
    return (benches.data ?? []).filter(
      (b) => keep(b, f) && (!q || `${b.name} ${b.source_name} ${b.classes.join(' ')} ${b.best?.name ?? ''}`.toLowerCase().includes(q)),
    );
  });
  const shown = computed(() => {
    const key = sortKey.get();
    const col = cols.find((c) => c.key === key) ?? cols[0]!;
    const sign = sortDir.get() === 'asc' ? 1 : -1;
    return filtered
      .get()
      .map((b) => ({ b, k: col.get(b) }))
      .sort((a, b) => {
        const x = a.k;
        const y = b.k;
        if (x === null || y === null) {
          if (x === y) return a.b.name.localeCompare(b.b.name);
          return x === null ? 1 : -1;
        }
        const d = typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y));
        return d * sign || a.b.name.localeCompare(b.b.name);
      })
      .map((x) => x.b);
  });
  // The header's facts line (design/UNIFIED.md §2, Page header): what the page holds, "4 benchmarks · 1,240 images ·
  // 9 entries · 1 running", whatever the filter; a narrowed list says what it shows under it.
  const facts = computed<string[] | null>(() => {
    const all = benches.data;
    if (!all?.length) return null;
    const active = count('active');
    return [
      `${num(all.length)} ${all.length === 1 ? 'benchmark' : 'benchmarks'}`,
      imagesText(all.reduce((a, b) => a + b.images, 0)),
      entriesText(all.reduce((a, b) => a + b.entries, 0)),
      active ? `${active} running` : '',
    ]
      .filter(Boolean);
  });

  const offline = computed(() => !benches.data && isOffline(benches.error));
  const hasData = computed(() => (benches.data ? benches.data.length > 0 : !benches.error));
  const hasRows = computed(() => (benches.data ? shown.get().length > 0 : !benches.error));
  const canExport = computed(() => !unavailable.get() && !!benches.data && shown.get().length > 0);
  const phone = new MediaQuery(media.phone, false);
  const searchEl = signal<HTMLInputElement | undefined>(undefined);
  const liveNote = signal('');
  function announce(message: string) {
    if (liveNote.peek() === message) {
      liveNote.set('');
      queueMicrotask(() => liveNote.set(message));
    } else {
      liveNote.set(message);
    }
  }
  const hasFiltered = signal(false);
  effect(() => {
    const active = filter.get() !== 'all' || query.get().trim().length > 0;
    if (active && !hasFiltered.peek()) hasFiltered.set(true);
  });
  const filterStatus = computed(() => {
    const all = benches.data;
    if (!all || !hasFiltered.get()) return '';
    const n = shown.get().length;
    return n < all.length
      ? `Showing ${n} of ${all.length} ${all.length === 1 ? 'benchmark' : 'benchmarks'}.`
      : `Showing all ${all.length} ${all.length === 1 ? 'benchmark' : 'benchmarks'}.`;
  });
  const open = (b: Bench) => router.go(`/benchmarks/${b.id}`);

  function exportList(format: 'csv' | 'json') {
    const list = shown.peek();
    if (!list.length) return;
    const payload = benchmarksExport(list);
    const ok =
      format === 'csv'
        ? exportTableCsv(payload.filenameBase, payload.headers, payload.rows)
        : exportTableJson(payload.filenameBase, payload.json);
    if (ok) announce(`Exported ${list.length} ${list.length === 1 ? 'benchmark' : 'benchmarks'} as ${format.toUpperCase()}.`);
  }
  const exportItems: MenuItem[] = [
    { label: 'Export CSV', icon: Download, onclick: () => exportList('csv') },
    { label: 'Export JSON', icon: Download, onclick: () => exportList('json') },
  ];
  function clearFilters() {
    query.set('');
    filter.set('all');
    announce('Cleared benchmark filters.');
    queueMicrotask(() => searchEl.peek()?.focus());
  }

  return (
    <>
      <Page
        title="Benchmarks"
        // What the page holds (what a benchmark is, frozen labeled images that grade any model, is the empty state's).
        sub={facts.get() ? () => <span class="num">{facts.get()!.join(' · ')}</span> : undefined}
        // Phones: the first two facts, beside the page's action (the rest are a wider screen's).
        subPhone={facts.get() ? () => <span class="num">{facts.get()!.slice(0, 2).join(' · ')}</span> : undefined}
        actions={() => (
          <>
            {canExport.get() ? (
              <Menu icon={Download} text={phone.current ? undefined : 'Export'} label="Export benchmarks" items={exportItems} />
            ) : null}
            {!unavailable.get() ? (
              <Gate reason={offline.get() ? OFFLINE_REASON : null}>
                {(why) => (
                  <Button variant="primary" icon={Plus} disabled={!!why()} aria-describedby={why()} onclick={() => creating.set(true)}>
                    {phone.current ? 'New' : 'New benchmark'}
                  </Button>
                )}
              </Gate>
            ) : null}
          </>
        )}
      >
        <PhoneSections
          label="Models and benchmarks"
          items={[
            { label: 'Models', path: '/models', current: false },
            { label: 'Benchmarks', path: '/benchmarks', current: true },
          ]}
        />{' '}
        {unsupported.get() ? (
          <Unsupported />
        ) : (
          <>
            <div class="list-toolbar">
              <div class="lt-filter">
                <Segmented
                  label="Filter benchmarks by where their images came from, or by what runs now"
                  bind:value={filter}
                  disabled={!hasData.get()}
                  options={[
                    { value: 'all', label: 'All', meta: benches.data?.length ? String(count('all')) : undefined },
                    { value: 'dataset', label: 'Datasets', meta: benches.data?.length ? String(count('dataset')) : undefined },
                    { value: 'label_job', label: 'Labeling jobs', meta: benches.data?.length ? String(count('label_job')) : undefined },
                    { value: 'active', label: 'Running', meta: benches.data?.length ? String(count('active')) : undefined },
                  ]}
                />
              </div>{' '}
              <div class="lt-search">
                <Input
                  bind:el={searchEl}
                  bind:value={query}
                  icon={Search}
                  placeholder="Filter by name, source or class"
                  aria-label="Filter benchmarks"
                  type="search"
                  disabled={!hasData.get()}
                />
              </div>
            </div>{' '}
            <Panel pad="none">
              {phone.current && hasRows.get() ? (
                shown.get().length ? (
                  /* Phones: the name, then where its images came from and how many; the leader's score on the right. */
                  <ul class="rows" aria-label="Benchmarks">
                    <For each={shown.get()} key={(b) => b.id}>
                      {(b) => (
                        <li>
                          <a href={href(`/benchmarks/${b().id}`)}>
                            <span class="m-main">
                              <span class="nm-row">
                                <span class="name truncate">{b().name}</span>
                                {b().active ? <StatusBadge plain dotOnly state={entryStatus({ status: 'running' })} /> : null}
                              </span>{' '}
                              <span class="m-meta num">
                                <Meta>
                                  <span>{b().source_name}</span>
                                  <span>{imagesText(b().images)}</span>
                                  <span>{entriesText(b().entries)}</span>
                                </Meta>
                              </span>
                            </span>{' '}
                            <span class="m-side">
                              {b().best ? (
                                <>
                                  <span class="num m-score">
                                    <span class="sr-only">Leader's score</span>
                                    {b().best!.partial ? <span class="partial">partial</span> : null}
                                    {scoreText(b().best!.score)}
                                  </span>
                                  <span class="m-who truncate">{b().best!.name}</span>
                                </>
                              ) : (
                                <NoValue label="no entries yet" />
                              )}
                            </span>
                          </a>
                        </li>
                      )}
                    </For>
                  </ul>
                ) : (
                  <ul class="rows" aria-hidden="true">
                    <For each={[0, 1, 2]} key={(i) => i}>
                      {() => (
                        <li>
                          <span class="row-sk">
                            <span class="m-main">
                              <Skeleton w="55%" h="14px" />
                              <Skeleton w="70%" h="10px" />
                            </span>
                            <Skeleton w="40px" h="14px" />
                          </span>
                        </li>
                      )}
                    </For>
                  </ul>
                )
              ) : (
                <div class="scroll" class:state={!hasRows.get()}>
                <table class="table benches" aria-busy={!benches.data && !benches.error} aria-label="Benchmarks" use:rowNav>
                  {hasRows.get() ? (
                    <thead>
                      <tr>
                        <For each={cols} key={(c) => c.key}>
                          {(c) => (
                            <SortHeader
                              label={c().label}
                              class={hideCls(c())}
                              align={c().num ? 'right' : 'left'}
                              width={c().w}
                              dir={sortKey.get() === c().key ? sortDir.get() : null}
                              disabled={!benches.data}
                              onsort={() => sortBy(c())}
                            />
                          )}
                        </For>
                      </tr>
                    </thead>
                  ) : null}
                  <tbody>
                    {benches.error && !benches.data ? (
                      <TableState cols={COLS}>
                        <ErrorState error={benches.error} title="Could not load benchmarks" retry={() => benches.refresh()} />
                      </TableState>
                    ) : benches.data?.length === 0 ? (
                      <TableState cols={COLS}>
                        <EmptyState
                          icon={Target}
                          title="No benchmarks yet"
                          description="A benchmark freezes a set of your labeled images and grades any model on them: unchanged, and under haze, low light, blur or whatever your camera meets. Make one to see how a model holds up before you rely on it."
                          actions={() => (
                            <Button icon={Plus} onclick={() => creating.set(true)}>
                              New benchmark
                            </Button>
                          )}
                        />
                      </TableState>
                    ) : benches.data && shown.get().length === 0 ? (
                      <TableState cols={COLS}>
                        <EmptyState
                          icon={Search}
                          title="No benchmarks match"
                          description="Try another source or search term."
                          actions={() => (
                            <Button size="sm" onclick={clearFilters}>
                              Clear filters
                            </Button>
                          )}
                        />
                      </TableState>
                    ) : !benches.data ? (
                      <For each={[0, 1, 2]} key={(i) => i}>
                        {() => (
                          <tr aria-hidden="true">
                            <For each={cols} key={(c) => c.key}>
                              {(c) => (
                                <td class={hideCls(c())} class:r={!!c().num}>
                                  <Skeleton h="12px" w={c().key === 'name' ? '140px' : c().key === 'leader' || c().key === 'source' ? '120px' : '32px'} />
                                </td>
                              )}
                            </For>
                          </tr>
                        )}
                      </For>
                    ) : (
                      <For each={shown.get()} key={(b) => b.id}>
                        {(b) => (
                          <tr class="link" data-row data-id={b().id} aria-label={b().name} onclick={() => open(b())}>
                            <td class="name-cell">
                              <span class="nm-row">
                                <a class="name truncate" href={href(`/benchmarks/${b().id}`)} onclick={(e) => e.stopPropagation()} title={b().name}>
                                  {b().name}
                                </a>{' '}
                                {b().active ? (
                                  <span class="run" use:tooltip={'An entry or a loop is running'}>
                                    <StatusBadge plain dotOnly state={entryStatus({ status: 'running' })} />
                                  </span>
                                ) : null}
                              </span>
                            </td>
                            <td class="hide-md src">
                              <Meta line>
                                <span>{b().source_name}</span>
                                <span>{sourceKind(b().source)}</span>
                              </Meta>
                            </td>
                            <td class="r num muted">{num(b().images)}</td>
                            <td class="r num muted hide-lg">{b().classes.length}</td>
                            <td class="r num muted hide-lg">{b().plan.conditions.length}</td>
                            <td class="r num muted">{num(b().entries)}</td>
                            {/* The leader cell (design/UNIFIED.md §4, Benchmarks): the score in text.primary 500, then the
                                entry's name, cut when long (its own page has it whole). */}
                            <td class="leader">
                              {b().best ? (
                                <span class="lead-row">
                                  <span class="l-score num">
                                    {b().best!.partial ? (
                                      <span class="partial" use:tooltip={'The leader’s score is partial: it compares only loosely with complete entries'}>
                                        partial
                                      </span>
                                    ) : null}
                                    {scoreText(b().best!.score)}
                                  </span>
                                  <span class="l-name truncate" title={b().best!.name}>
                                    {b().best!.name}
                                  </span>
                                </span>
                              ) : (
                                <NoValue label={b().entries ? 'no finished entry yet' : 'no entries yet'} />
                              )}
                            </td>
                            <td class="r faint hide-md" title={date(b().created)}>
                              {ago(b().created, clock.get())}
                            </td>
                          </tr>
                        )}
                      </For>
                    )}
                  </tbody>
                </table>
              </div>
              )}
            </Panel>{' '}
            {/* No totals line under the table (they are the header's facts): only a narrowed list says what it shows. */}
            {benches.data && shown.get().length < benches.data.length ? (
              <p class="list-foot num">
                {shown.get().length} of {benches.data.length} {benches.data.length === 1 ? 'benchmark' : 'benchmarks'}
              </p>
            ) : null}
            <p class="sr-only" role="status" aria-live="polite" data-filter-status>
              {filterStatus.get()}
            </p>
            <p class="sr-only" role="status" aria-live="polite" data-live-note>
              {liveNote.get()}
            </p>
          </>
        )}
      </Page>{' '}
      <NewBenchDialog bind:open={creating} />
    </>
  );
}

css`
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
  /* The leader: its score, then its name (cut when long; its own page has it whole). */
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
    flex-shrink: 0;
    color: var(--text-primary);
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
`;
