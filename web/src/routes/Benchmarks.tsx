import Plus from '$icons/plus';
import Search from '$icons/search';
import Target from '$icons/target';
import { computed, css, For, MediaQuery, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { Bench } from '$lib/api/types';
import { media } from '$lib/design/tokens';
import { ago, date, num } from '$lib/format';
import { href, router } from '$lib/router';
import Page from '$lib/shell/Page';
import PhoneSections from '$lib/shell/PhoneSections';
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
import { rowNav } from '$lib/ui/rowNav';
import Segmented from '$lib/ui/Segmented';
import Skeleton from '$lib/ui/Skeleton';
import SortHeader from '$lib/ui/SortHeader';
import StatusBadge from '$lib/ui/StatusBadge';
import TableState from '$lib/ui/TableState';
import { tooltip } from '$lib/ui/tooltip';
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
  { key: 'leader', label: 'Leader', get: (b) => b.best?.score ?? -1, w: '256px' },
  { key: 'created', label: 'Created', num: true, get: (b) => Date.parse(b.created), hide: 'md', w: '104px' },
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
  // Polled: an entry running somewhere changes a benchmark's leader and its count.
  const benches = new Resource((s) => hub.api.benches(s), { poll: 10_000 });
  const unsupported = computed(() => cannotBench(system.data, benches.error));

  // `#/benchmarks?new=1` opens the dialog (the palette's "New benchmark").
  const creating = signal(untrack(() => router.route.query.get('new') === '1'));

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
  const count = (f: Filter) => (benches.data ?? []).filter((b) => keep(b, f)).length;
  const shown = computed(() => {
    const q = query.get().trim().toLowerCase();
    const key = sortKey.get();
    const col = cols.find((c) => c.key === key) ?? cols[0]!;
    const sign = sortDir.get() === 'asc' ? 1 : -1;
    const f = filter.get();
    return (benches.data ?? [])
      .filter((b) => keep(b, f) && (!q || `${b.name} ${b.source_name} ${b.classes.join(' ')} ${b.best?.name ?? ''}`.toLowerCase().includes(q)))
      .sort((a, b) => {
        const x = col.get(a);
        const y = col.get(b);
        const d = typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y));
        return d * sign || a.name.localeCompare(b.name);
      });
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
  const phone = new MediaQuery(media.phone, false);
  const open = (b: Bench) => router.go(`/benchmarks/${b.id}`);

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
            {!unsupported.get() ? (
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
              <div class="scroll" class:state={!hasRows.get()}>
                <table class="table benches" aria-busy={!benches.data} aria-label="Benchmarks" use:rowNav>
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
                            <Button size="sm" onclick={() => (query.set(''), filter.set('all'))}>
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
              </div>{' '}
              {/* Phones: the name, then where its images came from and how many; the leader's score on the right. */}
              {shown.get().length ? (
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
              ) : !benches.data && !benches.error ? (
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
              ) : null}
            </Panel>{' '}
            {/* No totals line under the table (they are the header's facts): only a narrowed list says what it shows. */}
            {benches.data && shown.get().length < benches.data.length ? (
              <p class="list-foot num">
                {shown.get().length} of {benches.data.length} {benches.data.length === 1 ? 'benchmark' : 'benchmarks'}
              </p>
            ) : null}
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
