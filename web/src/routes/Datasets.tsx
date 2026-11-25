// The Datasets list (`#/datasets`): every registered dataset with its class balance and box count, and adding one
// (a folder to register, or a generated sample set). The parts are in ./datasets/.
import FolderOpen from '$icons/folder-open';
import Images from '$icons/images';
import Plus from '$icons/plus';
import Search from '$icons/search';
import Sparkles from '$icons/sparkles';
import Tags from '$icons/tags';
import { computed, css, effect, For, MediaQuery, onCleanup, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { Dataset } from '$lib/api/types';
import { media } from '$lib/design/tokens';
import ClassStrip from '$lib/detect/ClassStrip';
import { ago, count } from '$lib/format';
import { LABEL_FEATURE } from '$lib/label';
import { href, router } from '$lib/router';
import Page from '$lib/shell/Page';
import { ownCreateAction, setCrumbs } from '$lib/shell/crumbs';
import Badge from '$lib/ui/Badge';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Input from '$lib/ui/Input';
import Panel from '$lib/ui/Panel';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import Skeleton from '$lib/ui/Skeleton';
import Gate from '$lib/ui/Gate';
import { isOffline, OFFLINE_REASON } from '$lib/ui/offline';
import Meta from '$lib/ui/Meta';
import AddDatasetDialog from './datasets/AddDatasetDialog';
import DatasetCard, { cardFacts, DatasetCardSkeleton } from './datasets/DatasetCard';
import { imageCount } from './dataset/health';
import { datasetChanges, generationEnded, landed, pendingGeneration } from './datasets/changes';
import { emptyFilterSummary, leftForDataset, readState, returnQuery, writeState, type Format, type SortKey } from './datasets/listState';
import { datasetProblem, missingError, PROBLEM_LABEL, readDataset, type ListedDataset } from './datasets/problem';

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: 'name', label: 'Name' },
  { value: 'images', label: 'Most images' },
  { value: 'boxes', label: 'Most boxes' },
  { value: 'classes', label: 'Most classes' },
];
const cmpName = (a: ListedDataset, b: ListedDataset) =>
  a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }) || a.id.localeCompare(b.id);
const sameBase = (a: Dataset, b: Dataset) =>
  a.name === b.name &&
  a.path === b.path &&
  a.format === b.format &&
  a.splits.train === b.splits.train &&
  a.splits.val === b.splits.val &&
  a.splits.test === b.splits.test &&
  a.classes.length === b.classes.length &&
  a.classes.every((c, i) => c === b.classes[i]);
/** How often the list reads itself again while a sample set this tab asked for still generates. */
const GENERATION_POLL_MS = 3000;

export default function Datasets() {
  setCrumbs({ label: 'Datasets' });
  // The header's create action is the one in the top-right corner (the top bar's New run steps aside).
  ownCreateAction();

  // Render the list as soon as GET /api/datasets answers, then fill each dataset's stats and folder status in
  // independently so one slow dataset never holds back the whole page.
  // A dataset whose folder was found missing is not read again on every visit (./datasets/problem readDataset): only
  // when the list may have changed (`again`, below).
  const listed = new Resource((s) => hub.api.datasets(s));
  const details = signal<ReadonlyMap<string, ListedDataset>>(new Map());
  const detailTrigger = signal(0);
  let wantFresh = false;
  let wantRecover = false;
  let detailSeq = 0;
  let detailCtrl: AbortController | null = null;
  onCleanup(() => {
    detailSeq++;
    detailCtrl?.abort();
    detailCtrl = null;
  });

  const reread = () => {
    wantFresh = true;
    void listed.refresh();
  };

  // When the server comes back after an outage, `listed` refreshes automatically; mark `wantRecover` so even if the
  // registration list is unchanged (`reuse()`), transiently failed dataset detail reads are retried once it settles.
  let lastBack = untrack(() => hub.back);
  effect(() => {
    const now = hub.back;
    if (now === lastBack) return;
    lastBack = now;
    if (hub.mode !== 'connecting') wantRecover = true;
  });

  effect(() => {
    const list = listed.data;
    void detailTrigger.get();
    if (!list) {
      detailSeq++;
      detailCtrl?.abort();
      detailCtrl = null;
      wantFresh = false;
      wantRecover = false;
      details.set(new Map());
      return;
    }
    if (untrack(() => listed.error || listed.stale)) return;
    // Wait for an in-flight reread/recovery refresh to finish so we never consume its intent against the prior list.
    if (untrack(() => listed.refreshing && (wantFresh || wantRecover))) return;
    const api = untrack(() => hub.api);
    const fresh = wantFresh;
    const recover = wantRecover;
    wantFresh = false;
    wantRecover = false;
    const seq = ++detailSeq;
    detailCtrl?.abort();
    const ctl = (detailCtrl = new AbortController());
    const prev = untrack(() => details.get());
    const seed = new Map<string, ListedDataset>();
    const toFetch: Dataset[] = [];
    for (const d of list) {
      const existing = prev.get(d.id);
      if (!fresh && !recover && existing && !existing.pending && (existing.stats || existing.problem) && sameBase(existing, d)) {
        seed.set(d.id, existing);
        continue;
      }
      if (!fresh) {
        const known = missingError(api, d.id);
        if (known !== undefined) {
          seed.set(d.id, existing && !existing.pending && existing.problem === datasetProblem(known) && sameBase(existing, d) ? existing : { ...d, problem: datasetProblem(known) });
          continue;
        }
      }
      if (existing && !existing.pending) {
        // Keep the existing object identity when base fields did not change so cards do not re-trigger failed mosaics twice.
        seed.set(d.id, sameBase(existing, d) ? existing : { ...d, stats: existing.stats, problem: existing.problem });
      } else {
        seed.set(d.id, { ...d, pending: true });
      }
      toFetch.push(d);
    }
    details.set(seed);
    for (const d of toFetch) {
      void readDataset(api, d.id, () => api.dataset(d.id, ctl.signal), fresh).then(
        (full) => {
          if (ctl.signal.aborted || seq !== detailSeq) return;
          details.set((cur) => {
            const next = new Map(cur);
            next.set(d.id, full);
            return next;
          });
        },
        (e: unknown) => {
          if (ctl.signal.aborted || seq !== detailSeq) return;
          details.set((cur) => {
            const next = new Map(cur);
            next.set(d.id, { ...d, problem: datasetProblem(e) });
            return next;
          });
        },
      );
    }
  });

  // Resource.reuse() preserves `listed.data` when the list itself did not change; once a refresh settles cleanly,
  // bump `detailTrigger` if an explicit reread or outage recovery is still pending.
  let wasRefreshing = false;
  effect(() => {
    const refreshing = listed.refreshing;
    const ok = !!listed.data && !listed.error && !listed.stale;
    if (wasRefreshing && !refreshing && ok && (wantFresh || wantRecover)) {
      untrack(() => detailTrigger.set((n) => n + 1));
    }
    wasRefreshing = refreshing;
  });

  const datasetsData = computed<ListedDataset[] | undefined>(() => {
    const list = listed.data;
    if (!list) return undefined;
    const map = details.get();
    return list.map((d) => map.get(d.id) ?? { ...d, pending: true });
  });

  // The list goes stale otherwise: read it again after a dataset action in this tab, and when the tab comes back (one
  // added in another tab, by an agent, or a folder moved meanwhile).
  const seen = datasetChanges.peek();
  effect(() => {
    if (datasetChanges.get() !== seen) untrack(reread);
  });
  // A sample set this tab asked for of this server (or a folder it asked to register) is still on its way, also after
  // a reload: the list says so, and reads itself a few seconds after each read has settled (one slower than that is
  // never cut short by the next), while the tab is shown, until it is there (./datasets/changes).
  const waiting = signal<{ what: string; since: number } | null>(null);
  effect(() => {
    const list = listed.data;
    const server = untrack(() => hub.server);
    const g = untrack(() => pendingGeneration(server));
    if (!g || !list || landed(g, list)) {
      waiting.set(null);
      if (g && list) generationEnded(g);
      return;
    }
    waiting.set({ what: g.what ?? `Generating ${g.name}`, since: g.since });
    // A read that finds nothing new keeps the same list (Resource), so this runs on its own clock until one does.
    let live = true;
    let t: ReturnType<typeof setTimeout>;
    const later = () => {
      if (live) t = setTimeout(tick, GENERATION_POLL_MS);
    };
    const tick = () => {
      if (!live || !pendingGeneration(server)) return void waiting.set(null);
      // Its "started …" moves on with each read, even one that finds nothing new.
      waiting.set((w) => w && { ...w });
      if (document.visibilityState !== 'visible' || listed.loading || listed.refreshing) return later();
      void listed.refresh().finally(later);
    };
    later();
    return () => {
      live = false;
      clearTimeout(t);
    };
  });
  effect(() => {
    const back = () => document.visibilityState === 'visible' && !listed.loading && reread();
    document.addEventListener('visibilitychange', back);
    return () => document.removeEventListener('visibilitychange', back);
  });

  // `#/datasets?add=folder|synthetic` opens the dialog (the onboarding on an empty Overview, New run's empty state), also
  // when it is followed here, in place. Closing the dialog takes `?add=` out of the URL, so a reload does not open it
  // again.
  const addParam = computed(() => (router.route.name === 'datasets' ? router.route.query.get('add') : null));
  const asked = (p: string | null): p is 'folder' | 'synthetic' => p === 'folder' || p === 'synthetic';
  const first = untrack(() => addParam.get());
  const adding = signal(asked(first));
  const addMode = signal<'folder' | 'synthetic'>(first === 'synthetic' ? 'synthetic' : 'folder');
  let openedFromUrl = asked(first);
  function add(m: 'folder' | 'synthetic', fromUrl = false) {
    openedFromUrl = fromUrl;
    addMode.set(m);
    adding.set(true);
  }
  effect(() => {
    const p = addParam.get();
    if (asked(p)) untrack(() => add(p, true));
    else if (openedFromUrl && untrack(() => adding.get())) {
      openedFromUrl = false;
      untrack(() => adding.set(false));
    }
  });
  effect(() => {
    if (adding.get()) return;
    openedFromUrl = false;
    const q = untrack(() => router.route.query);
    if (untrack(() => router.route.name) !== 'datasets' || untrack(() => router.location.name) !== 'datasets' || !q.has('add')) return;
    const rest = new URLSearchParams(q);
    rest.delete('add');
    untrack(() => router.go(`/datasets${rest.toString() ? `?${rest}` : ''}`, { replace: true }));
  });

  // Filter, sort and search live in the URL query (./datasets/listState), replaced as they change so typing never
  // pollutes browser history while Back, reload and shared links restore the list as it was left.
  const back = untrack(() => returnQuery(router.entry, router.route.query));
  if (back !== null) router.go(`/datasets?${back}`, { replace: true });
  const startQuery = back !== null ? new URLSearchParams(back) : untrack(() => router.route.query);
  const start = readState(startQuery);
  const query = signal(start.q);
  // The toolbar's filter, at its left as on Runs and Models (one shape for the three list toolbars): by format.
  const format = signal<Format>(start.format);
  const sort = signal<SortKey>(start.sort);
  const norm = (q: URLSearchParams) => writeState(readState(q));
  let written = norm(startQuery);
  onCleanup(() => {
    if (router.location.name === 'dataset') leftForDataset(router.entry, written);
  });
  effect(() => {
    const next = writeState({ format: format.get(), sort: sort.get(), q: query.get() });
    // Guard against both `router.route` and `router.location` so a filter change while another route's chunk is still
    // loading never overwrites that pending navigation's history entry.
    if (next === written || untrack(() => router.route.name) !== 'datasets' || untrack(() => router.location.name) !== 'datasets') return;
    written = next;
    const full = writeState({ format: format.peek(), sort: sort.peek(), q: query.peek() }, untrack(() => router.route.query));
    router.go(`/datasets${full ? `?${full}` : ''}`, { replace: true });
  });
  effect(() => {
    const r = router.route;
    if (r.name !== 'datasets' || norm(r.query) === written) return;
    const st = readState(r.query);
    written = norm(r.query);
    format.set(st.format);
    sort.set(st.sort);
    query.set(st.q);
  });

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
    const active = format.get() !== 'all' || query.get().trim().length > 0;
    if (active && !hasFiltered.peek()) hasFiltered.set(true);
  });
  function clearFilters() {
    query.set('');
    format.set('all');
    announce('Cleared dataset filters.');
    queueMicrotask(() => {
      const input = searchEl.peek();
      if (input && input.isConnected && !input.disabled) input.focus();
    });
  }

  const formatCounts = computed(
    () => {
      let all = 0;
      let yolo = 0;
      let coco = 0;
      for (const d of listed.data ?? []) {
        all++;
        if (d.format === 'yolo') yolo++;
        else if (d.format === 'coco') coco++;
      }
      return { all, yolo, coco };
    },
    { equals: (a, b) => a.all === b.all && a.yolo === b.yolo && a.coco === b.coco },
  );
  const formatCount = (f: Format) => formatCounts.get()[f];
  const filtered = computed(() => {
    const q = query.get().trim().toLowerCase();
    const f = format.get();
    return (datasetsData.get() ?? []).filter(
      (d) => (f === 'all' || d.format === f) && (!q || `${d.name} ${d.path} ${d.format} ${d.id}`.toLowerCase().includes(q)),
    );
  });
  const shown = computed(() => {
    const s = sort.get();
    return filtered.get().slice().sort((a, b) => {
      if (s === 'images') return (b.problem ? -1 : imageCount(b)) - (a.problem ? -1 : imageCount(a)) || cmpName(a, b);
      if (s === 'boxes') return (b.problem ? -1 : (b.stats?.boxes ?? -1)) - (a.problem ? -1 : (a.stats?.boxes ?? -1)) || cmpName(a, b);
      if (s === 'classes') return b.classes.length - a.classes.length || cmpName(a, b);
      return cmpName(a, b);
    });
  });
  // The header's facts line (design/UNIFIED.md §2, Page header): what the page holds, "5 datasets · 9,400 images ·
  // 64,086 boxes", whatever the filter; a narrowed list says what it shows under it. A dataset that cannot be read
  // holds no images now: the counts of its registration are not added (stress s1-datasets-6).
  const facts = computed<string[] | null>(() => {
    const list = datasetsData.get();
    if (!list?.length) return null;
    const boxes = list.reduce((a, d) => a + (d.stats?.boxes ?? 0), 0);
    const images = list.reduce((a, d) => a + (d.problem ? 0 : imageCount(d)), 0);
    return [count(list.length, 'dataset'), count(images, 'image'), boxes ? count(boxes, 'box') : ''].filter(Boolean);
  });

  const emptySummary = computed(() => emptyFilterSummary({ format: format.get(), q: query.get() }));
  const filterStatus = computed(() => {
    const all = listed.data;
    if (!all?.length || !hasFiltered.get()) return '';
    const n = shown.get().length;
    return n < all.length
      ? `Showing ${n} of ${count(all.length, 'dataset')}.`
      : `Showing all ${count(all.length, 'dataset')}.`;
  });

  // Nothing answers: adding a dataset needs the server, so the action is disabled and says why (as on Runs).
  const offline = computed(() => !listed.data && isOffline(listed.error));
  /** Datasets to filter, or on their way: the toolbar is live (else it stays in place, disabled). */
  const hasData = computed(() => (listed.data ? listed.data.length > 0 : !listed.error));
  /** Cards to show (or their skeleton); with none a panel holds the state alone, where the grid would be. */
  const hasRows = computed(() => (listed.data ? shown.get().length > 0 : !listed.error));
  /** Agent labeling is hidden when the server's health explicitly reports it is unavailable (e.g. `folo serve --mock`). */
  const canLabel = computed(() => !hub.lacks(LABEL_FEATURE));
  const phone = new MediaQuery(media.phone, false);

  // Labeling images (agent labeling) lives under Datasets: a secondary action beside Add dataset (the ⋯ on phones).
  return (
    <>
      <Page
        title="Datasets"
        more={canLabel.get() ? [{ label: 'Label images', icon: Tags, href: href('/datasets/label') }] : []}
        moreLabel="More dataset actions"
        // What the page holds (what a dataset is, YOLO or COCO folders, is the empty state's).
        sub={facts.get() ? () => <span class="num">{facts.get()!.join(' · ')}</span> : undefined}
        // Phones: the first two facts, beside the page's action (the rest are a wider screen's).
        subPhone={facts.get() ? () => <span class="num">{facts.get()!.slice(0, 2).join(' · ')}</span> : undefined}
        actions={() => (
          <Gate reason={offline.get() ? OFFLINE_REASON : null}>
            {(why) => (
              <Button variant="primary" icon={Plus} disabled={!!why()} aria-describedby={why()} onclick={() => add('folder')}>
                Add dataset
              </Button>
            )}
          </Gate>
        )}
      >
        {/* Always here, so the grid never moves between states (DESIGN_DIRECTION §2.8): over an empty or failed list it
            is disabled and shows no counts. */}
        <div class="list-toolbar">
          <div class="lt-filter">
            <Segmented
              label="Filter datasets by format"
              bind:value={format}
              disabled={!hasData.get()}
              options={[
                { value: 'all', label: 'All', meta: listed.data?.length ? String(formatCount('all')) : undefined },
                { value: 'yolo', label: 'YOLO', meta: listed.data?.length ? String(formatCount('yolo')) : undefined },
                { value: 'coco', label: 'COCO', meta: listed.data?.length ? String(formatCount('coco')) : undefined },
              ]}
            />
          </div>{' '}
          <div class="lt-sort">
            <label for="datasets-sort">Sort</label>
            <Select id="datasets-sort" label="Sort datasets" bind:value={sort} disabled={!hasData.get()} options={SORT_OPTIONS} />
          </div>{' '}
          <div class="lt-search">
            <Input bind:el={searchEl} bind:value={query} icon={Search} placeholder="Filter by name or path" aria-label="Filter datasets" type="search" disabled={!hasData.get()} />
          </div>
        </div>{' '}
        {hasRows.get() ? (
          phone.current ? (
            /* Phones: the same parts as rows (DESIGN_DIRECTION §12, one list concept); desktop cards stay unmounted. */
            <div class="phone-list">
              <Panel pad="none">
              {listed.data ? (
                <ul class="rows" aria-label="Datasets">
                  <For each={shown.get()} key={(d) => d.id}>
                    {(d) => (
                      // The link covers the row (a stretched link); the class balance, its own disclosure, sits above it.
                      <li class="m-row">
                        <a href={href(`/datasets/${d().id}`)}>
                          <span class="m-main">
                            <span class="name truncate">{d().name}</span>{' '}
                            <span class="m-meta m-total num">
                              <Meta>
                                <span>{cardFacts(d())}</span>
                                <span>{d().format.toUpperCase()}</span>
                              </Meta>
                            </span>
                          </span>
                        </a>{' '}
                        {d().problem ? (
                          <span class="m-side">
                            <Badge tone="warning">{PROBLEM_LABEL[d().problem!]}</Badge>
                          </span>
                        ) : d().stats ? (
                          <span class="m-side">
                            <span class="m-strip">
                              <ClassStrip counts={d().stats!.class_counts} classes={d().classes} height={4} named />
                            </span>
                          </span>
                        ) : d().pending ? (
                          <span class="m-side">
                            <Skeleton w="104px" h="4px" />
                          </span>
                        ) : null}
                      </li>
                    )}
                  </For>
                </ul>
              ) : (
                // Phones, loading: the rows' shape (the name over its meta line, the class balance on the right).
                <ul class="rows" aria-hidden="true">
                  <For each={[0, 1, 2]} key={(i) => i}>
                    {() => (
                      <li>
                        <span class="row-sk">
                          <span class="m-main">
                            <Skeleton w="55%" h="14px" />
                            <Skeleton w="45%" h="10px" />
                          </span>{' '}
                          <span class="m-side">
                            <Skeleton w="104px" h="4px" />
                          </span>
                        </span>
                      </li>
                    )}
                  </For>
                </ul>
              )}
              </Panel>
            </div>
          ) : (
            /* The cards (design/UNIFIED.md §4, Datasets), as many columns of at least 300 px as fit, as Desktop's: three
               at 1440 px, four on a wide content column; a short last row keeps the columns' width. */
            <ul class="cards" aria-label="Datasets" aria-busy={!listed.data}>
              {listed.data ? (
                <For each={shown.get()} key={(d) => d.id}>
                  {(d) => <DatasetCard dataset={d()} />}
                </For>
              ) : (
                <For each={[0, 1, 2]} key={(i) => i}>
                  {() => <DatasetCardSkeleton />}
                </For>
              )}
            </ul>
          )
        ) : (
          // No cards (an error, no datasets, none that match): one panel holds the state where the grid would be.
          <Panel pad="none">
            <div class="state" aria-label="Datasets">
              {listed.error && !listed.data ? (
                <ErrorState error={listed.error} title="Could not load datasets" retry={() => reread()} />
              ) : listed.data?.length === 0 ? (
                // The header already holds the primary action: offer the quick start.
                <EmptyState
                  icon={Images}
                  title="No datasets yet"
                  description={
                    canLabel.get()
                      ? 'Register a folder of images you have labeled (YOLO or COCO), try FOLO in a minute with a generated set of shapes, or have an agent label your images.'
                      : 'Register a folder of images you have labeled (YOLO or COCO), or try FOLO in a minute with a generated set of shapes.'
                  }
                  actions={() => (
                    <>
                      <Button icon={FolderOpen} onclick={() => add('folder')}>
                        Register folder
                      </Button>
                      <Button icon={Sparkles} onclick={() => add('synthetic')}>
                        Generate sample set
                      </Button>
                      {canLabel.get() ? (
                        <Button icon={Tags} href={href('/datasets/label')}>
                          Label images
                        </Button>
                      ) : null}
                    </>
                  )}
                />
              ) : (
                <EmptyState
                  icon={Search}
                  title="No datasets match"
                  description={emptySummary.get()}
                  actions={() => (
                    <Button size="sm" onclick={clearFilters}>
                      Clear filters
                    </Button>
                  )}
                />
              )}
            </div>
          </Panel>
        )}{' '}
        {/* No totals line under the cards (they are the header's facts): only a narrowed list says what it shows, and a
            sample set on its way says so (also after a reload: nothing else on screen tells that one is coming,
            studio-data F1). */}
        {listed.data && (shown.get().length < listed.data.length || waiting.get()) ? (
          <p class="list-foot num">
            <Meta>
              {shown.get().length < listed.data.length ? (
                <span>
                  {shown.get().length} of {count(listed.data.length, 'dataset')}
                </span>
              ) : null}{' '}
              {waiting.get() ? (
                <span class="waiting" role="status">
                  {waiting.get()!.what}, started {ago(new Date(waiting.get()!.since).toISOString())}: it shows up here when it is done
                </span>
              ) : null}
            </Meta>
          </p>
        ) : null}
        <p class="sr-only" role="status" aria-live="polite" data-filter-status>
          {filterStatus.get()}
        </p>
        <p class="sr-only" role="status" aria-live="polite" data-live-note>
          {liveNote.get()}
        </p>
      </Page>{' '}
      <AddDatasetDialog bind:open={adding} bind:mode={addMode} known={listed.data} />
    </>
  );
}

css`
  .lt-sort {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-left: auto;
  }
  .lt-sort label {
    flex-shrink: 0;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .lt-sort :global(.select) {
    width: 160px;
  }
  .list-toolbar .lt-search {
    margin-left: 0;
  }
  @media (max-width: 1023px) {
    .lt-sort {
      margin-left: 0;
    }
  }
  @media (max-width: 767px) {
    .lt-sort {
      flex-basis: 100%;
    }
    .lt-sort :global(.select) {
      flex: 1;
      width: auto;
    }
  }
  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(min(100%, 300px), 1fr));
    gap: var(--space-4);
  }
  .state {
    display: flex;
    flex-direction: column;
  }
  .rows {
    display: block;
  }
  .rows .name {
    font-weight: var(--weight-medium);
  }
  .rows li + li {
    border-top: 1px solid var(--line-subtle);
  }
  .rows .m-row,
  .rows .row-sk {
    position: relative;
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-height: 56px;
    padding: var(--space-2) var(--space-4);
  }
  .m-row > a {
    flex: 1;
    min-width: 0;
    display: flex;
  }
  .m-row > a::after {
    content: '';
    position: absolute;
    inset: 0;
  }
  .m-row:has(> a:active) {
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
    color: var(--text-tertiary);
  }
  /* The dataset's size, read first: secondary ink. */
  .m-total {
    color: var(--text-secondary);
  }
  .m-side {
    position: relative;
    z-index: 1;
    display: grid;
    justify-items: end;
    gap: var(--space-1);
    flex-shrink: 0;
  }
  /* Wide enough for the strip's "Class balance" words over it (no column heading names it on phones). */
  .m-strip {
    width: 104px;
  }
  /* From 768 px the cards; phones the rows. */
  @media (min-width: 768px) {
    .phone-list {
      display: none;
    }
  }
  @media (max-width: 767px) {
    .cards {
      display: none;
    }
  }
`;
