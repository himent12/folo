import FolderOpen from '$icons/folder-open';
import Images from '$icons/images';
import Plus from '$icons/plus';
import Search from '$icons/search';
import Sparkles from '$icons/sparkles';
import Tags from '$icons/tags';
import { computed, css, effect, For, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import ClassStrip from '$lib/detect/ClassStrip';
import { ago, count } from '$lib/format';
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
import Skeleton from '$lib/ui/Skeleton';
import Gate from '$lib/ui/Gate';
import { isOffline, OFFLINE_REASON } from '$lib/ui/offline';
import Meta from '$lib/ui/Meta';
import { balanced } from '$lib/ui/balancedGrid';
import AddDatasetDialog from './datasets/AddDatasetDialog';
import DatasetCard, { cardFacts, DatasetCardSkeleton } from './datasets/DatasetCard';
import { imageCount } from './dataset/health';
import { datasetChanges, generationEnded, landed, pendingGeneration } from './datasets/changes';
import { datasetProblem, PROBLEM_LABEL, readDataset, type ListedDataset } from './datasets/problem';

type Format = 'all' | 'yolo' | 'coco';
/** A card's narrowest width (design/UNIFIED.md §4: at least 300). */
const CARD_MIN = 300;
/** The gap between cards. */
const CARD_GAP = 16;
/** How often the list reads itself again while a sample set this tab asked for still generates. */
const GENERATION_POLL_MS = 3000;

export default function Datasets() {
  setCrumbs({ label: 'Datasets' });
  // The header's create action is the one in the top-right corner (the top bar's New run steps aside).
  ownCreateAction();

  // The list endpoint has no stats; fetch each dataset for its class balance and box count (and to learn which ones
  // cannot be read: a folder that was moved is marked, not shown as if nothing were wrong).
  // A dataset whose folder was found missing is not read again on every visit (./datasets/problem readDataset): only
  // when the list may have changed (`again`, below).
  let again = false;
  const datasets = new Resource(async (s) => {
    const api = hub.api;
    const fresh = again;
    again = false;
    const list = await api.datasets(s);
    return Promise.all(
      list.map((d): Promise<ListedDataset> => readDataset(api, d.id, () => api.dataset(d.id, s), fresh).catch((e) => ({ ...d, problem: datasetProblem(e) }))),
    );
  });
  const reread = () => {
    again = true;
    void datasets.refresh();
  };
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
    const list = datasets.data;
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
      if (document.visibilityState !== 'visible' || datasets.loading || datasets.refreshing) return later();
      void datasets.refresh().finally(later);
    };
    later();
    return () => {
      live = false;
      clearTimeout(t);
    };
  });
  effect(() => {
    const back = () => document.visibilityState === 'visible' && !datasets.loading && reread();
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
  function add(m: 'folder' | 'synthetic') {
    addMode.set(m);
    adding.set(true);
  }
  effect(() => {
    const p = addParam.get();
    if (asked(p)) untrack(() => add(p));
  });
  effect(() => {
    if (adding.get()) return;
    const q = untrack(() => router.route.query);
    if (untrack(() => router.route.name) !== 'datasets' || !q.has('add')) return;
    const rest = new URLSearchParams(q);
    rest.delete('add');
    untrack(() => router.go(`/datasets${rest.toString() ? `?${rest}` : ''}`, { replace: true }));
  });

  const query = signal('');
  // The toolbar's filter, at its left as on Runs and Models (one shape for the three list toolbars): by format.
  const format = signal<Format>('all');
  const formatCount = (f: Format) => (datasets.data ?? []).filter((d) => f === 'all' || d.format === f).length;
  const shown = computed(() => {
    const q = query.get().trim().toLowerCase();
    const f = format.get();
    return (datasets.data ?? []).filter((d) => (f === 'all' || d.format === f) && (!q || `${d.name} ${d.path} ${d.format} ${d.id}`.toLowerCase().includes(q)));
  });
  // The header's facts line (design/UNIFIED.md §2, Page header): what the page holds, "5 datasets · 9,400 images ·
  // 64,086 boxes", whatever the filter; a narrowed list says what it shows under it.
  const facts = computed(() => {
    const list = datasets.data;
    if (!list?.length) return null;
    const boxes = list.reduce((a, d) => a + (d.stats?.boxes ?? 0), 0);
    return [count(list.length, 'dataset'), count(list.reduce((a, d) => a + imageCount(d), 0), 'image'), boxes ? count(boxes, 'box') : ''].filter(Boolean).join(' · ');
  });

  // Nothing answers: adding a dataset needs the server, so the action is disabled and says why (as on Runs).
  const offline = computed(() => !datasets.data && isOffline(datasets.error));
  /** Datasets to filter, or on their way: the toolbar is live (else it stays in place, disabled). */
  const hasData = computed(() => (datasets.data ? datasets.data.length > 0 : !datasets.error));
  /** Cards to show (or their skeleton); with none a panel holds the state alone, where the grid would be. */
  const hasRows = computed(() => (datasets.data ? shown.get().length > 0 : !datasets.error));

  // Labeling images (agent labeling) lives under Datasets: a secondary action beside Add dataset (the ⋯ on phones).
  return (
    <>
      <Page
        title="Datasets"
        more={[{ label: 'Label images', icon: Tags, href: href('/datasets/label') }]}
        moreLabel="More dataset actions"
        // What the page holds (what a dataset is, YOLO or COCO folders, is the empty state's).
        sub={facts.get() ? () => <span class="num">{facts.get()}</span> : undefined}
        // Phones: one header row, this line beside Add dataset.
        subPhone={facts.get() ? () => <span class="num">{facts.get()}</span> : undefined}
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
                { value: 'all', label: 'All', meta: datasets.data?.length ? String(formatCount('all')) : undefined },
                { value: 'yolo', label: 'YOLO', meta: datasets.data?.length ? String(formatCount('yolo')) : undefined },
                { value: 'coco', label: 'COCO', meta: datasets.data?.length ? String(formatCount('coco')) : undefined },
              ]}
            />
          </div>{' '}
          <div class="lt-search">
            <Input bind:value={query} icon={Search} placeholder="Filter by name or path" aria-label="Filter datasets" type="search" disabled={!hasData.get()} />
          </div>
        </div>{' '}
        {hasRows.get() ? (
          <>
            {/* The cards (design/UNIFIED.md §4, Datasets), as many columns of at least 300 px as fit, never one card
                alone on the last row (ui/balancedGrid). */}
            <ul class="cards" aria-label="Datasets" aria-busy={!datasets.data} use:balanced={{ n: datasets.data ? shown.get().length : 3, min: CARD_MIN, gap: CARD_GAP }}>
              {datasets.data ? (
                <For each={shown.get()} key={(d) => d.id}>
                  {(d) => <DatasetCard dataset={d()} />}
                </For>
              ) : (
                <For each={[0, 1, 2]} key={(i) => i}>
                  {() => <DatasetCardSkeleton />}
                </For>
              )}
            </ul>{' '}
            {/* Phones: the same parts as rows (DESIGN_DIRECTION §12, one list concept): the name, its images and
                classes, the format tag, the class balance on the right. */}
            <div class="phone-list">
              <Panel pad="none">
              {datasets.data ? (
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
          </>
        ) : (
          // No cards (an error, no datasets, none that match): one panel holds the state where the grid would be.
          <Panel pad="none">
            <div class="state" aria-label="Datasets">
              {datasets.error && !datasets.data ? (
                <ErrorState error={datasets.error} title="Could not load datasets" retry={() => datasets.refresh()} />
              ) : datasets.data?.length === 0 ? (
                // The header already holds the primary action: offer the quick start.
                <EmptyState
                  icon={Images}
                  title="No datasets yet"
                  description="Register a folder of images you have labeled (YOLO or COCO), try FOLO in a minute with a generated set of shapes, or have an agent label your images."
                  actions={() => (
                    <>
                      <Button icon={FolderOpen} onclick={() => add('folder')}>
                        Register folder
                      </Button>
                      <Button icon={Sparkles} onclick={() => add('synthetic')}>
                        Generate sample set
                      </Button>
                      <Button icon={Tags} href={href('/datasets/label')}>
                        Label images
                      </Button>
                    </>
                  )}
                />
              ) : (
                <EmptyState
                  icon={Search}
                  title="No datasets match"
                  description="Try another format or search term."
                  actions={() => (
                    <Button
                      size="sm"
                      onclick={() => {
                        query.set('');
                        format.set('all');
                      }}
                    >
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
        {datasets.data && ((shown.get().length && shown.get().length < datasets.data.length) || waiting.get()) ? (
          <p class="list-foot num">
            <Meta>
              {shown.get().length && shown.get().length < datasets.data.length ? (
                <span>
                  {shown.get().length} of {count(datasets.data.length, 'dataset')}
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
      </Page>{' '}
      <AddDatasetDialog bind:open={adding} bind:mode={addMode} known={datasets.data} />
    </>
  );
}

css`
  .cards {
    display: grid;
    grid-template-columns: repeat(var(--cols, 1), minmax(0, 1fr));
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
