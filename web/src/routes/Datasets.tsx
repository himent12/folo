import Images from '$icons/images';
import Plus from '$icons/plus';
import Search from '$icons/search';
import Sparkles from '$icons/sparkles';
import Tags from '$icons/tags';
import { computed, css, effect, For, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import ClassStrip from '$lib/detect/ClassStrip';
import { ago, count, num } from '$lib/format';
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
import TableState from '$lib/ui/TableState';
import Gate from '$lib/ui/Gate';
import { isOffline, OFFLINE_REASON } from '$lib/ui/offline';
import Meta from '$lib/ui/Meta';
import { rowNav } from '$lib/ui/rowNav';
import AddDatasetDialog from './datasets/AddDatasetDialog';
import { imageCount } from './dataset/health';
import { datasetChanges } from './datasets/changes';
import { datasetProblem, PROBLEM_LABEL, readDataset, type ListedDataset } from './datasets/problem';

type Format = 'all' | 'yolo' | 'coco';
/** Columns of the desktop table (for state rows that span it), the empty last one included. */
const COLS = 10;

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
  // What the rows shown hold, said once at the list's end (base.css .list-foot).
  const totals = computed(() => {
    const list = shown.get();
    return {
      images: list.reduce((a, d) => a + imageCount(d), 0),
      boxes: list.reduce((a, d) => a + (d.stats?.boxes ?? 0), 0),
    };
  });

  // Nothing answers: adding a dataset needs the server, so the action is disabled and says why (as on Runs).
  const offline = computed(() => !datasets.data && isOffline(datasets.error));
  /** Datasets to filter, or on their way: the toolbar is live (else it stays in place, disabled). */
  const hasData = computed(() => (datasets.data ? datasets.data.length > 0 : !datasets.error));
  /** Rows to head (or their skeleton): the column heads show; with none the frame holds the state alone. */
  const hasRows = computed(() => (datasets.data ? shown.get().length > 0 : !datasets.error));

  // Labeling images (agent labeling) lives under Datasets: a secondary action beside Add dataset (the ⋯ on phones).
  return (
    <>
      <Page
        title="Datasets"
        more={[{ label: 'Label images', icon: Tags, href: href('/datasets/label') }]}
        moreLabel="More dataset actions"
        // One fixed line: the counts live at the list's end, so nothing here changes as the list loads.
        sub={() => <span>YOLO and COCO folders registered with the server.</span>}
        // Phones: one header row, this line beside Add dataset.
        subPhone={() => <span>YOLO and COCO folders.</span>}
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
        {/* Always here, so the table never moves between states (DESIGN_DIRECTION §2.8): over an empty or failed list it
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
        <Panel pad="none">
          {/* With no rows (empty, no match, an error) the frame drops its column heads and holds the state alone. */}
          <div class="scroll" class:state={!hasRows.get()}>
            <table class="table ds" aria-busy={!datasets.data} aria-label="Datasets" use:rowNav>
              {hasRows.get() ? (
                <thead>
                  <tr>
                    {/* One geometry in every state (DESIGN_DIRECTION 7): fixed widths on the header row that loading and
                        loaded share. A few rows across a wide panel take bounded widths (the path 280 px, cut with
                        its full path in the title) and leave the slack to an empty last column, as Model's per class does;
                        below 1280, where the path and that column hide, the class balance takes what is left, at least
                        160 px (minmax(160px, 1fr)). */}
                    <th scope="col" style:width="160px">
                      Dataset
                    </th>
                    <th scope="col" class="hide-md" style:width="80px">
                      Format
                    </th>
                    <th scope="col" class="hide-lg" style:width="280px">
                      Path
                    </th>
                    <th scope="col" class="r" style:width="80px">
                      Train
                    </th>
                    <th scope="col" class="r" style:width="72px">
                      Val
                    </th>
                    <th scope="col" class="r" style:width="72px">
                      Test
                    </th>
                    <th scope="col" class="bal">
                      <span class="bal-h">Class balance</span>
                    </th>
                    <th scope="col" class="r hide-md" style:width="88px">
                      Boxes
                    </th>
                    <th scope="col" class="r hide-md" style:width="104px">
                      Created
                    </th>
                    <th class="hide-lg rest" aria-hidden="true"></th>
                  </tr>
                </thead>
              ) : null}
              <tbody>
                {datasets.error && !datasets.data ? (
                  <TableState cols={COLS}>
                    <ErrorState error={datasets.error} title="Could not load datasets" retry={() => datasets.refresh()} />
                  </TableState>
                ) : datasets.data?.length === 0 ? (
                  // The header already holds the primary action (and the subtitle says what a dataset is): offer the quick start.
                  <TableState cols={COLS}>
                    <EmptyState
                      icon={Images}
                      title="No datasets yet"
                      description="Try FOLO in a minute with a generated set of shapes, or have an agent label your own images."
                      actions={() => (
                        <>
                          <Button icon={Sparkles} onclick={() => add('synthetic')}>
                            Generate sample set
                          </Button>
                          <Button icon={Tags} href={href('/datasets/label')}>
                            Label images
                          </Button>
                        </>
                      )}
                    />
                  </TableState>
                ) : datasets.data && shown.get().length === 0 ? (
                  <TableState cols={COLS}>
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
                  </TableState>
                ) : !datasets.data ? (
                  <For each={[0, 1, 2]} key={(i) => i}>
                    {() => (
                      <tr aria-hidden="true">
                        <td>
                          <Skeleton w="96px" h="12px" />
                        </td>
                        <td class="hide-md">
                          <Skeleton w="32px" h="12px" />
                        </td>
                        <td class="hide-lg">
                          <Skeleton w="120px" h="12px" />
                        </td>
                        <td class="r">
                          <Skeleton w="36px" h="12px" />
                        </td>
                        <td class="r">
                          <Skeleton w="28px" h="12px" />
                        </td>
                        <td class="r">
                          <Skeleton w="28px" h="12px" />
                        </td>
                        <td>
                          <Skeleton h="6px" r="999px" />
                        </td>
                        <td class="r hide-md">
                          <Skeleton w="36px" h="12px" />
                        </td>
                        <td class="r hide-md">
                          <Skeleton w="44px" h="12px" />
                        </td>
                        <td class="hide-lg rest"></td>
                      </tr>
                    )}
                  </For>
                ) : (
                  <For each={shown.get()} key={(d) => d.id}>
                    {(d) => (
                      <tr class="link" data-row aria-label={d().name} onclick={() => router.go(`/datasets/${d().id}`)}>
                        <td>
                          <a class="name" href={href(`/datasets/${d().id}`)} onclick={(e) => e.stopPropagation()}>
                            {d().name}
                          </a>
                        </td>
                        <td class="hide-md faint">{d().format}</td>
                        <td class="hide-lg path-cell">
                          <span class="path mono truncate" title={d().path}>
                            {d().path}
                          </span>
                        </td>
                        <td class="r num muted">{num(d().splits.train)}</td>
                        <td class="r num muted">{num(d().splits.val)}</td>
                        <td class="r num muted">{num(d().splits.test)}</td>
                        <td>
                          {d().problem ? (
                            <Badge tone="warning">{PROBLEM_LABEL[d().problem!]}</Badge>
                          ) : d().stats ? (
                            <ClassStrip counts={d().stats!.class_counts} classes={d().classes} />
                          ) : (
                            <span class="faint">—</span>
                          )}
                        </td>
                        {/* Full numbers, like the split columns beside it: one format per column. */}
                        <td class="r num hide-md">{num(d().stats?.boxes)}</td>
                        <td class="r faint hide-md">{ago(d().created)}</td>
                        <td class="hide-lg rest" aria-hidden="true"></td>
                      </tr>
                    )}
                  </For>
                )}
              </tbody>
            </table>
          </div>{' '}
          {/* Phones: the name, then the image total and the class count, then the splits (each line wraps whole items, so
              no count is cut); the class balance on the right. */}
          {shown.get().length ? (
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
                            <span>{count(imageCount(d()), 'image')}</span>
                            <span>{count(d().classes.length, 'class')}</span>
                          </Meta>
                        </span>{' '}
                        <span class="m-meta num">
                          <Meta>
                            <span>{num(d().splits.train)} train</span>
                            <span>{num(d().splits.val)} val</span>
                            <span>{num(d().splits.test)} test</span>
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
          ) : !datasets.data && !datasets.error ? (
            // Phones, loading: the rows' shape (the name over two meta lines, the class balance on the right).
            <ul class="rows" aria-hidden="true">
              <For each={[0, 1, 2]} key={(i) => i}>
                {() => (
                  <li>
                    <span class="row-sk">
                      <span class="m-main">
                        <Skeleton w="55%" h="14px" />
                        <Skeleton w="45%" h="10px" />
                        <Skeleton w="65%" h="10px" />
                      </span>{' '}
                      <span class="m-side">
                        <Skeleton w="104px" h="4px" />
                      </span>
                    </span>
                  </li>
                )}
              </For>
            </ul>
          ) : null}
        </Panel>{' '}
        {datasets.data && shown.get().length ? (
          <p class="list-foot num">
            <Meta>
              <span>
                {shown.get().length < datasets.data.length ? `${shown.get().length} of ${datasets.data.length}` : shown.get().length}{' '}
                {datasets.data.length === 1 ? 'dataset' : 'datasets'}
              </span>{' '}
              <span>{count(totals.get().images, 'image')}</span>{' '}
              <span>{count(totals.get().boxes, 'box')}</span>
            </Meta>
          </p>
        ) : null}
      </Page>{' '}
      <AddDatasetDialog bind:open={adding} bind:mode={addMode} known={datasets.data} />
    </>
  );
}

css`
  .scroll {
    overflow-x: auto;
  }
  /* The path in its bounded column, cut with an ellipsis (the full path is in the title). */
  .path-cell {
    max-width: 0;
  }
  /* From 1280 the class balance is bounded too (the empty last column takes the slack); below it, where that column
     hides, it is the flexible one. */
  @media (min-width: 1280px) {
    .bal {
      width: 184px;
    }
  }
  .rest {
    padding: 0;
  }
  .path {
    display: block;
    color: var(--text-tertiary);
  }
  /* The class balance column: never under 160 px (the header holds it open in every state), the strip filling it. */
  .bal-h {
    display: block;
    min-width: 160px;
  }

  .rows {
    display: none;
  }
  @media (max-width: 767px) {
    /* Phones show the rows below; the table frame stays only to hold an empty or error state (no column heads at any width). */
    .scroll:not(.state) {
      display: none;
    }
    .rows .name {
      font-weight: var(--weight-medium);
    }
    .rows {
      display: block;
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
    /* The dataset's size, read first: secondary ink over the tertiary splits. */
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
  }
`;
