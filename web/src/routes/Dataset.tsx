import ArrowLeft from '$icons/arrow-left';
import Copy from '$icons/copy';
import ImageOff from '$icons/image-off';
import Play from '$icons/play';
import Scan from '$icons/scan';
import Trash from '$icons/trash-2';
import TriangleAlert from '$icons/triangle-alert';
import X from '$icons/x';
import { computed, css, effect, Index, Key, MediaQuery, signal, untrack } from 'fw';
import { ApiError, hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { Split } from '$lib/api/types';
import { media } from '$lib/design/tokens';
import { ago, compact, num } from '$lib/format';
import { href, router } from '$lib/router';
import Page from '$lib/shell/Page';
import { ownCreateAction, setCrumbs } from '$lib/shell/crumbs';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import Gate from '$lib/ui/Gate';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import { DialogStatus } from '$lib/ui/feedback';
import { gridView } from '$lib/ui/gridView';
import Menu, { type MenuItem } from '$lib/ui/Menu';
import Meta from '$lib/ui/Meta';
import Panel from '$lib/ui/Panel';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import Skeleton from '$lib/ui/Skeleton';
import Tabs from '$lib/ui/Tabs';
import type { Density } from '$lib/ui/TileGrid';
import { toast } from '$lib/ui/toast';
import ToggleButton from '$lib/ui/ToggleButton';
import ClassPicker from './dataset/ClassPicker';
import DisplayMenu from './dataset/DisplayMenu';
import { findingCount, findingLabel, healthChecks, imageCount, noTraining } from './dataset/health';
import ImageGrid from './dataset/ImageGrid';
import Inspector from './dataset/Inspector';
import { ImagePager } from './dataset/pager';
import StatsTab from './dataset/StatsTab';
import { datasetsChanged } from './datasets/changes';

type Tab = 'images' | 'stats';
type SplitFilter = 'all' | Split;

const DENSITIES: { value: Density; label: string }[] = [
  { value: 's', label: 'S' },
  { value: 'm', label: 'M' },
  { value: 'l', label: 'L' },
];
const SKELETON_CELLS = Array.from({ length: 12 });

export default function Dataset() {
  // Primitive deriveds: the route object is replaced on every query change, these only change when their value does.
  const id = computed(() => router.route.params.id ?? '');
  const tab = computed<Tab>(() => (router.route.query.get('tab') === 'stats' ? 'stats' : 'images'));
  const split = computed<SplitFilter>(() => {
    const s = router.route.query.get('split');
    return s === 'train' || s === 'val' || s === 'test' ? s : 'all';
  });
  const clsParam = computed(() => router.route.query.get('class'));
  const imageParam = computed(() => router.route.query.get('image'));

  const ds = new Resource((s) => hub.api.dataset(id.get(), s));
  const d = () => ds.data;
  /** The dataset could not be read: what went wrong. */
  const failure = computed(() => (ds.error && !d() ? ds.error : undefined));
  /** Its folder is gone (moved, renamed, deleted): the dataset is still registered, so it can be removed. */
  const gone = computed(() => {
    const e = failure.get();
    return e instanceof ApiError && e.code === 'path_not_found';
  });
  /** A dataset that cannot be read, as the list has it: its name for the page, and that it is there to remove. */
  const listed = new Resource(async (s) => (failure.get() ? ((await hub.api.datasets(s)).find((x) => x.id === id.get()) ?? null) : null));
  /** The dataset the header's actions (Copy path, Remove from FOLO) act on, whether or not it could be read. */
  const target = () => d() ?? listed.data ?? undefined;

  const cls = computed(() => {
    const p = clsParam.get();
    const n = p === null ? NaN : Number(p);
    const dd = d();
    return Number.isInteger(n) && n >= 0 && (!dd || n < dd.classes.length) ? n : undefined;
  });
  const only = computed(() => {
    const c = cls.get();
    return c === undefined ? null : new Set([c]);
  });

  effect(() => {
    setCrumbs({ label: 'Datasets', path: '/datasets' }, { label: target()?.name ?? id.get() });
  });
  // The header's "Train on this dataset" is this page's create action: the top bar's New run steps aside.
  ownCreateAction();

  // History: opening an image, and the first filter on an unfiltered view, add an entry, so Back closes the image or
  // undoes the filter instead of leaving the dataset; everything else (stepping through images, changing a filter, a
  // tab) replaces the entry. A change that comes back to the view an entry was added from goes Back to it, so Close
  // and Back agree and no entry is left over. `pushedFrom` holds those views (path and query), innermost last;
  // arriving at one (Back, or that change) drops it.
  const pushedFrom: string[] = [];
  const view = (path: string, q: URLSearchParams | string) => `${path}?${q}`;
  const here = computed(() => view(router.route.path, router.route.query));
  effect(() => {
    if (pushedFrom.at(-1) === here.get()) pushedFrom.pop();
  });

  /** Changes query params (null removes one): `push` adds a history entry, anything else replaces this one. */
  function setQuery(patch: Record<string, string | null>, push = false) {
    const before = router.route.query.toString();
    const q = new URLSearchParams(router.route.query);
    for (const [k, v] of Object.entries(patch)) {
      if (v === null) q.delete(k);
      else q.set(k, v);
    }
    const s = q.toString();
    if (s === before) return;
    const at = router.route.path;
    const path = `/datasets/${encodeURIComponent(id.peek())}${s ? `?${s}` : ''}`;
    if (push) {
      pushedFrom.push(view(at, before));
      router.go(path);
    } else if (pushedFrom.at(-1) === view(at, s)) history.back();
    else router.go(path, { replace: true });
  }

  /** A filter change: the first one on an unfiltered view adds a history entry. */
  function setFilter(patch: Record<string, string | null>) {
    setQuery({ ...patch, image: null }, !filtered.peek());
  }

  // ---------- image list for the current filter ----------
  const loaded = computed(() => !!d());
  const splitStart = computed(() => {
    const dd = d();
    const sp = split.get();
    return !dd || sp === 'all' || sp === 'train' ? 0 : sp === 'val' ? dd.splits.train : dd.splits.train + dd.splits.val;
  });
  const pager = computed(() => {
    if (!loaded.get() || hub.mode === 'connecting') return null;
    void hub.epoch;
    return new ImagePager(id.get(), split.get() === 'all' ? undefined : (split.get() as Split), cls.get(), splitStart.get());
  });
  effect(() => {
    const p = pager.get();
    return () => p?.dispose();
  });

  // ---------- per-viewer view preferences (every image grid's, ui/gridView) ----------
  const density = () => gridView.density;
  const boxes = () => gridView.boxes;
  const setBoxes = (on: boolean) => (gridView.boxes = on);
  const setDensity = (v: Density) => (gridView.density = v);

  // Phones get their own header and toolbar: one meta line, an icon Train button, and a one-row image toolbar.
  const phoneQuery = new MediaQuery(media.phone, false);
  const phone = () => phoneQuery.current;

  // ---------- header actions ----------
  async function copyPath() {
    const dd = target();
    if (!dd) return;
    // page-level: the header's Copy path, outside any dialog.
    try {
      await navigator.clipboard.writeText(dd.path);
      toast.success('Path copied', dd.path);
    } catch {
      toast.error(new Error('The browser blocked clipboard access.'), 'Could not copy the path');
    }
  }

  const confirmDelete = signal(false);
  const deleting = signal(false);
  /** A failed delete says so in its dialog, which stays open for another try (the toaster sits behind the modal). */
  const deleteStatus = new DialogStatus();
  async function remove() {
    const dd = target();
    if (!dd) return;
    deleting.set(true);
    try {
      await hub.api.deleteDataset(dd.id);
      datasetsChanged();
      confirmDelete.set(false);
      toast.success('Dataset removed', `${dd.name} is no longer registered. Files on disk were not touched.`);
      router.go('/datasets');
    } catch (e) {
      deleteStatus.error(e, 'Could not delete the dataset');
    } finally {
      deleting.set(false);
    }
  }

  // ---------- derived view data ----------
  const total = computed(() => {
    const dd = d();
    return dd ? imageCount(dd) : 0;
  });
  // The check's counts by severity (servers before them send none; a failure only leaves them out).
  const check = new Resource((s) => hub.api.datasetIssues(id.get(), { limit: 0 }, s));
  const counts = computed(() => check.data?.counts);
  const findings = computed(() => {
    const dd = d();
    return dd?.stats ? findingCount(healthChecks(dd, dd.stats, counts.get())) : 0;
  });
  const filtered = computed(() => cls.get() !== undefined || split.get() !== 'all');
  /** Why Train is off: no image is in the train split (a COCO test file registered alone puts them all in test). */
  const untrainable = computed(() => {
    const dd = d();
    return dd ? noTraining(dd) : null;
  });
  const splitOptions = computed(() => {
    const dd = d();
    if (!dd) return [];
    const opts: { value: SplitFilter; label: string; meta: string }[] = [{ value: 'all', label: 'All', meta: compact(total.get()) }];
    for (const s of ['train', 'val', 'test'] as const)
      if (dd.splits[s] > 0 || split.get() === s) opts.push({ value: s, label: s[0]!.toUpperCase() + s.slice(1), meta: compact(dd.splits[s]) });
    return opts;
  });

  function openImage(_i: number, imageId: string) {
    setQuery({ image: imageId }, true);
  }

  /** The image the inspector last showed, revealed in the grid when it closes (by Close, Esc or Back). */
  const reveal = signal<{ i: number } | null>(null);
  let lastImage: string | null = null;
  effect(() => {
    const im = imageParam.get();
    if (im === null && lastImage !== null) {
      const i = untrack(() => pager.get()?.indexOf(lastImage!) ?? -1);
      if (i >= 0) reveal.set({ i });
    }
    lastImage = im;
  });

  const moreItems: MenuItem[] = [
    { label: 'Copy path', icon: Copy, onclick: copyPath },
    { label: 'Remove from FOLO', icon: Trash, danger: true, onclick: () => confirmDelete.set(true) },
  ];

  const dd = () => d()!;
  const tg = () => target()!;
  const pg = () => pager.get()!;
  /** The filtered count, "…" until the first page answers. */
  const matched = () => (pager.get()?.total != null ? num(pager.get()!.total!) : '…');

  return (
    <>
      {failure.get() ? (
        /* Nothing to frame: the message frameless on the content edge (DESIGN_DIRECTION 8), as on Run and Not found;
           the way back is the header's action, as on Model. A dataset that is still registered keeps its name and its
           ⋯ menu, so it can always be removed. */
        <Page
          title={listed.data?.name ?? 'Dataset'}
          actions={() => (
            <>
              <Button icon={ArrowLeft} href={href('/datasets')}>
                All datasets
              </Button>
              {listed.data ? <Menu label="More dataset actions" items={moreItems} /> : null}
            </>
          )}
        >
          {gone.get() ? (
            <ErrorState
              error={failure.get()}
              title="Folder missing"
              hint="It was moved, renamed or deleted. Put it back and press Retry, or remove the dataset from FOLO: its files are never deleted."
              retry={() => ds.refresh()}
              actions={
                listed.data ? (
                  <Button size="sm" icon={Trash} onclick={() => confirmDelete.set(true)}>
                    Remove from FOLO
                  </Button>
                ) : undefined
              }
            />
          ) : failure.get() instanceof ApiError && (failure.get() as ApiError).status === 404 ? (
            <ErrorState error={failure.get()} title="Dataset not found" />
          ) : (
            <ErrorState
              error={failure.get()}
              title="Could not load this dataset"
              retry={() => ds.refresh()}
              actions={
                listed.data ? (
                  <Button size="sm" icon={Trash} onclick={() => confirmDelete.set(true)}>
                    Remove from FOLO
                  </Button>
                ) : undefined
              }
            />
          )}
        </Page>
      ) : !d() ? (
        <div class="sk-page" aria-busy="true" aria-label="Loading dataset">
          <div class="sk-head">
            <Skeleton w="48px" h="18px" />
            <Skeleton w="220px" h="24px" />
            <Skeleton w="360px" h="14px" />
          </div>
          <Skeleton h="37px" />
          <div class="sk-grid">
            <Index each={SKELETON_CELLS}>{() => <Skeleton h="auto" r="var(--radius-md)" class="sk-cell" />}</Index>
          </div>
        </div>
      ) : (
        <>
          <Page
            title={dd().name}
            sub={() => (
              <>
                {phone() ? (
                  /* The top bar names the dataset; the path is one tap away in the ⋯ menu (Copy path). */
                  <span class="meta-sm num">
                    <Meta line>
                      <span class="faint">{dd().format}</span>
                      <span>{num(total.get())} images</span>
                      <span>{dd().classes.length} classes</span>
                    </Meta>
                  </span>
                ) : (
                  <Meta>
                    <span class="faint">{dd().format}</span>
                    <span class="path mono truncate" title={dd().path}>
                      {dd().path}
                    </span>
                    <span class="num">{num(dd().splits.train)} train</span>
                    <span class="num">{num(dd().splits.val)} val</span>
                    <span class="num">{num(dd().splits.test)} test</span>
                    <span class="num">{dd().classes.length} classes</span>
                    <span class="faint">added {ago(dd().created)}</span>
                  </Meta>
                )}
              </>
            )}
            actions={() => (
              <>
                {/* A dataset with no training images cannot train: the action says so instead of leading to a run that fails. */}
                <Gate reason={untrainable.get()?.reason}>
                  {(why) =>
                    phone() ? (
                      /* Phones: the page's main action keeps a word (an icon-only lime square does not say what it does). */
                      <Button
                        variant="primary"
                        icon={Play}
                        aria-label="Train on this dataset"
                        disabled={!!why()}
                        aria-describedby={why()}
                        href={href(`/runs/new?dataset=${encodeURIComponent(dd().id)}`)}
                      >
                        Train
                      </Button>
                    ) : (
                      <Button variant="primary" icon={Play} disabled={!!why()} aria-describedby={why()} href={href(`/runs/new?dataset=${encodeURIComponent(dd().id)}`)}>
                        Train on this dataset
                      </Button>
                    )
                  }
                </Gate>
                <Menu label="More dataset actions" items={moreItems} />
              </>
            )}
          >
            <div class="tabs">
              <Tabs
                label="Dataset views"
                idPrefix="ds-tab"
                tabs={[
                  { value: 'images', label: 'Images', meta: compact(total.get()) },
                  { value: 'stats', label: 'Stats', meta: findingLabel(findings.get()) },
                ]}
                value={tab.get()}
                onchange={(t) => setQuery({ tab: t === 'images' ? null : t, image: null })}
              />
            </div>

            <div id="ds-tab-panel" role="tabpanel" aria-labelledby={`ds-tab-${tab.get()}`} class="panel">
              {untrainable.get() ? (
                <p class="no-train">
                  <TriangleAlert size={14} strokeWidth={1.75} aria-hidden="true" />
                  <span>
                    No training images: {untrainable.get()!.where}. Training needs a train split, such as <span class="mono">images/train/</span> in a YOLO
                    folder or a COCO <span class="mono">instances_train.json</span>.
                  </span>
                </p>
              ) : null}
              {tab.get() === 'stats' ? (
                <>
                  {dd().stats ? (
                    <StatsTab dataset={dd()} stats={dd().stats!} counts={counts.get()} onclass={(c) => setQuery({ tab: null, class: String(c), image: null }, !filtered.peek())} />
                  ) : (
                    <Panel>
                      <EmptyState icon={ImageOff} title="No statistics" description="The server did not return statistics for this dataset." />
                    </Panel>
                  )}
                </>
              ) : (
                <>
                  {/* One toolbar: what to show on the left, how to show it on the right. Phones keep it to one row. */}
                  <div class="ds-toolbar" class:phone={phone()}>
                    {phone() ? (
                      <span class="split-pick">
                        <Select
                          label="Split"
                          size="sm"
                          options={splitOptions.get().map((o) => ({ value: o.value, label: o.value === 'all' ? o.label : `${o.label} · ${o.meta}` }))}
                          value={split.get()}
                          onchange={(s) => setFilter({ split: s === 'all' ? null : s })}
                        />
                      </span>
                    ) : (
                      <Segmented
                        label="Split"
                        size="sm"
                        options={splitOptions.get()}
                        value={split.get()}
                        onchange={(s) => setFilter({ split: s === 'all' ? null : s })}
                      />
                    )}
                    <ClassPicker
                      block={phone()}
                      classes={dd().classes}
                      counts={dd().stats?.class_counts}
                      value={cls.get()}
                      onchange={(c) => setFilter({ class: c === undefined ? null : String(c) })}
                    />
                    {/* Phones, filtered: the count and Clear on a row of their own under the pickers ("1,873 of 2,400"). */}
                    {phone() && filtered.get() ? <span class="break" aria-hidden="true"></span> : null}
                    {/* Phones show no count until filtered; an empty span here would still take a gap after Display, 8 px
                        short of the content edge. The polite region stays, out of the row. */}
                    <span class="count num" class:none={phone() && !filtered.get()} aria-live="polite">
                      {phone() ? (
                        <>
                          {filtered.get() ? (
                            <>
                              {matched()} of {num(total.get())}
                              <span class="sr-only"> images</span>
                            </>
                          ) : null}
                        </>
                      ) : filtered.get() ? (
                        <>
                          {matched()} of {num(total.get())} images
                        </>
                      ) : (
                        <>{num(total.get())} images</>
                      )}
                    </span>
                    {filtered.get() ? (
                      <button type="button" class="clear" aria-label="Clear filters" onclick={() => setFilter({ split: null, class: null })}>
                        <X size={12} strokeWidth={2} aria-hidden="true" />
                        Clear
                      </button>
                    ) : null}
                    <span class="spacer"></span>
                    {phone() ? (
                      <DisplayMenu boxes={boxes()} density={density()} onboxes={setBoxes} ondensity={setDensity} />
                    ) : (
                      <>
                        <ToggleButton pressed={boxes()} label="Boxes" icon={Scan} onchange={setBoxes} />
                        <Segmented label="Thumbnail size" size="sm" options={DENSITIES} value={density()} onchange={setDensity} />
                      </>
                    )}
                  </div>

                  {pager.get()?.error && pager.get()!.total !== null ? (
                    /* Later pages failed: keep what loaded and offer a retry for the rest. */
                    <ErrorState error={pg().error} title="Some images could not be loaded" retry={() => pager.get()?.retry()} compact />
                  ) : null}
                  {pager.get()?.error && pager.get()!.total === null ? (
                    <Panel>
                      <ErrorState error={pg().error} title="Could not load images" retry={() => pager.get()?.retry()} />
                    </Panel>
                  ) : pager.get() && pager.get()!.total === 0 ? (
                    <Panel>
                      <EmptyState
                        icon={ImageOff}
                        title="No images match"
                        description={
                          cls.get() !== undefined
                            ? `No ${split.get() === 'all' ? '' : `${split.get()} `}image contains a ${dd().classes[cls.get()!]}.`
                            : `The ${split.get()} split is empty.`
                        }
                        actions={() => <Button onclick={() => setFilter({ split: null, class: null })}>Clear filters</Button>}
                      />
                    </Panel>
                  ) : pager.get() ? (
                    <Key when={pager.get()}>
                      <ImageGrid pager={pg()} classes={dd().classes} density={density()} boxes={boxes()} only={only.get()} reveal={reveal.get()} onopen={openImage} />
                    </Key>
                  ) : null}
                </>
              )}
            </div>
          </Page>

          {pager.get() ? (
            <Inspector
              pager={pg()}
              classes={dd().classes}
              imageId={imageParam.get()}
              only={only.get()}
              onnavigate={(im) => setQuery({ image: im })}
              onclose={() => setQuery({ image: null })}
              onmissing={(im) => {
                // page-level: a linked image that is not in this view; the inspector closes on the next line.
                toast.info('Image not in this view', `${im} does not match the current filters.`);
                setQuery({ image: null });
              }}
            />
          ) : null}

        </>
      )}
      {target() ? (
        <Dialog
          bind:open={confirmDelete}
          title={`Remove ${tg().name} from FOLO?`}
          size="sm"
          status={deleteStatus}
          footer={() => (
            <>
              <Button onclick={() => confirmDelete.set(false)}>Cancel</Button>
              <Button variant="destructive" icon={Trash} loading={deleting.get()} onclick={remove}>
                Remove from FOLO
              </Button>
            </>
          )}
        >
          <p class="confirm">
            <b>{tg().name}</b> will be unregistered from FOLO. The files in <span class="mono">{tg().path}</span> stay on disk, and runs that used it keep their results.
          </p>
        </Dialog>
      ) : null}
    </>
  );
}

css`
  .path {
    max-width: 360px;
  }
  .tabs {
    margin-bottom: calc(var(--space-1) * -1);
  }
  .panel {
    display: grid;
    gap: var(--space-3);
    min-width: 0;
  }
  .ds-toolbar {
    position: sticky;
    top: var(--size-topbar);
    z-index: calc(var(--z-sticky) - 1);
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-2) var(--space-3);
    margin: calc(var(--space-2) * -1) calc(var(--space-6) * -1) 0;
    padding: var(--space-2) var(--space-6);
    background: color-mix(in srgb, var(--bg-app) 92%, transparent);
    backdrop-filter: blur(8px);
  }
  .count {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .clear {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    height: var(--control-sm);
    margin-left: calc(var(--space-2) * -1);
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
    white-space: nowrap;
  }
  .clear:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }
  .spacer {
    flex: 1;
  }
  /* Phones: split, class and Display share one row; the class picker takes what the other two leave (its name
     truncates first), so Display ends on the 16 px gutter whatever the class reads ("All classes" left it 14 px short).
     Filtered, the count and Clear follow on a second row, so neither is squeezed to a number and a bare ×. */
  .ds-toolbar.phone {
    flex-wrap: wrap;
    gap: var(--space-2);
  }
  .ds-toolbar.phone .break {
    order: 1;
    flex-basis: 100%;
    height: 0;
  }
  .ds-toolbar.phone .count,
  .ds-toolbar.phone .clear {
    order: 2;
  }
  .ds-toolbar.phone .split-pick {
    flex-shrink: 0;
  }
  .ds-toolbar.phone :global(.picker) {
    flex: 1 1 0;
  }
  .ds-toolbar.phone .spacer {
    display: none;
  }
  .ds-toolbar.phone .count {
    flex-shrink: 0;
  }
  .ds-toolbar.phone .count.none {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip-path: inset(50%);
  }
  @media (pointer: coarse) {
    .clear {
      height: 44px;
    }
  }
  .no-train {
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .no-train :global(svg) {
    flex-shrink: 0;
    margin-top: 2px;
    color: var(--warning-text);
  }
  .confirm {
    color: var(--text-secondary);
  }
  .confirm b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .confirm .mono {
    color: var(--text-primary);
    overflow-wrap: anywhere;
  }
  .sk-page {
    max-width: calc(var(--size-content-max) + 2 * var(--space-6));
    margin: 0 auto;
    padding: var(--space-6);
    display: grid;
    gap: var(--space-5);
  }
  .sk-head {
    display: grid;
    gap: var(--space-2);
  }
  .sk-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(172px, 1fr));
    gap: var(--space-2);
  }
  .sk-grid :global(.sk-cell) {
    aspect-ratio: 4 / 3;
  }
  @media (max-width: 767px) {
    .ds-toolbar {
      margin-inline: calc(var(--space-4) * -1);
      padding-inline: var(--space-4);
    }
    .meta-sm {
      display: flex;
      align-items: center;
      min-width: 0;
      min-height: var(--control-md);
    }
    .sk-page {
      padding: var(--space-4);
    }
    .sk-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
`;
