<script lang="ts">
  import { MediaQuery } from 'svelte/reactivity';
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import Copy from '@lucide/svelte/icons/copy';
  import ImageOff from '@lucide/svelte/icons/image-off';
  import Play from '@lucide/svelte/icons/play';
  import Scan from '@lucide/svelte/icons/scan';
  import Trash from '@lucide/svelte/icons/trash-2';
  import X from '@lucide/svelte/icons/x';
  import { ApiError, hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Split } from '$lib/api/types';
  import { ago, compact, num } from '$lib/format';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { ownCreateAction, setCrumbs } from '$lib/shell/crumbs.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import { DialogStatus } from '$lib/ui/feedback.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import Menu, { type MenuItem } from '$lib/ui/Menu.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import Tabs from '$lib/ui/Tabs.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import Select from '$lib/ui/Select.svelte';
  import ClassPicker from './dataset/ClassPicker.svelte';
  import DisplayMenu from './dataset/DisplayMenu.svelte';
  import { findingCount, findingLabel, healthChecks, imageCount } from './dataset/health';
  import ImageGrid, { type Density } from './dataset/ImageGrid.svelte';
  import Inspector from './dataset/Inspector.svelte';
  import { ImagePager } from './dataset/pager.svelte';
  import StatsTab from './dataset/StatsTab.svelte';
  import { media } from '$lib/design/tokens';
  import Meta from '$lib/ui/Meta.svelte';

  type Tab = 'images' | 'stats';
  type SplitFilter = 'all' | Split;

  // Primitive deriveds: the route object is replaced on every query change, these only change when their value does.
  const id = $derived(router.route.params.id ?? '');
  const tab = $derived<Tab>(router.route.query.get('tab') === 'stats' ? 'stats' : 'images');
  const split = $derived.by<SplitFilter>(() => {
    const s = router.route.query.get('split');
    return s === 'train' || s === 'val' || s === 'test' ? s : 'all';
  });
  const clsParam = $derived(router.route.query.get('class'));
  const imageParam = $derived(router.route.query.get('image'));

  const ds = new Resource((s) => hub.api.dataset(id, s));
  const d = $derived(ds.data);

  const cls = $derived.by(() => {
    const n = clsParam === null ? NaN : Number(clsParam);
    return Number.isInteger(n) && n >= 0 && (!d || n < d.classes.length) ? n : undefined;
  });
  const only = $derived(cls === undefined ? null : new Set([cls]));

  $effect(() => {
    setCrumbs({ label: 'Datasets', path: '/datasets' }, { label: d?.name ?? id });
  });
  // The header's "Train on this dataset" is this page's create action: the top bar's New run steps aside.
  ownCreateAction();

  /** Replaces query params (null removes one) without adding history entries. */
  function setQuery(patch: Record<string, string | null>) {
    const q = new URLSearchParams(router.route.query);
    for (const [k, v] of Object.entries(patch)) {
      if (v === null) q.delete(k);
      else q.set(k, v);
    }
    const s = q.toString();
    router.go(`/datasets/${encodeURIComponent(id)}${s ? `?${s}` : ''}`, { replace: true });
  }

  // ---------- image list for the current filter ----------
  const loaded = $derived(!!d);
  const splitStart = $derived(
    !d || split === 'all' || split === 'train' ? 0 : split === 'val' ? d.splits.train : d.splits.train + d.splits.val,
  );
  const pager = $derived.by(() => {
    if (!loaded || hub.mode === 'connecting') return null;
    void hub.epoch;
    return new ImagePager(id, split === 'all' ? undefined : split, cls, splitStart);
  });
  $effect(() => {
    const p = pager;
    return () => p?.dispose();
  });

  // ---------- per-viewer view preferences ----------
  function readPref<T extends string>(key: string, ok: readonly T[], fallback: T): T {
    try {
      const v = localStorage.getItem(key);
      return v !== null && (ok as readonly string[]).includes(v) ? (v as T) : fallback;
    } catch {
      return fallback;
    }
  }
  function writePref(key: string, v: string) {
    try {
      localStorage.setItem(key, v);
    } catch {
      /* private mode: not remembered */
    }
  }
  let density = $state<Density>(readPref('folo.datasets.density', ['s', 'm', 'l'] as const, 'm'));
  let boxes = $state(readPref('folo.datasets.boxes', ['on', 'off'] as const, 'on') === 'on');
  function setBoxes(on: boolean) {
    boxes = on;
    writePref('folo.datasets.boxes', on ? 'on' : 'off');
  }
  function setDensity(v: Density) {
    density = v;
    writePref('folo.datasets.density', v);
  }

  // Phones get their own header and toolbar: one meta line, an icon Train button, and a one-row image toolbar.
  const phoneQuery = new MediaQuery(media.phone, false);
  const phone = $derived(phoneQuery.current);

  // ---------- header actions ----------
  async function copyPath() {
    if (!d) return;
    // page-level: the header's Copy path, outside any dialog.
    try {
      await navigator.clipboard.writeText(d.path);
      toast.success('Path copied', d.path);
    } catch {
      toast.error(new Error('The browser blocked clipboard access.'), 'Could not copy the path');
    }
  }

  let confirmDelete = $state(false);
  let deleting = $state(false);
  /** A failed delete says so in its dialog, which stays open for another try (the toaster sits behind the modal). */
  const deleteStatus = new DialogStatus();
  async function remove() {
    if (!d) return;
    deleting = true;
    try {
      await hub.api.deleteDataset(d.id);
      confirmDelete = false;
      toast.success('Dataset removed', `${d.name} is no longer registered. Files on disk were not touched.`);
      router.go('/datasets');
    } catch (e) {
      deleteStatus.error(e, 'Could not delete the dataset');
    } finally {
      deleting = false;
    }
  }

  // ---------- derived view data ----------
  const total = $derived(d ? imageCount(d) : 0);
  // The check's counts by severity (servers before them send none; a failure only leaves them out).
  const check = new Resource((s) => hub.api.datasetIssues(id, { limit: 0 }, s));
  const counts = $derived(check.data?.counts);
  const findings = $derived(d?.stats ? findingCount(healthChecks(d, d.stats, counts)) : 0);
  const filtered = $derived(cls !== undefined || split !== 'all');
  const splitOptions = $derived.by(() => {
    if (!d) return [];
    const opts: { value: SplitFilter; label: string; meta: string }[] = [{ value: 'all', label: 'All', meta: compact(total) }];
    for (const s of ['train', 'val', 'test'] as const)
      if (d.splits[s] > 0 || split === s) opts.push({ value: s, label: s[0]!.toUpperCase() + s.slice(1), meta: compact(d.splits[s]) });
    return opts;
  });

  function openImage(_i: number, imageId: string) {
    setQuery({ image: imageId });
  }

  const moreItems: MenuItem[] = [
    { label: 'Copy path', icon: Copy, onclick: copyPath },
    { label: 'Delete dataset', icon: Trash, danger: true, onclick: () => (confirmDelete = true) },
  ];
</script>

{#if ds.error && !d}
  <!-- Nothing to frame: the message frameless on the content edge (DESIGN_DIRECTION 8), as on Run and Not found; the
       way back is the header's action, as on Model. -->
  <Page title="Dataset">
    {#snippet actions()}<Button icon={ArrowLeft} href={href('/datasets')}>All datasets</Button>{/snippet}
    {#if ds.error instanceof ApiError && ds.error.status === 404}
      <ErrorState error={ds.error} title="Dataset not found" />
    {:else}
      <ErrorState error={ds.error} title="Could not load this dataset" retry={() => ds.refresh()} />
    {/if}
  </Page>
{:else if !d}
  <div class="sk-page" aria-busy="true" aria-label="Loading dataset">
    <div class="sk-head">
      <Skeleton w="48px" h="18px" />
      <Skeleton w="220px" h="24px" />
      <Skeleton w="360px" h="14px" />
    </div>
    <Skeleton h="37px" />
    <div class="sk-grid">{#each { length: 12 }, i (i)}<Skeleton h="auto" r="var(--radius-md)" class="sk-cell" />{/each}</div>
  </div>
{:else}
  <Page title={d.name}>
    {#snippet sub()}
      {#if phone}
        <!-- The top bar names the dataset; the path is one tap away in the ⋯ menu (Copy path). -->
        <span class="meta-sm num"><Meta line><span class="faint">{d.format}</span><span>{num(total)} images</span><span>{d.classes.length} classes</span></Meta></span>
      {:else}
        <Meta>
          <span class="faint">{d.format}</span>
          <span class="path mono truncate" title={d.path}>{d.path}</span>
          <span class="num">{num(d.splits.train)} train</span>
          <span class="num">{num(d.splits.val)} val</span>
          <span class="num">{num(d.splits.test)} test</span>
          <span class="num">{d.classes.length} classes</span>
          <span class="faint">added {ago(d.created)}</span>
        </Meta>
      {/if}
    {/snippet}
    {#snippet actions()}
      {#if phone}
        <!-- Phones: the page's main action keeps a word (an icon-only lime square does not say what it does). -->
        <Button variant="primary" icon={Play} aria-label="Train on this dataset" href={href(`/runs/new?dataset=${encodeURIComponent(d.id)}`)}>Train</Button>
      {:else}
        <Button variant="primary" icon={Play} href={href(`/runs/new?dataset=${encodeURIComponent(d.id)}`)}>Train on this dataset</Button>
      {/if}
      <Menu label="More dataset actions" items={moreItems} />
    {/snippet}

    <div class="tabs">
      <Tabs
        label="Dataset views"
        idPrefix="ds-tab"
        tabs={[
          { value: 'images', label: 'Images', meta: compact(total) },
          { value: 'stats', label: 'Stats', meta: findingLabel(findings) },
        ]}
        value={tab}
        onchange={(t) => setQuery({ tab: t === 'images' ? null : t, image: null })}
      />
    </div>

    <div id="ds-tab-panel" role="tabpanel" aria-labelledby="ds-tab-{tab}" class="panel">
      {#if tab === 'stats'}
        {#if d.stats}
          <StatsTab dataset={d} stats={d.stats} {counts} onclass={(c) => setQuery({ tab: null, class: String(c), image: null })} />
        {:else}
          <Panel><EmptyState icon={ImageOff} title="No statistics" description="The server did not return statistics for this dataset." /></Panel>
        {/if}
      {:else}
        <!-- One toolbar: what to show on the left, how to show it on the right. Phones keep it to one row. -->
        <div class="ds-toolbar" class:phone>
          {#if phone}
            <span class="split-pick">
              <Select
                label="Split"
                size="sm"
                options={splitOptions.map((o) => ({ value: o.value, label: o.value === 'all' ? o.label : `${o.label} · ${o.meta}` }))}
                value={split}
                onchange={(s) => setQuery({ split: s === 'all' ? null : s, image: null })}
              />
            </span>
          {:else}
            <Segmented
              label="Split"
              size="sm"
              options={splitOptions}
              value={split}
              onchange={(s) => setQuery({ split: s === 'all' ? null : s, image: null })}
            />
          {/if}
          <ClassPicker classes={d.classes} counts={d.stats?.class_counts} value={cls} onchange={(c) => setQuery({ class: c === undefined ? null : String(c), image: null })} />
          <!-- Phones, filtered: the count and Clear on a row of their own under the pickers ("1,873 of 2,400"). -->
          {#if phone && filtered}<span class="break" aria-hidden="true"></span>{/if}
          <span class="count num" aria-live="polite">
            {#if phone}
              {#if filtered}{pager?.total != null ? num(pager.total) : '…'} of {num(total)}<span class="sr-only"> images</span>{/if}
            {:else if filtered}
              {pager?.total != null ? num(pager.total) : '…'} of {num(total)} images
            {:else}
              {num(total)} images
            {/if}
          </span>
          {#if filtered}
            <button type="button" class="clear" aria-label="Clear filters" onclick={() => setQuery({ split: null, class: null, image: null })}>
              <X size={12} strokeWidth={2} aria-hidden="true" />Clear
            </button>
          {/if}
          <span class="spacer"></span>
          {#if phone}
            <DisplayMenu {boxes} {density} onboxes={setBoxes} ondensity={setDensity} />
          {:else}
            <button type="button" class="toggle" aria-pressed={boxes} onclick={() => setBoxes(!boxes)}>
              <Scan size={14} strokeWidth={1.75} aria-hidden="true" />Boxes
            </button>
            <Segmented
              label="Thumbnail size"
              size="sm"
              options={[
                { value: 's', label: 'S' },
                { value: 'm', label: 'M' },
                { value: 'l', label: 'L' },
              ]}
              value={density}
              onchange={setDensity}
            />
          {/if}
        </div>

        {#if pager?.error && pager.total !== null}
          <!-- Later pages failed: keep what loaded and offer a retry for the rest. -->
          <ErrorState error={pager.error} title="Some images could not be loaded" retry={() => pager?.retry()} compact />
        {/if}
        {#if pager?.error && pager.total === null}
          <Panel><ErrorState error={pager.error} title="Could not load images" retry={() => pager?.retry()} /></Panel>
        {:else if pager && pager.total === 0}
          <Panel>
            <EmptyState
              icon={ImageOff}
              title="No images match"
              description={cls !== undefined ? `No ${split === 'all' ? '' : `${split} `}image contains a ${d.classes[cls]}.` : `The ${split} split is empty.`}
            >
              {#snippet actions()}<Button onclick={() => setQuery({ split: null, class: null })}>Clear filters</Button>{/snippet}
            </EmptyState>
          </Panel>
        {:else if pager}
          {#key pager}
            <ImageGrid {pager} classes={d.classes} {density} {boxes} {only} onopen={openImage} />
          {/key}
        {/if}
      {/if}
    </div>
  </Page>

  {#if pager}
    <Inspector
      {pager}
      classes={d.classes}
      imageId={imageParam}
      {only}
      onnavigate={(im) => setQuery({ image: im })}
      onclose={() => setQuery({ image: null })}
      onmissing={(im) => {
        // page-level: a linked image that is not in this view; the inspector closes on the next line.
        toast.info('Image not in this view', `${im} does not match the current filters.`);
        setQuery({ image: null });
      }}
    />
  {/if}

  <Dialog bind:open={confirmDelete} title="Delete dataset?" size="sm" status={deleteStatus}>
    <p class="confirm">
      <b>{d.name}</b> will be unregistered from FOLO. The files in <span class="mono">{d.path}</span> stay on disk, and runs that used it keep their results.
    </p>
    {#snippet footer()}
      <Button variant="ghost" onclick={() => (confirmDelete = false)}>Cancel</Button>
      <Button variant="destructive" icon={Trash} loading={deleting} onclick={remove}>Delete dataset</Button>
    {/snippet}
  </Dialog>
{/if}

<style>
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
  /* Phones: split, class and Display share one row (the class name truncates first); filtered, the count and Clear
     follow on a second row, so neither is squeezed to a number and a bare ×. */
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
    flex: 0 1 auto;
  }
  .ds-toolbar.phone .count {
    flex-shrink: 0;
  }
  .toggle {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: var(--control-sm);
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    border: 1px solid var(--line-default);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    white-space: nowrap;
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .toggle:hover {
    color: var(--text-primary);
    border-color: var(--line-strong);
  }
  /* Pressed is neutral (lime is for the primary action, focus and live state): a stronger edge and primary text. */
  .toggle[aria-pressed='true'] {
    border-color: var(--text-tertiary);
    color: var(--text-primary);
  }
  @media (pointer: coarse) {
    .toggle,
    .clear {
      height: 44px;
    }
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
</style>
