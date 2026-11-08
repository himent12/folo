<script lang="ts">
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import Check from '@lucide/svelte/icons/check';
  import Copy from '@lucide/svelte/icons/copy';
  import ImageOff from '@lucide/svelte/icons/image-off';
  import Play from '@lucide/svelte/icons/play';
  import Scan from '@lucide/svelte/icons/scan';
  import Trash from '@lucide/svelte/icons/trash-2';
  import { ApiError, hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Split } from '$lib/api/types';
  import { classInk } from '$lib/design/classInk.svelte';
  import { ago, compact, num } from '$lib/format';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Badge from '$lib/ui/Badge.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import Tabs from '$lib/ui/Tabs.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import { tooltip } from '$lib/ui/tooltip';
  import { healthChecks, imageCount } from './dataset/health';
  import ImageGrid, { type Density } from './dataset/ImageGrid.svelte';
  import Inspector from './dataset/Inspector.svelte';
  import { ImagePager } from './dataset/pager.svelte';
  import StatsTab from './dataset/StatsTab.svelte';

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

  // ---------- header actions ----------
  let copied = $state(false);
  async function copyPath() {
    if (!d) return;
    try {
      await navigator.clipboard.writeText(d.path);
      copied = true;
      setTimeout(() => (copied = false), 1500);
    } catch {
      toast.error(new Error('The browser blocked clipboard access.'), 'Could not copy the path');
    }
  }

  let confirmDelete = $state(false);
  let deleting = $state(false);
  async function remove() {
    if (!d) return;
    deleting = true;
    try {
      await hub.api.deleteDataset(d.id);
      confirmDelete = false;
      toast.success('Dataset removed', `${d.name} is no longer registered. Files on disk were not touched.`);
      router.go('/datasets');
    } catch (e) {
      toast.error(e, 'Could not remove the dataset');
    } finally {
      deleting = false;
    }
  }

  // ---------- derived view data ----------
  const total = $derived(d ? imageCount(d) : 0);
  const issues = $derived(
    d?.stats ? healthChecks(d, d.stats).filter((h) => h.severity === 'danger' || h.severity === 'warning').length : 0,
  );
  const splitOptions = $derived.by(() => {
    if (!d) return [];
    const opts: { value: SplitFilter; label: string; meta: string }[] = [{ value: 'all', label: 'All', meta: compact(total) }];
    for (const s of ['train', 'val', 'test'] as const)
      if (d.splits[s] > 0 || split === s) opts.push({ value: s, label: s[0]!.toUpperCase() + s.slice(1), meta: compact(d.splits[s]) });
    return opts;
  });

  /** Fades a horizontally scrolling rail's edges (24 px) on the sides that have more to scroll to. */
  function edgeFade(el: HTMLElement) {
    const update = () => {
      const max = el.scrollWidth - el.clientWidth;
      el.style.setProperty('--fade-l', el.scrollLeft > 1 ? '24px' : '0px');
      el.style.setProperty('--fade-r', el.scrollLeft < max - 1 ? '24px' : '0px');
    };
    // Start at the beginning; scroll only when the active chip (a linked ?class=) is off-screen, and then just far
    // enough to show it clear of the right fade.
    el.scrollLeft = 0;
    const on = el.querySelector<HTMLElement>('[aria-pressed="true"]');
    if (on) {
      const right = on.getBoundingClientRect().right - el.getBoundingClientRect().left;
      if (right > el.clientWidth) el.scrollLeft = right - el.clientWidth + 32;
    }
    update();
    el.addEventListener('scroll', update, { passive: true });
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return {
      destroy() {
        el.removeEventListener('scroll', update);
        ro.disconnect();
      },
    };
  }

  function openImage(_i: number, imageId: string) {
    setQuery({ image: imageId });
  }
</script>

{#if ds.error && !d}
  <Page title="Dataset">
    <Panel>
      {#if ds.error instanceof ApiError && ds.error.status === 404}
        <ErrorState error={ds.error} title="Dataset not found" />
      {:else}
        <ErrorState error={ds.error} title="Could not load this dataset" retry={() => ds.refresh()} />
      {/if}
      <div class="back"><Button variant="ghost" icon={ArrowLeft} href={href('/datasets')}>All datasets</Button></div>
    </Panel>
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
    {#snippet lead()}
      <Badge tone="outline" mono>{d.format}</Badge>
      <span class="id mono">{d.id}</span>
    {/snippet}
    {#snippet sub()}
      <span class="path">
        <span class="mono truncate" title={d.path}>{d.path}</span>
        <span use:tooltip={copied ? 'Copied' : 'Copy path'}>
          <Button variant="ghost" size="sm" icon={copied ? Check : Copy} aria-label="Copy path" onclick={copyPath} />
        </span>
      </span>
      <span class="facts num">
        <span><b>{num(d.splits.train)}</b> train</span>
        <span><b>{num(d.splits.val)}</b> val</span>
        <span><b>{num(d.splits.test)}</b> test</span>
        <span class="dot" aria-hidden="true">·</span>
        <span><b>{d.classes.length}</b> classes</span>
        <span class="dot hide-sm" aria-hidden="true">·</span>
        <span class="hide-sm">added {ago(d.created)}</span>
      </span>
    {/snippet}
    {#snippet actions()}
      <Button icon={Trash} onclick={() => (confirmDelete = true)}>Delete</Button>
      <Button variant="primary" icon={Play} href={href(`/runs/new?dataset=${encodeURIComponent(d.id)}`)}>Train on this dataset</Button>
    {/snippet}

    <div class="tabs">
      <Tabs
        label="Dataset views"
        idPrefix="ds-tab"
        tabs={[
          { value: 'images', label: 'Images', meta: compact(total) },
          { value: 'stats', label: 'Stats', meta: issues ? `${issues} to review` : undefined },
        ]}
        value={tab}
        onchange={(t) => setQuery({ tab: t === 'images' ? null : t, image: null })}
      />
    </div>

    <div id="ds-tab-panel" role="tabpanel" aria-labelledby="ds-tab-{tab}" class="panel">
      {#if tab === 'stats'}
        {#if d.stats}
          <StatsTab dataset={d} stats={d.stats} onclass={(c) => setQuery({ tab: null, class: String(c), image: null })} />
        {:else}
          <Panel><EmptyState icon={ImageOff} title="No statistics" description="The server did not return statistics for this dataset." /></Panel>
        {/if}
      {:else}
        <div class="ds-toolbar">
          <div class="tb-row">
            <Segmented
              label="Split"
              size="sm"
              options={splitOptions}
              value={split}
              onchange={(s) => setQuery({ split: s === 'all' ? null : s, image: null })}
            />
            <span class="count num" aria-live="polite">
              {#if pager?.total != null}<b>{num(pager.total)}</b> {pager.total === 1 ? 'image' : 'images'}{:else}&nbsp;{/if}
            </span>
            <span class="spacer"></span>
            <button
              type="button"
              class="toggle"
              aria-pressed={boxes}
              onclick={() => {
                boxes = !boxes;
                writePref('folo.datasets.boxes', boxes ? 'on' : 'off');
              }}
            >
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
              bind:value={density}
              onchange={(v) => writePref('folo.datasets.density', v)}
            />
          </div>
          <div class="chips" role="group" aria-label="Filter by class" use:edgeFade>
            <button type="button" class="chip" aria-pressed={cls === undefined} onclick={() => setQuery({ class: null, image: null })}>All classes</button>
            {#each d.classes as c, i (i)}
              <button
                type="button"
                class="chip"
                aria-pressed={cls === i}
                onclick={() => setQuery({ class: cls === i ? null : String(i), image: null })}
              >
                <span class="sw" style:background={classInk(i)}></span>{c}
                {#if d.stats}<span class="cmeta num">{compact(d.stats.class_counts[i])}</span>{/if}
              </button>
            {/each}
          </div>
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
        toast.info('Image not in this view', `${im} does not match the current filters.`);
        setQuery({ image: null });
      }}
    />
  {/if}

  <Dialog bind:open={confirmDelete} title="Delete dataset?" size="sm">
    <p class="confirm">
      <b>{d.name}</b> will be unregistered from FOLO. The files in <span class="mono">{d.path}</span> stay on disk, and runs that used it keep their results.
    </p>
    {#snippet footer()}
      <Button variant="ghost" onclick={() => (confirmDelete = false)}>Cancel</Button>
      <Button variant="danger" icon={Trash} loading={deleting} onclick={remove}>Delete dataset</Button>
    {/snippet}
  </Dialog>
{/if}

<style>
  .id {
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .path {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    min-width: 0;
    max-width: 100%;
  }
  .path .mono {
    color: var(--text-secondary);
  }
  .facts {
    display: inline-flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-3);
    color: var(--text-tertiary);
  }
  .facts b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .dot {
    color: var(--text-disabled);
  }
  .tabs {
    margin-bottom: calc(var(--space-1) * -1);
  }
  .panel {
    display: grid;
    gap: var(--space-3);
    min-width: 0;
  }
  .back {
    padding: 0 var(--space-4) var(--space-4);
  }
  .ds-toolbar {
    position: sticky;
    top: var(--size-topbar);
    z-index: calc(var(--z-sticky) - 1);
    display: grid;
    gap: var(--space-2);
    margin: calc(var(--space-2) * -1) calc(var(--space-6) * -1) 0;
    padding: var(--space-2) var(--space-6) var(--space-3);
    background: color-mix(in srgb, var(--bg-app) 92%, transparent);
    backdrop-filter: blur(8px);
  }
  .tb-row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    flex-wrap: wrap;
  }
  .count {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .count b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .spacer {
    flex: 1;
  }
  .toggle,
  .chip {
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
      background-color var(--dur-fast) var(--ease-standard),
      border-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .toggle:hover,
  .chip:hover {
    color: var(--text-primary);
    border-color: var(--line-strong);
  }
  /* Selected is neutral (lime is for the primary action, focus and live state): a raised fill and a stronger
     edge (1.5 px via the inset ring, so the chip doesn't change size). */
  .toggle[aria-pressed='true'],
  .chip[aria-pressed='true'] {
    background: var(--bg-raised);
    border-color: var(--line-strong);
    box-shadow: inset 0 0 0 0.5px var(--line-strong);
    color: var(--text-primary);
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1-5);
  }
  .chip {
    border-radius: var(--radius-full);
    padding: 0 10px;
  }
  @media (pointer: coarse) {
    .toggle,
    .chip {
      height: 36px;
    }
    .chip {
      padding: 0 var(--space-3);
    }
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
  }
  .cmeta {
    color: var(--text-tertiary);
    font-weight: var(--weight-regular);
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
    max-width: var(--size-content-max);
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
    .chips {
      flex-wrap: nowrap;
      overflow-x: auto;
      scrollbar-width: none;
      margin-inline: calc(var(--space-4) * -1);
      padding-inline: var(--space-4);
      /* Clipped chips read as scrollable: the edges with more to scroll to fade out. */
      mask-image: linear-gradient(
        to right,
        transparent,
        #000 var(--fade-l, 0px),
        #000 calc(100% - var(--fade-r, 0px)),
        transparent
      );
    }
    .hide-sm {
      display: none;
    }
    .sk-page {
      padding: var(--space-4);
    }
    .sk-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
</style>
