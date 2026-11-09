<script lang="ts">
  import Images from '@lucide/svelte/icons/images';
  import Plus from '@lucide/svelte/icons/plus';
  import Search from '@lucide/svelte/icons/search';
  import Sparkles from '@lucide/svelte/icons/sparkles';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Dataset } from '$lib/api/types';
  import ClassStrip from '$lib/detect/ClassStrip.svelte';
  import { ago, num } from '$lib/format';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { ownCreateAction, setCrumbs } from '$lib/shell/crumbs.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import TableState from '$lib/ui/TableState.svelte';
  import Gate from '$lib/ui/Gate.svelte';
  import { isOffline, OFFLINE_REASON } from '$lib/ui/offline';
  import Meta from '$lib/ui/Meta.svelte';
  import { rowNav } from '$lib/ui/rowNav';
  import AddDatasetDialog from './datasets/AddDatasetDialog.svelte';
  import { imageCount } from './dataset/health';

  setCrumbs({ label: 'Datasets' });
  // The header's create action is the one in the top-right corner (the top bar's New run steps aside).
  ownCreateAction();

  // The list endpoint has no stats; fetch each dataset for its class balance and box count.
  const datasets = new Resource(async (s) => {
    const list = await hub.api.datasets(s);
    return Promise.all(list.map((d) => hub.api.dataset(d.id, s).catch(() => d)));
  });

  // `#/datasets?add=folder|synthetic` opens the dialog (used by the onboarding on an empty Overview).
  const addParam = router.route.query.get('add');
  let adding = $state(addParam === 'folder' || addParam === 'synthetic');
  let addMode = $state<'folder' | 'synthetic'>(addParam === 'synthetic' ? 'synthetic' : 'folder');
  function add(m: 'folder' | 'synthetic') {
    addMode = m;
    adding = true;
  }

  // What the rows shown hold, said once at the list's end (base.css .list-foot).
  const totals = $derived.by(() => {
    const list = shown;
    return {
      images: list.reduce((a, d) => a + imageCount(d), 0),
      boxes: list.reduce((a, d) => a + (d.stats?.boxes ?? 0), 0),
    };
  });

  let query = $state('');
  // The toolbar's filter, at its left as on Runs and Models (one shape for the three list toolbars): by format.
  let format = $state<'all' | 'yolo' | 'coco'>('all');
  const formatCount = (f: 'all' | 'yolo' | 'coco') => (datasets.data ?? []).filter((d) => f === 'all' || d.format === f).length;
  const shown = $derived.by(() => {
    const q = query.trim().toLowerCase();
    return (datasets.data ?? []).filter(
      (d) => (format === 'all' || d.format === format) && (!q || `${d.name} ${d.path} ${d.format} ${d.id}`.toLowerCase().includes(q)),
    );
  });
  /** Columns of the desktop table (for state rows that span it), the empty last one included. */
  const COLS = 10;

  // Nothing answers: adding a dataset needs the server, so the action is disabled and says why (as on Runs).
  const offline = $derived(!datasets.data && isOffline(datasets.error));
  /** Datasets to filter, or on their way: the toolbar is live (else it stays in place, disabled). */
  const hasData = $derived(datasets.data ? datasets.data.length > 0 : !datasets.error);
  /** Rows to head (or their skeleton): the column heads show; with none the frame holds the state alone. */
  const hasRows = $derived(datasets.data ? shown.length > 0 : !datasets.error);

  const classTip = (d: Dataset) => `${d.classes.length} classes: ${d.classes.join(', ')}`;
</script>

<Page title="Datasets">
  <!-- One fixed line: the counts live at the list's end, so nothing here changes as the list loads. -->
  {#snippet sub()}<span>YOLO and COCO folders registered with the server.</span>{/snippet}
  <!-- Phones: one header row, this line beside Add dataset. -->
  {#snippet subPhone()}<span>YOLO and COCO folders.</span>{/snippet}
  {#snippet actions()}
    <Gate reason={offline ? OFFLINE_REASON : null}>
      {#snippet children(why)}
        <Button variant="primary" icon={Plus} disabled={!!why} aria-describedby={why} onclick={() => add('folder')}>Add dataset</Button>
      {/snippet}
    </Gate>
  {/snippet}

  <!-- Always here, so the table never moves between states (DESIGN_DIRECTION §2.8): over an empty or failed list it is
       disabled and shows no counts. -->
  <div class="list-toolbar">
    <div class="lt-filter"><Segmented
      label="Filter datasets by format"
      bind:value={format}
      disabled={!hasData}
      options={[
        { value: 'all', label: 'All', meta: datasets.data?.length ? String(formatCount('all')) : undefined },
        { value: 'yolo', label: 'YOLO', meta: datasets.data?.length ? String(formatCount('yolo')) : undefined },
        { value: 'coco', label: 'COCO', meta: datasets.data?.length ? String(formatCount('coco')) : undefined },
      ]}
    /></div>
    <div class="lt-search"><Input bind:value={query} icon={Search} placeholder="Filter by name or path" aria-label="Filter datasets" type="search" disabled={!hasData} /></div>
  </div>

  <Panel pad="none">
    <!-- With no rows (empty, no match, an error) the frame drops its column heads and holds the state alone. -->
    <div class="scroll" class:state={!hasRows}>
      <table class="table ds" aria-busy={!datasets.data} aria-label="Datasets" use:rowNav>
        {#if hasRows}<thead>
          <tr>
            <!-- One geometry in every state (DESIGN_DIRECTION 7): fixed widths on the header row that loading and
                 loaded share. A few rows across a wide panel take bounded widths (the path 280 px, cut with
                 its full path in the title) and leave the slack to an empty last column, as Model's per class does;
                 below 1280, where the path and that column hide, the class balance takes what is left, at least
                 160 px (minmax(160px, 1fr)). -->
            <th scope="col" style:width="160px">Dataset</th>
            <th scope="col" class="hide-md" style:width="80px">Format</th>
            <th scope="col" class="hide-lg" style:width="280px">Path</th>
            <th scope="col" class="r" style:width="80px">Train</th>
            <th scope="col" class="r" style:width="72px">Val</th>
            <th scope="col" class="r" style:width="72px">Test</th>
            <th scope="col" class="bal"><span class="bal-h">Class balance</span></th>
            <th scope="col" class="r hide-md" style:width="88px">Boxes</th>
            <th scope="col" class="r hide-md" style:width="104px">Created</th>
            <th class="hide-lg rest" aria-hidden="true"></th>
          </tr>
        </thead>{/if}
        <tbody>
          {#if datasets.error && !datasets.data}
            <TableState cols={COLS}><ErrorState error={datasets.error} title="Could not load datasets" retry={() => datasets.refresh()} /></TableState>
          {:else if datasets.data?.length === 0}
            <!-- The header already holds the primary action (and the subtitle says what a dataset is): offer the quick start. -->
            <TableState cols={COLS}>
              <EmptyState icon={Images} title="No datasets yet" description="Try FOLO in a minute with a generated set of shapes.">
                {#snippet actions()}<Button icon={Sparkles} onclick={() => add('synthetic')}>Generate sample set</Button>{/snippet}
              </EmptyState>
            </TableState>
          {:else if datasets.data && shown.length === 0}
            <TableState cols={COLS}>
              <EmptyState icon={Search} title="No datasets match" description="Try another format or search term.">
                {#snippet actions()}<Button size="sm" onclick={() => ((query = ''), (format = 'all'))}>Clear filters</Button>{/snippet}
              </EmptyState>
            </TableState>
          {:else if !datasets.data}
            {#each [0, 1, 2] as i (i)}
              <tr aria-hidden="true">
                <td><Skeleton w="96px" h="12px" /></td>
                <td class="hide-md"><Skeleton w="32px" h="12px" /></td>
                <td class="hide-lg"><Skeleton w="120px" h="12px" /></td>
                <td class="r"><Skeleton w="36px" h="12px" /></td>
                <td class="r"><Skeleton w="28px" h="12px" /></td>
                <td class="r"><Skeleton w="28px" h="12px" /></td>
                <td><Skeleton h="6px" r="999px" /></td>
                <td class="r hide-md"><Skeleton w="36px" h="12px" /></td>
                <td class="r hide-md"><Skeleton w="44px" h="12px" /></td>
                <td class="hide-lg rest"></td>
              </tr>
            {/each}
          {:else}
            {#each shown as d (d.id)}
              <tr class="link" data-row aria-label={d.name} onclick={() => router.go(`/datasets/${d.id}`)}>
                <td><a class="name" href={href(`/datasets/${d.id}`)} onclick={(e) => e.stopPropagation()}>{d.name}</a></td>
                <td class="hide-md faint">{d.format}</td>
                <td class="hide-lg path-cell"><span class="path mono truncate" title={d.path}>{d.path}</span></td>
                <td class="r num muted">{num(d.splits.train)}</td>
                <td class="r num muted">{num(d.splits.val)}</td>
                <td class="r num muted">{num(d.splits.test)}</td>
                <td title={classTip(d)}>
                  {#if d.stats}<ClassStrip counts={d.stats.class_counts} classes={d.classes} />{:else}
<span class="faint">—</span>{/if}
                </td>
                <!-- Full numbers, like the split columns beside it: one format per column. -->
                <td class="r num hide-md">{num(d.stats?.boxes)}</td>
                <td class="r faint hide-md">{ago(d.created)}</td>
                <td class="hide-lg rest" aria-hidden="true"></td>
              </tr>
            {/each}
          {/if}
        </tbody>
      </table>
    </div>
    <!-- Phones: the name, then the image total and the class count, then the splits (each line wraps whole items, so no
         count is cut); the class balance on the right. -->
    {#if shown.length}
      <ul class="rows" aria-label="Datasets">
        {#each shown as d (d.id)}
          <li>
            <a href={href(`/datasets/${d.id}`)}>
              <span class="m-main">
                <span class="name truncate">{d.name}</span>
                <span class="m-meta m-total num"><Meta><span>{num(imageCount(d))} images</span><span>{d.classes.length} classes</span></Meta></span>
                <span class="m-meta num"><Meta><span>{num(d.splits.train)} train</span><span>{num(d.splits.val)} val</span><span>{num(d.splits.test)} test</span></Meta></span>
              </span>
              {#if d.stats}<span class="m-side"><span class="m-strip"><ClassStrip counts={d.stats.class_counts} classes={d.classes} height={4} /></span></span>{/if}
            </a>
          </li>
        {/each}
      </ul>
    {:else if !datasets.data && !datasets.error}
      <!-- Phones, loading: the rows' shape (the name over two meta lines, the class balance on the right). -->
      <ul class="rows" aria-hidden="true">
        {#each [0, 1, 2] as i (i)}
          <li>
            <span class="row-sk">
              <span class="m-main"><Skeleton w="55%" h="14px" /><Skeleton w="45%" h="10px" /><Skeleton w="65%" h="10px" /></span>
              <span class="m-side"><Skeleton w="64px" h="4px" /></span>
            </span>
          </li>
        {/each}
      </ul>
    {/if}
  </Panel>
  {#if datasets.data && shown.length}
    <p class="list-foot num">
      <Meta>
        <span>{shown.length < datasets.data.length ? `${shown.length} of ${datasets.data.length}` : shown.length} {datasets.data.length === 1 ? 'dataset' : 'datasets'}</span>
        <span>{num(totals.images)} images</span>
        <span>{num(totals.boxes)} boxes</span>
      </Meta>
    </p>
  {/if}
</Page>

<AddDatasetDialog bind:open={adding} bind:mode={addMode} known={datasets.data} />

<style>
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
    .rows a,
    .rows .row-sk {
      display: flex;
      align-items: center;
      gap: var(--space-3);
      min-height: 56px;
      padding: var(--space-2) var(--space-4);
    }
    .rows a:active {
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
      display: grid;
      justify-items: end;
      gap: var(--space-1);
      flex-shrink: 0;
    }
    .m-strip {
      width: 64px;
    }
  }
</style>
