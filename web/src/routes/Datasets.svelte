<script lang="ts">
  import FolderOpen from '@lucide/svelte/icons/folder-open';
  import Images from '@lucide/svelte/icons/images';
  import Plus from '@lucide/svelte/icons/plus';
  import Sparkles from '@lucide/svelte/icons/sparkles';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Dataset } from '$lib/api/types';
  import ClassStrip from '$lib/detect/ClassStrip.svelte';
  import { ago, compact, num } from '$lib/format';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Badge from '$lib/ui/Badge.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import { tooltip } from '$lib/ui/tooltip';
  import AddDatasetDialog from './datasets/AddDatasetDialog.svelte';
  import { imageCount } from './dataset/health';

  setCrumbs({ label: 'Datasets' });

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

  const totals = $derived.by(() => {
    const list = datasets.data ?? [];
    return {
      images: list.reduce((a, d) => a + imageCount(d), 0),
      boxes: list.reduce((a, d) => a + (d.stats?.boxes ?? 0), 0),
    };
  });

  const splitTip = (d: Dataset) =>
    `train ${num(d.splits.train)} · val ${num(d.splits.val)} · test ${num(d.splits.test)}`;
</script>

<Page title="Datasets">
  {#snippet sub()}
    {#if datasets.data && datasets.data.length}
      <span class="num">{datasets.data.length} {datasets.data.length === 1 ? 'dataset' : 'datasets'} · {compact(totals.images)} images · {compact(totals.boxes)} boxes</span>
    {:else}
      <span>YOLO and COCO folders registered with the server.</span>
    {/if}
  {/snippet}
  {#snippet actions()}
    <Button variant="primary" icon={Plus} onclick={() => add('folder')}>Add dataset</Button>
  {/snippet}

  {#if datasets.error && !datasets.data}
    <Panel><ErrorState error={datasets.error} title="Could not load datasets" retry={() => datasets.refresh()} /></Panel>
  {:else if !datasets.data}
    <div class="list" aria-busy="true" aria-label="Loading datasets">
      <div class="head" aria-hidden="true"><span>Dataset</span></div>
      {#each [0, 1, 2] as i (i)}
        <div class="row sk">
          <div class="c-name"><Skeleton w="120px" h="14px" /><Skeleton w="200px" h="10px" /></div>
          <div class="c-images"><Skeleton w="64px" h="14px" /></div>
          <div class="c-classes"><Skeleton w="24px" h="14px" /></div>
          <div class="c-balance"><Skeleton h="6px" r="999px" /></div>
          <div class="c-boxes"><Skeleton w="48px" h="14px" /></div>
          <div class="c-created"><Skeleton w="48px" h="12px" /></div>
        </div>
      {/each}
    </div>
  {:else if datasets.data.length === 0}
    <Panel>
      <EmptyState icon={Images} title="No datasets yet" description="Register a YOLO or COCO folder the server can read, or generate a synthetic shapes set to try FOLO in a minute.">
        {#snippet actions()}
          <Button variant="primary" icon={FolderOpen} onclick={() => add('folder')}>Register folder</Button>
          <Button icon={Sparkles} onclick={() => add('synthetic')}>Generate synthetic</Button>
        {/snippet}
      </EmptyState>
    </Panel>
  {:else}
    <div class="list" role="table" aria-label="Datasets">
      <div class="head" role="row">
        <span class="c-name" role="columnheader">Dataset</span>
        <span class="c-images r" role="columnheader">Images</span>
        <span class="c-classes r" role="columnheader">Classes</span>
        <span class="c-balance" role="columnheader">Class balance</span>
        <span class="c-boxes r" role="columnheader">Boxes</span>
        <span class="c-created r" role="columnheader">Created</span>
      </div>
      {#each datasets.data as d (d.id)}
        <a class="row" role="row" href={href(`/datasets/${d.id}`)}>
          <span class="c-name" role="cell">
            <span class="name-line">
              <span class="name truncate">{d.name}</span>
              <Badge tone="outline" mono>{d.format}</Badge>
            </span>
            <span class="path mono truncate" title={d.path}>{d.path}</span>
          </span>
          <span class="c-images r num" role="cell">
            <span class="big" use:tooltip={splitTip(d)}>{num(imageCount(d))}</span>
            <span class="splits">
              <span title="train">{compact(d.splits.train)}</span><i>/</i><span title="val">{compact(d.splits.val)}</span><i>/</i><span title="test">{compact(d.splits.test)}</span>
            </span>
          </span>
          <span class="c-classes r num" role="cell"><span class="big">{d.classes.length}</span><span class="lbl">classes</span></span>
          <span class="c-balance" role="cell">
            {#if d.stats}
              <ClassStrip counts={d.stats.class_counts} classes={d.classes} />
              <span class="classes truncate">{d.classes.join(', ')}</span>
            {:else}
              <span class="faint">—</span>
            {/if}
          </span>
          <span class="c-boxes r num" role="cell"><span class="big">{compact(d.stats?.boxes)}</span><span class="lbl">boxes</span></span>
          <span class="c-created r faint" role="cell">{ago(d.created)}</span>
        </a>
      {/each}
    </div>
  {/if}
</Page>

<AddDatasetDialog bind:open={adding} bind:mode={addMode} />

<style>
  .list {
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
    border: 1px solid var(--line-subtle);
    overflow: hidden;
  }
  :global([data-theme='light']) .list {
    border-color: var(--line-default);
  }
  .head,
  .row {
    display: grid;
    grid-template-columns: minmax(0, 2.2fr) 112px 72px minmax(160px, 1.6fr) 72px 88px;
    align-items: center;
    gap: var(--space-6);
    padding: 0 var(--space-4);
  }
  .head {
    height: 32px;
    border-bottom: 1px solid var(--line-subtle);
    font-size: var(--type-xs-size);
    font-weight: var(--weight-medium);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .row {
    min-height: 64px;
    padding-block: var(--space-3);
    border-bottom: 1px solid var(--line-subtle);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .row:last-child {
    border-bottom: 0;
  }
  a.row:hover {
    background: var(--bg-hover);
  }
  a.row:focus-visible {
    outline-offset: -2px;
    border-radius: 0;
  }
  .r {
    text-align: right;
    justify-self: end;
  }
  .c-name {
    display: grid;
    gap: 2px;
    min-width: 0;
  }
  .name-line {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .name {
    font-weight: var(--weight-medium);
    font-size: var(--type-md-size);
  }
  .path {
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .c-images,
  .c-classes,
  .c-boxes {
    display: grid;
    gap: 2px;
  }
  .big {
    font-weight: var(--weight-medium);
  }
  .lbl {
    display: none;
  }
  .splits,
  .lbl {
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .splits i {
    font-style: normal;
    margin: 0 2px;
    color: var(--text-disabled);
  }
  .c-balance {
    display: grid;
    gap: 6px;
    min-width: 0;
  }
  .classes {
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .c-created {
    font-size: var(--type-sm-size);
    white-space: nowrap;
  }
  .sk .c-name {
    gap: var(--space-2);
  }
  .sk .c-images,
  .sk .c-classes,
  .sk .c-boxes {
    justify-self: end;
  }
  @media (max-width: 1023px) {
    .head,
    .row {
      grid-template-columns: minmax(0, 2fr) 104px 64px minmax(120px, 1.4fr) 64px;
    }
    .c-created {
      display: none;
    }
  }
  @media (max-width: 767px) {
    .head {
      display: none;
    }
    .row {
      grid-template-columns: minmax(0, 1fr) auto auto;
      grid-template-areas:
        'name name name'
        'images classes boxes'
        'balance balance balance';
      gap: var(--space-3) var(--space-4);
      padding: var(--space-4);
    }
    .c-name {
      grid-area: name;
    }
    .c-images {
      grid-area: images;
    }
    .c-classes {
      grid-area: classes;
    }
    .c-boxes {
      grid-area: boxes;
    }
    .c-balance {
      grid-area: balance;
    }
    .c-images,
    .c-classes,
    .c-boxes {
      display: flex;
      align-items: baseline;
      gap: var(--space-1);
      justify-self: start;
      text-align: left;
    }
    .c-images .splits {
      display: none;
    }
    .lbl {
      display: inline;
    }
    .c-images::after {
      content: 'images';
      font-size: var(--type-xs-size);
      color: var(--text-tertiary);
    }
  }
</style>
