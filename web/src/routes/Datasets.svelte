<script lang="ts">
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
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
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

  const classTip = (d: Dataset) => `${d.classes.length} classes: ${d.classes.join(', ')}`;
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

  <Panel pad="none">
    {#if datasets.error && !datasets.data}
      <ErrorState error={datasets.error} title="Could not load datasets" retry={() => datasets.refresh()} />
    {:else if datasets.data?.length === 0}
      <!-- The header already holds the primary action; the empty state offers the quick start. -->
      <EmptyState icon={Images} title="No datasets yet" description="Register a YOLO or COCO folder the server can read, or generate a synthetic shapes set to try FOLO in a minute.">
        {#snippet actions()}<Button icon={Sparkles} onclick={() => add('synthetic')}>Generate synthetic</Button>{/snippet}
      </EmptyState>
    {:else}
      <div class="scroll">
        <table class="table ds" aria-busy={!datasets.data} aria-label="Datasets">
          <thead>
            <tr>
              <th scope="col">Dataset</th>
              <th scope="col" class="hide-md">Format</th>
              <th scope="col" class="hide-lg">Path</th>
              <th scope="col" class="r">Train</th>
              <th scope="col" class="r">Val</th>
              <th scope="col" class="r">Test</th>
              <th scope="col">Class balance</th>
              <th scope="col" class="r hide-md">Boxes</th>
              <th scope="col" class="r hide-md">Created</th>
            </tr>
          </thead>
          <tbody>
            {#if !datasets.data}
              {#each [0, 1, 2] as i (i)}
                <tr aria-hidden="true">
                  <td><Skeleton w="96px" h="12px" /></td>
                  <td class="hide-md"><Skeleton w="32px" h="12px" /></td>
                  <td class="hide-lg"><Skeleton w="120px" h="12px" /></td>
                  <td class="r"><Skeleton w="36px" h="12px" /></td>
                  <td class="r"><Skeleton w="28px" h="12px" /></td>
                  <td class="r"><Skeleton w="28px" h="12px" /></td>
                  <td><Skeleton w="160px" h="6px" r="999px" /></td>
                  <td class="r hide-md"><Skeleton w="36px" h="12px" /></td>
                  <td class="r hide-md"><Skeleton w="44px" h="12px" /></td>
                </tr>
              {/each}
            {:else}
              {#each datasets.data as d (d.id)}
                <tr class="link" onclick={() => router.go(`/datasets/${d.id}`)}>
                  <td><a class="name" href={href(`/datasets/${d.id}`)} onclick={(e) => e.stopPropagation()}>{d.name}</a></td>
                  <td class="hide-md faint">{d.format}</td>
                  <td class="hide-lg path-cell"><span class="path mono truncate" title={d.path}>{d.path}</span></td>
                  <td class="r num">{num(d.splits.train)}</td>
                  <td class="r num muted">{num(d.splits.val)}</td>
                  <td class="r num muted">{num(d.splits.test)}</td>
                  <td title={classTip(d)}>
                    {#if d.stats}<div class="strip"><ClassStrip counts={d.stats.class_counts} classes={d.classes} /></div>{:else}<span class="faint">—</span>{/if}
                  </td>
                  <td class="r num hide-md">{compact(d.stats?.boxes)}</td>
                  <td class="r faint hide-md">{ago(d.created)}</td>
                </tr>
              {/each}
            {/if}
          </tbody>
        </table>
      </div>
      <!-- Phones: two-line rows (name and one meta line, the image count and class balance on the right). -->
      <ul class="rows" aria-label="Datasets">
        {#each datasets.data ?? [] as d (d.id)}
          <li>
            <a href={href(`/datasets/${d.id}`)}>
              <span class="m-main">
                <span class="name truncate">{d.name}</span>
                <span class="m-meta num truncate">{d.format} · {num(d.splits.train)} / {num(d.splits.val)} / {num(d.splits.test)} · {d.classes.length} classes</span>
              </span>
              <span class="m-side">
                <span class="num">{compact(imageCount(d))}<span class="faint"> img</span></span>
                {#if d.stats}<span class="m-strip"><ClassStrip counts={d.stats.class_counts} classes={d.classes} height={4} /></span>{/if}
              </span>
            </a>
          </li>
        {/each}
      </ul>
    {/if}
  </Panel>
</Page>

<AddDatasetDialog bind:open={adding} bind:mode={addMode} />

<style>
  .scroll {
    overflow-x: auto;
  }
  .ds td {
    height: 40px;
  }
  .name {
    font-weight: var(--weight-medium);
  }
  .name:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
  }
  /* The path takes what the row has left, cut with an ellipsis (the full path is in the title). */
  .path-cell {
    width: 100%;
    max-width: 0;
  }
  .path {
    display: block;
    color: var(--text-tertiary);
  }
  .strip {
    width: 160px;
  }
  .rows {
    display: none;
  }
  @media (max-width: 1279px) {
    .hide-lg {
      display: none;
    }
  }
  @media (max-width: 1023px) {
    .hide-md {
      display: none;
    }
  }
  @media (max-width: 767px) {
    .scroll {
      display: none;
    }
    .rows {
      display: block;
    }
    .rows li + li {
      border-top: 1px solid var(--line-subtle);
    }
    .rows a {
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
    .m-meta {
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-tertiary);
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
