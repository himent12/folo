<script lang="ts" module>
  import type { LabelImage } from '$lib/api/types';

  export type Show = 'all' | 'pending' | 'labeled' | 'failed';

  /** Whether an image is in a filter: "To label" holds the images waiting and the ones with the agent now. */
  export const inFilter = (img: LabelImage, show: Show) =>
    show === 'all' || (show === 'pending' ? img.status === 'pending' || img.status === 'labeling' : img.status === show);
</script>

<script lang="ts">
  /**
   * A labeling job's images: a toolbar (which images, how they show, Add images), the upload in progress or its
   * result, and the virtualised grid. Image files dropped anywhere on it are uploaded. With no images yet, the grid's
   * place is a drop target (the Playground's look, and its rule: a touch device is told to choose, never to drop).
   */
  import FolderOpen from '@lucide/svelte/icons/folder-open';
  import ImageOff from '@lucide/svelte/icons/image-off';
  import ImagePlus from '@lucide/svelte/icons/image-plus';
  import Scan from '@lucide/svelte/icons/scan';
  import Upload from '@lucide/svelte/icons/upload';
  import { MediaQuery } from 'svelte/reactivity';
  import { media } from '$lib/design/tokens';
  import { num } from '$lib/format';
  import { TOUCH_ONLY } from '$lib/shell/shortcuts';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import { gridView } from '$lib/ui/gridView.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';
  import TileGrid, { dominantAspect } from '$lib/ui/TileGrid.svelte';
  import ToggleButton from '$lib/ui/ToggleButton.svelte';
  import Viewfinder from '$lib/ui/Viewfinder.svelte';
  import DisplayMenu from '../dataset/DisplayMenu.svelte';
  import LabelTile from './LabelTile.svelte';
  import UploadStatus from './UploadStatus.svelte';
  import type { Uploads } from './uploads.svelte';

  let {
    jobId,
    classes,
    images,
    show,
    uploads,
    onshow,
    onopen,
    onbrowse,
    onadd,
  }: {
    jobId: string;
    classes: readonly string[];
    images: readonly LabelImage[];
    show: Show;
    uploads: Uploads;
    onshow: (s: Show) => void;
    onopen: (id: string) => void;
    /** Opens the file picker. */
    onbrowse: () => void;
    /** Opens Add images, on its upload or its server folder tab. */
    onadd: (tab: 'upload' | 'folder') => void;
  } = $props();

  const touchOnly = new MediaQuery(TOUCH_ONLY, false);
  const phoneQuery = new MediaQuery(media.phone, false);
  const phone = $derived(phoneQuery.current);
  const touch = $derived(touchOnly.current || phone);

  const counts = $derived({
    all: images.length,
    pending: images.filter((i) => inFilter(i, 'pending')).length,
    labeled: images.filter((i) => i.status === 'labeled').length,
    failed: images.filter((i) => i.status === 'failed').length,
  });
  const shown = $derived(show === 'all' ? images : images.filter((i) => inFilter(i, show)));
  const options = $derived(
    (
      [
        ['all', 'All'],
        ['pending', 'To label'],
        ['labeled', 'Labeled'],
        ['failed', 'Failed'],
      ] as const
    ).map(([value, label]) => ({ value, label, meta: num(counts[value]) })),
  );
  // The cells take the job's commonest image shape (ui/TileGrid).
  const aspect = $derived(dominantAspect(images.slice(0, 64)));

  let dragging = $state(false);
  let depth = 0;
  const hasFiles = (e: DragEvent) => !!e.dataTransfer?.types.includes('Files');
  function ondragenter(e: DragEvent) {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth++;
    dragging = true;
  }
  function ondragleave() {
    depth = Math.max(0, depth - 1);
    if (!depth) dragging = false;
  }
  function ondrop(e: DragEvent) {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    dragging = false;
    void uploads.files([...(e.dataTransfer?.files ?? [])]);
  }

  const label = (i: number) => {
    const img = shown[i];
    return img ? `${img.file}, ${img.status === 'labeled' ? `${img.boxes.length} ${img.boxes.length === 1 ? 'box' : 'boxes'}` : img.status}` : 'Image';
  };
</script>

<!-- The whole area takes dropped files. -->
<section
  class="area"
  aria-label="Images"
  {ondragenter}
  ondragover={(e) => hasFiles(e) && e.preventDefault()}
  {ondragleave}
  {ondrop}
>
  <div class="lj-toolbar" class:phone>
    {#if phone}
      <span class="pick"><Select label="Filter images by status" size="sm" options={options.map((o) => ({ value: o.value, label: `${o.label} · ${o.meta}` }))} value={show} onchange={onshow} /></span>
      <DisplayMenu boxes={gridView.boxes} density={gridView.density} onboxes={(on) => (gridView.boxes = on)} ondensity={(d) => (gridView.density = d)} />
      <Button size="sm" icon={ImagePlus} aria-label="Add images" onclick={() => onadd('upload')} />
    {:else}
      <Segmented label="Filter images by status" size="sm" {options} value={show} onchange={onshow} disabled={!images.length} />
      <span class="spacer"></span>
      <ToggleButton pressed={gridView.boxes} label="Boxes" icon={Scan} onchange={(on) => (gridView.boxes = on)} />
      <Segmented
        label="Thumbnail size"
        size="sm"
        options={[
          { value: 's', label: 'S' },
          { value: 'm', label: 'M' },
          { value: 'l', label: 'L' },
        ]}
        value={gridView.density}
        onchange={(d) => (gridView.density = d)}
      />
      <Button size="sm" icon={ImagePlus} onclick={() => onadd('upload')}>Add images</Button>
    {/if}
  </div>

  <UploadStatus {uploads} />

  {#if images.length === 0}
    <div class="zone" class:dragging>
      <Viewfinder
        {dragging}
        title={touch ? 'Choose images to label' : dragging ? 'Release to add them' : 'Drop images to label'}
      >
        {#snippet description()}
          {#if touch}JPEG, PNG, BMP or WebP, from your photos or files.{:else}JPEG, PNG, BMP or WebP, as many as you like. They stay in this job until you make a dataset.{/if}
        {/snippet}
        {#snippet actions()}
          <Button variant="primary" icon={Upload} onclick={onbrowse}>{touch ? 'Choose images' : 'Browse files'}</Button>
          <Button icon={FolderOpen} onclick={() => onadd('folder')}>Add a folder on the server</Button>
        {/snippet}
      </Viewfinder>
    </div>
  {:else if shown.length === 0}
    <Panel>
      <EmptyState icon={ImageOff} title="No images here" description={show === 'failed' ? 'The agent labeled every image it was given.' : show === 'labeled' ? 'No image is labeled yet.' : 'Every image has labels.'}>
        {#snippet actions()}<Button onclick={() => onshow('all')}>Show all</Button>{/snippet}
      </EmptyState>
    </Panel>
  {:else}
    <div class="grid-wrap">
      {#key show}
        <TileGrid total={shown.length} density={gridView.density} {aspect} {label} onopen={(i) => shown[i] && onopen(shown[i].id)} toolbar=".lj-toolbar">
          {#snippet tile(i)}
            {@const img = shown[i]}
            {#if img}<LabelTile {img} {jobId} {classes} boxes={gridView.boxes} density={gridView.density} />{/if}
          {/snippet}
        </TileGrid>
      {/key}
      {#if dragging && !touch}
        <div class="over">
          <Viewfinder dragging title="Release to add them">
            {#snippet description()}They join the job as images to label.{/snippet}
          </Viewfinder>
        </div>
      {/if}
    </div>
  {/if}
</section>

<style>
  .area {
    display: grid;
    gap: var(--space-3);
    align-content: start;
    min-width: 0;
  }
  /* The images' toolbar sticks under the top bar as the grid scrolls, as a dataset's does. */
  .lj-toolbar {
    position: sticky;
    top: var(--size-topbar);
    z-index: calc(var(--z-sticky) - 1);
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-2) var(--space-3);
    margin: calc(var(--space-2) * -1) 0 0;
    padding: var(--space-2) 0;
    background: color-mix(in srgb, var(--bg-app) 92%, transparent);
    backdrop-filter: blur(8px);
  }
  .spacer {
    flex: 1;
  }
  .lj-toolbar.phone {
    flex-wrap: nowrap;
    gap: var(--space-2);
  }
  .pick {
    flex: 1;
    min-width: 0;
  }
  .zone {
    display: grid;
    border-radius: var(--radius-lg);
    background: var(--bg-inset);
    --vf-pad: var(--space-8) var(--space-6);
    min-height: 360px;
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .zone.dragging {
    background: color-mix(in srgb, var(--bg-inset), var(--accent-solid) 6%);
  }
  .grid-wrap {
    position: relative;
  }
  /* Files over a grid: the drop target over the whole of it. */
  .over {
    position: absolute;
    inset: 0;
    display: grid;
    border-radius: var(--radius-lg);
    background: color-mix(in srgb, var(--bg-app) 86%, transparent);
    pointer-events: none;
  }
  @media (max-width: 767px) {
    .zone {
      min-height: 0;
      --vf-pad: var(--space-8) var(--space-4);
    }
  }
</style>
