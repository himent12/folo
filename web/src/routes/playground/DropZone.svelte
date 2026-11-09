<script lang="ts">
  /** The Playground's empty state: a viewfinder to drop an image into, plus sample images to try. */
  import Images from '@lucide/svelte/icons/images';
  import Upload from '@lucide/svelte/icons/upload';
  import { MediaQuery } from 'svelte/reactivity';
  import type { Dataset, DatasetImage as Img } from '$lib/api/types';
  import DatasetImage from '$lib/detect/DatasetImage.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Kbd from '$lib/ui/Kbd.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import { TOUCH_ONLY } from '$lib/shell/shortcuts';

  let {
    dragging = false,
    samples,
    onbrowse,
    onpick,
    onsample,
  }: {
    dragging?: boolean;
    /** undefined while loading. */
    samples: { ds: Dataset; img: Img }[] | undefined;
    onbrowse: () => void;
    onpick: () => void;
    onsample: (ds: Dataset, img: Img) => void;
  } = $props();

  // A touch-only device (the shared detection, shell/shortcuts' TOUCH_ONLY) has no drag and drop and no Ctrl V: it
  // leads with choosing an image and shows neither hint; a device with a mouse or trackpad shows both.
  const touch = new MediaQuery(TOUCH_ONLY, false);
</script>

<!-- The viewfinder is the drop target and fills the canvas; the samples sit right under it, at the panel's foot. -->
<div class="zone" class:dragging>
  <div class="finder">
    <span class="corner tl" aria-hidden="true"></span>
    <span class="corner tr" aria-hidden="true"></span>
    <span class="corner bl" aria-hidden="true"></span>
    <span class="corner br" aria-hidden="true"></span>

    <div class="center">
      {#if touch.current}
        <p class="title">Choose an image to detect objects</p>
        <p class="desc">JPEG, PNG or WebP, from your photos, camera or files.</p>
      {:else}
        <p class="title">{dragging ? 'Release to detect' : 'Drop an image to detect objects'}</p>
        <p class="desc">JPEG, PNG or WebP. You can also paste from the clipboard with <span class="nowrap"><Kbd keys="Mod+V" />.</span></p>
      {/if}
      <div class="actions">
        <Button variant="primary" icon={Upload} onclick={onbrowse}>{touch.current ? 'Choose an image' : 'Browse files'}</Button>
        <Button icon={Images} onclick={onpick}>Pick from a dataset</Button>
      </div>
    </div>
  </div>

  {#if samples === undefined || samples.length > 0}
    <div class="samples">
      <p class="overline">Or try a sample</p>
      <ul>
        {#if samples === undefined}
          {#each { length: 6 } as _, i (i)}<li><Skeleton h="100%" r="var(--radius-sm)" image /></li>{/each}
        {:else}
          {#each samples as s (s.ds.id + s.img.id)}
            <li>
              <button type="button" class="sample" onclick={() => onsample(s.ds, s.img)} aria-label="Try {s.img.file} from {s.ds.name}" title="{s.ds.name} / {s.img.file}">
                <DatasetImage datasetId={s.ds.id} imageId={s.img.id} width={s.img.width} height={s.img.height} thumb={160} classes={s.ds.classes} fit="cover" compact />
                <span class="tag">{s.ds.name}</span>
              </button>
            </li>
          {/each}
        {/if}
      </ul>
    </div>
  {/if}
</div>

<style>
  .zone {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-4);
    height: 100%;
    padding: var(--space-4) var(--space-4) var(--space-8);
    background: var(--bg-inset);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  /* The drop target: a sparse state's compact composition (DESIGN_DIRECTION 8) at every width, 280–480 px tall, so the
     samples under it sit in the first screen. Beside the rail (from 1024 px) it is 480 px where the window has room
     for it and the samples under the page header (about 420 px of chrome), and gives way in a shorter one. Its message
     is centred between the corners. */
  .finder {
    position: relative;
    display: grid;
    place-items: center;
    width: 100%;
    flex: none;
    height: clamp(280px, calc(100dvh - 420px), 480px);
    padding: var(--space-8) var(--space-6);
  }
  .zone.dragging {
    background: color-mix(in srgb, var(--bg-inset), var(--accent-solid) 6%);
  }
  .corner {
    position: absolute;
    width: 28px;
    height: 28px;
    border: 0 solid var(--line-strong);
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      inset var(--dur-base) var(--ease-out);
  }
  .tl {
    top: 0;
    left: 0;
    border-top-width: 2px;
    border-left-width: 2px;
  }
  .tr {
    top: 0;
    right: 0;
    border-top-width: 2px;
    border-right-width: 2px;
  }
  .bl {
    bottom: 0;
    left: 0;
    border-bottom-width: 2px;
    border-left-width: 2px;
  }
  .br {
    bottom: 0;
    right: 0;
    border-bottom-width: 2px;
    border-right-width: 2px;
  }
  .dragging .corner {
    border-color: var(--accent-line);
  }
  .dragging .tl {
    top: var(--space-3);
    left: var(--space-3);
  }
  .dragging .tr {
    top: var(--space-3);
    right: var(--space-3);
  }
  .dragging .bl {
    bottom: var(--space-3);
    left: var(--space-3);
  }
  .dragging .br {
    bottom: var(--space-3);
    right: var(--space-3);
  }
  .center {
    position: relative;
    display: grid;
    justify-items: center;
    gap: var(--space-2);
    text-align: center;
    max-width: 420px;
  }
  .center::before {
    content: '';
    width: 10px;
    height: 10px;
    margin-bottom: var(--space-3);
    border-radius: 50%;
    background: var(--accent-line);
    box-shadow: 0 0 0 6px var(--accent-muted);
  }
  .title {
    font-size: var(--type-lg-size);
    line-height: var(--type-lg-line);
    letter-spacing: var(--type-lg-tracking);
    font-weight: var(--weight-medium);
  }
  .desc {
    color: var(--text-secondary);
  }
  .actions {
    display: flex;
    gap: var(--space-2);
    margin-top: var(--space-3);
    flex-wrap: wrap;
    justify-content: center;
  }
  .samples {
    display: grid;
    justify-items: center;
    gap: var(--space-2);
    width: 100%;
    max-width: 560px;
  }
  .samples ul {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: var(--space-2);
    width: 100%;
  }
  .samples li {
    aspect-ratio: 1;
  }
  .sample {
    position: relative;
    width: 100%;
    height: 100%;
    border-radius: var(--radius-sm);
    overflow: hidden;
    box-shadow: 0 0 0 1px var(--line-default);
    transition:
      box-shadow var(--dur-fast) var(--ease-standard),
      transform var(--dur-fast) var(--ease-standard);
  }
  .sample:hover {
    box-shadow: 0 0 0 1px var(--text-tertiary);
  }
  .sample:focus-visible {
    outline-offset: 2px;
  }
  .tag {
    position: absolute;
    left: 4px;
    bottom: 4px;
    padding: 0 4px;
    border-radius: var(--radius-xs);
    background: var(--bg-overlay);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    line-height: 16px;
  }
  @media (max-width: 1023px) {
    .finder {
      height: clamp(280px, 40dvh, 480px);
    }
  }
  @media (max-width: 767px) {
    .zone {
      gap: var(--space-6);
      padding: var(--space-3) var(--space-3) var(--space-6);
    }
    /* Phones: as tall as the message and buttons need. */
    .finder {
      height: auto;
      padding: var(--space-8) var(--space-4);
    }
    .samples ul {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .corner {
      width: 20px;
      height: 20px;
    }
  }
</style>
