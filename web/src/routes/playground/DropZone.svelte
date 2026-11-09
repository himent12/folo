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

  // Touch screens have no drag and drop or Ctrl+V: offer the picker instead of promising either.
  const touch = new MediaQuery('pointer: coarse', false);
</script>

<div class="zone" class:dragging>
  <span class="corner tl" aria-hidden="true"></span>
  <span class="corner tr" aria-hidden="true"></span>
  <span class="corner bl" aria-hidden="true"></span>
  <span class="corner br" aria-hidden="true"></span>
  <span class="cross" aria-hidden="true"></span>

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
    position: relative;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-8);
    padding: var(--space-10) var(--space-6);
    background: var(--bg-inset);
    transition: background-color var(--dur-fast) var(--ease-standard);
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
    top: var(--space-5);
    left: var(--space-5);
    border-top-width: 2px;
    border-left-width: 2px;
  }
  .tr {
    top: var(--space-5);
    right: var(--space-5);
    border-top-width: 2px;
    border-right-width: 2px;
  }
  .bl {
    bottom: var(--space-5);
    left: var(--space-5);
    border-bottom-width: 2px;
    border-left-width: 2px;
  }
  .br {
    bottom: var(--space-5);
    right: var(--space-5);
    border-bottom-width: 2px;
    border-right-width: 2px;
  }
  .dragging .corner {
    border-color: var(--accent-line);
  }
  .dragging .tl {
    top: var(--space-8);
    left: var(--space-8);
  }
  .dragging .tr {
    top: var(--space-8);
    right: var(--space-8);
  }
  .dragging .bl {
    bottom: var(--space-8);
    left: var(--space-8);
  }
  .dragging .br {
    bottom: var(--space-8);
    right: var(--space-8);
  }
  /* The focus dot at the centre of the viewfinder. */
  .cross {
    position: absolute;
    top: var(--space-5);
    left: 50%;
    width: 1px;
    height: 10px;
    background: var(--line-strong);
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
    font-size: var(--type-micro-size);
    line-height: 14px;
  }
  @media (max-width: 767px) {
    .zone {
      gap: var(--space-6);
      padding: var(--space-8) var(--space-4);
    }
    .samples ul {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .corner {
      width: 20px;
      height: 20px;
    }
    .tl,
    .tr {
      top: var(--space-3);
    }
    .bl,
    .br {
      bottom: var(--space-3);
    }
    .tl,
    .bl {
      left: var(--space-3);
    }
    .tr,
    .br {
      right: var(--space-3);
    }
  }
  .nowrap {
    white-space: nowrap;
  }
</style>
