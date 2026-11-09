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
  import Viewfinder from '$lib/ui/Viewfinder.svelte';
  import { TOUCH_ONLY } from '$lib/shell/shortcuts';
  import { media } from '$lib/design/tokens';

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
  // leads with choosing an image and shows neither hint; a device with a mouse or trackpad shows both. The phone tier
  // is the touch tier whatever reports the pointer, so a phone's copy is the same in every state (round 26: the
  // no-WASM and no-WebGPU phones said "Drop an image… Ctrl V" where the empty one said "Choose an image").
  const touchOnly = new MediaQuery(TOUCH_ONLY, false);
  const phone = new MediaQuery(media.phone, false);
  const touch = $derived(touchOnly.current || phone.current);
</script>

<!-- The viewfinder is the drop target and fills the canvas; the samples sit right under it, at the panel's foot. -->
<div class="zone" class:dragging>
  <div class="finder">
    <Viewfinder {dragging} title={touch ? 'Choose an image to detect objects' : dragging ? 'Release to detect' : 'Drop an image to detect objects'}>
      {#snippet description()}
        {#if touch}JPEG, PNG or WebP, from your photos, camera or files.{:else}JPEG, PNG or WebP. You can also paste from the clipboard with <span class="nowrap"><Kbd keys="Mod+V" />.</span>{/if}
      {/snippet}
      {#snippet actions()}
        <Button variant="primary" icon={Upload} onclick={onbrowse}>{touch ? 'Choose an image' : 'Browse files'}</Button>
        <Button icon={Images} onclick={onpick}>Pick from a dataset</Button>
      {/snippet}
    </Viewfinder>
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
  /* The drop target: a sparse state's compact composition (DESIGN_DIRECTION 8) at every width, so the samples under it
     sit in the first screen. Beside the rail (from 1024 px) it is 480 px where the window has room
     for it and the samples under the page header (about 420 px of chrome), and gives way in a shorter one. Its message
     is centred between the corners (ui/Viewfinder). */
  .finder {
    display: grid;
    width: 100%;
    flex: none;
    height: clamp(280px, calc(100dvh - 420px), 480px);
    --vf-pad: var(--space-8) var(--space-6);
  }
  .zone.dragging {
    background: color-mix(in srgb, var(--bg-inset), var(--accent-solid) 6%);
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
  /* Tablets: no rail beside it, so the whole empty frame (the prompt and the sample strip) is capped at about 360 px
     and Thresholds, Classes and Timing start in the first screen. The finder is as tall as its message and buttons;
     the frame grows only once an image is loaded (Playground's stage). */
  @media (min-width: 768px) and (max-width: 1023px) {
    .zone {
      padding-bottom: var(--space-4);
    }
    .finder {
      height: auto;
      --vf-pad: var(--space-6);
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
      --vf-pad: var(--space-8) var(--space-4);
    }
    .samples ul {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
  }
</style>
