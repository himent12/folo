<script lang="ts" module>
  export type AugKey = 'mosaic' | 'mixup' | 'hsv' | 'fliplr' | 'scale' | 'translate';
</script>

<script lang="ts">
  /**
   * What the augmentation settings do, shown on real images from the dataset: four seeded random draws with the
   * boxes transformed along with the pixels. Focusing one control isolates that augmentation at its configured
   * strength so its effect is obvious.
   */
  import type { Augment, DatasetImage as Img } from '$lib/api/types';
  import { rng } from '$lib/api/mock/rng';
  import DatasetImage from '$lib/detect/DatasetImage.svelte';

  let {
    datasetId,
    images,
    classes,
    augment,
    focus = null,
    seed = 1,
  }: { datasetId: string; images: Img[]; classes: string[]; augment: Augment; focus?: AugKey | null; seed?: number } = $props();

  interface Draw {
    mosaic: boolean;
    tiles: Img[];
    flip: boolean;
    scale: number;
    tx: number;
    ty: number;
    hue: number;
    sat: number;
    val: number;
    mix: Img | null;
    mixAlpha: number;
  }

  const on = (k: AugKey) => focus === null || focus === k;

  const draws = $derived.by((): Draw[] => {
    if (images.length === 0) return [];
    const a = augment;
    return [0, 1, 2, 3].map((i) => {
      const r = rng(seed * 7919 + i * 104729);
      const pickImg = () => images[Math.floor(r() * images.length)] as Img;
      const base = images[i % images.length] as Img;
      const mosaic = on('mosaic') && (focus === 'mosaic' ? a.mosaic > 0 : r() < a.mosaic);
      const sym = () => r() * 2 - 1;
      return {
        mosaic,
        tiles: mosaic ? [base, pickImg(), pickImg(), pickImg()] : [base],
        flip: on('fliplr') && (focus === 'fliplr' ? a.fliplr > 0 && i % 2 === 1 : r() < a.fliplr),
        scale: on('scale') ? 1 + (focus === 'scale' ? [-1, -0.5, 0.5, 1][i]! : sym()) * a.scale * 0.6 : 1,
        tx: on('translate') ? (focus === 'translate' ? [-1, 1, -1, 1][i]! : sym()) * a.translate : 0,
        ty: on('translate') ? (focus === 'translate' ? [-1, -1, 1, 1][i]! : sym()) * a.translate : 0,
        hue: on('hsv') ? (focus === 'hsv' ? [-1, -0.5, 0.5, 1][i]! : sym()) * a.hsv[0] * 360 : 0,
        sat: on('hsv') ? 1 + (focus === 'hsv' ? [1, -0.6, 0.6, -1][i]! : sym()) * a.hsv[1] : 1,
        val: on('hsv') ? 1 + (focus === 'hsv' ? [0.6, -0.6, -1, 1][i]! : sym()) * a.hsv[2] * 0.7 : 1,
        mix: on('mixup') && (focus === 'mixup' ? a.mixup > 0 : r() < a.mixup) ? pickImg() : null,
        mixAlpha: 0.5,
      };
    });
  });
</script>

{#snippet tile(img: Img)}
  <!-- crop-ok: a training crop, so the boxes cut at its edge are the ones the trainer clips too (DESIGN_DIRECTION §2.20). -->
  <DatasetImage
    datasetId={datasetId}
    imageId={img.id}
    width={img.width}
    height={img.height}
    thumb={320}
    boxes={img.boxes}
    {classes}
    compact
    fit="cover"
  />
{/snippet}

<div class="grid" aria-label="Augmentation preview: four random training samples" role="img">
  {#each draws as d, i (i)}
    <div class="frame">
      <div class="view">
        <div
          class="canvas"
          style:transform="translate({d.tx * 100}%, {d.ty * 100}%) scale({d.scale}) scaleX({d.flip ? -1 : 1})"
          style:filter="hue-rotate({d.hue}deg) saturate({d.sat}) brightness({d.val})"
        >
          {#if d.mosaic}
            <div class="mosaic">
              {#each d.tiles as t, k (k)}<div class="cell">{@render tile(t)}</div>{/each}
            </div>
          {:else}
            {@render tile(d.tiles[0] as Img)}
          {/if}
          {#if d.mix}<div class="mix" style:opacity={d.mixAlpha}>{@render tile(d.mix)}</div>{/if}
        </div>
      </div>
      <!-- What this draw applied, in its own strip under the image so it never covers a box. -->
      <span class="tag" aria-hidden="true">
        {#each [d.mosaic && 'mosaic', d.flip && 'flip', Math.abs(d.scale - 1) > 0.02 && `×${d.scale.toFixed(2)}`, d.mix && 'mixup'].filter(Boolean) as part, i (i)}{#if i > 0}<span class="sep">·</span>{/if}{part}{:else}original{/each}
      </span>
    </div>
  {:else}
    {#each [0, 1, 2, 3] as i (i)}<div class="frame empty"><div class="view"></div><span class="tag">&nbsp;</span></div>{/each}
  {/each}
</div>

<style>
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-2);
  }
  .frame {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    border-radius: var(--radius-md);
    border: 1px solid var(--line-subtle);
    background: var(--bg-raised);
  }
  .view {
    position: relative;
    aspect-ratio: 1;
    overflow: hidden;
    background:
      repeating-linear-gradient(45deg, var(--bg-inset) 0 6px, transparent 6px 12px),
      var(--bg-raised);
  }
  .canvas {
    position: absolute;
    inset: 0;
    transition:
      transform var(--dur-chart) var(--ease-standard),
      filter var(--dur-chart) var(--ease-standard);
  }
  .mosaic {
    position: absolute;
    inset: 0;
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
  }
  .cell {
    position: relative;
    overflow: hidden;
  }
  .mix {
    position: absolute;
    inset: 0;
  }
  .tag {
    flex: 1;
    padding: var(--space-1) var(--space-2);
    border-top: 1px solid var(--line-subtle);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    font-variant-numeric: tabular-nums;
  }
  .sep {
    margin: 0 var(--space-1);
    color: var(--text-tertiary);
  }
</style>
