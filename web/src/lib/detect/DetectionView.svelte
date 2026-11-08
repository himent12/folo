<script lang="ts" module>
  export interface ViewBox {
    class: number;
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    score?: number;
  }
</script>

<script lang="ts">
  /**
   * An image with detection boxes drawn over it in screen pixels (crisp at any size). Boxes: a 2 px class-colour
   * stroke inside a 1 px dark halo; label chips with the class name (and score) sit on the top-left corner.
   * Ground truth, when given, is drawn dashed underneath the predictions.
   */
  import { classColor } from '$lib/design/tokens';

  interface Props {
    src: string | null;
    width: number;
    height: number;
    boxes?: ViewBox[];
    truth?: ViewBox[];
    classes: readonly string[];
    labels?: boolean;
    scores?: boolean;
    /** Only show these classes (null = all). */
    only?: ReadonlySet<number> | null;
    /** Index into `boxes` to emphasise (e.g. hovered in a list). */
    active?: number | null;
    onhover?: (i: number | null) => void;
    fit?: 'contain' | 'cover';
    /** Thin strokes and no chips, for thumbnails. */
    compact?: boolean;
    alt?: string;
    /** Called once the image has loaded (or failed). */
    onload?: (ok: boolean) => void;
  }

  let {
    src,
    width,
    height,
    boxes = [],
    truth = [],
    classes,
    labels = true,
    scores = true,
    only = null,
    active = null,
    onhover,
    fit = 'contain',
    compact = false,
    alt = '',
    onload,
  }: Props = $props();

  let cw = $state(0);
  let ch = $state(0);
  let loaded = $state(false);

  const geo = $derived.by(() => {
    if (!cw || !ch || !width || !height) return null;
    const s = fit === 'contain' ? Math.min(cw / width, ch / height) : Math.max(cw / width, ch / height);
    return { s, ox: (cw - width * s) / 2, oy: (ch - height * s) / 2 };
  });

  const shown = (b: ViewBox) => !only || only.has(b.class);
  const name = (c: number) => classes[c] ?? `class ${c}`;

  $effect(() => {
    void src;
    loaded = false;
  });
</script>

<div class="view" class:compact bind:clientWidth={cw} bind:clientHeight={ch}>
  {#if src}
    <img
      {src}
      {alt}
      class:loaded
      style:object-fit={fit}
      draggable="false"
      onload={() => {
        loaded = true;
        onload?.(true);
      }}
      onerror={() => onload?.(false)}
    />
  {/if}
  {#if geo}
    <div class="layer" aria-hidden="true">
      {#each truth as b, i (i)}
        {#if shown(b)}
          <div
            class="box truth"
            style:--c={classColor(b.class)}
            style:left="{geo.ox + b.x1 * geo.s}px"
            style:top="{geo.oy + b.y1 * geo.s}px"
            style:width="{(b.x2 - b.x1) * geo.s}px"
            style:height="{(b.y2 - b.y1) * geo.s}px"
          ></div>
        {/if}
      {/each}
      {#each boxes as b, i (i)}
        {#if shown(b)}
          {@const top = geo.oy + b.y1 * geo.s}
          <!-- svelte-ignore a11y_no_static_element_interactions -->
          <div
            class="box"
            class:active={active === i}
            class:dim={active !== null && active !== i}
            style:--c={classColor(b.class)}
            style:left="{geo.ox + b.x1 * geo.s}px"
            style:top="{top}px"
            style:width="{(b.x2 - b.x1) * geo.s}px"
            style:height="{(b.y2 - b.y1) * geo.s}px"
            onpointerenter={onhover ? () => onhover(i) : undefined}
            onpointerleave={onhover ? () => onhover(null) : undefined}
          >
            {#if labels && !compact}
              <span class="chip" class:inside={top < 18}>
                {name(b.class)}{#if scores && b.score !== undefined}<span class="score">{b.score.toFixed(2)}</span>{/if}
              </span>
            {/if}
          </div>
        {/if}
      {/each}
    </div>
  {/if}
</div>

<style>
  .view {
    position: relative;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: var(--bg-inset);
    user-select: none;
  }
  img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    max-width: none;
    opacity: 0;
    transition: opacity var(--dur-base) var(--ease-standard);
  }
  img.loaded {
    opacity: 1;
  }
  .layer {
    position: absolute;
    inset: 0;
    pointer-events: none;
  }
  .box {
    position: absolute;
    border: 2px solid var(--c);
    box-shadow:
      0 0 0 1px var(--class-halo),
      inset 0 0 0 1px var(--class-halo);
    pointer-events: auto;
    transition:
      opacity var(--dur-fast) var(--ease-standard),
      background-color var(--dur-fast) var(--ease-standard);
  }
  .box.active {
    background: color-mix(in srgb, var(--c) 18%, transparent);
    z-index: 2;
  }
  .box.dim {
    opacity: 0.35;
  }
  .box.truth {
    border: 1.5px dashed var(--c);
    box-shadow: none;
    opacity: 0.85;
    pointer-events: none;
  }
  .compact .box {
    border-width: 1.5px;
    box-shadow: 0 0 0 1px var(--class-halo);
  }
  .compact .box.truth {
    border-width: 1px;
  }
  .chip {
    position: absolute;
    left: -2px;
    bottom: 100%;
    margin-bottom: 1px;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 16px;
    padding: 0 5px;
    border-radius: 3px 3px 3px 0;
    background: var(--c);
    color: var(--class-chip-text);
    font-size: 11px;
    line-height: 1;
    font-weight: var(--weight-medium);
    white-space: nowrap;
    box-shadow: 0 0 0 1px var(--class-halo);
  }
  .chip.inside {
    bottom: auto;
    top: -2px;
    margin: 0;
    border-radius: 0 0 3px 0;
  }
  .score {
    font-family: var(--font-mono);
    font-size: 10px;
    opacity: 0.8;
  }
</style>
