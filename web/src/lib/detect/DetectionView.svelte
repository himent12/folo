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
  import { CHIP_H, placeChips } from './chips';

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
    /** Index into `boxes` that is selected (persistent): drawn with a selection outline and a class-colour fill. */
    selected?: number | null;
    /** Click / tap on a box. Pointer only: keyboard users select from a list of the same boxes. */
    onselect?: (i: number) => void;
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
    selected = null,
    onselect,
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

  /** Label chips placed in screen pixels (see ./chips). Widths are estimated from the text (11 px Geist). */
  const chips = $derived.by(() => {
    const g = geo;
    if (!g || !labels || compact) return [];
    return placeChips(
      boxes.map((b) =>
        shown(b)
          ? {
              x: g.ox + b.x1 * g.s,
              y: g.oy + b.y1 * g.s,
              w: Math.ceil(name(b.class).length * 6.4 + 12 + (scores && b.score !== undefined ? 32 : 0)),
            }
          : null,
      ),
      {
        minX: Math.max(0, g.ox) + 4,
        maxX: Math.min(cw, g.ox + width * g.s) - 4,
        minY: Math.max(0, g.oy) + 2,
        maxY: Math.min(ch, g.oy + height * g.s) - 2,
      },
    );
  });

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
          <!-- svelte-ignore a11y_no_static_element_interactions, a11y_click_events_have_key_events -->
          <div
            class="box"
            class:active={active === i}
            class:selected={selected === i}
            class:dim={active !== null && active !== i}
            data-box={i}
            style:--c={classColor(b.class)}
            style:left="{geo.ox + b.x1 * geo.s}px"
            style:top="{top}px"
            style:width="{(b.x2 - b.x1) * geo.s}px"
            style:height="{(b.y2 - b.y1) * geo.s}px"
            onpointerenter={onhover ? () => onhover(i) : undefined}
            onpointerleave={onhover ? () => onhover(null) : undefined}
            onclick={onselect ? () => onselect(i) : undefined}
          >
            {#if labels && !compact && (!chips[i]?.hidden || active === i || selected === i)}
              {@const c = chips[i]}
              {#if c && (c.dy < -CHIP_H - 9 || (c.inside && c.dy > 8))}
                <!-- The chip had to move away from its box: a thin leader keeps ownership readable. -->
                <span
                  class="leader"
                  style:left="{Math.max(0, c.dx + 6) - 2}px"
                  style:top="{(c.dy < 0 ? c.dy + CHIP_H : 0) - 2}px"
                  style:height="{c.dy < 0 ? -c.dy - CHIP_H : c.dy}px"
                ></span>
              {/if}
              <span class="chip" class:inside={c?.inside} style:left="{(c?.dx ?? 0) - 2}px" style:top="{(c?.dy ?? -17) - 2}px">
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
  .box.active,
  .box.selected {
    background: color-mix(in srgb, var(--c) calc(var(--class-fill-opacity) * 100%), transparent);
    z-index: 2;
  }
  .box.selected {
    /* Selection ring outside the halo: 2 px in the accent, the same mark as focus and selection elsewhere. */
    outline: 2px solid var(--accent-line);
    outline-offset: 1px;
    z-index: 3;
  }
  .box.dim {
    /* Keep context readable: fade the stroke, never the label. */
    border-color: color-mix(in srgb, var(--c) 55%, transparent);
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--class-halo) 50%, transparent);
  }
  .box.truth {
    border: 1.5px dashed var(--c);
    box-shadow: none;
    /* A dark halo that follows the dashes keeps light classes (near-white) visible on bright photos. */
    filter: drop-shadow(0 0 0.75px var(--class-halo)) drop-shadow(0 0 0.75px var(--class-halo));
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
    /* left/top come from the placement pass (offsets include the 2 px border). */
    position: absolute;
    z-index: 1;
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
  .leader {
    position: absolute;
    width: 1px;
    background: var(--c);
    box-shadow: 0 0 0 1px var(--class-halo);
    pointer-events: none;
  }
  .chip.inside {
    border-radius: 0 0 3px 0;
  }
  .score {
    font-family: var(--font-mono);
    font-size: 10px;
    opacity: 0.8;
  }
</style>
