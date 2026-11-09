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
   * An image with detection boxes drawn over it in screen pixels (crisp at any size). Two styles, one per job:
   * - Full size (the inspector, the Playground): a 2 px class-colour stroke wrapped in a dark halo, 1.5 px outside and
   *   1 px inside, so a stroke holds on any colour under it; label chips with the class name (and score) sit on the
   *   top-left corner, or elsewhere next to their own box (see ./chips).
   * - Compact thumbnails: a 1 px stroke in a 1 px halo and no chips, so a small object is not buried under its frame.
   *   Only a hovered or selected box gets the full-size weight. A thumbnail that draws boxes shows the whole image
   *   (`fit="contain"`, letterboxed on bg.inset): a cropped one would cut boxes in half at its edge
   *   (./thumbFit.test.ts).
   * Ground truth, when given, is dashed (predictions are solid) and drawn underneath: in the class colour at full size,
   * neutral and just outside the prediction's halo in compact thumbnails (where the two would otherwise coincide).
   * A selection: the selected box takes a 3 px accent stroke and its chip an accent edge; every other box, chip and
   * truth mark recedes to 30 %; a selected box under 32 px on screen is also framed by lime viewfinder corners (the
   * product's mark), so it cannot be lost among bigger ones.
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
  // Small views (phones) get smaller chips. Boxes under 20 px on screen keep their label for hover/selection (a chip
  // would cover the object); their chips are placed and clamped but reserve no slot, so they never push others around.
  const small = $derived(cw > 0 && cw < 480);
  const chipH = $derived(small ? 14 : CHIP_H);
  const tiny = (b: ViewBox) => !!geo && Math.min(b.x2 - b.x1, b.y2 - b.y1) * geo.s < 20;
  /** Under 32 px on screen: a selected box this small also gets viewfinder corners. */
  const small32 = (b: ViewBox) => !!geo && Math.min(b.x2 - b.x1, b.y2 - b.y1) * geo.s < 32;
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
              w: Math.ceil(name(b.class).length * (small ? 5.8 : 6.4) + (small ? 10 : 12) + (scores && b.score !== undefined ? (small ? 28 : 32) : 0)),
              bw: (b.x2 - b.x1) * g.s,
              bh: (b.y2 - b.y1) * g.s,
              // A tiny box's label shows only on hover / selection: placed and clamped, but it reserves no slot.
              reserve: !tiny(b),
            }
          : null,
      ),
      {
        minX: Math.max(0, g.ox) + 4,
        maxX: Math.min(cw, g.ox + width * g.s) - 4,
        minY: Math.max(0, g.oy) + 2,
        maxY: Math.min(ch, g.oy + height * g.s) - 2,
      },
      chipH,
    );
  });

  $effect(() => {
    void src;
    loaded = false;
  });
</script>

<div class="view" class:compact class:small bind:clientWidth={cw} bind:clientHeight={ch}>
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
            class:fade={selected !== null}
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
            class:dim={active !== null && active !== i && selected !== i}
            class:recede={selected !== null && selected !== i && active !== i}
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
            {#if selected === i && small32(b)}<span class="vf"></span>{/if}
            {#if labels && !compact && ((!chips[i]?.hidden && !tiny(b)) || active === i || selected === i)}
              {@const c = chips[i]}
              <!-- With viewfinder corners, the chip steps 8 px further from the box so it never covers them. -->
              {@const clear = selected === i && small32(b) && !c?.inside ? (c?.side === 'below' ? 8 : -8) : 0}
              {#if c && c.leader > 0}
                <!-- Hung below its box: a leader from the box's bottom stroke keeps ownership readable. -->
                {@const bh = (b.y2 - b.y1) * geo.s}
                <span class="leader" style:left="{Math.max(0, c.dx + 6) - 2}px" style:top="{bh + 2}px" style:height="{c.dy - bh - 4 + clear}px"></span>
              {/if}
              <span class="chip" class:inside={c?.inside} class:below={c?.side === 'below'} style:left="{(c?.dx ?? 0) - 2}px" style:top="{(c?.dy ?? -17) - 2 + clear}px">
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
  /* Strokes sit outside the box (content-box plus a negative margin of the border width), so the object inside stays
     fully visible: a small circle no longer reads as a filled square under a 2 px border. The halo wraps the stroke on
     both sides (1.5 px out, 1 px in): a low-contrast pair (violet over an orange disc) still reads at 1x. */
  .box {
    position: absolute;
    box-sizing: content-box;
    margin: -2px;
    border: 2px solid var(--c);
    box-shadow:
      0 0 0 1.5px var(--class-halo),
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
    /* Selected: a 3 px accent stroke (lime reads on any photo inside its dark halo), drawn above everything else. */
    margin: -3px; /* grid-ok: equals the stroke width */
    border: 3px solid var(--accent-solid);
    box-shadow:
      0 0 0 1px var(--class-halo),
      0 0 0 3px color-mix(in srgb, var(--class-halo) 50%, transparent);
    z-index: 5;
  }
  /* The selected box's chip: an accent edge, so its label is found as fast as its box. */
  .box.selected .chip {
    z-index: 6;
    box-shadow:
      0 0 0 1.5px var(--accent-solid),
      0 0 0 2.5px var(--class-halo);
  }
  /* A selected box under 32 px on screen: lime viewfinder corners 4 px outside its stroke, in a dark halo. */
  .vf {
    position: absolute;
    inset: -8px;
    pointer-events: none;
    --arm: 6px;
    background:
      linear-gradient(var(--accent-solid) 0 0) left top / var(--arm) 2px,
      linear-gradient(var(--accent-solid) 0 0) left top / 2px var(--arm),
      linear-gradient(var(--accent-solid) 0 0) right top / var(--arm) 2px,
      linear-gradient(var(--accent-solid) 0 0) right top / 2px var(--arm),
      linear-gradient(var(--accent-solid) 0 0) left bottom / var(--arm) 2px,
      linear-gradient(var(--accent-solid) 0 0) left bottom / 2px var(--arm),
      linear-gradient(var(--accent-solid) 0 0) right bottom / var(--arm) 2px,
      linear-gradient(var(--accent-solid) 0 0) right bottom / 2px var(--arm);
    background-repeat: no-repeat;
    filter: drop-shadow(0 0 1px var(--class-halo)) drop-shadow(0 0 1px var(--class-halo));
  }
  /* Keep context readable: fade the stroke, never the label. Hovering one box fades the others' strokes a little... */
  .box.dim {
    border-color: color-mix(in srgb, var(--c) 55%, transparent);
    box-shadow: 0 0 0 1.5px color-mix(in srgb, var(--class-halo) 50%, transparent);
  }
  /* ...and with a selection the rest of the scene steps back to 30 %, strokes, chips, leaders and truth alike, so a
     small selected detection is never lost among full-strength chips. Hovering one of them brings it back. */
  .box.recede {
    border-color: color-mix(in srgb, var(--c) 30%, transparent);
    box-shadow: 0 0 0 1.5px color-mix(in srgb, var(--class-halo) 30%, transparent);
  }
  .box.recede .chip,
  .box.recede .leader {
    opacity: 0.3;
  }
  .box.truth.fade {
    opacity: 0.3;
  }
  /* Touch: a box smaller than a fingertip still gets a 32 px target, centred on it. */
  @media (pointer: coarse) {
    .box:not(.truth)::before {
      content: '';
      position: absolute;
      left: 50%;
      top: 50%;
      width: max(100%, 32px);
      height: max(100%, 32px);
      transform: translate(-50%, -50%);
    }
  }
  .box.truth {
    margin: -1.5px; /* grid-ok: equals the stroke width */
    border: 1.5px dashed var(--c);
    box-shadow: none;
    /* A dark halo that follows the dashes keeps light classes (near-white) visible on bright photos. */
    filter: drop-shadow(0 0 0.75px var(--class-halo)) drop-shadow(0 0 0.75px var(--class-halo));
    opacity: 0.85;
    pointer-events: none;
  }
  /* Thumbnails: the compact style, a 1 px stroke in a 1 px halo, so a small object is not buried under nested frames;
     the halo still holds every class edge against a pale photo. */
  .compact .box {
    margin: -1px;
    border-width: 1px;
    box-shadow: 0 0 0 1px var(--class-halo);
  }
  /* The strongest outline in a thumbnail is the hovered or selected object's: the full-size weight. */
  .compact .box.active,
  .compact .box.selected {
    margin: -2px;
    border-width: 2px;
    box-shadow: 0 0 0 1.5px var(--class-halo);
  }
  .compact .box.selected {
    border-color: var(--accent-solid);
  }
  /* Thumbnails: a prediction and its matching truth share a class colour and nearly the same rectangle, so a coloured
     dash would vanish under the solid stroke. Truth is neutral there (white 1 px dashes) and sits right outside the
     prediction's stroke and halo, so a match reads as one solid line with a dashed shadow and a miss as a lone ring. */
  .compact .box.truth {
    margin: 0;
    border: 0;
    outline: 1px dashed #ffffff;
    outline-offset: 2px;
    opacity: 1;
  }
  .chip {
    /* left/top come from the placement pass, relative to the box's own top-left (its stroke lies just outside). */
    position: absolute;
    z-index: 1;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    height: 16px;
    padding: 0 4px;
    border-radius: 3px 3px 3px 0;
    background: var(--c);
    color: var(--class-chip-text);
    font-size: var(--type-xs-size); /* small: image chip */
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
  .chip.below {
    border-radius: 3px;
  }
  .small .chip {
    height: 14px;
    padding: 0 4px;
    gap: 4px;
    font-size: var(--type-micro-size); /* small: image chip (views under 480 px) */
  }
  .small .score {
    font-size: var(--type-micro-size); /* small: image chip */
  }
  .score {
    font-variant-numeric: tabular-nums;
    font-size: var(--type-micro-size); /* small: image chip (the score after the name) */
    opacity: 0.8;
  }
</style>
