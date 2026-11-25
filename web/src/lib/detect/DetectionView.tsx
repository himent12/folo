/**
 * An image with detection boxes drawn over it in screen pixels (crisp at any size). Two styles, one per job:
 * - Full size (the inspector, the Playground): a 2 px class-colour stroke wrapped in a dark halo, 1.5 px outside and
 *   1 px inside, so a stroke holds on any colour under it; label chips with the class name (and score) sit on the
 *   top-left corner, or elsewhere next to their own box (see ./chips).
 * - Compact thumbnails: a 1 px stroke in a 1 px halo and no chips, so a small object is not buried under its frame.
 *   Only a hovered or selected box gets the full-size weight. A thumbnail that draws boxes shows the whole image
 *   (`fit="contain"`, letterboxed on bg.inset): a cropped one would cut boxes in half at its edge.
 * - The edges: a contained image clips only itself, never the overlay, so a box on the image's edge keeps its outer
 *   stroke and halo (up to 4 px outside it); a frame that shows the image edge to edge (the phone Playground) insets
 *   it by 4 px so that halo stays on screen. A cropped (`cover`) view clips everything at its edge.
 * Ground truth, when given, is dashed (predictions are solid) and drawn underneath: in the class colour at full size,
 * neutral and just outside the prediction's halo in compact thumbnails (where the two would otherwise coincide).
 * A selection: the selected box takes a 3 px accent stroke and its chip an accent edge; every other box, chip and
 * truth mark recedes to 30 %; a selected box under 32 px on screen is also framed by lime viewfinder corners (the
 * product's mark), so it cannot be lost among bigger ones.
 */
import { computed, css, defaults, effect, Index, onCleanup, signal } from 'fw';
import { classColor } from '$lib/design/tokens';
import { CHIP_H, placeChips } from './chips';

export interface ViewBox {
  class: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score?: number;
}

export interface DetectionViewProps {
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

const NO_BOXES: ViewBox[] = [];

export default function DetectionView(props: DetectionViewProps) {
  const p = defaults(props, {
    boxes: NO_BOXES,
    truth: NO_BOXES,
    labels: true,
    scores: true,
    only: null,
    active: null,
    selected: null,
    fit: 'contain',
    compact: false,
    alt: '',
  });

  const cw = signal(0);
  const ch = signal(0);
  const loaded = signal(false);

  const geo = computed(() => {
    const w = cw.get();
    const h = ch.get();
    const { width, height, fit } = p;
    if (!w || !h || !width || !height) return null;
    const s = fit === 'contain' ? Math.min(w / width, h / height) : Math.max(w / width, h / height);
    return { s, ox: (w - width * s) / 2, oy: (h - height * s) / 2 };
  });

  const shown = (b: ViewBox) => {
    const only = p.only;
    return !only || only.has(b.class);
  };
  // Small views (phones) get smaller chips, inside their box when they fit there, and never over another box (./chips).
  // Boxes under 20 px on screen keep their label for a tap (selection) or hover (a chip would cover the object), as
  // does a box whose chip has no free slot; their chips are placed and clamped but reserve no slot, so they never push
  // others around.
  const small = computed(() => cw.get() > 0 && cw.get() < 480);
  const chipH = computed(() => (small.get() ? 14 : CHIP_H));
  const tiny = (b: ViewBox) => {
    const g = geo.get();
    return !!g && Math.min(b.x2 - b.x1, b.y2 - b.y1) * g.s < 20;
  };
  /** Under 32 px on screen: a selected box this small also gets viewfinder corners. */
  const small32 = (b: ViewBox) => {
    const g = geo.get();
    return !!g && Math.min(b.x2 - b.x1, b.y2 - b.y1) * g.s < 32;
  };
  const name = (c: number) => p.classes[c] ?? `class ${c}`;

  /** Label chips placed in screen pixels (see ./chips). Widths are estimated from the text (11 px Geist). */
  const chips = computed(() => {
    const g = geo.get();
    if (!g || !p.labels || p.compact) return [];
    const sm = small.get();
    const scores = p.scores;
    const width = p.width;
    const height = p.height;
    const w = cw.get();
    const h = ch.get();
    return placeChips(
      p.boxes.map((b) =>
        shown(b)
          ? {
              x: g.ox + b.x1 * g.s,
              y: g.oy + b.y1 * g.s,
              w: Math.ceil(name(b.class).length * (sm ? 5.8 : 6.4) + (sm ? 10 : 12) + (scores && b.score !== undefined ? (sm ? 28 : 32) : 0)),
              bw: (b.x2 - b.x1) * g.s,
              bh: (b.y2 - b.y1) * g.s,
              // A tiny box's label shows only on hover / selection: placed and clamped, but it reserves no slot.
              reserve: !tiny(b),
            }
          : null,
      ),
      {
        minX: Math.max(0, g.ox) + 4,
        maxX: Math.min(w, g.ox + width * g.s) - 4,
        minY: Math.max(0, g.oy) + 2,
        maxY: Math.min(h, g.oy + height * g.s) - 2,
      },
      chipH.get(),
      // Phones: inside the box when the chip fits, outside otherwise, and never over another box (./chips).
      { small: sm },
    );
  });

  // A new image is not loaded yet. Through a computed: only a new `src` value counts, not a re-read of the parent's
  // expression for it (Svelte memoises a prop expression; DatasetImage's `src` re-reads its phase on load).
  const src = computed(() => p.src);
  effect(() => {
    void src.get();
    loaded.set(false);
  });

  let layerEl: HTMLDivElement | undefined;

  function boxIndex(el: EventTarget | null): number | null {
    const node = el instanceof Element ? el : el instanceof Node ? el.parentElement : null;
    const box = node?.closest<HTMLElement>('[data-box]');
    if (!box || (layerEl && !layerEl.contains(box))) return null;
    const i = Number(box.dataset.box);
    return Number.isInteger(i) ? i : null;
  }

  function onLayerPointerOver(e: PointerEvent) {
    if (!p.onhover) return;
    const to = boxIndex(e.target);
    if (to === null || to === boxIndex(e.relatedTarget)) return;
    p.onhover(to);
  }

  function onLayerPointerOut(e: PointerEvent) {
    if (!p.onhover) return;
    if (boxIndex(e.target) === null) return;
    if (boxIndex(e.relatedTarget) === null) p.onhover(null);
  }

  function onLayerClick(e: MouseEvent) {
    if (!p.onselect) return;
    const i = boxIndex(e.target);
    if (i !== null) p.onselect(i);
  }

  return (
    <div class="view" class:compact={p.compact} class:small={small.get()} class:cover={p.fit === 'cover'} bind:clientWidth={cw} bind:clientHeight={ch}>
      {p.src ? (
        <img
          ref={cancelOnRemove}
          src={p.src}
          alt={p.alt}
          decoding="async"
          class:loaded={loaded.get()}
          style:object-fit={p.fit}
          draggable="false"
          onload={() => {
            loaded.set(true);
            p.onload?.(true);
          }}
          onerror={(e) => {
            if ((e.target as HTMLImageElement).hasAttribute('src')) p.onload?.(false);
          }}
        />
      ) : null}
      {geo.get() ? (
        <div
          class="layer"
          ref={(el) => (layerEl = el)}
          aria-hidden="true"
          onpointerover={p.onhover ? onLayerPointerOver : undefined}
          onpointerout={p.onhover ? onLayerPointerOut : undefined}
          onclick={p.onselect ? onLayerClick : undefined}
        >
          <Index each={p.truth}>
            {(b) => (
              <>
                {shown(b()) ? (
                  <div
                    class="box truth"
                    class:fade={p.selected !== null}
                    style={{
                      '--c': classColor(b().class),
                      left: `${geo.get()!.ox + b().x1 * geo.get()!.s}px`,
                      top: `${geo.get()!.oy + b().y1 * geo.get()!.s}px`,
                      width: `${(b().x2 - b().x1) * geo.get()!.s}px`,
                      height: `${(b().y2 - b().y1) * geo.get()!.s}px`,
                    }}
                  ></div>
                ) : null}
              </>
            )}
          </Index>
          <Index each={p.boxes}>
            {(b, i) => {
              const top = () => geo.get()!.oy + b().y1 * geo.get()!.s;
              const c = () => chips.get()[i];
              // With viewfinder corners, the chip steps 8 px further from the box so it never covers them.
              const corners = () => p.selected === i && small32(b()) && !c()?.inside;
              const clear = () => (corners() && c()?.side !== 'beside' ? (c()?.side === 'below' ? 8 : -8) : 0);
              const clearX = () => (corners() && c()?.side === 'beside' ? (c()!.dx < 0 ? -8 : 8) : 0);
              return (
                <>
                  {shown(b()) ? (
                    <div
                      class="box"
                      class:active={p.active === i}
                      class:selected={p.selected === i}
                      class:dim={p.active !== null && p.active !== i && p.selected !== i}
                      class:recede={p.selected !== null && p.selected !== i && p.active !== i}
                      data-box={i}
                      style={{
                        '--c': classColor(b().class),
                        left: `${geo.get()!.ox + b().x1 * geo.get()!.s}px`,
                        top: `${top()}px`,
                        width: `${(b().x2 - b().x1) * geo.get()!.s}px`,
                        height: `${(b().y2 - b().y1) * geo.get()!.s}px`,
                      }}
                    >
                      {p.selected === i && small32(b()) ? <span class="vf"></span> : null}
                      {p.labels && !p.compact && ((!c()?.hidden && !tiny(b())) || p.active === i || p.selected === i) ? (
                        <>
                          {c() && c()!.leader > 0 ? (
                            /* Hung below its box: a leader from the box's bottom stroke keeps ownership readable. */
                            /* Each binding reads the chip again (`?.`): the list can shrink under a binding that has not
                               re-run yet (a threshold drag), so the chip checked above may be gone. */
                            <span
                              class="leader"
                              style:left={`${Math.max(0, (c()?.dx ?? 0) + 6) - 2}px`}
                              style:top={`${(b().y2 - b().y1) * geo.get()!.s + 2}px`}
                              style:height={`${(c()?.dy ?? 0) - (b().y2 - b().y1) * geo.get()!.s - 4 + clear()}px`}
                            ></span>
                          ) : null}
                          <span
                            class="chip"
                            class:inside={c()?.inside}
                            class:below={c()?.side === 'below'}
                            style:left={`${(c()?.dx ?? 0) - 2 + clearX()}px`}
                            style:top={`${(c()?.dy ?? -17) - 2 + clear()}px`}
                          >
                            {name(b().class)}
                            {p.scores && b().score !== undefined ? <span class="score">{b().score!.toFixed(2)}</span> : null}
                          </span>
                        </>
                      ) : null}
                    </div>
                  ) : null}
                </>
              );
            }}
          </Index>
        </div>
      ) : null}
    </div>
  );
}

css`
  .view {
    position: relative;
    width: 100%;
    height: 100%;
    background: var(--bg-inset);
    user-select: none;
  }
  /* Cropped: boxes run past the edge, so the view clips them with the image. */
  .view.cover {
    overflow: hidden;
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
`;

/**
 * An `<img>` taken out of the page goes on loading in Chrome (a fast scroll through a virtualised grid queued thousands
 * of thumbnails, and the image the user then opened waited behind them, stress s1-datasets-2); without its `src` it
 * stops. Registered on the owner that created the element, so it runs when that branch or the view goes.
 */
export function cancelOnRemove(el: HTMLImageElement): void {
  onCleanup(() => el.removeAttribute('src'));
}
