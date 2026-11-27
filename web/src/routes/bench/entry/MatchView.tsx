/**
 * An image of an entry with what it found against the truth (docs/BENCH.md, Matching): a right detection is a solid
 * line, a wrong one a solid line with a cross on its corner, a true box nothing found a dashed line; each also in its
 * own status ink (right, wrong, missed) in a dark halo, so it reads on any photo and without its colour. A true box a
 * detection found is not drawn again: the solid line is on it. Full size adds chips (class and confidence, or "missed
 * <class>"); a thumbnail keeps 1 px lines and the cross. `active` brings one box forward (the viewer's list).
 */
import { computed, css, effect, For, signal } from 'fw';
import type { Box, BenchDetection } from '$lib/api/types';
import { containFit, edgeRoom, insetFit, placeBox } from '$lib/bench/frame';
import { matchImage } from '$lib/bench/match';
import { placeChips } from '$lib/detect/chips';
import { cancelOnRemove } from '$lib/detect/DetectionView';

/** How a box came out, in the gallery's one encoding (never by colour alone: a solid line, a solid line with a cross, a dashed line). */
export type Outcome = 'right' | 'wrong' | 'missed';

export interface MatchViewProps {
  src: string | null;
  width: number;
  height: number;
  truth: readonly Box[];
  detections: readonly BenchDetection[];
  classes: readonly string[];
  full?: boolean;
  /** `d<i>` for detection i, `t<i>` for true box i. */
  active?: string | null;
  /** Show ground-truth boxes (missed GT in error mode, or all GT boxes when predictions are hidden). Default true. */
  showTruth?: boolean;
  /** Show predicted boxes (TP and FP). Default true. */
  showPred?: boolean;
  /** Highlight TP/FP/FN outcomes with status colors and FP crosses; when false, draws clean GT vs Pred boxes. Default true. */
  showErrors?: boolean;
  alt?: string;
}

interface Mark {
  key: string;
  box: Box;
  outcome: Outcome;
  /** Visual role rendered on the canvas: status outcome when `showErrors` is on, else plain truth/pred. */
  visual: Outcome | 'plain-pred' | 'plain-truth';
  score: number | undefined;
}

export default function MatchView(props: MatchViewProps) {
  const full = () => props.full ?? false;
  const active = () => props.active ?? null;
  const showTruth = () => props.showTruth ?? true;
  const showPred = () => props.showPred ?? true;
  const showErrors = () => props.showErrors ?? true;

  const cw = signal(0);
  const ch = signal(0);
  const loaded = signal(false);
  effect(() => {
    void props.src;
    loaded.set(false);
  });
  // The whole image in the frame, whatever either's shape (a portrait photo in a landscape tile sits between two bands
  // of the well); every box goes through the same fit, so it stays on its object ($lib/bench/frame).
  const m = computed(() => matchImage(props.truth, props.detections));
  /** What is drawn, respecting the GT / Pred / Error layer toggles while keeping default appearance identical. */
  const marks = computed<Mark[]>(() => {
    const mm = m.get();
    const err = showErrors();
    const predOn = showPred();
    const truthOn = showTruth();
    const out: Mark[] = [];
    if (predOn) {
      for (const d of mm.detections) {
        const outcome: Outcome = d.truth === null ? 'wrong' : 'right';
        out.push({
          key: `d${d.index}`,
          box: d.det as Box,
          outcome,
          visual: err ? outcome : 'plain-pred',
          score: d.det.score,
        });
      }
    }
    if (truthOn) {
      props.truth.forEach((t, i) => {
        const missed = mm.found[i] === null;
        if (missed) {
          out.push({ key: `t${i}`, box: t, outcome: 'missed', visual: err ? 'missed' : 'plain-truth', score: undefined });
        } else if (!predOn || !err) {
          // When predictions are hidden or error highlighting is off, show matched GT boxes too so full GT is visible.
          out.push({ key: `t${i}`, box: t, outcome: 'right', visual: 'plain-truth', score: undefined });
        }
      });
    }
    return out;
  });
  // A thumbnail's box marks reach outside it (the CSS below): its line and halo 2 px, a wrong box's cross 5 px, and half
  // a pixel more for a box drawn at the 1 px minimum about its centre (below). The
  // image (and so every box) keeps that much room from the tile's edge when a box on the image comes that close, so
  // none is cut off (issue #382). A box wholly off the image asks for none and is drawn with the plain fit, where it
  // was before (mostly cut off), and the large view, which has room around its image, is drawn as before.
  const on = (b: Box) => b.x2 > 0 && b.y2 > 0 && b.x1 < props.width && b.y1 < props.height;
  const plain = computed(() => containFit(cw.get(), ch.get(), props.width, props.height));
  const room = computed(() => {
    const f = plain.get();
    if (!f || full()) return 0;
    const r = edgeRoom(f, cw.get(), ch.get(), marks.get().flatMap((k) => (on(k.box) ? [{ ...k.box, reach: k.visual === 'wrong' ? 5.5 : 2.5 }] : [])));
    // Never so much that the image is gone (a long strip in a short tile).
    return 4 * r < Math.min(cw.get(), ch.get()) ? r : 0;
  });
  const geo = computed(() => insetFit(cw.get(), ch.get(), props.width, props.height, room.get()));
  const name = (c: number) => props.classes[c] ?? `class ${c}`;
  const chipText = (k: Mark) =>
    k.visual === 'missed'
      ? `missed ${name(k.box.class)}`
      : k.visual === 'plain-truth'
        ? `GT ${name(k.box.class)}`
        : `${name(k.box.class)}${k.score !== undefined && k.score < 1 ? ` ${k.score.toFixed(2)}` : ''}`;
  // Full size: each box's chip placed beside its own box, never over another chip (./lib/detect/chips, as the
  // inspector's); wrong and missed first, since they are what the view is for. A chip with no free slot shows on hover.
  const order = computed(() =>
    marks
      .get()
      .map((k, i) => ({ k, i }))
      .sort((a, b) => Number(a.k.outcome === 'right') - Number(b.k.outcome === 'right')),
  );
  const chips = computed(() => {
    const g = geo.get();
    if (!g || !full()) return [];
    const w = cw.get();
    const h = ch.get();
    const small = w < 480;
    const ord = order.get();
    const placed = placeChips(
      ord.map(({ k }) => ({
        x: g.ox + k.box.x1 * g.s,
        y: g.oy + k.box.y1 * g.s,
        w: Math.ceil(chipText(k).length * (small ? 5.8 : 6.4) + (small ? 10 : 12)) + (k.visual === 'wrong' ? 12 : 0),
        bw: (k.box.x2 - k.box.x1) * g.s,
        bh: (k.box.y2 - k.box.y1) * g.s,
      })),
      { minX: Math.max(0, g.ox) + 4, maxX: Math.min(w, g.ox + props.width * g.s) - 4, minY: Math.max(0, g.oy) + 2, maxY: Math.min(h, g.oy + props.height * g.s) - 2 },
      small ? 14 : 16,
      { small },
    );
    const out: (ReturnType<typeof placeChips>[number] | undefined)[] = [];
    ord.forEach(({ i }, n) => (out[i] = placed[n]));
    return out;
  });

  return (
    <div class="mv" class:full={full()} class:small={cw.get() > 0 && cw.get() < 480} bind:clientWidth={cw} bind:clientHeight={ch}>
      {props.src ? <img ref={cancelOnRemove} src={props.src} alt={props.alt ?? ''} class:loaded={loaded.get()} style:inset={`${room.get()}px`} style:width={`calc(100% - ${2 * room.get()}px)`} style:height={`calc(100% - ${2 * room.get()}px)`} draggable="false" onload={() => loaded.set(true)} /> : null}
      {geo.get() ? (
        <div class="layer" aria-hidden="true">
          <For each={marks.get()} key={(k) => k.key}>
            {(k, n) => {
              // A box under a pixel wide or high is drawn 1 px: on a thumbnail about its centre, so it stays inside the
              // room kept for it; in the large view from its corner, where its chip was placed (as before).
              const at = computed(() => {
                const off = room.get() > 0 && !on(k().box);
                const r = placeBox(off ? plain.get()! : geo.get()!, k().box);
                const w = Math.max(1, r.width);
                const h = Math.max(1, r.height);
                const c = full() ? 0 : 0.5;
                return { left: r.left - (w - r.width) * c, top: r.top - (h - r.height) * c, width: w, height: h };
              });
              const chip = () => chips.get()[n()];
              return (
                <div
                  class={`box ${k().visual}`}
                  class:on={active() === k().key}
                  class:dim={active() !== null && active() !== k().key}
                  style:left={`${at.get().left}px`}
                  style:top={`${at.get().top}px`}
                  style:width={`${at.get().width}px`}
                  style:height={`${at.get().height}px`}
                >
                  {k().visual === 'wrong' ? <span class="x"></span> : null}
                  {full() && chip() && (!chip()!.hidden || active() === k().key) ? (
                    <span class="chip" style:left={`${chip()!.dx - 2 + (k().visual === 'wrong' ? 12 : 0)}px`} style:top={`${chip()!.dy - 2}px`}>
                      {k().visual === 'missed' ? (
                        `missed ${name(k().box.class)}`
                      ) : k().visual === 'plain-truth' ? (
                        `GT ${name(k().box.class)}`
                      ) : (
                        <>
                          {name(k().box.class)}
                          {k().score !== undefined && k().score! < 1 ? <span class="sc">{k().score!.toFixed(2)}</span> : null}
                        </>
                      )}
                    </span>
                  ) : null}
                </div>
              );
            }}
          </For>
        </div>
      ) : null}
    </div>
  );
}

css`
  .mv {
    position: relative;
    width: 100%;
    height: 100%;
    background: var(--bg-inset);
    user-select: none;
    --ok: var(--success-solid);
    --bad: var(--danger-solid);
    --miss: var(--warning-solid);
  }
  img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
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
  /* Thumbnails: 1 px lines in a 1 px dark halo, outside the box so the object stays visible. */
  .box {
    position: absolute;
    box-sizing: content-box;
    margin: -1px;
    border: 1px solid var(--c);
    box-shadow: 0 0 0 1px var(--class-halo);
  }
  .box.right {
    --c: var(--ok);
  }
  .box.wrong {
    --c: var(--bad);
  }
  .box.missed,
  .box.plain-truth {
    --c: var(--miss);
    border-style: dashed;
    box-shadow: none;
    filter: drop-shadow(0 0 0.75px var(--class-halo)) drop-shadow(0 0 0.75px var(--class-halo));
  }
  .box.plain-truth {
    --c: var(--info-solid);
  }
  .box.plain-pred {
    --c: var(--accent-solid);
  }
  /* A wrong detection's cross: a small square on its top-left corner, the cross cut in the halo's ink. */
  .x {
    position: absolute;
    left: -4px;
    top: -4px;
    width: 8px;
    height: 8px;
    border-radius: 2px;
    background:
      linear-gradient(45deg, transparent 42%, var(--class-halo) 42% 58%, transparent 58%),
      linear-gradient(-45deg, transparent 42%, var(--class-halo) 42% 58%, transparent 58%),
      var(--bad);
    box-shadow: 0 0 0 1px var(--class-halo);
  }
  /* Full size: 2 px lines in a 1.5 px halo, chips with the class (and confidence) on the top-left corner. */
  .full .box {
    margin: -2px;
    border-width: 2px;
    box-shadow:
      0 0 0 1.5px var(--class-halo),
      inset 0 0 0 1px var(--class-halo);
  }
  .full .box.missed,
  .full .box.plain-truth {
    box-shadow: none;
  }
  .full .x {
    left: -8px;
    top: -8px;
    width: 14px;
    height: 14px;
  }
  .chip {
    position: absolute;
    z-index: 1;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    height: 16px;
    padding: 0 4px;
    border-radius: 3px;
    background: var(--c);
    color: var(--class-chip-text);
    font-size: var(--type-xs-size); /* small: image chip */
    line-height: 1;
    font-weight: var(--weight-medium);
    white-space: nowrap;
    box-shadow: 0 0 0 1px var(--class-halo);
  }
  .small .chip {
    height: 14px;
    font-size: var(--type-micro-size); /* small: image chip (views under 480 px) */
  }
  .sc {
    font-variant-numeric: tabular-nums;
    opacity: 0.85;
  }
  .box.on {
    z-index: 3;
  }
  .full .box.on {
    margin: -3px; /* grid-ok: equals the stroke width */
    border-width: 3px;
  }
  .box.dim {
    opacity: 0.35;
  }
`;
