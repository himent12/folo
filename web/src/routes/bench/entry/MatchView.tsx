/**
 * An image of an entry with what it found against the truth (docs/BENCH.md, Matching): a right detection is a solid
 * line, a wrong one a solid line with a cross on its corner, a true box nothing found a dashed line; each also in its
 * own status ink (right, wrong, missed) in a dark halo, so it reads on any photo and without its colour. A true box a
 * detection found is not drawn again: the solid line is on it. Full size adds chips (class and confidence, or "missed
 * <class>"); a thumbnail keeps 1 px lines and the cross. `active` brings one box forward (the viewer's list).
 */
import { computed, css, effect, For, signal } from 'fw';
import type { Box, BenchDetection } from '$lib/api/types';
import { containFit, placeBox } from '$lib/bench/frame';
import { matchImage } from '$lib/bench/match';
import { placeChips } from '$lib/detect/chips';

/** How a box came out, in the gallery's one encoding (never by colour alone: a solid line, a solid line with a cross, a dashed line). */
export type Outcome = 'right' | 'wrong' | 'missed';
export const OUTCOME_WORD: Record<Outcome, string> = { right: 'right', wrong: 'wrong', missed: 'missed' };

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
  alt?: string;
}

interface Mark {
  key: string;
  box: Box;
  outcome: Outcome;
  score: number | undefined;
}

export default function MatchView(props: MatchViewProps) {
  const full = () => props.full ?? false;
  const active = () => props.active ?? null;

  const cw = signal(0);
  const ch = signal(0);
  const loaded = signal(false);
  effect(() => {
    void props.src;
    loaded.set(false);
  });
  // The whole image in the frame, whatever either's shape (a portrait photo in a landscape tile sits between two bands
  // of the well); every box goes through the same fit, so it stays on its object ($lib/bench/frame).
  const geo = computed(() => containFit(cw.get(), ch.get(), props.width, props.height));
  const m = computed(() => matchImage(props.truth, props.detections));
  /** What is drawn: every detection, right or wrong, then every true box nothing found. */
  const marks = computed<Mark[]>(() => {
    const mm = m.get();
    return [
      ...mm.detections.map((d) => ({ key: `d${d.index}`, box: d.det as Box, outcome: (d.truth === null ? 'wrong' : 'right') as Outcome, score: d.det.score })),
      ...props.truth.flatMap((t, i) => (mm.found[i] === null ? [{ key: `t${i}`, box: t, outcome: 'missed' as Outcome, score: undefined }] : [])),
    ];
  });
  const name = (c: number) => props.classes[c] ?? `class ${c}`;
  const chipText = (k: Mark) =>
    k.outcome === 'missed' ? `missed ${name(k.box.class)}` : `${name(k.box.class)}${k.score !== undefined && k.score < 1 ? ` ${k.score.toFixed(2)}` : ''}`;
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
        w: Math.ceil(chipText(k).length * (small ? 5.8 : 6.4) + (small ? 10 : 12)) + (k.outcome === 'wrong' ? 12 : 0),
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
      {props.src ? <img src={props.src} alt={props.alt ?? ''} class:loaded={loaded.get()} draggable="false" onload={() => loaded.set(true)} /> : null}
      {geo.get() ? (
        <div class="layer" aria-hidden="true">
          <For each={marks.get()} key={(k) => k.key}>
            {(k, n) => {
              const at = computed(() => placeBox(geo.get()!, k().box));
              const chip = () => chips.get()[n()];
              return (
                <div
                  class={`box ${k().outcome}`}
                  class:on={active() === k().key}
                  class:dim={active() !== null && active() !== k().key}
                  style:left={`${at.get().left}px`}
                  style:top={`${at.get().top}px`}
                  style:width={`${Math.max(1, at.get().width)}px`}
                  style:height={`${Math.max(1, at.get().height)}px`}
                >
                  {k().outcome === 'wrong' ? <span class="x"></span> : null}
                  {full() && chip() && (!chip()!.hidden || active() === k().key) ? (
                    <span class="chip" style:left={`${chip()!.dx - 2 + (k().outcome === 'wrong' ? 12 : 0)}px`} style:top={`${chip()!.dy - 2}px`}>
                      {k().outcome === 'missed' ? (
                        `missed ${name(k().box.class)}`
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
  .box.missed {
    --c: var(--miss);
    border-style: dashed;
    box-shadow: none;
    filter: drop-shadow(0 0 0.75px var(--class-halo)) drop-shadow(0 0 0.75px var(--class-halo));
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
  .full .box.missed {
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
