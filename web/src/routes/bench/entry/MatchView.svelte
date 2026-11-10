<script lang="ts" module>
  /** How a box came out, in the gallery's one encoding (never by colour alone: a solid line, a solid line with a cross, a dashed line). */
  export type Outcome = 'right' | 'wrong' | 'missed';
  export const OUTCOME_WORD: Record<Outcome, string> = { right: 'right', wrong: 'wrong', missed: 'missed' };
</script>

<script lang="ts">
  /**
   * An image of an entry with what it found against the truth (docs/BENCH.md, Matching): a right detection is a solid
   * line, a wrong one a solid line with a cross on its corner, a true box nothing found a dashed line; each also in its
   * own status ink (right, wrong, missed) in a dark halo, so it reads on any photo and without its colour. A true box a
   * detection found is not drawn again: the solid line is on it. Full size adds chips (class and confidence, or "missed
   * <class>"); a thumbnail keeps 1 px lines and the cross. `active` brings one box forward (the viewer's list).
   */
  import type { Box, BenchDetection } from '$lib/api/types';
  import { containFit, placeBox } from '$lib/bench/frame';
  import { matchImage } from '$lib/bench/match';
  import { placeChips } from '$lib/detect/chips';

  let {
    src,
    width,
    height,
    truth,
    detections,
    classes,
    full = false,
    active = null,
    alt = '',
  }: {
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
  } = $props();

  let cw = $state(0);
  let ch = $state(0);
  let loaded = $state(false);
  $effect(() => {
    void src;
    loaded = false;
  });
  // The whole image in the frame, whatever either's shape (a portrait photo in a landscape tile sits between two bands
  // of the well); every box goes through the same fit, so it stays on its object ($lib/bench/frame).
  const geo = $derived(containFit(cw, ch, width, height));
  const m = $derived(matchImage(truth, detections));
  /** What is drawn: every detection, right or wrong, then every true box nothing found. */
  const marks = $derived([
    ...m.detections.map((d) => ({ key: `d${d.index}`, box: d.det as Box, outcome: (d.truth === null ? 'wrong' : 'right') as Outcome, score: d.det.score })),
    ...truth.flatMap((t, i) => (m.found[i] === null ? [{ key: `t${i}`, box: t, outcome: 'missed' as Outcome, score: undefined }] : [])),
  ]);
  const name = (c: number) => classes[c] ?? `class ${c}`;
  const chipText = (k: (typeof marks)[number]) => (k.outcome === 'missed' ? `missed ${name(k.box.class)}` : `${name(k.box.class)}${k.score !== undefined && k.score < 1 ? ` ${k.score.toFixed(2)}` : ''}`);
  // Full size: each box's chip placed beside its own box, never over another chip (./lib/detect/chips, as the
  // inspector's); wrong and missed first, since they are what the view is for. A chip with no free slot shows on hover.
  const order = $derived(marks.map((k, i) => ({ k, i })).sort((a, b) => Number(a.k.outcome === 'right') - Number(b.k.outcome === 'right')));
  const chips = $derived.by(() => {
    const g = geo;
    if (!g || !full) return [];
    const small = cw < 480;
    const placed = placeChips(
      order.map(({ k }) => ({
        x: g.ox + k.box.x1 * g.s,
        y: g.oy + k.box.y1 * g.s,
        w: Math.ceil(chipText(k).length * (small ? 5.8 : 6.4) + (small ? 10 : 12)) + (k.outcome === 'wrong' ? 12 : 0),
        bw: (k.box.x2 - k.box.x1) * g.s,
        bh: (k.box.y2 - k.box.y1) * g.s,
      })),
      { minX: Math.max(0, g.ox) + 4, maxX: Math.min(cw, g.ox + width * g.s) - 4, minY: Math.max(0, g.oy) + 2, maxY: Math.min(ch, g.oy + height * g.s) - 2 },
      small ? 14 : 16,
      { small },
    );
    const out: (ReturnType<typeof placeChips>[number] | undefined)[] = [];
    order.forEach(({ i }, n) => (out[i] = placed[n]));
    return out;
  });
</script>

<div class="mv" class:full class:small={cw > 0 && cw < 480} bind:clientWidth={cw} bind:clientHeight={ch}>
  {#if src}<img {src} {alt} class:loaded draggable="false" onload={() => (loaded = true)} />{/if}
  {#if geo}
    <div class="layer" aria-hidden="true">
      {#each marks as k, n (k.key)}
        {@const at = placeBox(geo, k.box)}
        <div
          class="box {k.outcome}"
          class:on={active === k.key}
          class:dim={active !== null && active !== k.key}
          style:left="{at.left}px"
          style:top="{at.top}px"
          style:width="{Math.max(1, at.width)}px"
          style:height="{Math.max(1, at.height)}px"
        >
          {#if k.outcome === 'wrong'}<span class="x"></span>{/if}
          {#if full && chips[n] && (!chips[n]!.hidden || active === k.key)}
            {@const p = chips[n]!}
            <span class="chip" style:left="{p.dx - 2 + (k.outcome === 'wrong' ? 12 : 0)}px" style:top="{p.dy - 2}px">{#if k.outcome === 'missed'}missed {name(k.box.class)}{:else}{name(k.box.class)}{#if k.score !== undefined && k.score < 1}<span class="sc">{k.score.toFixed(2)}</span>{/if}{/if}</span>
          {/if}
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
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
</style>
