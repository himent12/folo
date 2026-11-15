/**
 * Full-size image inspector: zoom (wheel / pinch / + - / double-click) and pan (drag, or Shift + arrows on the
 * focused canvas) over the image with its labels, a label list that highlights boxes, a class legend and image meta.
 * J/K or arrows step through the current filter; 0 fits the whole image; Esc closes. The open image lives in the URL
 * (`?image=`), so it is linkable.
 */
import ChevronLeft from '$icons/chevron-left';
import ChevronRight from '$icons/chevron-right';
import Maximize from '$icons/maximize';
import Minus from '$icons/minus';
import Plus from '$icons/plus';
import X from '$icons/x';
import { computed, css, cubicOut, effect, fly, Key, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import type { Box, DatasetImage as ImageInfo } from '$lib/api/types';
import DatasetImage from '$lib/detect/DatasetImage';
import { count, num } from '$lib/format';
import Button from '$lib/ui/Button';
import { DialogStatus } from '$lib/ui/feedback';
import Kbd from '$lib/ui/Kbd';
import { motion } from '$lib/ui/motion';
import Spinner from '$lib/ui/Spinner';
import { tooltip } from '$lib/ui/tooltip';
import type { ImagePager } from './pager';
import { stem } from './imagePath';
import InspectorRail from './InspectorRail';

export interface InspectorProps {
  pager: ImagePager;
  classes: readonly string[];
  imageId: string | null;
  only: ReadonlySet<number> | null;
  onnavigate: (id: string) => void;
  onclose: () => void;
  /** The linked image is not in this filter (or does not exist). */
  onmissing: (id: string) => void;
}

/** Shift + arrows on the focused canvas pan by a tenth of the view (clamped to the image like a drag). */
const PAN: Record<string, [number, number]> = { ArrowLeft: [1, 0], ArrowRight: [-1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
const clampAxis = (t: number, d: number, v: number) => (d <= v ? (v - d) / 2 : Math.min(0, Math.max(v - d, t)));

export default function Inspector(props: InspectorProps) {
  const id = computed(() => props.imageId);
  let dialog!: HTMLDialogElement;
  const viewport = signal<HTMLDivElement>();
  const vw = signal(0);
  const vh = signal(0);
  // A label (or box) under the pointer or focus is `hovered`; a label clicked to zoom to it stays `selected` until
  // clicked again or the image changes. The image draws the two apart, as the Playground does: the hovered box takes
  // its class fill and chip, the selected one the accent stroke above the rest, which recede.
  const hovered = signal<number | null>(null);
  const selected = signal<number | null>(null);

  const idx = computed(() => {
    const i = id.get();
    return i ? props.pager.indexOf(i) : -1;
  });
  const img = computed(() => (idx.get() >= 0 ? props.pager.get(idx.get()) : undefined));
  const total = computed(() => props.pager.total ?? 0);

  // ---------- open / close ----------
  effect(() => {
    if (id.get() && !dialog.open) {
      dialog.showModal();
      // The dialog takes focus as a container (as ui/Dialog does): no control or label row starts focused, so nothing
      // in the list looks picked before anything is.
      dialog.focus();
    } else if (!id.get() && dialog.open) dialog.close();
  });

  /** A copy's result, said beside the path in the dialog (the toaster sits behind the modal); it is about one image. */
  const status = new DialogStatus();
  effect(() => {
    void id.get();
    status.clear();
  });

  // Resolve a linked id that is not loaded yet, and the open one again whenever a page read before the folder changed
  // is read anew (it may be gone; after a failed read, when Retry's read lands).
  effect(() => {
    const want = id.get();
    const changed = props.pager.revision > 0 && props.pager.refreshed >= 0;
    if (!want || (!changed && untrack(() => props.pager.indexOf(want)) >= 0)) return;
    let live = true;
    void props.pager.find(want).then(
      (i) => {
        if (live && i < 0) props.onmissing(want);
      },
      () => {},
    );
    return () => {
      live = false;
    };
  });

  // Prefetch neighbours (metadata and the full-size image) so J/K feel instant. The next step, and closing the
  // inspector, drop the prefetches still loading: an image without its src stops (holding J would queue every
  // full-size image passed, and the one stopped at would wait behind them, stress s1-datasets-2).
  effect(() => {
    const at = idx.get();
    if (at < 0) return;
    props.pager.ensure(at - 1, at + 2);
    const ds = props.pager.datasetId;
    let live = true;
    const pre: HTMLImageElement[] = [];
    for (const j of [at + 1, at - 1]) {
      const n = untrack(() => props.pager.get(j));
      if (n)
        void hub.api.imageUrl(ds, n.id).then(
          (u) => {
            if (!live) return;
            const im = new Image();
            im.src = u;
            pre.push(im);
          },
          () => {},
        );
    }
    return () => {
      live = false;
      for (const im of pre) if (!im.complete) im.removeAttribute('src');
    };
  });

  /** The last step's direction: the next image slides in from that side (120 ms, none under reduced motion). */
  const stepDir = signal(0);
  async function step(d: number) {
    const at = idx.peek();
    if (at < 0) return;
    stepDir.set(d);
    const j = at + d;
    if (j < 0 || j >= total.peek()) return;
    const n = await props.pager.at(j);
    if (n) props.onnavigate(n.id);
  }

  // ---------- zoom & pan ----------
  /** z = 1 is "fit"; tx/ty is the image's top-left in viewport px (clamped on use). */
  const z = signal(1);
  const tx = signal(0);
  const ty = signal(0);

  const s0 = computed(() => {
    const im = img.get();
    return im && vw.get() && vh.get() ? Math.min(vw.get() / im.width, vh.get() / im.height) : 1;
  });
  const maxZ = computed(() => (img.get() ? Math.max(2, (8 / s0.get()) | 0) : 8));
  const dw = computed(() => (img.get() ? img.get()!.width * s0.get() * z.get() : 0));
  const dh = computed(() => (img.get() ? img.get()!.height * s0.get() * z.get() : 0));
  const x = computed(() => clampAxis(tx.get(), dw.get(), vw.get()));
  const y = computed(() => clampAxis(ty.get(), dh.get(), vh.get()));
  const zoomPct = computed(() => Math.round(s0.get() * z.get() * 100));
  /** The zoom readout: "Fit" at the fitted view (where it opens), the scale in % once zoomed. */
  const zoomLabel = computed(() => (z.get() <= 1.001 ? 'Fit' : `${zoomPct.get()}%`));

  // Fit whenever the image changes.
  effect(() => {
    void id.get();
    untrack(() => {
      z.set(1);
      hovered.set(null);
      selected.set(null);
    });
  });

  // Discrete moves (the buttons, + - 0, a double-click, a label's zoom) glide over 120 ms; continuous input (wheel,
  // pinch, drag) follows the hand with no transition. The CSS reduced-motion rule turns the glide off.
  const glide = signal(false);
  function zoomAt(f: number, px = vw.peek() / 2, py = vh.peek() / 2, eased = true) {
    glide.set(eased);
    const zz = z.peek();
    const s = s0.peek();
    const nz = Math.min(maxZ.peek(), Math.max(1, zz * f));
    const ux = (px - x.peek()) / (s * zz);
    const uy = (py - y.peek()) / (s * zz);
    z.set(nz);
    tx.set(px - ux * s * nz);
    ty.set(py - uy * s * nz);
  }

  function fit() {
    glide.set(true);
    z.set(1);
  }

  /** A label's click: select it and zoom to its box, or (already selected) let go of it and fit the image again. */
  function pick(i: number) {
    const b = img.peek()?.boxes[i];
    if (!b) return;
    if (selected.peek() === i) {
      selected.set(null);
      fit();
      return;
    }
    selected.set(i);
    zoomToBox(b);
  }

  function zoomToBox(b: Box) {
    if (!img.peek()) return;
    const w = vw.peek();
    const h = vh.peek();
    const s = s0.peek();
    const bw = Math.max(8, b.x2 - b.x1);
    const bh = Math.max(8, b.y2 - b.y1);
    const nz = Math.min(maxZ.peek(), Math.max(1, Math.min((w * 0.45) / (bw * s), (h * 0.45) / (bh * s))));
    glide.set(true);
    z.set(nz);
    tx.set(w / 2 - ((b.x1 + b.x2) / 2) * s * nz);
    ty.set(h / 2 - ((b.y1 + b.y2) / 2) * s * nz);
  }

  function local(e: { clientX: number; clientY: number }) {
    const r = viewport.peek()!.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top] as const;
  }

  // Wheel needs a non-passive listener to stop the page (and the browser's pinch-zoom) from scrolling.
  effect(() => {
    const v = viewport.get();
    if (!v) return;
    const onwheel = (e: WheelEvent) => {
      e.preventDefault();
      touched.set(true);
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? vh.peek() : 1;
      const [px, py] = local(e);
      zoomAt(Math.exp(-e.deltaY * unit * (e.ctrlKey ? 0.01 : 0.002)), px, py, false);
    };
    v.addEventListener('wheel', onwheel, { passive: false });
    return () => v.removeEventListener('wheel', onwheel);
  });

  const pointers = new Map<number, { x: number; y: number }>();
  let pinch: { d: number; mx: number; my: number } | null = null;
  const dragging = signal(false);
  /** The hint over the image (keys, gestures) fades once someone has used the viewer (a key, the wheel, a press), so
   *  it never sits on boxes in the corner for longer than it takes to read. */
  const touched = signal(false);

  function pinchState() {
    const [a, b] = [...pointers.values()];
    if (!a || !b) return null;
    return { d: Math.hypot(a.x - b.x, a.y - b.y), mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2 };
  }

  function onpointerdown(e: PointerEvent) {
    if (e.button !== 0) return;
    touched.set(true);
    viewport.peek()!.setPointerCapture(e.pointerId);
    const [px, py] = local(e);
    pointers.set(e.pointerId, { x: px, y: py });
    pinch = pointers.size === 2 ? pinchState() : null;
    dragging.set(true);
    glide.set(false);
  }

  function onpointermove(e: PointerEvent) {
    const prev = pointers.get(e.pointerId);
    if (!prev) return;
    const [px, py] = local(e);
    pointers.set(e.pointerId, { x: px, y: py });
    if (pointers.size >= 2) {
      const now = pinchState();
      if (now && pinch && pinch.d > 0) {
        tx.set(x.peek() + (now.mx - pinch.mx));
        ty.set(y.peek() + (now.my - pinch.my));
        zoomAt(now.d / pinch.d, now.mx, now.my, false);
      }
      pinch = now;
    } else {
      tx.set(x.peek() + (px - prev.x));
      ty.set(y.peek() + (py - prev.y));
    }
  }

  function onpointerup(e: PointerEvent) {
    pointers.delete(e.pointerId);
    pinch = pointers.size === 2 ? pinchState() : null;
    if (pointers.size === 0) dragging.set(false);
  }

  function ondblclick(e: MouseEvent) {
    const [px, py] = local(e);
    if (z.peek() >= maxZ.peek() * 0.99) fit();
    else zoomAt(2.5, px, py);
  }

  function onviewportkey(e: KeyboardEvent) {
    const d = PAN[e.key];
    if (!d || !e.shiftKey || e.metaKey || e.ctrlKey || e.altKey) return;
    e.preventDefault();
    e.stopPropagation();
    glide.set(true);
    tx.set(x.peek() + d[0] * Math.max(24, vw.peek() / 10));
    ty.set(y.peek() + d[1] * Math.max(24, vh.peek() / 10));
  }

  function onkeydown(e: KeyboardEvent) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    // Shift + arrows pan (on the canvas); they never step to another image.
    if (e.shiftKey && e.key in PAN) return;
    const k = e.key;
    if (k === 'j' || k === 'J' || k === 'ArrowRight') step(1);
    else if (k === 'k' || k === 'K' || k === 'ArrowLeft') step(-1);
    else if (k === '+' || k === '=') zoomAt(1.5);
    else if (k === '-' || k === '_') zoomAt(1 / 1.5);
    else if (k === '0') fit();
    else return;
    e.preventDefault();
    touched.set(true);
  }

  const zoomTools = () => (
    <div class="group" role="group" aria-label="Zoom">
      <span use:tooltip={{ text: 'Zoom out', keys: '-' }}>
        <Button variant="ghost" size="sm" icon={Minus} aria-label="Zoom out" disabled={z.get() <= 1} onclick={() => zoomAt(1 / 1.5)} />
      </span>
      <button type="button" class="zoom num" onclick={fit} use:tooltip={{ text: 'Fit to view', keys: '0' }} aria-label={`Zoom ${zoomPct.get()}%. Fit to view`}>
        {zoomLabel.get()}
      </button>
      <span use:tooltip={{ text: 'Zoom in', keys: '+' }}>
        <Button variant="ghost" size="sm" icon={Plus} aria-label="Zoom in" disabled={z.get() >= maxZ.get()} onclick={() => zoomAt(1.5)} />
      </span>
      <span use:tooltip={{ text: 'Fit', keys: '0' }}>
        <Button variant="ghost" size="sm" icon={Maximize} aria-label="Fit to view" onclick={fit} />
      </span>
    </div>
  );

  // As in the grid: bindings that re-run once more as the image goes read the last one, never undefined.
  let last: ImageInfo | undefined;
  const im = () => (last = img.get() ?? last)!;

  return (
    <dialog
      ref={(e) => (dialog = e)}
      tabindex={-1}
      class="insp"
      style={{ '--ar': img.get() ? im().width / im().height : undefined }}
      aria-label={img.get() ? `Image ${stem(im().file)}` : 'Image'}
      onclose={() => id.peek() && props.onclose()}
      onkeydown={onkeydown}
    >
      <header>
        <div class="title">
          {img.get() ? (
            <>
              <span class="file mono truncate" title={im().file}>
                {stem(im().file)}
              </span>
              {/* One line on phones too: "train", without the word split beside the cut file name. */}
              <span class="split" title="Split">
                {im().split}
                <span class="split-word"> split</span>
              </span>
            </>
          ) : (
            <span class="file faint">Loading…</span>
          )}
          {idx.get() >= 0 ? (
            <span class="pos num" aria-label={`Image ${num(idx.get() + 1)} of ${num(total.get())}`}>
              {num(idx.get() + 1)} <i>of</i> {num(total.get())}
            </span>
          ) : null}
        </div>
        <div class="tools">
          <div class="head-zoom">{zoomTools()}</div>
          <span class="sep head-zoom" aria-hidden="true"></span>
          <div class="group" role="group" aria-label="Navigate">
            <span use:tooltip={{ text: 'Previous', keys: 'K' }}>
              <Button variant="ghost" size="sm" icon={ChevronLeft} aria-label="Previous image" disabled={idx.get() <= 0} onclick={() => step(-1)} />
            </span>
            <span use:tooltip={{ text: 'Next', keys: 'J' }}>
              <Button
                variant="ghost"
                size="sm"
                icon={ChevronRight}
                aria-label="Next image"
                disabled={idx.get() < 0 || idx.get() >= total.get() - 1}
                onclick={() => step(1)}
              />
            </span>
          </div>
          <span class="sep" aria-hidden="true"></span>
          <span use:tooltip={{ text: 'Close', keys: 'Escape' }}>
            <Button variant="ghost" size="sm" icon={X} aria-label="Close" onclick={() => dialog.close()} />
          </span>
        </div>
      </header>

      <div class="body">
        {/* An image viewer, not an image: a focusable group that takes keys (pan, zoom, step), so never role="img"
            (whose content is flattened to its name); its keys are its description. */}
        <div
          class="viewport"
          class:zoomed={z.get() > 1}
          class:dragging={dragging.get()}
          ref={viewport}
          bind:clientWidth={vw}
          bind:clientHeight={vh}
          role="group"
          aria-roledescription="image viewer"
          tabindex={0}
          aria-label={img.get() ? `${stem(im().file)} with ${count(im().boxes.length, 'labelled box', 'labelled boxes')}` : 'Loading image'}
          aria-describedby="insp-keys"
          onkeydown={onviewportkey}
          onpointerdown={onpointerdown}
          onpointermove={onpointermove}
          onpointerup={onpointerup}
          onpointercancel={onpointerup}
          ondblclick={ondblclick}
        >
          {img.get() ? (
            <Key when={im().id}>
              <div
                class="stage"
                class:glide={glide.get()}
                style:width={`${dw.get()}px`}
                style:height={`${dh.get()}px`}
                style:transform={`translate3d(${x.get()}px, ${y.get()}px, 0)`}
                in:fly={{ x: 16 * stepDir.get(), duration: motion(120), easing: cubicOut }}
              >
                <DatasetImage
                  datasetId={props.pager.datasetId}
                  imageId={im().id}
                  rev={props.pager.revisionOf(idx.get())}
                  width={im().width}
                  height={im().height}
                  boxes={im().boxes}
                  classes={props.classes}
                  only={props.only}
                  active={hovered.get()}
                  selected={selected.get()}
                  onhover={(i) => hovered.set(i)}
                  alt={stem(im().file)}
                />
              </div>
            </Key>
          ) : (
            <div class="loading">
              <Spinner />
            </div>
          )}
          <div class="hint" class:touched={touched.get()} aria-hidden="true">
            <span class="fine">
              <Kbd keys="J" subtle />
              <Kbd keys="K" subtle /> step
            </span>
            <span class="fine">
              scroll to zoom · drag or <Kbd keys="Shift" subtle /> arrows to pan
            </span>
            <span class="touch">Pinch to zoom · drag to pan</span>
          </div>
        </div>
        <p id="insp-keys" class="sr-only">
          Left and right arrows, or J and K: previous and next image. Shift and arrows: pan. Plus and minus: zoom. 0: fit the
          whole image.
        </p>
        {/* Narrow screens: the zoom controls get their own 44 px bar under the image (never over the labels), with the
            gesture hint beneath it. */}
        <div class="zoom-bar">
          {zoomTools()}
          <p class="touch-hint">Pinch to zoom · drag to pan · double-tap to zoom in</p>
        </div>

        <InspectorRail img={img.get()} classes={props.classes} only={props.only} bind:hovered={hovered} selected={selected.get()} pick={pick} status={status} />
      </div>
    </dialog>
  );
}

css`
  /* Sized to the image (\`--ar\`, its width over height): the largest box of the canvas's shape plus the rail that fits
     the viewport, so a 4:3 photo on a laptop never sits between letterbox bands (the dialog is shorter instead). */
  .insp {
    --rail: 288px;
    --box-w: min(1480px, calc(100vw - 48px));
    --box-h: min(960px, calc(100dvh - 48px));
    width: min(var(--box-w), max(640px, calc((var(--box-h) - 48px) * var(--ar, 1.3333) + var(--rail))));
    height: min(var(--box-h), max(480px, calc((var(--box-w) - var(--rail)) / var(--ar, 1.3333) + 48px)));
    max-width: none;
    max-height: none;
    margin: auto;
    padding: 0;
    border: 0;
    border-radius: var(--radius-xl);
    background: var(--bg-overlay);
    color: var(--text-primary);
    box-shadow: var(--shadow-lg);
    overflow: hidden;
  }
  .insp:focus,
  .insp:focus-visible {
    outline: none;
  }
  .insp[open] {
    display: flex;
    flex-direction: column;
    animation: enter var(--dur-slow) var(--ease-out);
  }
  .insp::backdrop {
    background: var(--bg-scrim);
    backdrop-filter: blur(2px);
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    height: 48px;
    padding: 0 var(--space-2) 0 var(--space-4);
    border-bottom: 1px solid var(--line-subtle);
    flex-shrink: 0;
  }
  .title {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .file {
    font-weight: var(--weight-medium);
  }
  .split {
    flex-shrink: 0;
    white-space: nowrap;
    color: var(--text-tertiary);
  }
  .pos {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .pos i {
    font-style: normal;
    color: var(--text-disabled);
  }
  .tools {
    display: flex;
    align-items: center;
    gap: var(--space-1);
  }
  .group {
    display: flex;
    align-items: center;
    gap: 2px;
  }
  .tools span,
  .zoom-bar span {
    display: inline-flex;
  }
  .head-zoom {
    display: contents;
  }
  .zoom-bar {
    display: none;
  }
  .sep {
    width: 1px;
    height: 16px;
    margin: 0 var(--space-1);
    background: var(--line-default);
  }
  .zoom {
    min-width: 48px;
    height: var(--control-sm);
    padding: 0 6px;
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .zoom:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .body {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(0, 1fr) var(--rail);
  }
  /* The canvas is a darkroom well in both themes: photos (mostly dark) never sit in light-grey letterbox bands. The
     dark theme's bg.inset; in light, the ink token mixed into bg.inset gives the same near-black. */
  .viewport {
    position: relative;
    overflow: hidden;
    background: var(--bg-inset);
    touch-action: none;
    cursor: zoom-in;
  }
  :global([data-theme='light']) .viewport {
    background: color-mix(in srgb, var(--text-primary) 96%, var(--bg-inset));
  }
  .viewport:focus {
    outline: none;
  }
  .viewport:focus-visible {
    outline: 2px solid var(--focus);
    outline-offset: -2px;
  }
  .viewport.zoomed {
    cursor: grab;
  }
  .viewport.dragging.zoomed {
    cursor: grabbing;
  }
  .stage {
    position: absolute;
    left: 0;
    top: 0;
    will-change: transform;
  }
  .stage.glide {
    transition:
      transform var(--dur-fast) var(--ease-out),
      width var(--dur-fast) var(--ease-out),
      height var(--dur-fast) var(--ease-out);
  }
  .loading {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    color: var(--text-tertiary);
  }
  .hint {
    position: absolute;
    left: var(--space-3);
    bottom: var(--space-3);
    display: flex;
    gap: var(--space-3);
    padding: 4px 8px;
    border-radius: var(--radius-md);
    background: color-mix(in srgb, var(--bg-overlay) 82%, transparent);
    backdrop-filter: blur(6px);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    pointer-events: none;
  }
  .hint span {
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
  .hint .touch {
    display: none;
  }
  .hint {
    transition: opacity var(--dur-slow) var(--ease-in);
  }
  .hint.touched {
    opacity: 0;
  }
  @media (pointer: coarse) {
    .hint .fine {
      display: none;
    }
    .hint .touch {
      display: inline-flex;
    }
  }
  @keyframes enter {
    from {
      opacity: 0;
      transform: scale(0.985);
    }
  }
  @media (max-width: 1023px) {
    .insp {
      --rail: 248px;
    }
  }
  /* Tablets and laptops (768–1279 px): the dialog takes the viewport's height less 24 px above and below, whatever the
     image's shape. Sized to the image alone it was ~480 px tall in an 1180 px window, the rail cut after two of its
     classes; a dialog is sized to its content and the viewport, never to a fixed or image-only height
     (DESIGN_DIRECTION §2.2). The canvas letterboxes on its darkroom well instead. */
  @media (min-width: 768px) and (max-width: 1279px) {
    .insp {
      height: var(--box-h);
    }
  }
  @media (max-width: 767px) {
    .insp {
      width: 100vw;
      height: 100dvh;
      border-radius: 0;
    }
    header {
      padding-left: var(--space-3);
    }
    .split-word {
      display: none;
    }
    /* The zoom bar under the image carries the controls and the gesture hint. */
    .head-zoom,
    .hint {
      display: none;
    }
    /* The image sets the stage height (no empty bands); the label list takes the rest. */
    .body {
      position: relative;
      display: flex;
      flex-direction: column;
    }
    .viewport {
      flex: none;
      width: 100%;
      aspect-ratio: var(--ar, 4 / 3);
      max-height: 70vh;
    }
    .zoom-bar {
      flex: none;
      display: grid;
      justify-items: center;
      padding: 0 var(--space-3) var(--space-2);
      border-top: 1px solid var(--line-subtle);
    }
    .zoom-bar .group {
      height: 44px;
    }
    .zoom-bar .zoom {
      min-width: 56px;
      height: 44px;
    }
    .touch-hint {
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-tertiary);
    }
  }
`;
