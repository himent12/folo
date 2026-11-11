// fw port of EditCanvas.svelte.
/**
 * The editor's image: its boxes in the full-size style, the selected one with handles. A press on a box selects and
 * moves it, on a handle resizes it, on empty image draws a new box (in the class used last). Mouse, pen and touch
 * alike; the geometry is ./boxEdit's, in image pixels.
 */
import { bindable, computed, css, Index, signal } from 'fw';
import type { Box, LabelImage } from '$lib/api/types';
import DatasetImage from '$lib/detect/DatasetImage';
import { dragTo, fit, grab, HANDLE_REACH, HANDLE_REACH_TOUCH, HANDLES, handleAt, settle, toImage, toScreen, type Drag } from './boxEdit';

export interface EditCanvasProps {
  jobId: string;
  img: LabelImage;
  classes: readonly string[];
  boxes: Box[];
  selected?: number | null;
  /** A box the list points at. */
  hovered: number | null;
  /** The class a drawn box takes. */
  cls: number;
  disabled: boolean;
  onchange: (boxes: Box[]) => void;
}

export default function EditCanvas(props: EditCanvasProps) {
  const selected = bindable(props, 'selected');
  const cw = signal(0);
  const ch = signal(0);
  let el: HTMLDivElement | undefined;
  const drag = signal<Drag | null>(null);
  const shown = signal<Box | null>(null);

  const place = computed(() => (cw.get() && ch.get() ? fit(props.img.width, props.img.height, cw.get(), ch.get()) : null));
  /** The boxes as drawn now: the one in the gesture where it is, a new one at the end. */
  const view = computed(() => {
    const d = drag.get();
    const s = shown.get();
    if (!d || !s) return props.boxes;
    if (d.grip.kind === 'draw') return [...props.boxes, s];
    const i = d.grip.index;
    return props.boxes.map((b, k) => (k === i ? s : b));
  });
  const active = computed(() => (drag.get()?.grip.kind === 'draw' && shown.get() ? props.boxes.length : (selected.get() ?? null)));
  const handles = computed(() => {
    const a = active.get();
    const b = a === null ? undefined : view.get()[a];
    const p = place.get();
    if (!b || !p || props.disabled) return [];
    return HANDLES.map((h) => toScreen(p, ...handleAt(b, h)));
  });

  function point(e: PointerEvent): [number, number] {
    const r = el!.getBoundingClientRect();
    return toImage(place.peek()!, e.clientX - r.left, e.clientY - r.top);
  }

  function onpointerdown(e: PointerEvent) {
    const p = place.peek();
    if (props.disabled || !p || e.button !== 0) return;
    e.preventDefault();
    el!.setPointerCapture(e.pointerId);
    const [x, y] = point(e);
    const reach = (e.pointerType === 'touch' ? HANDLE_REACH_TOUCH : HANDLE_REACH) / p.s;
    const boxes = props.boxes;
    const grip = grab(boxes, selected.peek() ?? null, x, y, reach);
    if (grip.kind !== 'draw') selected.set(grip.index);
    const d: Drag = { grip, from: [x, y], start: grip.kind === 'draw' ? null : (boxes[grip.index] ?? null), cls: props.cls };
    drag.set(d);
    shown.set(dragTo(d, x, y, props.img.width, props.img.height));
  }

  function onpointermove(e: PointerEvent) {
    const d = drag.peek();
    if (!d) return;
    const [x, y] = point(e);
    shown.set(dragTo(d, x, y, props.img.width, props.img.height));
  }

  function onpointerup() {
    const d = drag.peek();
    const s = shown.peek();
    if (!d || !s) return;
    const boxes = props.boxes;
    const r = settle(boxes, d, s);
    const moved = r.boxes.length !== boxes.length || r.boxes.some((b, i) => b !== boxes[i]);
    selected.set(r.selected);
    drag.set(null);
    shown.set(null);
    if (moved) props.onchange(r.boxes);
  }

  return (
    <div class="canvas" bind:clientWidth={cw} bind:clientHeight={ch}>
      <DatasetImage
        jobId={props.jobId}
        imageId={props.img.id}
        width={props.img.width}
        height={props.img.height}
        boxes={view.get()}
        classes={props.classes}
        scores={false}
        active={props.hovered}
        selected={active.get()}
        alt={props.img.file}
      />
      {/* The pointer's way to draw and move boxes; the keyboard's is the box list beside it (add, select, class, delete)
          and the dialog's keys (arrows move or resize the selected box). */}
      <div
        class="hit"
        class:off={props.disabled}
        ref={(e) => (el = e)}
        onpointerdown={onpointerdown}
        onpointermove={onpointermove}
        onpointerup={onpointerup}
        onpointercancel={() => (drag.set(null), shown.set(null))}
      >
        <Index each={handles.get()}>{(h) => <span class="grip" style:left={`${h()[0]}px`} style:top={`${h()[1]}px`}></span>}</Index>
      </div>
    </div>
  );
}

css`
  .canvas {
    position: relative;
    width: 100%;
    height: 100%;
    min-height: 0;
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    overflow: hidden;
  }
  /* Over the image and its boxes: every press is a gesture of the editor's, never a page scroll or a sheet pull. */
  .hit {
    position: absolute;
    inset: 0;
    cursor: crosshair;
    touch-action: none;
  }
  .hit.off {
    cursor: default;
  }
  .grip {
    position: absolute;
    width: 8px;
    height: 8px;
    margin: -4px 0 0 -4px;
    border-radius: 2px;
    background: var(--accent-solid);
    box-shadow: 0 0 0 1px var(--class-halo);
    pointer-events: none;
  }
  @media (pointer: coarse) {
    .grip {
      width: 12px;
      height: 12px;
      margin: -6px 0 0 -6px;
    }
  }
`;
