/**
 * The editor's image: its boxes in the full-size style, the selected one with handles. A press on a box selects and
 * moves it, on a handle resizes it, on empty image draws a new box (in the class used last); a press that hardly moves
 * is a click, which selects and changes nothing. Mouse, pen and touch alike; the geometry is ./boxEdit's, in image
 * pixels. The whole frame takes the gestures, boxes included: the selected box is drawn above everything else, and a
 * press on it (or on its handles) is still the editor's. A press also takes focus from a field of the editor (the note),
 * so the keys that follow act on the boxes.
 */
import { bindable, computed, css, effect, Index, onCleanup, onWindow, signal } from 'fw';
import type { Box, LabelImage } from '$lib/api/types';
import DatasetImage from '$lib/detect/DatasetImage';
import { dragTo, fit, grab, HANDLE_REACH, HANDLE_REACH_TOUCH, HANDLES, handleAt, isClick, settle, toImage, toScreen, type Drag } from './boxEdit';

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
  /** Out: screen px per image px as shown (0 until the frame has a size), for the keyboard's steps. */
  scale?: number;
}

export default function EditCanvas(props: EditCanvasProps) {
  const selected = bindable(props, 'selected');
  const scale = bindable(props, 'scale', 0);
  const cw = signal(0);
  const ch = signal(0);
  let el: HTMLDivElement | undefined;
  const drag = signal<Drag | null>(null);
  const shown = signal<Box | null>(null);
  /** Where the gesture went down on screen, and whether the pointer has since left the click's slop. */
  let press: [number, number] = [0, 0];
  let dragged = false;
  /** The boxes and image the gesture started on: if they change under it (an undo, the agent, another image), it is dropped. */
  let pressedOn: Box[] = [];
  let pressedImgId: string | null = null;
  let activePointerId: number | null = null;

  function cancelDrag() {
    if (activePointerId !== null) {
      const id = activePointerId;
      activePointerId = null;
      try {
        el?.releasePointerCapture?.(id);
      } catch {
        // Capture may already have been released.
      }
    }
    drag.set(null);
    shown.set(null);
  }

  effect(() => {
    const boxes = props.boxes;
    const imgId = props.img.id;
    const off = props.disabled;
    if (drag.peek() && (off || boxes !== pressedOn || imgId !== pressedImgId)) cancelDrag();
  });
  onCleanup(cancelDrag);
  onWindow('blur', cancelDrag);

  const place = computed(() => (cw.get() && ch.get() ? fit(props.img.width, props.img.height, cw.get(), ch.get()) : null));
  effect(() => {
    scale.set(place.get()?.s ?? 0);
  });
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

  function point(e: PointerEvent): [number, number] | null {
    const p = place.peek();
    if (!p || !el) return null;
    const r = el.getBoundingClientRect();
    return toImage(p, e.clientX - r.left, e.clientY - r.top);
  }

  function onpointerdown(e: PointerEvent) {
    const p = place.peek();
    if (props.disabled || !p || e.button !== 0 || activePointerId !== null) return;
    const pt = point(e);
    if (!pt) return;
    // No default (no text selection, no native drag), so no focus move either: a field that had focus gives it up here.
    e.preventDefault();
    const dialog = el!.closest('dialog');
    if (dialog && document.activeElement !== dialog && dialog.contains(document.activeElement)) dialog.focus();
    activePointerId = e.pointerId;
    try {
      el!.setPointerCapture?.(e.pointerId);
    } catch {
      // Synthetic pointer events in tests may not have an active pointerId.
    }
    press = [e.clientX, e.clientY];
    dragged = false;
    pressedOn = props.boxes;
    pressedImgId = props.img.id;
    const [x, y] = pt;
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
    if (!d || activePointerId !== e.pointerId) return;
    if (e.pointerType === 'mouse' && (e.buttons & 1) === 0) {
      cancelDrag();
      return;
    }
    const pt = point(e);
    if (!pt) return;
    dragged ||= !isClick(press, [e.clientX, e.clientY]);
    const [x, y] = pt;
    shown.set(dragTo(d, x, y, props.img.width, props.img.height));
  }

  function onpointerup(e: PointerEvent) {
    const d = drag.peek();
    const s = shown.peek();
    if (!d || !s || activePointerId !== e.pointerId) return;
    const boxes = props.boxes;
    if (boxes !== pressedOn || props.img.id !== pressedImgId) {
      cancelDrag();
      return;
    }
    const r = settle(boxes, d, s, !dragged);
    const moved = r.boxes.length !== boxes.length || r.boxes.some((b, i) => b !== boxes[i]);
    selected.set(r.selected);
    cancelDrag();
    if (moved) props.onchange(r.boxes);
  }

  function onpointerabort(e: PointerEvent) {
    if (activePointerId === e.pointerId) cancelDrag();
  }

  return (
    <div
      class="canvas"
      class:off={props.disabled}
      bind:clientWidth={cw}
      bind:clientHeight={ch}
      ref={(e) => (el = e)}
      onpointerdown={onpointerdown}
      onpointermove={onpointermove}
      onpointerup={onpointerup}
      onpointercancel={onpointerabort}
      onlostpointercapture={onpointerabort}
    >
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
      <div class="hit">
        <Index each={handles.get()}>{(h) => <span class="grip" style:left={`${h()[0]}px`} style:top={`${h()[1]}px`}></span>}</Index>
      </div>
    </div>
  );
}

css`
  /* Every press in the frame is a gesture of the editor's, never a page scroll or a sheet pull. */
  .canvas {
    position: relative;
    width: 100%;
    height: 100%;
    min-height: 0;
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    overflow: hidden;
    cursor: crosshair;
    touch-action: none;
  }
  .canvas.off {
    cursor: default;
  }
  /* Over the image and its boxes (all but the selected one, which is drawn above it): the handles. */
  .hit {
    position: absolute;
    inset: 0;
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
