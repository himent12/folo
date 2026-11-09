<script lang="ts">
  /**
   * The editor's image: its boxes in the full-size style, the selected one with handles. A press on a box selects and
   * moves it, on a handle resizes it, on empty image draws a new box (in the class used last). Mouse, pen and touch
   * alike; the geometry is ./boxEdit's, in image pixels.
   */
  import type { Box, LabelImage } from '$lib/api/types';
  import DatasetImage from '$lib/detect/DatasetImage.svelte';
  import { dragTo, fit, grab, HANDLE_REACH, HANDLE_REACH_TOUCH, HANDLES, handleAt, settle, toImage, toScreen, type Drag } from './boxEdit';

  let {
    jobId,
    img,
    classes,
    boxes,
    selected = $bindable(),
    hovered,
    cls,
    disabled,
    onchange,
  }: {
    jobId: string;
    img: LabelImage;
    classes: readonly string[];
    boxes: Box[];
    selected: number | null;
    /** A box the list points at. */
    hovered: number | null;
    /** The class a drawn box takes. */
    cls: number;
    disabled: boolean;
    onchange: (boxes: Box[]) => void;
  } = $props();

  let cw = $state(0);
  let ch = $state(0);
  let el = $state<HTMLDivElement>();
  let drag = $state.raw<Drag | null>(null);
  let shown = $state.raw<Box | null>(null);

  const place = $derived(cw && ch ? fit(img.width, img.height, cw, ch) : null);
  /** The boxes as drawn now: the one in the gesture where it is, a new one at the end. */
  const view = $derived.by(() => {
    if (!drag || !shown) return boxes;
    if (drag.grip.kind === 'draw') return [...boxes, shown];
    const i = drag.grip.index;
    return boxes.map((b, k) => (k === i ? shown! : b));
  });
  const active = $derived(drag?.grip.kind === 'draw' && shown ? boxes.length : selected);
  const handles = $derived.by(() => {
    const b = active === null ? undefined : view[active];
    if (!b || !place || disabled) return [];
    return HANDLES.map((h) => toScreen(place, ...handleAt(b, h)));
  });

  function point(e: PointerEvent): [number, number] {
    const r = el!.getBoundingClientRect();
    return toImage(place!, e.clientX - r.left, e.clientY - r.top);
  }

  function onpointerdown(e: PointerEvent) {
    if (disabled || !place || e.button !== 0) return;
    e.preventDefault();
    el!.setPointerCapture(e.pointerId);
    const [x, y] = point(e);
    const reach = (e.pointerType === 'touch' ? HANDLE_REACH_TOUCH : HANDLE_REACH) / place.s;
    const grip = grab(boxes, selected, x, y, reach);
    if (grip.kind !== 'draw') selected = grip.index;
    drag = { grip, from: [x, y], start: grip.kind === 'draw' ? null : (boxes[grip.index] ?? null), cls };
    shown = dragTo(drag, x, y, img.width, img.height);
  }

  function onpointermove(e: PointerEvent) {
    if (!drag) return;
    const [x, y] = point(e);
    shown = dragTo(drag, x, y, img.width, img.height);
  }

  function onpointerup() {
    if (!drag || !shown) return;
    const r = settle(boxes, drag, shown);
    const moved = r.boxes.length !== boxes.length || r.boxes.some((b, i) => b !== boxes[i]);
    selected = r.selected;
    drag = null;
    shown = null;
    if (moved) onchange(r.boxes);
  }
</script>

<div class="canvas" bind:clientWidth={cw} bind:clientHeight={ch}>
  <DatasetImage {jobId} imageId={img.id} width={img.width} height={img.height} boxes={view} {classes} scores={false} active={hovered} selected={active} alt={img.file} />
  <!-- svelte-ignore a11y_no_static_element_interactions: the pointer's way to draw and move boxes; the keyboard's is the box list beside it (add, select, class, delete) and the dialog's keys (arrows move or resize the selected box). -->
  <div
    class="hit"
    class:off={disabled}
    bind:this={el}
    {onpointerdown}
    {onpointermove}
    {onpointerup}
    onpointercancel={() => ((drag = null), (shown = null))}
  >
    {#each handles as [x, y], i (i)}
      <span class="grip" style:left="{x}px" style:top="{y}px"></span>
    {/each}
  </div>
</div>

<style>
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
</style>
