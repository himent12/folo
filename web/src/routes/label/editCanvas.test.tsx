// @vitest-environment happy-dom
// The labeling editor's image under the pointer: a press anywhere in the frame is the editor's (on the selected box,
// which is drawn above the drawing layer, too), a press that hardly moves only selects, and a press takes focus from a
// field so the keys that follow reach the boxes.
import { flush as flushSync, signal } from 'fw';
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Box, LabelImage } from '$lib/api/types';
import { render } from '../../../framework/testing';
import EditCanvas from './EditCanvas';

// A 400 × 300 frame for a 400 × 300 image: one screen px per image px, at the frame's top left (no layout here).
const size = { w: Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientWidth'), h: Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientHeight') };
beforeAll(() => {
  Object.defineProperty(HTMLElement.prototype, 'clientWidth', { configurable: true, get: () => 400 });
  Object.defineProperty(HTMLElement.prototype, 'clientHeight', { configurable: true, get: () => 300 });
});
afterAll(() => {
  if (size.w) Object.defineProperty(HTMLElement.prototype, 'clientWidth', size.w);
  if (size.h) Object.defineProperty(HTMLElement.prototype, 'clientHeight', size.h);
});

let target: HTMLElement;
let c: { unmount(): void } | undefined;
afterEach(() => {
  c?.unmount();
  c = undefined;
  target.remove();
});

const img: LabelImage = { id: '1', file: 'a.jpg', width: 400, height: 300, status: 'labeled', boxes: [], source: 'agent', error: null, reviewed: false };
// Agent boxes with coordinates a move by 0 px would not give back exactly (y1 + (y2 - y1) !== y2 in floating point).
const start: Box[] = [
  { class: 0, x1: 32, y1: 187.68, x2: 160, y2: 307.68 - 100 },
  { class: 1, x1: 224, y1: 141.12, x2: 352, y2: 261.12 },
];

function mount(selected: number | null, boxes = signal<Box[]>(start)) {
  const api = new MockApi('normal', 0);
  api.labelImageUrl = async () => 'data:,a';
  hub.api = api;
  target = document.createElement('div');
  document.body.appendChild(target);
  const sel = signal<number | null>(selected);
  const onchange = vi.fn<(b: Box[]) => void>();
  c = render(
    () => <EditCanvas jobId="lb_1" img={img} classes={['a', 'b']} boxes={boxes.get()} bind:selected={sel} hovered={null} cls={0} disabled={false} onchange={onchange} />,
    target,
  );
  flushSync();
  return { sel, onchange, canvas: target.querySelector<HTMLElement>('.canvas')! };
}

/** The selected box's own element, once the image has loaded (the boxes are drawn with it). */
async function selectedBox(canvas: HTMLElement) {
  let el: Element | null = null;
  await vi.waitFor(() => {
    flushSync();
    const image = canvas.querySelector('img');
    if (!image) throw new Error('no image yet');
    image.dispatchEvent(new Event('load'));
    flushSync();
    if (!(el = canvas.querySelector('.box.selected'))) throw new Error('no selected box yet');
  });
  return el!;
}

/** A press at `from`, moved to `to` in a few steps, released there; dispatched on `on` (bubbling, as a browser does). */
function gesture(on: Element, from: [number, number], to: [number, number] = from) {
  const ev = (type: string, [x, y]: [number, number]) => on.dispatchEvent(new PointerEvent(type, { clientX: x, clientY: y, button: 0, pointerId: 1, bubbles: true, cancelable: true }));
  ev('pointerdown', from);
  flushSync();
  for (let i = 1; i <= 4; i++) ev('pointermove', [from[0] + ((to[0] - from[0]) * i) / 4, from[1] + ((to[1] - from[1]) * i) / 4]);
  ev('pointerup', to);
  flushSync();
}

describe('the editor canvas', () => {
  it("resizes the selected box by a handle pressed on the box's own element, which is drawn above the drawing layer", async () => {
    const { onchange, canvas } = mount(1);
    gesture(await selectedBox(canvas), [352, 261.12], [392, 291.12]);
    expect(onchange).toHaveBeenCalledTimes(1);
    expect(onchange.mock.calls[0]![0][1]).toEqual({ class: 1, x1: 224, y1: 141.12, x2: 392, y2: 291.12 });
  });

  it('moves the selected box again when it is dragged', async () => {
    const { onchange, canvas } = mount(1);
    gesture(await selectedBox(canvas), [300, 200], [320, 210]);
    expect(onchange.mock.calls[0]![0][1]).toMatchObject({ x1: 244, x2: 372 });
  });

  it('only selects on a click: the boxes stay the very same, so nothing counts as a change', () => {
    const { sel, onchange, canvas } = mount(null);
    gesture(canvas.querySelector('.hit')!, [96, 200], [98, 201]);
    expect(sel.peek()).toBe(0);
    expect(onchange).not.toHaveBeenCalled();
  });

  it('draws no box for a click with a little movement on empty image, and clears the selection', () => {
    const { sel, onchange, canvas } = mount(1);
    gesture(canvas.querySelector('.hit')!, [380, 20], [383, 23]);
    expect(onchange).not.toHaveBeenCalled();
    expect(sel.peek()).toBeNull();
    gesture(canvas.querySelector('.hit')!, [360, 10], [390, 40]);
    expect(onchange.mock.calls[0]![0]).toHaveLength(3);
  });

  it('drops a gesture whose boxes changed under it (an undo mid-drag), instead of writing over the new ones', async () => {
    const boxes = signal<Box[]>(start);
    const { onchange, canvas } = mount(1, boxes);
    const on = await selectedBox(canvas);
    on.dispatchEvent(new PointerEvent('pointerdown', { clientX: 300, clientY: 200, button: 0, pointerId: 1, bubbles: true, cancelable: true }));
    on.dispatchEvent(new PointerEvent('pointermove', { clientX: 330, clientY: 220, pointerId: 1, bubbles: true }));
    boxes.set([start[1]!, start[0]!]);
    flushSync();
    on.dispatchEvent(new PointerEvent('pointerup', { clientX: 330, clientY: 220, pointerId: 1, bubbles: true }));
    flushSync();
    expect(onchange).not.toHaveBeenCalled();
  });

  it('takes focus from a field of its dialog, so Delete and the arrows reach the boxes', () => {
    const dialog = document.createElement('dialog');
    dialog.tabIndex = -1;
    const note = document.createElement('input');
    dialog.append(note);
    document.body.append(dialog);
    const { canvas } = mount(null);
    dialog.append(target);
    note.focus();
    gesture(canvas.querySelector('.hit')!, [96, 200]);
    expect(document.activeElement).toBe(dialog);
    dialog.remove();
  });

  it('ignores secondary pointers during an active drag and cancels cleanly on lostpointercapture or mid-drag boxes change', async () => {
    const boxes = signal<Box[]>(start);
    const { onchange, canvas } = mount(1, boxes);
    const on = await selectedBox(canvas);
    on.dispatchEvent(new PointerEvent('pointerdown', { clientX: 300, clientY: 200, button: 0, pointerId: 1, bubbles: true, cancelable: true }));
    on.dispatchEvent(new PointerEvent('pointermove', { clientX: 320, clientY: 210, pointerId: 1, bubbles: true }));
    // A second pointer moving/lifting must neither hijack nor settle pointer 1's drag.
    on.dispatchEvent(new PointerEvent('pointermove', { clientX: 50, clientY: 50, pointerId: 2, bubbles: true }));
    on.dispatchEvent(new PointerEvent('pointerup', { clientX: 50, clientY: 50, pointerId: 2, bubbles: true }));
    flushSync();
    expect(onchange).not.toHaveBeenCalled();

    // Pointer 1 finishing commits its own drag.
    on.dispatchEvent(new PointerEvent('pointerup', { clientX: 320, clientY: 210, pointerId: 1, bubbles: true }));
    flushSync();
    expect(onchange).toHaveBeenCalledTimes(1);
    expect(onchange.mock.calls[0]![0][1]).toMatchObject({ x1: 244, x2: 372 });

    // Losing pointer capture aborts an in-flight draw without committing or wedging future gestures.
    const hit = canvas.querySelector('.hit')!;
    hit.dispatchEvent(new PointerEvent('pointerdown', { clientX: 360, clientY: 10, button: 0, pointerId: 3, bubbles: true, cancelable: true }));
    hit.dispatchEvent(new PointerEvent('pointermove', { clientX: 390, clientY: 40, pointerId: 3, bubbles: true }));
    flushSync();
    expect(canvas.querySelectorAll('.box').length).toBe(3);
    canvas.dispatchEvent(new PointerEvent('lostpointercapture', { pointerId: 3, bubbles: true }));
    flushSync();
    expect(canvas.querySelectorAll('.box').length).toBe(2);
    expect(onchange).toHaveBeenCalledTimes(1);

    // Changing boxes mid-drag drops the preview immediately, before pointerup.
    hit.dispatchEvent(new PointerEvent('pointerdown', { clientX: 360, clientY: 10, button: 0, pointerId: 4, bubbles: true, cancelable: true }));
    hit.dispatchEvent(new PointerEvent('pointermove', { clientX: 390, clientY: 40, pointerId: 4, bubbles: true }));
    flushSync();
    expect(canvas.querySelectorAll('.box').length).toBe(3);
    boxes.set([start[0]!]);
    flushSync();
    expect(canvas.querySelectorAll('.box').length).toBe(1);
  });

  it('cancels a stuck mouse drag on buttons=0 or window blur so subsequent gestures still work', async () => {
    const { onchange, canvas } = mount(1);
    await selectedBox(canvas);
    const hit = canvas.querySelector('.hit')!;
    hit.dispatchEvent(new PointerEvent('pointerdown', { clientX: 360, clientY: 10, button: 0, buttons: 1, pointerType: 'mouse', pointerId: 7, bubbles: true, cancelable: true }));
    hit.dispatchEvent(new PointerEvent('pointermove', { clientX: 390, clientY: 40, buttons: 1, pointerType: 'mouse', pointerId: 7, bubbles: true }));
    flushSync();
    expect(canvas.querySelectorAll('.box').length).toBe(3);

    // Mouse button released outside without pointerup: move with buttons=0 cancels cleanly.
    hit.dispatchEvent(new PointerEvent('pointermove', { clientX: 395, clientY: 45, buttons: 0, pointerType: 'mouse', pointerId: 7, bubbles: true }));
    flushSync();
    expect(canvas.querySelectorAll('.box').length).toBe(2);
    expect(onchange).not.toHaveBeenCalled();

    // Window blur mid-drag also cancels cleanly and frees activePointerId for the next gesture.
    hit.dispatchEvent(new PointerEvent('pointerdown', { clientX: 360, clientY: 10, button: 0, buttons: 1, pointerType: 'mouse', pointerId: 8, bubbles: true, cancelable: true }));
    hit.dispatchEvent(new PointerEvent('pointermove', { clientX: 390, clientY: 40, buttons: 1, pointerType: 'mouse', pointerId: 8, bubbles: true }));
    flushSync();
    expect(canvas.querySelectorAll('.box').length).toBe(3);
    window.dispatchEvent(new Event('blur'));
    flushSync();
    expect(canvas.querySelectorAll('.box').length).toBe(2);

    gesture(hit, [360, 10], [390, 40]);
    expect(onchange).toHaveBeenCalledTimes(1);
  });

  it('cancels an active draw drag on Escape and stops propagation before parent shortcuts run', async () => {
    const { onchange, canvas } = mount(1);
    await selectedBox(canvas);
    const hit = canvas.querySelector('.hit')!;
    let bubbled = 0;
    const onBubble = (e: KeyboardEvent) => {
      if (e.key === 'Escape') bubbled++;
    };
    window.addEventListener('keydown', onBubble);
    try {
      hit.dispatchEvent(new PointerEvent('pointerdown', { clientX: 360, clientY: 10, button: 0, buttons: 1, pointerType: 'mouse', pointerId: 11, bubbles: true, cancelable: true }));
      hit.dispatchEvent(new PointerEvent('pointermove', { clientX: 390, clientY: 40, buttons: 1, pointerType: 'mouse', pointerId: 11, bubbles: true }));
      flushSync();
      expect(canvas.querySelectorAll('.box').length).toBe(3);

      const esc = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
      window.dispatchEvent(esc);
      flushSync();
      expect(esc.defaultPrevented).toBe(true);
      expect(bubbled).toBe(0);
      expect(canvas.querySelectorAll('.box').length).toBe(2);
      expect(onchange).not.toHaveBeenCalled();

      // A second Escape with no active drag bubbles normally to parent shortcuts.
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
      expect(bubbled).toBe(1);
    } finally {
      window.removeEventListener('keydown', onBubble);
    }
  });
});
