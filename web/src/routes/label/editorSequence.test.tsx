// @vitest-environment happy-dom
// Labeling editor sequence accelerators: Duplicate selected box (`D`) and Copy boxes from previous labeled image (`P`),
// covering geometry helpers, cross-resolution scaling, filter fallback, undo/redo history, dirty state, and guards.
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { LabelImage } from '$lib/api/types';
import { parse, router } from '$lib/router';
import { render } from '../../../framework/testing';
import LabelJobPage from '../LabelJob';
import { adaptBoxesToImage, duplicateBox } from './boxEdit';
import { previousLabeledImage } from './editorNav';

let target: HTMLElement;
let mounted: { unmount(): void } | undefined;

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  target = document.createElement('div');
  document.body.appendChild(target);
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
});

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  target.remove();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

async function openEditor(image = '1', job = 'lb_7c21', extraQuery = '') {
  const at = `#/datasets/label/${job}?image=${image}${extraQuery}`;
  history.replaceState(history.state, '', at);
  router.route = parse(at);
  mounted = render(() => <LabelJobPage />, target);
  await vi.waitFor(() => {
    flushSync();
    if (!editor()?.open || !editor()?.querySelector('.ed')) throw new Error('loading');
  });
}

const editor = () => document.querySelector<HTMLDialogElement>('dialog.label-editor');
const boxCount = () => Number(editor()!.querySelector('h3 .meta')?.textContent);
const footer = () => editor()!.querySelector('footer')?.textContent ?? '';
const byText = (text: string) => [...editor()!.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === text)!;
const press = (k: string, init: KeyboardEventInit = {}, on: Element = editor()!) => {
  const e = new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true, ...init });
  on.dispatchEvent(e);
  flushSync();
  return e;
};

describe('boxEdit & editorNav sequence helpers', () => {
  it('duplicates a box with a visible offset and flips direction near the bottom-right corner', () => {
    const mid = duplicateBox({ class: 1, x1: 100, y1: 100, x2: 200, y2: 180 }, 640, 480);
    expect(mid.class).toBe(1);
    expect(mid.x1).toBeGreaterThan(100);
    expect(mid.y1).toBeGreaterThan(100);
    expect(mid.x2 - mid.x1).toBe(100);
    expect(mid.y2 - mid.y1).toBe(80);

    const corner = duplicateBox({ class: 0, x1: 540, y1: 400, x2: 640, y2: 480 }, 640, 480);
    expect(corner.x1).toBeLessThan(540);
    expect(corner.y1).toBeLessThan(400);
    expect(corner.x2).toBeLessThanOrEqual(640);
    expect(corner.y2).toBeLessThanOrEqual(480);
  });

  it('scales boxes across different image dimensions and filters invalid classes or tiny boxes', () => {
    const out = adaptBoxesToImage(
      [
        { class: 0, x1: 100, y1: 50, x2: 200, y2: 150 },
        { class: 9, x1: 10, y1: 10, x2: 50, y2: 50 }, // invalid class
        { class: 1, x1: 10, y1: 10, x2: 10.5, y2: 10.5 }, // scales to 1×1 (< MIN_SIDE=2)
      ],
      400,
      200,
      800,
      400,
      3,
    );
    expect(out).toEqual([{ class: 0, x1: 200, y1: 100, x2: 400, y2: 300 }]);
  });

  it('finds the nearest preceding image with boxes in the active filter, falling back to full job order', () => {
    const mk = (id: string, status: LabelImage['status'], boxes: LabelImage['boxes']): LabelImage => ({
      id,
      file: `${id}.jpg`,
      width: 640,
      height: 480,
      status,
      source: status === 'labeled' ? 'manual' : null,
      reviewed: false,
      boxes,
      error: null,
    });
    const all = [
      mk('1', 'labeled', [{ class: 0, x1: 10, y1: 10, x2: 50, y2: 50 }]),
      mk('2', 'labeled', []),
      mk('3', 'pending', []),
    ];
    // Under Pending filter, visibleImages only contains image 3 -> fallback finds image 1 in allImages.
    expect(previousLabeledImage(all, [all[2]!], '3')?.id).toBe('1');
    // First image has no preceding labeled image.
    expect(previousLabeledImage(all, all, '1')).toBeNull();
  });
});

describe('Labeling editor: Duplicate box (D) and Copy previous frame boxes (P)', () => {
  it('duplicates the selected box via button and D key, integrates with undo/redo and dirty state, and guards inputs', async () => {
    await openEditor('1');
    expect(editor()!.querySelector('.key-strip')?.textContent).toContain('Duplicate');
    expect(editor()!.querySelector('.key-strip')?.textContent).toContain('Copy prev');

    // No box selected -> D is a harmless no-op
    press('d');
    expect(boxCount()).toBe(0);

    // Add 1 box (auto-selected), duplicate via button -> 2 boxes, dirty, undoable
    byText('Add box').click();
    flushSync();
    expect(boxCount()).toBe(1);
    const dupBtn = editor()!.querySelector<HTMLButtonElement>('[data-duplicate-box]')!;
    expect(dupBtn).not.toBeNull();
    dupBtn.click();
    flushSync();
    expect(boxCount()).toBe(2);
    expect(footer()).toContain('Unsaved changes');

    // Duplicate again with D key -> 3 boxes; held key repeat must NOT create extra copies
    expect(press('d').defaultPrevented).toBe(true);
    expect(boxCount()).toBe(3);
    expect(press('d', { repeat: true }).defaultPrevented).toBe(true);
    expect(boxCount()).toBe(3);

    // Guarded when typing in the agent note input
    const noteInput = editor()!.querySelector<HTMLInputElement>('input[placeholder*="differently"]')!;
    noteInput.focus();
    expect(press('d', {}, noteInput).defaultPrevented).toBe(false);
    expect(boxCount()).toBe(3);

    // Ctrl+Z undoes one duplicate at a time, Ctrl+Shift+Z redoes it
    noteInput.blur();
    press('z', { ctrlKey: true });
    expect(boxCount()).toBe(2);
    press('z', { ctrlKey: true, shiftKey: true });
    expect(boxCount()).toBe(3);
  });

  it('copies boxes from the previous labeled image via button and P key, disables cleanly when unavailable, and undoes in one step', async () => {
    // Open fresh job image 1 -> no earlier image has boxes -> Copy prev button is disabled
    await openEditor('1');
    const copyBtn1 = editor()!.querySelector<HTMLButtonElement>('[data-copy-prev]')!;
    expect(copyBtn1.disabled).toBe(true);

    // Label image 1 with 2 boxes and save
    byText('Add box').click();
    flushSync();
    press('d');
    expect(boxCount()).toBe(2);
    byText('Save').click();
    await vi.waitFor(() => {
      flushSync();
      expect(footer()).not.toContain('Unsaved changes');
    });

    // Step to image 2 (pending, 0 boxes) -> Copy prev button enables showing (2)
    press('j');
    await vi.waitFor(() => {
      flushSync();
      expect(router.route.query.get('image')).toBe('2');
    });
    expect(boxCount()).toBe(0);
    const copyBtn2 = editor()!.querySelector<HTMLButtonElement>('[data-copy-prev]')!;
    expect(copyBtn2.disabled).toBe(false);
    expect(copyBtn2.textContent).toContain('(2)');

    // Click Copy prev -> copies both boxes into draft, marks dirty
    copyBtn2.click();
    flushSync();
    expect(boxCount()).toBe(2);
    expect(footer()).toContain('Unsaved changes');
    expect(editor()!.querySelector('[role="status"]')?.textContent).toContain('Copied 2 boxes');

    // Pressing P a second time does not pile duplicate boxes on top of identical ones
    press('p');
    expect(boxCount()).toBe(2);
    expect(editor()!.querySelector('[role="status"]')?.textContent).toContain('Already copied');

    // Single Ctrl+Z reverts the entire copied frame back to 0 boxes and clears dirty state
    press('z', { ctrlKey: true });
    expect(boxCount()).toBe(0);
    expect(footer()).not.toContain('Unsaved changes');

    // P key re-copies cleanly
    expect(press('P').defaultPrevented).toBe(true);
    expect(boxCount()).toBe(2);
  });

  it('finds the previous labeled image even when viewing the Pending filter', async () => {
    // Save 1 box on image 1 via API so image 1 is labeled (outside Pending filter), then open image 2 under ?show=pending
    await hub.api.updateLabelImage('lb_7c21', '1', [{ class: 0, x1: 20, y1: 20, x2: 120, y2: 120 }]);
    await openEditor('2', 'lb_7c21', '&show=pending');
    const copyBtn = editor()!.querySelector<HTMLButtonElement>('[data-copy-prev]')!;
    expect(copyBtn.disabled).toBe(false);
    expect(copyBtn.textContent).toContain('(1)');
    copyBtn.click();
    flushSync();
    expect(boxCount()).toBe(1);
  });
});
