// @vitest-environment happy-dom
// Labeling editor keyboard ergonomics and undo/redo feedback: 1–9 quick class selection (both before drawing and on
// a selected box), input-typing guards, visible hint strip, stack depth indicator, and contextual disabled tooltips.
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import { render } from '../../../framework/testing';
import LabelJobPage from '../LabelJob';

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

async function openEditor(image = '1', job = 'lb_7c21') {
  const at = `#/datasets/label/${job}?image=${image}`;
  history.replaceState(history.state, '', at);
  router.route = parse(at);
  mounted = render(() => <LabelJobPage />, target);
  await vi.waitFor(() => {
    flushSync();
    if (!editor()?.open || !editor()?.querySelector('.ed')) throw new Error('loading');
  });
}

const editor = () => document.querySelector<HTMLDialogElement>('dialog.label-editor');
const btn = (label: string) => editor()!.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`)!;
const byText = (text: string) => [...editor()!.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === text)!;
const press = (k: string, init: KeyboardEventInit = {}, on: Element = editor()!) => {
  const e = new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true, ...init });
  on.dispatchEvent(e);
  flushSync();
  return e;
};

describe('Labeling editor: 1–9 quick class selection and hint strip', () => {
  it('switches active draw class with 1–9 even when no box is selected, and reclassifies when selected', async () => {
    await openEditor('1');
    const chips = () => [...editor()!.querySelectorAll<HTMLButtonElement>('.class-keys button.ck')];
    expect(chips().length).toBeGreaterThanOrEqual(2);
    expect(editor()!.querySelector('.class-keys')?.hasAttribute('data-scrolls')).toBe(true);
    expect(editor()!.querySelector('.key-strip')?.textContent).toContain('Undo');

    // No box selected: pressing '2' switches the active draw class to index 1
    expect(chips()[0]!.getAttribute('aria-pressed')).toBe('true');
    press('2');
    expect(chips()[1]!.getAttribute('aria-pressed')).toBe('true');

    // Adding a box uses class index 1 (the second class)
    byText('Add box').click();
    flushSync();
    const firstBoxLabel = () => editor()!.querySelector('.boxes button.on')?.getAttribute('aria-label') ?? '';
    const secondClassName = chips()[1]!.querySelector('.cname')!.textContent!.trim();
    const firstClassName = chips()[0]!.querySelector('.cname')!.textContent!.trim();
    expect(firstBoxLabel()).toContain(secondClassName);

    // With box selected: pressing '1' reclassifies it to class 0
    press('1');
    expect(firstBoxLabel()).toContain(firstClassName);
    expect(chips()[0]!.getAttribute('aria-pressed')).toBe('true');

    // Clicking a quick-class chip also reclassifies the selected box
    chips()[1]!.click();
    flushSync();
    expect(firstBoxLabel()).toContain(secondClassName);
  });

  it('guards 1–9 hotkeys while typing inside the agent note input', async () => {
    await openEditor('1');
    byText('Add box').click();
    flushSync();
    const chips = () => [...editor()!.querySelectorAll<HTMLButtonElement>('.class-keys button.ck')];
    const firstClassName = chips()[0]!.querySelector('.cname')!.textContent!.trim();
    const noteInput = editor()!.querySelector<HTMLInputElement>('input[placeholder*="differently"]')!;
    noteInput.focus();
    const e = press('2', {}, noteInput);
    expect(e.defaultPrevented).toBe(false);
    expect(editor()!.querySelector('.boxes button.on')?.getAttribute('aria-label')).toContain(firstClassName);
    expect(chips()[0]!.getAttribute('aria-pressed')).toBe('true');
  });
});

describe('Labeling editor: Undo/Redo stack depth and disabled tooltips', () => {
  it('displays live undo/redo stack depth and contextual tooltips across edits', async () => {
    vi.useFakeTimers();
    try {
      await openEditor('1');
      const undoBtn = btn('Undo');
      const redoBtn = btn('Redo');
      expect(undoBtn.disabled).toBe(true);
      expect(redoBtn.disabled).toBe(true);
      expect(editor()!.querySelector('.hist-depth')).toBeNull();

      // Hover disabled Undo wrapper -> shows "Nothing to undo"
      undoBtn.parentElement!.dispatchEvent(new PointerEvent('pointerenter', { bubbles: true }));
      vi.advanceTimersByTime(500);
      expect(document.getElementById('folo-tooltip')?.textContent).toContain('Nothing to undo');
      undoBtn.parentElement!.dispatchEvent(new PointerEvent('pointerleave', { bubbles: true }));
      vi.advanceTimersByTime(200);

      // Make two edits -> depth reads "History: 2 undo · 0 redo"
      byText('Add box').click();
      flushSync();
      press('2');
      expect(editor()!.querySelector('.hist-depth')?.textContent?.trim()).toBe('History: 2 undo · 0 redo');

      // Undo once -> "History: 1 undo · 1 redo"
      undoBtn.click();
      flushSync();
      expect(editor()!.querySelector('.hist-depth')?.textContent?.trim()).toBe('History: 1 undo · 1 redo');

      // Hover active Undo wrapper -> shows step count
      undoBtn.parentElement!.dispatchEvent(new PointerEvent('pointerenter', { bubbles: true }));
      vi.advanceTimersByTime(500);
      expect(document.getElementById('folo-tooltip')?.textContent).toContain('Undo (1 step)');

      // New edit drops redo stack -> "History: 2 undo · 0 redo"
      byText('Add box').click();
      flushSync();
      expect(editor()!.querySelector('.hist-depth')?.textContent?.trim()).toBe('History: 2 undo · 0 redo');
    } finally {
      vi.useRealTimers();
    }
  });
});
