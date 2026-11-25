// @vitest-environment happy-dom
// Labeling editor keyboard ergonomics and undo/redo feedback: 1–9 quick class selection (both before drawing and on
// a selected box), input-typing guards, visible hint strip, stack depth indicator, and contextual disabled tooltips.
import { flush as flushSync, signal } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import { render } from '../../../framework/testing';
import LabelJobPage from '../LabelJob';
import EditorHints from './EditorHints';

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
      expect(undoBtn.parentElement!.classList.contains('off')).toBe(true);
      expect(redoBtn.parentElement!.classList.contains('off')).toBe(true);
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
      expect(undoBtn.parentElement!.classList.contains('off')).toBe(false);
      expect(redoBtn.parentElement!.classList.contains('off')).toBe(true);
      expect(editor()!.querySelector('.hist-depth')?.textContent?.trim()).toBe('History: 2 undo · 0 redo');

      // Undo twice -> "History: 0 undo · 2 redo", verify plural Redo tooltip
      undoBtn.click();
      flushSync();
      expect(editor()!.querySelector('.hist-depth')?.textContent?.trim()).toBe('History: 1 undo · 1 redo');

      // Hover active Undo wrapper -> shows singular step count
      undoBtn.parentElement!.dispatchEvent(new PointerEvent('pointerenter', { bubbles: true }));
      vi.advanceTimersByTime(500);
      expect(document.getElementById('folo-tooltip')?.textContent).toContain('Undo (1 step)');
      undoBtn.parentElement!.dispatchEvent(new PointerEvent('pointerleave', { bubbles: true }));
      vi.advanceTimersByTime(200);

      undoBtn.click();
      flushSync();
      expect(editor()!.querySelector('.hist-depth')?.textContent?.trim()).toBe('History: 0 undo · 2 redo');
      expect(redoBtn.parentElement!.classList.contains('off')).toBe(false);
      redoBtn.parentElement!.dispatchEvent(new PointerEvent('pointerenter', { bubbles: true }));
      vi.advanceTimersByTime(500);
      expect(document.getElementById('folo-tooltip')?.textContent).toContain('Redo (2 steps)');
      redoBtn.parentElement!.dispatchEvent(new PointerEvent('pointerleave', { bubbles: true }));
      vi.advanceTimersByTime(200);

      // New edit drops redo stack -> "History: 1 undo · 0 redo"
      byText('Add box').click();
      flushSync();
      expect(editor()!.querySelector('.hist-depth')?.textContent?.trim()).toBe('History: 1 undo · 0 redo');
      expect(redoBtn.parentElement!.classList.contains('off')).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it('blocks 1–9 keys and shows locked tooltips while the agent is labeling', async () => {
    vi.useFakeTimers();
    try {
      const api = new MockApi('normal', 0);
      const orig = api.labelImages.bind(api);
      api.labelImages = async (jobId, query) => {
        const page = await orig(jobId, query);
        return {
          ...page,
          images: page.images.map((im) => (im.id === '1' ? { ...im, status: 'labeling' as const } : im)),
        };
      };
      hub.api = api;
      await openEditor('1');

      const chips = [...editor()!.querySelectorAll<HTMLButtonElement>('.class-keys button.ck')];
      expect(chips[0]!.disabled).toBe(true);
      press('2');
      expect(chips[0]!.getAttribute('aria-pressed')).toBe('true');

      const undoBtn = btn('Undo');
      expect(undoBtn.parentElement!.classList.contains('off')).toBe(true);
      undoBtn.parentElement!.dispatchEvent(new PointerEvent('pointerenter', { bubbles: true }));
      vi.advanceTimersByTime(500);
      expect(document.getElementById('folo-tooltip')?.textContent).toContain('Undo unavailable while the agent is labeling');
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('EditorHints: overflow scroll, class bounds, and touch mode', () => {
  it('scrolls an overflowing chip strip to keep the active class in view', () => {
    const active = signal(0);
    const classes = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10'];
    mounted = render(
      () => <EditorHints classes={classes} activeClass={active.get()} hasSelection={false} disabled={false} touch={false} onselectclass={() => {}} />,
      target,
    );
    flushSync();

    const strip = target.querySelector<HTMLDivElement>('.class-keys')!;
    const chips = [...strip.querySelectorAll<HTMLButtonElement>('button.ck')];
    expect(chips).toHaveLength(9);
    expect(target.querySelector('.key-strip')?.getAttribute('role')).toBe('group');

    Object.defineProperty(strip, 'clientWidth', { configurable: true, value: 300 });
    Object.defineProperty(strip, 'scrollWidth', { configurable: true, value: 900 });
    strip.scrollLeft = 0;
    strip.getBoundingClientRect = () => new DOMRect(0, 0, 300, 32);
    chips.forEach((chip, i) => {
      chip.getBoundingClientRect = () => new DOMRect(i * 100 - strip.scrollLeft, 0, 90, 28);
    });

    // Switching to chip 9 (index 8, offscreen right at x=800..890) scrolls it fully into view
    active.set(8);
    flushSync();
    expect(strip.scrollLeft).toBeGreaterThanOrEqual(590);
    expect(strip.scrollLeft).toBeLessThanOrEqual(600);

    // Switching back to chip 1 (index 0, now offscreen left) scrolls back to 0
    active.set(0);
    flushSync();
    expect(strip.scrollLeft).toBe(0);
  });

  it('handles empty classes and hides keyboard hints + keycaps in touch mode', () => {
    mounted = render(
      () => <EditorHints classes={[]} activeClass={0} hasSelection={false} disabled={false} touch={true} onselectclass={() => {}} />,
      target,
    );
    flushSync();
    expect(target.querySelector('.class-keys')).toBeNull();
    expect(target.querySelector('.key-strip')).toBeNull();

    mounted.unmount();
    mounted = render(
      () => <EditorHints classes={['A', 'B']} activeClass={0} hasSelection={false} disabled={false} touch={true} onselectclass={() => {}} />,
      target,
    );
    flushSync();
    expect(target.querySelectorAll('.class-keys button.ck')).toHaveLength(2);
    expect(target.querySelector('.class-keys .kbd')).toBeNull();
    expect(target.querySelector('.key-strip')).toBeNull();
  });
});
