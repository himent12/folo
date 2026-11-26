// @vitest-environment happy-dom
// The labeling editor on the job page, over the mock: saving (also while another image opens or the boxes change),
// leaving with unsaved boxes by any route, saving an image not labeled yet as it is, its keys only while it is the
// topmost dialog, the keyboard's way to add, move and resize a box, and "Ask the agent again" following its run.
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Box, LabelImage } from '$lib/api/types';
import { parse, router } from '$lib/router';
import { tabHistory } from '$lib/tabHistory.testing';
import { toast } from '$lib/ui/toast';
import { render } from '../../../framework/testing';
import LabelJobPage from '../LabelJob';

/** Port of `mount(LabelJobPage, { target })`; `unmount(c)` is `c.unmount()`. */
const mount = (_: typeof LabelJobPage, o: { target: HTMLElement }) => render(() => <LabelJobPage />, o.target);
const unmount = (c: { unmount(): void }) => c.unmount();

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  toast.items = [];
  target = document.createElement('div');
  document.body.appendChild(target);
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => (resolve = r));
  return { promise, resolve };
}

/** The fresh job's page with image `image` open in the editor (its 40 images are pending, without boxes). */
async function openEditor(image = '1', job = 'lb_7c21') {
  const at = `#/datasets/label/${job}?image=${image}`;
  history.replaceState(history.state, '', at);
  router.route = parse(at);
  c = mount(LabelJobPage, { target });
  await vi.waitFor(() => {
    flushSync();
    if (!editor()?.open || !editor()?.querySelector('.ed')) throw new Error('loading');
  });
}
const editor = () => document.querySelector<HTMLDialogElement>('dialog.label-editor');
const asking = () => [...document.querySelectorAll<HTMLDialogElement>('dialog[open]')].find((d) => d.textContent?.includes('Save your changes?'));
const inEditor = (name: string) => [...editor()!.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === name);
const boxCount = () => Number(editor()!.querySelector('h3 .meta')?.textContent);
const footer = () => editor()!.querySelector('footer')?.textContent ?? '';
const click = (b: HTMLElement | undefined) => {
  b!.click();
  flushSync();
};
const key = (k: string, init: KeyboardEventInit = {}, on: Element = editor()!) => {
  const e = new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true, ...init });
  on.dispatchEvent(e);
  flushSync();
  return e;
};
/** The selected box's size, as the list says it. */
const selectedSize = () => editor()!.querySelector('.boxes button.on')?.getAttribute('aria-label');

describe('the labeling editor: saving', () => {
  it('applies a late answer to its own image only, not to the one opened meanwhile', async () => {
    await openEditor('1');
    const answer = deferred<LabelImage>();
    // The spy itself: `hub.api` hands out its sharing wrapper of the method (lib/api/share), which has no `.mock`.
    const update = vi.spyOn(hub.api, 'updateLabelImage').mockReturnValue(answer.promise);
    click(inEditor('Add box'));
    expect(boxCount()).toBe(1);
    click(inEditor('Save'));
    const [, , sent] = update.mock.calls[0]!;
    // Image 2, without saving anything more: the save of image 1 is still on its way.
    key('j');
    click([...asking()!.querySelectorAll('button')].find((b) => b.textContent?.includes('Discard')));
    await vi.waitFor(() => expect(router.route.query.get('image')).toBe('2'));
    flushSync();
    expect(boxCount()).toBe(0);
    // Two boxes drawn on image 2 by the time image 1's answer (one box) comes.
    click(inEditor('Add box'));
    click(inEditor('Add box'));
    const one = (await hub.api.labelImages('lb_7c21', { limit: 1 })).images[0]!;
    answer.resolve({ ...one, status: 'labeled', source: 'manual', boxes: sent });
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
    // Image 2 keeps its own boxes, unsaved; image 1's answer is in the list.
    expect(boxCount()).toBe(2);
    expect(footer()).toMatch(/Unsaved changes/);
    expect(document.querySelector('dialog.label-editor [role="status"]')?.textContent ?? '').not.toMatch(/Saved/);
  });

  it('keeps boxes changed while the save was on its way as unsaved changes', async () => {
    await openEditor('1');
    const answer = deferred<void>();
    const real = hub.api.updateLabelImage.bind(hub.api);
    vi.spyOn(hub.api, 'updateLabelImage').mockImplementation((job, id, boxes) => answer.promise.then(() => real(job, id, boxes)));
    click(inEditor('Add box'));
    click(inEditor('Save'));
    click(inEditor('Add box'));
    expect(boxCount()).toBe(2);
    answer.resolve();
    await vi.waitFor(() => {
      flushSync();
      expect(editor()!.querySelector('[role="status"]')?.textContent ?? document.body.textContent).toMatch(/Saved/);
    });
    expect(boxCount()).toBe(2);
    expect(footer()).toMatch(/Unsaved changes/);
  });

  it('says so when another tab saves the image under unsaved edits, and Discard goes back to those labels', async () => {
    await openEditor('1');
    click(inEditor('Add box'));
    expect(boxCount()).toBe(1);
    // Another tab saves two boxes on the same image: the stream brings them here.
    const theirs = [
      { class: 0, x1: 1, y1: 1, x2: 20, y2: 20 },
      { class: 1, x1: 30, y1: 30, x2: 60, y2: 60 },
    ];
    await hub.api.updateLabelImage('lb_7c21', '1', theirs);
    await vi.waitFor(() => {
      flushSync();
      expect(editor()!.querySelector('[role="status"]')?.textContent ?? '').toMatch(/Changed elsewhere/);
    });
    // The edits stay, unsaved; Discard shows what the other tab saved, not what this one opened with.
    expect(boxCount()).toBe(1);
    expect(footer()).toMatch(/Unsaved changes/);
    click(inEditor('Discard'));
    expect(boxCount()).toBe(2);
  });

  it('says so in the leave dialog too when another tab saves the image, also while that dialog is open', async () => {
    await openEditor('1');
    click(inEditor('Add box'));
    // J (the next image) asks first; the editor under the question is inert, so its status line is not where the eye is.
    key('j');
    expect(asking()).toBeDefined();
    expect(asking()!.textContent).not.toMatch(/another tab/);
    await hub.api.updateLabelImage('lb_7c21', '1', [{ class: 0, x1: 1, y1: 1, x2: 20, y2: 20 }]);
    await vi.waitFor(() => {
      flushSync();
      expect(asking()!.textContent).toMatch(/also saved from another tab or window while you edited/);
    });
    // Discard keeps those labels, and the note goes with the edits.
    click([...asking()!.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Discard changes'));
    expect(asking()).toBeUndefined();
  });

  it('saves an image not labeled yet as it is: labeled by hand, a background image without boxes', async () => {
    await openEditor('3');
    expect(boxCount()).toBe(0);
    expect(footer()).toMatch(/Save keeps it as labeled by hand, with no boxes/);
    const save = inEditor('Save')!;
    expect(save.disabled).toBe(false);
    click(save);
    await vi.waitFor(async () => {
      const three = (await hub.api.labelImages('lb_7c21', { limit: 3 })).images[2]!;
      expect(three).toMatchObject({ id: '3', status: 'labeled', source: 'manual', boxes: [] });
    });
    // Labeled now, with nothing changed: nothing to save.
    await vi.waitFor(() => {
      flushSync();
      expect(inEditor('Save')!.disabled).toBe(true);
    });
    expect(footer()).toMatch(/No changes yet/);
  });
});

describe('the labeling editor: leaving with unsaved boxes', () => {
  it('asks before any route change, and continues it after Discard', async () => {
    await openEditor('1');
    click(inEditor('Add box'));
    router.go('/datasets');
    flushSync();
    expect(router.route.path).toBe('/datasets/label/lb_7c21');
    expect(asking()).toBeTruthy();
    click([...asking()!.querySelectorAll('button')].find((b) => b.textContent?.includes('Discard')));
    expect(router.route.path).toBe('/datasets');
  });

  it('continues it after Save, with the boxes saved', async () => {
    await openEditor('1');
    click(inEditor('Add box'));
    // A followed link: the URL moves first, and the router brings the tab back while the question is open.
    const tab = tabHistory();
    tab.follow('#/runs');
    await tab.settle();
    flushSync();
    expect(router.route.path).toBe('/datasets/label/lb_7c21');
    expect(location.hash).toBe('#/datasets/label/lb_7c21?image=1');
    click([...asking()!.querySelectorAll('button')].find((b) => b.textContent?.trim() === 'Save'));
    await vi.waitFor(() => expect(router.route.path).toBe('/runs'));
    expect(tab.hashes()).toEqual(['#/datasets/label/lb_7c21?image=1', '#/runs']);
    expect((await hub.api.labelImages('lb_7c21', { limit: 1 })).images[0]!.boxes).toHaveLength(1);
  });

  it('drops it when the question is closed, and stays on the image', async () => {
    await openEditor('1');
    click(inEditor('Add box'));
    router.go('/datasets/label/lb_7c21?image=5', { replace: true });
    flushSync();
    click(asking()!.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!);
    expect(asking()).toBeUndefined();
    expect(router.route.query.get('image')).toBe('1');
    expect(boxCount()).toBe(1);
    expect(footer()).toMatch(/Unsaved changes/);
  });

  it('warns before the tab closes while boxes are unsaved, and not otherwise', async () => {
    await openEditor('1');
    const unload = () => {
      const e = new Event('beforeunload', { cancelable: true });
      window.dispatchEvent(e);
      return e.defaultPrevented;
    };
    expect(unload()).toBe(false);
    click(inEditor('Add box'));
    expect(unload()).toBe(true);
    click(inEditor('Discard'));
    expect(unload()).toBe(false);
  });
});

describe('the labeling editor: keys', () => {
  it('acts on the editor only while it is the topmost dialog', async () => {
    await openEditor('1');
    click(inEditor('Add box'));
    expect(boxCount()).toBe(1);
    // A dialog over it (the palette), focus on its Close button: Delete is not the editor's.
    const over = document.createElement('dialog');
    over.innerHTML = '<button type="button">Close</button>';
    document.body.appendChild(over);
    over.showModal();
    const close = over.querySelector('button')!;
    expect(key('Delete', {}, close).defaultPrevented).toBe(false);
    expect(key('j', {}, close).defaultPrevented).toBe(false);
    expect(boxCount()).toBe(1);
    expect(router.route.query.get('image')).toBe('1');
    over.remove();
    // Typing in the editor's own field is typing; with Ctrl or Cmd held it is someone else's shortcut.
    expect(key('Delete', {}, editor()!.querySelector('input')!).defaultPrevented).toBe(false);
    expect(key('Delete', { ctrlKey: true }).defaultPrevented).toBe(false);
    expect(boxCount()).toBe(1);
    expect(key('Delete').defaultPrevented).toBe(true);
    expect(boxCount()).toBe(0);
  });

  it('adds a box, then moves it with the arrows and resizes it with Alt+arrows, Shift by 10', async () => {
    await openEditor('1');
    const sent = vi.spyOn(hub.api, 'updateLabelImage');
    click(inEditor('Add box'));
    const { width, height } = (await hub.api.labelImages('lb_7c21', { limit: 1 })).images[0]!;
    const side = Math.round(Math.min(width, height) / 4);
    expect(selectedSize()).toBe(`bolt, ${side} by ${side} pixels`);
    key('ArrowRight');
    key('ArrowDown', { shiftKey: true });
    key('ArrowRight', { altKey: true });
    key('ArrowDown', { altKey: true, shiftKey: true });
    expect(selectedSize()).toBe(`bolt, ${side + 1} by ${side + 10} pixels`);
    click(inEditor('Save'));
    await vi.waitFor(() => expect(sent).toHaveBeenCalled());
    const x1 = Math.round((width - side) / 2) + 1;
    const y1 = Math.round((height - side) / 2) + 10;
    expect(sent.mock.calls[0]![2]).toEqual([{ class: 0, x1, y1, x2: x1 + side + 1, y2: y1 + side + 10 } satisfies Box]);
  });

  it('keeps working after Save: focus dropped to the page by the disabled button comes back to the editor', async () => {
    await openEditor('1');
    click(inEditor('Add box'));
    click(inEditor('Save'));
    await vi.waitFor(() => {
      flushSync();
      expect(footer()).toMatch(/No changes yet/);
    });
    // The browser moves focus off a button that turns disabled: to the body, outside the dialog.
    (document.activeElement as HTMLElement | null)?.blur();
    expect(key('j', {}, document.body).defaultPrevented).toBe(true);
    await vi.waitFor(() => expect(router.route.query.get('image')).toBe('2'));
    expect(document.activeElement).toBe(editor());
    // Not while another dialog is open over it.
    const over = document.createElement('dialog');
    document.body.appendChild(over);
    over.showModal();
    (document.activeElement as HTMLElement | null)?.blur();
    expect(key('k', {}, document.body).defaultPrevented).toBe(false);
    over.remove();
  });

  it('leaves the arrows alone when no box is selected', async () => {
    await openEditor('1');
    expect(key('ArrowRight').defaultPrevented).toBe(false);
  });

  it('drops the selection on Esc first, and leaves the next Esc to the dialog', async () => {
    await openEditor('1');
    click(inEditor('Add box'));
    expect(selectedSize()).toBeTruthy();
    expect(key('Escape').defaultPrevented).toBe(true);
    expect(selectedSize()).toBeUndefined();
    expect(editor()!.open).toBe(true);
    expect(key('Escape').defaultPrevented).toBe(false);
  });

  it('restores keyboard focus when deleting a box from the side list or Delete box button, and scrolls only .boxes to keep selected box visible', async () => {
    await openEditor('1');
    click(inEditor('Add box'));
    const list = editor()!.querySelector<HTMLElement>('.boxes')!;
    const dialogBody = editor()!.querySelector<HTMLElement>('.body')!;
    vi.spyOn(list, 'getBoundingClientRect').mockReturnValue({ top: 100, bottom: 200, left: 0, right: 200, width: 200, height: 100, x: 0, y: 100, toJSON: () => ({}) } as DOMRect);
    click(inEditor('Add box'));
    const secondBtn = list.querySelectorAll<HTMLButtonElement>('button')[1]!;
    vi.spyOn(secondBtn, 'getBoundingClientRect').mockReturnValue({ top: 230, bottom: 258, left: 0, right: 200, width: 200, height: 28, x: 0, y: 230, toJSON: () => ({}) } as DOMRect);
    await Promise.resolve();
    // Only `.boxes` scrolls (by 58px to bring bottom 258 up to list bottom 200); the enclosing dialog body never moves.
    expect(list.scrollTop).toBe(58);
    expect(dialogBody.scrollTop).toBe(0);
    expect(boxCount()).toBe(2);

    // Click Delete box while focused inside `.sel`: focus moves to the remaining box button in `.boxes`.
    const delBtn = inEditor('Delete box')!;
    delBtn.focus();
    click(delBtn);
    await Promise.resolve();
    expect(boxCount()).toBe(1);
    const remaining = editor()!.querySelector<HTMLButtonElement>('.boxes button')!;
    expect(document.activeElement).toBe(remaining);

    // Select that last box and press Delete while focused on its row button: focus moves to Add box.
    click(remaining);
    remaining.focus();
    key('Delete', {}, remaining);
    await Promise.resolve();
    expect(boxCount()).toBe(0);
    expect(document.activeElement).toBe(editor()!.querySelector<HTMLButtonElement>('[data-add-box]'));
  });

  it('clears hovered box highlight when stepping to another image', async () => {
    const cw = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientWidth');
    const ch = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientHeight');
    Object.defineProperty(HTMLElement.prototype, 'clientWidth', { configurable: true, get: () => 400 });
    Object.defineProperty(HTMLElement.prototype, 'clientHeight', { configurable: true, get: () => 300 });
    vi.spyOn(hub.api, 'labelImageUrl').mockResolvedValue('data:,a');
    try {
      await openEditor('1', 'lb_51d0');
      const loadCanvasImg = async () => {
        await vi.waitFor(() => {
          flushSync();
          const im = editor()!.querySelector('.canvas img');
          if (!im) throw new Error('no image yet');
          im.dispatchEvent(new Event('load'));
          flushSync();
          if (!editor()!.querySelector('.canvas .box')) throw new Error('no boxes yet');
        });
      };
      await loadCanvasImg();
      const firstBoxBtn = editor()!.querySelector<HTMLButtonElement>('.boxes button')!;
      expect(firstBoxBtn).not.toBeNull();
      firstBoxBtn.dispatchEvent(new PointerEvent('pointerenter', { pointerType: 'mouse', bubbles: true }));
      flushSync();
      expect(editor()!.querySelector('.canvas .box.active')).not.toBeNull();

      key('j');
      await vi.waitFor(() => {
        flushSync();
        expect(router.route.query.get('image')).toBe('2');
      });
      await loadCanvasImg();
      expect(editor()!.querySelector('.canvas .box.active')).toBeNull();
    } finally {
      if (cw) Object.defineProperty(HTMLElement.prototype, 'clientWidth', cw);
      if (ch) Object.defineProperty(HTMLElement.prototype, 'clientHeight', ch);
    }
  });
});

describe('the labeling editor: undo and redo (studio-label F4)', () => {
  /** The first box's size, as the list says it (an undo drops the selection). */
  const firstSize = () => editor()!.querySelector('.boxes button')?.getAttribute('aria-label');
  const button = (label: string) => editor()!.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`)!;

  it('steps back and forth through the edits with the keys, a run of arrow presses as one step, Discard too', async () => {
    await openEditor('1');
    expect(button('Undo').disabled).toBe(true);
    click(inEditor('Add box'));
    const added = firstSize();
    key('ArrowRight');
    key('ArrowRight');
    key('ArrowRight');
    expect(key('z', { ctrlKey: true }).defaultPrevented).toBe(true);
    expect(firstSize()).toBe(added);
    expect(boxCount()).toBe(1);
    key('z', { metaKey: true });
    expect(boxCount()).toBe(0);
    expect(footer()).not.toMatch(/Unsaved changes/);
    key('z', { ctrlKey: true, shiftKey: true });
    key('y', { ctrlKey: true });
    expect(boxCount()).toBe(1);
    click(inEditor('Discard'));
    expect(boxCount()).toBe(0);
    click(button('Undo'));
    expect(boxCount()).toBe(1);
    expect(footer()).toMatch(/Unsaved changes/);
  });

  it('makes separate arrow presses, and presses either side of a save, separate steps', async () => {
    await openEditor('1');
    click(inEditor('Add box'));
    const at = () => firstSize();
    key('ArrowRight', { altKey: true });
    const one = at();
    window.dispatchEvent(new KeyboardEvent('keyup', { key: 'ArrowRight' }));
    key('ArrowRight', { altKey: true });
    key('z', { ctrlKey: true });
    expect(at()).toBe(one);
    click(editor()!.querySelector<HTMLButtonElement>('.boxes button')!);
    key('ArrowDown', { altKey: true });
    const saved = at();
    key('s', { ctrlKey: true });
    await vi.waitFor(() => {
      flushSync();
      expect(footer()).toMatch(/Reviewed|No changes yet/);
    });
    // Still selected after the save.
    key('ArrowDown', { altKey: true });
    expect(at()).not.toBe(saved);
    key('z', { ctrlKey: true });
    expect(at()).toBe(saved);
    expect(footer()).not.toMatch(/Unsaved changes/);
  });

  it('drops the selection on undo, so Delete never hits another box', async () => {
    await openEditor('1');
    click(inEditor('Add box'));
    click(inEditor('Add box'));
    key('Delete');
    expect(boxCount()).toBe(1);
    click(editor()!.querySelector<HTMLButtonElement>('.boxes button')!);
    expect(selectedSize()).toBeTruthy();
    key('z', { ctrlKey: true });
    expect(boxCount()).toBe(2);
    expect(selectedSize()).toBeUndefined();
    expect(key('Delete').defaultPrevented).toBe(true);
    expect(boxCount()).toBe(2);
  });

  it('drops the selection when an undo swaps the boxes at the same count (Discard undone)', async () => {
    await openEditor('1');
    click(inEditor('Add box'));
    key('ArrowRight', { altKey: true, shiftKey: true });
    click(inEditor('Save'));
    await vi.waitFor(() => {
      flushSync();
      expect(footer()).not.toMatch(/Unsaved changes/);
    });
    const saved = selectedSize();
    key('Delete');
    click(inEditor('Add box'));
    click(inEditor('Discard'));
    click(editor()!.querySelector<HTMLButtonElement>('.boxes button')!);
    expect(selectedSize()).toBe(saved);
    key('z', { ctrlKey: true });
    expect(boxCount()).toBe(1);
    expect(selectedSize()).toBeUndefined();
    key('Delete');
    expect(boxCount()).toBe(1);
  });

  it('starts afresh on another image, and leaves Ctrl+Z in a field to the field', async () => {
    await openEditor('1');
    click(inEditor('Add box'));
    expect(key('z', { ctrlKey: true }, editor()!.querySelector('input')!).defaultPrevented).toBe(false);
    expect(boxCount()).toBe(1);
    click(inEditor('Discard'));
    click(editor()!.querySelector<HTMLButtonElement>('button[aria-label="Next image"]')!);
    await vi.waitFor(() => expect(router.route.query.get('image')).toBe('2'));
    flushSync();
    expect(button('Undo').disabled).toBe(true);
    expect(key('z', { ctrlKey: true }).defaultPrevented).toBe(true);
    expect(router.route.query.get('image')).toBe('2');
  });
});

describe('the labeling editor: asking the agent again', () => {
  it("follows the run it started: the line becomes the image's new boxes once the run ends", async () => {
    await openEditor('2');
    const ask = await vi.waitFor(() => {
      flushSync();
      const b = inEditor('Ask Claude Code');
      if (!b || b.disabled) throw new Error('agents loading');
      return b;
    });
    click(ask);
    const line = () => editor()!.querySelector('.dialog-status')?.textContent ?? '';
    await vi.waitFor(() => {
      flushSync();
      expect(line()).toMatch(/is labeling this image again/);
    });
    // The mock's agent answers a call 1.5 s after the run starts: its stream applies it on the next tick.
    const now = Date.now();
    vi.spyOn(Date, 'now').mockReturnValue(now + 2000);
    await vi.waitFor(() => {
      flushSync();
      expect(line()).toMatch(/New boxes/);
    });
    expect(line()).not.toMatch(/is labeling this image again/);
  });

  // review R3: the answer to "Ask the agent again" came after another image was opened, and that image's line said the
  // agent was labeling it, then "New boxes" with its own unchanged boxes.
  it.each([
    ['Next', () => void key('j')],
    ['closing and opening another image', () => (router.go('/datasets/label/lb_7c21'), router.go('/datasets/label/lb_7c21?image=3'))],
  ])('keeps its line and outcome on the image that asked, after %s', async (_, leave) => {
    await openEditor('2');
    const ask = await vi.waitFor(() => {
      flushSync();
      const b = inEditor('Ask Claude Code');
      if (!b || b.disabled) throw new Error('agents loading');
      return b;
    });
    const answer = deferred<void>();
    const start = hub.api.startLabeling.bind(hub.api);
    const started = vi.spyOn(hub.api, 'startLabeling').mockImplementation(async (...a) => (await answer.promise, start(...a)));
    click(ask);
    expect(started).toHaveBeenCalledWith('lb_7c21', expect.objectContaining({ image_ids: ['2'] }));
    leave();
    await vi.waitFor(() => {
      flushSync();
      expect(router.route.query.get('image')).toBe('3');
      expect(editor()?.open && editor()?.querySelector('.ed')).toBeTruthy();
    });
    const line = () => editor()!.querySelector('.dialog-status')?.textContent ?? '';
    answer.resolve();
    await started.mock.results[0]!.value;
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
    expect(line()).not.toMatch(/again/);
    // The run goes on and ends (the mock's agent answers 1.5 s after it starts); image 3's line says nothing of it.
    vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 2000);
    await vi.waitFor(async () => {
      flushSync();
      const two = (await hub.api.labelImages('lb_7c21', { limit: 5 })).images.find((i) => i.id === '2');
      expect(two?.status).toBe('labeled');
    });
    await new Promise((r) => setTimeout(r, 50));
    flushSync();
    expect(router.route.query.get('image')).toBe('3');
    expect(line()).not.toMatch(/again|New boxes/);
  });
});

describe('the labeling editor: reviewing', () => {
  // lb_51d0: 36 images labeled, the first 22 reviewed.
  it('marks labels that look right reviewed with Enter and opens the next image to review', async () => {
    await openEditor('23', 'lb_51d0');
    expect(footer()).toMatch(/Labels right\?/);
    expect(inEditor('Save')).toBeUndefined();
    expect(key('Enter').defaultPrevented).toBe(true);
    await vi.waitFor(() => expect(router.route.query.get('image')).toBe('24'));
    const page = await hub.api.labelImages('lb_51d0', { limit: 24 });
    expect(page.images[22]).toMatchObject({ id: '23', source: 'agent', reviewed: true });
    expect(page.images[23]!.reviewed).toBe(false);
  });

  it('says a reviewed image is reviewed, and can take it back; Enter on a button is the button’s', async () => {
    await openEditor('5', 'lb_51d0');
    expect(footer()).toMatch(/^Reviewed\./);
    expect(inEditor('Looks right')).toBeUndefined();
    expect(key('Enter').defaultPrevented).toBe(false);
    click(inEditor('Mark not reviewed'));
    await vi.waitFor(() => {
      flushSync();
      expect(inEditor('Looks right')).toBeDefined();
    });
    expect(key('Enter', {}, inEditor('Discard')!).defaultPrevented).toBe(false);
    expect(router.route.query.get('image')).toBe('5');
  });

  // #317 review: a labeled image whose last agent call failed keeps its labels and review; reopened, the editor
  // still says why the call failed, beside its labels' state.
  it('says why the last ask failed on a labeled image that kept its labels, and nothing on one without an error', async () => {
    const list = hub.api.labelImages.bind(hub.api);
    vi.spyOn(hub.api, 'labelImages').mockImplementation(async (...a) => {
      const page = await list(...a);
      return { ...page, images: page.images.map((i) => (i.id === '5' ? { ...i, error: 'the reply has no JSON object' } : i)) };
    });
    const description = () => editor()!.querySelector('header p')?.textContent ?? '';
    await openEditor('5', 'lb_51d0');
    expect(description()).toMatch(/reviewed\. Last ask failed: the reply has no JSON object$/);
    expect(footer()).toMatch(/^Reviewed\./);
    key('j');
    await vi.waitFor(() => {
      flushSync();
      expect(router.route.query.get('image')).toBe('6');
      expect(description()).toMatch(/reviewed$/);
    });
    expect(description()).not.toMatch(/Last ask failed/);
  });

  it('resets selection, undo history, note, and status when closed and reopened on the same image', async () => {
    await openEditor('1');
    click(inEditor('Add box'));
    expect(boxCount()).toBe(1);
    const note = editor()!.querySelector<HTMLInputElement>('input[placeholder*="differently"]')!;
    note.value = 'catch small bolts';
    note.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    // Save so closing has no unsaved changes and a "Saved" status banner is shown.
    click(inEditor('Save'));
    await vi.waitFor(() => {
      flushSync();
      expect(editor()!.querySelector('.dialog-status')?.textContent ?? '').toMatch(/Saved/);
    });
    // Close the editor by clearing `?image=`, then reopen the exact same image.
    router.go('/datasets/label/lb_7c21', { replace: true });
    await vi.waitFor(() => {
      flushSync();
      expect(editor()?.open).toBe(false);
    });
    router.go('/datasets/label/lb_7c21?image=1', { replace: true });
    await vi.waitFor(() => {
      flushSync();
      expect(editor()?.open && editor()?.querySelector('.ed')).toBeTruthy();
    });
    expect(selectedSize()).toBeUndefined();
    expect(editor()!.querySelector<HTMLInputElement>('input[placeholder*="differently"]')!.value).toBe('');
    expect(editor()!.querySelector('.dialog-status')?.textContent?.trim() ?? '').toBe('');
    expect(editor()!.querySelector<HTMLButtonElement>('button[aria-label="Undo"]')!.disabled).toBe(true);
  });

  it('does not navigate away if the leave dialog is cancelled while Save is still in flight', async () => {
    await openEditor('1');
    const answer = deferred<LabelImage>();
    vi.spyOn(hub.api, 'updateLabelImage').mockReturnValue(answer.promise);
    click(inEditor('Add box'));
    key('j');
    expect(asking()).toBeDefined();
    click([...asking()!.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Save'));
    // User closes the leave dialog while save is still pending.
    click(asking()!.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!);
    expect(asking()).toBeUndefined();
    const one = (await hub.api.labelImages('lb_7c21', { limit: 1 })).images[0]!;
    answer.resolve({ ...one, status: 'labeled', source: 'manual', boxes: [{ class: 0, x1: 10, y1: 10, x2: 50, y2: 50 }] });
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
    expect(router.route.query.get('image')).toBe('1');
  });

  it('does not leak a failed Save error onto another image, disables Discard while saving, and guards Remove against double submit/close', async () => {
    await openEditor('1');
    let rejectSave!: (e: unknown) => void;
    const failSave = new Promise<LabelImage>((_, r) => (rejectSave = r));
    vi.spyOn(hub.api, 'updateLabelImage').mockReturnValueOnce(failSave);

    click(inEditor('Add box'));
    const discardBtn = () => inEditor('Discard')!;
    expect(discardBtn().disabled).toBe(false);
    click(inEditor('Save'));
    expect(discardBtn().disabled).toBe(true);

    // Switch to image 2 while image 1's save is still pending, then fail image 1's save.
    key('j');
    // Leave dialog opens because image 1 is dirty; discard to switch to image 2.
    expect(asking()).toBeDefined();
    click([...asking()!.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Discard changes'));
    await vi.waitFor(() => {
      flushSync();
      expect(router.route.query.get('image')).toBe('2');
    });
    rejectSave(new Error('network blip'));
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
    expect(editor()!.querySelector('.dialog-status')?.textContent?.trim() ?? '').toBe('');

    // Open Remove dialog and start delete: re-clicking Remove or trying to close while busy must be blocked.
    const del = deferred<void>();
    const delSpy = vi.spyOn(hub.api, 'deleteLabelImage').mockReturnValue(del.promise);
    click(inEditor('Remove image'));
    const removeDlg = () =>
      [...document.querySelectorAll<HTMLDialogElement>('dialog[open]')].find((d) => d.querySelector('h2')?.textContent?.startsWith('Remove '));
    expect(removeDlg()).toBeDefined();
    const confirmRemove = () => [...removeDlg()!.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Remove image')!;
    const keepBtn = () => [...removeDlg()!.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Keep it')!;
    click(confirmRemove());
    click(confirmRemove());
    expect(delSpy).toHaveBeenCalledTimes(1);
    expect(keepBtn().disabled).toBe(true);
    click(removeDlg()!.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!);
    expect(removeDlg()).toBeDefined();
    del.resolve();
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
    expect(removeDlg()).toBeUndefined();
  });

  it('dismisses the Remove dialog instead of retargeting when URL switches images during pending delete', async () => {
    await openEditor('1');
    const del = deferred<void>();
    vi.spyOn(hub.api, 'deleteLabelImage').mockReturnValue(del.promise);
    click(inEditor('Remove image'));
    const removeDlg = () =>
      [...document.querySelectorAll<HTMLDialogElement>('dialog[open]')].find((d) => d.querySelector('h2')?.textContent?.startsWith('Remove '));
    expect(removeDlg()?.querySelector('h2')?.textContent).toContain('0901');
    click([...removeDlg()!.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Remove image'));

    // Navigate to image 3 via URL while image 1 deletion is still pending: dialog must dismiss and not retarget.
    router.go('/datasets/label/lb_7c21?image=3', { replace: true });
    await vi.waitFor(() => {
      flushSync();
      expect(router.route.query.get('image')).toBe('3');
    });
    expect(removeDlg()).toBeUndefined();

    // Settling image 1's delete must not navigate away from image 3.
    del.resolve();
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
    expect(router.route.query.get('image')).toBe('3');
    expect(removeDlg()).toBeUndefined();
  });

  it('does not redirect back to the labeling job when leaving the page while a delete is pending', async () => {
    await openEditor('1');
    const del = deferred<void>();
    vi.spyOn(hub.api, 'deleteLabelImage').mockReturnValue(del.promise);
    click(inEditor('Remove image'));
    const removeDlg = [...document.querySelectorAll<HTMLDialogElement>('dialog[open]')].find((d) =>
      d.querySelector('h2')?.textContent?.startsWith('Remove '),
    );
    click([...removeDlg!.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Remove image'));

    // Navigate away to /runs and unmount the labeling page before delete settles.
    router.go('/runs', { replace: true });
    unmount(c!);
    c = undefined;
    flushSync();
    expect(router.route.path).toBe('/runs');

    del.resolve();
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
    expect(router.route.path).toBe('/runs');
  });

  it('disables Discard changes in the leave dialog once Save is in flight', async () => {
    await openEditor('1');
    const answer = deferred<LabelImage>();
    vi.spyOn(hub.api, 'updateLabelImage').mockReturnValue(answer.promise);
    click(inEditor('Add box'));
    key('j');
    expect(asking()).toBeDefined();
    const leaveBtn = (label: string) => [...asking()!.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === label)!;
    expect(leaveBtn('Discard changes').disabled).toBe(false);
    click(leaveBtn('Save'));
    // Discard must be disabled while Save is pending so it cannot leave while the draft is still being saved.
    expect(leaveBtn('Discard changes').disabled).toBe(true);
    leaveBtn('Discard changes').click();
    flushSync();
    expect(router.route.query.get('image')).toBe('1');
    const one = (await hub.api.labelImages('lb_7c21', { limit: 1 })).images[0]!;
    answer.resolve({ ...one, status: 'labeled', source: 'manual', boxes: [{ class: 0, x1: 10, y1: 10, x2: 50, y2: 50 }] });
    await vi.waitFor(() => {
      flushSync();
      expect(router.route.query.get('image')).toBe('2');
    });
  });

  it('does not override navigation to /runs when delete settles while router.holdWhile holds the labeling route', async () => {
    await openEditor('1');
    const del = deferred<void>();
    vi.spyOn(hub.api, 'deleteLabelImage').mockReturnValue(del.promise);
    click(inEditor('Remove image'));
    const removeDlg = [...document.querySelectorAll<HTMLDialogElement>('dialog[open]')].find((d) =>
      d.querySelector('h2')?.textContent?.startsWith('Remove '),
    );
    click([...removeDlg!.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Remove image'));

    // Hold displayed route on label-job (as App.tsx does while /runs chunk loads), leaving Editor mounted.
    const release = router.holdWhile((to) => to.name === 'runs');
    try {
      router.go('/runs', { replace: true });
      flushSync();
      expect(router.location.path).toBe('/runs');
      expect(router.route.name).toBe('label-job');

      del.resolve();
      await new Promise((r) => setTimeout(r, 0));
      flushSync();
      expect(router.location.path).toBe('/runs');
    } finally {
      release();
    }
  });

  it('guards Ask agent against double-submit and ignores late settlement after unmount', async () => {
    await openEditor('1');
    let rejectPending!: (err: unknown) => void;
    const pending = new Promise<Awaited<ReturnType<typeof hub.api.startLabeling>>>((_, rej) => {
      rejectPending = rej;
    });
    const startSpy = vi.spyOn(hub.api, 'startLabeling').mockReturnValue(pending);
    const askBtn = [...document.querySelectorAll<HTMLButtonElement>('dialog[open] button')].find((b) => b.textContent?.includes('Ask '))!;
    askBtn.click();
    askBtn.click();
    flushSync();
    expect(startSpy).toHaveBeenCalledTimes(1);

    unmount(c!);
    c = undefined;
    rejectPending(new Error('network down'));
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
  });

  it('reopens the unsaved-changes confirmation when J is pressed immediately after Escape before the async close event', async () => {
    await openEditor('1');
    click(inEditor('Add box'));
    key('j');
    flushSync();
    const confirmDlg = asking()!;
    expect(confirmDlg).toBeDefined();

    // Match Chromium's HTML dialog close steps: `close()` clears `open` synchronously without dispatching `close` yet.
    const closeSpy = vi.spyOn(confirmDlg, 'close').mockImplementation(() => {
      confirmDlg.open = false;
    });
    try {
      // Simulate Chromium Escape on the confirmation dialog: `cancel` fires before the queued `close` event.
      confirmDlg.dispatchEvent(new Event('cancel', { cancelable: true }));
      expect(confirmDlg.open).toBe(false);

      // Press J immediately before either a framework flush or the queued native `close` event runs: exercises
      // synchronous `onDialogClose()`, `closedTick`, and `open.peek()` advancing the unbound prop memo.
      key('j');
      flushSync();
      expect(asking()).toBeDefined();
      expect(confirmDlg.open).toBe(true);

      // A stale queued `close` event arriving after reopen must not dismiss the newly reopened confirmation.
      confirmDlg.dispatchEvent(new Event('close'));
      flushSync();
      expect(asking()).toBeDefined();
      expect(confirmDlg.open).toBe(true);
    } finally {
      closeSpy.mockRestore();
    }
  });

  it('blocks concurrent save while review is in flight and cancels deferred save-and-leave navigation when hub.epoch changes', async () => {
    await openEditor('23', 'lb_51d0');
    // 1. Start review ("Looks right") on an unreviewed labeled image, then add a box and press Ctrl+S while review is
    // in flight: the mutual busy guard must prevent starting a concurrent save.
    const reviewGate = deferred<Awaited<ReturnType<typeof hub.api.reviewLabelImage>>>();
    const reviewSpy = vi.spyOn(hub.api, 'reviewLabelImage').mockReturnValue(reviewGate.promise);
    const saveSpy = vi.spyOn(hub.api, 'updateLabelImage');
    key('Enter');
    flushSync();
    expect(reviewSpy).toHaveBeenCalledTimes(1);
    click(inEditor('Add box'));
    key('s', { ctrlKey: true });
    flushSync();
    expect(saveSpy).not.toHaveBeenCalled();

    // Finish the review cleanly, then open a fresh editor on image 1 to test stale save-and-leave across hub.epoch++.
    const img23 = (await hub.api.labelImages('lb_51d0', { ids: ['23'] })).images[0]!;
    reviewGate.resolve({ ...img23, reviewed: true });
    await new Promise((r) => setTimeout(r, 10));
    unmount(c!);
    c = undefined;
    reviewSpy.mockRestore();
    saveSpy.mockRestore();

    await openEditor('1', 'lb_51d0');
    const one = (await hub.api.labelImages('lb_51d0', { ids: ['1'] })).images[0]!;
    const saveGate = deferred<Awaited<ReturnType<typeof hub.api.updateLabelImage>>>();
    vi.spyOn(hub.api, 'updateLabelImage').mockReturnValue(saveGate.promise);
    click(inEditor('Add box'));
    key('j');
    flushSync();
    const confirmDlg = asking()!;
    expect(confirmDlg).toBeDefined();
    click([...confirmDlg.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Save'));
    flushSync();

    // Advance hub.epoch while the save-and-leave request is in flight, then resolve the old save.
    hub.epoch++;
    flushSync();
    saveGate.resolve({ ...one, status: 'labeled', source: 'manual', reviewed: true });
    await new Promise((r) => setTimeout(r, 10));
    flushSync();
    // Stale save must return false and NEVER execute deferred navigation to image 2.
    expect(router.route.query.get('image')).toBe('1');
  });
});
