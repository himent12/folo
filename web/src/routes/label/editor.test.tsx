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
async function openEditor(image = '1') {
  const at = `#/datasets/label/lb_7c21?image=${image}`;
  history.replaceState(null, '', at);
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
});

describe('the labeling editor: a linked image the job does not have', () => {
  it('closes instead of opening empty, and says why', async () => {
    // `?image=` of an image removed since (a shared link, another tab, the agent's own removal): the editor opened as an
    // empty dialog titled "Image", with only Save and Discard.
    const at = '#/datasets/label/lb_7c21?image=img_gone';
    history.replaceState(null, '', at);
    router.route = parse(at);
    const info = vi.spyOn(toast, 'info');
    c = mount(LabelJobPage, { target });
    await vi.waitFor(() => {
      flushSync();
      if (!document.querySelector('.grid, [role=grid], .tile, .empty, .tiles')) throw new Error('loading');
    });
    await vi.waitFor(() => {
      flushSync();
      expect(router.route.query.get('image')).toBeNull();
    });
    expect(editor()?.open).toBeFalsy();
    expect(info).toHaveBeenCalled();
  });
});
