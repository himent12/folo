// @vitest-environment happy-dom
// The labeling editor on the job page, over the mock: external-change conflicts (sweep 4 #318), keyboard save
// shortcuts, missing linked images, and browser history navigation for opening/closing/removing images.
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { LabelImage } from '$lib/api/types';
import { boxRevision } from '$lib/label';
import { parse, router } from '$lib/router';
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

describe('the labeling editor: boxes changed elsewhere while a click was on its way (sweep 4 #318)', () => {
  /** Holds the editor's next call of `method` until `go()`; the others (another client's) go through. */
  function hold<K extends 'reviewLabelImage' | 'updateLabelImage'>(method: K) {
    const gate = deferred<void>();
    const real = hub.api[method].bind(hub.api) as (...a: unknown[]) => Promise<LabelImage>;
    const spy = vi.spyOn(hub.api, method).mockImplementation(((...a: unknown[]) => gate.promise.then(() => real(...a))) as never);
    return { real, spy, go: () => gate.resolve() };
  }
  const status = () => editor()!.querySelector('[role="status"]')?.textContent ?? '';

  it('refuses "Looks right" once the boxes it was clicked on were replaced, and shows the new ones to review', async () => {
    await openEditor('23', 'lb_51d0');
    const shown = boxCount();
    expect(shown).toBeGreaterThan(0);
    const review = hold('reviewLabelImage');
    click(inEditor('Looks right'));
    // Meanwhile the CLI clears the boxes and takes the review back (folo label set --clear, folo label review --undo).
    await hub.api.updateLabelImage('lb_51d0', '23', []);
    await review.real('lb_51d0', '23', false);
    await vi.waitFor(() => {
      flushSync();
      expect(boxCount()).toBe(0);
    });
    review.go();
    await vi.waitFor(() => {
      flushSync();
      expect(status()).toMatch(/Changed elsewhere/);
    });
    expect(review.spy.mock.calls[0]![3]).toMatch(new RegExp(`^${shown}-`));
    const page = await hub.api.labelImages('lb_51d0', { limit: 23 });
    expect(page.images[22]).toMatchObject({ id: '23', boxes: [], reviewed: false });
    // Still on this image, its new labels to look at.
    expect(router.route.query.get('image')).toBe('23');
    expect(inEditor('Looks right')).toBeDefined();
  });

  it('refuses a save once another client saved other boxes, and keeps the edits over those', async () => {
    await openEditor('1');
    const save = hold('updateLabelImage');
    click(inEditor('Add box'));
    click(inEditor('Save'));
    const theirs = [
      { class: 0, x1: 1, y1: 1, x2: 20, y2: 20 },
      { class: 1, x1: 30, y1: 30, x2: 60, y2: 60 },
    ];
    await save.real('lb_7c21', '1', theirs);
    save.go();
    await vi.waitFor(() => {
      flushSync();
      expect(status()).toMatch(/Changed elsewhere while you edited/);
    });
    expect((await hub.api.labelImages('lb_7c21', { limit: 1 })).images[0]!.boxes).toEqual(theirs);
    expect(boxCount()).toBe(1);
    expect(footer()).toMatch(/Unsaved changes/);
    // Discard goes back to their boxes, not to the ones this editor opened with.
    click(inEditor('Discard'));
    expect(boxCount()).toBe(2);
  });

  // #334 review: the job's stream, its image events held back from `hold()` until `release()`; `send` tells one now.
  function stream() {
    const real = hub.api.labelEvents.bind(hub.api);
    let held: LabelImage[] | null = null;
    let tell: ((img: LabelImage) => void) | undefined;
    vi.spyOn(hub.api, 'labelEvents').mockImplementation((id, h) => {
      tell = h.image;
      return real(id, { ...h, image: (img) => (held ? held.push(img) : tell?.(img)) });
    });
    return {
      hold: () => (held = []),
      release: () => {
        const due = held ?? [];
        held = null;
        due.forEach((img) => tell?.(img));
      },
      send: (img: LabelImage) => tell?.(img),
    };
  }
  /** Holds the editor's load of image `id` alone (after a refused mark) until `go()`, or fails it with `fail()`. */
  function holdLoad() {
    const gate = deferred<boolean>();
    const list = hub.api.labelImages.bind(hub.api);
    const spy = vi.spyOn(hub.api, 'labelImages').mockImplementation((...a) =>
      a[1]?.ids ? gate.promise.then((ok) => (ok ? list(...a) : Promise.reject(new Error('down')))) : list(...a),
    );
    return { spy, go: () => gate.resolve(true), fail: () => gate.resolve(false) };
  }
  /** Image 23 of the reviewed job open, its boxes cleared and its review taken back by the CLI, the stream not told yet. */
  async function clearedUntold() {
    const events = stream();
    await openEditor('23', 'lb_51d0');
    const shown = boxCount();
    expect(shown).toBeGreaterThan(0);
    events.hold();
    await hub.api.updateLabelImage('lb_51d0', '23', []);
    await hub.api.reviewLabelImage('lb_51d0', '23', false);
    return { events, shown };
  }

  it('after a refused "Looks right" the stream has not explained, loads the new boxes before asking to look again', async () => {
    const { events, shown } = await clearedUntold();
    const load = holdLoad();
    click(inEditor('Looks right'));
    await vi.waitFor(() => expect(load.spy).toHaveBeenCalledWith('lb_51d0', { ids: ['23'] }));
    flushSync();
    // Refused, but not yet asking for a look at boxes it has not loaded.
    expect(status()).not.toMatch(/Changed elsewhere/);
    expect(boxCount()).toBe(shown);
    load.go();
    await vi.waitFor(() => {
      flushSync();
      expect(status()).toMatch(/Changed elsewhere.*Check the labels shown now/);
    });
    expect(boxCount()).toBe(0);
    expect(router.route.query.get('image')).toBe('23');
    // The stream's news, late, changes nothing more.
    events.release();
    flushSync();
    expect(boxCount()).toBe(0);
    expect(inEditor('Looks right')).toBeDefined();
  });

  it('says the new boxes could not be loaded, never that the ones shown are current, when that load fails', async () => {
    const { shown } = await clearedUntold();
    holdLoad().fail();
    click(inEditor('Looks right'));
    await vi.waitFor(() => {
      flushSync();
      expect(editor()!.querySelector('[role="alert"]')?.textContent).toMatch(/Changed elsewhere.*could not be loaded/);
    });
    expect(editor()!.textContent).not.toMatch(/shown now/);
    expect(boxCount()).toBe(shown);
  });

  it('marks boxes saved again as another revision that looks the same here (0 as -0) with that newest revision', async () => {
    const events = stream();
    await openEditor('23', 'lb_51d0');
    const img = (await hub.api.labelImages('lb_51d0', { ids: ['23'] })).images[0]!;
    const zero = { ...img, boxes: img.boxes.map((b, i) => (i ? b : { ...b, x1: 0 })) };
    const negZero = { ...zero, boxes: zero.boxes.map((b, i) => (i ? b : { ...b, x1: -0 })) };
    expect(boxRevision(negZero.boxes)).not.toBe(boxRevision(zero.boxes));
    events.send(zero);
    await new Promise((r) => requestAnimationFrame(r));
    flushSync();
    events.send(negZero);
    await new Promise((r) => requestAnimationFrame(r));
    flushSync();
    // The server has the -0 boxes now and takes a mark made on their revision only.
    const review = vi.spyOn(hub.api, 'reviewLabelImage').mockImplementation(async (_job, _id, reviewed, expect) => {
      if (expect !== boxRevision(negZero.boxes)) throw new ApiError(409, 'label_boxes_changed', 'other boxes');
      return { ...negZero, reviewed };
    });
    click(inEditor('Looks right'));
    await vi.waitFor(() => {
      flushSync();
      expect(router.route.query.get('image')).not.toBe('23');
    });
    expect(review).toHaveBeenCalledTimes(1);
  });
});

describe('the labeling editor: saving from the keyboard (v-models-label N3)', () => {
  it('saves with Ctrl+S and stays, never the browser’s Save page', async () => {
    await openEditor('23', 'lb_51d0');
    const sent = vi.spyOn(hub.api, 'updateLabelImage');
    // Nothing to save: the key is still the editor's.
    expect(key('s', { ctrlKey: true }).defaultPrevented).toBe(true);
    expect(sent).not.toHaveBeenCalled();
    click(inEditor('Add box'));
    expect(footer()).toMatch(/Unsaved changes\. .* saves them and opens the next image to review\./);
    expect(key('s', { metaKey: true }).defaultPrevented).toBe(true);
    await vi.waitFor(() => expect(sent).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => {
      flushSync();
      expect(footer()).toMatch(/^Reviewed\./);
    });
    expect(router.route.query.get('image')).toBe('23');
  });

  it('saves with Enter when there are changes, and opens the next image to review', async () => {
    await openEditor('23', 'lb_51d0');
    click(inEditor('Add box'));
    expect(key('Enter').defaultPrevented).toBe(true);
    await vi.waitFor(() => expect(router.route.query.get('image')).toBe('24'));
    const page = await hub.api.labelImages('lb_51d0', { limit: 24 });
    expect(page.images[22]).toMatchObject({ id: '23', source: 'manual', reviewed: true });
    expect(page.images[22]!.boxes.length).toBeGreaterThan(0);
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

describe('LabelJob editor history', () => {
  const hash = () => decodeURIComponent(location.hash);

  it('pushes a history entry when opening an image from the grid so Back and Close return to the job', async () => {
    const at = '#/datasets/label/lb_7c21';
    history.replaceState(history.state, '', at);
    router.route = parse(at);
    c = mount(LabelJobPage, { target });
    const cell = await vi.waitFor(() => {
      flushSync();
      const el = target.querySelector<HTMLElement>('[data-i="0"]');
      if (!el) throw new Error('waiting for grid tile');
      return el;
    });
    const depth = history.length;
    cell.click();
    await vi.waitFor(() => {
      flushSync();
      if (!hash().includes('image=') || !editor()?.open) throw new Error('waiting for editor open');
    });
    expect(history.length).toBe(depth + 1);

    // Stepping to the next image replaces the entry instead of pushing another one.
    key('j');
    await vi.waitFor(() => {
      flushSync();
      expect(router.route.query.get('image')).toBe('2');
    });
    expect(history.length).toBe(depth + 1);

    // Closing goes Back to the grid entry rather than leaving a spare history entry.
    const back = vi.spyOn(history, 'back');
    editor()!.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!.click();
    expect(back).toHaveBeenCalledOnce();
    await vi.waitFor(() => {
      flushSync();
      expect(hash()).toBe('#/datasets/label/lb_7c21');
    });
    expect(editor()?.open).toBeFalsy();
  });

  it('closes on browser Back and preserves originating view across Back and Forward', async () => {
    const at = '#/datasets/label/lb_7c21';
    history.replaceState(history.state, '', at);
    router.route = parse(at);
    c = mount(LabelJobPage, { target });
    const cell = await vi.waitFor(() => {
      flushSync();
      const el = target.querySelector<HTMLElement>('[data-i="0"]');
      if (!el) throw new Error('waiting for grid tile');
      return el;
    });
    cell.click();
    await vi.waitFor(() => {
      flushSync();
      if (!hash().includes('image=') || !editor()?.open) throw new Error('waiting for editor open');
    });
    history.back();
    await vi.waitFor(() => {
      flushSync();
      expect(hash()).toBe('#/datasets/label/lb_7c21');
    });
    expect(editor()?.open).toBeFalsy();

    history.forward();
    await vi.waitFor(() => {
      flushSync();
      expect(hash()).toContain('image=');
      expect(editor()?.open).toBe(true);
    });
    const back = vi.spyOn(history, 'back');
    const replace = vi.spyOn(history, 'replaceState');
    editor()!.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!.click();
    expect(back).toHaveBeenCalledOnce();
    expect(replace).not.toHaveBeenCalled();
    await vi.waitFor(() => {
      flushSync();
      expect(hash()).toBe('#/datasets/label/lb_7c21');
    });
  });

  it('replaces in place when closing an image opened directly via link (no grid entry behind it)', async () => {
    await openEditor('1');
    const back = vi.spyOn(history, 'back');
    editor()!.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!.click();
    expect(back).not.toHaveBeenCalled();
    await vi.waitFor(() => {
      flushSync();
      expect(hash()).toBe('#/datasets/label/lb_7c21');
    });
  });

  it('removes the last image in a job opened from the grid with a single Back, no bogus toast, and scrubs Forward', async () => {
    // Only 1 image in the job so removing it leaves no next/previous image and closes the editor.
    const realList = hub.api.labelImages.bind(hub.api);
    let dropped = false;
    vi.spyOn(hub.api, 'labelImages').mockImplementation(async (jobId, q) => {
      const page = await realList(jobId, q);
      const only = dropped ? [] : page.images.slice(0, 1);
      return { total: only.length, images: only };
    });
    vi.spyOn(hub.api, 'deleteLabelImage').mockImplementation(async () => {
      dropped = true;
    });

    const at = '#/datasets/label/lb_7c21';
    history.replaceState(history.state, '', at);
    router.route = parse(at);
    c = mount(LabelJobPage, { target });
    const cell = await vi.waitFor(() => {
      flushSync();
      const el = target.querySelector<HTMLElement>('[data-i="0"]');
      if (!el) throw new Error('waiting for sole grid tile');
      return el;
    });
    cell.click();
    await vi.waitFor(() => {
      flushSync();
      if (!hash().includes('image=1') || !editor()?.open) throw new Error('waiting for editor open');
    });

    const back = vi.spyOn(history, 'back');
    const info = vi.spyOn(toast, 'info');
    click(inEditor('Remove image'));
    const confirm = await vi.waitFor(() => {
      flushSync();
      const d = [...document.querySelectorAll<HTMLDialogElement>('dialog[open]')].find((el) => el !== editor() && el.textContent?.includes('Remove'));
      const b = d ? [...d.querySelectorAll<HTMLButtonElement>('button')].find((x) => x.textContent?.trim() === 'Remove image') : undefined;
      if (!b) throw new Error('waiting for Remove confirm');
      return b;
    });
    confirm.click();

    await vi.waitFor(() => {
      flushSync();
      expect(hash()).toBe('#/datasets/label/lb_7c21');
    });
    // Exactly one history.back() (stays on the job grid, never exits to /datasets/label) and no spurious toast.
    expect(back).toHaveBeenCalledTimes(1);
    expect(info).not.toHaveBeenCalled();

    // Pressing Forward into the deleted image scrubs the dead URL in place instead of bouncing back.
    back.mockClear();
    history.forward();
    await vi.waitFor(() => {
      flushSync();
      expect(info).toHaveBeenCalledWith('Image not in this job', '1 has been removed from it.');
    });
    await vi.waitFor(() => {
      flushSync();
      expect(hash()).toBe('#/datasets/label/lb_7c21');
    });
    expect(back).not.toHaveBeenCalled();
  });

  it('removes the last image even with unsaved edits without triggering a save prompt or stranding the editor', async () => {
    const realList = hub.api.labelImages.bind(hub.api);
    let dropped = false;
    vi.spyOn(hub.api, 'labelImages').mockImplementation(async (jobId, q) => {
      const page = await realList(jobId, q);
      const only = dropped ? [] : page.images.slice(0, 1);
      return { total: only.length, images: only };
    });
    vi.spyOn(hub.api, 'deleteLabelImage').mockImplementation(async () => {
      dropped = true;
    });

    const at = '#/datasets/label/lb_7c21';
    history.replaceState(history.state, '', at);
    router.route = parse(at);
    c = mount(LabelJobPage, { target });
    const cell = await vi.waitFor(() => {
      flushSync();
      const el = target.querySelector<HTMLElement>('[data-i="0"]');
      if (!el) throw new Error('waiting for sole grid tile');
      return el;
    });
    cell.click();
    await vi.waitFor(() => {
      flushSync();
      if (!hash().includes('image=1') || !editor()?.open) throw new Error('waiting for editor open');
    });

    // Make an unsaved edit before deleting the image.
    click(inEditor('Add box'));
    expect(footer()).toMatch(/Unsaved changes/);

    const back = vi.spyOn(history, 'back');
    const info = vi.spyOn(toast, 'info');
    click(inEditor('Remove image'));
    const confirm = await vi.waitFor(() => {
      flushSync();
      const d = [...document.querySelectorAll<HTMLDialogElement>('dialog[open]')].find((el) => el !== editor() && el.textContent?.includes('Remove'));
      const b = d ? [...d.querySelectorAll<HTMLButtonElement>('button')].find((x) => x.textContent?.trim() === 'Remove image') : undefined;
      if (!b) throw new Error('waiting for Remove confirm');
      return b;
    });
    confirm.click();

    await vi.waitFor(() => {
      flushSync();
      expect(hash()).toBe('#/datasets/label/lb_7c21');
    });
    expect(asking()).toBeUndefined();
    expect(editor()?.open).toBeFalsy();
    expect(back).toHaveBeenCalledTimes(1);
    expect(info).not.toHaveBeenCalled();
  });
});

describe('the labeling editor: unsaved draft while another tab starts relabeling (playtest R11)', () => {
  function captureStream() {
    const real = hub.api.labelEvents.bind(hub.api);
    let onImage: ((img: LabelImage) => void) | undefined;
    vi.spyOn(hub.api, 'labelEvents').mockImplementation((id, h) => {
      onImage = h.image;
      return real(id, h);
    });
    return { send: (img: LabelImage) => onImage?.(img) };
  }

  const promptBtn = (label: string) =>
    [...(asking()?.querySelectorAll<HTMLButtonElement>('button') ?? [])].find((b) => b.textContent?.trim() === label);
  const dismissPrompt = () => {
    click(asking()!.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!);
    expect(asking()).toBeUndefined();
  };

  it('asks before discarding unsaved boxes on Close, Esc, J/K, route leave, and beforeunload while locked, and lets Save work once unlocked', async () => {
    const sse = captureStream();
    await openEditor('5', 'lb_51d0');
    const orig = (await hub.api.labelImages('lb_51d0', { ids: ['5'] })).images[0]!;
    const before = boxCount();
    expect(before).toBeGreaterThan(0);

    // Delete one box without saving.
    const firstBox = editor()!.querySelector<HTMLButtonElement>('.boxes button')!;
    click(firstBox);
    key('Delete');
    expect(boxCount()).toBe(before - 1);

    // Another tab starts relabeling: SSE marks image 5 as `labeling` (locked).
    sse.send({ ...orig, status: 'labeling' });
    await vi.waitFor(() => {
      flushSync();
      expect(editor()!.textContent).toContain('The agent is labeling this image');
    });
    expect(boxCount()).toBe(before - 1);
    expect(inEditor('Save')!.disabled).toBe(true);
    expect(footer()).not.toContain('saves them');

    // Ctrl+S and Enter must not save while locked.
    const saveSpy = vi.spyOn(hub.api, 'updateLabelImage');
    key('s', { ctrlKey: true });
    key('Enter');
    expect(saveSpy).not.toHaveBeenCalled();

    // 1. Close button asks instead of silently closing.
    click(editor()!.querySelector<HTMLButtonElement>('header button[aria-label="Close"]')!);
    expect(asking()).toBeDefined();
    expect(asking()!.textContent).toContain('The agent is labeling this image right now');
    expect(promptBtn('Save')!.disabled).toBe(true);
    expect(promptBtn('Discard changes')!.disabled).toBe(false);
    dismissPrompt();
    expect(router.route.query.get('image')).toBe('5');
    expect(boxCount()).toBe(before - 1);

    // 2. Esc (native dialog cancel) asks and dismissing keeps draft.
    editor()!.dispatchEvent(new Event('cancel', { cancelable: true }));
    flushSync();
    expect(asking()).toBeDefined();
    expect(promptBtn('Save')!.disabled).toBe(true);
    dismissPrompt();
    expect(boxCount()).toBe(before - 1);

    // 3. J and K ask and dismissing keeps draft on image 5.
    key('j');
    expect(asking()).toBeDefined();
    expect(promptBtn('Save')!.disabled).toBe(true);
    dismissPrompt();
    expect(router.route.query.get('image')).toBe('5');
    key('k');
    expect(asking()).toBeDefined();
    dismissPrompt();
    expect(router.route.query.get('image')).toBe('5');

    // 4. Route navigation (as browser Back triggers router.guard) asks and dismissing keeps draft.
    router.go('/datasets/label/lb_51d0');
    flushSync();
    expect(asking()).toBeDefined();
    expect(promptBtn('Save')!.disabled).toBe(true);
    dismissPrompt();
    expect(router.route.query.get('image')).toBe('5');
    expect(boxCount()).toBe(before - 1);

    // 5. beforeunload still warns while locked with unsaved edits.
    const unload = new Event('beforeunload', { cancelable: true });
    window.dispatchEvent(unload);
    expect(unload.defaultPrevented).toBe(true);

    // 6. Agent finishes and releases the image: Save becomes enabled and saves the user's draft boxes cleanly.
    sse.send({ ...orig, status: 'labeled' });
    await vi.waitFor(() => {
      flushSync();
      expect(inEditor('Save')!.disabled).toBe(false);
    });
    click(inEditor('Save'));
    await vi.waitFor(() => {
      flushSync();
      expect(saveSpy).toHaveBeenCalledTimes(1);
      expect(footer()).not.toContain('Unsaved changes');
    });
    expect(saveSpy.mock.calls[0]![2]).toHaveLength(before - 1);
  });

  it('explicitly discarding from the prompt while locked navigates away without saving', async () => {
    const sse = captureStream();
    await openEditor('5', 'lb_51d0');
    const orig = (await hub.api.labelImages('lb_51d0', { ids: ['5'] })).images[0]!;
    const before = boxCount();
    click(editor()!.querySelector<HTMLButtonElement>('.boxes button')!);
    key('Delete');
    expect(boxCount()).toBe(before - 1);

    sse.send({ ...orig, status: 'labeling' });
    await vi.waitFor(() => {
      flushSync();
      expect(editor()!.textContent).toContain('The agent is labeling this image');
    });
    const saveSpy = vi.spyOn(hub.api, 'updateLabelImage');

    // Click Close -> prompt appears -> click Discard changes -> closes editor and leaves image 5.
    click(editor()!.querySelector<HTMLButtonElement>('header button[aria-label="Close"]')!);
    expect(asking()).toBeDefined();
    click(promptBtn('Discard changes'));
    await vi.waitFor(() => {
      flushSync();
      expect(router.route.query.get('image')).toBeNull();
    });
    expect(asking()).toBeUndefined();
    expect(saveSpy).not.toHaveBeenCalled();
  });
});
