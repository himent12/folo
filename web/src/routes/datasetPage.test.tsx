// @vitest-environment happy-dom
// The dataset page: what it offers for the dataset it shows.
import { flush as flushSync } from 'fw';
import { render } from '../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import Dataset from './Dataset';

let target: HTMLElement;
let c: { unmount(): void } | undefined;
let api: MockApi;

beforeEach(() => {
  api = new MockApi('normal', 0);
  hub.api = api;
  hub.mode = 'mock';
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
});

afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
  vi.restoreAllMocks();
});

function open(path: string) {
  history.replaceState(history.state, '', `#${path}`);
  router.route = parse(`#${path}`);
  target = document.createElement('div');
  document.body.appendChild(target);
  c = render(() => <Dataset />, target);
  flushSync();
}

const until = <T,>(f: () => T | null | undefined, what: string) =>
  vi.waitFor(() => {
    flushSync();
    const v = f();
    if (!v) throw new Error(`still waiting for ${what}`);
    return v;
  });

describe('Dataset page', () => {
  it('says a dataset with no training images cannot train, and turns Train off with the reason', async () => {
    const read = api.dataset.bind(api);
    vi.spyOn(api, 'dataset').mockImplementation(async (id: string) => ({ ...(await read(id)), splits: { train: 0, val: 0, test: 72 } }));
    open('/datasets/ds_shapes');
    const note = await until(() => target.querySelector('.no-train'), 'the notice');
    expect(note.textContent).toContain('No training images: all 72 images are in the test split.');
    const train = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Train on this dataset'))!;
    expect(train.disabled).toBe(true);
    expect(document.getElementById(train.getAttribute('aria-describedby')!)?.textContent).toBe('No training images in this dataset');
  });

  it('reads the dataset and its images again when the window comes back after the files changed', async () => {
    const read = api.dataset.bind(api);
    let train: number | null = null;
    let answered = 0;
    const one = vi.spyOn(api, 'dataset').mockImplementation(async (id: string) => {
      const d = await read(id);
      answered++;
      return train === null ? d : { ...d, splits: { ...d.splits, train } };
    });
    const list = vi.spyOn(api, 'datasetImages');
    open('/datasets/ds_shapes');
    await until(() => target.querySelector('[data-i="0"]'), 'the grid');
    const header = () => target.querySelector('header')?.textContent ?? '';
    expect(header()).not.toContain('7 train');
    const reads = list.mock.calls.length;

    // Nothing changed: coming back reads the dataset, and the image list stays as it is.
    window.dispatchEvent(new Event('focus'));
    expect(one).toHaveBeenCalledTimes(2);
    await until(() => answered === 2, 'the second answer');
    await new Promise((r) => setTimeout(r));
    flushSync();
    expect(list.mock.calls.length).toBe(reads);

    // Two images went (the folder changed on disk): the header says so, and the rows in view are read again.
    train = 7;
    document.dispatchEvent(new Event('visibilitychange'));
    expect(one).toHaveBeenCalledTimes(3);
    await until(() => header().includes('7 train'), 'the new count');
    await until(() => list.mock.calls.length > reads, 'the image list read again');
  });

  it('reloads images when issue counts change on focus even if dataset summary counts are unchanged', async () => {
    const readIssues = api.datasetIssues.bind(api);
    let extraErr = 0;
    vi.spyOn(api, 'datasetIssues').mockImplementation(async (id, q) => {
      const res = await readIssues(id, q);
      return extraErr && res.counts ? { ...res, counts: { ...res.counts, error: res.counts.error + extraErr } } : res;
    });
    const list = vi.spyOn(api, 'datasetImages');
    open('/datasets/ds_shapes');
    await until(() => target.querySelector('[data-i="0"]'), 'the grid');
    const reads = list.mock.calls.length;

    extraErr = 1;
    window.dispatchEvent(new Event('focus'));
    await until(() => list.mock.calls.length > reads, 'the image list reloaded after issue count changed');
  });

  it("reads the check after the dataset when the window comes back, so it never answers from the server's old index", async () => {
    const read = api.dataset.bind(api);
    let hold: Promise<void> | null = null;
    vi.spyOn(api, 'dataset').mockImplementation(async (id: string) => {
      if (hold) await hold;
      return read(id);
    });
    const issues = vi.spyOn(api, 'datasetIssues');
    open('/datasets/ds_shapes');
    await until(() => target.querySelector('[data-i="0"]'), 'the grid');
    await until(() => issues.mock.calls.length === 1, 'the first check');
    let release!: () => void;
    hold = new Promise<void>((r) => (release = r));
    window.dispatchEvent(new Event('focus'));
    await new Promise((r) => setTimeout(r));
    expect(issues).toHaveBeenCalledTimes(1);
    release();
    await until(() => issues.mock.calls.length === 2, 'the check read again');
  });

  it('closes an open image that left the folder, saying so, when the window comes back', async () => {
    const { total } = await api.datasetImages('ds_shapes', { offset: 0, limit: 1 });
    const last = (await api.datasetImages('ds_shapes', { offset: total - 1, limit: 1 })).images[0]!;
    const read = api.dataset.bind(api);
    const list = api.datasetImages.bind(api);
    let gone = false;
    vi.spyOn(api, 'dataset').mockImplementation(async (id: string) => {
      const d = await read(id);
      return gone ? { ...d, splits: { ...d.splits, test: d.splits.test - 1 } } : d;
    });
    vi.spyOn(api, 'datasetImages').mockImplementation(async (id, q) => {
      const page = await list(id, q);
      return gone ? { total: page.total - 1, images: page.images.filter((im) => im.id !== last.id) } : page;
    });
    const info = vi.spyOn(toast, 'info');
    open(`/datasets/ds_shapes?image=${last.id}`);
    await until(() => document.querySelector('dialog.insp')?.textContent?.includes(last.file.split('/').pop()!), 'the image');
    gone = true;
    window.dispatchEvent(new Event('focus'));
    await until(() => info.mock.calls.length > 0, 'the note');
    expect(info).toHaveBeenCalledWith('Image not in this view', `${last.id} does not match the current filters, or is no longer in the folder.`);
    await until(() => !decodeURIComponent(location.hash).includes('image='), 'the inspector closed');
  });

  it('says the folder is missing when it went while the page was open, instead of showing what it held', async () => {
    const read = api.dataset.bind(api);
    let gone = false;
    vi.spyOn(api, 'dataset').mockImplementation(async (id: string) => {
      if (gone) throw new ApiError(404, 'path_not_found', `the folder of dataset 'shapes' is gone`);
      return read(id);
    });
    open('/datasets/ds_shapes');
    await until(() => target.querySelector('[data-i="0"]'), 'the grid');
    gone = true;
    window.dispatchEvent(new Event('focus'));
    await until(() => target.textContent?.includes('Folder missing'), 'the missing folder');
    expect(target.querySelector('[data-i="0"]')).toBeNull();
  });

  it('says the dataset is not found when another tab removed it while the page was open (stress s1-datasets-1)', async () => {
    const read = api.dataset.bind(api);
    let removed = false;
    vi.spyOn(api, 'dataset').mockImplementation(async (id: string) => {
      if (removed) throw new ApiError(404, 'dataset_not_found', `no dataset '${id}'`);
      return read(id);
    });
    open('/datasets/ds_shapes');
    await until(() => target.querySelector('[data-i="0"]'), 'the grid');
    removed = true;
    window.dispatchEvent(new Event('focus'));
    await until(() => target.textContent?.includes('Dataset not found'), 'the removed dataset');
    expect(target.querySelector('[data-i="0"]')).toBeNull();
    expect(target.textContent).not.toContain('Train on this dataset');
  });

  it('says the dataset cannot be loaded when its folder stopped being a dataset while the page was open', async () => {
    const read = api.dataset.bind(api);
    let broken = false;
    vi.spyOn(api, 'dataset').mockImplementation(async (id: string) => {
      if (broken) throw new ApiError(400, 'invalid_dataset', 'data.yaml: expected a mapping');
      return read(id);
    });
    open('/datasets/ds_shapes');
    await until(() => target.querySelector('[data-i="0"]'), 'the grid');
    broken = true;
    window.dispatchEvent(new Event('focus'));
    await until(() => target.textContent?.includes('Could not load this dataset'), 'the broken dataset');
    expect(target.querySelector('[data-i="0"]')).toBeNull();
    expect(target.textContent).not.toContain('Train on this dataset');
  });

  it('keeps the page when a refresh fails for a reason that is not about the dataset (offline)', async () => {
    const read = api.dataset.bind(api);
    let offline = false;
    const spy = vi.spyOn(api, 'dataset').mockImplementation(async (id: string) => {
      if (offline) throw new TypeError('Failed to fetch');
      return read(id);
    });
    open('/datasets/ds_shapes');
    await until(() => target.querySelector('[data-i="0"]'), 'the grid');
    const before = spy.mock.calls.length;
    offline = true;
    window.dispatchEvent(new Event('focus'));
    await until(() => spy.mock.calls.length > before, 'the refresh');
    // Let the failed read settle.
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
    expect(target.querySelector('[data-i="0"]')).not.toBeNull();
    expect(target.textContent).not.toContain('Could not load this dataset');
  });

  it('offers Train as a link for a dataset that can train', async () => {
    open('/datasets/ds_shapes');
    const train = await until(() => [...target.querySelectorAll('a')].find((a) => a.textContent?.includes('Train on this dataset')), 'Train');
    expect(train.getAttribute('href')).toBe('#/runs/new?dataset=ds_shapes');
    expect(target.querySelector('.no-train')).toBeNull();
  });

  // design/UNIFIED.md §4: the name, the format as a small tag, one meta line; the tabs with their counts.
  it('tags the name with the format and gives the tabs their counts ("Images 2,400", "Statistics · 3 findings")', async () => {
    const ds = await api.dataset('ds_shapes_hard');
    open('/datasets/ds_shapes_hard');
    const tag = await until(() => target.querySelector('.page header .title-line .badge-slot'), 'the format tag');
    expect(tag.textContent).toBe(ds.format);
    expect(target.querySelector('.page header .sub')?.textContent).not.toContain(ds.format);
    const tabs = await until(() => {
      const t = [...target.querySelectorAll('[role="tab"]')].map((x) => x.textContent?.replace(/\s+/g, ' ').trim());
      return t.length === 2 && /findings?$/.test(t[1] ?? '') ? t : null;
    }, 'the tabs with the findings');
    expect(tabs[0]).toMatch(/^Images ?[\d,]+$/);
    expect(tabs[0]).toContain('5,000');
    expect(tabs[1]).toMatch(/^Statistics ?· \d+ findings?$/);
  });
});

describe('Dataset page history', () => {
  const hash = () => decodeURIComponent(location.hash);

  it('opens an image as a history entry that Back (or Close) takes away, leaving the grid', async () => {
    open('/datasets/ds_shapes');
    const cell = await until(() => target.querySelector<HTMLElement>('[data-i="3"]'), 'the grid');
    const depth = history.length;
    cell.click();
    await until(() => hash().includes('image='), 'the image in the URL');
    expect(history.length).toBe(depth + 1);
    const back = vi.spyOn(history, 'back');
    document.querySelector('dialog.insp')!.dispatchEvent(new Event('close'));
    expect(back).toHaveBeenCalledOnce();
    await until(() => hash() === '#/datasets/ds_shapes', 'the grid again');
  });

  it('still knows where an image was opened from after Back and Forward, so Close goes back instead of doubling the grid', async () => {
    open('/datasets/ds_shapes');
    const cell = await until(() => target.querySelector<HTMLElement>('[data-i="2"]'), 'the grid');
    cell.click();
    await until(() => hash().includes('image='), 'the image in the URL');
    history.back();
    await until(() => hash() === '#/datasets/ds_shapes', 'the grid after Back');
    history.forward();
    await until(() => hash().includes('image='), 'the image after Forward');
    const back = vi.spyOn(history, 'back');
    const replace = vi.spyOn(history, 'replaceState');
    await until(() => document.querySelector('dialog.insp'), 'the inspector');
    document.querySelector('dialog.insp')!.dispatchEvent(new Event('close'));
    expect(back).toHaveBeenCalledOnce();
    expect(replace).not.toHaveBeenCalled();
    await until(() => hash() === '#/datasets/ds_shapes', 'the grid again');
  });

  it('replaces an image entry reached by a link when it closes (there is no grid entry behind it)', async () => {
    open('/datasets/ds_shapes?image=1');
    await until(() => document.querySelector('dialog.insp'), 'the inspector');
    const back = vi.spyOn(history, 'back');
    document.querySelector('dialog.insp')!.dispatchEvent(new Event('close'));
    expect(back).not.toHaveBeenCalled();
    await until(() => hash() === '#/datasets/ds_shapes', 'the grid in place of the image');
  });

  it('adds an entry for the first filter only, and Clear goes back to the unfiltered view', async () => {
    open('/datasets/ds_shapes');
    await until(() => target.querySelector('[data-i="0"]'), 'the grid');
    const depth = history.length;
    const pick = (label: string) => [...target.querySelectorAll<HTMLButtonElement>('.ds-toolbar button')].find((b) => b.textContent?.startsWith(label))!;
    pick('Val').click();
    await until(() => hash().includes('split=val'), 'the val filter');
    pick('Train').click();
    await until(() => hash().includes('split=train'), 'the train filter');
    expect(history.length).toBe(depth + 1);
    const back = vi.spyOn(history, 'back');
    target.querySelector<HTMLButtonElement>('button[aria-label="Clear filters"]')!.click();
    expect(back).toHaveBeenCalledOnce();
    await until(() => hash() === '#/datasets/ds_shapes', 'the unfiltered view');
  });

  it('filters images via interactive class-balance chips and clears empty states in one click', async () => {
    const readImages = api.datasetImages.bind(api);
    let emptyClass: number | null = null;
    vi.spyOn(api, 'datasetImages').mockImplementation(async (id, q) => {
      if (emptyClass !== null && q?.class === emptyClass) return { total: 0, images: [] };
      return readImages(id, q);
    });

    open('/datasets/ds_shapes');
    await until(() => target.querySelector('[data-i="0"]'), 'the grid');
    const chips = [...target.querySelectorAll<HTMLButtonElement>('.class-strip button.chip')];
    expect(chips.length).toBeGreaterThan(0);

    // Click first class chip -> updates URL query and highlights chip
    chips[0]!.click();
    await until(() => router.route.query.get('class') !== null, 'class filter set');
    const selectedClass = Number(router.route.query.get('class'));
    expect(chips[0]!.getAttribute('aria-pressed')).toBe('true');

    // Click active chip again -> clears filter back to unfiltered view
    chips[0]!.click();
    await until(() => router.route.query.get('class') === null, 'class filter cleared');
    expect(chips[0]!.getAttribute('aria-pressed')).toBe('false');

    // Empty-state reset when class filter matches 0 images
    emptyClass = selectedClass;
    chips[0]!.click();
    const clearBtn = await until(
      () => [...target.querySelectorAll<HTMLButtonElement>('.empty button')].find((b) => b.textContent?.includes('Clear filters')),
      'empty state Clear filters button',
    );
    clearBtn.focus();
    clearBtn.click();
    await until(() => router.route.query.get('class') === null, 'empty state cleared');
    expect(document.activeElement).toBe(target.querySelector('.ds-toolbar .picker .trigger'));
  });

  it('returns keyboard focus to the class picker when the toolbar Clear filters button unmounts', async () => {
    open('/datasets/ds_shapes?class=0');
    const clearBtn = await until(() => target.querySelector<HTMLButtonElement>('button[aria-label="Clear filters"]'), 'toolbar Clear button');
    clearBtn.focus();
    expect(document.activeElement).toBe(clearBtn);
    clearBtn.click();
    await until(() => router.route.query.get('class') === null, 'class filter cleared');
    expect(target.querySelector('button[aria-label="Clear filters"]')).toBeNull();
    expect(document.activeElement).toBe(target.querySelector('.ds-toolbar .picker .trigger'));
  });

  it('does not hijack navigation if an async missing-image callback resolves after router.location moved away', async () => {
    let releaseImages!: () => void;
    const gate = new Promise<void>((r) => (releaseImages = r));
    vi.spyOn(api, 'datasetImages').mockImplementation(async () => {
      await gate;
      return { total: 0, images: [] };
    });
    const info = vi.spyOn(toast, 'info');
    const hold = router.holdWhile((to) => to.name === 'runs');
    try {
      open('/datasets/ds_shapes?image=img_missing');
      await until(() => target.querySelector('h1'), 'dataset loaded');
      // User clicks away to /runs while the /runs chunk is loading (router.location moved, router.route held on dataset).
      router.go('/runs');
      expect(router.location.path).toBe('/runs');
      releaseImages();
      await new Promise((r) => setTimeout(r, 20));
      flushSync();
      expect(router.location.path).toBe('/runs');
      expect(info).not.toHaveBeenCalled();
    } finally {
      hold();
    }
  });

  it('moves keyboard focus to the Statistics tab when activating the DatasetInsights Stats shortcut', async () => {
    open('/datasets/ds_shapes_hard');
    const statsBtn = await until(() => target.querySelector<HTMLButtonElement>('[data-dataset-insights] .stats-link'), 'insights Stats button');
    statsBtn.focus();
    statsBtn.click();
    await until(() => router.route.query.get('tab') === 'stats', 'switched to stats tab');
    await until(() => document.activeElement?.id === 'ds-tab-stats', 'focus moved to Statistics tab');
  });
});
