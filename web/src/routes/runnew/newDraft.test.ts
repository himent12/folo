// @vitest-environment happy-dom
import { createRoot, flush, signal } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Draft } from './draft';
import { DRAFT_DEBOUNCE_MS, useDraftPersistence } from './newDraft';

function makeDraft(overrides: Partial<Draft> = {}): Draft {
  return {
    server: 'mock',
    query: '',
    datasetId: 'ds_1',
    arch: 'folo-s',
    epochs: 100,
    batch: '16',
    imgsz: '640',
    imgszChosen: false,
    device: 'auto',
    optimizer: 'auto',
    amp: false,
    e2e: false,
    lr0: 0.01,
    seed: '0',
    seedOver: false,
    weights: 'none',
    name: 'run-1',
    nameEdited: false,
    augment: { mosaic: 1, mixup: 0, hsv: [0.015, 0.7, 0.4], fliplr: 0.5, scale: 0.5, translate: 0.1, close_mosaic: 10 },
    closeEdited: false,
    advancedOpen: false,
    augOpen: false,
    cloud: { target: 'local', offerId: null, offerChosen: false, hours: null, hoursEdited: false },
    ...overrides,
  };
}

function memoryStore() {
  const map = new Map<string, string>();
  let writes = 0;
  const storage = {
    get length() {
      return map.size;
    },
    clear: () => map.clear(),
    key: (i: number) => [...map.keys()][i] ?? null,
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => {
      writes++;
      map.set(k, v);
    },
    removeItem: (k: string) => map.delete(k),
  } as Storage;
  return { storage, writes: () => writes, read: () => map.get('folo.runNew.draft') ?? null };
}

describe('runnew/newDraft', () => {
  afterEach(() => vi.useRealTimers());

  it('writes the initial ready draft immediately, then debounces rapid edits into one storage write', () => {
    vi.useFakeTimers();
    const mem = memoryStore();
    const name = signal('initial');
    const ready = signal(false);

    const dispose = createRoot((d) => {
      useDraftPersistence({
        ready: () => ready.get(),
        build: () => makeDraft({ name: name.get() }),
        store: () => mem.storage,
      });
      return d;
    });

    flush();
    expect(mem.writes()).toBe(0);

    ready.set(true);
    flush();
    expect(mem.writes()).toBe(1);
    expect(JSON.parse(mem.read()!).name).toBe('initial');

    // Rapid edits coalesce into a single debounced write.
    name.set('a');
    flush();
    name.set('ab');
    flush();
    name.set('abc');
    flush();
    expect(mem.writes()).toBe(1);

    vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS - 1);
    expect(mem.writes()).toBe(1);

    vi.advanceTimersByTime(1);
    expect(mem.writes()).toBe(2);
    expect(JSON.parse(mem.read()!).name).toBe('abc');

    dispose();
  });

  it('flushes pending debounced edits synchronously on cleanup/unmount unless overwritten externally', () => {
    vi.useFakeTimers();
    const mem = memoryStore();
    const epochs = signal(100);

    const dispose = createRoot((d) => {
      useDraftPersistence({
        ready: () => true,
        build: () => makeDraft({ epochs: epochs.get() }),
        store: () => mem.storage,
      });
      return d;
    });

    flush();
    expect(mem.writes()).toBe(1);

    epochs.set(42);
    flush();
    expect(mem.writes()).toBe(1);

    dispose();
    expect(mem.writes()).toBe(2);
    expect(JSON.parse(mem.read()!).epochs).toBe(42);

    // An externally overwritten draft is preserved on unmount.
    const epochs2 = signal(50);
    const dispose2 = createRoot((d) => {
      useDraftPersistence({
        ready: () => true,
        build: () => makeDraft({ epochs: epochs2.get() }),
        store: () => mem.storage,
      });
      return d;
    });
    flush();
    epochs2.set(60);
    flush();
    mem.storage.setItem('folo.runNew.draft', '{"external":true}');
    dispose2();
    expect(mem.read()).toBe('{"external":true}');
  });

  it('markStarted cancels pending writes and never clears a newer form’s draft', () => {
    vi.useFakeTimers();
    const mem = memoryStore();
    const name1 = signal('first');
    let ctrl1!: ReturnType<typeof useDraftPersistence>;

    const dispose1 = createRoot((d) => {
      ctrl1 = useDraftPersistence({
        ready: () => true,
        build: () => makeDraft({ name: name1.get() }),
        store: () => mem.storage,
      });
      return d;
    });

    flush();
    expect(mem.read()).not.toBeNull();

    const name2 = signal('second');
    const dispose2 = createRoot((d) => {
      useDraftPersistence({
        ready: () => true,
        build: () => makeDraft({ name: name2.get() }),
        store: () => mem.storage,
      });
      return d;
    });
    flush();
    name2.set('second-edited');
    flush();

    ctrl1.markStarted();
    vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
    expect(JSON.parse(mem.read()!).name).toBe('second-edited');

    dispose1();
    dispose2();
  });
});
