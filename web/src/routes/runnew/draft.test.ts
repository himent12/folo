import { describe, expect, it } from 'vitest';
import { readDraft, saveDraft, type Draft } from './draft';

const KEY = 'folo.runNew.draft';

function memory(): Storage {
  const m = new Map<string, string>();
  return {
    get length() {
      return m.size;
    },
    clear: () => m.clear(),
    getItem: (k) => m.get(k) ?? null,
    key: (i) => [...m.keys()][i] ?? null,
    removeItem: (k) => void m.delete(k),
    setItem: (k, v) => void m.set(k, v),
  };
}

const SERVER = 'http://127.0.0.1:7878';

const draft: Draft = {
  server: SERVER,
  query: '',
  datasetId: 'ds_shapes',
  arch: 'folo-s',
  // Cleared fields, kept as typed.
  epochs: null,
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
  name: 'shapes-s-1',
  nameEdited: false,
  augment: { mosaic: 1, mixup: 0, hsv: [0.015, 0.7, 0.4], fliplr: 0.5, scale: 0.5, translate: 0.1, close_mosaic: 10 },
  closeEdited: false,
  advancedOpen: false,
  augOpen: false,
  cloud: { target: 'local', offerId: null, offerChosen: false, hours: null, hoursEdited: false },
};

describe('New run draft', () => {
  it('reads back a draft as it was saved', () => {
    const store = memory();
    saveDraft(draft, () => store);
    expect(readDraft(SERVER, () => store)).toEqual(draft);
  });

  it('drops a draft made against another server, or the mock, and removes it', () => {
    for (const other of ['http://127.0.0.1:7979', 'mock']) {
      const store = memory();
      saveDraft(draft, () => store);
      expect(readDraft(other, () => store)).toBeNull();
      expect(store.getItem(KEY)).toBeNull();
    }
  });

  // review R4: valid JSON of another shape was taken for a draft, and the form threw on it.
  it.each<[string, (d: Draft) => unknown]>([
    ['hsv null', (d) => ({ ...d, augment: { ...d.augment, hsv: null } })],
    ['hsv of two', (d) => ({ ...d, augment: { ...d.augment, hsv: [0.1, 0.2] } })],
    ['hsv of strings', (d) => ({ ...d, augment: { ...d.augment, hsv: ['0.1', 0.2, 0.3] } })],
    ['mosaic out of range', (d) => ({ ...d, augment: { ...d.augment, mosaic: 2 } })],
    ['scale of 1', (d) => ({ ...d, augment: { ...d.augment, scale: 1 } })],
    ['augment an array', (d) => ({ ...d, augment: [] })],
    ['an unknown size', (d) => ({ ...d, arch: 'folo-z' })],
    ['an unknown optimizer', (d) => ({ ...d, optimizer: 'adam' })],
    ['epochs as text', (d) => ({ ...d, epochs: '7' })],
    ['a missing flag', ({ amp: _, ...d }) => d],
    ['cloud without its target', (d) => ({ ...d, cloud: { ...d.cloud, target: 1 } })],
    ['not an object', () => [1, 2]],
    ['null', () => null],
    ['no server', ({ server: _, ...d }) => d],
  ])('drops a draft with %s, and removes it', (_, change) => {
    const store = memory();
    store.setItem(KEY, JSON.stringify(change(draft)));
    expect(readDraft(SERVER, () => store)).toBeNull();
    expect(store.getItem(KEY)).toBeNull();
  });

  it('drops a draft that is not JSON, and storage that cannot be read', () => {
    const store = memory();
    store.setItem(KEY, '{');
    expect(readDraft(SERVER, () => store)).toBeNull();
    expect(store.getItem(KEY)).toBeNull();
    expect(
      readDraft(SERVER, () => {
        throw new Error('blocked');
      }),
    ).toBeNull();
  });
});
