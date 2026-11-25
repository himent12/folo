import { describe, expect, it } from 'vitest';
import type { TrainConfig } from '$lib/api/types';
import type { Draft } from './draft';
import { parseInitialQuery, parseQueryInt } from './newPrefill';
import { estimateKey, withServerCliName } from './serverEstimate';

const sampleDraft: Draft = {
  server: 'mock',
  query: 'dataset=ds_a',
  datasetId: 'ds_a',
  arch: 'folo-m',
  epochs: 50,
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
  weights: 'm_old',
  name: 'old-name',
  nameEdited: true,
  augment: { mosaic: 1, mixup: 0, hsv: [0.015, 0.7, 0.4], fliplr: 0.5, scale: 0.5, translate: 0.1, close_mosaic: 10 },
  closeEdited: false,
  advancedOpen: false,
  augOpen: false,
  cloud: { target: 'local', offerId: null, offerChosen: false, hours: null, hoursEdited: false },
};

const sampleConfig: TrainConfig = {
  name: 'run-alpha',
  dataset_id: 'ds_shapes',
  arch: 'folo-s',
  epochs: 100,
  batch: 16,
  imgsz: 640,
  device: 'auto',
  optimizer: 'auto',
  lr0: 0.01,
  weights: null,
  augment: { mosaic: 1, mixup: 0, hsv: [0.015, 0.7, 0.4], fliplr: 0.5, scale: 0.5, translate: 0.1, close_mosaic: 10 },
  seed: 0,
};

describe('runnew/newPrefill', () => {
  it('parses ?weights=, ?dataset=, and ?imgsz= over another link’s draft, while letting an own draft win', () => {
    const q = new URLSearchParams('dataset=ds_b&imgsz=960&weights=m_fast_n');
    expect(parseInitialQuery(q, null, sampleDraft)).toEqual({
      fromId: null,
      datasetId: 'ds_b',
      queryImgsz: true,
      imgsz: '960',
      imgszChosen: true,
      queryWeights: 'm_fast_n',
      weights: 'm_fast_n',
    });

    // When reopening the exact same link (`own` draft), saved edits win over query params.
    expect(parseInitialQuery(q, sampleDraft, sampleDraft)).toEqual({
      fromId: null,
      datasetId: 'ds_a',
      queryImgsz: false,
      imgsz: '640',
      imgszChosen: false,
      queryWeights: null,
      weights: 'm_old',
    });

    // Blank or "none" ?weights= does not schedule a weights lookup.
    expect(parseInitialQuery(new URLSearchParams('weights=none'), null, null).queryWeights).toBeNull();
    expect(parseInitialQuery(new URLSearchParams('weights=   '), null, null).queryWeights).toBeNull();
  });

  it('validates integer query overrides against range and step', () => {
    const q = new URLSearchParams('epochs=40&bad=0&imgsz=960&odd=650&blank=');
    expect(parseQueryInt(q, 'epochs', 1, 10000)).toBe('40');
    expect(parseQueryInt(q, 'bad', 1, 10000)).toBeNull();
    expect(parseQueryInt(q, 'imgsz', 64, 4096, 32)).toBe('960');
    expect(parseQueryInt(q, 'odd', 64, 4096, 32)).toBeNull();
    expect(parseQueryInt(q, 'blank', 1, 100)).toBeNull();
  });
});

describe('runnew/serverEstimate name independence', () => {
  it('omits run name from estimateKey so name edits do not trigger re-estimates', () => {
    expect(estimateKey(sampleConfig)).toBe(estimateKey({ ...sampleConfig, name: 'completely-different-name' }));
    expect(estimateKey(sampleConfig)).not.toBe(estimateKey({ ...sampleConfig, epochs: 50 }));
  });

  it('updates --name in server-supplied CLI commands without a roundtrip', () => {
    const base = 'folo train --data D:/x --weights D:/w.folo --name shapes-n';
    expect(withServerCliName(base, 'shapes-n', 'shapes-n')).toBe(base);
    expect(withServerCliName(base, 'shapes-n', 'custom-run')).toBe('folo train --data D:/x --weights D:/w.folo --name custom-run');
    expect(withServerCliName(base, 'shapes-n', 'my run')).toBe("folo train --data D:/x --weights D:/w.folo --name 'my run'");
    expect(withServerCliName(base, 'shapes-n', '-dash')).toBe("folo train --data D:/x --weights D:/w.folo --name=-dash");
    expect(withServerCliName(base, 'shapes-n', '')).toBe('folo train --data D:/x --weights D:/w.folo');
  });
});
