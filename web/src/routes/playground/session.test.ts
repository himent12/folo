import { describe, expect, it } from 'vitest';
import { hub } from '$lib/api/index';
import type { Dataset, DatasetImage } from '$lib/api/types';
import { fromUrl, playgroundPath, session, type Session } from './session';

const ds = { id: 'ds_1', name: 'shapes' } as Dataset;
const img = { id: 'im 7', file: 'a.jpg', split: 'val' } as DatasetImage;

describe('Playground session', () => {
  it('belongs to the server it was made against: another one (or the mock) starts without it', () => {
    const s = { source: null, pred: null, runOn: 'server', conf: 0.4, iou: 0.7, maxDet: 300, labels: true, scores: true, truth: false, hidden: [] } satisfies Session;
    hub.mode = 'live';
    session.set(s);
    expect(session.get()).toBe(s);
    hub.mode = 'mock';
    expect(session.get()).toBeNull();
    hub.mode = 'live';
    expect(session.get()).toBe(s);
    session.set(null);
  });
});

describe('Playground URL', () => {
  it('names the dataset image and the thresholds that differ from the defaults, and reads them back', () => {
    const path = playgroundPath({ model: 'm_a', source: { kind: 'dataset', ds, img, at: 12 }, conf: 0.5, iou: 0.7, maxDet: 300 });
    expect(path).toBe('/playground?model=m_a&dataset=ds_1&image=im+7&split=val&at=12&conf=0.5');
    expect(fromUrl(new URLSearchParams(path.split('?')[1]))).toEqual({
      model: 'm_a',
      image: { dataset: 'ds_1', image: 'im 7', split: 'val', at: 12 },
      conf: 0.5,
      iou: null,
      maxDet: null,
    });
  });

  it('leaves out an uploaded file and the defaults', () => {
    const file = { kind: 'file' as const, file: new Blob(), name: 'x.png', url: 'blob:x', width: 1, height: 1 };
    expect(playgroundPath({ model: null, source: file, conf: 0.25, iou: 0.7, maxDet: 300 })).toBe('/playground');
  });

  it('ignores values out of range or not numbers', () => {
    expect(fromUrl(new URLSearchParams('conf=2&iou=x&max=0&dataset=ds_1&image=i&split=nope'))).toEqual({ model: null, image: null, conf: null, iou: null, maxDet: null });
  });

  it('preserves an in-flight imageRef while source is still null', () => {
    expect(
      playgroundPath({
        model: 'm_b',
        source: null,
        imageRef: { dataset: 'ds_1', image: 'im 7', split: 'val', at: 12 },
        conf: 0.25,
        iou: 0.7,
        maxDet: 300,
      }),
    ).toBe('/playground?model=m_b&dataset=ds_1&image=im+7&split=val&at=12');
  });
});
