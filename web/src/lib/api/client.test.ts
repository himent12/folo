import { afterEach, describe, expect, it, vi } from 'vitest';
import { HttpApi } from './client';

describe('HttpApi.predict', () => {
  afterEach(() => vi.unstubAllGlobals());

  function stub() {
    const calls: { url: string; init: RequestInit }[] = [];
    vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      return new Response(JSON.stringify({ model_id: 'm', image: { width: 1, height: 1 }, boxes: [], timing: {}, device: 'cpu' }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    });
    return calls;
  }

  it('sends conf, iou and max_det for dataset images (JSON)', async () => {
    const calls = stub();
    await new HttpApi('').predict({ kind: 'dataset', dataset_id: 'ds', image_id: 'img_1' }, { model_id: 'm', conf: 0.01, iou: 1, max_det: 1000 });
    const body = JSON.parse(String(calls[0]!.init.body));
    expect(body).toEqual({ model_id: 'm', dataset_id: 'ds', image_id: 'img_1', conf: 0.01, iou: 1, max_det: 1000 });
    expect(calls[0]!.url).toBe('/api/predict');
  });

  it('sends conf, iou and max_det for uploads (multipart)', async () => {
    const calls = stub();
    await new HttpApi('').predict({ kind: 'file', file: new Blob(['x'], { type: 'image/png' }) }, { model_id: 'm', conf: 0.01, iou: 1, max_det: 1000 });
    const fd = calls[0]!.init.body as FormData;
    expect(fd.get('model_id')).toBe('m');
    expect(fd.get('conf')).toBe('0.01');
    expect(fd.get('iou')).toBe('1');
    expect(fd.get('max_det')).toBe('1000');
  });
});
