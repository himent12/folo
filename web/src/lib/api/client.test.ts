import { afterEach, describe, expect, it, vi } from 'vitest';
import { attachmentName, HttpApi, onnxFileName } from './client';

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

describe('HttpApi: queue and ONNX (docs/API.md, #129)', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('moves a run with one POST body', async () => {
    const calls: { url: string; init: RequestInit }[] = [];
    vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      return new Response(JSON.stringify({ id: 'run_1', queue_position: 1 }), { status: 200 });
    });
    const r = await new HttpApi('').moveRun('run_1', { action: 'start' });
    expect(r.queue_position).toBe(1);
    expect(calls[0]!.url).toBe('/api/runs/run_1/queue');
    expect(calls[0]!.init.method).toBe('POST');
    expect(JSON.parse(String(calls[0]!.init.body))).toEqual({ action: 'start' });
  });

  it('says a run that already started as run_not_queued (409)', async () => {
    vi.stubGlobal('fetch', async () => new Response(JSON.stringify({ error: { code: 'run_not_queued', message: 'run is running', hint: null } }), { status: 409 }));
    await expect(new HttpApi('').moveRun('run_1', { action: 'up' })).rejects.toMatchObject({ status: 409, code: 'run_not_queued' });
  });

  it('downloads ONNX with the switches that are on as the query, under the server file name', async () => {
    const urls: string[] = [];
    vi.stubGlobal('fetch', async (url: string) => {
      urls.push(url);
      return new Response(new Uint8Array(4), { status: 200, headers: { 'content-disposition': 'attachment; filename="fasteners-n-best-f16.onnx"' } });
    });
    const f = await new HttpApi('http://h').onnxFile('m_1', { decode: true, dynamic: false, half: true });
    expect(urls[0]).toBe('http://h/api/models/m_1/weights?format=onnx&decode=true&half=true');
    expect(f.name).toBe('fasteners-n-best-f16.onnx');
    expect(f.size).toBe(4);
  });

  it('names files by the server rule', () => {
    expect(onnxFileName('fasteners n/best', false)).toBe('fasteners_n_best.onnx');
    expect(onnxFileName('shapes-s-best', true)).toBe('shapes-s-best-f16.onnx');
    expect(attachmentName('attachment; filename="a.onnx"')).toBe('a.onnx');
    expect(attachmentName(null)).toBeNull();
  });
});
