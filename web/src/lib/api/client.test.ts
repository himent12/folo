import { afterEach, describe, expect, it, vi } from 'vitest';
import { attachmentName, f16FileName, HttpApi, onnxFileName } from './client';

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

// s6-shell-3: a list answer without its list read as an empty list ("No runs yet"), one of the wrong type made screens
// throw (`.filter is not a function`) and show a skeleton forever.
describe('HttpApi: a list answer of the wrong shape is an error', () => {
  afterEach(() => vi.unstubAllGlobals());
  const answer = (body: string) => vi.stubGlobal('fetch', async () => new Response(body, { status: 200, headers: { 'content-type': 'application/json' } }));

  for (const [what, body] of [
    ['the key missing', '{}'],
    ['null', '{"runs":null}'],
    ['an object', '{"runs":{"a":1}}'],
    ['a string', '{"runs":"run_1"}'],
    ['no body', ''],
    ['a bare array', '[]'],
  ] as const) {
    it(`runs: ${what}`, async () => {
      answer(body);
      await expect(new HttpApi('').runs()).rejects.toMatchObject({ code: 'bad_response', request: 'GET /api/runs' });
    });
  }

  it('a list of the right shape comes through', async () => {
    answer('{"benchmarks":[{"id":"b1"}]}');
    expect(await new HttpApi('').benches()).toEqual([{ id: 'b1' }]);
    answer('{"jobs":[]}');
    expect(await new HttpApi('').labelJobs()).toEqual([]);
  });

  // Review round 1: offers of null left "Loading offers…" forever; a page without images showed an empty grid or threw.
  const api = new HttpApi('');
  const reads = {
    offers: () => api.cloudOffers('cx_1'),
    'dataset images': () => api.datasetImages('ds_1', {}),
    'bench images': () => api.benchImages('b_1'),
    'entry images': () => api.benchEntryImages('b_1', 'e_1'),
    // Review round 2: 1000 images and no total stopped LiveLabelJob's paging and showed a truncated job as whole.
    'label images': () => api.labelImages('lj_1'),
  };
  const bad: Record<keyof typeof reads, string[]> = {
    offers: ['{"offers":null,"fetched":"t"}', '{"fetched":"t"}', '[]', ''],
    'dataset images': ['{"total":12}', '{"total":12,"images":null}', '{"images":[]}', '{"total":"12","images":[]}', ''],
    'bench images': ['{"total":12}', '{"images":{}}', ''],
    'entry images': ['{"total":12}', '{"images":[],"total":null}', ''],
    'label images': ['{"images":[{"id":"i"}]}', '{"images":[],"total":"1000"}', '{"total":3}', '{"total":3,"images":{}}', ''],
  };
  for (const [what, bodies] of Object.entries(bad) as [keyof typeof reads, string[]][]) {
    for (const body of bodies) {
      it(`${what}: ${body || 'no body'}`, async () => {
        answer(body);
        await expect(reads[what]()).rejects.toMatchObject({ code: 'bad_response' });
      });
    }
  }

  it('pages and offers of the right shape come through whole', async () => {
    answer('{"offers":[],"fetched":"t"}');
    expect(await reads.offers()).toEqual({ offers: [], fetched: 't' });
    answer('{"total":3,"images":[{"id":"i"}]}');
    expect(await reads['dataset images']()).toEqual({ total: 3, images: [{ id: 'i' }] });
    answer('{"total":0,"images":[]}');
    expect(await reads['entry images']()).toEqual({ total: 0, images: [] });
    answer('{"total":1001,"images":[{"id":"i"}]}');
    expect(await reads['label images']()).toEqual({ total: 1001, images: [{ id: 'i' }] });
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

  it('downloads the f16 file under the server file name', async () => {
    const urls: string[] = [];
    vi.stubGlobal('fetch', async (url: string) => {
      urls.push(url);
      return new Response(new Uint8Array(6), { status: 200, headers: { 'content-disposition': 'attachment; filename="fasteners-n-best-f16.folo"' } });
    });
    const f = await new HttpApi('http://h').f16File('m_1');
    expect(urls[0]).toBe('http://h/api/models/m_1/weights?format=f16');
    expect(f.name).toBe('fasteners-n-best-f16.folo');
    expect(f.size).toBe(6);
  });

  it('names files by the server rule', () => {
    expect(f16FileName('fasteners n/best')).toBe('fasteners_n_best-f16.folo');
    expect(onnxFileName('fasteners n/best', false)).toBe('fasteners_n_best.onnx');
    expect(onnxFileName('shapes-s-best', true)).toBe('shapes-s-best-f16.onnx');
    expect(onnxFileName('shapes-s-best', false, true)).toBe('shapes-s-best-decoded.onnx');
    expect(onnxFileName('shapes-s-best', true, true)).toBe('shapes-s-best-decoded-f16.onnx');
    expect(attachmentName('attachment; filename="a.onnx"')).toBe('a.onnx');
    expect(attachmentName(null)).toBeNull();
  });
});

describe('HttpApi: cloud compute (docs/API.md, Cloud compute)', () => {
  afterEach(() => vi.unstubAllGlobals());

  function stub(answer: (url: string) => Response) {
    const calls: { url: string; init: RequestInit }[] = [];
    vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      return answer(url);
    });
    return calls;
  }
  const json = (v: unknown, status = 200) => new Response(JSON.stringify(v), { status });

  it('unwraps the lists and asks the provider again only on refresh', async () => {
    const calls = stub((url) =>
      url.endsWith('/providers') ? json({ providers: [{ id: 'runpod' }] }) : url.includes('/offers') ? json({ offers: [], fetched: 't' }) : json({ connections: [], instances: [] }),
    );
    const api = new HttpApi('');
    expect(await api.cloudProviders()).toEqual([{ id: 'runpod' }]);
    expect(await api.cloudConnections()).toEqual([]);
    expect(await api.cloudInstances()).toEqual([]);
    await api.cloudOffers('cc 1');
    await api.cloudOffers('cc 1', undefined, true);
    expect(calls.map((c) => c.url)).toEqual([
      '/api/cloud/providers',
      '/api/cloud/connections',
      '/api/cloud/instances',
      '/api/cloud/connections/cc%201/offers',
      '/api/cloud/connections/cc%201/offers?refresh=true',
    ]);
  });

  it('connects with one POST, checks, disconnects and terminates on their own paths', async () => {
    const calls = stub((url) => (url.endsWith('/connections/cc_1') ? new Response(null, { status: 204 }) : json({ id: 'x' })));
    const api = new HttpApi('');
    await api.connectCloud({ provider: 'runpod', name: '', values: { api_key: 'rpa_x' } });
    await api.checkCloudConnection('cc_1');
    await api.disconnectCloud('cc_1');
    await api.terminateCloudInstance('ci_1');
    expect(calls.map((c) => `${c.init.method} ${c.url}`)).toEqual([
      'POST /api/cloud/connections',
      'POST /api/cloud/connections/cc_1/check',
      'DELETE /api/cloud/connections/cc_1',
      'DELETE /api/cloud/instances/ci_1',
    ]);
    expect(JSON.parse(String(calls[0]!.init.body))).toEqual({ provider: 'runpod', name: '', values: { api_key: 'rpa_x' } });
  });

  it('passes a rejected credential on as its code, with the server message', async () => {
    stub(() => json({ error: { code: 'cloud_auth_failed', message: 'RunPod rejected the API key', hint: null } }, 422));
    await expect(new HttpApi('').connectCloud({ provider: 'runpod', values: { api_key: 'bad' } })).rejects.toMatchObject({
      status: 422,
      code: 'cloud_auth_failed',
      message: 'RunPod rejected the API key',
    });
  });
});
