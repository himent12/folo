import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, HttpApi } from './client';
import type { LabelImagesAdded, LabelJob } from './types';
import { chunkFiles, mergeAdded, UploadIncomplete } from './upload';

const MiB = 2 ** 20;
const sized = (...mib: number[]) => mib.map((m, i) => ({ name: `f${i}`, size: m * MiB }));

describe('chunkFiles', () => {
  it('splits by bytes, in order', () => {
    const parts = chunkFiles(sized(20, 20, 20, 5, 30), 48 * MiB, 50);
    expect(parts.map((p) => p.map((f) => f.name))).toEqual([['f0', 'f1'], ['f2', 'f3'], ['f4']]);
  });

  it('splits by count', () => {
    const files = Array.from({ length: 120 }, (_, i) => ({ name: `f${i}`, size: 1000 }));
    expect(chunkFiles(files).map((p) => p.length)).toEqual([50, 50, 20]);
  });

  it('sends a file larger than a request alone, never dropping it', () => {
    expect(chunkFiles(sized(1, 70, 1), 48 * MiB, 50).map((p) => p.map((f) => f.name))).toEqual([['f0'], ['f1'], ['f2']]);
    expect(chunkFiles([])).toEqual([]);
  });
});

describe('mergeAdded', () => {
  it('keeps every image added and skipped, and the job after the last request', () => {
    const job = (n: number) => ({ id: 'j', counts: { images: n } }) as LabelJob;
    const a = { added: [{ id: '1' }], skipped: [{ file: 'x.txt', reason: 'not an image' }], job: job(1) } as LabelImagesAdded;
    const b = { added: [{ id: '2' }, { id: '3' }], skipped: [], job: job(3) } as unknown as LabelImagesAdded;
    expect(mergeAdded([a, b])).toEqual({ added: [{ id: '1' }, { id: '2' }, { id: '3' }], skipped: [{ file: 'x.txt', reason: 'not an image' }], job: job(3) });
    expect(mergeAdded([])).toBeNull();
  });
});

describe('HttpApi.addLabelImages', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('posts one multipart request per chunk, one image part per file, and reports each', async () => {
    const bodies: FormData[] = [];
    vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
      expect(url).toBe('/api/label/jobs/lb_1/images');
      const fd = init.body as FormData;
      bodies.push(fd);
      const added = fd.getAll('image').map((f, i) => ({ id: `${bodies.length}-${i}`, file: (f as File).name }));
      return new Response(JSON.stringify({ added, skipped: [], job: { id: 'lb_1', counts: { images: bodies.length } } }), { status: 201 });
    });
    const files = Array.from({ length: 60 }, (_, i) => new File(['x'], `${i}.jpg`, { type: 'image/jpeg' }));
    const progress: number[] = [];
    const r = await new HttpApi('').addLabelImages('lb_1', files, (p) => progress.push(p.sent));
    expect(bodies.map((b) => b.getAll('image').length)).toEqual([50, 10]);
    expect(progress).toEqual([50, 60]);
    expect(r.added).toHaveLength(60);
    expect(r.job.counts.images).toBe(2);
  });

  it('keeps what the first requests added when a later one fails, with how many files were not sent and why', async () => {
    let n = 0;
    vi.stubGlobal('fetch', async (_url: string, init: RequestInit) => {
      if (++n === 2) return new Response(JSON.stringify({ error: { code: 'payload_too_large', message: 'the request is too large' } }), { status: 413 });
      const added = (init.body as FormData).getAll('image').map((f, i) => ({ id: `${i}`, file: (f as File).name }));
      return new Response(JSON.stringify({ added, skipped: [{ file: 'x.txt', reason: 'not a readable image' }], job: { id: 'lb_1' } }), { status: 201 });
    });
    const files = Array.from({ length: 60 }, (_, i) => new File(['x'], `${i}.jpg`, { type: 'image/jpeg' }));
    const e = await new HttpApi('').addLabelImages('lb_1', files).catch((x: unknown) => x);
    expect(e).toBeInstanceOf(UploadIncomplete);
    const inc = e as UploadIncomplete;
    expect(inc.added.added).toHaveLength(50);
    expect(inc.added.skipped).toEqual([{ file: 'x.txt', reason: 'not a readable image' }]);
    expect(inc.unsent).toBe(10);
    expect(inc.cause).toBeInstanceOf(ApiError);
    expect(inc.message).toMatch(/too large/);
  });

  it("throws the first request's own failure as is: nothing was added", async () => {
    vi.stubGlobal('fetch', async () => new Response(JSON.stringify({ error: { code: 'label_job_not_found', message: 'no job' } }), { status: 404 }));
    const e = await new HttpApi('').addLabelImages('lb_1', [new File(['x'], 'a.jpg')]).catch((x: unknown) => x);
    expect(e).toBeInstanceOf(ApiError);
  });
});
