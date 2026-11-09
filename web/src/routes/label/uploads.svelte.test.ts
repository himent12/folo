// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, type Api } from '$lib/api/client';
import { hub } from '$lib/api/index.svelte';
import type { LiveLabelJob } from '$lib/api/liveLabel.svelte';
import type { LabelImage, LabelImagesAdded, LabelJob } from '$lib/api/types';
import { UploadIncomplete } from '$lib/api/upload';
import UploadStatus from './UploadStatus.svelte';
import { Uploads } from './uploads.svelte';

const img = (id: string): LabelImage => ({ id, file: `${id}.jpg`, width: 64, height: 48, status: 'pending', boxes: [], source: null, error: null });
const job = { id: 'lb_1', counts: { images: 2 } } as LabelJob;

describe('Uploads', () => {
  const saved = hub.api;
  afterEach(() => {
    hub.api = saved;
    document.body.innerHTML = '';
  });

  it('when a later request fails: shows what the first ones added and skipped, puts those images in the list, and says how many files were not sent and why', async () => {
    const sofar: LabelImagesAdded = { added: [img('1'), img('2')], skipped: [{ file: 'notes.txt', reason: 'not a readable image' }], job };
    const cause = new ApiError(413, 'payload_too_large', 'the request is too large');
    hub.api = { addLabelImages: async () => Promise.reject(new UploadIncomplete(sofar, 50, cause)) } as unknown as Api;
    const put = vi.fn();
    const live = { mark: () => 7, act: async (call: () => Promise<LabelJob>) => call(), put } as unknown as LiveLabelJob;
    const uploads = new Uploads(() => 'lb_1', live);

    await uploads.files([new File(['x'], 'a.jpg')]);
    expect(put).toHaveBeenCalledWith(sofar.added, 7);
    expect(uploads.result).toBe(sofar);
    expect(uploads.error).toBeInstanceOf(UploadIncomplete);

    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = mount(UploadStatus, { target, props: { uploads } });
    flushSync();
    const text = target.textContent ?? '';
    expect(text).toMatch(/2 added/);
    expect(text).toMatch(/1 skipped/);
    expect(text).toMatch(/50 files were not sent/);
    expect(text).toMatch(/The request is too large/i);
    // One way to put it away: the failure's Dismiss.
    const dismiss = [...target.querySelectorAll('button')].filter((b) => /Dismiss/.test(b.textContent ?? '') || b.getAttribute('aria-label') === 'Dismiss');
    expect(dismiss).toHaveLength(1);
    unmount(c);
  });

  it('says a failure of the first request alone, as before', async () => {
    hub.api = { addLabelImages: async () => Promise.reject(new ApiError(404, 'label_job_not_found', 'no job')) } as unknown as Api;
    const put = vi.fn();
    const live = { mark: () => 0, act: async (call: () => Promise<LabelJob>) => call(), put } as unknown as LiveLabelJob;
    const uploads = new Uploads(() => 'lb_1', live);
    await uploads.files([new File(['x'], 'a.jpg')]);
    expect(put).not.toHaveBeenCalled();
    expect(uploads.result).toBeNull();
    expect(uploads.error).toBeInstanceOf(ApiError);
  });
});
