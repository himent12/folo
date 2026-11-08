// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from 'vitest';
import type { Api } from '$lib/api/client';
import { hub } from '$lib/api/index.svelte';
import { MockApi } from '$lib/api/mock';
import { ImagePager } from './pager.svelte';

describe('ImagePager', () => {
  const saved = hub.api;
  afterEach(() => {
    hub.api = saved;
  });

  it('surfaces a failed later page and loads it again on retry', async () => {
    const mock = new MockApi('normal', 0);
    let failOnce = true;
    const api: Partial<Api> = {
      datasetImages: (id, q, s) => {
        if (q.offset === 100 && failOnce) {
          failOnce = false;
          return Promise.reject(new Error('flaky'));
        }
        void s;
        return mock.datasetImages(id, q);
      },
    };
    hub.api = api as Api;
    const pager = new ImagePager('ds_fasteners', undefined, undefined);
    await pager.at(0);
    expect(pager.total).toBe(2400);
    await pager.at(150);
    expect(pager.error).toBeTruthy();
    expect(pager.get(150)).toBeUndefined();
    pager.retry();
    await pager.at(150);
    expect(pager.error).toBeUndefined();
    expect(pager.get(150)?.id).toBeDefined();
  });
});
