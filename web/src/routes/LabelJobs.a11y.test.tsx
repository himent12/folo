// @vitest-environment happy-dom
// Live regions and Clear filters focus return on LabelJobs and label/ImagesArea.
import { flush, signal } from 'fw';
import { render } from '../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { LabelImage } from '$lib/api/types';
import { router } from '$lib/router';
import LabelJobs from './LabelJobs';
import ImagesArea, { type Show } from './label/ImagesArea';
import type { Uploads } from './label/uploads';

let target: HTMLElement;

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  hub.epoch++;
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
  window.matchMedia ??= (() => ({ matches: false, addEventListener() {}, removeEventListener() {} })) as unknown as typeof window.matchMedia;
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  target.remove();
});

describe('LabelJobs and ImagesArea filter live regions', () => {
  it('announces filter counts in one live region and restores focus to search on Clear filters in LabelJobs', async () => {
    router.go('/datasets/label', { replace: true });
    const view = render(() => <LabelJobs />, target);
    try {
      await vi.waitFor(() => {
        flush();
        expect(target.querySelectorAll('tbody tr[data-row]').length).toBeGreaterThan(0);
      });

      const status = () => target.querySelector<HTMLElement>('[data-filter-status]')!;
      expect(status().getAttribute('role')).toBe('status');
      expect(status().getAttribute('aria-live')).toBe('polite');
      expect(status().textContent?.trim()).toBe('');

      const search = target.querySelector<HTMLInputElement>('input[type="search"]')!;
      search.value = 'no-match-xyz';
      search.dispatchEvent(new Event('input', { bubbles: true }));
      flush();

      expect(status().textContent).toMatch(/^Showing 0 of \d+ labeling jobs?\.$/);

      const clearBtn = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Clear filters')!;
      expect(clearBtn).toBeDefined();
      clearBtn.click();
      flush();
      await Promise.resolve();
      flush();

      // Single combined announcement (no second competing live region).
      expect(status().textContent).toMatch(/^Cleared labeling job filters\. Showing all \d+ labeling jobs?\.$/);
      expect(document.activeElement).toBe(search);
    } finally {
      view.unmount();
    }
  });

  it('announces filtered image counts in label/ImagesArea only on filter changes, not on live SSE image updates', () => {
    const show = signal<Show>('all');
    const images = signal<LabelImage[]>([
      { id: 'i1', file: 'a.jpg', width: 640, height: 480, status: 'pending', boxes: [], source: null, error: null, reviewed: false },
      { id: 'i2', file: 'b.jpg', width: 640, height: 480, status: 'labeled', boxes: [], source: 'agent', error: null, reviewed: false },
      { id: 'i3', file: 'c.jpg', width: 640, height: 480, status: 'labeled', boxes: [], source: 'manual', error: null, reviewed: true },
    ]);
    const uploads = { progress: null, result: null, error: null, files: async () => {}, dismiss: () => {} } as unknown as Uploads;
    const view = render(
      () => (
        <ImagesArea
          jobId="lj_1"
          classes={['bolt']}
          images={images.get()}
          show={show.get()}
          uploads={uploads}
          onshow={(s) => show.set(s)}
          onopen={() => {}}
          onbrowse={() => {}}
          onadd={() => {}}
        />
      ),
      target,
    );
    try {
      const status = () => target.querySelector<HTMLElement>('[data-filter-status]')!;
      expect(status().getAttribute('role')).toBe('status');
      expect(status().getAttribute('aria-live')).toBe('polite');
      expect(status().textContent?.trim()).toBe('');

      show.set('review');
      flush();
      expect(status().textContent).toBe('Showing 1 of 3 images.');

      // Live agent progress mutating images does NOT re-announce while the filter stays on 'review'.
      images.set([
        { id: 'i1', file: 'a.jpg', width: 640, height: 480, status: 'labeled', boxes: [], source: 'agent', error: null, reviewed: false },
        { id: 'i2', file: 'b.jpg', width: 640, height: 480, status: 'labeled', boxes: [], source: 'agent', error: null, reviewed: false },
        { id: 'i3', file: 'c.jpg', width: 640, height: 480, status: 'labeled', boxes: [], source: 'manual', error: null, reviewed: true },
      ]);
      flush();
      expect(status().textContent).toBe('Showing 1 of 3 images.');

      show.set('all');
      flush();
      expect(status().textContent).toBe('Showing all 3 images.');
    } finally {
      view.unmount();
    }
  });
});
