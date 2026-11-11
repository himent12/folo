// // Paged, cached access to a dataset's image list for one filter (split, class). The virtualised grid asks for the
// rows it shows; the inspector asks for neighbours and resolves linked image ids. Pages load once and stay cached.

import { signal } from 'fw';
import { hub } from '$lib/api/index';
import type { DatasetImage, Split } from '$lib/api/types';

const PAGE = 100;
/** Page size used when scanning for an image id that is not loaded yet (the API caps `limit` at 500). */
const SCAN = 500;

export class ImagePager {
  #total = signal<number | null>(null);
  #error = signal<unknown>(undefined);
  /** Images matching the filter; null until the first page answers. */
  get total(): number | null {
    return this.#total.get();
  }
  set total(v: number | null) {
    this.#total.set(v);
  }
  get error(): unknown {
    return this.#error.get();
  }
  set error(v: unknown) {
    this.#error.set(v);
  }
  /** Bumps whenever a page lands, so `get` / `indexOf` readers update. */
  #version = signal(0);
  #pages = new Map<number, DatasetImage[]>();
  #inflight = new Map<number, Promise<void>>();
  #index = new Map<string, number>();
  /** Pages whose last request failed; `retry()` loads them again. */
  #failed = new Set<number>();
  #ctrl = new AbortController();

  constructor(
    readonly datasetId: string,
    readonly split: Split | undefined,
    readonly cls: number | undefined,
    /** Index of the first image of `split` in the unfiltered list (lets `find` guess an id's position). */
    readonly splitStart = 0,
  ) {}

  /** Stops pending requests (call when the filter changes). */
  dispose() {
    this.#ctrl.abort();
  }

  get(i: number): DatasetImage | undefined {
    void this.#version.get();
    return this.#pages.get(Math.floor(i / PAGE))?.[i % PAGE];
  }

  indexOf(id: string): number {
    void this.#version.get();
    return this.#index.get(id) ?? -1;
  }

  /** Loads the pages covering images [lo, hi). */
  ensure(lo: number, hi: number): void {
    const max = this.total ?? Infinity;
    // Until the first page answers the total is unknown (and the grid has no rows): always load page 0.
    if (this.total === null) hi = Math.max(hi, 1);
    const a = Math.max(0, Math.floor(lo / PAGE));
    const b = Math.floor((Math.min(hi, max) - 1) / PAGE);
    for (let p = a; p <= b; p++) void this.#load(p);
  }

  /** The image at `i`, loading its page if needed. */
  async at(i: number): Promise<DatasetImage | undefined> {
    if (i < 0 || (this.total !== null && i >= this.total)) return undefined;
    await this.#load(Math.floor(i / PAGE));
    return this.#pages.get(Math.floor(i / PAGE))?.[i % PAGE];
  }

  /** Finds an image id in this filter: loaded pages, then a guess from `img_<n>` ids, then a scan. -1 if absent. */
  async find(id: string): Promise<number> {
    const hit = this.#index.get(id);
    if (hit !== undefined) return hit;
    const n = /(\d+)$/.exec(id);
    if (n && this.cls === undefined) {
      const guess = Number(n[1]) - this.splitStart;
      if (guess >= 0) {
        await this.at(guess).catch(() => undefined);
        const g = this.#index.get(id);
        if (g !== undefined) return g;
      }
    }
    if (this.total === null) await this.#load(0);
    const total = this.total ?? 0;
    for (let off = 0; off < total; off += SCAN) {
      const pages = Array.from({ length: SCAN / PAGE }, (_, k) => off / PAGE + k).filter((p) => p * PAGE < total);
      if (pages.every((p) => this.#pages.has(p))) continue;
      const res = await hub.api.datasetImages(this.datasetId, this.#query(off, SCAN), this.#ctrl.signal);
      for (const p of pages) this.#store(p, res.images.slice((p - off / PAGE) * PAGE, (p - off / PAGE + 1) * PAGE), res.total);
      const f = this.#index.get(id);
      if (f !== undefined) return f;
    }
    return -1;
  }

  #query(offset: number, limit: number) {
    return { split: this.split, class: this.cls, offset, limit };
  }

  #store(p: number, images: DatasetImage[], total: number) {
    if (this.#pages.has(p)) return;
    this.#pages.set(p, images);
    images.forEach((im, k) => this.#index.set(im.id, p * PAGE + k));
    this.#total.set(total);
    this.#version.set((v) => v + 1);
  }

  #load(p: number): Promise<void> {
    if (this.#pages.has(p)) return Promise.resolve();
    const pending = this.#inflight.get(p);
    if (pending) return pending;
    const req = hub.api
      .datasetImages(this.datasetId, this.#query(p * PAGE, PAGE), this.#ctrl.signal)
      .then(
        (res) => {
          this.#failed.delete(p);
          if (this.#failed.size === 0) this.#error.set(undefined);
          this.#store(p, res.images, res.total);
        },
        (e: unknown) => {
          if (e instanceof DOMException && e.name === 'AbortError') return;
          this.#failed.add(p);
          this.#error.set(e);
        },
      )
      .finally(() => this.#inflight.delete(p));
    this.#inflight.set(p, req);
    return req;
  }

  /** Clears the error and loads every page that failed (the first page when nothing has loaded yet). */
  retry() {
    this.#error.set(undefined);
    const pages = this.#failed.size ? [...this.#failed] : [0];
    this.#failed.clear();
    for (const p of pages) void this.#load(p);
  }
}
