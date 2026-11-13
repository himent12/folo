// // Paged, cached access to a dataset's image list for one filter (split, class). The virtualised grid asks for the
// rows it shows; the inspector asks for neighbours and resolves linked image ids. Pages load once and stay cached,
// until `reload()` says the dataset changed on disk.

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
  /** Pages read before the dataset last changed (`reload()`): shown until their new read answers. */
  #stale = new Set<number>();
  /** Bumped by `reload()`: an answer to a read made before it is dropped, even one already on its way. */
  #gen = 0;
  #revision = signal(0);
  #ctrl = new AbortController();

  /** The `reload()` each page was read after (0: before any). */
  #pageRev = new Map<number, number>();
  #refreshed = signal(0);

  /** Bumps when a page read before a change is replaced by its new read (an image may have gone or moved). */
  get refreshed(): number {
    return this.#refreshed.get();
  }

  /** How many times the dataset changed on disk while shown (`reload()`). */
  get revision(): number {
    return this.#revision.get();
  }

  /**
   * The revision image `i`'s page was read at. Image ids are positions, so its pixels are fetched again under a new
   * one (lib/detect/DatasetImage `rev`), together with the boxes of that read, never before it (an id past the new
   * end would be asked for).
   */
  revisionOf(i: number): number {
    void this.#version.get();
    return this.#pageRev.get(Math.floor(i / PAGE)) ?? 0;
  }

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
    const p = Math.floor(i / PAGE);
    // A stale page is read again by whoever still shows it (the grid's rows in view, the inspector's image), so only
    // what is on screen is asked for after a change; one whose read failed waits for Retry.
    if (this.#stale.has(p) && !this.#failed.has(p)) void this.#load(p);
    return this.#pages.get(p)?.[i % PAGE];
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
    if (hit !== undefined) {
      // A place read before the dataset changed counts once its page is read again (the id may be gone since).
      if (!this.#fresh(Math.floor(hit / PAGE))) await this.#load(Math.floor(hit / PAGE));
      const now = this.#index.get(id);
      if (now !== undefined && this.#fresh(Math.floor(now / PAGE))) return now;
    }
    const n = /(\d+)$/.exec(id);
    if (n && this.cls === undefined) {
      const guess = Number(n[1]) - this.splitStart;
      if (guess >= 0) {
        await this.at(guess).catch(() => undefined);
        const g = this.#index.get(id);
        if (g !== undefined && this.#fresh(Math.floor(g / PAGE))) return g;
      }
    }
    if (this.total === null) await this.#load(0);
    const total = this.total ?? 0;
    for (let off = 0; off < total; off += SCAN) {
      const pages = Array.from({ length: SCAN / PAGE }, (_, k) => off / PAGE + k).filter((p) => p * PAGE < total);
      if (pages.every((p) => this.#fresh(p))) continue;
      const gen = this.#gen;
      const res = await hub.api.datasetImages(this.datasetId, this.#query(off, SCAN), this.#ctrl.signal).catch((e: unknown) => {
        if (gen === this.#gen) throw e;
      });
      // A reload() meanwhile: look again in the dataset as it is now.
      if (!res || gen !== this.#gen) return this.find(id);
      for (const p of pages) this.#store(p, res.images.slice((p - off / PAGE) * PAGE, (p - off / PAGE + 1) * PAGE), res.total);
      const f = this.#index.get(id);
      if (f !== undefined) return f;
    }
    return -1;
  }

  #query(offset: number, limit: number) {
    return { split: this.split, class: this.cls, offset, limit };
  }

  /** Page `p` is loaded and was read since the dataset last changed. */
  #fresh(p: number): boolean {
    return this.#pages.has(p) && !this.#stale.has(p);
  }

  #store(p: number, images: DatasetImage[], total: number) {
    if (this.#fresh(p)) return;
    const was = this.#stale.has(p);
    // A stale page's ids may now name other places (or nothing): they give way to the new read's.
    for (const [k, im] of (this.#pages.get(p) ?? []).entries()) if (this.#index.get(im.id) === p * PAGE + k) this.#index.delete(im.id);
    this.#stale.delete(p);
    this.#pageRev.set(p, this.#gen);
    this.#pages.set(p, images);
    images.forEach((im, k) => this.#index.set(im.id, p * PAGE + k));
    this.#total.set(total);
    this.#version.set((v) => v + 1);
    if (was) this.#refreshed.set((n) => n + 1);
  }

  #load(p: number): Promise<void> {
    if (this.#fresh(p)) return Promise.resolve();
    const pending = this.#inflight.get(p);
    if (pending) return pending;
    const gen = this.#gen;
    const req = hub.api
      .datasetImages(this.datasetId, this.#query(p * PAGE, PAGE), this.#ctrl.signal)
      .then(
        (res) => {
          if (gen !== this.#gen) return;
          this.#failed.delete(p);
          if (this.#failed.size === 0) this.#error.set(undefined);
          this.#store(p, res.images, res.total);
        },
        (e: unknown) => {
          if (gen !== this.#gen || (e instanceof DOMException && e.name === 'AbortError')) return;
          // A stale page that cannot be read again keeps what it showed (still stale, so Retry reads it).
          this.#failed.add(p);
          this.#error.set(e);
        },
      )
      .finally(() => {
        // A request a reload() abandoned must not drop the one that replaced it.
        if (this.#inflight.get(p) === req) this.#inflight.delete(p);
      });
    this.#inflight.set(p, req);
    return req;
  }

  /**
   * The dataset changed on disk: every page is read again when something reads it (the rows in view, the inspector's
   * image), and shows what it had until then, so nothing blinks or jumps. Requests in flight are dropped (they answer
   * for the dataset as it was).
   */
  reload() {
    this.#gen++;
    this.#ctrl.abort();
    this.#ctrl = new AbortController();
    // A page on its way is read again too, by whoever waited for it.
    for (const p of this.#inflight.keys()) this.#stale.add(p);
    this.#inflight.clear();
    for (const p of this.#pages.keys()) this.#stale.add(p);
    for (const p of this.#failed) this.#stale.add(p);
    this.#failed.clear();
    this.#error.set(undefined);
    // Readers run again and ask for their pages (`get`); with no row shown (an empty filter), nothing would ask.
    if (!this.total) void this.#load(0);
    this.#revision.set(this.#gen);
    this.#version.set((v) => v + 1);
  }

  /** Clears the error and loads every page that failed (the first page when nothing has loaded yet). */
  retry() {
    this.#error.set(undefined);
    const pages = this.#failed.size ? [...this.#failed] : [0];
    this.#failed.clear();
    for (const p of pages) void this.#load(p);
  }
}
