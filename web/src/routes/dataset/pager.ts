// Paged, cached access to a dataset's image list for one filter (split, class). The virtualised grid asks for the
// rows it shows; the inspector asks for neighbours and resolves linked image ids. Pages react individually (one signal
// per page, so landing page N never wakes tiles on page M) and are bounded by an LRU cap (`MAX_CACHED_PAGES`), until
// `reload()` says the dataset changed on disk.

import { batch, createSubscriber, isTracking, signal } from 'fw';
import { hub } from '$lib/api/index';
import type { DatasetImage, Split } from '$lib/api/types';

export const PAGE = 100;
/** Page size used when scanning for an image id that is not loaded yet (the API caps `limit` at 500). */
const SCAN = 500;
/** Maximum pages kept cached (LRU): caps memory on large datasets while keeping nearby pages warm. */
export const MAX_CACHED_PAGES = 50;

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
  /** Per-page subscriber functions: `get(i)` and `revisionOf(i)` wake only when their own page updates; auto-pruned on detach. */
  #pageSlots = new Map<number, () => void>();
  /** Active `createSubscriber` update callbacks per observed page (protected from eviction when possible and refetched on reload). */
  #pageUpdaters = new Map<number, Set<() => void>>();
  /** Monotonic counter bumped on every `#bumpPage`, so a detached computed reattaching after a load invalidates immediately. */
  #bumpSeq = 0;
  /** Bumps when `#index` changes, for `indexOf(id)` readers. */
  #indexVersion = signal(0);
  /** Insertion-ordered LRU cache of loaded pages (oldest first, most-recently-used last). */
  #pages = new Map<number, DatasetImage[]>();
  #inflight = new Map<number, Promise<DatasetImage[] | undefined>>();
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
  readonly maxPages: number;

  /** Bumps when a page read before a change is replaced by its new read (an image may have gone or moved). */
  get refreshed(): number {
    return this.#refreshed.get();
  }

  /** How many times the dataset changed on disk while shown (`reload()`). */
  get revision(): number {
    return this.#revision.get();
  }

  /** Number of pages currently held in the LRU cache. */
  get cachedPages(): number {
    return this.#pages.size;
  }

  /**
   * The revision image `i`'s page was read at. Image ids are positions, so its pixels are fetched again under a new
   * one (lib/detect/DatasetImage `rev`), together with the boxes of that read, never before it (an id past the new
   * end would be asked for).
   */
  revisionOf(i: number): number {
    if (i < 0) return 0;
    const p = Math.floor(i / PAGE);
    this.#trackPage(p);
    return this.#pageRev.get(p) ?? 0;
  }

  constructor(
    readonly datasetId: string,
    readonly split: Split | undefined,
    readonly cls: number | undefined,
    /** Index of the first image of `split` in the unfiltered list (lets `find` guess an id's position). */
    readonly splitStart = 0,
    maxPages = MAX_CACHED_PAGES,
  ) {
    this.maxPages = Math.max(1, maxPages);
  }

  /** Stops pending requests (call when the filter changes). */
  dispose() {
    this.#ctrl.abort();
  }

  get(i: number): DatasetImage | undefined {
    if (i < 0) return undefined;
    const p = Math.floor(i / PAGE);
    this.#trackPage(p);
    // A stale page is read again by whoever still shows it (the grid's rows in view, the inspector's image), so only
    // what is on screen is asked for after a change; one whose read failed waits for Retry.
    if (this.#stale.has(p) && !this.#failed.has(p)) void this.#load(p);
    const page = this.#pages.get(p);
    if (page !== undefined) this.#touch(p, page);
    return page?.[i % PAGE];
  }

  /**
   * Cached first-page sample for grid aspect-ratio inference: tracks page 0 reactively without triggering a stale
   * network reload when the user has scrolled deep into the dataset.
   */
  sampleFirstPage(limit = 64): readonly DatasetImage[] | undefined {
    this.#trackPage(0);
    const page = this.#pages.get(0);
    if (!page || page.length === 0 || limit <= 0) return undefined;
    return page.length <= limit ? page : page.slice(0, limit);
  }

  indexOf(id: string): number {
    void this.#indexVersion.get();
    return this.#index.get(id) ?? -1;
  }

  /** Loads the pages covering images [lo, hi), requesting visible pages [visLo, visHi) before overscan. */
  ensure(lo: number, hi: number, visLo = lo, visHi = hi): void {
    const max = this.total ?? Infinity;
    // Until the first page answers the total is unknown (and the grid has no rows): always load page 0.
    if (this.total === null) {
      hi = Math.max(hi, 1);
      visHi = Math.max(visHi, 1);
    }
    if (lo >= hi) return;
    const a = Math.max(0, Math.floor(lo / PAGE));
    const b = Math.floor((Math.min(hi, max) - 1) / PAGE);
    if (a > b) return;
    const va = Math.max(a, Math.floor(visLo / PAGE));
    const vb = Math.min(b, Math.floor((Math.min(visHi, max) - 1) / PAGE));
    if (va <= vb) {
      for (let p = va; p <= vb; p++) void this.#load(p);
      for (let p = a; p < va; p++) void this.#load(p);
      for (let p = vb + 1; p <= b; p++) void this.#load(p);
    } else {
      for (let p = a; p <= b; p++) void this.#load(p);
    }
  }

  /** The image at `i`, loading its page if needed. */
  async at(i: number): Promise<DatasetImage | undefined> {
    if (i < 0 || (this.total !== null && i >= this.total)) return undefined;
    const p = Math.floor(i / PAGE);
    const gen = this.#gen;
    const loaded = await this.#load(p);
    const page = this.#pages.get(p);
    if (page !== undefined) {
      this.#touch(p, page);
      return page[i % PAGE];
    }
    // If another concurrent load evicted `p` before we resumed, still hand back the page this call loaded.
    return gen === this.#gen ? loaded?.[i % PAGE] : undefined;
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
      const found = res.images.findIndex((im) => im.id === id);
      if (found >= 0) {
        const idx = off + found;
        const p = Math.floor(idx / PAGE);
        // Cache the matched page (leaving other cached pages untouched unless capacity is free).
        this.#store(p, res.images.slice((p - off / PAGE) * PAGE, (p - off / PAGE + 1) * PAGE), res.total);
        return idx;
      }
      // Scan misses only fill spare capacity — they must never evict pages the grid or inspector already holds.
      this.#total.set(res.total);
      for (const p of pages) {
        if (this.#pages.size >= this.maxPages) break;
        this.#store(p, res.images.slice((p - off / PAGE) * PAGE, (p - off / PAGE + 1) * PAGE), res.total);
      }
    }
    return -1;
  }

  #trackPage(p: number): void {
    if (!isTracking()) return;
    let watch = this.#pageSlots.get(p);
    if (!watch) {
      let seenSeq = this.#bumpSeq;
      const sub = createSubscriber((update) => {
        let set = this.#pageUpdaters.get(p);
        if (!set) this.#pageUpdaters.set(p, (set = new Set()));
        set.add(update);
        if (!this.#pageSlots.has(p)) this.#pageSlots.set(p, sub);
        if (seenSeq !== this.#bumpSeq) {
          seenSeq = this.#bumpSeq;
          update();
        }
        return () => {
          seenSeq = this.#bumpSeq;
          set!.delete(update);
          if (set!.size === 0) this.#pageUpdaters.delete(p);
          if (this.#pageSlots.get(p) === sub) this.#pageSlots.delete(p);
        };
      });
      watch = sub;
    }
    watch();
  }

  #bumpPage(p: number): void {
    this.#bumpSeq++;
    const set = this.#pageUpdaters.get(p);
    if (set) for (const update of [...set]) update();
  }

  #touch(p: number, images: DatasetImage[]): void {
    this.#pages.delete(p);
    this.#pages.set(p, images);
  }

  #evict(keep?: number): void {
    while (this.#pages.size > this.maxPages) {
      let oldest: number | undefined;
      for (const p of this.#pages.keys()) {
        if (p !== keep && !this.#pageUpdaters.has(p)) {
          oldest = p;
          break;
        }
      }
      if (oldest === undefined) {
        for (const p of this.#pages.keys()) {
          if (p !== keep) {
            oldest = p;
            break;
          }
        }
      }
      if (oldest === undefined) break;
      const dropped = this.#pages.get(oldest);
      this.#pages.delete(oldest);
      this.#stale.delete(oldest);
      this.#pageRev.delete(oldest);
      if (dropped) {
        for (let k = 0; k < dropped.length; k++) {
          const im = dropped[k]!;
          if (this.#index.get(im.id) === oldest * PAGE + k) this.#index.delete(im.id);
        }
      }
      if (this.#pageUpdaters.has(oldest)) this.#bumpPage(oldest);
    }
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
    this.#touch(p, images);
    for (let k = 0; k < images.length; k++) this.#index.set(images[k]!.id, p * PAGE + k);
    this.#evict(p);
    batch(() => {
      this.#total.set(total);
      this.#bumpPage(p);
      this.#indexVersion.set((v) => v + 1);
      if (was) this.#refreshed.set((n) => n + 1);
    });
  }

  #load(p: number): Promise<DatasetImage[] | undefined> {
    if (this.#fresh(p)) {
      const page = this.#pages.get(p)!;
      this.#touch(p, page);
      return Promise.resolve(page);
    }
    const pending = this.#inflight.get(p);
    if (pending) return pending;
    const gen = this.#gen;
    const req = hub.api
      .datasetImages(this.datasetId, this.#query(p * PAGE, PAGE), this.#ctrl.signal)
      .then(
        (res) => {
          if (gen !== this.#gen) return undefined;
          this.#failed.delete(p);
          if (this.#failed.size === 0) this.#error.set(undefined);
          this.#store(p, res.images, res.total);
          return res.images;
        },
        (e: unknown) => {
          if (gen !== this.#gen || (e instanceof DOMException && e.name === 'AbortError')) return undefined;
          // A stale page that cannot be read again keeps what it showed (still stale, so Retry reads it).
          this.#failed.add(p);
          this.#error.set(e);
          return undefined;
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
    for (const p of this.#pageUpdaters.keys()) this.#stale.add(p);
    this.#failed.clear();
    // Readers run again and ask for their pages (`get`); with no row shown (an empty filter), nothing would ask.
    if (!this.total) void this.#load(0);
    batch(() => {
      this.#error.set(undefined);
      this.#revision.set(this.#gen);
      this.#indexVersion.set((v) => v + 1);
      for (const p of this.#stale) this.#bumpPage(p);
    });
  }

  /** Clears the error and loads every page that failed (the first page when nothing has loaded yet). */
  retry() {
    this.#error.set(undefined);
    const pages = this.#failed.size ? [...this.#failed] : [0];
    this.#failed.clear();
    for (const p of pages) void this.#load(p);
  }
}
