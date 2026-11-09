// A labeling job kept live (docs/API.md, Agent labeling): follows its SSE stream and loads the job, its images and its
// log. As for a run (LiveRun), the stream replays nothing, so we subscribe *before* loading and merge the two: an
// image event that arrived while the images were loading is applied over the answer, and a `job` event over the job
// snapshot. An event may describe a state the answer already passed, but a newer event always follows it, so the view
// settles on the server's. A reconnect loads again, and so does `reload`, which the server sends instead of an event
// per image when it has more to say than the stream carries (a bulk add, a client that fell behind). An action's
// answer (an upload, boxes saved) shows at once, except for an image the stream has told of since the action began.
// The images (at most 10,000 a job) are all kept, in the order they were added: the page filters them and the grid
// draws only the ones in view.

import { untrack } from 'svelte';
import { hub } from './index.svelte';
import { mergeLogs, STREAM_GRACE_MS } from './live.svelte';
import type { LabelImage, LabelJob, LogLine } from './types';

/** Images per request while loading (the API's cap). */
const PAGE = 1000;
/** Loads of every page at most, while removals keep moving the offsets. */
const LOAD_ATTEMPTS = 3;

/** `images` with `img` in place of the image with its id, or added at the end (a new image). */
export function upsertImage(images: readonly LabelImage[], img: LabelImage): LabelImage[] {
  const i = images.findIndex((x) => x.id === img.id);
  if (i < 0) return [...images, img];
  const out = [...images];
  out[i] = img;
  return out;
}

/**
 * A list with changes applied over it (null: removed), in one pass: a changed image keeps its place, a new one goes at
 * the end in the order given (an id given twice keeps its first place and its last value). For a loaded list, what the
 * stream reported since the request was sent; for an action's answer, its images.
 */
export function applyChanges(images: readonly LabelImage[], changes: Iterable<readonly [string, LabelImage | null]>): LabelImage[] {
  const left = new Map(changes);
  if (!left.size) return [...images];
  const out: LabelImage[] = [];
  for (const x of images) {
    if (!left.has(x.id)) out.push(x);
    else {
      const img = left.get(x.id);
      left.delete(x.id);
      if (img) out.push(img);
    }
  }
  for (const img of left.values()) if (img) out.push(img);
  return out;
}

/** Whether a job's agent run is active (`running` or `stopping`). */
export const isActive = (job: LabelJob | undefined) => !!job && job.status !== 'idle';

export class LiveLabelJob {
  job = $state.raw<LabelJob | undefined>(undefined);
  images = $state.raw<LabelImage[]>([]);
  error = $state.raw<unknown>(undefined);
  loading = $state(true);
  connection = $state<'idle' | 'open' | 'reconnecting'>('idle');
  /** When the stream was last heard from, set once it has been down for `STREAM_GRACE_MS`; null while it is up. */
  lost = $state<number | null>(null);
  /** The job's log, oldest first; `version` bumps when it changes (run/Logs reads it in place). */
  logs: LogLine[] = [];
  version = $state(0);
  #retry = $state(0);
  /** Every `job` event bumps it: a snapshot or an action's answer older than the latest event is not applied. */
  #jobSeq = 0;
  /** The image changes heard while a load is in flight, one map per load. */
  #inflight = new Set<Map<string, LabelImage | null>>();
  /** Every image event (`image`, `removed`) bumps it; `#heard` keeps, per image, the value at its latest event. */
  #imageSeq = 0;
  #heard = new Map<string, number>();
  /** Every `removed` event bumps it: a load in pages that saw one may have skipped an image (the offsets shifted). */
  #removals = 0;

  constructor(id: () => string | null) {
    $effect(() => {
      const jobId = id();
      if (hub.mode === 'connecting' || !jobId) return;
      void hub.epoch;
      void this.#retry;
      const api = hub.api;
      let closed = false;
      let heard = Date.now();
      let lostTimer: ReturnType<typeof setTimeout> | undefined;
      let syncTimer: ReturnType<typeof setTimeout> | undefined;
      untrack(() => {
        this.job = undefined;
        this.images = [];
        this.logs = [];
        this.error = undefined;
        this.loading = true;
        this.lost = null;
        this.connection = 'idle';
        this.version++;
      });

      // An image removed between two pages moves the ones after it an offset down, and the next page starts past one
      // of them: a load of several pages that heard a removal loads again (a few times at most; the stream's events
      // are applied over whatever it ends with).
      const allImages = async () => {
        for (let attempt = 1; ; attempt++) {
          const removals = this.#removals;
          const out: LabelImage[] = [];
          let pages = 0;
          for (let total = Infinity; out.length < total; pages++) {
            const page = await api.labelImages(jobId, { offset: out.length, limit: PAGE });
            out.push(...page.images);
            total = page.images.length ? page.total : out.length;
          }
          if (pages <= 1 || this.#removals === removals || attempt >= LOAD_ATTEMPTS) return out;
        }
      };

      // Answers can arrive out of order (a load per reconnect): one older than the last applied is dropped.
      let issued = 0;
      let applied = 0;
      const load = async () => {
        const my = ++issued;
        const seq = this.#jobSeq;
        const changes = new Map<string, LabelImage | null>();
        this.#inflight.add(changes);
        try {
          const [job, images, logs] = await Promise.all([api.labelJob(jobId), allImages(), api.labelLogs(jobId, 500).catch(() => ({ lines: [] }))]);
          if (closed || my < applied) return;
          applied = my;
          if (this.#jobSeq === seq || !this.job) this.job = job;
          this.images = applyChanges(images, changes);
          this.logs = mergeLogs(logs.lines, this.logs);
          this.version++;
          this.loading = false;
          this.error = undefined;
          heard = Date.now();
          if (this.lost !== null) this.lost = this.connection === 'reconnecting' ? heard : null;
        } finally {
          this.#inflight.delete(changes);
        }
      };
      // Re-synchronising (on open, on reconnect) never gives up silently: a gap would stay.
      const sync = (attempt = 0) => {
        clearTimeout(syncTimer);
        load().catch(() => {
          if (!closed) syncTimer = setTimeout(() => sync(attempt + 1), Math.min(15_000, 1000 * 2 ** attempt));
        });
      };
      const image = (id: string, img: LabelImage | null) => {
        if (closed) return;
        heard = Date.now();
        this.lost = null;
        this.#heard.set(id, ++this.#imageSeq);
        if (!img) this.#removals++;
        for (const m of this.#inflight) m.set(id, img);
        this.images = img ? upsertImage(this.images, img) : this.images.filter((x) => x.id !== id);
      };

      const unsub = api.labelEvents(jobId, {
        connection: (s) => {
          if (closed) return;
          // Events between the first load and the stream opening would be lost: load again once it is open.
          if (s === 'open' && this.connection === 'idle') sync();
          this.connection = s;
          if (s === 'open') {
            clearTimeout(lostTimer);
            lostTimer = undefined;
          } else if (this.lost === null && lostTimer === undefined) {
            const since = heard;
            lostTimer = setTimeout(() => {
              lostTimer = undefined;
              if (!closed && this.connection === 'reconnecting') this.lost = since;
            }, STREAM_GRACE_MS);
          }
        },
        job: (j) => {
          if (closed) return;
          heard = Date.now();
          this.lost = null;
          this.#jobSeq++;
          this.job = j;
        },
        image: (img) => image(img.id, img),
        removed: ({ id }) => image(id, null),
        reload: () => {
          if (!closed) sync();
        },
        log: (l) => {
          if (closed) return;
          heard = Date.now();
          this.logs = mergeLogs(this.logs, [l]);
          this.version++;
        },
        reconnect: () => sync(),
      });

      load().catch((e) => {
        if (closed) return;
        this.error = e;
        this.loading = false;
      });

      return () => {
        closed = true;
        clearTimeout(lostTimer);
        clearTimeout(syncTimer);
        unsub();
      };
    });
  }

  /** Loads again from scratch (after a load error). */
  retry() {
    this.#retry++;
  }

  /** Runs an action that answers with the job (start, stop, edit) and shows its answer, unless an event was newer. */
  async act(call: () => Promise<LabelJob>): Promise<LabelJob> {
    const seq = this.#jobSeq;
    const job = await call();
    if (this.#jobSeq === seq && this.job?.id === job.id) this.job = job;
    return job;
  }

  /** Where the stream is now: an action takes a mark before it starts, and `put` gives its answer with it. */
  mark(): number {
    return this.#imageSeq;
  }

  /**
   * Shows images as an action's answer has them (hand-drawn boxes, an upload), except the ones the stream has told of
   * since `since` (the action's `mark()`): that event is as new as the answer or newer (a removal, a later edit).
   */
  put(images: readonly LabelImage[], since: number) {
    const fresh = images.filter((img) => (this.#heard.get(img.id) ?? -1) <= since);
    if (!fresh.length) return;
    const changes = fresh.map((img) => [img.id, img] as const);
    // A load in flight was sent before this answer: it applies the answer over what it gets, as an event.
    for (const m of this.#inflight) for (const [id, img] of changes) m.set(id, img);
    this.images = applyChanges(this.images, changes);
  }

  /** Drops a deleted image (also from a load in flight, which may still have it). */
  drop(id: string) {
    for (const m of this.#inflight) m.set(id, null);
    this.images = this.images.filter((x) => x.id !== id);
  }
}
