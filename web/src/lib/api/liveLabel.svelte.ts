// A labeling job kept live (docs/API.md, Agent labeling): follows its SSE stream and loads the job, its images and its
// log. As for a run (LiveRun), the stream replays nothing, so we subscribe *before* loading and merge the two: an
// image event that arrived while the images were loading is applied over the answer, and a `job` event over the job
// snapshot. An event may describe a state the answer already passed, but a newer event always follows it, so the view
// settles on the server's. A reconnect loads again. The images (at most 10,000 a job) are all kept, in the order they
// were added: the page filters them and the grid draws only the ones in view.

import { untrack } from 'svelte';
import { hub } from './index.svelte';
import { mergeLogs, STREAM_GRACE_MS } from './live.svelte';
import type { LabelImage, LabelJob, LogLine } from './types';

/** Images per request while loading (the API's cap). */
const PAGE = 1000;

/** `images` with `img` in place of the image with its id, or added at the end (a new image). */
export function upsertImage(images: readonly LabelImage[], img: LabelImage): LabelImage[] {
  const i = images.findIndex((x) => x.id === img.id);
  if (i < 0) return [...images, img];
  const out = [...images];
  out[i] = img;
  return out;
}

/** A loaded list with what the stream reported since the request was sent applied over it (null: removed). */
export function applyChanges(images: readonly LabelImage[], changes: ReadonlyMap<string, LabelImage | null>): LabelImage[] {
  let out = [...images];
  for (const [id, img] of changes) out = img ? upsertImage(out, img) : out.filter((x) => x.id !== id);
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

      const allImages = async () => {
        const out: LabelImage[] = [];
        for (let total = Infinity; out.length < total; ) {
          const page = await api.labelImages(jobId, { offset: out.length, limit: PAGE });
          out.push(...page.images);
          total = page.images.length ? page.total : out.length;
        }
        return out;
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

  /** Shows an image as an action's answer has it (hand-drawn boxes, an upload). */
  put(...images: LabelImage[]) {
    let out = this.images;
    for (const img of images) out = upsertImage(out, img);
    this.images = out;
  }

  /** Drops a deleted image. */
  drop(id: string) {
    this.images = this.images.filter((x) => x.id !== id);
  }
}
