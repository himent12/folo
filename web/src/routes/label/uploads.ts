// fw port of uploads.svelte.ts (the same Uploads, on fw signals).
// Adding images to a labeling job, from this browser (uploads in chunks, with progress) or from a folder on the
// server's machine: what is under way and how the last addition went, shown in place over the images. An upload that
// stops part way shows what its first requests added as well as why the rest was not sent.

import { signal } from 'fw';
import { hub } from '$lib/api/index';
import type { LiveLabelJob } from '$lib/api/liveLabel';
import type { LabelImagesAdded } from '$lib/api/types';
import { UploadIncomplete, type UploadProgress } from '$lib/api/upload';

export class Uploads {
  #progress = signal<UploadProgress | null>(null);
  #result = signal<LabelImagesAdded | null>(null);
  #error = signal<unknown>(undefined);
  /** While files are sent: how many of them. */
  get progress(): UploadProgress | null {
    return this.#progress.get();
  }
  set progress(v: UploadProgress | null) {
    this.#progress.set(v);
  }
  /** How the last addition went (added and skipped), until dismissed. */
  get result(): LabelImagesAdded | null {
    return this.#result.get();
  }
  set result(v: LabelImagesAdded | null) {
    this.#result.set(v);
  }
  /** Why the last upload failed (a folder's failure is said in its dialog); an `UploadIncomplete` beside its `result`. */
  get error(): unknown {
    return this.#error.get();
  }
  set error(v: unknown) {
    this.#error.set(v);
  }

  constructor(
    readonly jobId: () => string,
    readonly live: LiveLabelJob,
  ) {}

  /** Uploads `files`; the result (or the error) is shown in place. */
  async files(files: File[]) {
    if (!files.length || this.#progress.peek()) return;
    this.error = undefined;
    this.result = null;
    this.progress = { sent: 0, total: files.length };
    try {
      await this.#apply(() => hub.api.addLabelImages(this.jobId(), files, (p) => (this.progress = p)));
    } catch (e) {
      this.error = e;
    } finally {
      this.progress = null;
    }
  }

  /** Adds a folder on the server's machine; throws for the dialog that asked to say. */
  async folder(path: string) {
    this.error = undefined;
    this.result = null;
    await this.#apply(() => hub.api.importLabelImages(this.jobId(), path));
  }

  dismiss() {
    this.result = null;
    this.error = undefined;
  }

  /** Runs an addition and shows its answer (also the part an incomplete upload added, before it throws). */
  async #apply(call: () => Promise<LabelImagesAdded>) {
    const mark = this.live.mark();
    let answer: LabelImagesAdded | undefined;
    let incomplete: UploadIncomplete | undefined;
    await this.live.act(async () => {
      try {
        answer = await call();
      } catch (e) {
        if (!(e instanceof UploadIncomplete)) throw e;
        incomplete = e;
        answer = e.added;
      }
      return answer.job;
    });
    if (!answer) return;
    this.live.put(answer.added, mark);
    this.result = answer;
    if (incomplete) throw incomplete;
  }
}
