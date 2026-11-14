// Adding images to a labeling job, from this browser (uploads in chunks, with progress) or from a folder on the
// server's machine: what is under way and how the last addition went, shown in place over the images. An upload that
// stops part way shows what its first requests added as well as why the rest was not sent.

import { signal } from 'fw';
import { hub } from '$lib/api/index';
import type { LiveLabelJob } from '$lib/api/liveLabel';
import type { LabelImagesAdded, SkippedImage } from '$lib/api/types';
import { UploadIncomplete, type UploadProgress } from '$lib/api/upload';
import { count } from '$lib/format';

/** A file's reason said of several files ("40 files: the job already has them"). */
const PLURAL: Record<string, string> = { 'the job already has this file': 'the job already has them' };

/**
 * Why a folder added nothing, in one line from what the server skipped: its one reason (no images there, not an image
 * file, already in the job), or each reason with how many files it took.
 */
export function whyNothingAdded(skipped: readonly SkippedImage[]): string {
  const [first] = skipped;
  if (!first) return 'The folder has no images to add.';
  const cap = (r: string) => r.charAt(0).toUpperCase() + r.slice(1);
  if (skipped.length === 1) return `${cap(first.reason)}.`;
  const reasons = new Map<string, number>();
  for (const s of skipped) reasons.set(s.reason, (reasons.get(s.reason) ?? 0) + 1);
  return `${[...reasons].map(([r, n]) => `${count(n, 'file')}: ${n > 1 ? (PLURAL[r] ?? r) : r}`).join('; ')}.`;
}

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

  /**
   * Adds a folder on the server's machine; throws for the dialog that asked to say. An answer that added nothing is
   * the dialog's to explain too (it stays open), so it is returned rather than shown over the images.
   */
  async folder(path: string): Promise<LabelImagesAdded | null> {
    this.error = undefined;
    this.result = null;
    await this.#apply(() => hub.api.importLabelImages(this.jobId(), path));
    // Read again: the addition set it (TypeScript still sees the null set above).
    const answer = this.#result.peek();
    if (answer && !answer.added.length) this.result = null;
    return answer;
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
