// Adding images to a labeling job, from this browser (uploads in chunks, with progress) or from a folder on the
// server's machine: what is under way and how the last addition went, shown in place over the images.

import { hub } from '$lib/api/index.svelte';
import type { LiveLabelJob } from '$lib/api/liveLabel.svelte';
import type { LabelImagesAdded } from '$lib/api/types';
import type { UploadProgress } from '$lib/api/upload';

export class Uploads {
  /** While files are sent: how many of them. */
  progress = $state<UploadProgress | null>(null);
  /** How the last addition went (added and skipped), until dismissed. */
  result = $state.raw<LabelImagesAdded | null>(null);
  /** Why the last upload failed (a folder's failure is said in its dialog). */
  error = $state.raw<unknown>(undefined);

  constructor(
    readonly jobId: () => string,
    readonly live: LiveLabelJob,
  ) {}

  /** Uploads `files`; the result (or the error) is shown in place. */
  async files(files: File[]) {
    if (!files.length || this.progress) return;
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

  async #apply(call: () => Promise<LabelImagesAdded>) {
    let answer: LabelImagesAdded | undefined;
    await this.live.act(async () => (answer = await call()).job);
    if (!answer) return;
    this.live.put(...answer.added);
    this.result = answer;
  }
}
