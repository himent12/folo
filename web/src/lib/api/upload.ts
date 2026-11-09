// Uploading many images to a labeling job (docs/API.md, POST /api/label/jobs/{id}/images): the server takes at most
// 64 MiB per request, so a large selection goes in several requests, each well under the cap, and their answers are
// merged into one.

import type { LabelImagesAdded } from './types';

/** What one request carries at most: bytes (well under the server's 64 MiB) and files. */
export const UPLOAD_CHUNK_BYTES = 48 * 2 ** 20;
export const UPLOAD_CHUNK_FILES = 50;

/**
 * `files` in order, split into requests of at most `maxBytes` and `maxFiles` each. A file larger than `maxBytes` goes
 * alone (the server then says whether it takes it), so no file is ever dropped here.
 */
export function chunkFiles<F extends { size: number }>(files: readonly F[], maxBytes = UPLOAD_CHUNK_BYTES, maxFiles = UPLOAD_CHUNK_FILES): F[][] {
  const out: F[][] = [];
  let cur: F[] = [];
  let bytes = 0;
  for (const f of files) {
    if (cur.length && (cur.length >= maxFiles || bytes + f.size > maxBytes)) {
      out.push(cur);
      cur = [];
      bytes = 0;
    }
    cur.push(f);
    bytes += f.size;
  }
  if (cur.length) out.push(cur);
  return out;
}

/** The answers of several requests as one: every image added and skipped, in order, and the job after the last. */
export function mergeAdded(parts: readonly LabelImagesAdded[]): LabelImagesAdded | null {
  const last = parts.at(-1);
  if (!last) return null;
  return { added: parts.flatMap((p) => p.added), skipped: parts.flatMap((p) => p.skipped), job: last.job };
}

/** How far an upload has got: files sent so far of all of them. */
export interface UploadProgress {
  sent: number;
  total: number;
}
