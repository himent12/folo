// Where the Playground's image comes from (a file or a dataset's image), and its bytes for the in-browser runtime.
import { hub } from '$lib/api/index';
import type { Dataset, DatasetImage as Img } from '$lib/api/types';

/** Where the Playground's image comes from: a file (picked, dropped or pasted) or a dataset's image. */
export type Source =
  | { kind: 'file'; file: Blob; name: string; url: string; width: number; height: number; /** When true, object URL lifetime is managed by the batch session rather than `usePlaygroundSource`. */ sharedUrl?: boolean }
  | { kind: 'dataset'; ds: Dataset; img: Img; at?: number };

/** A stable identity per file blob so browser inference can cache decoded pixels across re-runs. */
const blobIds = new WeakMap<Blob, number>();
let nextBlobId = 0;

/** Stable cache key for a Playground image source (file identity or dataset+image id). */
export function sourceImageKey(s: Source): string {
  if (s.kind === 'dataset') return `dataset:${s.ds.id}:${s.img.id}`;
  let id = blobIds.get(s.file);
  if (id === undefined) blobIds.set(s.file, (id = ++nextBlobId));
  return `file:${id}`;
}

/** The image's bytes, for the in-browser runtime. */
export async function imageBlob(s: Source, signal: AbortSignal): Promise<Blob> {
  if (s.kind === 'file') return s.file;
  const res = await fetch(await hub.api.imageUrl(s.ds.id, s.img.id), { signal });
  if (!res.ok) throw new Error(`Could not read ${s.img.file} (HTTP ${res.status}).`);
  return res.blob();
}
