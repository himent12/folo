// Where the Playground's image comes from (a file or a dataset's image), and its bytes for the in-browser runtime.
import { hub } from '$lib/api/index';
import type { Dataset, DatasetImage as Img } from '$lib/api/types';

/** Where the Playground's image comes from: a file (picked, dropped or pasted) or a dataset's image. */
export type Source = { kind: 'file'; file: Blob; name: string; url: string; width: number; height: number } | { kind: 'dataset'; ds: Dataset; img: Img; at?: number };

/** The image's bytes, for the in-browser runtime. */
export async function imageBlob(s: Source, signal: AbortSignal): Promise<Blob> {
  if (s.kind === 'file') return s.file;
  const res = await fetch(await hub.api.imageUrl(s.ds.id, s.img.id), { signal });
  if (!res.ok) throw new Error(`Could not read ${s.img.file} (HTTP ${res.status}).`);
  return res.blob();
}
