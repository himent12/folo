/** What the form reads of a dataset's training images: a few for the augmentation preview, and their sizes. */
import { hub } from '$lib/api/index';
import type { DatasetImage } from '$lib/api/types';

/** A few training images with boxes (the augmentation preview's); none without a dataset. */
export async function sampleImagesOf(id: string, s: AbortSignal): Promise<DatasetImage[]> {
  if (!id) return [];
  return (await hub.api.datasetImages(id, { split: 'train', offset: 0, limit: 12 }, s)).images.filter((i) => i.boxes.length > 0);
}

// The training images' sizes, to train a dataset of small images at its own size. Read only for a dataset small
// enough to list in a few pages (the stats do not carry image sizes); otherwise null, and the size stays 640.
export async function trainSizesOf(id: string, s: AbortSignal) {
  if (!id) return null;
  const MAX = 2000;
  const sizes: (readonly [number, number])[] = [];
  let total = Infinity;
  for (let page = 0; page < 8 && sizes.length < Math.min(total, MAX); page++) {
    const r = await hub.api.datasetImages(id, { split: 'train', offset: sizes.length, limit: 1000 }, s);
    total = r.total;
    if (total > MAX || !r.images.length) break;
    sizes.push(...r.images.map((i) => [i.width, i.height] as const));
  }
  return { id, sizes: sizes.length === total ? sizes : null };
}
