import { computed, type ReadSignal } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { Dataset, DatasetImage as Img, Model } from '$lib/api/types';

/**
 * Sample images for the empty state: a few from each dataset, the selected model's own dataset first. Undefined while
 * they load. Call in a component's body.
 */
export function sampleImages(model: ReadSignal<Model | null>): ReadSignal<{ ds: Dataset; img: Img }[] | undefined> {
  const pool = new Resource(async (s) => {
    const list = await hub.api.datasets(s);
    const per = await Promise.all(
      list.slice(0, 4).map((d) =>
        hub.api
          .datasetImages(d.id, { split: d.splits.val ? 'val' : 'train', limit: 3 }, s)
          .then((r) => r.images.map((img) => ({ ds: d, img })))
          .catch(() => []),
      ),
    );
    return per;
  });
  return computed(() => {
    const groups = pool.data;
    if (!groups) return pool.error ? [] : undefined;
    const key = model.get()?.classes.join('|');
    const sorted = [...groups].sort((a, b) => Number(b[0]?.ds.classes.join('|') === key) - Number(a[0]?.ds.classes.join('|') === key));
    return sorted.flat().slice(0, 6);
  });
}
