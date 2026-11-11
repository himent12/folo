/**
 * Pick an image from a registered dataset: dataset + split, then a page of thumbnails. A tile is never blank without a
 * reason: while it loads it is the viewfinder skeleton (detect/DatasetImage), and an image with no boxes (a background
 * example, often a plain surface) says "No boxes" on it, so it never reads as a tile that failed to load (round 27).
 */
import ChevronLeft from '$icons/chevron-left';
import ChevronRight from '$icons/chevron-right';
import Images from '$icons/images';
import { bindable, computed, css, defaults, effect, For, Index, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { Dataset, DatasetImage as Img, Split } from '$lib/api/types';
import DatasetImage from '$lib/detect/DatasetImage';
import { num } from '$lib/format';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import Skeleton from '$lib/ui/Skeleton';

export interface DatasetPickerProps {
  open?: boolean;
  /** `at`: the image's place in its split (a link finds it again with one request). */
  onpick: (d: Dataset, img: Img, at: number) => void;
  onclose?: () => void;
  /** Classes of the selected model: a dataset with the same classes is shown first. */
  prefer?: readonly string[];
}

const PAGE = 24;
/** The twelve placeholder tiles while a page loads. */
const TWELVE = Array.from({ length: 12 });
const SPLITS: Split[] = ['val', 'train', 'test'];

export default function DatasetPicker(props: DatasetPickerProps) {
  const p = defaults(props, { prefer: [] as readonly string[] });
  const open = bindable(props, 'open', false);

  const datasets = new Resource((s) => (open.get() ? hub.api.datasets(s) : Promise.resolve(null)));

  const chosen = signal<string | null>(null);
  const split = signal<Split>('val');
  const offset = signal(0);

  const list = computed(() => datasets.data ?? []);
  const preferKey = computed(() => p.prefer.join('|'));
  const ds = computed(
    () => list.get().find((d) => d.id === chosen.get()) ?? list.get().find((d) => d.classes.join('|') === preferKey.get()) ?? list.get()[0] ?? null,
  );
  const dsId = computed(() => ds.get()?.id ?? null);
  const splitN = computed(() => {
    const d = ds.get();
    return d ? d.splits[split.get()] : 0;
  });

  // When the dataset changes, start at its first page and on a split it actually has (val, else train, else test).
  effect(() => {
    const d = ds.get();
    if (!d) return;
    untrack(() => {
      offset.set(0);
      if (d.splits[split.get()] === 0) split.set(SPLITS.find((x) => d.splits[x] > 0) ?? 'train');
    });
  });

  const images = new Resource((s) => {
    const id = dsId.get();
    const sp = split.get();
    const off = offset.get();
    if (!id || !open.get()) return Promise.resolve(null);
    return hub.api.datasetImages(id, { split: sp, offset: off, limit: PAGE }, s);
  });

  function pick(img: Img) {
    const d = ds.peek();
    if (!d) return;
    p.onpick(d, img, offset.peek() + Math.max(0, images.data?.images.indexOf(img) ?? 0));
    open.set(false);
  }

  const splits = computed(() => {
    const d = ds.get();
    return SPLITS.filter((s) => !d || d.splits[s] > 0).map((s) => ({ value: s, label: s, meta: d ? num(d.splits[s]) : undefined }));
  });

  return (
    <Dialog
      bind:open={open}
      title="Pick an image"
      description="Run the model on an image from a registered dataset."
      size="lg"
      onclose={p.onclose}
      footer={() => (
        <>
          <span class="range num">
            {splitN.get() ? (
              <>
                {num(offset.get() + 1)}–{num(Math.min(splitN.get(), offset.get() + PAGE))} of {num(splitN.get())}
              </>
            ) : null}
          </span>
          <Button
            size="sm"
            variant="ghost"
            icon={ChevronLeft}
            aria-label="Previous page"
            disabled={offset.get() === 0}
            onclick={() => offset.set(Math.max(0, offset.get() - PAGE))}
          />
          <Button
            size="sm"
            variant="ghost"
            icon={ChevronRight}
            aria-label="Next page"
            disabled={offset.get() + PAGE >= splitN.get()}
            onclick={() => offset.set(offset.get() + PAGE)}
          />
        </>
      )}
    >
      {datasets.error && !datasets.data ? (
        <ErrorState error={datasets.error} retry={() => datasets.refresh()} compact />
      ) : datasets.data && list.get().length === 0 ? (
        <EmptyState icon={Images} title="No datasets registered" description="Register a dataset first, or drop an image file instead." />
      ) : (
        <>
          <div class="bar">
            <div class="ds-select">
              <Select
                label="Dataset"
                value={ds.get()?.id ?? ''}
                options={list.get().map((d) => ({ value: d.id, label: d.name }))}
                disabled={!list.get().length}
                onchange={(v) => {
                  chosen.set(v);
                  offset.set(0);
                }}
              />
            </div>
            <Segmented label="Split" size="sm" bind:value={split} options={splits.get()} onchange={() => offset.set(0)} />
          </div>

          {/* Errors show even with an older page retained, so a stale page of another dataset is never pickable. */}
          {images.error ? (
            <ErrorState error={images.error} retry={() => images.refresh()} compact />
          ) : (
            <div class="grid" aria-busy={images.loading}>
              {!images.data || images.loading ? (
                <Index each={TWELVE}>{() => <Skeleton h="auto" r="var(--radius-sm)" class="thumb-sk" image />}</Index>
              ) : (
                <For each={images.data?.images ?? []} key={(img) => img.id}>
                  {(img) => (
                    <button
                      type="button"
                      class="thumb"
                      onclick={() => pick(img())}
                      aria-label={`Use ${img().file}${img().boxes.length ? '' : ', no boxes'}`}
                      title={img().file}
                    >
                      <DatasetImage
                        datasetId={ds.get()?.id ?? ''}
                        imageId={img().id}
                        width={img().width}
                        height={img().height}
                        thumb={192}
                        classes={ds.get()?.classes ?? []}
                        fit="cover"
                        compact
                      />
                      {!img().boxes.length ? (
                        <span class="bare" aria-hidden="true">
                          No boxes
                        </span>
                      ) : null}
                    </button>
                  )}
                </For>
              )}
            </div>
          )}
        </>
      )}
    </Dialog>
  );
}

css`
  .bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    margin-bottom: var(--space-3);
    flex-wrap: wrap;
  }
  .ds-select {
    width: 220px;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: var(--space-2);
  }
  .grid :global(.thumb-sk) {
    aspect-ratio: 1;
  }
  .thumb {
    position: relative;
    aspect-ratio: 1;
    border-radius: var(--radius-sm);
    overflow: hidden;
    box-shadow: 0 0 0 1px var(--line-subtle);
    transition: box-shadow var(--dur-fast) var(--ease-standard);
  }
  .thumb:hover {
    box-shadow: 0 0 0 1px var(--text-tertiary);
  }
  .thumb:focus-visible {
    outline-offset: 2px;
  }
  /* A background example's mark, as the image chips are drawn: on a scrim, at the tile's bottom left. */
  .bare {
    position: absolute;
    left: 4px;
    bottom: 4px;
    padding: 0 var(--space-1);
    border-radius: var(--radius-xs);
    background: var(--bg-scrim);
    color: var(--text-primary);
    font-size: var(--type-sm-size);
    line-height: 18px;
    pointer-events: none;
  }
  :global([data-theme='light']) .bare {
    background: color-mix(in srgb, var(--bg-overlay) 88%, transparent);
  }
  .range {
    margin-right: auto;
    align-self: center;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  @media (max-width: 767px) {
    .grid {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .ds-select {
      width: 100%;
    }
  }
`;
