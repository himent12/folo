/**
 * A dataset as a card (design/UNIFIED.md §4, Datasets): a 2 × 2 mosaic of its first images with their boxes drawn as
 * the dataset grid's tiles draw them, then its name, "2,400 images · 8 classes" and its format as a small tag, and the
 * class-balance strip along the card's bottom edge (its breakdown is a popover, ui/InfoButton through ClassStrip). The
 * path, the splits and the box count are on the dataset's page.
 *
 * The mosaic loads when the card comes into view (`GET /api/datasets/{id}/images?limit=4`, each image as a 256 px
 * thumbnail); an image that fails shows the tile's own unavailable state, with Retry. The name is a stretched link over
 * the card; the strip and a Retry sit above it, so each keeps its own click.
 */
import ImageOff from '$icons/image-off';
import { computed, css, For, onCleanup, onMount, signal } from 'fw';
import { hub } from '$lib/api/index';
import type { DatasetImage as ImageInfo } from '$lib/api/types';
import ClassStrip from '$lib/detect/ClassStrip';
import DatasetImage from '$lib/detect/DatasetImage';
import { count } from '$lib/format';
import { href } from '$lib/router';
import Badge from '$lib/ui/Badge';
import Skeleton from '$lib/ui/Skeleton';
import { imageCount } from '../dataset/health';
import { PROBLEM_LABEL, type ListedDataset } from './problem';

/** The mosaic's images: the dataset's first four. */
export const MOSAIC = 4;

/** What a card says under its name: "2,400 images · 8 classes". */
export const cardFacts = (d: ListedDataset) => `${count(imageCount(d), 'image')} · ${count(d.classes.length, 'class')}`;

export default function DatasetCard(props: { dataset: ListedDataset }) {
  const d = () => props.dataset;
  /** The first images, once the card has been in view; null while they load, [] when none could be read. */
  const images = signal<ImageInfo[] | null>(null);
  const failed = signal(false);
  let el!: HTMLElement;

  const ctl = new AbortController();
  onCleanup(() => ctl.abort());
  function load() {
    hub.api.datasetImages(d().id, { limit: MOSAIC }, ctl.signal).then(
      (r) => images.set(r.images.slice(0, MOSAIC)),
      () => {
        if (!ctl.signal.aborted) failed.set(true);
      },
    );
  }
  // Lazily: a long list reads only the mosaics in view (and a little ahead). A folder found missing has none to read.
  onMount(() => {
    if (d().problem) return;
    if (typeof IntersectionObserver === 'undefined') return void load();
    const io = new IntersectionObserver(
      (entries) => {
        if (!entries.some((e) => e.isIntersecting)) return;
        io.disconnect();
        load();
      },
      { rootMargin: '200px' },
    );
    io.observe(el);
    return () => io.disconnect();
  });
  const tiles = computed(() => Array.from({ length: MOSAIC }, (_, i) => i));

  return (
    <li class="card" data-id={d().id} data-name={d().name} ref={(node) => (el = node)}>
      <div class="mosaic">
        <For each={tiles.get()} key={(i) => i}>
          {(i) => {
            const im = () => images.get()?.[i()];
            return (
              <span class="tile">
                {d().problem || failed.get() ? (
                  i() === 0 ? (
                    <span class="none">
                      <ImageOff size={16} strokeWidth={1.75} aria-hidden="true" />
                    </span>
                  ) : null
                ) : images.get() === null ? (
                  <Skeleton w="100%" h="100%" r="0" image />
                ) : im() ? (
                  <DatasetImage
                    datasetId={d().id}
                    imageId={im()!.id}
                    width={im()!.width}
                    height={im()!.height}
                    thumb={256}
                    boxes={im()!.boxes}
                    classes={d().classes}
                    compact
                    fit="cover"
                    alt=""
                  />
                ) : null}
              </span>
            );
          }}
        </For>
      </div>
      <div class="body">
        <a class="name truncate" href={href(`/datasets/${d().id}`)} title={d().name}>
          {d().name}
        </a>
        <span class="facts">
          <span class="num truncate">{cardFacts(d())}</span>
          <span class="tag">{d().format.toUpperCase()}</span>
        </span>
      </div>
      <div class="foot">
        {d().problem ? (
          <Badge tone="warning">{PROBLEM_LABEL[d().problem!]}</Badge>
        ) : d().stats ? (
          <ClassStrip counts={d().stats!.class_counts} classes={d().classes} height={4} />
        ) : (
          <span class="no-strip"></span>
        )}
      </div>
    </li>
  );
}

/** A card's shape while the list loads: the mosaic's well, the name and its facts line, the strip. */
export function DatasetCardSkeleton() {
  return (
    <li class="card" aria-hidden="true">
      <div class="mosaic">
        <span class="tile">
          <Skeleton w="100%" h="100%" r="0" image />
        </span>
        <span class="tile">
          <Skeleton w="100%" h="100%" r="0" image />
        </span>
        <span class="tile">
          <Skeleton w="100%" h="100%" r="0" image />
        </span>
        <span class="tile">
          <Skeleton w="100%" h="100%" r="0" image />
        </span>
      </div>
      <div class="body">
        <Skeleton w="45%" h="14px" />
        <Skeleton w="60%" h="10px" />
      </div>
      <div class="foot">
        <Skeleton h="4px" r="999px" />
      </div>
    </li>
  );
}

css`
  .card {
    position: relative;
    display: flex;
    flex-direction: column;
    min-width: 0;
    overflow: hidden;
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    transition: border-color var(--dur-fast) var(--ease-standard);
  }
  .card:hover {
    border-color: var(--line-strong);
  }
  .card:has(.name:focus-visible) {
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: var(--size-focus-offset);
  }
  /* The 2 × 2 mosaic: four 4:3 cells with 1 px seams, on the inset well an image letterboxes on. */
  .mosaic {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    aspect-ratio: 8 / 6;
    background: var(--line-subtle);
    border-bottom: 1px solid var(--line-subtle);
  }
  .tile {
    position: relative;
    display: flex;
    min-width: 0;
    min-height: 0;
    overflow: hidden;
    background: var(--bg-inset);
  }
  .tile > :global(*) {
    flex: 1;
  }
  .none {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    color: var(--text-tertiary);
  }
  /* An unavailable image's Retry stays its own target, above the card's stretched link. */
  .tile :global(.retry) {
    position: relative;
    z-index: 2;
  }
  .body {
    display: grid;
    gap: var(--space-1);
    padding: var(--space-3) var(--space-4) var(--space-2);
  }
  .name {
    display: block;
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    letter-spacing: var(--type-md-tracking);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  .name:focus-visible {
    outline: none;
  }
  /* The stretched link: the whole card opens the dataset. */
  .name::after {
    content: '';
    position: absolute;
    inset: 0;
    z-index: 1;
  }
  .facts {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  /* The format, a small neutral tag. */
  .tag {
    flex-shrink: 0;
    margin-left: auto;
    padding: 0 var(--space-1-5);
    border-radius: var(--radius-xs);
    background: var(--bg-active);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    line-height: var(--control-xs);
    font-weight: var(--weight-medium);
  }
  /* The class balance along the bottom edge, above the stretched link so its breakdown opens on its own (the rest of
     the row still opens the dataset). */
  .foot {
    position: relative;
    z-index: 2;
    pointer-events: none;
    display: flex;
    align-items: center;
    min-height: var(--control-sm);
    padding: 0 var(--space-4) var(--space-3);
  }
  .foot > :global(*) {
    flex: 1;
    min-width: 0;
    pointer-events: auto;
  }
  .no-strip {
    height: 4px;
    border-radius: var(--radius-full);
    background: var(--bg-active);
  }
`;
