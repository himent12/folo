/**
 * A dataset's thumbnails in a virtualised grid (ui/TileGrid): image pages load as their rows come into view. The
 * cells take the dataset's dominant image shape.
 */
import { computed, css } from 'fw';
import DatasetImage from '$lib/detect/DatasetImage';
import TileGrid, { dominantAspect, type Density } from '$lib/ui/TileGrid';
import type { DatasetImage as ImageInfo } from '$lib/api/types';
import { count } from '$lib/format';
import type { ImagePager } from './pager';

export interface ImageGridProps {
  pager: ImagePager;
  classes: readonly string[];
  density: Density;
  boxes: boolean;
  only: ReadonlySet<number> | null;
  onopen: (index: number, id: string) => void;
  /** An image to scroll to and focus, each time a new one is given (the one the inspector closed on). */
  reveal?: { i: number } | null;
}

const NO_BOXES: never[] = [];

const stem = (file: string) => file.split('/').pop() ?? file;

export default function ImageGrid(props: ImageGridProps) {
  /** The cells take the shape of the dataset's first 64 images (4:3 until they load). */
  const aspect = computed(() =>
    dominantAspect(Array.from({ length: Math.min(props.pager.total ?? 0, 64) }, (_, i) => props.pager.get(i))),
  );

  function label(i: number) {
    const img = props.pager.get(i);
    return img ? `${stem(img.file)}, ${img.split}, ${count(img.boxes.length, 'box')}` : 'Loading image';
  }

  function open(i: number) {
    const img = props.pager.get(i);
    if (img) props.onopen(i, img.id);
  }

  // A tile reads its image through one computed: a page landing elsewhere (the pager's version moves) re-checks it
  // once, and its bindings re-run only when this cell's image itself changed (Svelte re-ran each `{@const}` reader).
  const tile = (i: () => number) => {
    const img = computed(() => props.pager.get(i()));
    // The image's bindings may re-run once more as it goes (before the branch that shows it is taken down: a link to an
    // image opened from the Stats tab did): they read the last image there was, never undefined.
    let last: ImageInfo | undefined;
    const im = () => (last = img.get() ?? last)!;
    return (
      <>
        {img.get() ? (
          <span class="tile" class:bare={im().boxes.length === 0}>
            <DatasetImage
              datasetId={props.pager.datasetId}
              imageId={im().id}
              rev={props.pager.revisionOf(i())}
              width={im().width}
              height={im().height}
              thumb={256}
              boxes={props.boxes ? im().boxes : NO_BOXES}
              classes={props.classes}
              only={props.only}
              compact
              fit="contain"
              alt=""
            />
            {props.density !== 's' ? (
              <span class="meta">
                <span class="file mono truncate">{stem(im().file)}</span>
                <span class="n num">{im().boxes.length}</span>
              </span>
            ) : null}
          </span>
        ) : (
          <span class="ph" class:still={!!props.pager.error}></span>
        )}
      </>
    );
  };

  return (
    <TileGrid
      total={props.pager.total ?? 0}
      density={props.density}
      aspect={aspect.get()}
      label={label}
      onopen={open}
      onrange={(lo, hi) => props.pager.ensure(lo, hi)}
      toolbar=".ds-toolbar"
      reveal={props.reveal}
      tile={tile}
    />
  );
}

css`
  .tile {
    position: absolute;
    inset: 0;
  }
  .meta {
    position: absolute;
    left: 4px;
    right: 4px;
    bottom: 4px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-1);
    pointer-events: none;
    opacity: 0;
    transform: translateY(2px);
    transition:
      opacity var(--dur-fast) var(--ease-standard),
      transform var(--dur-fast) var(--ease-standard);
  }
  .meta > span {
    height: 18px;
    display: inline-flex;
    align-items: center;
    padding: 0 4px;
    border-radius: var(--radius-xs);
    background: var(--bg-scrim);
    color: var(--text-primary);
    font-size: var(--type-micro-size); /* small: image chip (the thumbnail's box count) */
    backdrop-filter: blur(4px);
  }
  :global([data-theme='light']) .meta > span {
    background: color-mix(in srgb, var(--bg-overlay) 88%, transparent);
    color: var(--text-primary);
  }
  :global(.cell:hover) .meta,
  :global(.cell:focus-visible) .meta {
    opacity: 1;
    transform: none;
  }
  /* An image with no boxes (a background example) keeps its "0" at rest, so a plain surface never reads as a tile that
     failed to load; its name still waits for hover or focus. */
  .tile.bare .meta {
    opacity: 1;
    transform: none;
  }
  :global(.cell:not(:hover, :focus-visible)) .tile.bare .meta .file {
    visibility: hidden;
  }
  .ph {
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent, var(--bg-hover), transparent);
    background-size: 200% 100%;
    /* Loading motion is bounded (DESIGN_DIRECTION §2.9): a few sweeps while the page of images is requested, none once
       a request has failed (the grid says so), so a cell that never fills never shimmers forever. */
    animation: shimmer 1.4s linear 6;
  }
  .ph.still {
    animation: none;
  }
  @keyframes shimmer {
    from {
      background-position: 100% 0;
    }
    to {
      background-position: -100% 0;
    }
  }
`;
