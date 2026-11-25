/**
 * A dataset's thumbnails in a virtualised grid (ui/TileGrid): image pages load as their rows come into view. The
 * cells take the dataset's dominant image shape.
 */
import { computed, css, effect, onCleanup, signal, untrack } from 'fw';
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
  /** The cells take the shape of the dataset's first 64 images (4:3 until they load; latched across page-0 eviction). */
  let lastAspect = 4 / 3;
  const aspect = computed(() => {
    const sample = Array.from({ length: Math.min(props.pager.total ?? 0, 64) }, (_, i) => props.pager.get(i));
    if (sample.some(Boolean)) lastAspect = dominantAspect(sample);
    return lastAspect;
  });

  function label(i: number) {
    const img = props.pager.get(i);
    return img ? `${stem(img.file)}, ${img.split}, ${count(img.boxes.length, 'box')}` : 'Loading image';
  }

  let openSeq = 0;
  onCleanup(() => {
    openSeq++;
  });
  function open(i: number) {
    const seq = ++openSeq;
    const img = props.pager.get(i);
    if (img) props.onopen(i, img.id);
    else void props.pager.at(i).then((loaded) => loaded && seq === openSeq && props.onopen(i, loaded.id));
  }

  // A tile reads its image through one computed subscribed only to its own page's signal, so a page landing elsewhere
  // never wakes this tile or re-runs its bindings. Visible tiles mount their thumbnail immediately once metadata is
  // available; off-screen overscan tiles wait one animation frame after metadata arrives so visible thumbnails claim
  // the connection pool first (and fast-scrolled overscan tiles cancel before fetching).
  const tile = (i: () => number, inView: () => boolean) => {
    const img = computed(() => props.pager.get(i()));
    const initImg = untrack(() => img.get());
    const initOn = untrack(inView) && !!initImg;
    let readyKey = initOn ? `${initImg.id}:${untrack(() => props.pager.revisionOf(i()))}` : '';
    const ready = signal(initOn);
    effect(() => {
      const cur = img.get();
      if (!cur) {
        readyKey = '';
        ready.set(false);
        return;
      }
      const key = `${cur.id}:${props.pager.revisionOf(i())}`;
      if (inView() || typeof requestAnimationFrame === 'undefined') {
        readyKey = key;
        ready.set(true);
        return;
      }
      if (readyKey === key && ready.peek()) return;
      ready.set(false);
      const id = requestAnimationFrame(() => {
        readyKey = key;
        ready.set(true);
      });
      return () => cancelAnimationFrame(id);
    });
    // The image's bindings may re-run once more as it goes (before the branch that shows it is taken down: a link to an
    // image opened from the Stats tab did): they read the last image there was, never undefined.
    let last: ImageInfo | undefined;
    const im = () => (last = img.get() ?? last)!;
    return (
      <>
        {img.get() && ready.get() ? (
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
      onrange={(lo, hi, visLo, visHi) => props.pager.ensure(lo, hi, visLo, visHi)}
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
