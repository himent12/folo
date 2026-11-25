/**
 * A labeling job's images: a toolbar (which images, how they show, Add images), the upload in progress or its
 * result, and the virtualised grid. Image files dropped anywhere on it are uploaded. With no images yet, the grid's
 * place is a drop target (the Playground's look, and its rule: a touch device is told to choose, never to drop).
 */
import FolderOpen from '$icons/folder-open';
import ImageOff from '$icons/image-off';
import ImagePlus from '$icons/image-plus';
import Scan from '$icons/scan';
import Upload from '$icons/upload';
import { computed, createRoot, css, effect, getOwner, Key, MediaQuery, onCleanup, signal, untrack, type Signal } from 'fw';
import type { LabelImage } from '$lib/api/types';
import { media } from '$lib/design/tokens';
import { count, num } from '$lib/format';
import { TOUCH_ONLY } from '$lib/shell/shortcuts';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import { gridView } from '$lib/ui/gridView';
import Panel from '$lib/ui/Panel';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import TileGrid, { dominantAspect } from '$lib/ui/TileGrid';
import ToggleButton from '$lib/ui/ToggleButton';
import Viewfinder from '$lib/ui/Viewfinder';
import DisplayMenu from '../dataset/DisplayMenu';
import LabelTile from './LabelTile';
import UploadStatus from './UploadStatus';
import type { Uploads } from './uploads';

export type Show = 'all' | 'pending' | 'review' | 'reviewed' | 'failed';

/**
 * Whether an image is in a filter: "To label" holds the images waiting and the ones with the agent now; a labeled
 * image is "To review" until a person checked it, then "Reviewed".
 */
export const inFilter = (img: LabelImage, show: Show) =>
  show === 'all' ||
  (show === 'pending'
    ? img.status === 'pending' || img.status === 'labeling'
    : show === 'review' || show === 'reviewed'
      ? img.status === 'labeled' && !!img.reviewed === (show === 'reviewed')
      : img.status === show);

export interface ImagesAreaProps {
  jobId: string;
  classes: readonly string[];
  images: readonly LabelImage[];
  show: Show;
  uploads: Uploads;
  onshow: (s: Show) => void;
  onopen: (id: string) => void;
  /** Opens the file picker. */
  onbrowse: () => void;
  /** Opens Add images, on its upload or its server folder tab. */
  onadd: (tab: 'upload' | 'folder') => void;
}

const FILTERS = [
  ['all', 'All'],
  ['pending', 'To label'],
  ['review', 'To review'],
  ['reviewed', 'Reviewed'],
  ['failed', 'Failed'],
] as const;

export default function ImagesArea(props: ImagesAreaProps) {
  const touchOnly = new MediaQuery(TOUCH_ONLY, false);
  const phoneQuery = new MediaQuery(media.phone, false);
  const phone = computed(() => phoneQuery.current);
  const touch = computed(() => touchOnly.current || phone.get());

  // One pass over the images (up to 10,000, re-counted on every image event of a run) instead of one per filter.
  const counts = computed(() => {
    const c = { all: props.images.length, pending: 0, review: 0, reviewed: 0, failed: 0 };
    for (const i of props.images) {
      if (i.status === 'pending' || i.status === 'labeling') c.pending++;
      else if (i.status === 'labeled') c[i.reviewed ? 'reviewed' : 'review']++;
      else if (i.status === 'failed') c.failed++;
    }
    return c;
  });
  const shown = computed(() => (props.show === 'all' ? props.images : props.images.filter((i) => inFilter(i, props.show))));
  // Announce filter counts only when the active filter changes, never on live agent SSE image updates.
  const filterStatus = signal('');
  let prevShow = untrack(() => props.show);
  effect(() => {
    const cur = props.show;
    if (cur === prevShow) return;
    prevShow = cur;
    const total = untrack(() => props.images.length);
    if (!total) {
      filterStatus.set('');
      return;
    }
    const n = shown.peek().length;
    filterStatus.set(n < total ? `Showing ${num(n)} of ${count(total, 'image')}.` : `Showing all ${count(total, 'image')}.`);
  });
  const options = computed(() => FILTERS.map(([value, label]) => ({ value, label, meta: num(counts.get()[value]) })));
  // The cells take the job's commonest image shape (ui/TileGrid).
  const aspect = computed(() => dominantAspect(props.images.slice(0, 64)));

  const dragging = signal(false);
  let depth = 0;
  const hasFiles = (e: DragEvent) => !!e.dataTransfer?.types.includes('Files');
  function ondragenter(e: DragEvent) {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth++;
    dragging.set(true);
  }
  function ondragleave() {
    depth = Math.max(0, depth - 1);
    if (!depth) dragging.set(false);
  }
  function ondrop(e: DragEvent) {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    dragging.set(false);
    void props.uploads.files([...(e.dataTransfer?.files ?? [])]);
  }

  const label = (i: number) => {
    const img = shown.get()[i];
    return img
      ? `${img.file}, ${img.status === 'labeled' ? `${count(img.boxes.length, 'box')}${img.reviewed ? ', reviewed' : ''}` : img.status}`
      : 'Image';
  };

  // ---------- the tiles, keyed by image id ----------
  // TileGrid's cells are places (the item at row × columns + column); a tile belongs to an image. Each image shown has
  // one tile, made once, and a cell shows its image's tile, moving it in from the cell that showed it before. So when
  // the images shift (an image labeled leaves "To label", one is removed, the column count changes) the tiles move with
  // their images, already loaded, instead of each cell's tile loading the next image (Svelte reloaded every tile after
  // the change: a skeleton flash across the grid on every image a run labeled). Rows that stay in view while the grid
  // scrolls keep their cells, so their tiles are untouched; an image replaced by a newer copy of itself (the agent
  // labeled it) updates its tile in place. A tile no cell shows any more goes at the end of the turn (one cell may give
  // it up before the next takes it).
  interface Tile {
    node: Node;
    img: Signal<LabelImage>;
    /** The cell showing it (its anchor), or null. */
    at: Comment | null;
    dispose: () => void;
  }
  const owner = getOwner();
  const tiles = new Map<string, Tile>();
  let sweeping = false;
  function sweep() {
    if (sweeping) return;
    sweeping = true;
    queueMicrotask(() => {
      sweeping = false;
      for (const [id, t] of tiles) {
        if (t.at) continue;
        t.dispose();
        tiles.delete(id);
      }
    });
  }
  const make = (img: LabelImage): Tile =>
    createRoot((dispose) => {
      const sig = signal(img);
      const node = (
        <LabelTile img={sig.get()} jobId={props.jobId} classes={props.classes} boxes={gridView.boxes} density={gridView.density} />
      ) as Node;
      return { node, img: sig, at: null, dispose };
    }, owner);
  // The tiles' roots are this component's: they go with it.
  onCleanup(() => tiles.clear());

  const tile = (i: () => number) => {
    const img = computed(() => shown.get()[i()]);
    // The cell's place for its tile (comments are not rendered).
    const anchor = document.createComment('');
    let held: Tile | null = null;
    const letGo = () => {
      if (held?.at === anchor) {
        held.node.parentNode?.removeChild(held.node);
        held.at = null;
      }
      held = null;
      sweep();
    };
    effect(() => {
      const cur = img.get();
      const t = cur && (tiles.get(cur.id) ?? tiles.set(cur.id, make(cur)).get(cur.id)!);
      if (held && held !== t) letGo();
      if (!cur || !t) return;
      t.img.set(cur);
      if (t.at !== anchor) {
        anchor.parentNode?.insertBefore(t.node, anchor);
        t.at = anchor;
      }
      held = t;
    });
    onCleanup(letGo);
    return anchor;
  };

  return (
    /* The whole area takes dropped files. */
    <section
      class="area"
      aria-label="Images"
      ondragenter={ondragenter}
      ondragover={(e) => hasFiles(e) && e.preventDefault()}
      ondragleave={ondragleave}
      ondrop={ondrop}
    >
      <div class="lj-toolbar" class:phone={phone.get()}>
        {phone.get() ? (
          <>
            <span class="pick">
              <Select
                label="Filter images by status"
                size="sm"
                options={options.get().map((o) => ({ value: o.value, label: `${o.label} · ${o.meta}` }))}
                value={props.show}
                onchange={props.onshow}
              />
            </span>
            <DisplayMenu
              boxes={gridView.boxes}
              density={gridView.density}
              onboxes={(on) => (gridView.boxes = on)}
              ondensity={(d) => (gridView.density = d)}
            />
            <Button size="sm" icon={ImagePlus} aria-label="Add images" onclick={() => props.onadd('upload')} />
          </>
        ) : (
          <>
            <Segmented
              label="Filter images by status"
              size="sm"
              options={options.get()}
              value={props.show}
              onchange={props.onshow}
              disabled={!props.images.length}
            />
            <span class="spacer"></span>
            <ToggleButton pressed={gridView.boxes} label="Boxes" icon={Scan} onchange={(on) => (gridView.boxes = on)} />
            <Segmented
              label="Thumbnail size"
              size="sm"
              options={[
                { value: 's', label: 'S' },
                { value: 'm', label: 'M' },
                { value: 'l', label: 'L' },
              ]}
              value={gridView.density}
              onchange={(d) => (gridView.density = d)}
            />
            <Button size="sm" icon={ImagePlus} onclick={() => props.onadd('upload')}>
              Add images
            </Button>
          </>
        )}
      </div>

      <UploadStatus uploads={props.uploads} />
      <p class="sr-only" role="status" aria-live="polite" data-filter-status>
        {filterStatus.get()}
      </p>

      {props.images.length === 0 ? (
        <div class="zone" class:dragging={dragging.get()}>
          <Viewfinder
            dragging={dragging.get()}
            title={touch.get() ? 'Choose images to label' : dragging.get() ? 'Release to add them' : 'Drop images to label'}
            description={() => (
              <>
                {touch.get()
                  ? 'JPEG, PNG, BMP or WebP, from your photos or files.'
                  : 'JPEG, PNG, BMP or WebP, as many as you like. They stay in this job until you make a dataset.'}
              </>
            )}
            actions={() => (
              <>
                <Button variant="primary" icon={Upload} onclick={props.onbrowse}>
                  {touch.get() ? 'Choose images' : 'Browse files'}
                </Button>
                <Button icon={FolderOpen} onclick={() => props.onadd('folder')}>
                  Add a folder on the server
                </Button>
              </>
            )}
          />
        </div>
      ) : shown.get().length === 0 ? (
        <Panel>
          <EmptyState
            icon={ImageOff}
            title="No images here"
            description={
              props.show === 'failed'
                ? 'The agent labeled every image it was given.'
                : props.show === 'review'
                  ? 'Every labeled image is reviewed.'
                  : props.show === 'reviewed'
                    ? 'No image is reviewed yet: open one and press Enter when its labels look right.'
                    : 'Every image has labels.'
            }
            actions={() => <Button onclick={() => props.onshow('all')}>Show all</Button>}
          />
        </Panel>
      ) : (
        <div class="grid-wrap">
          <Key when={props.show}>
            <TileGrid
              total={shown.get().length}
              density={gridView.density}
              aspect={aspect.get()}
              label={label}
              onopen={(i) => {
                const img = shown.peek()[i];
                if (img) props.onopen(img.id);
              }}
              toolbar=".lj-toolbar"
              tile={tile}
            />
          </Key>
          {dragging.get() && !touch.get() ? (
            <div class="over">
              <Viewfinder dragging title="Release to add them" description={() => <>They join the job as images to label.</>} />
            </div>
          ) : null}
        </div>
      )}
    </section>
  );
}

css`
  .area {
    display: grid;
    gap: var(--space-3);
    align-content: start;
    min-width: 0;
  }
  /* The images' toolbar sticks under the top bar as the grid scrolls, as a dataset's does. */
  .lj-toolbar {
    position: sticky;
    top: var(--size-topbar);
    z-index: calc(var(--z-sticky) - 1);
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-2) var(--space-3);
    margin: calc(var(--space-2) * -1) 0 0;
    padding: var(--space-2) 0;
    background: color-mix(in srgb, var(--bg-app) 92%, transparent);
    backdrop-filter: blur(8px);
  }
  .spacer {
    flex: 1;
  }
  .lj-toolbar.phone {
    flex-wrap: nowrap;
    gap: var(--space-2);
  }
  .pick {
    flex: 1;
    min-width: 0;
  }
  .zone {
    display: grid;
    border-radius: var(--radius-lg);
    background: var(--bg-inset);
    --vf-pad: var(--space-8) var(--space-6);
    min-height: 360px;
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .zone.dragging {
    background: color-mix(in srgb, var(--bg-inset), var(--accent-solid) 6%);
  }
  .grid-wrap {
    position: relative;
  }
  /* Files over a grid: the drop target over the whole of it. */
  .over {
    position: absolute;
    inset: 0;
    display: grid;
    border-radius: var(--radius-lg);
    background: color-mix(in srgb, var(--bg-app) 86%, transparent);
    pointer-events: none;
  }
  @media (max-width: 767px) {
    .zone {
      min-height: 0;
      --vf-pad: var(--space-8) var(--space-4);
    }
  }
`;
