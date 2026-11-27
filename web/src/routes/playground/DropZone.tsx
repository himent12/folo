/** The Playground's empty state: a viewfinder to drop an image into, plus sample images to try. */
import Images from '$icons/images';
import Play from '$icons/play';
import Scan from '$icons/scan';
import Upload from '$icons/upload';
import { css, defaults, For, Index, MediaQuery } from 'fw';
import type { Dataset, DatasetImage as Img } from '$lib/api/types';
import DatasetImage from '$lib/detect/DatasetImage';
import Button from '$lib/ui/Button';
import Kbd from '$lib/ui/Kbd';
import Skeleton from '$lib/ui/Skeleton';
import Viewfinder from '$lib/ui/Viewfinder';
import { TOUCH_ONLY } from '$lib/shell/shortcuts';
import { media } from '$lib/design/tokens';
import { FORMATS } from './intake';

export interface DropZoneProps {
  dragging?: boolean;
  /** undefined while loading. */
  samples: { ds: Dataset; img: Img }[] | undefined;
  onbrowse: () => void;
  onpick: () => void;
  onvideo?: () => void;
  oncamera?: () => void;
  onsample: (ds: Dataset, img: Img) => void;
}

/** The six placeholder tiles while the samples load. */
const SIX = Array.from({ length: 6 });

export default function DropZone(props: DropZoneProps) {
  const p = defaults(props, { dragging: false });
  // A touch-only device (the shared detection, shell/shortcuts' TOUCH_ONLY) has no drag and drop and no Ctrl V: it
  // leads with choosing an image and shows neither hint; a device with a mouse or trackpad shows both. The phone tier
  // is the touch tier whatever reports the pointer, so a phone's copy is the same in every state (round 26: the
  // no-WASM and no-WebGPU phones said "Drop an image… Ctrl V" where the empty one said "Choose an image").
  const touchOnly = new MediaQuery(TOUCH_ONLY, false);
  const phone = new MediaQuery(media.phone, false);
  const touch = () => touchOnly.current || phone.current;

  // The viewfinder is the drop target and fills the canvas; the samples sit right under it, at the panel's foot.
  return (
    <div class="zone" class:dragging={p.dragging}>
      <div class="finder">
        <Viewfinder
          dragging={p.dragging}
          title={touch() ? 'Choose an image to detect objects' : p.dragging ? 'Release to detect' : 'Drop an image to detect objects'}
          description={() => (
            <>
              {touch() ? (
                `${FORMATS} (other images are converted), from your photos, camera or files.`
              ) : (
                <>
                  {FORMATS} (other images are converted). You can also paste from the clipboard with{' '}
                  <span class="nowrap">
                    <Kbd keys="Mod+V" />.
                  </span>
                </>
              )}
            </>
          )}
          actions={() => (
            <>
              <Button variant="primary" icon={Upload} onclick={p.onbrowse}>
                {touch() ? 'Choose an image' : 'Browse files'}
              </Button>
              <Button icon={Images} onclick={p.onpick}>
                Pick from a dataset
              </Button>
              {p.onvideo ? (
                <Button icon={Play} onclick={p.onvideo}>
                  Open video
                </Button>
              ) : null}
              {p.oncamera ? (
                <Button icon={Scan} onclick={p.oncamera}>
                  Live camera
                </Button>
              ) : null}
            </>
          )}
        />
      </div>

      {p.samples === undefined || p.samples.length > 0 ? (
        <div class="samples">
          <p class="overline">Or try a sample</p>
          <ul>
            {p.samples === undefined ? (
              <Index each={SIX}>
                {() => (
                  <li>
                    <Skeleton h="100%" r="var(--radius-sm)" image />
                  </li>
                )}
              </Index>
            ) : (
              <For each={p.samples} key={(s) => s.ds.id + s.img.id}>
                {(s) => (
                  <li>
                    <button
                      type="button"
                      class="sample"
                      onclick={() => p.onsample(s().ds, s().img)}
                      aria-label={`Try ${s().img.file} from ${s().ds.name}`}
                      title={`${s().ds.name} / ${s().img.file}`}
                    >
                      <DatasetImage
                        datasetId={s().ds.id}
                        imageId={s().img.id}
                        width={s().img.width}
                        height={s().img.height}
                        thumb={160}
                        classes={s().ds.classes}
                        fit="cover"
                        compact
                      />
                      <span class="tag">{s().ds.name}</span>
                    </button>
                  </li>
                )}
              </For>
            )}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

css`
  .zone {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-4);
    height: 100%;
    padding: var(--space-4) var(--space-4) var(--space-8);
    background: var(--bg-inset);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  /* The drop target: a sparse state's compact composition (DESIGN_DIRECTION 8) at every width, so the samples under it
     sit in the first screen. Beside the rail (from 1024 px) it is 480 px where the window has room
     for it and the samples under the page header (about 420 px of chrome), and gives way in a shorter one. Its message
     is centred between the corners (ui/Viewfinder). */
  .finder {
    display: grid;
    width: 100%;
    flex: none;
    height: clamp(280px, calc(100dvh - 420px), 480px);
    --vf-pad: var(--space-8) var(--space-6);
  }
  .zone.dragging {
    background: color-mix(in srgb, var(--bg-inset), var(--accent-solid) 6%);
  }
  .samples {
    display: grid;
    justify-items: center;
    gap: var(--space-2);
    width: 100%;
    max-width: 560px;
  }
  .samples ul {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: var(--space-2);
    width: 100%;
  }
  .samples li {
    aspect-ratio: 1;
  }
  .sample {
    position: relative;
    width: 100%;
    height: 100%;
    border-radius: var(--radius-sm);
    overflow: hidden;
    box-shadow: 0 0 0 1px var(--line-default);
    transition:
      box-shadow var(--dur-fast) var(--ease-standard),
      transform var(--dur-fast) var(--ease-standard);
  }
  .sample:hover {
    box-shadow: 0 0 0 1px var(--text-tertiary);
  }
  .sample:focus-visible {
    outline-offset: 2px;
  }
  /* One line over the image, cut with an ellipsis (the button's title has the whole name), never wrapping over it. */
  .tag {
    position: absolute;
    left: 4px;
    bottom: 4px;
    max-width: calc(100% - 8px);
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
    padding: 0 4px;
    border-radius: var(--radius-xs);
    background: var(--bg-overlay);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    line-height: 16px;
  }
  /* Tablets: no rail beside it, so the whole empty frame (the prompt and the sample strip) is capped at about 360 px
     and Thresholds, Classes and Timing start in the first screen. The finder is as tall as its message and buttons;
     the frame grows only once an image is loaded (Playground's stage). */
  @media (min-width: 768px) and (max-width: 1023px) {
    .zone {
      padding-bottom: var(--space-4);
    }
    .finder {
      height: auto;
      --vf-pad: var(--space-6);
    }
  }
  @media (max-width: 767px) {
    .zone {
      gap: var(--space-6);
      padding: var(--space-3) var(--space-3) var(--space-6);
    }
    /* Phones: as tall as the message and buttons need. */
    .finder {
      height: auto;
      --vf-pad: var(--space-8) var(--space-4);
    }
    .samples ul {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
  }
`;
