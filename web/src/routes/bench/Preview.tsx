/**
 * What a condition does to an image of this benchmark: the image unchanged and under the condition, split by a
 * divider the person drags (or moves with the arrow keys), with the true boxes on both sides (a geometric condition
 * moves them: `distance`'s boxes shrink with the scene). Picks the condition, its severity (1–5) and the image. The
 * changed image is the one the server serves contestants (the mock draws an approximation).
 */
import ChevronLeft from '$icons/chevron-left';
import ChevronRight from '$icons/chevron-right';
import Scan from '$icons/scan';
import { bindable, computed, createUniqueId, css, effect, signal } from 'fw';
import { hub } from '$lib/api/index';
import type { Bench, BenchCatalog, BenchImage } from '$lib/api/types';
import { GROUP_NAME, SEVERITIES } from '$lib/bench/rules';
import DetectionView from '$lib/detect/DetectionView';
import { num } from '$lib/format';
import Button from '$lib/ui/Button';
import ErrorState from '$lib/ui/ErrorState';
import Panel from '$lib/ui/Panel';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import Skeleton from '$lib/ui/Skeleton';
import { dominantAspect } from '$lib/ui/TileGrid';
import ToggleButton from '$lib/ui/ToggleButton';

export interface PreviewProps {
  bench: Bench;
  catalog: BenchCatalog;
  /** Bindable: the condition shown. */
  condition?: string;
  /** Bindable: its severity, 1–5. */
  severity?: number;
}

/** The images people step through: the benchmark's first ones (a seeded shuffle, so a fair look). */
const PAGE = 24;

export default function Preview(props: PreviewProps) {
  const condition = bindable(props, 'condition', '');
  const severity = bindable(props, 'severity', 3);

  const uid = createUniqueId();
  const index = signal(0);
  const images = signal<BenchImage[]>([]);
  const total = signal(0);
  /** The truth under the condition; null while it is not known (loading, or its request failed): never another
   *  condition's boxes over this one's image. */
  const changedBoxes = signal<BenchImage['boxes'] | null>([]);
  const truthError = signal<unknown>(undefined);
  const truthTry = signal(0);
  const plainUrl = signal<string | null>(null);
  const changedUrl = signal<string | null>(null);
  const error = signal<unknown>(undefined);
  const boxes = signal(true);
  /** Where the divider is, 0–100 from the left: unchanged on its left, the condition on its right. */
  const split = signal(50);

  // The first images, once per benchmark: by its id, as the page hands down a new benchmark on every `bench` event.
  const benchId = computed(() => props.bench.id);
  effect(() => {
    const id = benchId.get();
    if (hub.mode === 'connecting') return;
    void hub.epoch;
    let gone = false;
    error.set(undefined);
    hub.api.benchImages(id, { limit: PAGE }).then(
      (p) => {
        if (gone) return;
        images.set(p.images);
        total.set(p.total);
        index.set((i) => Math.min(i, Math.max(0, p.images.length - 1)));
      },
      (e) => !gone && error.set(e),
    );
    return () => (gone = true);
  });

  const image = computed(() => images.get()[index.get()]);

  // The image unchanged, and under the condition with its truth (a geometric condition moves the boxes).
  effect(() => {
    const im = image.get();
    if (!im) return;
    let gone = false;
    plainUrl.set(null);
    void hub.api.benchImageUrl(benchId.get(), im.id).then(
      (u) => !gone && plainUrl.set(u),
      (e) => !gone && error.set(e),
    );
    return () => (gone = true);
  });
  effect(() => {
    const im = image.get();
    const c = condition.get();
    const s = severity.get();
    if (!im) return;
    let gone = false;
    changedUrl.set(null);
    const id = benchId.get();
    void hub.api.benchImageUrl(id, im.id, { condition: c, severity: s }).then(
      (u) => !gone && changedUrl.set(u),
      (e) => !gone && error.set(e),
    );
    return () => (gone = true);
  });
  // The truth under it: only a geometric condition moves it; the others keep the image's own. A failed request says
  // so with Retry (it used to be swallowed, leaving the previous condition's boxes drawn over this image).
  effect(() => {
    const im = image.get();
    const c = condition.get();
    const s = severity.get();
    void truthTry.get();
    truthError.set(undefined);
    if (!im) return;
    if (c !== 'distance') {
      changedBoxes.set(im.boxes);
      return;
    }
    let gone = false;
    changedBoxes.set(null);
    void hub.api.benchImages(benchId.get(), { offset: index.get(), limit: 1, condition: c, severity: s }).then(
      (p) => !gone && changedBoxes.set(p.images[0]?.boxes ?? im.boxes),
      (e) => !gone && truthError.set(e),
    );
    return () => (gone = true);
  });

  const options = computed(() => props.catalog.conditions.map((c) => ({ value: c.id, label: c.name, group: GROUP_NAME[c.group] })));
  const name = computed(() => props.catalog.conditions.find((c) => c.id === condition.get())?.name ?? condition.get());
  const desc = computed(() => props.catalog.conditions.find((c) => c.id === condition.get())?.description ?? '');
  /**
   * The stage's shape: the commonest among the benchmark's first images (within 3:4 to 16:9), so stepping from a
   * landscape photo to a portrait one or a panorama never makes the panel jump; each image is shown whole in it, both
   * halves fitted alike (DetectionView's contain), so the divider cuts the same point of both and the boxes stay on
   * their objects.
   */
  const ratio = computed(() => (images.get().length ? dominantAspect(images.get()) : 4 / 3));
  /** The severity as the Segmented's string value (Svelte's function binding). */
  const severityText = {
    get: () => String(severity.get()),
    set: (v: string) => severity.set(Number(v)),
  };
  const severityOptions = SEVERITIES.map((s) => ({ value: String(s), label: String(s) }));

  return (
    <Panel
      title="Preview"
      meta={image.get() ? `image ${num(index.get() + 1)} of ${num(total.get())}` : undefined}
      pad="none"
      actions={() => <ToggleButton pressed={boxes.get()} label="Boxes" icon={Scan} onchange={(v) => boxes.set(v)} />}
    >
      <div class="pv">
        <div class="controls">
          <div class="cond">
            <label class="sr-only" for={`${uid}-cond`}>
              Condition
            </label>
            <Select id={`${uid}-cond`} bind:value={condition} options={options.get()} label="Condition to preview" />
          </div>
          <Segmented label="Severity" size="sm" bind:value={severityText} options={severityOptions} />
        </div>
        {error.get() && !image.get() ? (
          <ErrorState error={error.get()} title="Could not load the images" compact />
        ) : (
          <div class="stage" style:aspect-ratio={ratio.get()} style={{ '--ar': ratio.get() }}>
            {image.get() && plainUrl.get() && changedUrl.get() ? (
              <>
                <div class="side plain" style:clip-path={`inset(0 ${100 - split.get()}% 0 0)`}>
                  <DetectionView
                    src={plainUrl.get()!}
                    width={image.get()!.width}
                    height={image.get()!.height}
                    truth={boxes.get() ? image.get()!.boxes : []}
                    classes={props.bench.classes}
                    labels={false}
                    alt={`${image.get()!.file}, unchanged`}
                  />
                </div>
                <div class="side changed" style:clip-path={`inset(0 0 0 ${split.get()}%)`}>
                  <DetectionView
                    src={changedUrl.get()!}
                    width={image.get()!.width}
                    height={image.get()!.height}
                    truth={boxes.get() ? (changedBoxes.get() ?? []) : []}
                    classes={props.bench.classes}
                    labels={false}
                    alt={`${image.get()!.file} under ${name.get()}, severity ${severity.get()}`}
                  />
                </div>
                <span class="divider" style:left={`${split.get()}%`} aria-hidden="true">
                  <span class="knob"></span>
                </span>
                <span class="tag l" aria-hidden="true">
                  Unchanged
                </span>
                <span class="tag r" aria-hidden="true">
                  {name.get()} {severity.get()}
                </span>
                <input
                  class="drag"
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  bind:value={split}
                  aria-label={`Divider between the unchanged image and ${name.get()}, severity ${severity.get()}`}
                  aria-valuetext={`${split.get()}% unchanged`}
                />
              </>
            ) : (
              <Skeleton h="100%" r="0" image />
            )}
          </div>
        )}
        {truthError.get() && image.get() ? (
          <ErrorState error={truthError.get()} title={`Could not load the boxes under ${name.get()}`} retry={() => truthTry.set((n) => n + 1)} compact />
        ) : null}
        <div class="foot">
          <p class="what">{desc.get()}</p>
          <div class="step">
            <Button
              variant="ghost"
              size="sm"
              icon={ChevronLeft}
              aria-label="Previous image"
              disabled={index.get() === 0}
              onclick={() => index.set((i) => Math.max(0, i - 1))}
            />
            <span class="file truncate" title={image.get()?.file}>
              {image.get()?.file ?? ''}
            </span>
            <Button
              variant="ghost"
              size="sm"
              icon={ChevronRight}
              aria-label="Next image"
              disabled={index.get() >= images.get().length - 1}
              onclick={() => index.set((i) => Math.min(images.get().length - 1, i + 1))}
            />
          </div>
        </div>
      </div>
    </Panel>
  );
}

css`
  .pv {
    display: grid;
    gap: var(--space-3);
    padding: 0 var(--space-4) var(--space-4);
  }
  /* The condition takes what the severities leave; where that is too little for its name (a phone's 16 px touch
     text), the severities wrap under it. */
  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
  .cond {
    flex: 1 1 176px;
    min-width: 0;
  }
  .stage {
    position: relative;
    width: 100%;
    border-radius: var(--radius-md);
    overflow: hidden;
    background: var(--bg-inset);
  }
  /* Below 1024 px the preview spans the column up to half the viewport's height, so the plan starts in the first
     screen (as the Playground's canvas does on a tablet): a tall or square stage takes the width that height allows,
     centred. */
  @media (max-width: 1023px) {
    .stage {
      width: min(100%, calc(50dvh * var(--ar)));
      margin-inline: auto;
    }
  }
  .side {
    position: absolute;
    inset: 0;
    display: grid;
  }
  .side :global(.view) {
    width: 100%;
    height: 100%;
  }
  /* The divider: a hairline in the image's own light, with a grip at its centre. */
  .divider {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 2px;
    margin-left: -1px;
    background: #ffffffd9;
    box-shadow: 0 0 0 1px #00000040;
    pointer-events: none;
  }
  .knob {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 16px;
    height: 24px;
    margin: -12px 0 0 -8px;
    border-radius: 8px;
    background: #ffffffe6;
    box-shadow: 0 1px 4px #00000059;
  }
  .tag {
    position: absolute;
    top: var(--space-2);
    padding: 2px 6px;
    border-radius: var(--radius-sm);
    background: #000000a6;
    color: #ffffff;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    pointer-events: none;
  }
  .tag.l {
    left: var(--space-2);
  }
  .tag.r {
    right: var(--space-2);
  }
  /* The range input is the divider's control: invisible over the image, full height, so a drag anywhere moves it. */
  .drag {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    margin: 0;
    opacity: 0;
    cursor: ew-resize;
  }
  .stage:has(.drag:focus-visible) {
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: var(--size-focus-offset);
  }
  .foot {
    display: grid;
    gap: var(--space-2);
  }
  .what {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    min-height: 32px;
  }
  .step {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-2);
  }
  .file {
    text-align: center;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
`;
