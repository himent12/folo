/**
 * Evaluation samples (lib/evaluated `scoresTitle`): the same 16 images every epoch, predictions (solid, class colour) over ground truth (dashed,
 * neutral in the thumbnails). Follows the
 * latest epoch while the run is live; pick any earlier epoch to see how detections improved.
 */
import ImageOff from '$icons/image-off';
import { computed, css, effect, For, Index, MediaQuery, signal } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { Sample } from '$lib/api/types';
import DatasetImage from '$lib/detect/DatasetImage';
import SampleInspector from './SampleInspector';
import ErrorState from '$lib/ui/ErrorState';
import Panel from '$lib/ui/Panel';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select.tsx';
import Skeleton from '$lib/ui/Skeleton';
import Slider from '$lib/ui/Slider';
import { media } from '$lib/design/tokens';
import { scoresTitle } from '$lib/evaluated';

export interface SamplesProps {
  runId: string;
  classes: string[];
  epochsDone: number;
  /** "Validation samples", or the split they are really from ("Training-split samples", lib/evaluated `scoresTitle`). */
  title?: string;
  /** The run's history failed to load (so `epochsDone` is unknown): one quiet line, not "appear after…"; the metrics
   * panel has the error and its Retry. */
  unavailable?: boolean;
  /** The run's dataset was removed: its images are gone, and the tiles say so (nothing to retry). */
  datasetGone?: boolean;
  /** The run has ended (stopped, failed): with no validated epoch, none will come, and the panel says so. */
  ended?: boolean;
}

export default function Samples(props: SamplesProps) {
  const pick = signal<string>('latest');
  const show = signal<'both' | 'pred' | 'truth'>('both');
  const conf = signal(0.25);
  /** The sample open in the inspector (an index into the epoch's samples), null: none. */
  const open = signal<number | null>(null);
  // About eight samples show by default, always as full rows (5 columns: 10, 3 columns: 9); the rest are one click
  // away.
  const cols = signal(4);
  const FIRST = computed(() => cols.get() * Math.max(1, Math.round(8 / cols.get())));
  const gridEl = signal<HTMLElement>();
  effect(() => {
    const el = gridEl.get();
    if (!el) return;
    const measure = () => cols.set(Math.max(1, getComputedStyle(el).gridTemplateColumns.split(' ').filter(Boolean).length));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  });
  const showAll = signal(false);
  // Phones drop the header's epoch meta (the picker names it).
  const phone = new MediaQuery(media.phone, false);
  /** From 1280 px the header holds every control (design/UNIFIED.md §4 Run); narrower, the slider is on the key's row. */
  const wide = new MediaQuery(media.desktop, false);

  // Memoise the props the fetcher reads: the parent passes values derived from a run object that changes every
  // step, and an unmemoised prop read would refetch on each one.
  const rid = computed(() => props.runId);
  const done = computed(() => props.epochsDone);
  // Epochs are 1-based (docs/API.md); 'latest' is the last finished one.
  const epoch = computed(() => (pick.get() === 'latest' ? done.get() : Number(pick.get())));
  const samples = new Resource((s) => (done.get() > 0 ? hub.api.runSamples(rid.get(), epoch.get(), s) : Promise.resolve(null)));

  const options = computed(() => {
    const n = props.epochsDone;
    return [
      { value: 'latest', label: `Latest (epoch ${Math.max(1, n)})` },
      ...Array.from({ length: n }, (_, i) => n - i)
        .filter((e) => e === 1 || e % 5 === 0)
        .map((e) => ({ value: String(e), label: `Epoch ${e}` })),
    ];
  });

  const preds = (s: Sample) => (show.get() === 'truth' ? [] : s.pred.filter((p) => p.score >= conf.get()));
  const truth = (s: Sample) => (show.get() === 'pred' ? [] : s.truth);
  const hits = (s: Sample) => s.pred.filter((p) => p.score >= conf.get()).length;

  const shown = computed(() => {
    const all = samples.data?.samples ?? [];
    return showAll.get() ? all : all.slice(0, FIRST.get());
  });

  const confSlider = () => (
    <div class="conf">
      <Slider label="Min confidence" bind:value={conf} min={0.05} max={0.95} step={0.05} inline />
    </div>
  );

  return (
    <>
      {/* The epoch picker names the epoch, so phones (and wide screens, whose header holds every control) drop the meta
          that repeats it. */}
      <Panel
        title={props.title ?? `${scoresTitle(null)} samples`}
        meta={samples.data && !phone.current && !wide.current ? `epoch ${samples.data.epoch}` : undefined}
        id="samples"
        actions={() => (
          // Before the first validation there is nothing to filter: the frame holds only its one line.
          <>
            {props.epochsDone > 0 ? (
              <div class="ctl">
                <Segmented
                  size="sm"
                  label="Show boxes"
                  bind:value={show}
                  options={[
                    { value: 'both', label: 'Both' },
                    { value: 'pred', label: 'Predicted' },
                    { value: 'truth', label: 'Truth' },
                  ]}
                />
                <Select size="sm" label="Epoch" bind:value={pick} options={options.get()} />
                {wide.current ? confSlider() : null}
              </div>
            ) : null}
          </>
        )}
      >
        {/* One toolbar row under the header: the key on the left and, below 1280 px, the confidence slider (label, track,
            value) on the right. A narrow panel (the laptop column, phones) shortens the key instead of wrapping it. */}
        {props.epochsDone > 0 ? (
          <div class="bar-c">
            <div class="bar">
              <div class="legend" aria-hidden="true">
                <span>
                  <i class="solid"></i>prediction
                </span>
                <span>
                  <i class="dashed"></i>
                  <span class="long">ground truth</span>
                  <span class="short">truth</span>
                </span>
                {/* What a tile's corner count means, as a key: letters, never a count that reads like data. */}
                <span class="count-key">
                  <span class="long">n/m = predictions / true boxes</span>
                  <span class="short">n/m = predicted / true</span>
                </span>
              </div>
              {!wide.current ? confSlider() : null}
            </div>
          </div>
        ) : null}
        {props.unavailable ? (
          <p class="none">Samples show once the run's metrics load.</p>
        ) : props.epochsDone === 0 ? (
          <p class="none">{props.ended ? 'No samples: it ended before its first validation.' : 'Samples appear after the first validation pass.'}</p>
        ) : samples.error && !samples.data && props.datasetGone ? (
          <p class="none">The dataset was removed, so its sample images are gone.</p>
        ) : samples.error && !samples.data ? (
          <ErrorState error={samples.error} retry={() => samples.refresh()} compact />
        ) : !samples.data ? (
          <div class="grid" ref={gridEl}>
            <Index each={Array.from({ length: FIRST.get() })}>{() => <Skeleton h="auto" class="cell-sk" r="var(--radius-md)" />}</Index>
          </div>
        ) : (
          <>
            <div class="grid" ref={gridEl}>
              <For each={shown.get()} key={(s) => s.image_id}>
                {(s) => (
                  <button
                    class="cell"
                    style:aspect-ratio={`${s().width} / ${s().height}`}
                    onclick={() => {
                      open.set(samples.data?.samples.indexOf(s()) ?? null);
                    }}
                    aria-label={`Open sample ${s().image_id}: ${hits(s())} predictions, ${s().truth.length} ground-truth boxes`}
                  >
                    {props.datasetGone ? (
                      <span class="gone">
                        <ImageOff size={16} strokeWidth={1.75} aria-hidden="true" />
                        The dataset was removed
                      </span>
                    ) : (
                      <DatasetImage
                        datasetId={s().dataset_id}
                        imageId={s().image_id}
                        width={s().width}
                        height={s().height}
                        thumb={384}
                        boxes={preds(s())}
                        truth={truth(s())}
                        classes={props.classes}
                        compact
                      />
                    )}
                    <span class="count num">
                      {hits(s())}
                      <span class="faint">/{s().truth.length}</span>
                    </span>
                  </button>
                )}
              </For>
            </div>
            {samples.data!.samples.length > FIRST.get() ? (
              <button type="button" class="more" aria-expanded={showAll.get()} onclick={() => showAll.set(!showAll.peek())}>
                {showAll.get() ? 'Show fewer' : `Show all ${samples.data!.samples.length} samples`}
              </button>
            ) : null}
          </>
        )}
      </Panel>

      <SampleInspector
        samples={samples.data?.samples ?? []}
        index={open.get()}
        onindex={(i) => open.set(i)}
        epoch={samples.data?.epoch}
        classes={props.classes}
        preds={preds}
        truth={truth}
        describe={(x) => `${hits(x)} predictions ≥ ${conf.get().toFixed(2)} · ${x.truth.length} ground-truth boxes`}
        datasetGone={props.datasetGone}
      />
    </>
  );
}

css`
  .ctl {
    display: flex;
    gap: var(--space-2);
    align-items: center;
  }
  .bar-c {
    container-type: inline-size;
    margin-bottom: var(--space-3);
  }
  .bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2) var(--space-4);
    min-height: var(--control-sm);
  }
  .legend .short {
    display: none;
  }
  /* Container queries, not tiers: the panel's width depends on the rail beside it, so the key shortens by its own width. */
  @container (max-width: 640px) {
    .legend .long {
      display: none;
    }
    .legend .short {
      display: inline;
    }
  }
  /* Too narrow for one row (phones): the key, then the slider across the full width. */
  @container (max-width: 560px) {
    .bar {
      flex-direction: column;
      align-items: stretch;
    }
    .bar .conf {
      flex: none;
      width: 100%;
      min-width: 0;
    }
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-4);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .legend > span {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }
  .legend i {
    width: 14px;
    height: 10px;
    border-radius: 1px;
  }
  /* The keys mirror the thumbnails: predictions are solid in their class colour (three palette hues stand for "any
     class"), truth is a neutral dashed ring. */
  .solid {
    border: 2px solid;
    border-image: linear-gradient(90deg, var(--class-0), var(--class-1), var(--class-2)) 1;
  }
  .dashed {
    border: 1px dashed var(--text-secondary);
  }
  .conf {
    flex: 0 1 260px;
    min-width: 220px;
  }
  /* Thumbnails at least ~150 px wide, so boxes and prediction / truth outlines stay readable. */
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(152px, 1fr));
    gap: var(--space-2);
  }
  .more {
    align-self: flex-start;
    margin-top: var(--space-3);
    height: 28px;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .more:hover {
    color: var(--text-primary);
  }
  @media (pointer: coarse) {
    .more {
      height: 44px;
    }
  }
  .grid :global(.cell-sk) {
    aspect-ratio: 4 / 3;
  }
  .cell {
    position: relative;
    display: block;
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: 0 0 0 1px var(--line-subtle);
    transition: box-shadow var(--dur-fast) var(--ease-standard);
  }
  .cell:hover {
    box-shadow: 0 0 0 1px var(--line-strong);
  }
  .cell:focus-visible {
    outline-offset: 2px;
  }
  .count {
    position: absolute;
    right: 4px;
    bottom: 4px;
    padding: 1px 6px;
    border-radius: var(--radius-xs);
    background: #000000b3;
    color: #fff;
    font-size: var(--type-micro-size); /* small: image chip (the thumbnail's box count) */
    line-height: 14px;
  }
  .count .faint {
    color: #ffffff99;
  }
  /* The state-height rule (DESIGN_DIRECTION §2.8): the header, then one compact line with the panel's padding. */
  .none {
    color: var(--text-tertiary);
  }
  .gone {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-1);
    width: 100%;
    height: 100%;
    padding: var(--space-2);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    text-align: center;
    background: var(--bg-inset);
  }
  @media (max-width: 767px) {
    .grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .ctl :global(.seg) {
      display: none;
    }
  }
`;
