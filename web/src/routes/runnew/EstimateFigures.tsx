/**
 * The Summary's figures (the rail's, and its phone sheet's): the schedule, the estimated time with what it stands
 * on, a cloud run's cost and limit, GPU memory with its meter, and the warnings right above Start.
 */
import ChevronRight from '$icons/chevron-right';
import TriangleAlert from '$icons/triangle-alert';
import { bindable, computed, css, For, type Signal } from 'fw';
import type { OomFix } from '$lib/oom';
import { bytes, num, usdRange } from '$lib/format';
import { limitText } from '$lib/cloud';
import { prefs } from '$lib/prefs';
import { timeRange } from '$lib/throughput';
import Skeleton from '$lib/ui/Skeleton';
import Spinner from '$lib/ui/Spinner';
import InfoButton from '$lib/ui/InfoButton';
import NoValue from '$lib/ui/NoValue';
import { accumulationText } from '../run/resolved';
import type { CloudPick } from './cloudPick';
import type { Estimate, ShortRun, Warmup } from './estimate';
import { roughBytes, type Shown } from './summary';

export interface EstimateFiguresProps {
  where: string;
  /**
   * Settings no figure can stand for (an epoch count out of range, a dataset with nothing to train on): dashes, never a
   * figure for settings Start refuses.
   */
  blocked: boolean;
  estReady: boolean;
  sched: Warmup;
  batchN: number;
  est: Estimate;
  stepsOpen?: boolean;
  shown: Shown;
  estStale: boolean;
  estimating: boolean;
  memShown: boolean;
  vram: number;
  onCloud: boolean;
  cloudPick: CloudPick;
  cloudCost: { lo: number; hi: number } | null;
  gpuMem: number | null;
  scale: number;
  onCpu: boolean;
  vramLevel: 'danger' | 'warning' | null;
  vramPct: number;
  noFit: boolean;
  fix: OomFix | null;
  fixLoading: boolean;
  short: ShortRun | null;
  shortWhy?: boolean;
  epochs?: number;
}

const pct2 = (v: number) => `${Math.round(v * 100)}%`;

export default function EstimateFigures(props: EstimateFiguresProps) {
  // Bound by the page (`$bindable` in the Svelte original): never undefined.
  const stepsOpen = bindable(props, 'stepsOpen') as Signal<boolean>;
  const shortWhy = bindable(props, 'shortWhy') as Signal<boolean>;
  const epochs = bindable(props, 'epochs') as Signal<number>;

  /** The batches behind the steps, opened under the Summary's Schedule row (ui/InfoButton): one line in the rail. */
  const batchesOf = computed(() => `${num(props.est.stepsPerEpoch)} ${props.est.stepsPerEpoch === 1 ? 'batch' : 'batches'} of ${props.batchN}`);
  const stepsWhy = computed(() =>
    props.sched.accumulate > 1 ? `${batchesOf.get()} an epoch, ${props.sched.accumulate} to each optimizer step.` : `${batchesOf.get()} an epoch, one optimizer step each.`,
  );

  return (
    <>
      {props.blocked ? (
        <dl class="est">
          <dt>Schedule</dt>
          <dd class="num">
            <NoValue label="no estimate" />
          </dd>
          <dt>Est. time</dt>
          <dd class="num">
            <NoValue label="no estimate" />
          </dd>
        </dl>
      ) : !props.estReady ? (
        <dl class="est" aria-busy="true">
          <dt>Schedule</dt>
          <dd class="num">
            <Skeleton w="88px" inline />
          </dd>
          <dt>Est. time</dt>
          <dd class="num rough">Estimating…</dd>
        </dl>
      ) : (
        <>
          <dl class="est">
            {/* The trainer's count (the short-run warning's unit). The batches behind it, one tap, click or Enter away
                (ui/InfoButton: the value is its button), as a line under the row: in a list of rows a popover would
                cover its neighbours (round 24's covered Start from), so it opens in place and pushes the rows below
                down. */}
            <dt>Schedule</dt>
            <dd class="num sched">
              <InfoButton
                label="How the steps are counted"
                controls={`${props.where}-steps-why`}
                bind:open={stepsOpen}
                class="steps"
                trigger={() => (
                  <>
                    {num(props.sched.stepsPerEpoch)} optimizer steps <span class="faint">/ epoch</span>
                  </>
                )}
              />
            </dd>
            {/* Batches accumulate into one optimizer step below a batch of 64 (docs/MODEL.md §8.3): said beside the
                schedule, in the one wording Run's Config and Compare use (run/resolved `accumulationText`), never "per
                step" for a batch. */}
            {props.sched.accumulate > 1 ? <dd class="sched-acc num">{accumulationText(props.sched.accumulate, props.batchN)}</dd> : null}
            {stepsOpen.get() ? (
              <dd class="steps-why" id={`${props.where}-steps-why`}>
                {stepsWhy.get()}
              </dd>
            ) : null}
            {/* A range, never one precise figure; a rough one (low confidence) in secondary ink. What it stands on is the
                line under it, which opens the phases behind it. */}
            <dt>Est. time</dt>
            <dd class="num" class:rough={props.shown.rough} class:stale={props.estStale} aria-busy={props.estimating || undefined}>
              {timeRange(props.shown.lo, props.shown.hi)}
            </dd>
            <dd class="basis">
              <details class="est-how" class:stale={props.estStale}>
                {/* While the answer for these settings is out, the line says so in place of the old basis (`estStale`). */}
                <summary>
                  {props.estStale ? (
                    <span class="updating">
                      <Spinner size={12} />
                      Updating…
                    </span>
                  ) : (
                    <span>{props.shown.basis}</span>
                  )}
                  <ChevronRight size={12} strokeWidth={2} aria-hidden="true" />
                </summary>
                {/* One grid for every row: the term wraps in its own column, the value sits on the term's first line at
                    the right (never pushed to a line of its own), one word for a source, never a phrase in the value
                    column. */}
                <dl class="how-list num">
                  <For each={props.shown.phases} key={(p) => p.phase}>
                    {(p) => (
                      <div>
                        <dt>
                          {num(p().epochs)} {p().epochs === 1 ? 'epoch' : 'epochs'}, {p().phase === 'mosaic' ? 'mosaic' : 'no mosaic'}
                        </dt>
                        <dd>
                          {num(p().epoch_s, p().epoch_s < 10 ? 1 : 0)} s each · {num(p().train_imgs_per_s)} img/s
                        </dd>
                      </div>
                    )}
                  </For>
                  <div>
                    <dt>Validation{props.shown.val.reference ? ', reference' : ''}</dt>
                    <dd>
                      {num(props.shown.val.images)} images · {num(props.shown.val.rate)} img/s
                    </dd>
                  </div>
                  <div>
                    <dt>First epoch</dt>
                    <dd>+{num(props.shown.startup, 1)} s startup</dd>
                  </div>
                  {props.memShown ? (
                    // The Summary's figure, with what it stands on: a row's value is a number, never a source alone.
                    <div>
                      <dt>Peak GPU memory</dt>
                      <dd>
                        {roughBytes(props.vram)} {props.shown.vramMeasured ? 'measured' : 'reference'}
                      </dd>
                    </div>
                  ) : null}
                </dl>
                <p class="how-note">
                  {props.shown.how}
                  {props.shown.updated ? <>{` Last measured ${props.shown.updated}.`}</> : null}
                </p>
              </details>
            </dd>
            {props.onCloud && props.cloudPick.offer && props.cloudPick.hours !== null ? (
              // Money before it is spent: the training's range, with what comes on top, and the most the run can cost,
              // beside the time they follow from (the memory and its meter stay together, last).
              <>
                {props.cloudCost ? (
                  <>
                    <dt>Est. cost</dt>
                    <dd class="num" class:rough={props.shown.rough} class:stale={props.estStale}>
                      {usdRange(props.cloudCost.lo, props.cloudCost.hi)}
                    </dd>
                    <dd class="cost-note">Training only: booting and the upload come on top.</dd>
                  </>
                ) : null}
                <dt>Limit</dt>
                <dd class="num">{limitText(props.cloudPick.offer!.price_per_hour, props.cloudPick.hours!)}</dd>
              </>
            ) : null}
            {props.memShown ? (
              <>
                <dt>Est. GPU memory</dt>
                {/* The figure is a metric, in ink; its severity is the meter's and the line's under it (§2.6). A GPU that
                    does not report its memory (0) gets no denominator. */}
                <dd class="num" class:stale={props.estStale}>
                  {roughBytes(props.vram)}
                  {props.gpuMem ? (
                    <>
                      {' '}
                      <span class="faint">/ {bytes(props.gpuMem, prefs.units)}</span>
                    </>
                  ) : null}
                </dd>
              </>
            ) : null}
          </dl>
          {/* Some adapters report no memory size (`memory_bytes: 0`, e.g. through Vulkan): the estimate stands alone,
              said once. */}
          {props.memShown && !props.gpuMem ? (
            <p class="calib">{props.onCloud ? 'The offer does not say how much memory its GPU has.' : 'This GPU’s memory size isn’t reported.'}</p>
          ) : null}
          {props.scale > 1 && !props.onCpu ? <p class="calib">Scaled up from the failed run's settings, which did not fit.</p> : null}
          {props.memShown && props.gpuMem ? (
            <div class="vram" class:stale={props.estStale} aria-hidden="true">
              <i class={props.vramLevel} style:width={pct2(Math.min(1, props.vramPct))}></i>
            </div>
          ) : null}
          {/* Warnings in the rail are one-line callouts: the status's icon in its hue, the words and the Why? link in
              ink, no tinted fill (fills are for banners), right above Start. */}
          {props.noFit ? (
            // The change that fits is the action under it (Use batch 8); with none, the line says what to lower.
            <p class="callout danger" id={`${props.where}-nofit`} role="alert">
              <TriangleAlert size={14} aria-hidden="true" />
              <span>
                Likely won't fit: {props.onCloud && props.cloudPick.offer ? `the ${props.cloudPick.offer.gpu}` : 'this GPU'} has{' '}
                {bytes(props.gpuMem, prefs.units)}.
                {!props.fix && !props.fixLoading ? <>{' '}No smaller batch, image size or model fits with 10% to spare.</> : null}
              </span>
            </p>
          ) : props.onCpu ? (
            <p class="callout">
              <TriangleAlert size={14} aria-hidden="true" />
              <span>Training on the CPU is roughly 14× slower than on a GPU.</span>
            </p>
          ) : null}
        </>
      )}
      {props.short ? (
        // The trainer's warnings (folo_vision::train::schedule_warnings) as one line; the reasons and the fix behind Why?.
        <div class="short">
          <p class="callout">
            <TriangleAlert size={14} aria-hidden="true" />
            <span>
              {props.short.scratch ? (
                <>Too short to learn: {num(props.short.steps)} optimizer steps.</>
              ) : (
                <>
                  Mostly warmup: {num(props.short.warmup)} of {num(props.short.steps)} optimizer steps.
                </>
              )}
            </span>
            <button
              type="button"
              class="more"
              aria-expanded={shortWhy.get()}
              aria-controls={`${props.where}-short-why`}
              onclick={() => shortWhy.set(!shortWhy.get())}
            >
              Why?
            </button>
          </p>
          {shortWhy.get() ? (
            <div class="short-why" id={`${props.where}-short-why`}>
              {props.short.scratch ? (
                // Two short sentences: the batches behind the steps open under the Schedule row.
                <p>
                  {num(epochs.get())} epochs × {num(props.short.stepsPerEpoch)} optimizer steps. From scratch, a detector needs a few hundred, so
                  expect an mAP near&nbsp;0.
                </p>
              ) : null}
              {props.short.warmupHeavy ? (
                <p>
                  Warmup takes {num(props.short.warmup)} of the {num(props.short.steps)} optimizer steps, so the learning rate peaks late.
                </p>
              ) : null}
              <button type="button" class="fix" onclick={() => epochs.set(props.short!.epochs)}>
                Train {num(props.short.epochs)} epochs
              </button>
            </div>
          ) : null}
        </div>
      ) : null}
    </>
  );
}

css`
  .est {
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 6px var(--space-3);
    font-size: var(--type-sm-size);
  }
  .est dt {
    color: var(--text-tertiary);
  }
  .est dd {
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  /* What a cloud run's cost range leaves out: a quiet line under it, across the rail from its label's edge. */
  .est dd.cost-note {
    grid-column: 1 / -1;
    margin-top: -2px;
    color: var(--text-tertiary);
    text-align: left;
    white-space: normal;
  }
  /* A rough estimate (nothing close measured here) reads one step quieter than a measured one. */
  .est dd.rough {
    color: var(--text-secondary);
  }
  /* A figure for earlier settings while these settings' is out (\`estStale\`): kept, so nothing jumps, but tertiary, so it
     never reads as current. The dock's figures dim the same way. */
  .est dd.stale,
  .est-how.stale .how-list dd {
    color: var(--text-tertiary);
  }
  .updating {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
  }
  /* What the time stands on: one quiet line under it, across the row, opening the phases behind it. */
  .est dd.basis {
    grid-column: 1 / -1;
    margin-top: -2px;
    overflow: visible;
    white-space: normal;
  }
  .est-how summary {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    max-width: 100%;
    border-radius: var(--radius-xs);
    color: var(--text-tertiary);
    cursor: pointer;
    list-style: none;
  }
  .est-how summary::-webkit-details-marker {
    display: none;
  }
  .est-how summary > span {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .est-how summary :global(svg) {
    flex-shrink: 0;
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .est-how[open] summary :global(svg) {
    transform: rotate(90deg);
  }
  .est-how summary:hover {
    color: var(--text-secondary);
  }
  .how-list {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 2px var(--space-3);
    margin-top: var(--space-2);
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    text-align: left;
  }
  .how-list > div {
    display: contents;
  }
  .how-list dt {
    color: var(--text-tertiary);
  }
  .how-list dd {
    align-self: start;
    color: var(--text-secondary);
    text-align: right;
    white-space: nowrap;
  }
  /* Below 1280 the rail (300 px) and the summary block are too narrow for a term and its value side by side ("90
     epochs, / mosaic" wrapped while its value held one line): each phase is its term over its value, one column. */
  @media (max-width: 1279px) {
    .how-list {
      grid-template-columns: minmax(0, 1fr);
      row-gap: 0;
    }
    .how-list dd {
      text-align: left;
      white-space: normal;
    }
    .how-list > div + div > dt {
      margin-top: 6px;
    }
  }
  .how-note {
    max-inline-size: var(--measure);
    margin-top: var(--space-2);
    text-align: left;
    color: var(--text-tertiary);
    text-wrap: pretty;
  }
  @media (pointer: coarse) {
    .est-how summary {
      min-height: 44px;
    }
  }
  /* The steps open their batches (ui/InfoButton): a button, so the value keeps room for its focus ring. */
  .est dd.sched {
    overflow: visible;
  }
  /* Accumulation, beside the schedule: a meta line under the row, across the rail from its label's edge, wrapping
     rather than cut. Left-aligned, as prose is: a right-aligned sentence that wraps leaves an orphan at the far edge. */
  .est dd.sched-acc {
    grid-column: 1 / -1;
    margin-top: -2px;
    color: var(--text-tertiary);
    text-align: left;
    white-space: normal;
  }
  /* The batches behind the steps: one quiet sentence under the row, from the label's edge across the rail. */
  .est dd.steps-why {
    grid-column: 1 / -1;
    margin-top: -2px;
    color: var(--text-secondary);
    text-align: left;
    white-space: normal;
    text-wrap: pretty;
  }
  /* A neutral meter; the warning colour within 10 % of the GPU's memory, danger over it (the line under it says so). */
  .vram {
    height: 4px;
    margin-top: calc(-1 * var(--space-2));
    border-radius: 2px;
    background: var(--bg-active);
    overflow: hidden;
  }
  .vram i {
    display: block;
    height: 100%;
    background: var(--text-secondary);
    transition: width var(--dur-chart) var(--ease-standard);
  }
  .vram i.warning {
    background: var(--warning-solid);
  }
  .vram i.danger {
    background: var(--danger-solid);
  }
  /* The meter draws the same kept figure (\`estStale\`): tertiary with it, whatever its level. */
  .vram.stale i {
    background: var(--text-tertiary);
  }
  /* A callout: one line at 12 px after its icon, no fill (tinted fills are for banners). The status shows through the
     icon alone: the words are ink and Why? an ink link (DESIGN_DIRECTION §2.6), so no amber text turns weak in light. */
  .callout {
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-primary);
    text-wrap: pretty;
  }
  .callout > :global(svg) {
    color: var(--warning-text);
  }
  .callout.danger > :global(svg) {
    color: var(--danger-text);
  }
  .callout > :global(svg) {
    flex-shrink: 0;
    margin-top: 3px; /* grid-ok: the 14 px icon centred on the 20 px first line */
  }
  .callout > span {
    min-width: 0;
  }
  .short {
    display: grid;
    gap: var(--space-2);
  }
  /* The reasons and the fix, under the line, indented to its text. */
  .short-why {
    display: grid;
    justify-items: start;
    max-inline-size: calc(var(--measure) + 22px);
    gap: var(--space-2);
    padding-left: 22px; /* grid-ok: the icon (14) and the gap (8) */
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    text-wrap: pretty;
  }
  /* "Why?" opens the reasons and the fix, right after the line's text. */
  .more {
    flex-shrink: 0;
    color: var(--text-primary);
    text-decoration: underline;
    text-decoration-color: color-mix(in srgb, currentColor 40%, transparent);
    text-underline-offset: 3px;
    border-radius: var(--radius-xs);
  }
  .more:hover {
    text-decoration-color: currentColor;
  }
  /* The fix a short-run warning suggests, under its reasons. */
  .fix {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
    text-align: left;
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
    border-radius: var(--radius-xs);
  }
  .fix:hover {
    text-decoration-color: currentColor;
  }
  @media (pointer: coarse) {
    .fix,
    .more {
      min-height: 44px;
    }
  }
  .calib {
    margin-top: calc(-1 * var(--space-1));
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  @media (max-width: 1023px) {
    :global(.summary) .est {
      padding-top: 0;
      border-top: 0;
    }
  }
  @media (max-width: 767px) {
    :global(.summary) .est {
      padding-top: var(--space-3);
      border-top: 1px solid var(--line-subtle);
    }
  }
`;
