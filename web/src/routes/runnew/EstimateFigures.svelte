<script lang="ts">
  /**
   * The Summary's figures (the rail's, and its phone sheet's): the schedule, the estimated time with what it stands
   * on, a cloud run's cost and limit, GPU memory with its meter, and the warnings right above Start.
   */
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import type { OomFix } from '$lib/oom';
  import { bytes, num, usdRange } from '$lib/format';
  import { limitText } from '$lib/cloud';
  import { prefs } from '$lib/prefs.svelte';
  import { timeRange } from '$lib/throughput';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import Spinner from '$lib/ui/Spinner.svelte';
  import InfoButton from '$lib/ui/InfoButton.svelte';
  import { accumulationText } from '../run/resolved';
  import type { CloudPick } from './cloudPick.svelte';
  import type { Estimate, ShortRun, Warmup } from './estimate';
  import { roughBytes, type Shown } from './summary';

  let {
    where,
    estReady,
    sched,
    batchN,
    est,
    stepsOpen = $bindable(),
    shown,
    estStale,
    estimating,
    memShown,
    vram,
    onCloud,
    cloudPick,
    cloudCost,
    gpuMem,
    scale,
    onCpu,
    vramLevel,
    vramPct,
    noFit,
    fix,
    fixLoading,
    short,
    shortWhy = $bindable(),
    epochs = $bindable(),
  }: {
    where: string;
    estReady: boolean;
    sched: Warmup;
    batchN: number;
    est: Estimate;
    stepsOpen: boolean;
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
    shortWhy: boolean;
    epochs: number;
  } = $props();

  /** The batches behind the steps, opened under the Summary's Schedule row (ui/InfoButton): one line in the rail. */
  const batchesOf = $derived(`${num(est.stepsPerEpoch)} ${est.stepsPerEpoch === 1 ? 'batch' : 'batches'} of ${batchN}`);
  const stepsWhy = $derived(
    sched.accumulate > 1
      ? `${batchesOf} an epoch, ${sched.accumulate} to each optimizer step.`
      : `${batchesOf} an epoch, one optimizer step each.`,
  );
  const pct2 = (v: number) => `${Math.round(v * 100)}%`;
</script>

{#if !estReady}
  <dl class="est" aria-busy="true">
    <dt>Schedule</dt><dd class="num"><Skeleton w="88px" inline /></dd>
    <dt>Est. time</dt><dd class="num rough">Estimating…</dd>
  </dl>
{:else}
<dl class="est">
  <!-- The trainer's count (the short-run warning's unit). The batches behind it, one tap, click or Enter away
       (ui/InfoButton: the value is its button), as a line under the row: in a list of rows a popover would cover
       its neighbours (round 24's covered Start from), so it opens in place and pushes the rows below down. -->
  <dt>Schedule</dt>
  <dd class="num sched">
    <InfoButton label="How the steps are counted" controls="{where}-steps-why" bind:open={stepsOpen} class="steps"
      >{#snippet trigger()}{num(sched.stepsPerEpoch)} optimizer steps <span class="faint">/ epoch</span>{/snippet}</InfoButton
    >
  </dd>
  <!-- Batches accumulate into one optimizer step below a batch of 64 (docs/MODEL.md §8.3): said beside the schedule,
       in the one wording Run's Config and Compare use (run/resolved `accumulationText`), never "per step" for a batch. -->
  {#if sched.accumulate > 1}<dd class="sched-acc num">{accumulationText(sched.accumulate, batchN)}</dd>{/if}
  {#if stepsOpen}<dd class="steps-why" id="{where}-steps-why">{stepsWhy}</dd>{/if}
  <!-- A range, never one precise figure; a rough one (low confidence) in secondary ink. What it stands on is the
       line under it, which opens the phases behind it. -->
  <dt>Est. time</dt><dd class="num" class:rough={shown.rough} class:stale={estStale} aria-busy={estimating || undefined}>{timeRange(shown.lo, shown.hi)}</dd>
  <dd class="basis">
    <details class="est-how" class:stale={estStale}>
      <!-- While the answer for these settings is out, the line says so in place of the old basis (`estStale`). -->
      <summary
        >{#if estStale}<span class="updating"><Spinner size={12} />Updating…</span>{:else}<span>{shown.basis}</span>{/if}<ChevronRight
          size={12}
          strokeWidth={2}
          aria-hidden="true"
        /></summary
      >
      <!-- One grid for every row: the term wraps in its own column, the value sits on the term's first line at the
           right (never pushed to a line of its own), one word for a source, never a phrase in the value column. -->
      <dl class="how-list num">
        {#each shown.phases as p (p.phase)}
          <div>
            <dt>{num(p.epochs)} {p.epochs === 1 ? 'epoch' : 'epochs'}, {p.phase === 'mosaic' ? 'mosaic' : 'no mosaic'}</dt>
            <dd>{num(p.epoch_s, p.epoch_s < 10 ? 1 : 0)} s each · {num(p.train_imgs_per_s)} img/s</dd>
          </div>
        {/each}
        <div>
          <dt>Validation{shown.val.reference ? ', reference' : ''}</dt>
          <dd>{num(shown.val.images)} images · {num(shown.val.rate)} img/s</dd>
        </div>
        <div><dt>First epoch</dt><dd>+{num(shown.startup, 1)} s startup</dd></div>
        {#if memShown}
          <!-- The Summary's figure, with what it stands on: a row's value is a number, never a source alone. -->
          <div><dt>Peak GPU memory</dt><dd>{roughBytes(vram)} {shown.vramMeasured ? 'measured' : 'reference'}</dd></div>
        {/if}
      </dl>
      <p class="how-note">{shown.how}{#if shown.updated}{' '}Last measured {shown.updated}.{/if}</p>
    </details>
  </dd>
  {#if onCloud && cloudPick.offer && cloudPick.hours !== null}
    <!-- Money before it is spent: the training's range, with what comes on top, and the most the run can cost, beside the time
         they follow from (the memory and its meter stay together, last). -->
    {#if cloudCost}
      <dt>Est. cost</dt>
      <dd class="num" class:rough={shown.rough} class:stale={estStale}>{usdRange(cloudCost.lo, cloudCost.hi)}</dd>
      <dd class="cost-note">Training only: booting and the upload come on top.</dd>
    {/if}
    <dt>Limit</dt>
    <dd class="num">{limitText(cloudPick.offer.price_per_hour, cloudPick.hours)}</dd>
  {/if}
  {#if memShown}
    <dt>Est. GPU memory</dt>
    <!-- The figure is a metric, in ink; its severity is the meter's and the line's under it (§2.6). A GPU that does not
         report its memory (0) gets no denominator. -->
    <dd class="num" class:stale={estStale}>{roughBytes(vram)}{#if gpuMem}{' '}<span class="faint">/ {bytes(gpuMem, prefs.units)}</span>{/if}</dd>
  {/if}
</dl>
<!-- Some adapters report no memory size (`memory_bytes: 0`, e.g. through Vulkan): the estimate stands alone, said once. -->
{#if memShown && !gpuMem}<p class="calib">{onCloud ? 'The offer does not say how much memory its GPU has.' : 'This GPU’s memory size isn’t reported.'}</p>{/if}
{#if scale > 1 && !onCpu}<p class="calib">Scaled up from the failed run's settings, which did not fit.</p>{/if}
{#if memShown && gpuMem}
  <div class="vram" class:stale={estStale} aria-hidden="true"><i class={vramLevel} style:width={pct2(Math.min(1, vramPct))}></i></div>
{/if}
<!-- Warnings in the rail are one-line callouts: the status's icon in its hue, the words and the Why? link in ink, no
     tinted fill (fills are for banners), right above Start. -->
{#if noFit}
  <!-- The change that fits is the action under it (Use batch 8); with none, the line says what to lower. -->
  <p class="callout danger" id="{where}-nofit" role="alert">
    <TriangleAlert size={14} aria-hidden="true" /><span
      >Likely won't fit: {onCloud && cloudPick.offer ? `the ${cloudPick.offer.gpu}` : 'this GPU'} has {bytes(gpuMem, prefs.units)}.{#if !fix && !fixLoading}{' '}No smaller batch, image size or model fits with 10% to spare.{/if}</span
    >
  </p>
{:else if onCpu}
  <p class="callout"><TriangleAlert size={14} aria-hidden="true" /><span>Training on the CPU is roughly 14× slower than on a GPU.</span></p>
{/if}
{/if}
{#if short}
  <!-- The trainer's warnings (folo_vision::train::schedule_warnings) as one line; the reasons and the fix behind Why?. -->
  <div class="short">
    <p class="callout">
      <TriangleAlert size={14} aria-hidden="true" />
      <span>{#if short.scratch}Too short to learn: {num(short.steps)} optimizer steps.{:else}Mostly warmup: {num(short.warmup)} of {num(short.steps)} optimizer steps.{/if}</span>
      <button type="button" class="more" aria-expanded={shortWhy} aria-controls="{where}-short-why" onclick={() => (shortWhy = !shortWhy)}>Why?</button>
    </p>
    {#if shortWhy}
      <div class="short-why" id="{where}-short-why">
        {#if short.scratch}
          <!-- Two short sentences: the batches behind the steps open under the Schedule row. -->
          <p>{num(epochs)} epochs × {num(short.stepsPerEpoch)} optimizer steps. From scratch, a detector needs a few hundred, so expect an mAP near&nbsp;0.</p>
        {/if}
        {#if short.warmupHeavy}
          <p>Warmup takes {num(short.warmup)} of the {num(short.steps)} optimizer steps, so the learning rate peaks late.</p>
        {/if}
        <button type="button" class="fix" onclick={() => (epochs = short.epochs)}>Train {num(short.epochs)} epochs</button>
      </div>
    {/if}
  </div>
{/if}

<style>
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
  /* A figure for earlier settings while these settings' is out (`estStale`): kept, so nothing jumps, but tertiary, so it
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
  /* The meter draws the same kept figure (`estStale`): tertiary with it, whatever its level. */
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
</style>
