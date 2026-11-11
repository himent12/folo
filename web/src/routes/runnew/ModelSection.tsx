// fw port of ModelSection.svelte.
/**
 * New run, section 2: the model size (a specification table; on phones a segmented pick with the table folded
 * behind "Compare sizes") and the weights the run starts from.
 */
import Check from '$icons/check';
import ChevronRight from '$icons/chevron-right';
import { bindable, computed, css, For, MediaQuery, signal } from 'fw';
import type { Resource } from '$lib/api/resource';
import type { Arch, Model, Run, ThroughputTable } from '$lib/api/types';
import { scoreText, type ModelChoice, type ScoreContext } from '$lib/scoreSummary';
import { FROM_SCRATCH } from '$lib/lineage';
import { compact, num } from '$lib/format';
import { latencyMs, latencyRange, measuredLatency } from '$lib/throughput';
import Field from '$lib/ui/Field.tsx';
import Panel from '$lib/ui/Panel';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import Ident from '$lib/ui/Ident';
import Meta from '$lib/ui/Meta';
import InfoButton from '$lib/ui/InfoButton';
import { arches } from './estimate';
import type { FormErrors } from './jump.fw';
import { media } from '$lib/design/tokens';

export interface ModelSectionProps {
  arch?: Arch;
  weights?: string;
  tp: Resource<ThroughputTable | null>;
  oomRun: Run | null;
  was: (key: string) => string | undefined;
  pickedPublished: Model | null;
  selArch: (typeof arches)[number] | undefined;
  startHint: string;
  pickedAbout: string | null;
  errors: FormErrors;
  weightChoices: ModelChoice<Model>[];
  publishedOptions: Model[];
  scoreCtx: (m: Model) => ScoreContext;
  pickWeights: (id: string) => void;
}

/** Where the architectures' reference latency was measured (docs/BENCHMARKS.md), where the line already says 640 px
 *  and end to end. */
const REF_WHERE = 'RTX 4080 SUPER, f32';

export default function ModelSection(props: ModelSectionProps) {
  const arch = bindable(props, 'arch') as { get(): Arch; set(v: Arch): void; peek(): Arch };
  const weights = bindable(props, 'weights') as { get(): string; set(v: string): void; peek(): string };
  /** Phones: whether Compare sizes is open (its rows are then the one size picker). */
  const sizesOpen = signal(false);
  // Phones pick the model size from a segmented control; the full specification table folds behind "Compare sizes".
  const phone = new MediaQuery(media.phone, false);
  const aboutOpen = signal(false);

  // The model sizes, one specification row each (a table on wide screens; behind "Compare sizes" on phones).
  // One specification row per size, the column header on the same grid (subgrid), so every value sits under its
  // heading at every width.
  const archTable = () => (
    <>
      <div class="arches" role="radiogroup" aria-label="Model size">
        <div class="a-head" aria-hidden="true">
          <span class="a-head-lead">Size</span>
          <span>Params</span>
          <span>GFLOPs</span>
          <span class="a-head-lat">Latency</span>
          <span></span>
        </div>
        <For each={arches} key={(a) => a.arch}>
          {(a) => {
            const on = () => a().arch === arch.get();
            const here = computed(() => measuredLatency(props.tp.data, a().arch));
            // Each figure says where it comes from in its own cell (a tertiary word), so the caption under the table
            // stays one line: "measured" on this GPU, "reference" for the architecture's benchmark. A dash is explained
            // once, in the caption (DESIGN_DIRECTION 23), its word slot kept empty so dashes and figures share one edge.
            return (
              <label class="arch" class:on={on()}>
                <input type="radio" name="arch" value={a().arch} bind:group={arch} class="sr-only" />
                <span class="a-size">{a().size}</span>
                <Ident value={a().arch} class="a-name" />
                <span class="a-note">{a().note}</span>
                <span class="a-v a-params num">{compact(a().params)}</span>
                <span class="a-v a-flops num">{num(a().flops, 1)}</span>
                {here.get() ? (
                  <span class="a-v lat">
                    <span class="num here">{num(here.get()!.latency_ms, 2)} ms</span>
                    <span class="a-src">measured</span>
                  </span>
                ) : a().ms !== undefined ? (
                  <span class="a-v lat">
                    <span class="num">{num(a().ms!, 2)} ms</span>
                    <span class="a-src">reference</span>
                  </span>
                ) : (
                  <span class="a-v lat">
                    <span class="faint" title="Not measured yet">
                      —
                    </span>
                    <span class="a-src" aria-hidden="true"></span>
                  </span>
                )}
                <span class="tick" aria-hidden="true">
                  {on() ? <Check size={14} strokeWidth={2.25} /> : null}
                </span>
              </label>
            );
          }}
        </For>
      </div>
      {/* One line: what the column measures and where the reference comes from; each cell names its own source. Never
          these weights' own timing. */}
      <p class="ref num">
        <Meta>
          <span>Latency at 640 px, end to end</span>
          <span>reference: {REF_WHERE}</span>
          <span>—: not measured yet</span>
        </Meta>
      </p>
    </>
  );

  /** Measured on this server: the chosen size's spread and count, the table's "measured" figure in full. */
  const selHere = computed(() => (props.selArch ? measuredLatency(props.tp.data, props.selArch.arch) : undefined));

  return (
    <Panel title="2 · Model" id="sec-model" meta={props.was('model') ? `was ${props.oomRun?.config.arch}` : undefined}>
      {props.pickedPublished ? (
        // Published weights set the size (a fine-tune trains the weights' own): said where the size is picked.
        <p class="a-from" role="status">
          <Ident value={props.pickedPublished.arch} />, from <Ident value={props.pickedPublished.name} />.{' '}
          <span class="faint">The run trains the weights' size; another size starts from scratch.</span>
        </p>
      ) : null}
      {phone.current ? (
        <>
          {/* Phones: the size as one segmented pick and one short recommendation in words; its figures (params, compute,
              latency and where each was measured) fold into "Compare sizes", so the section stays the pick, the
              recommendation and Start from. One selection surface at a time: while Compare sizes is open its rows are
              the selector, and the compact pick and its sentence step aside until it closes. */}
          {!sizesOpen.get() ? (
            <div class="a-pick">
              <Segmented id="arch-pick" label="Model size" bind:value={arch} options={arches.map((a) => ({ value: a.arch, label: a.size }))} />
              {props.selArch ? (
                <p class="a-sel">
                  <Ident value={props.selArch.arch} /> {props.selArch.note}
                </p>
              ) : null}
            </div>
          ) : null}
          <details class="advanced a-compare" bind:open={sizesOpen}>
            <summary>
              <ChevronRight size={14} aria-hidden="true" /> Compare sizes <span class="faint">params, compute, latency</span>
            </summary>
            {archTable()}
            {props.selArch && selHere.get() ? (
              <>
                <p class="a-lat num">
                  <Ident value={props.selArch.arch} /> measured here <span class="a-lat-v">{latencyMs(selHere.get()!.latency_ms)}</span>
                </p>
                <p class="a-specs num">
                  {latencyRange(selHere.get()!.latency_ms_low, selHere.get()!.latency_ms_high)} · {num(selHere.get()!.samples)} predictions · 640 px ·{' '}
                  {selHere.get()!.precision} · end to end
                </p>
              </>
            ) : null}
          </details>
        </>
      ) : (
        archTable()
      )}
      <div class="row2">
        {/* Closed, the select names the weights alone; their score and what it is on are the hint under it (never cut
            at the select's edge), with published weights' own description (`Model.about`, also each option's tooltip).
            The open list keeps the groups and each option's context. Published weights: the plain line is the hint;
            where they come from (the server's own words) opens as a line under the field (InfoButton `controls`), never
            a popover over the fields below. */}
        <Field
          label="Start from"
          for="weights"
          hint={props.startHint}
          error={props.errors.weights}
          aside={() => (
            <>{props.pickedAbout ? <InfoButton label="Where these weights come from" controls="weights-about" bind:open={aboutOpen} /> : null}</>
          )}
        >
          <Select
            id="weights"
            bind:value={weights}
            onchange={props.pickWeights}
            invalid={!!props.errors.weights}
            options={[
              { value: 'none', label: FROM_SCRATCH },
              ...props.weightChoices.map((c) => ({ value: c.model.id, label: c.label, group: c.group, short: c.short })),
              ...props.publishedOptions.map((m) => ({
                value: m.id,
                label: m.metrics ? `${m.name} · ${scoreText(m.metrics.map50_95, props.scoreCtx(m))}` : m.name,
                group: 'Published weights',
                title: m.about,
                short: m.name,
              })),
            ]}
          />
        </Field>
        {props.pickedAbout && aboutOpen.get() ? (
          <p class="weights-about" id="weights-about">
            {props.pickedAbout}
          </p>
        ) : null}
      </div>
    </Panel>
  );
}

css`
  .arch {
    position: relative;
    border-radius: var(--radius-md);
    border: 1px solid var(--line-default);
    background: var(--bg-raised);
    cursor: pointer;
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      background-color var(--dur-fast) var(--ease-standard),
      box-shadow var(--dur-fast) var(--ease-standard);
  }
  .arch:hover {
    border-color: var(--line-strong);
  }
  /* A model size is a row in a list: the one selection rule, bg.active and the 2 px ink bar, plus the check. */
  .arch.on {
    background: var(--bg-active);
    box-shadow: var(--select-bar);
  }
  .tick {
    display: grid;
    color: var(--text-primary);
  }
  .arch:has(:focus-visible) {
    outline: 2px solid var(--focus);
    outline-offset: 2px;
  }
  /* Model sizes: single-line rows (letter, name, note, the three specs, the check) on one grid with their header.
     A subgrid's edge tracks hold its border and padding, so the first and last are 12 + 1 px wider than their content. */
  .arches {
    display: grid;
    grid-template-columns: 28px auto minmax(0, 1fr) repeat(2, 64px) auto 28px;
    column-gap: var(--space-3);
    row-gap: var(--space-1);
  }
  .a-head,
  .arch {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: subgrid;
    align-items: center;
  }
  .a-head {
    padding: 0 calc(var(--space-3) + 1px) var(--space-1);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    text-align: right;
  }
  .a-head-lead {
    grid-column: 1 / 4;
    text-align: left;
  }
  /* The latency column holds a figure and its source word side by side: its heading starts where the column does
     (phones stack the word under its figure, right-aligned, and keep the heading on the right). */
  @media (min-width: 768px) {
    .a-head-lat {
      text-align: left;
    }
  }
  .arch {
    min-height: 36px;
    padding: 0 var(--space-3);
  }
  .a-size {
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    font-weight: var(--weight-semibold);
  }
  .arch :global(.a-name) {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .arch.on :global(.a-name) {
    color: var(--text-secondary);
  }
  /* A sentence is never cut (DESIGN_DIRECTION 2.3): where its column is short it wraps, and below 1280 it takes a line
     of its own under the size's name, so the row grows instead (round 24's "Fastest; edge devices, qâ€¦" at 1024). */
  .a-note {
    min-width: 0;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  @media (min-width: 768px) and (max-width: 1279px) {
    .arch {
      grid-template-rows: auto auto;
      row-gap: 0;
      padding-block: 6px;
    }
    .a-size,
    .a-v,
    .tick {
      grid-row: 1 / 3;
    }
    .arch :global(.a-name) {
      grid-column: 2 / 4;
      grid-row: 1;
    }
    .a-note {
      grid-column: 2 / 4;
      grid-row: 2;
    }
  }
  .a-v {
    font-size: var(--type-sm-size);
    text-align: right;
    white-space: nowrap;
  }
  .a-v.num,
  .lat .num {
    color: var(--text-secondary);
  }
  .arch.on .a-v.num,
  .arch.on .lat .num {
    color: var(--text-primary);
  }
  /* Measured on this server: ink in every row (its cell says "measured"). */
  .lat .num.here {
    color: var(--text-primary);
  }
  /* The figure's source, one step quieter than the figure, in a slot of one width so the figures keep one edge. */
  .a-src {
    display: inline-block;
    min-width: 10ch;
    margin-left: var(--space-2);
    text-align: left;
    color: var(--text-tertiary);
  }
  /* A caption, not a paragraph: one meta line under the table (it wraps whole items on phones). */
  .ref {
    margin-top: var(--space-2);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  /* Start from on the same half column as the fields below it (Epochs, Device): one grid of two tracks with the same
     gap, the field in the first; one column on phones. */
  .row2 {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    column-gap: var(--space-5);
    margin-top: var(--space-4);
  }
  .row2 > :global(*) {
    grid-column: 1;
  }
  /* Where published weights come from, opened from Start from's ⓘ: a quiet line under the field, at the measure. */
  .weights-about {
    margin-top: var(--space-2);
    max-inline-size: var(--measure);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  /* Published weights set the size: one line above the picker, the identifiers mono (ui/Ident). */
  .a-from {
    margin-bottom: var(--space-3);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    text-wrap: pretty;
  }
  /* Phones: the chosen size in words under its segmented pick, its specs in one line. */
  .a-sel {
    margin-top: var(--space-2);
    color: var(--text-primary);
  }
  .a-sel :global(.ident) {
    margin-right: var(--space-1);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .a-specs {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .a-lat {
    margin-top: var(--space-1);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .a-lat-v {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .a-compare[open] .arches {
    margin-top: var(--space-3);
  }
  .advanced {
    margin-top: var(--space-4);
    border-top: 1px solid var(--line-subtle);
    padding-top: var(--space-3);
  }
  .advanced summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: var(--control-md);
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
    cursor: pointer;
    list-style: none;
    border-radius: var(--radius-sm);
  }
  .advanced summary::-webkit-details-marker {
    display: none;
  }
  .advanced summary :global(svg) {
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .advanced[open] summary :global(svg) {
    transform: rotate(90deg);
  }
  .advanced summary:hover {
    color: var(--text-primary);
  }
  .advanced summary .faint {
    font-weight: var(--weight-regular);
  }
  @media (pointer: coarse) {
    .advanced summary {
      height: 44px;
    }
  }
  @media (max-width: 767px) {
    /* Model rows: the note gives way (it sits under the rows) so params and compute keep their columns; latency and
       its source are the name's meta line ("5.90 ms measured"), so every row is the same two lines, never a figure
       wrapping its source word in some rows and not others (round 26). */
    .arches {
      grid-template-columns: 28px minmax(0, 1fr) repeat(2, 52px) 28px;
      column-gap: var(--space-2);
    }
    .a-head-lat {
      display: none;
    }
    .a-head-lead {
      grid-column: 1 / 3;
    }
    .arch {
      min-height: 56px;
      grid-template-rows: auto auto;
      row-gap: 0;
      padding-block: 6px;
    }
    .a-size {
      grid-column: 1;
      grid-row: 1 / 3;
    }
    .arch :global(.a-name) {
      grid-column: 2;
      grid-row: 1;
    }
    .a-params {
      grid-column: 3;
      grid-row: 1 / 3;
    }
    .a-flops {
      grid-column: 4;
      grid-row: 1 / 3;
    }
    .a-v.lat {
      grid-column: 2;
      grid-row: 2;
      text-align: left;
    }
    .tick {
      grid-column: 5;
      grid-row: 1 / 3;
    }
    .a-src {
      min-width: 0;
      margin-left: var(--space-1);
    }
    .a-note {
      display: none;
    }
    .row2 {
      grid-template-columns: minmax(0, 1fr);
    }
  }
`;
