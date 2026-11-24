/**
 * The legend: one card per run (its key, name and value under the cursor, how its line ended and how it started),
 * each a toggle that keeps its run's line in front; on phones, two columns of compact keys and an Edit button.
 */
import X from '$icons/x';
import ArrowUpRight from '$icons/arrow-up-right';
import RotateCw from '$icons/rotate-cw';
import { bindable, css, For } from 'fw';
import type { Run } from '$lib/api/types';
import { runEnd } from '$lib/charts/runAxis';
import { metric } from '$lib/format';
import { FROM_SCRATCH } from '$lib/lineage';
import { href } from '$lib/router';
import Ident from '$lib/ui/Ident';

export interface RunChipsProps {
  hint: string | null;
  readoutAt: string;
  managing?: boolean;
  chosen: Run[];
  valueOf: (id: string) => number | null;
  startOf: (r: Run) => string | null;
  pinned?: string | null;
  hovered?: string | null;
  fill: (id: string) => string;
  SLOT_PATTERN: readonly (undefined | 'dash' | 'dot')[];
  slotOf: (id: string) => number;
  ids: string[];
  setIds: (next: string[]) => void;
  narrow: boolean;
  MAX: number;
  candidates: Run[];
  /** Add run…, as the page draws it. */
  adder: JSX.Element;
  /** Per-run metrics error, when this run's curve could not load or refresh. */
  metricErrorOf?: (id: string) => unknown;
  /** Whether this run's metrics are currently retrying. */
  retryingOf?: (id: string) => boolean;
  /** Retry loading this run's metrics. */
  onRetryMetric?: (id: string) => void;
}

export default function RunChips(props: RunChipsProps) {
  const managing = bindable(props, 'managing', false);
  const pinned = bindable(props, 'pinned', null);
  const hovered = bindable(props, 'hovered', null);

  const startLine = (r: Run) => {
    const from = props.startOf(r);
    return from ? `Fine-tuned from ${from}` : FROM_SCRATCH;
  };

  // The run chips are the legend: colour key, value under the cursor, link and remove.
  return (
    <div class="chips" role="group" aria-label="Runs being compared">
      <div class="chips-head">
        {/* The readout row: where the values are read, or how to read them while the plot has keyboard focus or on a
            first touch. */}
        <span class="at num" class:hint={!!props.hint} aria-hidden="true">
          {props.hint ?? props.readoutAt}
        </span>{' '}
        {/* Phones: opening, removing and adding runs sit in a sheet, so the legend stays two rows of keys. */}
        <button type="button" class="manage" aria-haspopup="dialog" aria-expanded={managing.get()} onclick={() => managing.set(true)}>
          Edit
        </button>
      </div>
      <For each={props.chosen} key={(r) => r.id}>
        {(r) => {
          const v = () => props.valueOf(r().id);
          const from = () => props.startOf(r());
          const end = () => runEnd(r().status);
          const failed = () => props.metricErrorOf?.(r().id) !== undefined;
          const retrying = () => !!props.retryingOf?.(r().id);
          return (
            <span
              class="chip"
              class:pinned={pinned.get() === r().id}
              style={{ '--c': props.fill(r().id) }}
              onpointerenter={() => hovered.set(r().id)}
              onpointerleave={() => hovered.set(null)}
              onfocusin={() => hovered.set(r().id)}
              onfocusout={() => hovered.set(null)}
            >
              {/* The legend entry (key and name) is the toggle: it keeps the run's line in front. */}
              <button
                type="button"
                class="key"
                aria-pressed={pinned.get() === r().id}
                title={pinned.get() === r().id ? 'Let the line fall back' : 'Keep this line in front'}
                onclick={() => pinned.set(pinned.peek() === r().id ? null : r().id)}
              >
                <i aria-hidden="true" data-pattern={props.SLOT_PATTERN[props.slotOf(r().id)]}></i>
                <span class="nm">{r().name}</span>
              </button>{' '}
              {/* How the line ended (live, stopped, failed) in words beside its value. */}
              <span class="v num">
                {v() === null ? '—' : metric(v())}
                {failed() ? (
                  <span class="st">
                    <span class="sep">{' · '}</span>
                    metrics failed
                  </span>
                ) : end() ? (
                  <span class="st">
                    <span class="sep">{' · '}</span>
                    {end()}
                  </span>
                ) : null}
              </span>{' '}
              {/* How the run started: fine-tuned vs from scratch often explains a gap between two curves. */}
              <span class="from" class:tuned={!!from()} title={startLine(r())} data-fact>
                {from() ? (
                  <>
                    From <Ident value={from()!} class="src" />
                  </>
                ) : (
                  FROM_SCRATCH
                )}
              </span>{' '}
              <span class="acts">
                <a class="open" href={href(`/runs/${r().id}`)} aria-label={`Open ${r().name}`} title="Open run">
                  <ArrowUpRight size={12} />
                </a>{' '}
                {failed() && props.onRetryMetric ? (
                  <>
                    <button
                      type="button"
                      class="rm"
                      aria-label={`Retry metrics for ${r().name}`}
                      title="Retry metrics"
                      disabled={retrying()}
                      onclick={() => props.onRetryMetric?.(r().id)}
                    >
                      <RotateCw size={12} />
                    </button>{' '}
                  </>
                ) : null}
                <button class="rm" aria-label={`Remove ${r().name}`} onclick={() => props.setIds(props.ids.filter((id) => id !== r().id))}>
                  <X size={12} />
                </button>
              </span>
            </span>
          );
        }}
      </For>
      {!props.narrow && props.ids.length < props.MAX && props.candidates.length ? props.adder : null}
    </div>
  );
}

css`
  /* The legend: one card per run (key, name, value under the cursor; how it started below), side by side. */
  .chips {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(232px, 1fr));
    gap: var(--space-2);
    align-items: stretch;
  }
  /* Laptops and tablets: a 2 × 2 grid, so four runs never leave one card orphaned on a row of its own. */
  @media (min-width: 768px) and (max-width: 1279px) {
    .chips {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
  .chips-head {
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .at {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .chip {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto auto;
    grid-template-rows: auto auto;
    column-gap: var(--space-1);
    min-width: 0;
    padding: var(--space-1) var(--space-1) var(--space-1) var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-panel);
    border: 1px solid var(--line-default);
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      background-color var(--dur-fast) var(--ease-standard);
  }
  .chip:hover {
    border-color: var(--line-strong);
  }
  /* The run kept in front: the one selection rule (bg.active and the 2 px ink bar). */
  .chip.pinned {
    background: var(--bg-active);
  }
  .chip.pinned::before {
    content: '';
    position: absolute;
    left: -1px;
    top: -1px;
    bottom: -1px;
    width: 2px;
    border-radius: var(--radius-md) 0 0 var(--radius-md);
    background: var(--text-primary);
  }
  /* The legend entry is the toggle: key and name, over both lines of the card. */
  .key {
    grid-column: 1;
    grid-row: 1 / 3;
    position: relative;
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
    padding: var(--space-1);
    margin-left: calc(-1 * var(--space-1));
    border-radius: var(--radius-sm);
    min-width: 0;
    text-align: left;
  }
  .key:hover {
    background: var(--bg-hover);
  }
  /* On the name's centre line (20 px line). */
  .key i {
    margin-top: 8px;
  }
  .nm {
    font-weight: var(--weight-medium);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .chip .v {
    grid-column: 2;
    grid-row: 1;
    align-self: start;
    padding: var(--space-1) var(--space-1) 0 0;
    min-width: 3.2em;
    color: var(--text-primary);
    font-weight: var(--weight-medium);
    font-variant-numeric: tabular-nums;
    text-align: right;
    white-space: nowrap;
  }
  .chip .st {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  /* Indented under the name, past the key (16 px swatch + 8 px gap); clicks fall through to the toggle. */
  .from {
    grid-column: 1 / 3;
    grid-row: 2;
    align-self: end;
    position: relative;
    min-width: 0;
    padding: 0 var(--space-1) var(--space-1) var(--space-6);
    pointer-events: none;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .from.tuned {
    color: var(--text-secondary);
  }
  /* Breaks only after a hyphen (fasteners-n-/best), never mid-word and never with an ellipsis. */
  .from :global(.ident) {
    font-size: var(--type-sm-size);
    overflow-wrap: normal;
    word-break: normal;
  }
  .acts {
    grid-column: 3;
    grid-row: 1 / 3;
    display: grid;
    align-content: center;
  }
  .open,
  .chip .rm {
    display: grid;
    place-items: center;
    width: 24px;
    height: 24px;
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
  }
  .open:hover,
  .chip .rm:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  @media (pointer: coarse) {
    .chip .rm {
      position: relative;
    }
    .chip .rm::after {
      content: '';
      position: absolute;
      inset: -8px;
    }
  }
  .manage {
    display: none;
  }
  .at.hint {
    color: var(--text-secondary);
  }
  @media (max-width: 767px) {
    /* Phones: the legend is two columns of compact keys above the plot, each one line of text that wraps only where it
       must: the swatch, the run's whole name (wrapping at its hyphens, never cut: "shapes-hard-s" beside
       "shapes-hard-m"), its value right after it, and how the line ended ("failed") under the name, so name and value
       read together and four runs take two short rows. The whole key is the toggle that keeps a line in front; its hit
       area is the card. Opening, removing and adding runs live in the Edit sheet. */
    .chips {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0 var(--space-2);
    }
    /* The legend's caption ("Latest · ep 37", or how to read it on a first touch) and Edit: the first line of the
       legend, not a row of its own. Edit is a real 32 px target (44 on touch), never a hit area over the controls above. */
    .chips-head {
      min-height: 32px;
    }
    .manage {
      display: inline-flex;
      align-items: center;
      height: 32px;
      padding: 0 var(--space-2);
      margin-right: calc(-1 * var(--space-2));
      border-radius: var(--radius-sm);
      font-size: var(--type-sm-size);
      color: var(--text-secondary);
    }
    .manage:hover {
      color: var(--text-primary);
    }
    /* A hanging indent: a wrapped line starts under the name, past the swatch (12 px and its 6 px gap). */
    .chips .chip {
      display: block;
      min-height: 32px;
      padding: var(--space-1-5) var(--space-2) var(--space-1-5) calc(var(--space-2) + 18px);
      text-indent: -18px;
      border: 0;
      background: none;
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
    }
    .chips .chip.pinned {
      background: var(--bg-active);
    }
    .chips .chip.pinned::before {
      top: 0;
      bottom: 0;
      left: 0;
    }
    /* The space before the value is the name's margin, so a value wrapped under the name starts on its left edge. */
    .chips .chip .key {
      display: inline;
      position: static;
      margin: 0 var(--space-1) 0 0;
      padding: 0;
      font: inherit;
    }
    .chips .chip .key:hover {
      background: none;
    }
    .chips .chip .key::after {
      content: '';
      position: absolute;
      inset: 0;
      border-radius: var(--radius-md);
    }
    .chips .chip .key:focus-visible {
      outline: none;
    }
    .chips .chip .key:focus-visible::after {
      outline: 2px solid var(--focus);
      outline-offset: -2px;
    }
    /* On the text's centre line (16 px line). */
    .chips .chip .key i {
      display: inline-block;
      width: 12px;
      margin: 0 6px 0 0;
      vertical-align: middle;
    }
    .chips .nm {
      white-space: normal;
      overflow: visible;
      overflow-wrap: break-word;
    }
    /* The value and how the line ended ("0.494 · failed") are one unit that never breaks: beside the name when they
       fit, else together on the line under it, never the status alone under the name with its value above. */
    .chips .chip .v {
      display: inline-block;
      min-width: 0;
      margin: 0;
      padding: 0;
      text-align: left;
      text-indent: 0;
      white-space: nowrap;
      pointer-events: none;
    }
    .chips .chip .from,
    .chips .chip .acts {
      display: none;
    }
  }
  /* Touch: Edit is a real 44 px target (no invisible hit area over the controls above or the keys below). */
  @media (max-width: 767px) and (pointer: coarse) {
    .chips-head,
    .manage {
      min-height: 44px;
    }
  }
  /* Same patterns as the lines, so a run's key matches its curve without relying on colour. A key is a fill: the
     palette hue (--c is chart.fill) with the swatch hairline, which the pattern's mask cuts along with the dashes. */
  .chip i {
    flex-shrink: 0;
    width: 16px;
    height: 4px;
    border-radius: 2px;
    background: var(--c);
    box-shadow: inset 0 0 0 1px var(--chart-swatch-edge);
  }
  i[data-pattern='dash'] {
    mask-image: repeating-linear-gradient(90deg, #000 0 5px, transparent 5px 7px);
    border-radius: 0;
  }
  i[data-pattern='dot'] {
    mask-image: repeating-linear-gradient(90deg, #000 0 3px, transparent 3px 5px);
    border-radius: 0;
  }
`;
