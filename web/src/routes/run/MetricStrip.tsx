/**
 * The top of the run's metrics: the epoch as a full-width row (count, progress bar with the hatched no-mosaic end, and
 * its caption) over the KPI tabs, each with its current value; the selected tab names the chart under them.
 */
import { bindable, css, For } from 'fw';
import { tooltip } from '$lib/ui/tooltip';
import type { Run } from '$lib/api/types';
import { progressText } from '$lib/api/progress';
import { duration, eta } from '$lib/format';
import Meta from '$lib/ui/Meta';

export type Tab = 'map' | 'map50' | 'pr' | 'loss' | 'lr' | 'ips';
export interface TabDef {
  id: Tab;
  label: string;
  value: string;
  /** Set one step smaller after the value; plain text (the phone picker's option reads it as is). */
  unit?: string;
  /** A power of ten's exponent ("−4"): the tab draws the unit "× 10" with a real superscript instead. */
  exp?: string;
  caption: string;
  /** Tablets (768–1023 px): a shorter label and caption, so all six tabs fit without clipping. */
  short?: string;
  capShort?: string;
  title: string;
  qualifier: string;
  /** The qualifier's power-of-ten exponent, drawn as a superscript after it. */
  qExp?: string;
}

export interface MetricStripProps {
  run: Run;
  live: boolean;
  progress: number;
  mosaicOff: boolean;
  closeFrom: number;
  closeId: string;
  closeText: { cap: string; full: string };
  elapsed: number | null;
  tabs: TabDef[];
  tab?: Tab;
  /** A tab's unit, as the chart's readout draws it. */
  unitOf: (t: () => TabDef) => JSX.Element;
}

export default function MetricStrip(props: MetricStripProps) {
  const tab = bindable(props, 'tab');

  function onkeydown(e: KeyboardEvent) {
    const tabs = props.tabs;
    const i = tabs.findIndex((t) => t.id === tab.peek());
    let j = i;
    if (e.key === 'ArrowRight') j = (i + 1) % tabs.length;
    else if (e.key === 'ArrowLeft') j = (i - 1 + tabs.length) % tabs.length;
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = tabs.length - 1;
    else return;
    e.preventDefault();
    tab.set(tabs[j]!.id);
    (e.currentTarget as HTMLElement).querySelector<HTMLElement>(`[data-tab="${tab.peek()}"]`)?.focus();
  }

  const completed = () => props.run.status === 'completed';

  // Phones: a finished run's strip only repeats the completed banner above it (status, duration), so it goes.
  // One structure at every width: the epoch as a full-width row (count, bar, caption) over the KPI tabs.
  return (
    <div class="strip" class:finished={completed()}>
      <div class="epoch">
        <span class="sr-only">Epoch</span>
        <span class="big num">
          {props.run.epoch}
          <span class="of">/ {props.run.epochs}</span>
        </span>
        <div class="bar" class:has-close={props.mosaicOff}>
          <div
            class="track"
            role="progressbar"
            aria-label="Training progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={completed() ? 100 : Math.floor(props.progress * 100 + 1e-9)}
            aria-describedby={props.mosaicOff ? props.closeId : undefined}
          >
            <div
              class="fill"
              class:done={completed()}
              class:bad={props.run.status === 'failed'}
              style:transform={`scaleX(${completed() ? 1 : props.progress})`}
            ></div>
            {/* Over the filled part of the bar (a finished run) the hatch is cut out of the fill in the panel colour. */}
            {props.mosaicOff ? (
              <div class="close" class:over={completed() || props.progress >= props.closeFrom} style:left={`${props.closeFrom * 100}%`}></div>
            ) : null}
          </div>
          {/* The hatched end says what it is: a caption under it, the same words on hover (a hit area taller than the
              4 px bar) and as the bar's accessible description, with the trainer's capped count. */}
          {props.mosaicOff ? (
            <>
              <span class="close-hit" aria-hidden="true" style:left={`${props.closeFrom * 100}%`} use:tooltip={props.closeText.full}></span>
              <span class="close-cap" aria-hidden="true" use:tooltip={props.closeText.full}>
                {props.closeText.cap}
              </span>
              <span class="sr-only" id={props.closeId}>
                {props.closeText.full}
              </span>
            </>
          ) : null}
        </div>
        <span class="cap num">
          {props.run.status === 'queued' ? (
            'Not started'
          ) : props.live ? (
            <Meta line>
              <span>{progressText(props.run)}</span>
              <span>{props.run.eta_s !== null ? `${eta(props.run.eta_s)} left` : 'estimating…'}</span>
            </Meta>
          ) : props.elapsed !== null ? (
            `${duration(props.elapsed)} total`
          ) : (
            '—'
          )}
        </span>
      </div>
      {/* Queued: the tabs read "—" until the first step, so the frame is the one a running run fills. */}
      <div class="tabs" role="tablist" aria-label="Metric to chart" data-scrolls onkeydown={onkeydown}>
        <For each={props.tabs} key={(t) => t.id}>
          {(t) => (
            <button
              role="tab"
              class="tab"
              class:on={tab.get() === t().id}
              aria-selected={tab.get() === t().id}
              aria-controls="run-chart"
              tabindex={tab.get() === t().id ? 0 : -1}
              data-tab={t().id}
              onclick={() => tab.set(t().id)}
            >
              <span class="label">
                <span class="full">{t().label}</span>
                <span class="short" aria-hidden="true">
                  {t().short ?? t().label}
                </span>
              </span>
              <span class="val num">
                {t().value}
                {t().unit && t().value !== '—' ? props.unitOf(t) : null}
              </span>
              <span class="cap num">
                <span class="full">{t().caption}</span>
                <span class="short" aria-hidden="true">
                  {t().capShort ?? t().caption}
                </span>
              </span>
            </button>
          )}
        </For>
      </div>
    </div>
  );
}

css`
  .strip {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    border-bottom: 1px solid var(--line-subtle);
  }
  /* The epoch: one full-width row over the tabs at every width, "38 / 100", the bar, and its caption. */
  .epoch {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    align-items: center;
    column-gap: var(--space-4);
    padding: var(--space-3) var(--space-4);
    border-bottom: 1px solid var(--line-subtle);
  }
  /* Each tab is at least as wide as its value and caption, so no value is ever cut off; spare width is shared. Every
     tier is sized to fit all six; a sideways scroll is only the fallback for values far wider than real ones. */
  .tabs {
    display: grid;
    grid-template-columns: repeat(6, minmax(max-content, 1fr));
    min-width: 0;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .tab {
    position: relative;
    display: grid;
    grid-template-rows: 16px 40px 16px;
    gap: 6px;
    align-content: start;
    padding: var(--space-4);
    text-align: left;
    min-width: 0;
    border-left: 1px solid var(--line-subtle);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .tab:first-child {
    border-left: 0;
  }
  .tab:hover {
    background: var(--bg-hover);
  }
  /* The selected metric: an ink bar on the edge it shares with its chart (inside the tab, which the row may clip). */
  .tab::after {
    content: '';
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 2px;
    background: var(--text-primary);
    transform: scaleX(0);
    transition: transform var(--dur-base) var(--ease-out);
  }
  .tab.on::after {
    transform: scaleX(1);
  }
  .tab:focus-visible {
    outline: 2px solid var(--focus);
    outline-offset: -2px;
  }
  .label {
    font-size: var(--type-sm-size);
    line-height: 16px;
    color: var(--text-tertiary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .tab.on .label {
    color: var(--text-primary);
  }
  .big {
    font-size: var(--type-2xl-size);
    line-height: 32px;
    letter-spacing: var(--type-2xl-tracking);
    font-weight: var(--weight-semibold);
  }
  .of {
    margin-left: 6px;
    font-size: var(--type-lg-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    letter-spacing: 0;
  }
  /* A fixed line box: the value's baseline sits at the same height in every tab, with or without a unit, an exponent or
     a fallback glyph (the unit's own line-height is 1, so a 12 px span can never deepen the line). */
  .val {
    align-self: end;
    height: 32px;
    overflow: visible;
    font-size: var(--type-2xl-size);
    line-height: 32px;
    letter-spacing: var(--type-2xl-tracking);
    font-weight: var(--weight-semibold);
    color: var(--text-secondary);
    white-space: nowrap;
  }
  .tab.on .val {
    color: var(--text-primary);
  }
  /* A tab's caption holds figures people read ("best 0.604 · ep 150", "warmup 360 steps · linear"): secondary, not
     the tertiary of labels people skim (DESIGN_DIRECTION §2.3). */
  .cap {
    font-size: var(--type-sm-size);
    line-height: 16px;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .track {
    position: relative;
    height: 4px;
    border-radius: 2px;
    background: var(--bg-active);
    overflow: hidden;
  }
  .fill {
    position: absolute;
    inset: 0;
    transform-origin: left;
    background: var(--accent-line);
    border-radius: 2px;
    transition: transform var(--dur-slow) var(--ease-standard);
  }
  .fill.done {
    background: var(--text-secondary);
  }
  .fill.bad {
    background: var(--danger-solid);
  }
  .bar {
    position: relative;
    align-self: center;
    min-width: 0;
  }
  .close-hit {
    position: absolute;
    top: -12px;
    bottom: -12px;
    right: 0;
  }
  /* The hatch's caption ("Mosaic off after epoch 90") under it, right-aligned to the bar's end: in the epoch row's spare
     height from the tablet up, in a line the bar keeps for it on phones. */
  .close-cap {
    position: absolute;
    top: calc(100% + var(--space-1));
    right: 0;
    font-size: var(--type-sm-size);
    line-height: 16px;
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .close {
    position: absolute;
    top: 0;
    bottom: 0;
    right: 0;
    background: repeating-linear-gradient(135deg, transparent 0 3px, var(--line-strong) 3px 4px);
    border-left: 1px solid var(--line-strong);
  }
  .close.over {
    background: repeating-linear-gradient(135deg, transparent 0 3px, var(--bg-panel) 3px 4.5px);
    border-left-color: var(--bg-panel);
  }
  /* Laptops: the same six tabs in one row, a step more compact (12 px sides, 20 px values), so all six fit from
     1024 px up beside the collapsed sidebar. */
  @media (max-width: 1279px) {
    .tab {
      grid-template-rows: 16px 32px 16px;
      gap: var(--space-1);
      padding: var(--space-3);
    }
    .val {
      font-size: var(--type-xl-size);
      line-height: 32px;
      letter-spacing: var(--type-xl-tracking);
    }
  }
  /* Tablets: six tabs share about 700 px, so each says it in fewer words ("P · R", "ep 37"), the full words their
     accessible name; nothing clips. */
  .short {
    display: none;
  }
  @media (min-width: 768px) and (max-width: 1023px) {
    .tab .full {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip-path: inset(50%);
      white-space: nowrap;
    }
    .tab .short {
      display: inline;
    }
  }
  @media (max-width: 767px) {
    /* Phones: six tiles would push the chart down: one compact metric picker in the chart header instead, in place of
       the title (the qualifier stays beside it). */
    .tabs {
      display: none;
    }
    .epoch {
      border-bottom: 0;
    }
    /* "23 / 100" and its meta on one baseline, the bar under the whole row 8 px below; the chart follows. */
    .epoch {
      grid-template-columns: auto minmax(0, 1fr);
      align-items: baseline;
      row-gap: var(--space-2);
    }
    .big {
      grid-row: 1;
      grid-column: 1;
      align-self: baseline;
    }
    .epoch .cap {
      grid-row: 1;
      grid-column: 2;
      align-self: baseline;
      text-align: right;
    }
    .epoch .bar {
      grid-row: 2;
      grid-column: 1 / -1;
      align-self: center;
    }
    .epoch .bar.has-close {
      margin-bottom: var(--space-5);
    }
    .val {
      height: 28px;
      font-size: var(--type-lg-size);
      line-height: 28px;
    }
    .strip.finished {
      display: none;
    }
  }
`;
