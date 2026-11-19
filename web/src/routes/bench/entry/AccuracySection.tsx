/**
 * Accuracy on the unchanged images (docs/BENCH.md, Accuracy): mAP@0.5:0.95 with its 95% interval drawn on the whole
 * 0–1 scale (honest about where it sits) and read in a line; mAP@0.5 and @0.75; precision, recall and F1 at the
 * best-F1 confidence; mAP by object size (a size the benchmark has none of says so); and AP per class as a sorted bar
 * list with how many true boxes each class has, a class the model does not know marked as such.
 */
import ChevronDown from '$icons/chevron-down';
import { computed, css, For, signal } from 'fw';
import type { Bench, BenchEntry, BenchResult } from '$lib/api/types';
import { unitTop } from '$lib/charts/domain';
import { atBestF1, count, metric, num } from '$lib/format';
import Panel from '$lib/ui/Panel';
import { bestF1Conf, intervalReading, intervalText } from '../words';

export interface AccuracySectionProps {
  result: BenchResult;
  entry: BenchEntry;
  bench: Bench;
}

const SHOWN = 10;
const TICKS = [0, 0.25, 0.5, 0.75, 1];
const at = (v: number) => `${(Math.max(0, Math.min(1, v)) * 100).toFixed(2)}%`;

export default function AccuracySection(props: AccuracySectionProps) {
  const a = computed(() => props.result.accuracy);
  const missing = computed(() => new Set(props.entry.missing_classes.map((c) => c.toLowerCase())));
  /** Classes with true boxes, best AP first; the ones it does not know last. */
  const classes = computed(() => {
    const names = props.bench.classes;
    const miss = missing.get();
    return a
      .get()
      .per_class.map((c) => ({ ...c, name: names[c.class] ?? `class ${c.class}`, unknown: miss.has((names[c.class] ?? '').toLowerCase()) }))
      .sort((x, y) => Number(x.unknown) - Number(y.unknown) || y.map50_95 - x.map50_95 || x.name.localeCompare(y.name));
  });
  /** Bars on 0–1 while any class reaches 0.5, else scaled just over the best (the table rule, DESIGN_DIRECTION §2.7). */
  const barTop = computed(() => {
    const best = Math.max(0, ...a.get().per_class.map((c) => c.map50_95));
    return best >= 0.5 ? 1 : unitTop(best);
  });
  const all = signal(false);
  const visible = computed(() => (all.get() ? classes.get() : classes.get().slice(0, SHOWN)));
  const sizes = computed(() => [
    { name: 'Small', range: 'under 32² px', v: a.get().map_small },
    { name: 'Medium', range: '32²–96² px', v: a.get().map_medium },
    { name: 'Large', range: 'over 96² px', v: a.get().map_large },
  ]);

  return (
    <Panel title="Accuracy" meta="on the unchanged images" id="accuracy">
      <div class="acc">
        <div class="left">
          <div class="head">
            <p class="hv">
              <span class="value num">{metric(a.get().map50_95)}</span>
              <span class="unit">mAP50-95</span>
            </p>
            <p class="ci num">95% interval {intervalText(a.get().ci95)}</p>
          </div>
          {/* The interval on the whole 0–1 scale: where it sits, and how wide it is for this many images. */}
          <div
            class="strip"
            role="img"
            aria-label={`mAP50-95 ${metric(a.get().map50_95)} with its 95% interval ${intervalText(a.get().ci95)}, on a scale from 0 to 1`}
          >
            <span class="track"></span>
            <span class="iv" style:left={at(a.get().ci95[0])} style:width={`calc(${at(a.get().ci95[1])} - ${at(a.get().ci95[0])})`}></span>
            <span class="pt" style:left={at(a.get().map50_95)}></span>
            <For each={TICKS} key={(t) => t}>
              {(t) => (
                <span class="t num" style:left={at(t())}>
                  {t() === 0 || t() === 1 ? t() : t().toFixed(2)}
                </span>
              )}
            </For>
          </div>
          <p class="reading">{intervalReading(a.get().ci95, props.entry.images)}</p>

          <dl class="kv">
            <div>
              <dt>mAP50</dt>
              <dd class="num">{metric(a.get().map50)}</dd>
            </div>
            <div>
              <dt>mAP75</dt>
              <dd class="num">{metric(a.get().map75)}</dd>
            </div>
            <div>
              <dt>Precision</dt>
              <dd class="num">{metric(a.get().precision)}</dd>
            </div>
            <div>
              <dt>Recall</dt>
              <dd class="num">{metric(a.get().recall)}</dd>
            </div>
            <div>
              <dt>F1</dt>
              <dd class="num">{metric(a.get().f1)}</dd>
            </div>
          </dl>
          <p class="cap">
            {!props.entry.scored
              ? 'Precision, recall and F1 of all its boxes (it gives no confidences), class means.'
              : bestF1Conf(a.get()) === null
                ? 'No confidence gives a right detection: precision, recall and F1 are 0 at every one, so there is no best F1.'
                : `Precision, recall and F1 ${atBestF1(a.get().best_conf)}, class means.`}
          </p>

          <h3 class="group-label sub-h">By object size</h3>
          <ul class="sizes" aria-label="mAP50-95 by object size">
            <For each={sizes.get()} key={(s) => s.name}>
              {(s) => (
                <li>
                  <span class="s-name">
                    {s().name}
                    <span class="s-range">{s().range}</span>
                  </span>
                  {s().v === null ? (
                    // Measured on the images it ran: an entry on a part of them may miss a size the others have.
                    <span class="s-none">
                      {props.entry.images < props.bench.images ? `None in the ${count(props.entry.images, 'image')} it ran` : 'None in this benchmark'}
                    </span>
                  ) : (
                    <>
                      <span class="bar" aria-hidden="true">
                        <span class="fill" style:width={at(s().v!)}></span>
                      </span>
                      <span class="s-v num">{metric(s().v!)}</span>
                    </>
                  )}
                </li>
              )}
            </For>
          </ul>
        </div>

        <div class="right">
          <div class="pc-head">
            <h3 class="group-label">AP50-95 per class</h3>
            <span class="pc-meta num">{barTop.get() < 1 ? `bars 0–${barTop.get()}` : 'bars 0–1'} · true boxes</span>
          </div>
          <ul class="pc" aria-label="AP50-95 per class, highest first">
            <For each={visible.get()} key={(c) => c.class}>
              {(c) => (
                <li class:unknown={c().unknown}>
                  <span class="c-name truncate" title={c().name}>
                    {c().name}
                  </span>
                  {c().unknown ? (
                    <span class="c-unk">Does not know this class</span>
                  ) : (
                    <>
                      <span class="bar" aria-hidden="true">
                        <span class="fill" style:width={`${(Math.min(1, c().map50_95 / barTop.get()) * 100).toFixed(2)}%`}></span>
                      </span>
                      <span class="c-v num">{metric(c().map50_95)}</span>
                    </>
                  )}
                  <span class="c-n num" title={`True boxes of ${c().name}`}>
                    {num(c().truths)}
                  </span>
                </li>
              )}
            </For>
          </ul>
          {classes.get().length > SHOWN ? (
            <button type="button" class="more" aria-expanded={all.get()} onclick={() => all.set(!all.get())}>
              {all.get() ? `Show the first ${SHOWN}` : `Show all ${classes.get().length} classes`}
              <ChevronDown size={14} aria-hidden="true" />
            </button>
          ) : null}
          {props.bench.classes.length > a.get().per_class.length ? (
            <p class="cap">
              {props.bench.classes.length - a.get().per_class.length} of the benchmark’s {props.bench.classes.length} classes have no true box in the
              images it ran, so no AP.
            </p>
          ) : null}
        </div>
      </div>
    </Panel>
  );
}

css`
  .acc {
    display: grid;
    grid-template-columns: minmax(0, 5fr) minmax(0, 7fr);
    gap: var(--space-6) var(--space-8);
    align-items: start;
  }
  .left,
  .right {
    display: grid;
    gap: var(--space-3);
    min-width: 0;
    align-content: start;
  }
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-1) var(--space-4);
  }
  .hv {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
  }
  /* A KPI value (24/600), proportional figures as a standalone number. */
  .value {
    font-size: var(--type-2xl-size);
    line-height: var(--type-2xl-line);
    letter-spacing: var(--type-2xl-tracking);
    font-weight: var(--weight-semibold);
    font-variant-numeric: proportional-nums;
  }
  .unit {
    color: var(--text-secondary);
  }
  .ci {
    color: var(--text-secondary);
  }
  /* The interval strip: a hairline track over 0–1, the interval as a 6 px ink band, the value a dot on it, ticks under. */
  .strip {
    position: relative;
    height: 32px;
    margin: 0 var(--space-2);
  }
  .track {
    position: absolute;
    left: 0;
    right: 0;
    top: 8px;
    height: 1px;
    background: var(--chart-axis);
  }
  .iv {
    position: absolute;
    top: 4px;
    height: 8px;
    min-width: 2px;
    border-radius: 3px;
    background: var(--text-tertiary);
  }
  .pt {
    position: absolute;
    top: 2px;
    width: 12px;
    height: 12px;
    margin-left: -6px;
    border-radius: 50%;
    background: var(--text-primary);
    box-shadow: 0 0 0 2px var(--bg-panel);
  }
  .t {
    position: absolute;
    top: 16px;
    transform: translateX(-50%);
    color: var(--chart-label);
    font-size: var(--type-xs-size); /* small: chart tick */
  }
  .reading,
  .cap {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    max-inline-size: var(--measure-dialog);
  }
  .cap {
    color: var(--text-tertiary);
  }
  .kv {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: var(--space-3);
    margin-top: var(--space-2);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
  }
  .kv div {
    display: grid;
    gap: 2px;
  }
  .kv dt {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .kv dd {
    font-size: var(--type-lg-size);
    line-height: var(--type-lg-line);
    font-weight: var(--weight-semibold);
  }
  .sub-h {
    margin-top: var(--space-3);
  }
  .sizes li,
  .pc li {
    display: grid;
    align-items: center;
    column-gap: var(--space-3);
    min-height: 28px;
  }
  .sizes li {
    grid-template-columns: 168px minmax(0, 1fr) 48px;
  }
  .s-name {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
  }
  .s-range {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .s-none {
    grid-column: 2 / span 2;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .bar {
    height: 6px;
    border-radius: 3px;
    background: var(--bg-active);
    overflow: hidden;
  }
  .fill {
    display: block;
    height: 100%;
    background: var(--text-secondary);
  }
  .s-v,
  .c-v,
  .c-n {
    text-align: right;
  }
  .pc-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .pc-meta {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .pc li {
    grid-template-columns: minmax(96px, 200px) minmax(0, 1fr) 48px 56px;
    border-top: 1px solid var(--line-subtle);
  }
  .c-n {
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .c-unk {
    grid-column: 2 / span 2;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .pc li.unknown .c-name {
    color: var(--text-secondary);
  }
  .more {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    justify-self: start;
    min-height: 32px;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .more[aria-expanded='true'] :global(svg) {
    transform: rotate(180deg);
  }
  @media (hover: hover) {
    .more:hover {
      color: var(--text-primary);
    }
  }
  @media (pointer: coarse) {
    .more {
      min-height: 44px;
    }
  }
  @media (max-width: 1023px) {
    .acc {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  @media (max-width: 767px) {
    .kv {
      gap: var(--space-2);
    }
    .kv dd {
      font-size: var(--type-md-size);
      line-height: var(--type-md-line);
    }
    .sizes li {
      grid-template-columns: minmax(0, 1fr) 96px 48px;
    }
    .s-range {
      display: none;
    }
    .pc li {
      grid-template-columns: minmax(0, 1fr) 72px 48px 40px;
    }
  }
`;
