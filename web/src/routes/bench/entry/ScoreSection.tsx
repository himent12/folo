/**
 * An entry's score made visible (docs/BENCH.md, The score): the total as the page's one hero number, the partial mark
 * with its reason, then the five axes in the score's order, one row each: what it asks, its value on 0–1 as a bar,
 * the number behind it in its own unit, its weight, and the points it adds (weight × value, over the measured
 * weights), so the rows add up to the total under them. An axis not measured says why in its place.
 */
import { computed, css, For } from 'fw';
import type { Bench, BenchEntry, BenchResult } from '$lib/api/types';
import { AXIS_ASKS, AXIS_NAME } from '$lib/bench/rules';
import { axisShares } from '$lib/bench/score';
import { num, pct } from '$lib/format';
import Panel from '$lib/ui/Panel';
import Skeleton from '$lib/ui/Skeleton';
import { tooltip } from '$lib/ui/tooltip';
import { axisPoints, axisRaw, axisWhyNot, partialWhy, roundToTotal, scoreText, sharesNote } from '../words';

export interface ScoreSectionProps {
  entry: BenchEntry;
  bench: Bench;
  result: BenchResult | undefined;
  rank: { at: number; of: number } | null;
}

const ordinal = (n: number) =>
  `${n}${n % 10 === 1 && n % 100 !== 11 ? 'st' : n % 10 === 2 && n % 100 !== 12 ? 'nd' : n % 10 === 3 && n % 100 !== 13 ? 'rd' : 'th'}`;

export default function ScoreSection(props: ScoreSectionProps) {
  const score = computed(() => props.entry.score!);
  const shares = computed(() => axisShares(score.get().axes));
  const points = computed(() => {
    const axes = score.get().axes;
    const p = axisPoints(axes);
    const shown = roundToTotal(
      axes.map((a) => p[a.axis]),
      Number(scoreText(score.get().total).replace(/,/g, '')),
    );
    return Object.fromEntries(axes.map((a, i) => [a.axis, shown[i] ?? null])) as Record<string, number | null>;
  });
  const note = computed(() => sharesNote(score.get().axes));

  return (
    <Panel title="Score" meta="0–100, the weighted mean of five axes" id="score">
      <div class="score">
        <div class="hero">
          <p class="total num">
            <span class="big">{scoreText(score.get().total)}</span>
            <span class="of">of 100</span>
          </p>
          {/* The total as the five axes' points side by side on 0–100: the rows beside it, added up. */}
          <span class="stack" aria-hidden="true">
            <For each={score.get().axes} key={(a) => a.axis}>
              {(a, i) => (
                <>
                  {points.get()[a().axis] ? (
                    <span
                      class="seg"
                      style={{ width: `${points.get()[a().axis]}%`, '--k': 1 - i() * 0.16 }}
                      use:tooltip={`${AXIS_NAME[a().axis]} +${num(points.get()[a().axis], 1)}`}
                    ></span>
                  ) : null}
                </>
              )}
            </For>
          </span>
          {props.rank ? (
            <p class="rank num">
              {ordinal(props.rank.at)} of {props.rank.of} on this benchmark
            </p>
          ) : null}
          {score.get().partial ? (
            <p class="partial">
              <span class="p-word">Partial</span> {partialWhy(props.entry, props.bench).replace(/^Partial: /, '')}
            </p>
          ) : (
            <p class="complete">Complete: every image, every condition of the plan, every axis.</p>
          )}
        </div>

        <div class="rows" role="table" aria-label="The score's axes">
          <div class="r head" role="row">
            <span role="columnheader">Axis</span>
            <span role="columnheader" class="c-bar">
              Value, 0–1
            </span>
            <span role="columnheader" class="c-raw">
              Measured as
            </span>
            <span role="columnheader" class="c-w">
              Weight
            </span>
            <span role="columnheader" class="c-pts">
              Points
            </span>
          </div>
          <For each={score.get().axes} key={(a) => a.axis}>
            {(a) => {
              const measured = () => a().value !== null;
              return (
                <div class="r" class:none={!measured()} role="row" data-axis={a().axis}>
                  <span class="c-name" role="cell">
                    <span class="nm">{AXIS_NAME[a().axis]}</span>
                    <span class="asks">{AXIS_ASKS[a().axis]}</span>
                  </span>
                  {measured() ? (
                    <>
                      <span class="c-bar" role="cell">
                        <span class="bar" aria-hidden="true">
                          <span class="fill" style:width={`${Math.max(0, Math.min(1, a().value!)) * 100}%`}></span>
                        </span>
                        <span class="v num">{num(a().value, 2)}</span>
                      </span>
                      <span class="c-raw" role="cell">
                        {props.result ? axisRaw(a().axis, props.result, props.entry.scored) : <Skeleton w="120px" h="12px" />}
                      </span>
                    </>
                  ) : (
                    <span class="c-why" role="cell">
                      <span class="nm-not">Not measured.</span> {axisWhyNot(a().axis, props.entry, props.bench)}
                    </span>
                  )}
                  <span class="c-w num" role="cell">
                    {/* Phones hide the header row: the figure says what it is. */}
                    <span class="w-word">weight </span>
                    {pct(measured() ? shares.get()[a().axis] : a().weight, 0)}
                    {measured() && Math.abs(shares.get()[a().axis] - a().weight) > 1e-6 ? (
                      <span class="base">{pct(a().weight, 0)} in the plan</span>
                    ) : null}
                  </span>
                  <span class="c-pts num" role="cell">
                    {points.get()[a().axis] !== null ? `+${num(points.get()[a().axis], 1)}` : <span class="faint">—</span>}
                  </span>
                </div>
              );
            }}
          </For>
          <div class="r sum" role="row">
            <span class="c-name" role="cell">
              <span class="nm">Score</span>
            </span>
            <span class="c-sumline" role="cell">
              {note.get() ?? 'The five rows add up to the score.'}
            </span>
            <span class="c-pts num" role="cell">
              {scoreText(score.get().total)}
            </span>
          </div>
        </div>
      </div>
    </Panel>
  );
}

css`
  /* The hero on the page grid's first 3 columns, the axes on the other 9; stacked under 1024 px. */
  .score {
    display: grid;
    grid-template-columns: minmax(0, 3fr) minmax(0, 9fr);
    gap: var(--space-4) var(--space-8);
    align-items: start;
  }
  .hero {
    display: grid;
    gap: var(--space-2);
    align-content: start;
  }
  .total {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
  }
  /* The page's one hero number (DESIGN_DIRECTION §2.3: at most one 32/600 per page), proportional figures. */
  .big {
    font-size: var(--type-3xl-size);
    line-height: var(--type-3xl-line);
    letter-spacing: var(--type-3xl-tracking);
    font-weight: var(--type-3xl-weight);
    font-variant-numeric: proportional-nums;
  }
  .of {
    color: var(--text-tertiary);
  }
  /* 0–100 as a track, each axis's points a segment in ink, darker to lighter in the rows' order, 2 px apart. */
  .stack {
    display: flex;
    gap: 2px;
    height: 8px;
    margin-bottom: var(--space-1);
    border-radius: 4px;
    background: var(--bg-active);
    overflow: hidden;
  }
  .seg {
    height: 100%;
    background: color-mix(in oklab, var(--text-primary) calc(var(--k) * 100%), var(--bg-panel));
  }
  .rank {
    color: var(--text-secondary);
  }
  .partial,
  .complete {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    max-inline-size: var(--measure);
  }
  .p-word {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  /* One template for every row: the axis and what it asks, its bar and value, the number behind it, its weight, its
     points; a not-measured row's reason spans the bar and the number. */
  .rows {
    display: grid;
    min-width: 0;
  }
  .r {
    display: grid;
    grid-template-columns: minmax(0, 1.25fr) minmax(120px, 1fr) minmax(0, 1.1fr) 88px 56px;
    align-items: center;
    column-gap: var(--space-4);
    min-height: 48px;
    padding: var(--space-2) 0;
    border-top: 1px solid var(--line-subtle);
  }
  .r.head {
    min-height: 32px;
    padding: 0;
    border-top: 0;
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-tertiary);
  }
  .r.head .c-w,
  .r.head .c-pts {
    text-align: right;
  }
  .c-name {
    display: grid;
    min-width: 0;
  }
  .nm {
    font-weight: var(--weight-medium);
  }
  .asks {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .r.none .nm {
    color: var(--text-secondary);
  }
  .c-bar {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 40px;
    align-items: center;
    gap: var(--space-3);
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
  .v {
    text-align: right;
  }
  .c-raw {
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .c-why {
    grid-column: 2 / span 2;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .nm-not {
    color: var(--text-primary);
  }
  .c-w {
    text-align: right;
    color: var(--text-secondary);
  }
  /* Its catalog weight under the share it took here, when axes were left out. */
  .w-word {
    display: none;
    color: var(--text-tertiary);
  }
  .base {
    display: block;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .c-pts {
    text-align: right;
  }
  .r.sum {
    border-top: 1px solid var(--line-default);
  }
  .c-sumline {
    grid-column: 2 / span 3;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .r.sum .c-pts {
    font-weight: var(--weight-semibold);
  }
  @media (max-width: 1023px) {
    .score {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  /* Phones: the axis and what it asks over a line of its bar, weight and points; the number behind it under them. */
  @media (max-width: 767px) {
    .r {
      grid-template-columns: minmax(0, 1fr) 72px 48px;
      row-gap: var(--space-2);
    }
    .r.head {
      display: none;
    }
    .c-name {
      grid-column: 1 / -1;
    }
    .c-raw {
      grid-column: 1 / -1;
      grid-row: 3;
    }
    .c-why {
      grid-column: 1;
    }
    .c-sumline {
      grid-column: 1 / span 2;
    }
    .w-word {
      display: inline;
    }
  }
`;
