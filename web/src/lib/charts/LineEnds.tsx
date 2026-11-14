/**
 * What sits at the lines' ends, inside the chart's SVG: each line's end mark (a dot, or how the run ended), the
 * direct labels with their leaders, and the best markers (the hollow ring, its pulse and its focus brackets).
 */
import { css, For, Key } from 'fw';
import type { Placed } from './endLabels';
import { MORE_ID } from './endLabels';
import { endMarkInk, LIVE_GROW } from './endMarks';
import { paintOrder } from './layers';
import { inkOf, swatchDash, type Marker, type Series } from './lineSeries';

export type EndLabel = Placed & {
  key: Series | null;
  tx: number;
  color: string;
  text: string;
  lead: { x1: number; y1: number; x2: number; y2: number } | null;
};

export interface LineEndsProps {
  endDots: boolean;
  keyed: Series[];
  receding: boolean;
  highlight: string | null;
  contextual: (s: Series) => boolean;
  mergedInto: (id: string) => Marker | null;
  lifted: (items: Series[], id: (s: Series) => string) => Series[];
  x: (v: number) => number;
  y: (v: number) => number;
  pulseBase: number;
  focusBase: number;
  showEnds: boolean;
  ends: EndLabel[];
  markers: Marker[];
  merges: (string | null)[];
  xExtent: [number, number];
  width: number;
  margin: { top: number; right: number; bottom: number; left: number };
}

/** The run just completed: lime viewfinder corners close in on the best ring and fade. */
const brackets = (h: number) => {
  const k = 3.5;
  return <path class="focus" d={`M${-h} ${-h + k}V${-h}H${-h + k}M${h - k} ${-h}H${h}V${-h + k}M${h} ${h - k}V${h}H${h - k}M${-h + k} ${h}H${-h}V${h - k}`} />;
};

export default function LineEnds(props: LineEndsProps) {
  return (
    <>
      {props.endDots ? (
        <For
          each={props.lifted(
            paintOrder(props.keyed, (s) => s),
            (s) => s.id,
          )}
          key={(s) => s.id}
        >
          {(s) => {
            const p = () => s().points[s().points.length - 1];
            const r = () => (props.receding && s().id === props.highlight ? 4 : 3);
            const ink = () => (props.contextual(s()) ? 'var(--chart-context)' : inkOf(s()));
            const best = () => props.mergedInto(s().id);
            const px = () => props.x(p()!.x);
            const py = () => props.y(p()!.y);
            return (
              <>
                {p() && Number.isFinite(p()!.y) ? (
                  <>
                    {best() ? (
                      // The best is the line's end: its ink ring around the end mark, one mark (./endMarks).
                      <g class="marker" data-merged={s().id} transform={`translate(${px()},${py()})`}>
                        {best()!.pulse && best()!.pulse! > props.pulseBase ? (
                          <Key when={best()!.pulse}>
                            <circle class="pulse" r={(s().end === 'live' ? r() + LIVE_GROW : r()) + 3} />
                          </Key>
                        ) : null}
                        <circle class="ring" r={(s().end === 'live' ? r() + LIVE_GROW : r()) + 3} />
                        {best()!.focus && best()!.focus! > props.focusBase ? (
                          <Key when={best()!.focus}>{brackets((s().end === 'live' ? r() + LIVE_GROW : r()) + 3 + 4)}</Key>
                        ) : null}
                      </g>
                    ) : null}
                    {s().end === 'failed' ? (
                      <path
                        class="end-x"
                        d={`M${px() - r()} ${py() - r()}l${2 * r()} ${2 * r()}m0 ${-2 * r()}l${-2 * r()} ${2 * r()}`}
                        style:stroke={endMarkInk(s().end, ink())}
                      />
                    ) : s().end === 'stopped' ? (
                      <rect class="end-dot" x={px() - r()} y={py() - r()} width={2 * r()} height={2 * r()} style:fill={ink()} />
                    ) : s().end === 'live' ? (
                      <>
                        {/* Still training: a filled lime dot with the heartbeat, StatusBadge's running mark. */}
                        <circle class="end-beat" cx={px()} cy={py()} r={r() + LIVE_GROW} />
                        <circle class="end-dot end-live" cx={px()} cy={py()} r={r() + LIVE_GROW} style:fill={endMarkInk(s().end, ink())} />
                      </>
                    ) : (
                      <circle class="end-dot" cx={px()} cy={py()} r={r()} style:fill={ink()} />
                    )}
                  </>
                ) : null}
              </>
            );
          }}
        </For>
      ) : null}

      {props.showEnds ? (
        <>
          <For each={props.ends} key={(e) => e.id}>
            {(e) => (
              // A label moved off its line's end (to keep clear of lines and labels) is tied to its marker by a leader.
              <>{e().lead ? <line class="leader" x1={e().lead!.x1} y1={e().lead!.y1} x2={e().lead!.x2} y2={e().lead!.y2} style:stroke={e().color} /> : null}</>
            )}
          </For>
          <For each={props.ends} key={(e) => e.id}>
            {(e) => (
              <>
                {e().key ? (
                  // Keyed: the label could not sit beside its line's end, so it starts with the line's own swatch.
                  <line
                    class="end-key"
                    x1={e().x + 1}
                    x2={e().x + 13}
                    y1={e().y}
                    y2={e().y}
                    style:stroke={props.receding && e().id !== props.highlight ? 'var(--chart-context)' : inkOf(e().key!)}
                    style:stroke-dasharray={swatchDash(e().key!)}
                  />
                ) : null}
                <text
                  class="end-label"
                  class:hl={e().id === props.highlight}
                  class:more={e().id === MORE_ID}
                  x={e().tx}
                  y={e().y}
                  dy="0.32em"
                  style:fill={e().text}
                >
                  {e().label}
                </text>
              </>
            )}
          </For>
        </>
      ) : null}
      <For each={props.markers} key={(m) => m.label + m.x}>
        {(m, i) => {
          const at = () => props.x(m().x);
          const right = () => at() > props.width - props.margin.right - 40;
          const left = () => at() < props.margin.left + 40;
          return (
            <>
              {m().x >= props.xExtent[0] && m().x <= props.xExtent[1] && props.merges[i()] == null ? (
                <g class="marker" transform={`translate(${at()},${props.y(m().y)})`}>
                  {m().pulse && m().pulse! > props.pulseBase ? (
                    <Key when={m().pulse}>
                      <circle class="pulse" r="4.5" />
                    </Key>
                  ) : null}
                  <circle class="ring" r="4.5" />
                  {m().focus && m().focus! > props.focusBase ? <Key when={m().focus}>{brackets(8.5)}</Key> : null}
                  {!m().quiet ? (
                    <text y="-10" text-anchor={right() ? 'end' : left() ? 'start' : 'middle'} dx={right() ? 6 : left() ? -6 : 0}>
                      {m().label}
                    </text>
                  ) : null}
                </g>
              ) : null}
            </>
          );
        }}
      </For>
    </>
  );
}

css`
  .end-dot {
    stroke: var(--plot-bg, var(--bg-panel));
    stroke-width: 1.5;
  }
  /* A failed run's line ends in a cross, over a halo in the panel colour like the dots. */
  .end-x {
    fill: none;
    stroke-width: 2;
    stroke-linecap: round;
    filter: drop-shadow(0 0 1px var(--plot-bg, var(--bg-panel)));
  }
  /* Still training: a filled lime dot (style:fill) whose heartbeat, a ring every 2.4 s, is StatusBadge's \`live-pulse\`
     drawn in SVG; off under reduced motion, where the dot still says "now". */
  .end-beat {
    fill: none;
    stroke: var(--accent-line);
    stroke-width: 1.5;
    opacity: 0;
    transform-box: fill-box;
    transform-origin: center;
    animation: end-beat 2.4s var(--ease-out) infinite;
  }
  @keyframes end-beat {
    0% {
      opacity: 0.55;
      transform: scale(1);
    }
    33%,
    100% {
      opacity: 0;
      transform: scale(2.2);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .end-beat {
      display: none;
    }
  }
  .end-label {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    paint-order: stroke;
    stroke: var(--plot-bg, var(--bg-panel));
    stroke-width: 3px;
  }
  .end-label.hl {
    font-weight: var(--weight-semibold);
  }
  /* The column's "+N more": not a line's name, so the quiet ink and the regular weight. */
  .end-label.more {
    fill: var(--text-tertiary);
    font-weight: var(--weight-regular);
  }
  /* A keyed label's swatch: the legend's (2 px, round caps). */
  .end-key {
    stroke-width: 2;
    stroke-linecap: round;
  }
  .leader {
    stroke-width: 1;
    stroke-linecap: round;
  }
  /* The best epoch: a hollow ink ring, 1.5 px, the header's "○ best" glyph (./endMarks). On its own it is filled with
     the panel, so the line does not run through it; around an end mark it is only the ring. */
  .marker .ring {
    fill: none;
    stroke: var(--text-primary);
    stroke-width: 1.5;
  }
  .marker:not([data-merged]) .ring {
    fill: var(--plot-bg, var(--bg-panel));
  }
  /* A new best: a ring grows out of the marker and fades, once, over the chart duration. */
  .marker .pulse {
    fill: none;
    stroke: var(--text-primary);
    stroke-width: 1.5;
    opacity: 0;
    transform-box: fill-box;
    transform-origin: center;
    animation: best-pulse var(--dur-chart) var(--ease-out);
  }
  @keyframes best-pulse {
    from {
      opacity: 0.9;
      transform: scale(1);
    }
    to {
      opacity: 0;
      transform: scale(2.4);
    }
  }
  /* The run completed: four lime corners close in on the best ring over 200 ms, hold, then fade (1 s in all). */
  .marker .focus {
    fill: none;
    stroke: var(--accent-line);
    stroke-width: 1.5;
    stroke-linecap: square;
    opacity: 0;
    transform-box: fill-box;
    transform-origin: center;
    animation: best-focus 1s var(--ease-out);
  }
  @keyframes best-focus {
    0% {
      opacity: 0;
      transform: scale(1.8);
    }
    20%,
    70% {
      opacity: 1;
      transform: scale(1);
    }
    100% {
      opacity: 0;
      transform: scale(1);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .marker .pulse,
    .marker .focus {
      display: none;
    }
  }
  .marker text {
    fill: var(--text-secondary);
    font-size: var(--type-sm-size);
    font-variant-numeric: tabular-nums;
    paint-order: stroke;
    stroke: var(--plot-bg, var(--bg-panel));
    stroke-width: 3px;
  }
`;
