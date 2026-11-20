/**
 * Could it label for itself? (docs/BENCH.md, Self-labeling): the precision of the labels it keeps and the share of all
 * objects they find, against the confidence threshold they are kept from (the 101-point curve); the 90% target as a
 * reference level and the threshold t* where precision first reaches it, with what it covers there. The readout in
 * words, the verdict with what it means, and, for a model whose classes are the benchmark's, the next step: a loop
 * that tests whether those labels train a better model. Designed for a curve that never reaches 90% (no t*) and for
 * a contestant without confidences (one operating point, no curve to draw).
 */
import Repeat from '$icons/repeat';
import { computed, css, For, MediaQuery, signal } from 'fw';
import type { BenchEntry, BenchResult } from '$lib/api/types';
import { TARGET_PRECISION } from '$lib/bench/rules';
import XYChart from '$lib/charts/XYChart';
import { media } from '$lib/design/tokens';
import { conf, count, num, pct } from '$lib/format';
import Button from '$lib/ui/Button';
import Panel from '$lib/ui/Panel';
import { labelCurves, labelPointAt } from '../charts';
import { MIN_LABELS } from '$lib/bench/score';
import { fine, givesNone, labelsText, perImageText, READINESS, READINESS_MEANS, selfLabelReading, share, shareApart } from '../words';

export interface SelfLabelSectionProps {
  result: BenchResult;
  entry: BenchEntry;
  /** The loop it could start: `why` null when it can (a model of this server with the benchmark's classes). */
  loop: { why: string | null; start: () => void } | null;
}

export default function SelfLabelSection(props: SelfLabelSectionProps) {
  const s = computed(() => props.result.self_labeling);
  const phone = new MediaQuery(media.phone, false);
  const curves = computed(() => labelCurves(s.get().curve));
  const scored = computed(() => props.entry.scored && s.get().curve.length > 1);
  const at = computed(() => (s.get().threshold !== null ? labelPointAt(s.get().curve, s.get().threshold!) : null));
  const cursor = signal<number | null>(null);
  const stops = computed(() =>
    s
      .get()
      .curve.filter((p) => p.labels > 0)
      .map((p) => p.confidence),
  );
  const one = computed(() => s.get().curve[0]);
  const meters = computed(() => {
    const o = one.get();
    return o
      ? [
          { k: 'Right (precision)', v: o.precision ?? 0, target: true },
          { k: 'Objects found (recall)', v: o.recall, target: false },
        ]
      : [];
  });

  const readout = (c: () => number | null, hint: () => string | null) => {
    const p = computed(() => (c() === null ? (at.get() ?? s.get().curve[0]) : labelPointAt(s.get().curve, c()!)));
    return (
      <>
        {hint() ? (
          <span class="hint">{hint()}</span>
        ) : (
          <>
            <span class="t">{c() === null ? (at.get() ? `At t* ${conf(s.get().threshold)}` : 'All its detections') : `From ${conf(c())} up`}</span>
            <span class="key">
              <i class="ln" aria-hidden="true"></i>Right {p.get()?.precision != null ? fine(p.get()!.precision!) : '—'}
            </span>
            <span class="key">
              <i class="ln dash" aria-hidden="true"></i>Objects found {p.get() ? share(p.get()!.recall) : '—'}
            </span>
            {p.get() ? (
              <span class="n">
                {c() === null && !at.get() ? (
                  <>{num(p.get()!.labels)} detections, none kept</>
                ) : (
                  <>
                    {labelsText(p.get()!.labels)}
                    {p.get()!.labels > 0 && p.get()!.labels < MIN_LABELS ? <>, too few to rest a threshold on</> : null}
                  </>
                )}
              </span>
            ) : null}
          </>
        )}
      </>
    );
  };

  return (
    <Panel title="Self-labeling" meta="the labels it keeps, by confidence threshold" id="self-labeling">
      <div class="sl">
        <div class="chart">
          {scored.get() ? (
            <XYChart
              label="Share of the labels kept that is right, and share of all objects they find, by the confidence they are kept from"
              series={[
                { id: 'p', label: 'Right (precision)', color: 'var(--text-primary)', points: curves.get().precision },
                { id: 'r', label: 'Objects found (recall)', color: 'var(--text-tertiary)', points: curves.get().recall, pattern: 'dash', thin: true },
              ]}
              xDomain={[0, 1]}
              yDomain={[0, 1]}
              xTicks={phone.current ? [0, 0.5, 1] : [0, 0.2, 0.4, 0.6, 0.8, 1]}
              yTicks={[0, 0.25, 0.5, 0.75, 1]}
              xFormat={(v) => (v === 0 || v === 1 ? String(v) : v.toFixed(1))}
              yFormat={(v) => (v === 0 || v === 1 ? String(v) : v.toFixed(2))}
              xLabel="Kept from confidence"
              yLabel="Share"
              height={phone.current ? 240 : 280}
              refs={[
                { axis: 'y', value: TARGET_PRECISION, label: '90% right: enough to train on', below: s.get().threshold !== null },
                ...(s.get().threshold !== null ? [{ axis: 'x' as const, value: s.get().threshold!, label: `t* ${conf(s.get().threshold)}` }] : []),
              ]}
              marks={
                at.get()
                  ? [
                      {
                        x: at.get()!.confidence,
                        y: at.get()!.recall,
                        label: `${share(at.get()!.recall)} of objects, ${labelsText(at.get()!.labels)}`,
                        side: at.get()!.confidence > 0.6 ? 'left' : 'right',
                      },
                    ]
                  : []
              }
              stops={stops.get()}
              bind:cursor={cursor}
              readout={readout}
              speak={(c) => {
                const p = labelPointAt(s.get().curve, c);
                return p
                  ? `From ${conf(c)} up: ${labelsText(p.labels)}, ${p.precision !== null ? pct(p.precision, 0) : 'none'} right, ${share(p.recall)} of all objects`
                  : `From ${conf(c)} up: no labels`;
              }}
              table={{
                caption: 'Labels kept by confidence threshold',
                head: ['Kept from', 'Right', 'Objects found', 'Labels'],
                rows: s
                  .get()
                  .curve.filter((p) => p.labels > 0 && Math.round(p.confidence * 100) % 5 === 0)
                  .map((p) => [conf(p.confidence), p.precision !== null ? pct(p.precision, 0) : '—', share(p.recall), num(p.labels)]),
              }}
            />
          ) : one.get() ? (
            // No confidences: one operating point, read as two shares against the 90% a label set needs.
            <div class="one">
              <p class="one-h">One operating point: it {props.entry.scored ? 'gives no confidences' : givesNone(props.entry)}, so no threshold can trade labels for precision.</p>
              <For each={meters.get()} key={(row) => row.k}>
                {(row) => (
                  <div class="meter">
                    <span class="m-k">{row().k}</span>
                    <span class="m-bar" aria-hidden="true">
                      <span class="m-fill" style:width={`${row().v * 100}%`}></span>
                      {row().target ? <span class="m-target" style:left={`${TARGET_PRECISION * 100}%`}></span> : null}
                    </span>
                    <span class="m-v num">
                      {row().target
                        ? one.get()!.labels >= MIN_LABELS
                          ? shareApart(row().v, TARGET_PRECISION)
                          : pct(row().v, 0)
                        : share(row().v)}
                    </span>
                  </div>
                )}
              </For>
              <p class="one-cap">
                The mark on the first bar is the 90% a label set to train on needs. {labelsText(one.get()!.labels)} on {count(props.entry.images, 'image')}
                {one.get()!.labels < MIN_LABELS ? <>: too few to rest a label set on (it takes at least {MIN_LABELS})</> : null}.
              </p>
            </div>
          ) : null}
        </div>
        <aside class="read">
          <h3 class="verdict">{READINESS[s.get().readiness]}</h3>
          <p class="means">{READINESS_MEANS[s.get().readiness]}</p>
          <p class="sentence">{selfLabelReading(s.get(), scored.get(), props.entry.scored ? undefined : props.entry)}</p>
          <dl class="kv">
            <div>
              <dt>Threshold t*</dt>
              <dd class="num">{s.get().threshold !== null ? conf(s.get().threshold) : '—'}</dd>
            </div>
            <div>
              <dt>Axis (objects at 90%)</dt>
              <dd class="num">{share(s.get().recall)}</dd>
            </div>
            {/* The labels the threshold rests on (all its detections without one), and per image as a reader can see it. */}
            {/* Without confidences, a point that reaches 90% keeps every box it gave as a label (the axis is their recall). */}
            <div>
              <dt>{s.get().threshold !== null ? 'Labels at t*' : !scored.get() && s.get().recall > 0 ? 'Labels' : 'Detections (none kept)'}</dt>
              <dd class="num">
                {num(s.get().labels)}
                <span class="per">{perImageText(s.get().labels_per_image, s.get().labels)}</span>
              </dd>
            </div>
            <div>
              <dt>Mean IoU</dt>
              <dd class="num">{s.get().mean_iou !== null ? num(s.get().mean_iou!, 2) : '—'}</dd>
            </div>
          </dl>
          {props.loop ? (
            <div class="next">
              <p class="next-h">
                Does training on these labels help? A loop tests it: this model labels a pool of images, a student trains on those labels alone, and this
                benchmark grades the student.
              </p>
              {props.loop.why ? (
                <p class="next-why" data-gate-why>
                  {props.loop.why}
                </p>
              ) : (
                <Button variant="primary" icon={Repeat} onclick={() => props.loop!.start()}>
                  Test it: start a loop
                </Button>
              )}
            </div>
          ) : null}
        </aside>
      </div>
    </Panel>
  );
}

css`
  .sl {
    display: grid;
    grid-template-columns: minmax(0, 8fr) minmax(0, 4fr);
    gap: var(--space-4) var(--space-8);
    align-items: start;
  }
  .chart {
    min-width: 0;
  }
  .read {
    display: grid;
    gap: var(--space-3);
    align-content: start;
  }
  .verdict {
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    font-weight: var(--weight-medium);
  }
  .means {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    max-inline-size: var(--measure);
  }
  .sentence {
    max-inline-size: var(--measure);
  }
  .kv {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-3);
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
  .next {
    display: grid;
    justify-items: start;
    gap: var(--space-3);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
  }
  .next-h,
  .next-why {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    max-inline-size: var(--measure);
  }
  .key {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-left: var(--space-3);
    color: var(--text-primary);
  }
  .ln {
    width: 14px;
    height: 0;
    border-top: 2px solid var(--text-primary);
  }
  .ln.dash {
    border-top: 1px dashed var(--text-tertiary);
  }
  .t,
  .n {
    color: var(--text-tertiary);
  }
  .n {
    margin-left: var(--space-3);
  }
  .hint {
    color: var(--text-secondary);
  }
  .one {
    display: grid;
    gap: var(--space-3);
  }
  .one-h {
    color: var(--text-secondary);
    max-inline-size: var(--measure-dialog);
  }
  .meter {
    display: grid;
    grid-template-columns: 176px minmax(0, 1fr) 48px;
    align-items: center;
    gap: var(--space-3);
  }
  .m-bar {
    position: relative;
    height: 8px;
    border-radius: 4px;
    background: var(--bg-active);
  }
  .m-fill {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    border-radius: 4px;
    background: var(--text-secondary);
  }
  /* The 90% target: an ink tick across the bar, 4 px over each edge. */
  .m-target {
    position: absolute;
    top: -4px;
    bottom: -4px;
    width: 2px;
    margin-left: -1px;
    background: var(--text-primary);
  }
  .m-v {
    text-align: right;
  }
  .one-cap {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  @media (max-width: 1023px) {
    .sl {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  @media (max-width: 767px) {
    .meter {
      grid-template-columns: minmax(0, 1fr) 48px;
    }
    .m-k {
      grid-column: 1 / -1;
    }
    .key,
    .n {
      margin-left: var(--space-2);
    }
  }
  /* Per image beside the count, small, on its line: the next value's baseline is not pushed down. */
  .per {
    display: inline;
    margin-left: var(--space-2);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-secondary);
  }
`;
