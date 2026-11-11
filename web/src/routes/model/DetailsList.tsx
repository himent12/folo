/** The model's details: what it is, how it was trained and stored, its classes, where it came from and its latency. */
import { bindable, computed, css, For } from 'fw';
import type { Resource } from '$lib/api/resource';
import type { Dataset, Model, Run, ThroughputTable } from '$lib/api/types';
import type { Evaluation } from '$lib/api/evaluation';
import { e2eMs } from '$lib/benchmarks';
import { classInk } from '$lib/design/classInk';
import { splitWord, type EvalSource } from '$lib/evaluated';
import { bytes, compact, date, num } from '$lib/format';
import { latencyMs, latencyRange, measuredLatency } from '$lib/throughput';
import { prefs } from '$lib/prefs';
import { href } from '$lib/router';
import InfoButton from '$lib/ui/InfoButton';
import { publishedSummary } from '$lib/published';
import Ident from '$lib/ui/Ident';
import { FROM_SCRATCH, type Origin } from '$lib/lineage';
import Lineage from '$lib/ui/Lineage';
import Meta from '$lib/ui/Meta';
// The head and precision in the words Run's Config and Compare use.
import { HEAD_LABEL, PRECISION_LABEL } from '../run/resolved';

export interface DetailsListProps {
  m: Model;
  /** Folded rail (up to 1199 px) and phones: labels beside their values, in two columns. */
  two: boolean;
  tp: Resource<ThroughputTable | null>;
  run: Run | null | undefined;
  /** The source run, its dataset and the model it was fine-tuned from. */
  context: Resource<{ run: Run | null; dataset: Dataset | null; parent: Model | null }>;
  source: EvalSource;
  ev: Evaluation | null;
  lineage: Origin | null;
  /** No registered dataset has the model's classes: there is no evaluation to describe. */
  unevaluated?: boolean;
  stored: string | null;
  head: 'o2m' | 'o2o';
  aboutOpen?: boolean;
}

/**
 * The architecture's end-to-end time from docs/BENCHMARKS.md (640 px, one RTX 4080 SUPER), or undefined when not
 * measured. It is a benchmark of the architecture, never this model's own timing, and is labelled as such.
 */
const latency = (m: Model): number | undefined => e2eMs[m.arch];

export default function DetailsList(props: DetailsListProps) {
  const aboutOpen = bindable(props, 'aboutOpen');
  const here = computed(() => measuredLatency(props.tp.data, props.m.arch, props.m.imgsz));
  const m = () => props.m;
  const src = () => props.source;
  const h = () => here.get()!;

  return (
    <dl class="details" class:two={props.two} id="model-details">
      <div class="kv">
        <dt>Architecture</dt>
        <dd>
          <Ident value={m().arch} />
        </dd>
      </div>
      <div class="kv">
        <dt>Input size</dt>
        <dd class="num">
          {m().imgsz} × {m().imgsz} px
        </dd>
      </div>
      <div class="kv">
        <dt>Parameters</dt>
        <dd class="num">
          {compact(m().params)} <span class="faint">({m().params.toLocaleString('en-US')})</span>
        </dd>
      </div>
      <div class="kv">
        <dt>Compute</dt>
        <dd class="num">{num(m().flops_g, 1)} GFLOPs</dd>
      </div>
      <div class="kv">
        <dt>Head</dt>
        <dd>{HEAD_LABEL[props.head]}</dd>
      </div>
      {/* How it was trained, from the file's training record: none for exports, imported weights and older
          servers, so nothing is said then. */}
      {m().trained_with ? (
        <>
          <div class="kv">
            <dt>Trained in</dt>
            <dd>{m().trained_with!.amp ? PRECISION_LABEL.amp : PRECISION_LABEL.f32}</dd>
          </div>
          <div class="kv">
            <dt>Trained for</dt>
            <dd class="num">
              {num(m().trained_with!.epochs)} {m().trained_with!.epochs === 1 ? 'epoch' : 'epochs'} at {m().trained_with!.imgsz} px
            </dd>
          </div>
        </>
      ) : null}
      {props.stored ? (
        <div class="kv">
          <dt>Stored as</dt>
          <dd>{props.stored}</dd>
        </div>
      ) : null}
      <div class="kv">
        <dt>File size</dt>
        <dd class="num">{bytes(m().bytes, prefs.units)}</dd>
      </div>
      <div class="kv wide">
        <dt>Classes</dt>
        <dd class="classes">
          <For each={m().classes.slice(0, 10)} key={(_, i) => i}>
            {(c, i) => (
              <span class="chip">
                <span class="sw" style:background={classInk(i())} aria-hidden="true"></span>
                {c()}
              </span>
            )}
          </For>
          {m().classes.length > 10 ? <span class="faint">+{m().classes.length - 10} more</span> : null}
        </dd>
      </div>
      <div class="kv" class:stack={!m().run_id && m().source === 'published'}>
        <dt>Source</dt>
        <dd>
          {m().run_id ? (
            <>
              <a class="link" href={href(`/runs/${m().run_id}`)}>
                {props.run?.name ?? m().run_id}
              </a>{' '}
              {props.context.data?.dataset ? (
                <>
                  <span class="faint">{' on '}</span>
                  <a class="link" href={href(`/datasets/${props.context.data.dataset.id}`)}>
                    {props.context.data.dataset.name}
                  </a>
                </>
              ) : null}
            </>
          ) : m().source === 'published' ? (
            // `folo weights pull` weights: what they are, in the words `folo weights list` prints. What they are for in
            // plain words; the server's own description, its provenance, in the ⓘ.
            <>
              <span class="pub">
                Published weights
                {m().about ? <InfoButton label="Where these weights come from" controls="model-about" bind:open={aboutOpen} /> : null}
              </span>{' '}
              <span class="about">{publishedSummary(m())}</span>{' '}
              {m().about && aboutOpen.get() ? (
                <span class="about" id="model-about">
                  {m().about}
                </span>
              ) : null}
            </>
          ) : (
            <span class="muted">Imported .folo file</span>
          )}
        </dd>
      </div>
      {/* What the curves, the per-class table and the matrix were measured on (/eval's provenance). */}
      <div class="kv">
        <dt>Evaluated on</dt>
        {/* The dataset as the value, the split and its images under it, so the row keeps the rail's alignment. */}
        <dd class="evald">
          {props.unevaluated ? (
            <span class="muted">Not evaluated: no registered dataset has its classes</span>
          ) : src().datasetName && src().datasetId ? (
            <a class="link" href={href(`/datasets/${encodeURIComponent(src().datasetId!)}`)}>
              {src().datasetName}
            </a>
          ) : src().datasetName ? (
            <span>{src().datasetName}</span>
          ) : null}{' '}
          {props.unevaluated ? null : (
            <span class="num" class:faint={!!src().datasetName}>
              {src().split ? `${splitWord(src().split!)} split` : 'split not reported'}
              {props.ev ? `, ${num(props.ev.images)} ${props.ev.images === 1 ? 'image' : 'images'}` : ''}
            </span>
          )}
        </dd>
      </div>
      {props.lineage ? (
        <div class="kv">
          <dt>Started from</dt>
          <dd>{props.lineage.kind !== 'scratch' ? <Lineage origin={props.lineage} /> : <span class="muted">{FROM_SCRATCH}</span>}</dd>
        </div>
      ) : null}
      <div class="kv">
        <dt>Model id</dt>
        <dd>
          <Ident value={m().id} class="muted" />
        </dd>
      </div>
      <div class="kv">
        <dt>Created</dt>
        <dd>{date(m().created)}</dd>
      </div>
      {/* The architecture's number, never this model's own timing: measured on this server where it has
          predicted with the architecture at the model's size (its spread and count beside it), the reference
          (docs/BENCHMARKS.md) otherwise or under it. The label says whose it is; the conditions sit beside the
          value. Last, apart from the facts about this model. */}
      <div class="kv wide ref">
        <dt>{here.get() ? 'Architecture latency' : 'Architecture reference latency'}</dt>
        {here.get() ? (
          // Two labelled rows, each value with its conditions under it, so no fact is left alone on a line.
          <dd class="latency lat-rows">
            <span class="lat-k">Measured</span>
            <span class="num">{latencyMs(h().latency_ms)}</span>
            {/* Each row: where it was measured, then the conditions, as two lines of the same anatomy. */}
            <Meta class="figs num">
              <span>on this server</span>
              <span>{latencyRange(h().latency_ms_low, h().latency_ms_high)}</span>
            </Meta>
            <Meta class="figs num">
              <span>{num(h().samples)} predictions</span>
              <span>{m().imgsz} px</span>
              <span>{h().precision}</span>
            </Meta>
            <span class="lat-k">Reference</span>
            {latency(m()) !== undefined ? (
              <>
                <span class="num">{latency(m())!.toFixed(2)} ms</span>
                <Meta class="figs num">
                  <span>on an RTX 4080 SUPER</span>
                </Meta>
                <Meta class="figs num">
                  <Ident value={m().arch} />
                  <span>640 px</span>
                </Meta>
              </>
            ) : (
              <span class="faint">
                Not measured for <Ident value={m().arch} /> yet
              </span>
            )}
          </dd>
        ) : (
          <dd class="latency">
            {latency(m()) !== undefined ? (
              <>
                <span class="num">{latency(m())!.toFixed(2)} ms</span>
                <Meta class="figs">
                  <Ident value={m().arch} />
                  <span>640 px</span>
                  <span>RTX 4080 SUPER</span>
                  <span>end to end</span>
                </Meta>
              </>
            ) : (
              <span class="faint">
                Not measured for <Ident value={m().arch} /> yet
              </span>
            )}
          </dd>
        )}
      </div>
    </dl>
  );
}

css`
  .link {
    color: var(--text-primary);
  }
  .link:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  .details {
    display: grid;
    gap: var(--space-3);
  }
  .kv {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: var(--space-4);
  }
  /* A value with a sentence under it (published weights and what they are): the label over it, the sentence at the
     measure, in the rail; the folded rail keeps its label column (\`.two .kv\` below). */
  .kv.stack {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 2px;
  }
  .kv.stack dd {
    display: grid;
    gap: 2px;
    text-align: left;
  }
  .pub {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
  }
  .about {
    max-inline-size: var(--measure);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    text-wrap: pretty;
  }
  /* Folded rail (up to 1199 px): labels beside their values, in two columns. */
  .details.two {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-2) var(--space-8);
  }
  .two .kv {
    display: grid;
    grid-template-columns: 112px minmax(0, 1fr);
    gap: var(--space-3);
  }
  .two .kv.wide {
    grid-column: 1 / -1;
  }
  .two dd {
    text-align: left;
  }
  .two .classes {
    justify-content: flex-start;
  }
  .evald {
    display: grid;
    justify-items: end;
    text-align: right;
  }
  .evald .faint {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .two .evald {
    justify-items: start;
    text-align: left;
  }
  dt {
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  dd {
    min-width: 0;
    text-align: right;
    overflow-wrap: anywhere;
  }
  /* Stacked in every layout: the label is long, and the value keeps its conditions on the same line. */
  .kv.ref,
  .two .kv.ref {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 2px;
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
  }
  /* The value, then each line of its conditions as its own meta line: items wrap whole, so a fact ("RTX 4080 SUPER")
     never breaks inside (dd's overflow-wrap would split it). Figures people read are secondary (§2.3). */
  .latency {
    display: grid;
    justify-items: start;
    gap: 2px;
    text-align: left;
    overflow-wrap: normal;
  }
  .latency .faint,
  .latency :global(.figs) {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .latency :global(.figs) {
    color: var(--text-secondary);
  }
  /* Measured and Reference: a label column, the value beside it and its conditions under the value. */
  .lat-rows {
    grid-template-columns: max-content minmax(0, 1fr);
    column-gap: var(--space-3);
  }
  /* Global: the conditions line is ui/Meta's own element. */
  .lat-rows > :global(:not(.lat-k)) {
    grid-column: 2;
  }
  .lat-k {
    grid-column: 1;
    color: var(--text-tertiary);
  }
  .lat-rows > .lat-k:not(:first-child),
  .lat-rows > .lat-k:not(:first-child) + :global(*) {
    margin-top: var(--space-2);
  }
  .classes {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: var(--space-1);
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 20px;
    padding: 0 6px;
    border-radius: var(--radius-xs);
    background: var(--bg-active);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  @media (max-width: 767px) {
    .details.two {
      grid-template-columns: minmax(0, 1fr);
    }
    .two .kv {
      grid-template-columns: 96px minmax(0, 1fr);
    }
  }
`;
