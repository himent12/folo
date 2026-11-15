/**
 * The model's details: what it is, how it was trained and stored, its classes, where it came from and its latency.
 * A full-width panel's facts, each label beside its value: three columns of rows from 1280 px (as Desktop's), the rows
 * two to a line on laptops and tablets, one on phones.
 */
import { bindable, computed, css, For } from 'fw';
import type { Resource } from '$lib/api/resource';
import type { Dataset, Model, Run, ThroughputTable } from '$lib/api/types';
import type { Evaluation } from '$lib/api/evaluation';
import { e2eMs } from '$lib/benchmarks';
import { classInk } from '$lib/design/classInk';
import { splitWord, type EvalSource } from '$lib/evaluated';
import { bytes, compact, count, date, num } from '$lib/format';
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
import { trainedFor, weightsEpoch } from './weightsEpoch';

/**
 * The source run, its dataset, the model it was fine-tuned from, and the epochs the run finished (from its history,
 * ./weightsEpoch finishedEpochs; null while not known).
 */
export interface ModelContext {
  run: Run | null;
  dataset: Dataset | null;
  parent: Model | null;
  epochs: number | null;
}

export interface DetailsListProps {
  m: Model;
  tp: Resource<ThroughputTable | null>;
  run: Run | null | undefined;
  context: Resource<ModelContext>;
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
/** Bytes per parameter of the FOLO f16 export (as the Export dialog estimates it). */
const F16_BYTES_PER_PARAM = 2.03;

export default function DetailsList(props: DetailsListProps) {
  const aboutOpen = bindable(props, 'aboutOpen');
  const here = computed(() => measuredLatency(props.tp.data, props.m.arch, props.m.imgsz));
  const m = () => props.m;
  const done = () => props.context.data?.epochs ?? null;
  const src = () => props.source;
  const h = () => here.get()!;

  // Three columns of label / value rows (as Desktop's Details): what the model is, how it was trained and stored,
  // where it came from and how fast its architecture runs. One label width and one row rhythm in every column; a long
  // value (the classes, the file-size note) wraps inside its row.
  return (
    <dl class="details" id="model-details">
      <div class="col">
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
        {m().trained_with ? (
          <div class="kv">
            <dt>Trained in</dt>
            <dd>{m().trained_with!.amp ? PRECISION_LABEL.amp : PRECISION_LABEL.f32}</dd>
          </div>
        ) : null}
      </div>
      <div class="col">
        {m().trained_with ? (
          <div class="kv" id="model-trained-for">
            <dt>Trained for</dt>
              <dd class="num">{trainedFor(m(), props.run, done())}</dd>
          </div>
        ) : null}
        {weightsEpoch(m(), props.run, done()) ? (
          <div class="kv">
            <dt>Weights from</dt>
            <dd class="num">{weightsEpoch(m(), props.run, done())}</dd>
          </div>
        ) : null}
        {props.stored ? (
          <div class="kv">
            <dt>Stored as</dt>
            <dd>{props.stored}</dd>
          </div>
        ) : null}
        <div class="kv">
          <dt>File size</dt>
          {/* A training checkpoint (it has a training record) also holds the optimizer state and the averaged weights:
              say why it is bigger than what Export writes. */}
          <dd class="evald">
            <span class="num">{bytes(m().bytes, prefs.units)}</span>
            {m().trained_with && !m().fused ? (
              <span class="faint">training checkpoint with optimizer state; exported as FOLO f16, about {bytes(m().params * F16_BYTES_PER_PARAM, prefs.units)}</span>
            ) : null}
          </dd>
        </div>
        <div class="kv">
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
      </div>
      <div class="col">
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
                {props.ev ? `, ${count(props.ev.images, 'image')}` : ''}
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
        {/* The architecture's numbers, never this model's own timing: measured on this server where it has predicted
            with the architecture at the model's size, and the reference (docs/BENCHMARKS.md). Rows of the same grid,
            each value with its conditions as its secondary lines. */}
        {here.get() ? (
          <div class="kv">
            <dt>Measured latency</dt>
            <dd class="latency">
              <span class="num">{latencyMs(h().latency_ms)}</span>
              <Meta class="figs num">
                <span>on this server</span>
                <span>{latencyRange(h().latency_ms_low, h().latency_ms_high)}</span>
              </Meta>
              <Meta class="figs num">
                <span>{num(h().samples)} predictions</span>
                <span>{m().imgsz} px</span>
                <span>{h().precision}</span>
              </Meta>
            </dd>
          </div>
        ) : null}
        <div class="kv">
          <dt>Reference latency</dt>
          <dd class="latency">
            {latency(m()) !== undefined ? (
              <>
                <span class="num">{latency(m())!.toFixed(2)} ms</span>
                <Meta class="figs num">
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
        </div>
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
  /* Three columns, each a list of rows: one label width (112 px) and one 28 px row rhythm everywhere; a value with
     secondary lines (the file-size note, the split, the latency's conditions) grows its row. */
  .details {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    column-gap: var(--space-8);
    align-items: start;
  }
  .col {
    display: grid;
    align-content: start;
    min-width: 0;
  }
  .kv {
    display: grid;
    grid-template-columns: 112px minmax(0, 1fr);
    align-items: baseline;
    gap: var(--space-3);
    min-height: var(--control-md);
    padding: var(--space-1) 0;
  }
  .kv.stack dd {
    display: grid;
    gap: 2px;
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
  .evald {
    display: grid;
    justify-items: start;
  }
  .evald .faint {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  dt {
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  dd {
    min-width: 0;
    overflow-wrap: anywhere;
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
  .classes {
    display: flex;
    flex-wrap: wrap;
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
  /* Laptops and tablets: the rows two to a line, in the columns' order. */
  @media (max-width: 1279px) {
    .details {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .col {
      display: contents;
    }
  }
  @media (max-width: 767px) {
    .details {
      grid-template-columns: minmax(0, 1fr);
    }
  }
`;
