/**
 * Settings › Throughput: the speed this server has measured (GET /api/system/throughput), as a table of training or
 * prediction rows (Training / Inference in the panel's header) with their spread, sample count and age; phones keep a
 * two-line list. The reference rows (docs/BENCHMARKS.md) are not listed, one sentence says they fill the rest. Where the
 * server has `benchmark`, Measure now runs its probe (1–10 s, the elapsed time counted honestly beside it) and reloads
 * the rows.
 */
import Gauge from '$icons/gauge';
import { computed, css, defaults, effect, For, MediaQuery, signal } from 'fw';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { InferThroughput, TrainThroughput } from '$lib/api/types';
import { media } from '$lib/design/tokens';
import { deviceName, num } from '$lib/format';
import { latencyMs, latencyRange, measuredHere, measuredRows, updatedWords } from '$lib/throughput';
import { referenceSource } from '$lib/reference';
import Button from '$lib/ui/Button';
import Ident from '$lib/ui/Ident';
import Panel from '$lib/ui/Panel';
import Segmented from '$lib/ui/Segmented';
import { toast } from '$lib/ui/toast';

type Row = TrainThroughput | InferThroughput;
type View = 'training' | 'inference';

/**
 * "640 px · batch 16 · f32" / "640 px · f16 · tensor cores", and where it was measured when that is not this server's
 * adapter. A prediction row is one precision with tensor cores on or off: the two are separate measurements.
 */
function setting(r: Row): string {
  const parts = [`${r.imgsz} px`];
  if ('amp' in r) parts.push(`batch ${r.batch}`, r.amp ? 'amp' : 'f32');
  else parts.push(r.precision, ...(r.tensor_cores ? ['tensor cores'] : []));
  if (r.device === 'cpu') parts.push('CPU');
  return parts.join(' · ');
}

/** "amp" / "f32"; "f16 · tensor cores". */
const precision = (r: Row) => ('amp' in r ? (r.amp ? 'amp' : 'f32') : `${r.precision}${r.tensor_cores ? ' · tensor cores' : ''}`);
/** The server's device the row is for: `gpu:0` as "GPU 0", `cpu` as "CPU". */
const deviceLabel = (d: string) => (d === 'cpu' ? 'CPU' : d.replace(/^gpu:(\d+)$/, 'GPU $1'));

const rowKey = (r: Row) => `${r.device}/${r.measured_on}/${r.arch}/${r.imgsz}/${'amp' in r ? `${r.batch}/${r.amp}` : `${r.precision}/${r.tensor_cores}`}`;

export default function Throughput(props: { benchmark?: boolean }) {
  const p = defaults(props, { benchmark: false });
  const table = new Resource((s) => hub.api.throughput(s));
  const rows = computed(() => (table.data ? measuredRows(table.data) : null));
  const empty = computed(() => {
    const r = rows.get();
    return !!r && !r.training.length && !r.inference.length;
  });

  // The age of each row reads against a clock that moves while the page is open.
  const now = signal(Date.now());
  effect(() => {
    const t = setInterval(() => now.set(Date.now()), 30_000);
    return () => clearInterval(t);
  });

  const elsewhere = (r: Row) => (table.data && !measuredHere(table.data, r) ? deviceName(r.measured_on) : null);
  const samples = (r: Row) => {
    const probes = 'probes' in r && r.probes > 0 ? (r.probes === r.samples ? (r.samples === 1 ? ' (a probe)' : ' (probes)') : `, ${num(r.probes)} from probes`) : '';
    return `${num(r.samples)} ${r.samples === 1 ? 'sample' : 'samples'}${probes}`;
  };
  const meta = (r: Row) => `${samples(r)} · ${updatedWords(r, now.get())}`;
  const phone = new MediaQuery(media.phone, false);

  // ---- Measure now ----
  const measuring = signal(false);
  let started = 0;
  const elapsed = signal(0);
  const failure = signal<{ message: string; hint?: string | null } | null>(null);
  effect(() => {
    if (!measuring.get()) return;
    const t = setInterval(() => elapsed.set(Math.floor((Date.now() - started) / 1000)), 250);
    return () => clearInterval(t);
  });

  async function measure() {
    if (measuring.peek()) return;
    measuring.set(true);
    failure.set(null);
    started = Date.now();
    elapsed.set(0);
    try {
      const r = await hub.api.benchmark({});
      await table.refresh();
      toast.success('Measured', `${r.training.arch} at ${r.training.imgsz} px: ${num(r.train_imgs_per_s)} img/s training, ${latencyMs(r.latency_ms)} a prediction.`);
    } catch (e) {
      failure.set(e instanceof ApiError ? { message: e.message, hint: e.hint } : { message: e instanceof Error ? e.message : String(e) });
    } finally {
      measuring.set(false);
    }
  }

  /** The group shown: Training first, Inference when only it has rows. */
  const view = signal<View>('training');
  let picked = false;
  effect(() => {
    const r = rows.get();
    if (!picked && r && !r.training.length && r.inference.length) view.set('inference');
  });
  const list = computed<Row[]>(() => (view.get() === 'training' ? (rows.get()?.training ?? []) : (rows.get()?.inference ?? [])));

  const source = (r: Row) => {
    const other = elsewhere(r);
    return other ? `Measured on ${other} · ${samples(r)} · not used here` : `Measured · ${samples(r)}`;
  };
  const range = (r: Row) => {
    if ('amp' in r) return r.mosaic && r.mosaic.samples > 1 ? `${num(r.mosaic.low)}–${num(r.mosaic.high)}` : null;
    return r.samples > 1 ? latencyRange(r.latency_ms_low, r.latency_ms_high) : null;
  };
  const figure = (r: Row) => ('amp' in r ? num(r.imgs_per_s) : latencyMs(r.latency_ms));

  /** From 768 px: one row a measurement, a column a fact (design/UNIFIED.md §4). */
  const grid = () => (
    <div class="tp-scroll">
      <table class="table" aria-label={view.get() === 'training' ? 'Training' : 'Predictions, end to end'}>
        <thead>
          <tr>
            <th scope="col">Device</th>
            <th scope="col">Architecture</th>
            <th scope="col" class="r">
              Image size
            </th>
            <th scope="col" class="r">
              Batch
            </th>
            <th scope="col">Precision</th>
            <th scope="col" class="r">
              {view.get() === 'training' ? 'img/s' : 'Latency'}
            </th>
            <th scope="col" class="r">
              Range
            </th>
            <th scope="col">Source</th>
            <th scope="col" class="r">
              Updated
            </th>
          </tr>
        </thead>
        <tbody>
          <For each={list.get()} key={rowKey}>
            {(r) => (
              <tr class="tp-row" class:other={!!elsewhere(r())}>
                <td class="muted tp-dev">{deviceLabel(r().device)}</td>
                <td class="lead">
                  <Ident value={r().arch} />
                </td>
                <td class="r num muted">{r().imgsz} px</td>
                <td class="r num muted">{r().batch}</td>
                <td class="muted">{precision(r())}</td>
                <td class="r num tp-fig">{figure(r())}</td>
                <td class="r num muted">{range(r()) ?? '—'}</td>
                <td class="muted tp-src" title={elsewhere(r()) ? `Measured on another GPU (${elsewhere(r())}), so not used for this one` : undefined}>
                  {source(r())}
                </td>
                <td class="r num muted">{updatedWords(r(), now.get())}</td>
              </tr>
            )}
          </For>
        </tbody>
      </table>
    </div>
  );

  /** Phones: each measurement two lines, the figure and its spread on the right. */
  const phoneList = () => (
    <ul class="tp-list" aria-label={view.get() === 'training' ? 'Training' : 'Predictions, end to end'}>
      <For each={list.get()} key={rowKey}>
        {(r) => {
          const other = () => elsewhere(r());
          // The spaces between the blocks render nothing; they keep the row's text as Svelte's ("f32 99 samples").
          return (
            <li class="tp-row">
              <div class="tp-what">
                <p>
                  <Ident value={r().arch} /> <span class="tp-set num">{setting(r())}</span>
                </p>{' '}
                <p class="tp-meta num">
                  {meta(r())}
                  {other() ? <>{' '}· measured on another GPU ({other()}), so not used for this one</> : null}
                </p>
              </div>{' '}
              <div class="tp-val num">
                <p>{'amp' in r() ? `${figure(r())} img/s` : figure(r())}</p>{' '}
                {range(r()) ? <p class="tp-meta">{range(r())}</p> : null}
              </div>
            </li>
          );
        }}
      </For>
    </ul>
  );

  const measureButton = () =>
    p.benchmark ? (
      <Button size="sm" icon={Gauge} loading={measuring.get()} onclick={measure}>
        {measuring.get() ? 'Measuring' : 'Measure now'}
      </Button>
    ) : null;

  return (
    <Panel
      title="Throughput"
      id="throughput"
      divided
      pad="none"
      actions={
        <>
          {rows.get() && !empty.get() ? (
            <Segmented
              size="sm"
              label="Measurements shown"
              bind:value={view}
              onchange={() => (picked = true)}
              options={[
                { value: 'training', label: 'Training', meta: String(rows.get()!.training.length) },
                { value: 'inference', label: 'Inference', meta: String(rows.get()!.inference.length) },
              ]}
            />
          ) : null}
          {phone.current ? null : measureButton()}
        </>
      }
    >
      <div class="tp">
        <p class="tp-note">
          Speed measured on {table.data ? <>this server ({deviceName(table.data.device_name)})</> : 'this server'} by its runs, predictions
          {p.benchmark ? ' and probes' : ''}; predictions are timed end to end. Estimates use it; where nothing is measured, they come from{' '}
          {referenceSource(false)}, scaled to each setting.
        </p>
        {phone.current && p.benchmark ? <div class="tp-measure">{measureButton()}</div> : null}
        {measuring.get() ? (
          /* The probe trains for a few seconds, then predicts: 1–10 s plus setup. The counter is the time so far. */
          <p class="tp-status" role="status">
            Training, then predicting on {table.data ? deviceName(table.data.device_name) : 'this server'}:{' '}
            <span class="tp-clock num">
              {elapsed.get()} s <span class="faint">of up to 10 s</span>
            </span>
          </p>
        ) : failure.get() ? (
          <p class="tp-status err" role="alert">
            {failure.get()!.message}
            {failure.get()!.hint ? (
              <>
                {' '}
                <span class="faint">{failure.get()!.hint}</span>
              </>
            ) : null}
          </p>
        ) : null}
      </div>

      {table.error && !table.data ? (
        <p class="tp-empty">Could not read the measurements.</p>
      ) : empty.get() ? (
        <p class="tp-empty">Nothing measured here yet: runs and predictions add their speed as they go{p.benchmark ? ', or measure now' : ''}.</p>
      ) : rows.get() && !list.get().length ? (
        <p class="tp-empty">{view.get() === 'training' ? 'No training measured here yet.' : 'No predictions measured here yet.'}</p>
      ) : rows.get() ? (
        phone.current ? (
          phoneList()
        ) : (
          grid()
        )
      ) : null}
    </Panel>
  );
}

css`
  .tp {
    display: grid;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
  }
  .tp-note {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .tp-status {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .tp-status.err {
    color: var(--danger-text);
  }
  .tp-clock {
    white-space: nowrap;
  }
  .tp-status .faint {
    color: var(--text-tertiary);
  }
  .tp-empty {
    padding: 0 var(--space-4) var(--space-4);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  /* The table keeps its columns: on a tablet it scrolls sideways inside the panel rather than squeeze them. */
  .tp-scroll {
    overflow-x: auto;
  }
  .tp-scroll .table {
    min-width: 640px;
  }
  .tp-scroll .table td,
  .tp-scroll .table th {
    white-space: nowrap;
  }
  /* The architecture is the row's name (.lead); the device it is for reads as the other settings do. */
  .tp-scroll .tp-dev {
    font-weight: var(--weight-regular);
  }
  /* Below 1024 px the nine columns take 8 px gaps, so the table fits a tablet's column. */
  @media (max-width: 1023px) {
    .tp-scroll .table :is(th, td):not(:first-child):not(:last-child) {
      padding: 0 var(--space-2);
    }
  }
  /* The source is a phrase: it wraps (the row grows) before Updated leaves the panel. */
  .tp-scroll .table td.tp-src {
    min-width: 120px;
    padding-top: var(--space-1);
    padding-bottom: var(--space-1);
    white-space: normal;
  }
  .tp-fig {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  /* A row measured on another adapter is listed, not used: one step quieter. */
  .tp-row.other td {
    color: var(--text-tertiary);
  }
  .tp-list {
    padding: 0 var(--space-4) var(--space-2);
  }
  .tp-list .tp-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-4);
    padding: var(--space-2) 0;
    border-top: 1px solid var(--line-subtle);
  }
  .tp-what {
    min-width: 0;
  }
  .tp-set {
    color: var(--text-secondary);
  }
  .tp-meta {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .tp-val {
    text-align: right;
    white-space: nowrap;
  }
  .tp-measure :global(.btn) {
    width: 100%;
  }
`;
