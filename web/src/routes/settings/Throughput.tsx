/**
 * Settings › System › Throughput: the speed this server has measured (GET /api/system/throughput), training and
 * prediction rows with their spread, sample count and age; the reference rows (docs/BENCHMARKS.md) are not listed,
 * one sentence says they fill the rest. Where the server has `benchmark`, Measure now runs its probe (1–10 s, the
 * elapsed time counted honestly beside it) and reloads the rows.
 */
import Gauge from '$icons/gauge';
import { computed, css, defaults, effect, For, signal } from 'fw';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { InferThroughput, TrainThroughput } from '$lib/api/types';
import { deviceName, num } from '$lib/format';
import { latencyMs, latencyRange, measuredHere, measuredRows, updatedWords } from '$lib/throughput';
import { referenceSource } from '$lib/reference';
import Button from '$lib/ui/Button';
import Ident from '$lib/ui/Ident';
import { toast } from '$lib/ui/toast';

type Row = TrainThroughput | InferThroughput;

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
  const meta = (r: Row) => {
    const probes = 'probes' in r && r.probes > 0 ? (r.probes === r.samples ? (r.samples === 1 ? ' (a probe)' : ' (probes)') : `, ${num(r.probes)} from probes`) : '';
    return `${num(r.samples)} ${r.samples === 1 ? 'sample' : 'samples'}${probes} · ${updatedWords(r, now.get())}`;
  };

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

  const group = (title: string, list: () => Row[]) => (
    <>
      {list().length ? (
        <div class="tp-group">
          <h3 class="tp-h group-label">{title}</h3>
          <ul aria-label={title}>
            <For each={list()} key={rowKey}>
              {(r) => {
                const other = () => elsewhere(r());
                const train = () => {
                  const x = r();
                  return 'amp' in x ? x : null;
                };
                const infer = () => r() as InferThroughput;
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
                      {train() ? (
                        <>
                          <p>{num(train()!.imgs_per_s)} img/s</p>{' '}
                          {train()!.mosaic && train()!.mosaic!.samples > 1 ? (
                            <p class="tp-meta">
                              {num(train()!.mosaic!.low)}–{num(train()!.mosaic!.high)}
                            </p>
                          ) : null}
                        </>
                      ) : (
                        <>
                          <p>{latencyMs(infer().latency_ms)}</p>{' '}
                          {infer().samples > 1 ? <p class="tp-meta">{latencyRange(infer().latency_ms_low, infer().latency_ms_high)}</p> : null}
                        </>
                      )}
                    </div>
                  </li>
                );
              }}
            </For>
          </ul>
        </div>
      ) : null}
    </>
  );

  return (
    <div class="tp">
      <div class="tp-head">
        <div class="setting-text">
          <p>Throughput</p>
          <p class="note">
            Speed measured on this server by its runs, predictions{p.benchmark ? ' and probes' : ''}. Estimates use it; where nothing is measured, they come from{' '}
            {referenceSource(false)}, scaled to each setting.
          </p>
        </div>
        {p.benchmark ? (
          <Button size="sm" icon={Gauge} loading={measuring.get()} onclick={measure}>
            {measuring.get() ? 'Measuring' : 'Measure now'}
          </Button>
        ) : null}
      </div>
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

      {table.error && !table.data ? (
        <p class="tp-empty">Could not read the measurements.</p>
      ) : empty.get() ? (
        <p class="tp-empty">Nothing measured here yet: runs and predictions add their speed as they go{p.benchmark ? ', or measure now' : ''}.</p>
      ) : rows.get() ? (
        <>
          {group('Training', () => rows.get()?.training ?? [])}
          {group('Predictions, end to end', () => rows.get()?.inference ?? [])}
        </>
      ) : null}
    </div>
  );
}

css`
  .tp {
    display: grid;
    gap: var(--space-3);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
  }
  .tp-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4);
  }
  .setting-text {
    min-width: 0;
  }
  .note {
    max-inline-size: var(--measure);
    margin-top: 2px;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .tp-head :global(.btn) {
    flex-shrink: 0;
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
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .tp-group {
    display: grid;
  }
  /* Type and ink: the shared group label (base.css .group-label). */
  .tp-h {
    padding-bottom: var(--space-1);
  }
  .tp-row {
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
  @media (max-width: 767px) {
    .tp-head {
      flex-direction: column;
      align-items: stretch;
    }
  }
`;
