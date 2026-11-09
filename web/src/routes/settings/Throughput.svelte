<script lang="ts">
  /**
   * Settings › System › Throughput: the speed this server has measured (GET /api/system/throughput), training and
   * prediction rows with their spread, sample count and age; the reference rows (docs/BENCHMARKS.md) are not listed,
   * one sentence says they fill the rest. Where the server has `benchmark`, Measure now runs its probe (1–10 s, the
   * elapsed time counted honestly beside it) and reloads the rows.
   */
  import Gauge from '@lucide/svelte/icons/gauge';
  import { ApiError, hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { InferThroughput, TrainThroughput } from '$lib/api/types';
  import { deviceName, num } from '$lib/format';
  import { latencyMs, latencyRange, measuredHere, measuredRows, updatedWords } from '$lib/throughput';
  import Button from '$lib/ui/Button.svelte';
  import Ident from '$lib/ui/Ident.svelte';
  import { toast } from '$lib/ui/toast.svelte';

  let { benchmark = false }: { benchmark?: boolean } = $props();

  const table = new Resource((s) => hub.api.throughput(s));
  const rows = $derived(table.data ? measuredRows(table.data) : null);
  const empty = $derived(!!rows && !rows.training.length && !rows.inference.length);

  // The age of each row reads against a clock that moves while the page is open.
  let now = $state(Date.now());
  $effect(() => {
    const t = setInterval(() => (now = Date.now()), 30_000);
    return () => clearInterval(t);
  });

  /**
   * "640 px · batch 16 · f32" / "640 px · f16 · tensor cores", and where it was measured when that is not this server's
   * adapter. A prediction row is one precision with tensor cores on or off: the two are separate measurements.
   */
  function setting(r: TrainThroughput | InferThroughput): string {
    const parts = [`${r.imgsz} px`];
    if ('amp' in r) parts.push(`batch ${r.batch}`, r.amp ? 'amp' : 'f32');
    else parts.push(r.precision, ...(r.tensor_cores ? ['tensor cores'] : []));
    if (r.device === 'cpu') parts.push('CPU');
    return parts.join(' · ');
  }
  const elsewhere = (r: TrainThroughput | InferThroughput) => (table.data && !measuredHere(table.data, r) ? deviceName(r.measured_on) : null);
  const meta = (r: TrainThroughput | InferThroughput) => {
    const probes = 'probes' in r && r.probes > 0 ? (r.probes === r.samples ? (r.samples === 1 ? ' (a probe)' : ' (probes)') : `, ${num(r.probes)} from probes`) : '';
    return `${num(r.samples)} ${r.samples === 1 ? 'sample' : 'samples'}${probes} · ${updatedWords(r, now)}`;
  };

  // ---- Measure now ----
  let measuring = $state(false);
  let started = $state(0);
  let elapsed = $state(0);
  let failure = $state<{ message: string; hint?: string | null } | null>(null);
  $effect(() => {
    if (!measuring) return;
    const t = setInterval(() => (elapsed = Math.floor((Date.now() - started) / 1000)), 250);
    return () => clearInterval(t);
  });

  async function measure() {
    if (measuring) return;
    measuring = true;
    failure = null;
    started = Date.now();
    elapsed = 0;
    try {
      const r = await hub.api.benchmark({});
      await table.refresh();
      toast.success('Measured', `${r.training.arch} at ${r.training.imgsz} px: ${num(r.train_imgs_per_s)} img/s training, ${latencyMs(r.latency_ms)} a prediction.`);
    } catch (e) {
      failure = e instanceof ApiError ? { message: e.message, hint: e.hint } : { message: e instanceof Error ? e.message : String(e) };
    } finally {
      measuring = false;
    }
  }
</script>

<div class="tp">
  <div class="tp-head">
    <div class="setting-text">
      <p>Throughput</p>
      <p class="note">Speed measured on this server by its runs, predictions{benchmark ? ' and probes' : ''}. Estimates use it; where nothing is measured, the reference figures (docs/BENCHMARKS.md) stand in.</p>
    </div>
    {#if benchmark}
      <Button size="sm" icon={Gauge} loading={measuring} onclick={measure}>{measuring ? 'Measuring' : 'Measure now'}</Button>
    {/if}
  </div>
  {#if measuring}
    <!-- The probe trains for a few seconds, then predicts: 1–10 s plus setup. The counter is the time so far. -->
    <p class="tp-status" role="status">
      Training, then predicting on {table.data ? deviceName(table.data.device_name) : 'this server'}:{' '}<span class="tp-clock num"
        >{elapsed} s <span class="faint">of up to 10 s</span></span
      >
    </p>
  {:else if failure}
    <p class="tp-status err" role="alert">{failure.message}{#if failure.hint}{' '}<span class="faint">{failure.hint}</span>{/if}</p>
  {/if}

  {#if table.error && !table.data}
    <p class="tp-empty">Could not read the measurements.</p>
  {:else if empty}
    <p class="tp-empty">Nothing measured here yet: runs and predictions add their speed as they go{benchmark ? ', or measure now' : ''}.</p>
  {:else if rows}
    {#snippet group(title: string, list: (TrainThroughput | InferThroughput)[])}
      {#if list.length}
        <div class="tp-group">
          <h3 class="tp-h">{title}</h3>
          <ul aria-label={title}>
            {#each list as r (`${r.device}/${r.measured_on}/${r.arch}/${r.imgsz}/${'amp' in r ? `${r.batch}/${r.amp}` : `${r.precision}/${r.tensor_cores}`}`)}
              {@const other = elsewhere(r)}
              <li class="tp-row">
                <div class="tp-what">
                  <p><Ident value={r.arch} /> <span class="tp-set num">{setting(r)}</span></p>
                  <p class="tp-meta num">{meta(r)}{#if other}{' '}· measured on {other}, not used{/if}</p>
                </div>
                <div class="tp-val num">
                  {#if 'amp' in r}
                    <p>{num(r.imgs_per_s)} img/s</p>
                    {#if r.mosaic && r.mosaic.samples > 1}<p class="tp-meta">{num(r.mosaic.low)}–{num(r.mosaic.high)}</p>{/if}
                  {:else}
                    <p>{latencyMs(r.latency_ms)}</p>
                    {#if r.samples > 1}<p class="tp-meta">{latencyRange(r.latency_ms_low, r.latency_ms_high)}</p>{/if}
                  {/if}
                </div>
              </li>
            {/each}
          </ul>
        </div>
      {/if}
    {/snippet}
    {@render group('Training', rows.training)}
    {@render group('Predictions, end to end', rows.inference)}
  {/if}
</div>

<style>
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
    max-width: 60ch;
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
  .tp-h {
    padding-bottom: var(--space-1);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    font-weight: var(--weight-medium);
    color: var(--text-tertiary);
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
</style>
