<script lang="ts">
  /** The metric charts of a run. All share one crosshair (in epochs), so hovering one reads out all of them. */
  import { smooth, type Pt } from '$lib/charts/downsample';
  import LineChart, { type Series } from '$lib/charts/LineChart.svelte';
  import type { Epoch, Run, Step } from '$lib/api/types';
  import { lr as fmtLr, num } from '$lib/format';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';

  let {
    run,
    steps,
    epochs,
    version,
    loading,
  }: { run: Run; steps: Step[]; epochs: Epoch[]; version: number; loading: boolean } = $props();

  let cursor = $state<number | null>(null);
  let smoothing = $state<'0' | '0.6' | '0.9' | '0.97'>('0.9');
  let scale = $state<'lin' | 'log'>('lin');

  const xDomain = $derived<[number, number]>([0, run.epochs]);
  const epochTitle = (x: number) => `Epoch ${x.toFixed(x % 1 === 0 ? 0 : 1)}`;

  const loss = $derived.by((): Series[] => {
    void version;
    const spe = Math.max(1, run.steps_per_epoch);
    const a = Number(smoothing);
    const pts = (f: (s: Step) => number): Pt[] => steps.map((s) => ({ x: s.step / spe, y: f(s) }));
    const total = pts((s) => s.loss.total);
    return [
      { id: 'raw', label: 'total (raw)', color: 'var(--chart-1)', ghost: true, points: a > 0 ? total : [] },
      { id: 'total', label: 'total', color: 'var(--chart-1)', points: smooth(total, a) },
      { id: 'box', label: 'box', color: 'var(--chart-2)', points: smooth(pts((s) => s.loss.box), a) },
      { id: 'cls', label: 'cls', color: 'var(--chart-3)', points: smooth(pts((s) => s.loss.cls), a) },
      { id: 'dfl', label: 'dfl', color: 'var(--chart-4)', points: smooth(pts((s) => s.loss.dfl), a) },
    ];
  });

  const ep = <T,>(f: (e: Epoch) => T) => epochs.map((e) => ({ x: e.epoch + 1, y: f(e) as number }));

  const map = $derived.by((): Series[] => {
    void version;
    return [
      { id: 'm95', label: 'mAP50-95', color: 'var(--chart-1)', area: true, points: ep((e) => e.val.map50_95) },
      { id: 'm50', label: 'mAP50', color: 'var(--chart-2)', points: ep((e) => e.val.map50) },
    ];
  });

  const pr = $derived.by((): Series[] => {
    void version;
    return [
      { id: 'p', label: 'precision', color: 'var(--chart-3)', points: ep((e) => e.val.precision) },
      { id: 'r', label: 'recall', color: 'var(--chart-4)', points: ep((e) => e.val.recall) },
    ];
  });

  const thin = (n: number) => Math.max(1, Math.floor(steps.length / n));
  const lrS = $derived.by((): Series[] => {
    void version;
    const spe = Math.max(1, run.steps_per_epoch);
    const k = thin(600);
    return [{ id: 'lr', label: 'learning rate', color: 'var(--chart-5)', points: steps.filter((_, i) => i % k === 0).map((s) => ({ x: s.step / spe, y: s.lr })) }];
  });

  const ips = $derived.by((): Series[] => {
    void version;
    const spe = Math.max(1, run.steps_per_epoch);
    const pts = steps.map((s) => ({ x: s.step / spe, y: s.imgs_per_s }));
    return [{ id: 'ips', label: 'images/s', color: 'var(--chart-6)', area: true, points: smooth(pts, 0.9) }];
  });
  // Throughput reads best against zero: a steady line near the top, dips stand out.
  const ipsMax = $derived.by(() => {
    let m = 1;
    for (const p of ips[0]?.points ?? []) if (p.y > m) m = p.y;
    return m;
  });

  const best = $derived(run.best ? [{ x: run.best.epoch + 1, y: run.best.map50_95, label: `best ${run.best.map50_95.toFixed(3)}` }] : []);
</script>

{#snippet chartSk()}<Skeleton h="180px" r="var(--radius-md)" />{/snippet}

<div class="grid">
  <Panel title="Loss" class="span2" id="chart-loss">
    {#snippet actions()}
      <Segmented
        size="sm"
        label="Smoothing"
        bind:value={smoothing}
        options={[
          { value: '0', label: 'Raw' },
          { value: '0.6', label: '0.6' },
          { value: '0.9', label: '0.9' },
          { value: '0.97', label: '0.97' },
        ]}
      />
      <Segmented size="sm" label="Y scale" bind:value={scale} options={[{ value: 'lin', label: 'Lin' }, { value: 'log', label: 'Log' }]} />
    {/snippet}
    {#if loading}{@render chartSk()}{:else}
      <LineChart
        series={loss}
        height={220}
        {xDomain}
        yLog={scale === 'log'}
        yFormat={(v) => (v >= 10 ? v.toFixed(0) : v.toFixed(1))}
        valueFormat={(v) => v.toFixed(3)}
        xTitle={epochTitle}
        bind:cursor
        label="Training loss components by epoch"
        empty="Waiting for the first step…"
      />
    {/if}
  </Panel>

  <Panel title="Validation mAP" id="chart-map">
    {#if loading}{@render chartSk()}{:else}
      <LineChart
        series={map}
        height={220}
        {xDomain}
        yDomain={[0, 1]}
        yFormat={(v) => v.toFixed(1)}
        valueFormat={(v) => v.toFixed(3)}
        xTitle={epochTitle}
        curve="monotone"
        markers={best}
        bind:cursor
        label="Validation mAP50 and mAP50-95 by epoch"
        empty="Waiting for the first validation…"
      />
    {/if}
  </Panel>

  <Panel title="Precision / recall" id="chart-pr">
    {#if loading}{@render chartSk()}{:else}
      <LineChart
        series={pr}
        height={180}
        {xDomain}
        yDomain={[0, 1]}
        yFormat={(v) => v.toFixed(1)}
        valueFormat={(v) => v.toFixed(3)}
        xTitle={epochTitle}
        curve="monotone"
        bind:cursor
        label="Validation precision and recall by epoch"
        empty="Waiting for the first validation…"
      />
    {/if}
  </Panel>

  <Panel title="Learning rate" meta="warmup 3 · cosine" id="chart-lr">
    {#if loading}{@render chartSk()}{:else}
      <LineChart
        series={lrS}
        height={180}
        {xDomain}
        yFormat={(v) => fmtLr(v)}
        xTitle={epochTitle}
        bind:cursor
        label="Learning rate by epoch"
        legend={false}
      />
    {/if}
  </Panel>

  <Panel title="Throughput" meta="images/s" id="chart-ips">
    {#if loading}{@render chartSk()}{:else}
      <LineChart
        series={ips}
        height={180}
        {xDomain}
        yDomain={[0, ipsMax * 1.2]}
        yFormat={(v) => num(v, 0)}
        valueFormat={(v) => `${num(v, 0)} img/s`}
        xTitle={epochTitle}
        bind:cursor
        label="Training throughput in images per second"
        legend={false}
      />
    {/if}
  </Panel>
</div>

<style>
  .grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-4);
  }
  .grid :global(.span2) {
    grid-column: span 2;
  }
  @media (max-width: 1023px) {
    .grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
  @media (max-width: 767px) {
    .grid {
      grid-template-columns: minmax(0, 1fr);
    }
    .grid :global(.span2) {
      grid-column: auto;
    }
  }
</style>
