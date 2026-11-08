<script lang="ts">
  /**
   * The run's metrics as one instrument: the epoch progress, a row of KPI tabs (each with its current value) and the one
   * chart the selected tab names. A single run is drawn in ink (primary metric in text.primary, secondary ones
   * tertiary with dash patterns); hue is reserved for entities (runs in Compare, classes).
   */
  import { tooltip } from '$lib/ui/tooltip';
  import ChartSkeleton from '$lib/charts/ChartSkeleton.svelte';
  import { smooth, type Pt } from '$lib/charts/downsample';
  import LineChart, { type Marker, type Series } from '$lib/charts/LineChart.svelte';
  import { runHorizon } from '$lib/charts/horizon';
  import type { Epoch, Run, Step } from '$lib/api/types';
  import { duration, lr as fmtLr, metric, num, pow10 } from '$lib/format';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';

  let {
    run,
    steps,
    epochs,
    version,
    loading,
  }: { run: Run; steps: Step[]; epochs: Epoch[]; version: number; loading: boolean } = $props();

  type Tab = 'map' | 'map50' | 'pr' | 'loss' | 'lr' | 'ips';
  let tab = $state<Tab>('map');

  // Phones get a shorter plot so the samples start within the first screen.
  let narrow = $state(typeof matchMedia !== 'undefined' && matchMedia('(max-width: 767px)').matches);
  $effect(() => {
    const mq = matchMedia('(max-width: 767px)');
    const on = () => (narrow = mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  });

  const live = $derived(['queued', 'running', 'stopping'].includes(run.status));
  const progress = $derived(Math.min(1, run.step / Math.max(1, run.steps_per_epoch * run.epochs)));
  const closeFrom = $derived(1 - run.config.augment.close_mosaic / run.epochs);
  const elapsed = $derived(run.started ? ((run.finished ? Date.parse(run.finished) : Date.now()) - Date.parse(run.started)) / 1000 : null);

  const lastStep = $derived.by(() => {
    void version;
    return steps[steps.length - 1];
  });
  const lastEpoch = $derived.by(() => {
    void version;
    return epochs[epochs.length - 1];
  });

  // ---- chart data ----
  let smoothing = $state<'0' | '0.6' | '0.9' | '0.97'>('0.9');
  let scale = $state<'lin' | 'log'>('lin');
  let horizon = $state<'observed' | 'full'>('observed');
  const observed = $derived.by(() => {
    void version;
    const spe = Math.max(1, run.steps_per_epoch);
    return Math.max(lastStep ? lastStep.step / spe : 0, lastEpoch ? lastEpoch.epoch + 1 : 0);
  });
  const xDomain = $derived(runHorizon(observed, run.epochs, live, horizon));
  const epochTitle = (x: number) => `Epoch ${x.toFixed(x % 1 === 0 ? 0 : 1)}`;

  const INK = 'var(--text-primary)';
  const INK2 = 'var(--text-tertiary)';
  const ep = (f: (e: Epoch) => number): Pt[] => epochs.map((e) => ({ x: e.epoch + 1, y: f(e) }));
  const perStep = (f: (s: Step) => number): Pt[] => {
    const spe = Math.max(1, run.steps_per_epoch);
    return steps.map((s) => ({ x: s.step / spe, y: f(s) }));
  };

  const series = $derived.by((): Series[] => {
    void version;
    switch (tab) {
      case 'map':
        return [
          { id: 'm95', label: 'mAP50-95', color: INK, area: true, points: ep((e) => e.val.map50_95) },
          { id: 'm50', label: 'mAP50', color: INK2, pattern: 'dash', points: ep((e) => e.val.map50) },
        ];
      case 'map50':
        return [
          { id: 'm50', label: 'mAP50', color: INK, area: true, points: ep((e) => e.val.map50) },
          { id: 'm95', label: 'mAP50-95', color: INK2, pattern: 'dash', points: ep((e) => e.val.map50_95) },
        ];
      case 'pr':
        return [
          { id: 'p', label: 'Precision', color: INK, points: ep((e) => e.val.precision) },
          { id: 'r', label: 'Recall', color: INK2, pattern: 'dash', points: ep((e) => e.val.recall) },
        ];
      case 'loss': {
        const a = Number(smoothing);
        const total = perStep((s) => s.loss.total);
        return [
          { id: 'raw', label: 'total (raw)', color: INK, ghost: true, points: a > 0 ? total : [] },
          { id: 'total', label: 'Total', color: INK, emphasis: true, points: smooth(total, a) },
          { id: 'box', label: 'box', color: INK2, pattern: 'dash', points: smooth(perStep((s) => s.loss.box), a) },
          { id: 'cls', label: 'cls', color: INK2, pattern: 'dot', points: smooth(perStep((s) => s.loss.cls), a) },
          { id: 'dfl', label: 'dfl', color: INK2, pattern: 'dashdot', points: smooth(perStep((s) => s.loss.dfl), a) },
        ];
      }
      case 'lr': {
        const k = Math.max(1, Math.floor(steps.length / 600));
        return [{ id: 'lr', label: 'Learning rate', color: INK, points: perStep((s) => s.lr).filter((_, i) => i % k === 0) }];
      }
      case 'ips':
        return [{ id: 'ips', label: 'Images/s', color: INK, points: smooth(perStep((s) => s.imgs_per_s), 0.9) }];
    }
  });

  // LR ticks share one power of ten (stated in the header): "2" "4" with "× 10⁻³" instead of "2.00e-3".
  const lrExp = $derived.by(() => {
    void version;
    let m = 0;
    for (const s of steps) if (s.lr > m) m = s.lr;
    return m > 0 ? Math.floor(Math.log10(m)) : 0;
  });
  const ipsMax = $derived.by(() => {
    let m = 1;
    if (tab === 'ips') for (const p of series[0]?.points ?? []) if (p.y > m) m = p.y;
    return m;
  });

  const best = $derived<Marker[]>(
    run.best && (tab === 'map' || tab === 'map50')
      ? [{ x: run.best.epoch + 1, y: tab === 'map' ? run.best.map50_95 : (epochs.find((e) => e.epoch === run.best!.epoch)?.val.map50 ?? 0), label: 'best', color: 'var(--accent-line)', quiet: true }]
      : [],
  );

  interface TabDef {
    id: Tab;
    label: string;
    value: string;
    unit?: string;
    caption: string;
    title: string;
    qualifier: string;
  }
  const tabs = $derived.by((): TabDef[] => [
    {
      id: 'map',
      label: 'mAP50-95',
      value: metric(lastEpoch?.val.map50_95),
      caption: run.best ? `best ${metric(run.best.map50_95)} · ep ${run.best.epoch + 1}` : 'after the first epoch',
      title: 'Validation mAP50-95',
      qualifier: run.best ? `by epoch · best ${metric(run.best.map50_95)} at epoch ${run.best.epoch + 1} (ring)` : 'by epoch',
    },
    { id: 'map50', label: 'mAP50', value: metric(lastEpoch?.val.map50), caption: 'last epoch', title: 'Validation mAP50', qualifier: 'by epoch' },
    {
      id: 'pr',
      label: 'Precision · Recall',
      value: lastEpoch ? `${num(lastEpoch.val.precision, 2)} · ${num(lastEpoch.val.recall, 2)}` : '—',
      caption: 'last epoch',
      title: 'Precision and recall',
      qualifier: 'by epoch',
    },
    {
      id: 'loss',
      label: 'Loss',
      value: num(lastStep?.loss.total, 3),
      caption: 'last step, raw',
      title: 'Training loss',
      qualifier: smoothing === '0' ? 'by epoch · raw per step' : `by epoch · EMA ${smoothing}`,
    },
    {
      id: 'lr',
      label: 'Learning rate',
      value: fmtLr(lastStep?.lr),
      caption: 'warmup 3 · cosine',
      title: 'Learning rate',
      qualifier: `by epoch · × ${pow10(lrExp)}`,
    },
    {
      id: 'ips',
      label: 'Throughput',
      value: num(lastStep?.imgs_per_s, 0),
      unit: 'img/s',
      caption: 'last step',
      title: 'Throughput',
      qualifier: 'images/s by epoch',
    },
  ]);
  const current = $derived(tabs.find((t) => t.id === tab)!);

  function onkeydown(e: KeyboardEvent) {
    const i = tabs.findIndex((t) => t.id === tab);
    let j = i;
    if (e.key === 'ArrowRight') j = (i + 1) % tabs.length;
    else if (e.key === 'ArrowLeft') j = (i - 1 + tabs.length) % tabs.length;
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = tabs.length - 1;
    else return;
    e.preventDefault();
    tab = tabs[j]!.id;
    (e.currentTarget as HTMLElement).querySelector<HTMLElement>(`[data-tab="${tab}"]`)?.focus();
  }
</script>

<section class="metrics" aria-label="Run metrics">
  <div class="strip">
    <div class="epoch">
      <span class="label">Epoch</span>
      <span class="big num">{run.epoch}<span class="of">/ {run.epochs}</span></span>
      <div
        class="track"
        aria-hidden="true"
        use:tooltip={closeFrom < 1 && live ? `Hatched: the last ${run.config.augment.close_mosaic} epochs train without mosaic` : ''}
      >
        <div class="fill" class:done={run.status === 'completed'} class:bad={run.status === 'failed'} style:transform="scaleX({run.status === 'completed' ? 1 : progress})"></div>
        {#if closeFrom < 1 && live}<div class="close" style:left="{closeFrom * 100}%"></div>{/if}
      </div>
      {#if closeFrom < 1 && live}<span class="sr-only">The hatched end of the bar marks the last {run.config.augment.close_mosaic} epochs, trained without mosaic.</span>{/if}
      <span class="cap num">
        {#if live}{(progress * 100).toFixed(1)}% · {run.eta_s !== null ? `${duration(run.eta_s)} left` : 'estimating…'}{#if closeFrom < 1}<span class="hatch-key"> · <i aria-hidden="true"></i>no mosaic</span>{/if}{:else}{elapsed !== null ? `${duration(elapsed)} total` : '—'}{/if}
      </span>
    </div>
    <!-- Phones: one picker instead of six cells; the chart below shows what it names. -->
    <div class="phone-metric">
      <Select
        label="Metric to chart"
        bind:value={tab}
        options={tabs.map((t) => ({ value: t.id, label: `${t.label} · ${t.value}${t.unit ? ` ${t.unit}` : ''}` }))}
      />
    </div>
    <!-- svelte-ignore a11y_interactive_supports_focus -->
    <div class="tabs" role="tablist" aria-label="Metric to chart" {onkeydown}>
      {#each tabs as t (t.id)}
        <button
          role="tab"
          class="tab"
          class:on={tab === t.id}
          aria-selected={tab === t.id}
          aria-controls="run-chart"
          tabindex={tab === t.id ? 0 : -1}
          data-tab={t.id}
          onclick={() => (tab = t.id)}
        >
          <span class="label">{t.label}</span>
          <span class="val num">{t.value}{#if t.unit}<span class="unit">{t.unit}</span>{/if}</span>
          <span class="cap num">{t.caption}</span>
        </button>
      {/each}
    </div>
  </div>

  <div class="chart" id="run-chart" role="tabpanel" aria-label={current.title}>
    <header>
      <h2><span class="t">{current.title}</span><span class="q">{current.qualifier}</span></h2>
      <div class="ctl">
        {#if tab === 'loss'}
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
        {/if}
        {#if live}
          <Segmented
            size="sm"
            label="Chart range"
            bind:value={horizon}
            options={[
              { value: 'observed', label: 'So far' },
              { value: 'full', label: `All ${run.epochs}` },
            ]}
          />
        {/if}
      </div>
    </header>
    {#if loading}
      <ChartSkeleton height={narrow ? 220 : 320} />
    {:else}
      {#key tab}
        <LineChart
          {series}
          height={narrow ? 220 : 320}
          {xDomain}
          yDomain={tab === 'map' || tab === 'map50' || tab === 'pr' ? [0, 1] : tab === 'ips' ? [0, ipsMax * 1.15] : undefined}
          yLog={tab === 'loss' && scale === 'log'}
          yFormat={tab === 'lr' ? (v) => String(+(v / 10 ** lrExp).toFixed(2)) : undefined}
          valueFormat={tab === 'lr' ? (v) => fmtLr(v) : tab === 'ips' ? (v) => `${num(v, 0)} img/s` : (v) => v.toFixed(3)}
          xTitle={epochTitle}
          curve={tab === 'map' || tab === 'map50' || tab === 'pr' ? 'monotone' : 'linear'}
          markers={best}
          legendNote={live ? 'Latest' : 'Last'}
          label="{current.title} by epoch"
          empty={tab === 'map' || tab === 'map50' || tab === 'pr' ? 'Waiting for the first validation…' : 'Waiting for the first step…'}
        />
      {/key}
    {/if}
  </div>
</section>

<style>
  .metrics {
    background: var(--bg-panel);
    border: 1px solid var(--line-default);
    border-radius: var(--radius-lg);
    min-width: 0;
  }
  :global([data-theme='dark']) .metrics {
    border-color: var(--line-subtle);
  }
  .strip {
    display: grid;
    grid-template-columns: 240px minmax(0, 1fr);
    border-bottom: 1px solid var(--line-subtle);
  }
  .epoch {
    display: grid;
    grid-template-rows: 16px 40px auto 16px;
    gap: 6px;
    padding: var(--space-4);
    border-right: 1px solid var(--line-subtle);
  }
  .tabs {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    min-width: 0;
  }
  .tab {
    position: relative;
    display: grid;
    grid-template-rows: 16px 40px 16px;
    gap: 6px;
    align-content: start;
    padding: var(--space-4);
    text-align: left;
    min-width: 0;
    border-left: 1px solid var(--line-subtle);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .tab:first-child {
    border-left: 0;
  }
  .tab:hover {
    background: var(--bg-hover);
  }
  /* The selected metric: an ink bar on the edge it shares with its chart. */
  .tab::after {
    content: '';
    position: absolute;
    left: 0;
    right: 0;
    bottom: -1px;
    height: 2px;
    background: var(--text-primary);
    transform: scaleX(0);
    transition: transform var(--dur-base) var(--ease-out);
  }
  .tab.on::after {
    transform: scaleX(1);
  }
  .tab:focus-visible {
    outline: 2px solid var(--focus);
    outline-offset: -2px;
  }
  .label {
    font-size: var(--type-sm-size);
    line-height: 16px;
    color: var(--text-tertiary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .phone-metric {
    display: none;
  }
  .tab.on .label {
    color: var(--text-primary);
  }
  .big {
    align-self: end;
    font-size: var(--type-3xl-size);
    line-height: 40px;
    letter-spacing: var(--type-3xl-tracking);
    font-weight: var(--weight-semibold);
  }
  .of {
    margin-left: 6px;
    font-size: var(--type-lg-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    letter-spacing: 0;
  }
  .val {
    align-self: end;
    font-size: var(--type-2xl-size);
    line-height: 32px;
    letter-spacing: var(--type-2xl-tracking);
    font-weight: var(--weight-semibold);
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .tab.on .val {
    color: var(--text-primary);
  }
  .unit {
    margin-left: 4px;
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    letter-spacing: 0;
  }
  .cap {
    font-size: var(--type-sm-size);
    line-height: 16px;
    color: var(--text-tertiary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .track {
    position: relative;
    align-self: center;
    height: 4px;
    border-radius: 2px;
    background: var(--bg-active);
    overflow: hidden;
  }
  .fill {
    position: absolute;
    inset: 0;
    transform-origin: left;
    background: var(--accent-line);
    border-radius: 2px;
    transition: transform var(--dur-slow) var(--ease-standard);
  }
  .fill.done {
    background: var(--text-secondary);
  }
  .fill.bad {
    background: var(--danger-solid);
  }
  /* A key for the hatched end of the bar, in the caption beside it. */
  .hatch-key i {
    display: inline-block;
    width: 10px;
    height: 6px;
    margin: 0 4px 0 2px;
    vertical-align: 1px;
    border-radius: 1px;
    background: repeating-linear-gradient(135deg, transparent 0 2px, var(--line-strong) 2px 3px);
    box-shadow: inset 0 0 0 1px var(--line-strong);
  }
  .close {
    position: absolute;
    top: 0;
    bottom: 0;
    right: 0;
    background: repeating-linear-gradient(135deg, transparent 0 3px, var(--line-strong) 3px 4px);
    border-left: 1px solid var(--line-strong);
  }
  .chart {
    display: grid;
    gap: var(--space-2);
    padding: 0 var(--space-4) var(--space-4);
    min-width: 0;
  }
  .chart header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    min-height: 48px;
    flex-wrap: wrap;
    padding-top: var(--space-2);
  }
  h2 {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
    min-width: 0;
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    font-weight: var(--weight-medium);
  }
  .q {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .ctl {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex-wrap: wrap;
  }
  @media (max-width: 1199px) {
    .strip {
      grid-template-columns: minmax(0, 1fr);
    }
    .epoch {
      grid-template-columns: auto minmax(0, 1fr) auto;
      grid-template-rows: auto;
      align-items: center;
      column-gap: var(--space-4);
      border-right: 0;
      border-bottom: 1px solid var(--line-subtle);
      padding: var(--space-3) var(--space-4);
    }
    .epoch .label {
      display: none;
    }
    .big {
      font-size: var(--type-2xl-size);
      line-height: 32px;
    }
    .tabs {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .tab:nth-child(4) {
      border-left: 0;
    }
    .tab:nth-child(n + 4) {
      border-top: 1px solid var(--line-subtle);
    }
  }
  @media (max-width: 767px) {
    /* Phones: the tabs become one scrolling row; the chart follows directly. */
    .epoch {
      grid-template-columns: auto minmax(0, 1fr);
    }
    .epoch .cap {
      grid-column: 2;
    }
    .epoch .track {
      grid-column: 2;
      grid-row: 1;
    }
    .big {
      grid-row: 1 / span 2;
    }
    .tabs {
      display: none;
    }
    .phone-metric {
      display: grid;
      gap: var(--space-1);
      padding: var(--space-3);
    }
    .val {
      font-size: var(--type-lg-size);
      line-height: 28px;
    }
    .chart {
      padding: 0 var(--space-3) var(--space-3);
    }
    /* The picker above already names the metric: the header keeps only its qualifier (best value and epoch). */
    h2 .t {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip-path: inset(50%);
      white-space: nowrap;
    }
    h2 {
      font-size: var(--type-sm-size);
      font-weight: var(--weight-regular);
    }
    .q {
      white-space: normal;
    }
  }
</style>
