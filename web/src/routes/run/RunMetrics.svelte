<script lang="ts" module>
  import { lr as fmtLr } from '$lib/format';

  /** "3.42" and "× 10⁻⁴" from 3.42e-4: the value set large, its power of ten as the unit, so a tab never cuts it off. */
  export function lrParts(v: number | null | undefined): { value: string; unit?: string } {
    const [m, e] = fmtLr(v).split(' × ');
    return e ? { value: m!, unit: `× ${e}` } : { value: m! };
  }
</script>

<script lang="ts">
  /**
   * The run's metrics as one instrument: the epoch progress, a row of KPI tabs (each with its current value) and the one
   * chart the selected tab names. A single run is drawn in ink (primary metric in text.primary, secondary ones
   * tertiary with dash patterns); hue is reserved for entities (runs in Compare, classes).
   */
  import { tooltip } from '$lib/ui/tooltip';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import ChartSkeleton from '$lib/charts/ChartSkeleton.svelte';
  import { BestPulse } from '$lib/charts/bestPulse';
  import LossOptions, { type Scale, type Smoothing } from '$lib/charts/LossOptions.svelte';
  import { holdDomain, logDomain, niceDomain, unitDomain, type Fit, type Held } from '$lib/charts/domain';
  import { smooth, type Pt } from '$lib/charts/downsample';
  import LineChart, { type Marker, type Series } from '$lib/charts/LineChart.svelte';
  import { runHorizon } from '$lib/charts/horizon';
  import type { Epoch, Run, Step } from '$lib/api/types';
  import { runProgress, stepsPerEpoch } from '$lib/api/progress';
  import { duration, eta, metric, num, pow10 } from '$lib/format';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';

  let {
    run,
    steps,
    epochs,
    version,
    loading,
    queued = false,
    folded = false,
  }: {
    run: Run;
    steps: Step[];
    epochs: Epoch[];
    version: number;
    loading: boolean;
    /**
     * A queued run with nothing yet: the same frame as a running run (KPI tabs reading "—", the empty plot); on phones a
     * one-line "Metrics appear when training starts" that opens it.
     */
    queued?: boolean;
    /** Phones: another section is picked, so the chart folds to its one-line readout (it opens on a tap). */
    folded?: boolean;
  } = $props();

  type Tab = 'map' | 'map50' | 'pr' | 'loss' | 'lr' | 'ips';
  let tab = $state<Tab>('map');

  // Below 960 px the six tabs become one metric picker in the chart header, where it replaces the title; phones also
  // get a shorter plot, so the samples start within the first screen.
  const media = (q: string) => typeof matchMedia !== 'undefined' && matchMedia(q).matches;
  let narrow = $state(media('(max-width: 767px)'));
  let picking = $state(media('(max-width: 959px)'));
  $effect(() => {
    const phone = matchMedia('(max-width: 767px)');
    const tablet = matchMedia('(max-width: 959px)');
    const on = () => ((narrow = phone.matches), (picking = tablet.matches));
    phone.addEventListener('change', on);
    tablet.addEventListener('change', on);
    return () => (phone.removeEventListener('change', on), tablet.removeEventListener('change', on));
  });
  // Phones: a queued run's plot is the axes and one line, folded until asked for.
  const height = $derived(queued && narrow ? 120 : narrow ? 168 : 320);
  let queuedOpen = $state(false);
  const queuedFold = $derived(queued && narrow && !queuedOpen);
  // Folded on phones while another section is picked; a tap on the readout opens it until the section changes.
  let peek = $state(false);
  $effect(() => {
    void folded;
    peek = false;
  });
  const fold = $derived(folded && narrow && !peek && !queued);

  const live = $derived(['queued', 'running', 'stopping'].includes(run.status));
  const progress = $derived(runProgress(run));
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
  let smoothing = $state<Smoothing>('0.9');
  // Log by default: the total starts near 8 while box / cls / dfl sit between 0.4 and 3, which a linear axis squashes
  // into its bottom quarter.
  let scale = $state<Scale>('log');
  let horizon = $state<'observed' | 'full'>('observed');
  const observed = $derived.by(() => {
    void version;
    const spe = Math.max(1, stepsPerEpoch(run, steps));
    return Math.max(lastStep ? lastStep.step / spe : 0, lastEpoch ? lastEpoch.epoch : 0);
  });
  const xDomain = $derived(runHorizon(observed, run.epochs, live, horizon));
  const epochTitle = (x: number) => `Epoch ${x.toFixed(x % 1 === 0 ? 0 : 1)}`;

  const INK = 'var(--text-primary)';
  const INK2 = 'var(--text-tertiary)';
  const ep = (f: (e: Epoch) => number): Pt[] => epochs.map((e) => ({ x: e.epoch, y: f(e) }));
  const perStep = (f: (s: Step) => number): Pt[] => {
    const spe = Math.max(1, stepsPerEpoch(run, steps));
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
          // Said in the legend, so the smoothed total never seems to contradict the KPI's raw last step.
          { id: 'total', label: a > 0 ? `Total, smoothed (EMA ${smoothing})` : 'Total, raw', endLabel: 'total', color: INK, emphasis: true, points: smooth(total, a) },
          // The components as design/README.md maps them in both apps: box amber, cls violet dashed, dfl teal dotted.
          // Box and dfl run together late in a run (0.85 vs 0.83): box is painted over dfl's dots, and each line is
          // labelled at its end, so both stay readable where they meet.
          { id: 'box', label: 'box', color: 'var(--chart-2)', front: true, points: smooth(perStep((s) => s.loss.box), a) },
          { id: 'cls', label: 'cls', color: 'var(--chart-5)', pattern: 'dash', points: smooth(perStep((s) => s.loss.cls), a) },
          { id: 'dfl', label: 'dfl', color: 'var(--chart-4)', pattern: 'dot', points: smooth(perStep((s) => s.loss.dfl), a) },
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

  // ---- y domain ----
  // Fitted to the data and "nice" (it ends on a gridline): a failed run's mAP of 0.5 fills the plot instead of half
  // of a 0–1 axis. A live chart re-fits at most once per epoch (./domain holdDomain), and values never tween.
  const unit = $derived(tab === 'map' || tab === 'map50' || tab === 'pr');
  // The plot inside LineChart's margins (8 px top, 22 px bottom).
  const plotPx = $derived(height - 30);
  let held: Held<Fit & { log?: boolean }> | null = null;
  const fit = $derived.by((): (Fit & { log?: boolean }) | null => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of series)
      for (const p of s.points) {
        if (!Number.isFinite(p.y) || (tab === 'loss' && scale === 'log' && p.y <= 0)) continue;
        if (p.y < lo) lo = p.y;
        if (p.y > hi) hi = p.y;
      }
    if (!Number.isFinite(lo)) return null;
    let next: Fit & { log?: boolean };
    if (unit) next = unitDomain(hi, plotPx);
    else if (tab === 'loss' && scale === 'log') next = { domain: logDomain(lo, hi), ticks: [], log: true };
    // Throughput starts at 0 (a rate), with a little headroom over its peak.
    else if (tab === 'ips') next = niceDomain(0, hi * 1.05);
    else next = niceDomain(lo, hi);
    const epochNow = live ? Math.floor(observed) : -1;
    held = holdDomain(held, `${tab}|${scale}|${smoothing}|${height}|${epochNow}`, next);
    return held.value;
  });

  // A new best while the run trains pulses its ring once (LineChart, over the chart duration; nothing moves under reduced motion).
  const pulser = new BestPulse();
  let pulse = $state(0);
  $effect.pre(() => {
    pulse = pulser.see(run.best?.epoch ?? null, run.status === 'running');
  });
  const best = $derived<Marker[]>(
    run.best && (tab === 'map' || tab === 'map50')
      ? [
          {
            x: run.best.epoch,
            y: tab === 'map' ? run.best.map50_95 : (epochs.find((e) => e.epoch === run.best!.epoch)?.val.map50 ?? 0),
            label: 'best',
            color: 'var(--accent-line)',
            quiet: true,
            pulse,
          },
        ]
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
    /** The value is the last step's raw reading (the chart may draw it smoothed): the phone picker says so. */
    raw?: boolean;
  }
  const tabs = $derived.by((): TabDef[] => [
    {
      id: 'map',
      label: 'mAP50-95',
      value: metric(lastEpoch?.val.map50_95),
      caption: run.best ? `best ${metric(run.best.map50_95)} · ep ${run.best.epoch}` : 'after the first epoch',
      title: 'Validation mAP50-95',
      qualifier: run.best ? `by epoch · ○ best ${metric(run.best.map50_95)} at epoch ${run.best.epoch}` : 'by epoch',
    },
    { id: 'map50', label: 'mAP50', value: metric(lastEpoch?.val.map50), caption: 'last epoch', title: 'Validation mAP50', qualifier: 'by epoch' },
    {
      id: 'pr',
      label: 'Precision · Recall',
      value: lastEpoch ? `${metric(lastEpoch.val.precision)} · ${metric(lastEpoch.val.recall)}` : '—',
      caption: 'last epoch',
      title: 'Precision and recall',
      qualifier: 'by epoch',
    },
    {
      id: 'loss',
      label: 'Loss',
      value: num(lastStep?.loss.total, 3),
      caption: 'last step, raw',
      raw: true,
      title: 'Training loss',
      qualifier: 'by epoch',
    },
    {
      id: 'lr',
      label: 'Learning rate',
      // The power of ten is the value's unit, set smaller, so the tab never cuts the value off.
      ...lrParts(lastStep?.lr),
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

{#snippet picker()}
  <Select
    label="Metric to chart"
    bind:value={tab}
    options={tabs.map((t) => ({ value: t.id, label: `${t.label} · ${t.value}${t.unit && t.value !== '—' ? ` ${t.unit}` : ''}${t.raw && t.value !== '—' ? ' raw' : ''}` }))}
  />
{/snippet}

<section class="metrics" aria-label="Run metrics">
  {#if queued && narrow}
    <!-- Phones, queued: one line until there is something to chart; the banner above says when it starts. -->
    <button type="button" class="pending" aria-expanded={queuedOpen} aria-controls="run-metrics-body" onclick={() => (queuedOpen = !queuedOpen)}>
      <span class="p-t">Metrics appear when training starts</span>
      <span class="r-a">{queuedOpen ? 'Hide chart' : 'Show chart'}<ChevronDown size={14} aria-hidden="true" /></span>
    </button>
  {/if}
  {#if !queuedFold}
  <div class="metrics-body" id="run-metrics-body">
  <!-- Phones: a finished run's strip only repeats the completed banner above it (status, duration), so it goes. -->
  <div class="strip" class:finished={run.status === 'completed'}>
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
        {#if run.status === 'queued'}Not started{#if closeFrom < 1}<span class="hatch-key">&nbsp;· <i aria-hidden="true"></i>no mosaic</span>{/if}{:else if live}{(progress * 100).toFixed(1)}% · {run.eta_s !== null ? `${eta(run.eta_s)} left` : 'estimating…'}{#if closeFrom < 1}<span class="hatch-key">&nbsp;· <i aria-hidden="true"></i>no mosaic</span>{/if}{:else}{elapsed !== null ? `${duration(elapsed)} total` : '—'}{/if}
      </span>
    </div>
    <!-- Queued: the tabs read "—" until the first step, so the frame is the one a running run fills. -->
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
            <span class="val num">{t.value}{#if t.unit && t.value !== '—'}<span class="unit">{t.unit}</span>{/if}</span>
            <span class="cap num">{t.caption}</span>
          </button>
        {/each}
      </div>
  </div>

  <div class="chart" class:folded={fold} id="run-chart" role="tabpanel" aria-label={current.title}>
    {#if folded && narrow && !queued}
      <!-- Phones, another section picked: the chart as its one-line readout, which opens it in place. -->
      <button type="button" class="readout" aria-expanded={!fold} aria-controls="run-chart-body" onclick={() => (peek = !peek)}>
        <span class="r-l">{current.label}</span>
        <span class="r-v num">{current.value}{#if current.unit && current.value !== '—'}<span class="unit">{current.unit}</span>{/if}</span>
        <span class="r-c">{current.caption}</span>
        <span class="r-a">{fold ? 'Show chart' : 'Hide chart'}<ChevronDown size={14} aria-hidden="true" /></span>
      </button>
    {/if}
    {#if !fold}
      <div class="chart-body" id="run-chart-body">
        <header>
          <!-- Below 960 px the picker names the metric in place of the title (the tabs above are hidden). -->
          {#if picking}<div class="pick">{@render picker()}</div>{/if}
          <h2><span class="t">{current.title}</span><span class="q">{current.qualifier}</span></h2>
          <div class="ctl">
            <!-- One control for the loss view (smoothing and y scale behind one button), so the header keeps at most one
                 segmented group. -->
            {#if tab === 'loss'}<LossOptions bind:smoothing bind:scale />{/if}
            {#if live && run.status !== 'queued'}
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
          <ChartSkeleton {height} />
        {:else}
          {#key tab}
            <LineChart
              {series}
              {height}
              {xDomain}
              xInteger
              yDomain={fit?.domain}
              yTickValues={fit && !fit.log ? fit.ticks : undefined}
              yLog={tab === 'loss' && scale === 'log'}
              yFormat={tab === 'lr' ? (v) => String(+(v / 10 ** lrExp).toFixed(2)) : undefined}
              valueFormat={tab === 'lr' ? (v) => fmtLr(v) : tab === 'ips' ? (v) => `${num(v, 0)} img/s` : (v) => metric(v)}
              xTitle={epochTitle}
              curve={unit ? 'monotone' : 'linear'}
              markers={best}
              legendNote={live ? 'Latest' : 'Last'}
              label="{current.title} by epoch"
              endLabels={tab === 'loss'}
              empty={queued ? 'Curves appear here once training starts' : unit ? 'Waiting for the first validation…' : 'Waiting for the first step…'}
            />
          {/key}
        {/if}
      </div>
    {/if}
  </div>
  </div>
  {/if}
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
  /* Each tab is at least as wide as its value and caption, so no value is ever cut off; spare width is shared. */
  .tabs {
    display: grid;
    grid-template-columns: repeat(6, minmax(max-content, 1fr));
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
  .chart-body {
    display: grid;
    gap: var(--space-2);
    min-width: 0;
  }
  /* Phones, queued: the frame as one 48 px line that opens it. */
  .pending {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    width: 100%;
    min-height: 48px;
    padding: var(--space-1) var(--space-3);
    text-align: left;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    border-radius: var(--radius-lg);
  }
  .pending[aria-expanded='true'] {
    border-bottom: 1px solid var(--line-subtle);
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
  }
  .pending:focus-visible {
    outline-offset: -2px;
  }
  .p-t {
    color: var(--text-secondary);
  }
  .pending:hover .r-a {
    color: var(--text-primary);
  }
  .pending[aria-expanded='true'] .r-a :global(svg) {
    transform: rotate(180deg);
  }
  /* Phones, another section picked: the chart folded to one 44 px readout row that opens it. */
  .chart.folded {
    padding-bottom: var(--space-1);
  }
  .readout {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 100%;
    min-height: 44px;
    margin-top: var(--space-1);
    text-align: left;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    border-radius: var(--radius-md);
  }
  .readout:focus-visible {
    outline-offset: -2px;
  }
  .r-l {
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .r-v {
    color: var(--text-primary);
    font-size: var(--type-base-size);
    font-weight: var(--weight-semibold);
    white-space: nowrap;
  }
  .r-c {
    flex: 1;
    min-width: 0;
    color: var(--text-tertiary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .r-a {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    color: var(--text-secondary);
    white-space: nowrap;
  }
  .readout:hover .r-a {
    color: var(--text-primary);
  }
  .r-a :global(svg) {
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .readout[aria-expanded='true'] .r-a :global(svg) {
    transform: rotate(180deg);
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
  @media (min-width: 1200px) and (max-width: 1299px) {
    /* The narrow end of the wide tier: a slimmer epoch cell and tab sides keep every value whole beside it; a
       scroll is the fallback for values wider than the mock's. */
    .strip {
      grid-template-columns: 200px minmax(0, 1fr);
    }
    .tabs {
      overflow-x: auto;
      scrollbar-width: thin;
    }
    .tab {
      padding-inline: var(--space-3);
    }
    .tab::after {
      bottom: 0;
    }
  }
  @media (max-width: 1199px) {
    /* Laptops: the epoch becomes a one-line row over the six tabs, which keep their values in one row (scrolling
       sideways only if a narrow window cannot fit them). */
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
    /* A window too narrow for all six scrolls them sideways rather than cutting a value off. */
    .tabs {
      overflow-x: auto;
      scrollbar-width: thin;
    }
    .tab {
      grid-template-rows: 16px 32px 16px;
      gap: var(--space-1);
      padding: var(--space-3) var(--space-4);
    }
    .tab::after {
      bottom: 0;
    }
  }
  @media (max-width: 1099px) {
    /* The low end of the tier: values a step smaller and tighter sides, so all six tabs fit at 960 px without
       scrolling. */
    .tab {
      padding: var(--space-3);
    }
    .val {
      font-size: var(--type-xl-size);
      line-height: 32px;
      letter-spacing: var(--type-xl-tracking);
    }
  }
  @media (max-width: 959px) {
    /* Below 960 px six tiles would push the chart down: one compact metric picker in the chart header instead, in
       place of the title (the qualifier stays beside it). */
    .tabs {
      display: none;
    }
    .epoch {
      border-bottom: 0;
    }
    .chart header {
      padding-top: var(--space-3);
      row-gap: var(--space-2);
    }
    .pick {
      flex: 0 1 280px;
      min-width: 0;
    }
    h2 {
      flex: 1 1 200px;
    }
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
  }
  @media (max-width: 767px) {
    /* Phones: "23 / 100" and its meta on one baseline, the bar under the whole row 8 px below; the chart follows. */
    .epoch {
      grid-template-columns: auto minmax(0, 1fr);
      align-items: baseline;
      row-gap: var(--space-2);
    }
    .big {
      grid-row: 1;
      grid-column: 1;
      align-self: baseline;
    }
    .epoch .cap {
      grid-row: 1;
      grid-column: 2;
      align-self: baseline;
      text-align: right;
    }
    .epoch .track {
      grid-row: 2;
      grid-column: 1 / -1;
      align-self: center;
    }
    .val {
      font-size: var(--type-lg-size);
      line-height: 28px;
    }
    .strip.finished {
      display: none;
    }
    .chart {
      padding: 0 var(--space-3) var(--space-3);
    }
    /* One row: the metric picker and the range control; the qualifier (best value and epoch) on a line below. */
    .chart header {
      min-height: 0;
    }
    .pick {
      flex: 1 1 160px;
      max-width: 240px;
    }
    .ctl {
      margin-left: auto;
    }
    h2 {
      order: 1;
      flex-basis: 100%;
    }
    .q {
      white-space: normal;
    }
  }
</style>
