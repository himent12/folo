<script lang="ts">
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import Download from '@lucide/svelte/icons/download';
  import ScanSearch from '@lucide/svelte/icons/scan-search';
  import { MediaQuery } from 'svelte/reactivity';
  import { hub } from '$lib/api/index.svelte';
  import { fromModelEval, reduceConfusion } from '$lib/api/evaluation';
  import { archInfo } from '$lib/api/mock/runs';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Dataset, Model, Run } from '$lib/api/types';
  import ConfusionMatrix from '$lib/charts/ConfusionMatrix.svelte';
  import { classInk } from '$lib/design/classInk.svelte';
  import { ago, bytes, compact, date, metric, num } from '$lib/format';
  import { prefs } from '$lib/prefs.svelte';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Button from '$lib/ui/Button.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import { cmIndex } from './model/confusion';
  import ExportDialog from './model/ExportDialog.svelte';
  import PrCurves from './model/PrCurves.svelte';

  const id = $derived(router.route.params.id ?? '');
  const model = new Resource((s) => hub.api.model(id, s));

  $effect(() => {
    setCrumbs({ label: 'Models', path: '/models' }, { label: model.data?.name ?? id });
  });

  // The source run and its dataset, for the provenance links.
  const runId = $derived(model.data?.run_id ?? null);
  const context = new Resource<{ run: Run | null; dataset: Dataset | null }>(async (s) => {
    const rid = runId;
    if (!rid) return { run: null, dataset: null };
    const run = await hub.api.run(rid, s).catch(() => null);
    const dataset = run ? await hub.api.dataset(run.config.dataset_id, s).catch(() => null) : null;
    return { run, dataset };
  });

  // PR curves, confusion matrix and per-class metrics from GET /api/models/{id}/eval.
  const evaluation = new Resource((s) => hub.api.modelEval(id, s));
  const ev = $derived(evaluation.data ? fromModelEval(evaluation.data) : null);

  // Large class sets (COCO) get a readable matrix of the most frequent classes.
  const CM_MAX = 12;
  const cm = $derived.by(() => {
    const m = model.data;
    if (!m || !ev) return null;
    const n = m.classes.length;
    if (n <= CM_MAX) return { classes: m.classes, ids: undefined, matrix: ev.confusion, partial: false };
    const keep = ev.perClass
      .map((c) => c)
      .sort((a, b) => b.instances - a.instances)
      .slice(0, CM_MAX)
      .map((c) => c.class)
      .sort((a, b) => a - b);
    return {
      classes: [...keep.map((i) => m.classes[i] ?? `class ${i}`), 'Other'],
      ids: [...keep, -1],
      matrix: reduceConfusion(ev.confusion, keep),
      partial: true,
    };
  });

  /**
   * The largest off-diagonal confusion between two real classes (background excluded), row-normalised over the full
   * matrix. Rows with very few instances are skipped so one stray box can't top the list.
   */
  const confused = $derived.by(() => {
    if (!ev) return null;
    const conf = ev.confusion;
    const k = conf.length - 1;
    let best: { r: number; c: number; v: number } | null = null;
    for (let r = 0; r < k; r++) {
      const row = conf[r] ?? [];
      const sum = row.reduce((a, b) => a + b, 0);
      if (sum < 20) continue;
      for (let c = 0; c < k; c++) {
        if (c === r) continue;
        const v = (row[c] ?? 0) / sum;
        if (v > 0 && (!best || v > best.v)) best = { r, c, v };
      }
    }
    return best;
  });

  /**
   * Missed objects: the share of real instances whose only match was background (the matrix's last column), overall
   * and for the class that misses most (rows with very few instances skipped, as above).
   */
  const missed = $derived.by(() => {
    if (!ev) return null;
    const conf = ev.confusion;
    const k = conf.length - 1;
    let total = 0;
    let miss = 0;
    let worst: { r: number; v: number } | null = null;
    for (let r = 0; r < k; r++) {
      const row = conf[r] ?? [];
      const sum = row.reduce((a, b) => a + b, 0);
      const m = row[k] ?? 0;
      total += sum;
      miss += m;
      if (sum >= 20 && m > 0 && (!worst || m / sum > worst.v)) worst = { r, v: m / sum };
    }
    return total ? { v: miss / total, worst } : null;
  });

  // The summary points into the matrix: each figure is a button that reveals its cell (on a phone the matrix scrolls
  // sideways, so the background column starts out of view).
  let matrixView = $state<ConfusionMatrix>();
  const confusedCell = $derived.by(() => {
    if (!cm || !confused) return null;
    const r = cmIndex(confused.r, cm.ids);
    const c = cmIndex(confused.c, cm.ids);
    return r === null || c === null ? null : { r, c };
  });
  const missedCell = $derived.by(() => {
    if (!cm || !missed?.worst) return null;
    const r = cmIndex(missed.worst.r, cm.ids);
    return r === null ? null : { r, c: cm.classes.length };
  });

  let exportOpen = $state(false);
  // Up to 1199 px the rail stacks under the PR panel: the details fold into one line so the matrix follows soon.
  const narrow = new MediaQuery('max-width: 1199px', false);
  let detailsOpen = $state(false);

  /** The architecture's published 640 px reference time, scaled by input area. */
  const latency = (m: Model) => archInfo[m.arch].ms * (m.imgsz / 640) ** 2;
  const kpis = (m: Model) => [
    { label: 'mAP50-95', value: m.metrics?.map50_95, sub: 'IoU 0.50:0.95', rest: ', all classes' },
    { label: 'mAP50', value: m.metrics?.map50, sub: 'IoU 0.50', rest: '' },
    { label: 'Precision', value: m.metrics?.precision, sub: 'boxes', rest: ', all classes' },
    { label: 'Recall', value: m.metrics?.recall, sub: 'boxes', rest: ', all classes' },
  ];
</script>

{#if model.error && !model.data}
  <Page title="Model">
    {#snippet actions()}<Button icon={ArrowLeft} href={href('/models')}>All models</Button>{/snippet}
    <Panel><ErrorState error={model.error} retry={() => model.refresh()} title="Could not load this model" compact /></Panel>
  </Page>
{:else if !model.data}
  <Page title="Model">
    {#snippet sub()}<Skeleton w="320px" h="14px" />{/snippet}
    <Skeleton h="106px" r="var(--radius-lg)" />
    <div class="detail"><Skeleton h="560px" r="var(--radius-lg)" /><Skeleton h="380px" r="var(--radius-lg)" /></div>
  </Page>
{:else}
  {@const m = model.data}
  {@const run = context.data?.run}
  <Page title={m.name}>
    {#snippet sub()}
      <span class="mono">{m.arch}</span>
      <span class="num">{m.classes.length} classes · {m.imgsz} px · {bytes(m.bytes, prefs.units)}</span>
      {#if m.run_id}
        <span>trained by <a class="link" href={href(`/runs/${m.run_id}`)}>{run?.name ?? m.run_id}</a></span>
      {:else}
        <span>imported</span>
      {/if}
      <span class="faint" title={date(m.created)}>created {ago(m.created)}</span>
    {/snippet}
    {#snippet actions()}
      <Button icon={Download} onclick={() => (exportOpen = true)}>Export</Button>
      <Button variant="primary" icon={ScanSearch} href={href(`/playground?model=${encodeURIComponent(m.id)}`)}>Try in Playground</Button>
    {/snippet}

    <!-- One strip of numbers in ink: the headline metric, then the ones that support it. Phones: the headline gets
         its own row, the other three a subordinate row sharing one qualifier line. -->
    <section class="kpis" aria-label="Validation metrics">
      {#each kpis(m) as k, i (k.label)}
        <div class="kpi" class:hero={i === 0}>
          <span class="k-label">{k.label}</span>
          <span class="k-value num">{metric(k.value)}</span>
          <span class="k-sub">{k.sub}<span class="k-rest">{k.rest}</span></span>
        </div>
      {/each}
      <p class="k-note">All four over all classes · mAP50 at IoU 0.50</p>
    </section>

    <!-- Evidence in the main column, reference in a sticky rail. -->
    <div class="detail">
      <div class="main-col">
        <Panel title="Precision–recall" meta="precision by recall · IoU 0.5" pad="none" id="pr">
          {#if ev}
            <PrCurves classes={m.classes} {ev} map50={m.metrics?.map50 ?? null} />
          {:else if evaluation.error}
            <ErrorState error={evaluation.error} retry={() => evaluation.refresh()} compact title="Could not load the evaluation" />
          {:else}
            <div class="sk"><Skeleton h="24px" w="200px" /><Skeleton h="300px" /></div>
          {/if}
        </Panel>
      </div>

      <aside class="rail" aria-label="Model details">
        {#snippet detailList(two: boolean)}
          <dl class="details" class:two id="model-details">
            <div class="kv">
              <dt>Architecture</dt>
              <dd class="mono">{m.arch}</dd>
            </div>
            <div class="kv">
              <dt>Input size</dt>
              <dd class="num">{m.imgsz} × {m.imgsz} px</dd>
            </div>
            <div class="kv">
              <dt>Parameters</dt>
              <dd class="num">{compact(m.params)} <span class="faint">({m.params.toLocaleString('en-US')})</span></dd>
            </div>
            <div class="kv">
              <dt>Compute</dt>
              <dd class="num">{num(m.flops_g, 1)} GFLOPs</dd>
            </div>
            <div class="kv">
              <dt>File size</dt>
              <dd class="num">{bytes(m.bytes, prefs.units)}</dd>
            </div>
            <div class="kv">
              <dt>Latency</dt>
              <dd class="latency">
                <span class="num">Reference ~{latency(m).toFixed(1)} ms</span>
                <!-- A scaled hint, not a measurement: say exactly what it assumes. -->
                <span class="faint">RTX 4080-class GPU · {m.imgsz} px · fp32</span>
              </dd>
            </div>
            <div class="kv wide">
              <dt>Classes</dt>
              <dd class="classes">
                {#each m.classes.slice(0, 10) as c, i (i)}
                  <span class="chip"><span class="sw" style:background={classInk(i)} aria-hidden="true"></span>{c}</span>
                {/each}
                {#if m.classes.length > 10}<span class="faint">+{m.classes.length - 10} more</span>{/if}
              </dd>
            </div>
            <div class="kv">
              <dt>Source</dt>
              <dd>
                {#if m.run_id}
                  <a class="link" href={href(`/runs/${m.run_id}`)}>{run?.name ?? m.run_id}</a>
                  {#if context.data?.dataset}<span class="faint">{' on '}</span><a class="link" href={href(`/datasets/${context.data.dataset.id}`)}>{context.data.dataset.name}</a>{/if}
                {:else}
                  <span class="muted">Imported .folo file</span>
                {/if}
              </dd>
            </div>
            <div class="kv">
              <dt>Model id</dt>
              <dd class="mono muted">{m.id}</dd>
            </div>
            <div class="kv">
              <dt>Created</dt>
              <dd>{date(m.created)}</dd>
            </div>
          </dl>
        {/snippet}
        {#if narrow.current}
          <Panel pad="none" id="details">
            <button type="button" class="d-line" aria-expanded={detailsOpen} aria-controls="model-details" onclick={() => (detailsOpen = !detailsOpen)}>
              <span class="d-sum num truncate">
                <span class="mono">{m.arch}</span><span class="sep" aria-hidden="true">·</span>{m.imgsz} px<span class="d-lat"
                  ><span class="sep" aria-hidden="true">·</span><span title="Reference latency: RTX 4080-class GPU, {m.imgsz} px, fp32"
                    >~{latency(m).toFixed(1)} ms ref.</span
                  ></span
                ><span class="sep" aria-hidden="true">·</span>{bytes(m.bytes, prefs.units)}
              </span>
              <span class="d-verb">Details <ChevronRight size={14} aria-hidden="true" /></span>
            </button>
            {#if detailsOpen}<div class="d-body">{@render detailList(true)}</div>{/if}
          </Panel>
        {:else}
          <Panel title="Details" id="details">{@render detailList(false)}</Panel>
        {/if}
      </aside>
    </div>
    <!-- Full width: the matrix grows with the class count, and the rail beside it would sit empty. -->
    <Panel title="Confusion matrix" meta={cm?.partial ? `top ${CM_MAX} of ${m.classes.length} classes` : 'row-normalised'} id="cm">
      {#if cm}
        <!-- Matrix left (square, so it stays compact), what to read from it on the right. Up to 1199 px the two
             consequential cells lead (a phone's matrix scrolls sideways, hiding the background column), and the
             reading notes follow the matrix. -->
        <div class="cm-layout">
          <div class="cm-main"><ConfusionMatrix bind:this={matrixView} classes={cm.classes} classIds={cm.ids} matrix={cm.matrix} label="Confusion matrix of {m.name}" sparseLabels minCell={32} scale /></div>
          {#if confused || missed}
            <dl class="cm-sum">
              {#if confused}
                <div>
                  <dt>Most confused</dt>
                  <dd>
                    {#snippet pair()}<b>{m.classes[confused.r]}</b> → <b>{m.classes[confused.c]}</b> <span class="num">{num(confused.v * 100, 1)}%</span>{/snippet}
                    {#if confusedCell}
                      {@const at = confusedCell}
                      <button type="button" class="go" title="Show in the matrix" onclick={() => matrixView?.reveal(at.r, at.c)}>{@render pair()}</button>
                    {:else}{@render pair()}{/if}
                  </dd>
                </div>
              {/if}
              {#if missed}
                <div>
                  <dt>Missed (background)</dt>
                  <dd>
                    {#if missed.worst}
                      {@const w = missed.worst}
                      {#snippet worst()}<b>{m.classes[w.r]}</b> <span class="num">{num(w.v * 100, 1)}%</span>{/snippet}
                      {#if missedCell}
                        {@const at = missedCell}
                        <button type="button" class="go" title="Show in the matrix" onclick={() => matrixView?.reveal(at.r, at.c)}>{@render worst()}</button>
                      {:else}{@render worst()}{/if}<span class="sep" aria-hidden="true">·</span>
                    {/if}<span
                      class="num">{num(missed.v * 100, 1)}%</span> overall
                  </dd>
                </div>
              {/if}
            </dl>
          {/if}
          <div class="cm-notes">
            <p>Rows are the true class, columns the prediction; each row sums to 100%. The diagonal is correct; the last
              column is missed objects and the last row false detections on background.</p>
            {#if cm.partial}
              <p>Top {CM_MAX} classes by instances; the other {m.classes.length - CM_MAX} are folded into Other.</p>
            {/if}
          </div>
        </div>
      {:else if evaluation.error}
        <p class="none">No evaluation available.</p>
      {:else}
        <Skeleton h="300px" />
      {/if}
    </Panel>
  </Page>
  <ExportDialog model={m} bind:open={exportOpen} />
{/if}

<style>
  .link {
    color: var(--text-primary);
  }
  .link:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
  }
  /* 1 px gaps over a hairline-coloured background draw the dividers in any column count. */
  .kpis {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1px;
    background: var(--line-subtle);
    border: 1px solid var(--line-default);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }
  :global([data-theme='dark']) .kpis {
    border-color: var(--line-subtle);
  }
  .kpi {
    display: grid;
    grid-template-rows: 16px 40px 16px;
    gap: 6px;
    align-content: start;
    padding: var(--space-4);
    background: var(--bg-panel);
    min-width: 0;
  }
  .k-label,
  .k-sub {
    font-size: var(--type-sm-size);
    line-height: 16px;
    color: var(--text-tertiary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .k-value {
    align-self: end;
    font-size: var(--type-2xl-size);
    line-height: 32px;
    letter-spacing: var(--type-2xl-tracking);
    font-weight: var(--weight-semibold);
  }
  .hero .k-label {
    color: var(--text-secondary);
  }
  .k-note {
    display: none;
  }
  .hero .k-value {
    font-size: var(--type-3xl-size);
    line-height: 40px;
    letter-spacing: var(--type-3xl-tracking);
  }
  .detail {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    gap: var(--space-4);
    align-items: start;
  }
  .main-col,
  .rail {
    display: grid;
    gap: var(--space-4);
    min-width: 0;
  }
  .rail {
    position: sticky;
    top: calc(var(--size-topbar) + var(--space-4));
  }
  .sk {
    display: grid;
    gap: var(--space-2);
    padding: 0 var(--space-4) var(--space-4);
  }
  .cm-main {
    grid-area: main;
    min-width: 0;
  }
  .cm-layout {
    display: grid;
    grid-template-columns: minmax(0, auto) minmax(220px, 320px);
    grid-template-rows: auto 1fr;
    grid-template-areas:
      'main sum'
      'main notes';
    gap: var(--space-4) var(--space-8);
    align-items: start;
    justify-content: start;
  }
  @media (max-width: 1199px) {
    .cm-layout {
      grid-template-columns: minmax(0, 1fr);
      grid-template-rows: none;
      grid-template-areas: 'sum' 'main' 'notes';
    }
  }
  .cm-sum {
    grid-area: sum;
    display: grid;
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .cm-sum > div {
    display: grid;
    grid-template-columns: 128px minmax(0, 1fr);
    gap: var(--space-3);
    /* The label sits on the figure's baseline, also when the figure is a taller touch target. */
    align-items: baseline;
  }
  .cm-sum dt {
    color: var(--text-tertiary);
  }
  .cm-sum dd {
    text-align: left;
    color: var(--text-secondary);
  }
  .cm-sum b,
  .cm-sum .num {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  /* A figure that points at its cell: reads as a link (ink, underline), the cell takes the selection stroke. */
  .go {
    display: inline;
    color: inherit;
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
    text-align: left;
  }
  .go:hover {
    text-decoration-color: currentColor;
  }
  @media (pointer: coarse) {
    /* 20 px of text + 12 px above and below: a 44 px target. */
    .go {
      display: inline-block;
      padding-block: var(--space-3);
    }
  }
  .cm-notes {
    grid-area: notes;
    display: grid;
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .none {
    color: var(--text-tertiary);
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
  .d-line {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    width: 100%;
    min-height: 44px;
    padding: 0 var(--space-4);
    border-radius: inherit;
    text-align: left;
  }
  .d-line:hover {
    background: var(--bg-hover);
  }
  .d-line:focus-visible {
    outline-offset: -2px;
  }
  .d-sum {
    flex: 1;
    min-width: 0;
    color: var(--text-secondary);
  }
  .d-sum .mono {
    color: var(--text-primary);
  }
  .sep {
    margin-inline: var(--space-2);
    color: var(--text-tertiary);
  }
  .d-verb {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    flex-shrink: 0;
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .d-verb :global(svg) {
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .d-line[aria-expanded='true'] .d-verb :global(svg) {
    transform: rotate(90deg);
  }
  .d-body {
    padding: var(--space-3) var(--space-4) var(--space-4);
    border-top: 1px solid var(--line-subtle);
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
  .latency {
    display: grid;
    gap: 2px;
  }
  .latency .faint {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
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
  @media (max-width: 1199px) {
    .detail {
      grid-template-columns: minmax(0, 1fr);
    }
    .rail {
      position: static;
    }
  }
  @media (max-width: 767px) {
    /* The reference latency stays in the opened list; the line keeps what fits. */
    .d-lat {
      display: none;
    }
    .details.two {
      grid-template-columns: minmax(0, 1fr);
    }
    .two .kv {
      grid-template-columns: 96px minmax(0, 1fr);
    }
    /* Headline row: label, then the value with its qualifier on the same baseline. */
    .kpis {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .kpi {
      grid-template-rows: 16px 24px;
      gap: var(--space-1);
      padding: var(--space-3);
    }
    .kpi.hero {
      grid-column: 1 / -1;
      grid-template-columns: auto minmax(0, 1fr);
      grid-template-rows: 16px 40px;
      column-gap: var(--space-3);
      align-items: baseline;
    }
    .hero .k-label {
      grid-column: 1 / -1;
    }
    .hero .k-value {
      line-height: 40px;
    }
    /* On the value's baseline: 40 px line, 32 px figures; 16 px line, 12 px text. */
    .hero .k-sub {
      align-self: end;
      padding-bottom: 6px;
    }
    .k-rest {
      display: none;
    }
    .k-value {
      font-size: var(--type-xl-size);
      line-height: 24px;
      letter-spacing: var(--type-xl-tracking);
    }
    .kpi:not(.hero) .k-sub {
      display: none;
    }
    .k-note {
      display: block;
      grid-column: 1 / -1;
      padding: var(--space-2) var(--space-3);
      background: var(--bg-panel);
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-tertiary);
    }
  }
</style>
