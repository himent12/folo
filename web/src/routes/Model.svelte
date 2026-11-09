<script lang="ts">
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import Download from '@lucide/svelte/icons/download';
  import Info from '@lucide/svelte/icons/info';
  import ScanSearch from '@lucide/svelte/icons/scan-search';
  import { tick } from 'svelte';
  import { MediaQuery } from 'svelte/reactivity';
  import { hub } from '$lib/api/index.svelte';
  import { fromModelEval, reduceConfusion } from '$lib/api/evaluation';
  import { e2eMs } from '$lib/benchmarks';
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
  import Menu from '$lib/ui/Menu.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import { cmIndex } from './model/confusion';
  import ExportDialog from './model/ExportDialog.svelte';
  import { origin } from '$lib/lineage';
  import PrCurves from './model/PrCurves.svelte';
  import { MODEL_SECTIONS, prView, sectionShown, type ModelSection } from './model/sections';

  const id = $derived(router.route.params.id ?? '');
  const model = new Resource((s) => hub.api.model(id, s));

  $effect(() => {
    setCrumbs({ label: 'Models', path: '/models' }, { label: model.data?.name ?? id });
  });

  // The source run, its dataset and the model it was fine-tuned from (its `weights`), for the provenance links.
  const runId = $derived(model.data?.run_id ?? null);
  const context = new Resource<{ run: Run | null; dataset: Dataset | null; parent: Model | null }>(async (s) => {
    const rid = runId;
    if (!rid) return { run: null, dataset: null, parent: null };
    const run = await hub.api.run(rid, s).catch(() => null);
    const dataset = run ? await hub.api.dataset(run.config.dataset_id, s).catch(() => null) : null;
    const from = origin(run?.config.weights);
    const parent = from.kind === 'model' ? await hub.api.model(from.id, s).catch(() => null) : null;
    return { run, dataset, parent };
  });
  const lineage = $derived(context.data?.run ? origin(context.data.run.config.weights, context.data.parent ? [context.data.parent] : []) : null);

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
  // Phones: after the headline metrics, one section at a time (as on the run page) instead of a 4,000 px column.
  const phone = new MediaQuery('max-width: 767px', false);
  let section = $state<ModelSection>('curves');
  const showing = (part: ModelSection) => sectionShown(phone.current, section, part);

  /** Points the matrix at a cell, first bringing the Confusion section up on a phone. */
  async function reveal(at: { r: number; c: number }) {
    if (phone.current && section !== 'confusion') {
      section = 'confusion';
      await tick();
    }
    matrixView?.reveal(at.r, at.c);
  }

  /**
   * The architecture's end-to-end time from docs/BENCHMARKS.md (640 px, one RTX 4080 SUPER), or undefined when not
   * measured. It is a benchmark of the architecture, never this model's own timing, and is labelled as such.
   */
  const latency = (m: Model): number | undefined => e2eMs[m.arch];
  const kpis = (m: Model) => [
    { label: 'mAP50-95', value: m.metrics?.map50_95, sub: 'IoU 0.50:0.95' },
    { label: 'mAP50', value: m.metrics?.map50, sub: 'IoU 0.50' },
    { label: 'Precision', value: m.metrics?.precision, sub: 'at the best-F1 confidence' },
    { label: 'Recall', value: m.metrics?.recall, sub: 'at the best-F1 confidence' },
  ];
  // What the four numbers mean, one tap away (the strip itself keeps to values and a short qualifier).
  let helpOpen = $state(false);
  // Phones: one summary line under the top bar; the rest of the provenance behind it.
  let metaOpen = $state(false);
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
  {#snippet lineageNote()}
    <!-- Where the weights started: the run's `weights`, a model (linked), an imported file, or nothing. -->
    {#if lineage?.kind === 'model'}
      {@const lid = lineage.id}
      <span>fine-tuned from <a class="link" href={href(`/models/${encodeURIComponent(lid)}`)}>{context.data?.parent?.name ?? lid}</a></span>
    {:else if lineage?.kind === 'file'}
      <span>fine-tuned from <span class="mono" title={lineage.path}>{lineage.label}</span></span>
    {:else if lineage?.kind === 'scratch'}
      <span>trained from scratch</span>
    {/if}
  {/snippet}
  {#snippet provenance()}
    {#if m.run_id}
      <span>trained by <a class="link" href={href(`/runs/${m.run_id}`)}>{run?.name ?? m.run_id}</a></span>
      {@render lineageNote()}
    {:else}
      <span>imported</span>
    {/if}
    <span class="faint" title={date(m.created)}>created {ago(m.created)}</span>
  {/snippet}
  <Page title={m.name} inlineActions={phone.current}>
    {#snippet sub()}
      {#if phone.current}
        <button type="button" class="meta-line num" aria-expanded={metaOpen} aria-controls="model-meta" onclick={() => (metaOpen = !metaOpen)}>
          <span class="truncate"><span class="mono">{m.arch}</span> · {m.classes.length} classes · {m.imgsz} px</span>
          <ChevronDown size={14} aria-hidden="true" />
        </button>
        {#if metaOpen}
          <div class="meta-more" id="model-meta">
            <span class="num">{bytes(m.bytes, prefs.units)}</span>
            {@render provenance()}
          </div>
        {/if}
      {:else}
        <span class="mono">{m.arch}</span>
        <span class="num">{m.classes.length} classes · {m.imgsz} px · {bytes(m.bytes, prefs.units)}</span>
        {@render provenance()}
      {/if}
    {/snippet}
    {#snippet actions()}
      {#if phone.current}
        <Button variant="primary" icon={ScanSearch} aria-label="Try in Playground" href={href(`/playground?model=${encodeURIComponent(m.id)}`)}>Try</Button>
        <Menu label="More model actions" items={[{ label: 'Export', icon: Download, onclick: () => (exportOpen = true) }]} />
      {:else}
        <Button icon={Download} onclick={() => (exportOpen = true)}>Export</Button>
        <Button variant="primary" icon={ScanSearch} href={href(`/playground?model=${encodeURIComponent(m.id)}`)}>Try in Playground</Button>
      {/if}
    {/snippet}

    <!-- One strip of numbers in ink: the headline metric, then the ones that support it, each with a short qualifier;
         the ⓘ opens what they mean. Phones: one row of four values, qualifiers in the help. -->
    <section class="kpis" aria-label="Validation metrics">
      {#each kpis(m) as k, i (k.label)}
        <div class="kpi" class:hero={i === 0}>
          <span class="k-label">{k.label}</span>
          <span class="k-value num">{metric(k.value)}</span>
          <span class="k-sub">{k.sub}</span>
        </div>
      {/each}
      <button type="button" class="k-help" aria-label="What these metrics mean" aria-expanded={helpOpen} aria-controls="metric-help" onclick={() => (helpOpen = !helpOpen)}
        ><Info size={14} aria-hidden="true" /></button
      >
      {#if helpOpen}
        <dl class="k-body" id="metric-help">
          <div>
            <dt>AP and mAP</dt>
            <dd>AP is the area under one class's precision–recall curve. mAP is the mean of AP over all {m.classes.length} classes; per-class tables show AP.</dd>
          </div>
          <div>
            <dt>50 and 50-95</dt>
            <dd>The overlap (IoU) a box needs with a true box to count as found: 0.50, or AP averaged over IoU 0.50 to 0.95 in 0.05 steps.</dd>
          </div>
          <div>
            <dt>Precision, recall</dt>
            <dd>At the one confidence threshold that maximises the class-mean F1 at IoU 0.50. The server does not report which threshold that was.</dd>
          </div>
          <div>
            <dt>Evaluated on</dt>
            <dd>
              {#if context.data?.dataset}The validation split of {context.data.dataset.name}{:else}The validation split of the model's dataset{/if}{ev
                ? ` (${num(ev.images)} images)`
                : ''}, at {m.imgsz} px with NMS IoU 0.7.
            </dd>
          </div>
        </dl>
      {/if}
    </section>

    {#if phone.current}
      <!-- Stays reachable under the top bar while a section scrolls. -->
      <div class="phone-pick">
        <Segmented label="Show" bind:value={section} options={MODEL_SECTIONS} />
      </div>
    {/if}

    <!-- Evidence in the main column, reference in a sticky rail. -->
    <div class="detail" class:off={!showing('curves') && !showing('classes') && !showing('details')}>
      <div class="main-col" class:off={!showing('curves') && !showing('classes')}>
        <Panel
          title={phone.current && section === 'classes' ? 'Per-class AP' : 'Precision–recall'}
          meta={phone.current && section === 'classes' ? `${m.classes.length} classes · IoU 0.5` : 'precision by recall · IoU 0.5'}
          pad="none"
          id="pr"
        >
          {#if ev}
            <PrCurves classes={m.classes} {ev} map50={m.metrics?.map50 ?? null} view={prView(phone.current, section)} />
          {:else if evaluation.error}
            <ErrorState error={evaluation.error} retry={() => evaluation.refresh()} compact title="Could not load the evaluation" />
          {:else}
            <div class="sk"><Skeleton h="24px" w="200px" /><Skeleton h="300px" /></div>
          {/if}
        </Panel>
      </div>

      <aside class="rail" class:off={!showing('details')} aria-label="Model details">
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
              <dt>Benchmark</dt>
              <dd class="latency">
                <!-- The architecture's number (docs/BENCHMARKS.md), never this model's timing: say exactly what it is. -->
                {#if latency(m) !== undefined}
                  <span class="num">{latency(m)!.toFixed(2)} ms</span>
                  <span class="faint">{m.arch} benchmark (640 px), end to end on an RTX 4080 SUPER; not timed on this model</span>
                {:else}
                  <span class="faint">No {m.arch} benchmark yet</span>
                {/if}
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
            {#if lineage}
              <div class="kv">
                <dt>Started from</dt>
                <dd>
                  {#if lineage.kind === 'model'}
                    <a class="link" href={href(`/models/${encodeURIComponent(lineage.id)}`)}>{context.data?.parent?.name ?? lineage.id}</a>
                  {:else if lineage.kind === 'file'}
                    <span class="mono" title={lineage.path}>{lineage.label}</span>
                  {:else}
                    <span class="muted">Random weights (from scratch)</span>
                  {/if}
                </dd>
              </div>
            {/if}
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
        {#if phone.current}
          <!-- Its own section on phones: the whole list, no fold. -->
          <Panel title="Details" id="details">{@render detailList(true)}</Panel>
        {:else if narrow.current}
          <Panel pad="none" id="details">
            <button type="button" class="d-line" aria-expanded={detailsOpen} aria-controls="model-details" onclick={() => (detailsOpen = !detailsOpen)}>
              <span class="d-sum num truncate">
                <span class="mono">{m.arch}</span><span class="sep" aria-hidden="true">·</span>{m.imgsz} px<span class="sep" aria-hidden="true">·</span
                >{num(m.flops_g, 1)} GFLOPs<span class="sep" aria-hidden="true">·</span>{bytes(m.bytes, prefs.units)}
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
    <div class="cm-part" class:off={!showing('confusion')}>
      <Panel title="Confusion matrix" meta={cm?.partial ? (phone.current ? `${CM_MAX} most common + Other` : `${CM_MAX} most common classes + Other + background`) : 'share of each true class'} id="cm">
        {#if cm}
          <!-- The matrix at its own size (cells up to 72 px), what to read from it directly beside it, so the panel has
               no dead middle. Up to 1199 px the two consequential cells lead (a phone's matrix scrolls sideways,
               hiding the background column), and the reading notes follow the matrix. -->
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
                        <button type="button" class="go" title="Show in the matrix" onclick={() => reveal(at)}>{@render pair()}</button>
                      {:else}{@render pair()}{/if}
                    </dd>
                  </div>
                {/if}
                {#if missed}
                  <div>
                    <dt>Most missed</dt>
                    <dd>
                      {#if missed.worst}
                        {@const w = missed.worst}
                        {#snippet worst()}<b>{m.classes[w.r]}</b> <span class="num">{num(w.v * 100, 1)}%</span>{/snippet}
                        {#if missedCell}
                          {@const at = missedCell}
                          <button type="button" class="go" title="Show in the matrix" onclick={() => reveal(at)}>{@render worst()}</button>
                        {:else}{@render worst()}{/if}<span class="sep" aria-hidden="true">·</span>
                      {/if}<span
                        class="num">{num(missed.v * 100, 1)}%</span> overall
                    </dd>
                  </div>
                {/if}
              </dl>
            {/if}
            <div class="cm-notes">
              <p>Each row is one true class, split by what the model predicted for its objects. The diagonal is right;
                the background column is objects it missed.</p>
              <details class="how">
                <summary><ChevronDown size={14} aria-hidden="true" />How the shares are computed</summary>
                <div class="how-body">
                  <p>Each row sums to 100% of that class's ground-truth boxes at IoU 0.5. Cells under 5% are not
                    labelled; hover or tap one for its count. An empty cell had no objects at all.</p>
                  <p>The background row splits the false detections (boxes on no object) by the class they were given.</p>
                  {#if cm.partial}
                    <p>The {CM_MAX} classes with the most ground-truth boxes are shown by name; the other {m.classes.length - CM_MAX} are folded into Other.</p>
                  {/if}
                </div>
              </details>
            </div>
          </div>
        {:else if evaluation.error}
          <p class="none">No evaluation available.</p>
        {:else}
          <Skeleton h="300px" />
        {/if}
      </Panel>
    </div>
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
  .kpis {
    position: relative;
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
  /* The help toggle sits in the strip's top-right corner, over the last cell's empty label end. */
  .k-help {
    position: absolute;
    top: var(--space-3);
    right: var(--space-3);
    display: grid;
    place-items: center;
    width: 24px;
    height: 24px;
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
  }
  .k-help:hover,
  .k-help[aria-expanded='true'] {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .k-body {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-3) var(--space-8);
    padding: var(--space-4);
    background: var(--bg-panel);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .k-body > div {
    display: grid;
    grid-template-columns: 112px minmax(0, 1fr);
    gap: var(--space-3);
    max-width: 72ch;
  }
  .k-body dt {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .k-body dd {
    text-align: left;
    color: var(--text-secondary);
  }
  .meta-line {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    min-width: 0;
    max-width: 100%;
    min-height: var(--control-md);
    color: var(--text-secondary);
    text-align: left;
  }
  @media (pointer: coarse) {
    .meta-line {
      min-height: 44px;
    }
  }
  .meta-line :global(svg) {
    flex-shrink: 0;
    color: var(--text-tertiary);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .meta-line[aria-expanded='true'] :global(svg) {
    transform: rotate(180deg);
  }
  .meta-more {
    flex-basis: 100%;
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-3);
    margin-top: calc(var(--space-2) * -1);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .hero .k-value {
    font-size: var(--type-3xl-size);
    line-height: 40px;
    letter-spacing: var(--type-3xl-tracking);
  }
  /* The rail is as tall as its content and sticks under the top bar while the long class table scrolls past. */
  .detail {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    align-items: start;
    gap: var(--space-4);
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
  /* Phones: the sections the picker is not showing. */
  .cm-part {
    display: grid;
    min-width: 0;
  }
  /* After every display rule of the parts it hides. */
  .off {
    display: none;
  }
  /* As on the run page: full width, just under the top bar, segments sharing the width. */
  .phone-pick {
    position: sticky;
    top: calc(var(--size-topbar) + var(--space-2));
    z-index: var(--z-sticky);
    margin-bottom: calc(-1 * var(--space-1));
  }
  .phone-pick :global(.seg) {
    display: flex;
    width: 100%;
  }
  .phone-pick :global(.seg > *) {
    flex: 1;
    justify-content: center;
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
  /* The matrix takes the width its cells want (up to 72 px each), the notes follow 32 px after it and take the rest
     (at least 320 px): an 8-class matrix fills about two thirds of a full-width panel and the notes the remainder,
     while a large one shrinks its cells before the notes drop under 320 px. */
  .cm-layout {
    display: grid;
    grid-template-columns: minmax(0, max-content) minmax(320px, 1fr);
    grid-template-rows: auto 1fr;
    grid-template-areas:
      'main sum'
      'main notes';
    gap: var(--space-4) var(--space-8);
    align-items: start;
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
    max-width: 60ch;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  /* Secondary explanation: how the figures are made, folded until asked for. */
  .how summary {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    min-height: 24px;
    list-style: none;
    cursor: pointer;
    color: var(--text-tertiary);
    border-radius: var(--radius-sm);
  }
  .how summary::-webkit-details-marker {
    display: none;
  }
  .how summary:hover {
    color: var(--text-primary);
  }
  .how summary :global(svg) {
    transform: rotate(-90deg);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .how[open] summary :global(svg) {
    transform: none;
  }
  .how-body {
    display: grid;
    gap: var(--space-2);
    padding-top: var(--space-1);
    color: var(--text-tertiary);
  }
  @media (pointer: coarse) {
    .how summary {
      min-height: 44px;
    }
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
    .k-body {
      grid-template-columns: minmax(0, 1fr);
    }
    .detail {
      grid-template-columns: minmax(0, 1fr);
      align-items: start;
    }
    .rail {
      position: static;
    }
  }
  @media (pointer: coarse) {
    .k-help {
      top: 0;
      right: 0;
      width: 44px;
      height: 44px;
    }
  }
  @media (max-width: 767px) {
    .details.two {
      grid-template-columns: minmax(0, 1fr);
    }
    .two .kv {
      grid-template-columns: 96px minmax(0, 1fr);
    }
    /* One row of four values: the headline a step larger; the qualifiers live in the help. */
    .kpis {
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }
    .kpi {
      grid-template-rows: 16px 28px;
      gap: var(--space-1);
      padding: var(--space-3);
    }
    .k-sub {
      display: none;
    }
    .k-value {
      font-size: var(--type-xl-size);
      line-height: 28px;
      letter-spacing: var(--type-xl-tracking);
    }
    .hero .k-value {
      font-size: var(--type-2xl-size);
      line-height: 28px;
      letter-spacing: var(--type-2xl-tracking);
    }
    .k-body {
      grid-template-columns: minmax(0, 1fr);
      padding: var(--space-3);
    }
    .k-body > div {
      grid-template-columns: minmax(0, 1fr);
      gap: 2px;
    }
  }
</style>
