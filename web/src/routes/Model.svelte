<script lang="ts">
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import Download from '@lucide/svelte/icons/download';
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
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import { cmIndex } from './model/confusion';
  import ExportDialog from './model/ExportDialog.svelte';
  import PrCurves from './model/PrCurves.svelte';
  import { MODEL_SECTIONS, prView, sectionShown, type ModelSection } from './model/sections';

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

  /** The architecture's measured end-to-end time at 640 px (docs/BENCHMARKS.md), or undefined when not measured. */
  const latency = (m: Model): number | undefined => e2eMs[m.arch];
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
              <dt>Latency</dt>
              <dd class="latency">
                <!-- Measured for the architecture (docs/BENCHMARKS.md), never scaled or estimated: say exactly what it is. -->
                {#if latency(m) !== undefined}
                  <span class="num">{latency(m)!.toFixed(2)} ms{m.imgsz !== 640 ? ' at 640 px' : ''}</span>
                  <span class="faint">End to end, one image · RTX 4080 SUPER · fp32</span>
                {:else}
                  <span class="faint">Not measured for {m.arch} yet</span>
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
                <span class="mono">{m.arch}</span><span class="sep" aria-hidden="true">·</span>{m.imgsz} px{#if latency(m) !== undefined}<span class="d-lat"
                    ><span class="sep" aria-hidden="true">·</span><span title="Measured end to end at 640 px: RTX 4080 SUPER, fp32, one image"
                      >{latency(m)!.toFixed(2)} ms</span
                    ></span
                  >{/if}<span class="sep" aria-hidden="true">·</span>{bytes(m.bytes, prefs.units)}
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
      <Panel title="Confusion matrix" meta={cm?.partial ? `top ${CM_MAX} of ${m.classes.length} classes` : 'share of each true class'} id="cm">
        {#if cm}
          <!-- 8 + 4, on the grid of the panels above: the matrix in the main column, what to read from it in the side
               column (aligned with the Details rail). Up to 1199 px the two consequential cells lead (a phone's
               matrix scrolls sideways, hiding the background column), and the reading notes follow the matrix. -->
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
                  <p>Each row sums to 100% of that class's ground-truth boxes at IoU 0.5. Cells under 10% are not
                    labelled; hover or tap one for its count.</p>
                  <p>The background row splits the false detections (boxes on no object) by the class they were given.</p>
                  {#if cm.partial}
                    <p>The {CM_MAX} most frequent classes are shown; the other {m.classes.length - CM_MAX} are folded into Other.</p>
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
  /* Panels in a row end together: the rail stretches to the PR panel's height. */
  .detail {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
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
  /* The same 2 : 1 split as the panels above. Inside the panel's 16 px padding, an 80 px column gap puts the side
     column's text on the Details rail's text edge (rail edge + 16). */
  .cm-layout {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    grid-template-rows: auto 1fr;
    grid-template-areas:
      'main sum'
      'main notes';
    gap: var(--space-4) calc(var(--space-16) + var(--space-4));
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
    .detail {
      grid-template-columns: minmax(0, 1fr);
      align-items: start;
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
