<script lang="ts">
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import Download from '@lucide/svelte/icons/download';
  import Copy from '@lucide/svelte/icons/copy';
  import { copyText, pageCommands, type PageCommand } from '$lib/shell/pageCommands.svelte';
  import ScanSearch from '@lucide/svelte/icons/scan-search';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import { tick } from 'svelte';
  import { MediaQuery } from 'svelte/reactivity';
  import { hub } from '$lib/api/index.svelte';
  import { fromModelEval } from '$lib/api/evaluation';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Dataset, Model, Run } from '$lib/api/types';
  import { fallbackNotice, metricsLabel, sourceShort, splitWord, type EvalSource } from '$lib/evaluated';
  import { ago, bytes, date, num } from '$lib/format';
  import { loadThroughput } from '$lib/throughput';
  import { prefs } from '$lib/prefs.svelte';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Button from '$lib/ui/Button.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Menu from '$lib/ui/Menu.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Tabs from '$lib/ui/Tabs.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import Ident from '$lib/ui/Ident.svelte';
  import ExportDialog from './model/ExportDialog.svelte';
  import { FROM_SCRATCH_META, origin } from '$lib/lineage';
  import Lineage from '$lib/ui/Lineage.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import PrCurves from './model/PrCurves.svelte';
  import Callout from './run/Callout.svelte';
  import { MODEL_SECTIONS, sectionShown, type ModelSection } from './model/sections';
  import { media } from '$lib/design/tokens';
  import KpiSummary from './model/KpiSummary.svelte';
  import DetailsList from './model/DetailsList.svelte';
  import ConfusionPanel from './model/ConfusionPanel.svelte';

  const id = $derived(router.route.params.id ?? '');
  const model = new Resource((s) => hub.api.model(id, s));

  $effect(() => {
    setCrumbs({ label: 'Models', path: '/models' }, { label: model.data?.name ?? id });
  });

  // The source run, its dataset and the model it was fine-tuned from (its `weights`), for the provenance links.
  const runId = $derived(model.data?.run_id ?? null);
  // The palette's This page: the header's actions and the id.
  pageCommands((): PageCommand[] => {
    const m = model.data;
    if (!m) return [];
    return [
      { id: 'model-export', label: `Export ${m.name}`, match: ['Export model', 'ONNX'], icon: Download, run: () => (exportOpen = true) },
      { id: 'model-try', label: `Try ${m.name} in Playground`, match: ['Try model'], icon: ScanSearch, run: () => router.go(`/playground?model=${encodeURIComponent(m.id)}`) },
      { id: 'model-copy-id', label: 'Copy model id', match: [m.id], icon: Copy, run: () => void copyText(m.id, 'model id') },
    ];
  });
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
  // This server's measured latency for the architecture (GET /api/system/throughput), where it has predicted with it.
  const tp = new Resource((s) => loadThroughput(hub.api, s));
  const ev = $derived.by(() => {
    if (!evaluation.data) return null;
    const e = fromModelEval(evaluation.data);
    // An older server leaves the dataset out; /eval runs on the run's (docs/API.md), so its name stands in. The split
    // stays unknown: a missing split is not `val`.
    const ds = context.data?.dataset;
    if (ds && e.source.datasetId === null && e.source.datasetName === null) e.source = { ...e.source, datasetId: ds.id, datasetName: ds.name };
    return e;
  });
  /**
   * What the page's numbers were measured on. Before the evaluation loads (or when it fails), the run's dataset: its
   * validation split, or its training split when it has no validation images, as training and /eval both fall back.
   */
  const source = $derived.by((): EvalSource => {
    if (ev) return ev.source;
    const ds = context.data?.dataset ?? null;
    if (!ds) return { datasetId: null, datasetName: null, split: null, fallback: null };
    const empty = ds.splits.val === 0;
    return {
      datasetId: ds.id,
      datasetName: ds.name,
      split: empty ? 'train' : 'val',
      fallback: empty ? { requested: 'val', reason: 'empty_split', message: '' } : null,
    };
  });
  /** Said once on the page, under the headline metrics, when the numbers are not on the validation split. */
  const notice = $derived(fallbackNotice(source));
  // The curves' qualifier names what was evaluated ("fasteners · validation split · IoU 0.5"); phones keep the split.
  const prMeta = $derived.by(() => {
    const where = phone.current ? (source.split ? `${splitWord(source.split)} split` : null) : sourceShort(source);
    return `${where ?? 'precision by recall'} · IoU 0.5`;
  });
  /** The curves panel's band / every-curve switch (many classes). */
  let allCurves = $state(false);
  let exportOpen = $state(false);
  // Below the desktop tier the rail stacks under the PR panel: the details fold into one line so the matrix follows soon.
  const narrow = new MediaQuery(media.upToLaptop, false);
  let detailsOpen = $state(false);
  // Phones: after the headline metrics, one section at a time (as on the run page) instead of a 4,000 px column.
  const phone = new MediaQuery(media.phone, false);
  let section = $state<ModelSection>('curves');
  /** The sticky picker's height: a section's own sticky bar (Classes' selection) holds just under it. */
  let pickH = $state(0);
  const showing = (part: ModelSection) => sectionShown(phone.current, section, part);

  const smooth = (): ScrollBehavior => (matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth');
  // Phones: a class's "View curve" (in Classes) shows its traced curve, with a way back to the class it came from.
  let backTo = $state<number | null>(null);
  // The classes traced (clicked) and pointed at in the per-class table, drawn in the curves' panel.
  let prSelected = $state<number[]>([]);
  let prHovered = $state<number | null>(null);
  $effect(() => {
    if (section !== 'curves') backTo = null;
  });
  async function viewCurve(cls: number) {
    section = 'curves';
    backTo = cls;
    await tick();
    document.getElementById('pr')?.scrollIntoView({ block: 'start', behavior: smooth() });
    document.querySelector<HTMLElement>('#pr .back')?.focus({ preventScroll: true });
  }
  async function backToClasses() {
    const cls = backTo;
    section = 'classes';
    await tick();
    const row = document.querySelector<HTMLElement>(`#per-class .disc[data-cls="${cls}"]`);
    row?.focus({ preventScroll: true });
    row?.scrollIntoView({ block: 'center', behavior: smooth() });
  }

  // Phones: the work before the summary (DESIGN_DIRECTION 12). The section picker sits right under the header, so it
  // never moves between sections; every section, Curves included, opens on one line, the headline and "All metrics",
  // which unfolds the 2 × 2 grid, so the chart is the first thing under the picker. What the numbers mean is behind
  // the grid's ⓘ.
  let kpiOpen = $state(false);
  const kpiFolded = $derived(phone.current);
  /**
   * How the weights are stored (the file's folo.dtype and folo.fused, docs/API.md): "f16 · fused, inference only".
   * Null when the server does not say; a dtype this client has no name for reads as unknown, as the API asks.
   */
  const stored = $derived.by(() => {
    const m = model.data;
    const p = m?.precision === 'f32' || m?.precision === 'f16' ? m.precision : null;
    const fused = m?.fused === true;
    if (!p && !fused) return null;
    return [p, fused ? 'fused, inference only' : null].filter(Boolean).join(' · ');
  });
  // What the four numbers mean, one tap away (the strip itself keeps to values and a short qualifier).
  let helpOpen = $state(false);
  let aboutOpen = $state(false);
  // Phones: one summary line under the top bar; the rest of the provenance behind it.
  let metaOpen = $state(false);
</script>

{#if model.error && !model.data}
  <Page title="Model">
    {#snippet actions()}<Button icon={ArrowLeft} href={href('/models')}>All models</Button>{/snippet}
    <!-- Nothing to frame: the message frameless on the content edge (DESIGN_DIRECTION 8), as on Run and Not found. -->
    <ErrorState error={model.error} retry={() => model.refresh()} title="Could not load this model" />
  </Page>
{:else if !model.data}
  <Page title="Model">
    {#snippet sub()}<Skeleton w="320px" h="14px" />{/snippet}
    <Skeleton h="106px" r="var(--radius-lg)" />
    <div class="detail"><Skeleton h="480px" r="var(--radius-lg)" /><Skeleton h="480px" r="var(--radius-lg)" /></div>
  </Page>
{:else}
  {@const m = model.data}
  {@const run = context.data?.run}
  {@const head = m.head ?? 'o2m'}
  {#snippet lineageNote()}
    <!-- Where the weights started: the run's `weights`, a model (linked), an imported file, or nothing. -->
    {#if lineage && lineage.kind !== 'scratch'}
      <span>fine-tuned from <Lineage origin={lineage} /></span>
    {:else if lineage?.kind === 'scratch'}
      <span>{FROM_SCRATCH_META}</span>
    {/if}
  {/snippet}
  {#snippet provenance()}
    {#if m.run_id}
      <span>trained by <a class="link" href={href(`/runs/${m.run_id}`)}>{run?.name ?? m.run_id}</a></span>
      {@render lineageNote()}
    {:else if m.source === 'published'}
      <span title={m.about}>published weights</span>
    {:else}
      <span>imported</span>
    {/if}
    <span class="faint" title={date(m.created)}>created {ago(m.created)}</span>
  {/snippet}
  <Page title={m.name} inlineActions={phone.current}>
    {#snippet sub()}
      {#if phone.current}
        <button type="button" class="meta-line num" aria-expanded={metaOpen} aria-controls="model-meta" onclick={() => (metaOpen = !metaOpen)}>
          <Meta line><Ident value={m.arch} /><span>{m.classes.length} classes</span><span>{m.imgsz} px</span></Meta>
          <ChevronDown size={14} aria-hidden="true" />
        </button>
        {#if metaOpen}
          <div class="meta-more" id="model-meta">
            <Meta>
              <span class="num">{bytes(m.bytes, prefs.units)}</span>
              {@render provenance()}
            </Meta>
          </div>
        {/if}
      {:else}
        <!-- One line, its last item ("created yesterday") never alone on a second one (DESIGN_DIRECTION §10): on tablets
             the file size steps out (Details has it). -->
        <Meta>
          <Ident value={m.arch} />
          <span class="num">{m.classes.length} classes</span>
          <span class="num">{m.imgsz} px</span>
          <span class="num file-size">{bytes(m.bytes, prefs.units)}</span>
          {@render provenance()}
        </Meta>
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
         the ⓘ opens what they mean. Phones: a 2 × 2 summary of label over value, the headline first, inside the
         sections (below). -->
    {#if !phone.current}
    <KpiSummary {m} {source} {ev} {head} bind:helpOpen />
    {/if}

    {#if notice}
      <!-- Once on the page, in every layout and phone section: these are not validation numbers. -->
      <Callout tone="warning" icon={TriangleAlert} title={notice.title} role="note" label="Evaluation split">{notice.why}</Callout>
    {/if}

    {#if phone.current}
      <!-- Stays reachable under the top bar while a section scrolls. -->
      <div class="phone-pick" bind:offsetHeight={pickH}>
        <Tabs label="Model sections" idPrefix="model-sec" bind:value={section} tabs={MODEL_SECTIONS} />
      </div>
    {/if}

    <!-- The parts, in one frame: on phones it is the sections' tab panel; wider, a column with the page's own gap. -->
    <div class="parts" id="model-sec-panel" role={phone.current ? 'tabpanel' : undefined} aria-labelledby={phone.current ? `model-sec-${section}` : undefined}>
    {#if phone.current}
      <KpiSummary {m} {source} {ev} {head} grid {kpiFolded} bind:kpiOpen bind:helpOpen />
    {/if}

    <!-- Evidence beside reference: the curves (8 columns) and the Details rail (4) stretch to one height, the plot
         taking what the rail adds; the per-class table, the curves' legend, follows in a row of its own. Up to 1199 px
         the table follows the curves and the details fold into one line under it. Both PR panels share the traced and
         pointed-at classes. -->
    <div
      class="detail"
      class:off={!showing('curves') && !showing('classes') && !showing('details')}
      style:--pick-bottom={phone.current ? `calc(var(--size-topbar) + var(--space-2) + ${pickH}px)` : undefined}
    >
      <div class="main-col" class:off={!showing('curves')}>
        <Panel title="Precision–recall" meta={prMeta} pad="none" id="pr">
          <!-- Many classes: the band / every-curve switch is the chart's one header control, so the legend row keeps one
               line in the laptop's narrower column. Phones put it on the Trace row (PrCurves), never on a header row of
               its own that wraps under the title. -->
          {#snippet actions()}
            {#if ev && m.classes.length > 10 && !phone.current}
              <Segmented
                size="sm"
                label="Curves shown"
                bind:value={() => (allCurves ? 'all' : 'band'), (v) => (allCurves = v === 'all')}
                options={[
                  { value: 'band', label: 'Band' },
                  { value: 'all', label: 'All curves' },
                ]}
              />
            {/if}
          {/snippet}
          {#if ev}
            <PrCurves
              classes={m.classes}
              {ev}
              map50={m.metrics?.map50 ?? null}
              view="curves"
              bind:allCurves
              modeInHeader
              bind:selected={prSelected}
              bind:hovered={prHovered}
              onback={backTo !== null ? backToClasses : undefined}
            />
          {:else if evaluation.error}
            <ErrorState error={evaluation.error} retry={() => evaluation.refresh()} compact title="Could not load the evaluation" />
          {:else}
            <div class="sk"><Skeleton h="24px" w="200px" /><Skeleton h="360px" /></div>
          {/if}
        </Panel>
      </div>
      <div class="cls-part" class:off={!showing('classes')}>
        <Panel title="Per class" meta="{m.classes.length} classes · IoU 0.5" pad="none" id="per-class">
          {#if ev}
            <PrCurves
              classes={m.classes}
              {ev}
              map50={m.metrics?.map50 ?? null}
              view="classes"
              bind:selected={prSelected}
              bind:hovered={prHovered}
              onviewcurve={phone.current ? viewCurve : undefined}
            />
          {:else if evaluation.error}
            <!-- The failure is said once, in the curves' panel (or here on a phone, where that panel is another section). -->
            {#if phone.current}
              <ErrorState error={evaluation.error} retry={() => evaluation.refresh()} compact title="Could not load the evaluation" />
            {:else}
              <p class="quiet">The per-class figures show once the evaluation loads.</p>
            {/if}
          {:else}
            <div class="sk">{#each [0, 1, 2, 3, 4, 5] as i (i)}<Skeleton h="16px" />{/each}</div>
          {/if}
        </Panel>
      </div>

      <aside class="rail" class:off={!showing('details')} aria-label="Model details">
        {#snippet detailList(two: boolean)}
          <DetailsList {m} {two} {tp} {run} {context} {source} {ev} {lineage} {stored} {head} bind:aboutOpen />
        {/snippet}
        {#if phone.current}
          <!-- Its own section on phones: the whole list, no fold, and no title repeating the Details picker above it. -->
          <Panel id="details">{@render detailList(true)}</Panel>
        {:else if narrow.current}
          <Panel pad="none" id="details">
            <button type="button" class="d-line" aria-expanded={detailsOpen} aria-controls="model-details" onclick={() => (detailsOpen = !detailsOpen)}>
              <span class="d-sum num"
                ><Meta line><Ident value={m.arch} /><span>{m.imgsz} px</span><span>{num(m.flops_g, 1)} GFLOPs</span><span>{bytes(m.bytes, prefs.units)}</span></Meta></span
              >
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
      <ConfusionPanel {m} {ev} {source} {phone} bind:section {evaluation} />
    </div>
    </div>
  </Page>
  <ExportDialog model={m} bind:open={exportOpen} />
{/if}

<style>
  .link,
  :global(a.ident.link) {
    color: var(--text-primary);
  }
  .link:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
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
  /* Under the button that opens it, never pulled up over it (a control is never covered by its neighbour: a -8 px
     margin here lay over the meta line's lower half, so a tap there missed it). */
  .meta-more {
    flex-basis: 100%;
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-3);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  /* One rule at every class count, decided by structure, not by stretching: the curves (8) at their fixed plot height,
     the per-class table (their legend) under them in the same column, and Details (4) a sticky properties rail beside
     both (DESIGN_DIRECTION §2.2, as on Linear's issue view). No panel is stretched to its neighbour, so the plot never
     grows to a long rail and no frame holds empty space. */
  .detail {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    grid-template-areas: 'pr rail' 'cls rail';
    align-items: start;
    gap: var(--space-4);
  }
  @media (min-width: 1024px) {
    .detail {
      grid-template-rows: auto 1fr;
    }
    .rail {
      position: sticky;
      top: calc(var(--size-topbar) + var(--space-4));
    }
  }

  .main-col {
    grid-area: pr;
  }
  .rail {
    grid-area: rail;
  }
  .cls-part {
    grid-area: cls;
  }
  .main-col,
  .rail,
  .cls-part {
    display: grid;
    gap: var(--space-4);
    min-width: 0;
  }
  .quiet {
    padding: 0 var(--space-4) var(--space-4);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
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
  /* As on the run page: full width, just under the top bar, the tabs sharing the width. */
  .phone-pick {
    position: sticky;
    top: var(--size-topbar);
    z-index: var(--z-sticky);
    margin-top: calc(-1 * var(--space-2));
    background: var(--bg-app);
  }
  .phone-pick :global([role='tab']) {
    flex: 1;
    justify-content: center;
  }
  .parts {
    display: flex;
    flex-direction: column;
    gap: inherit;
    min-width: 0;
  }
  .sk {
    display: grid;
    gap: var(--space-2);
    padding: 0 var(--space-4) var(--space-4);
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
  .d-sum :global(.ident) {
    color: var(--text-primary);
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
  /* Tablets: the rail is folded to one line of facts (Details ›), so it sits right under the KPI strip, the reference
     before the evidence, never under a long per-class table. Phones show it as a section of its own. */
  @media (max-width: 1023px) {
    .detail {
      grid-template-columns: minmax(0, 1fr);
      grid-template-areas: 'rail' 'pr' 'cls';
    }
  }
  @media (max-width: 767px) {
    /* One section at a time behind the picker: no named rows, so the hidden sections leave no empty tracks or gaps
       and every section starts 12 px under the picker (as on Run; the shoot's QA checks it). */
    .detail {
      grid-template-areas: none;
      gap: 0;
    }
    .main-col,
    .rail,
    .cls-part {
      grid-area: auto;
    }
    /* "View curve" and "Back to Classes" land the curves panel under the top bar and the section picker. */
    .main-col :global(#pr) {
      scroll-margin-top: calc(var(--size-topbar) + var(--space-16));
    }
  }
  @media (min-width: 768px) and (max-width: 1023px) {
    .file-size {
      display: none;
    }
  }
</style>
