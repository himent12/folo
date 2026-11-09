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
  import { fromModelEval, reduceConfusion } from '$lib/api/evaluation';
  import { e2eMs } from '$lib/benchmarks';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Dataset, Model, Run } from '$lib/api/types';
  import ConfusionMatrix from '$lib/charts/ConfusionMatrix.svelte';
  import { sharePct } from '$lib/charts/confusionInk';
  import { classInk } from '$lib/design/classInk.svelte';
  import { evaluatedOn, fallbackNotice, matrixCounted, matrixVsPr, metricsLabel, sourceShort, splitWord, type EvalSource } from '$lib/evaluated';
  import { ago, atBestF1, bytes, compact, conf, date, metric, num } from '$lib/format';
  import { latencyMs, latencyRange, loadThroughput, measuredLatency } from '$lib/throughput';
  import { prefs } from '$lib/prefs.svelte';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Button from '$lib/ui/Button.svelte';
  import InfoButton from '$lib/ui/InfoButton.svelte';
  import { publishedSummary } from '$lib/published';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Menu from '$lib/ui/Menu.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Tabs from '$lib/ui/Tabs.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import Ident from '$lib/ui/Ident.svelte';
  import { cmIndex, rankConfusions, type ConfusionEntry } from './model/confusion';
  import ExportDialog from './model/ExportDialog.svelte';
  import { FROM_SCRATCH, FROM_SCRATCH_META, origin } from '$lib/lineage';
  import Lineage from '$lib/ui/Lineage.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import PrCurves from './model/PrCurves.svelte';
  import Callout from './run/Callout.svelte';
  import { MODEL_SECTIONS, sectionShown, type ModelSection } from './model/sections';
  // The head and precision in the words Run's Config and Compare use.
  import { HEAD_LABEL, PRECISION_LABEL } from './run/resolved';
  import { media } from '$lib/design/tokens';

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
  // The matrix's own thresholds and what it counted, never precision and recall's operating point.
  const matrixLine = $derived(ev ? matrixCounted(source, ev.images, ev.matrixAt) : '');
  const matrixHow = $derived(ev ? matrixVsPr(ev.matrixAt, atBestF1(ev.bestF1Conf)) : '');

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

  // The ranked lists point into the matrix: each line is a button that reveals its cell (on a phone the matrix scrolls
  // sideways, so the background column starts out of view).
  let matrixView = $state<ConfusionMatrix>();
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

  // What goes wrong most, ranked: beside the matrix in a rail on wider screens; on phones the Confusion section opens on
  // these lists and the full matrix is one step further.
  let cmExplore = $state(false);
  // Two short groups, so misses (usually the larger shares) never crowd out the confusions between classes.
  const RANK_MAX = 5;
  const ranked = $derived.by(() => {
    const all = ev ? rankConfusions(ev.confusion, Infinity) : [];
    return [
      { kind: 'confused', title: 'Most confused', rows: all.filter((e) => e.kind === 'confused').slice(0, RANK_MAX) },
      { kind: 'missed', title: 'Most missed', rows: all.filter((e) => e.kind === 'missed').slice(0, RANK_MAX) },
    ].filter((g) => g.rows.length);
  });
  /** Any confusion or miss at all, counting the classes too small to rank (fewer than 20 objects). */
  const anyErrors = $derived(!!ev && rankConfusions(ev.confusion, 1, 1).length > 0);
  /** A ranked line's cell in the shown matrix (null when its class is folded into Other). */
  function cellOf(e: ConfusionEntry): { r: number; c: number } | null {
    if (!cm) return null;
    const r = cmIndex(e.r, cm.ids);
    const c = e.kind === 'missed' ? cm.classes.length : cmIndex(e.c, cm.ids);
    return r === null || c === null ? null : { r, c };
  }
  const smooth = (): ScrollBehavior => (matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth');

  /** Points the matrix at a cell, first bringing the Confusion section and its matrix up on a phone. */
  async function reveal(at: { r: number; c: number } | null) {
    if (phone.current && (section !== 'confusion' || !cmExplore)) {
      section = 'confusion';
      cmExplore = true;
      await tick();
    }
    // The pressed line is gone with the list: focus lands on the cell, or (a class folded into Other) on the way back.
    if (at) matrixView?.reveal(at.r, at.c);
    else focusBack();
  }
  /** Phones: the whole matrix, focus on its way back to the list (the pressed button is gone with the list). */
  async function openMatrix() {
    cmExplore = true;
    await tick();
    focusBack();
  }
  const focusBack = () => document.querySelector<HTMLElement>('#cm .to-list')?.focus({ preventScroll: true });
  /** Back from the matrix to the ranked list, keeping the reader's place at the section's top. */
  async function closeMatrix() {
    cmExplore = false;
    await tick();
    document.querySelector<HTMLElement>('#cm .explore')?.focus({ preventScroll: true });
  }

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

  /**
   * The architecture's end-to-end time from docs/BENCHMARKS.md (640 px, one RTX 4080 SUPER), or undefined when not
   * measured. It is a benchmark of the architecture, never this model's own timing, and is labelled as such.
   */
  const latency = (m: Model): number | undefined => e2eMs[m.arch];
  const kpis = (m: Model) => [
    { label: 'mAP50-95', value: m.metrics?.map50_95, sub: 'IoU 0.50:0.95' },
    { label: 'mAP50', value: m.metrics?.map50, sub: 'IoU 0.50' },
    // The stored metrics' own threshold (the epoch that wrote the weights), never /eval's: one source per strip.
    { label: 'Precision', value: m.metrics?.precision, sub: atBestF1(m.metrics?.best_f1_conf) },
    { label: 'Recall', value: m.metrics?.recall, sub: atBestF1(m.metrics?.best_f1_conf) },
  ];
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
  {#snippet kHelp()}
    <!-- What the four numbers mean: one place in every layout, after the strip's first label (the strip and the phone's
         2 × 2 grid alike), so it never lands on whichever cell happens to be last or top-right. -->
    <InfoButton label="What these metrics mean" controls="metric-help" bind:open={helpOpen} class="k-help" />
  {/snippet}
  {#snippet metricHelp()}
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
        <dd>Measured at the confidence that gives the best F1 (precision and recall balanced, averaged over classes, at IoU 0.50): the model's best trade-off between missed objects and false boxes.{' '}{#if m.metrics?.best_f1_conf != null}Here
            conf {conf(m.metrics.best_f1_conf)}, from the epoch that wrote these weights.{:else}This model's file does not say which
            threshold that was.{/if}</dd>
      </div>
      <div>
        <dt>Evaluated on</dt>
        <dd>
          {evaluatedOn(source, ev ? ev.images : null)}, at {m.imgsz} px, {head === 'o2o' ? 'end to end (the one-to-one head, no NMS)' : 'with NMS IoU 0.7'}.
        </dd>
      </div>
    </dl>
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
    <section class="kpis" aria-label={metricsLabel(source)}>
      {#each kpis(m) as k, i (k.label)}
        <div class="kpi" class:hero={i === 0}>
          <span class="k-label"><span class="truncate">{k.label}</span>{#if i === 0}{@render kHelp()}{/if}</span>
          <span class="k-value num">{metric(k.value)}</span>
          <span class="k-sub">{k.sub}</span>
        </div>
      {/each}
      {#if helpOpen}{@render metricHelp()}{/if}
    </section>
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
      <!-- Phones: the summary heads its section, one line that unfolds the grid. -->
      <section class="kpi-grid num" class:folded={kpiFolded} aria-label={metricsLabel(source)}>
        {#if kpiFolded}
          <button type="button" class="kpi-line" aria-expanded={kpiOpen} aria-controls="kpi-cells" onclick={() => (kpiOpen = !kpiOpen)}>
            <span class="kl-name">mAP50-95</span>
            <span class="kl-value">{metric(m.metrics?.map50_95)}</span>
            <span class="kl-verb">All metrics <ChevronDown size={14} aria-hidden="true" /></span>
          </button>
        {/if}
        {#if !kpiFolded || kpiOpen}
          <dl class="kg" id="kpi-cells">
            {#each kpis(m) as k, i (k.label)}
              <div class="kg-cell" class:hero={i === 0}>
                <dt>{k.label}{#if i === 0}{@render kHelp()}{/if}</dt>
                <dd>{metric(k.value)}</dd>
              </div>
            {/each}
          </dl>
          {#if helpOpen}{@render metricHelp()}{/if}
        {/if}
      </section>
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
               line in the laptop's narrower column. -->
          {#snippet actions()}
            {#if ev && m.classes.length > 10}
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
          {@const here = measuredLatency(tp.data, m.arch, m.imgsz)}
          <dl class="details" class:two id="model-details">
            <div class="kv">
              <dt>Architecture</dt>
              <dd><Ident value={m.arch} /></dd>
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
              <dt>Head</dt>
              <dd>{HEAD_LABEL[head]}</dd>
            </div>
            <!-- How it was trained, from the file's training record: none for exports, imported weights and older
                 servers, so nothing is said then. -->
            {#if m.trained_with}
              <div class="kv">
                <dt>Trained in</dt>
                <dd>{m.trained_with.amp ? PRECISION_LABEL.amp : PRECISION_LABEL.f32}</dd>
              </div>
              <div class="kv">
                <dt>Trained for</dt>
                <dd class="num">{num(m.trained_with.epochs)} {m.trained_with.epochs === 1 ? 'epoch' : 'epochs'} at {m.trained_with.imgsz} px</dd>
              </div>
            {/if}
            {#if stored}
              <div class="kv">
                <dt>Stored as</dt>
                <dd>{stored}</dd>
              </div>
            {/if}
            <div class="kv">
              <dt>File size</dt>
              <dd class="num">{bytes(m.bytes, prefs.units)}</dd>
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
            <div class="kv" class:stack={!m.run_id && m.source === 'published'}>
              <dt>Source</dt>
              <dd>
                {#if m.run_id}
                  <a class="link" href={href(`/runs/${m.run_id}`)}>{run?.name ?? m.run_id}</a>
                  {#if context.data?.dataset}<span class="faint">{' on '}</span><a class="link" href={href(`/datasets/${context.data.dataset.id}`)}>{context.data.dataset.name}</a>{/if}
                {:else if m.source === 'published'}
                  <!-- `folo weights pull` weights: what they are, in the words `folo weights list` prints. -->
                  <!-- What they are for in plain words; the server's own description, its provenance, in the ⓘ. -->
                  <span class="pub">Published weights{#if m.about}<InfoButton label="Where these weights come from" controls="model-about" bind:open={aboutOpen} />{/if}</span>
                  <span class="about">{publishedSummary(m)}</span>
                  {#if m.about && aboutOpen}<span class="about" id="model-about">{m.about}</span>{/if}
                {:else}
                  <span class="muted">Imported .folo file</span>
                {/if}
              </dd>
            </div>
            <!-- What the curves, the per-class table and the matrix were measured on (/eval's provenance). -->
            <div class="kv">
              <dt>Evaluated on</dt>
              <!-- The dataset as the value, the split and its images under it, so the row keeps the rail's alignment. -->
              <dd class="evald">
                {#if source.datasetName && source.datasetId}
                  <a class="link" href={href(`/datasets/${encodeURIComponent(source.datasetId)}`)}>{source.datasetName}</a>
                {:else if source.datasetName}
                  <span>{source.datasetName}</span>
                {/if}
                <span class="num" class:faint={!!source.datasetName}
                  >{source.split ? `${splitWord(source.split)} split` : 'split not reported'}{ev ? `, ${num(ev.images)} ${ev.images === 1 ? 'image' : 'images'}` : ''}</span
                >
              </dd>
            </div>
            {#if lineage}
              <div class="kv">
                <dt>Started from</dt>
                <dd>
                  {#if lineage.kind !== 'scratch'}
                    <Lineage origin={lineage} />
                  {:else}
                    <span class="muted">{FROM_SCRATCH}</span>
                  {/if}
                </dd>
              </div>
            {/if}
            <div class="kv">
              <dt>Model id</dt>
              <dd><Ident value={m.id} class="muted" /></dd>
            </div>
            <div class="kv">
              <dt>Created</dt>
              <dd>{date(m.created)}</dd>
            </div>
            <!-- The architecture's number, never this model's own timing: measured on this server where it has
                 predicted with the architecture at the model's size (its spread and count beside it), the reference
                 (docs/BENCHMARKS.md) otherwise or under it. The label says whose it is; the conditions sit beside the
                 value. Last, apart from the facts about this model. -->
            <div class="kv wide ref">
              <dt>{here ? 'Architecture latency' : 'Architecture reference latency'}</dt>
              {#if here}
                <!-- Two labelled rows, each value with its conditions under it, so no fact is left alone on a line. -->
                <dd class="latency lat-rows">
                  <span class="lat-k">Measured</span>
                  <span class="num">{latencyMs(here.latency_ms)}</span>
                  <!-- Each row: where it was measured, then the conditions, as two lines of the same anatomy. -->
                  <Meta class="figs num"><span>on this server</span><span>{latencyRange(here.latency_ms_low, here.latency_ms_high)}</span></Meta>
                  <Meta class="figs num"><span>{num(here.samples)} predictions</span><span>{m.imgsz} px</span><span>{here.precision}</span></Meta>
                  <span class="lat-k">Reference</span>
                  {#if latency(m) !== undefined}
                    <span class="num">{latency(m)!.toFixed(2)} ms</span>
                    <Meta class="figs num"><span>on an RTX 4080 SUPER</span></Meta>
                    <Meta class="figs num"><Ident value={m.arch} /><span>640 px</span></Meta>
                  {:else}
                    <span class="faint">Not measured for <Ident value={m.arch} /> yet</span>
                  {/if}
                </dd>
              {:else}
                <dd class="latency">
                  {#if latency(m) !== undefined}
                    <span class="num">{latency(m)!.toFixed(2)} ms</span>
                    <Meta class="figs"><Ident value={m.arch} /><span>640 px</span><span>RTX 4080 SUPER</span><span>end to end</span></Meta>
                  {:else}
                    <span class="faint">Not measured for <Ident value={m.arch} /> yet</span>
                  {/if}
                </dd>
              {/if}
            </div>
          </dl>
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
    <!-- Phones, exploring the matrix: the way back is in the panel's heading row, so the only control row above the
         cells is the matrix's own (overview and pages). -->
    {#snippet cmBack()}<button type="button" class="to-list" onclick={closeMatrix}><ArrowLeft size={14} aria-hidden="true" />Back to lists</button>{/snippet}
    <!-- Full width: the matrix grows with the class count, and the rail beside it would sit empty. -->
    <div class="cm-part" class:off={!showing('confusion')}>
      <Panel
        title={phone.current && !cmExplore ? 'Confusions and misses' : 'Confusion matrix'}
        meta={phone.current && !cmExplore
          ? 'largest share of a class first'
          : cm?.partial
            ? phone.current
              ? `top ${CM_MAX} + Other`
              : `${CM_MAX} most common classes + Other + background`
            : phone.current
              ? undefined
              : 'share of each true class'}
        actions={phone.current && cmExplore && cm ? cmBack : undefined}
        id="cm"
      >
        {#if cm && phone.current && !cmExplore}
          <!-- Phones: what goes wrong most, as a list (share of the true class and the counts behind it); each line
               opens the matrix at its cell, and the whole matrix is one explicit step away. -->
          {#each ranked as g (g.kind)}
            <h3 class="rank-h" id="cm-rank-{g.kind}">{g.title}</h3>
            <ol class="rank" aria-labelledby="cm-rank-{g.kind}">
              {#each g.rows as e (`${e.r}-${e.c}`)}
                <li>
                  <button type="button" class="rk" onclick={() => reveal(cellOf(e))}>
                    <span class="rk-what">
                      <span class="sw" style:background={classInk(e.r)} aria-hidden="true"></span>
                      {#if e.kind === 'missed'}<b>{m.classes[e.r]}</b> missed{:else}<b>{m.classes[e.r]}</b> → <b>{m.classes[e.c]}</b>{/if}
                    </span>
                    <span class="rk-pct num">{sharePct(e.v)}%</span>
                    <span class="rk-sub num">
                      {num(e.count)} of {num(e.total)}
                      {e.kind === 'missed' ? 'objects found no box' : `objects predicted as ${m.classes[e.c]}`}
                    </span>
                  </button>
                </li>
              {/each}
            </ol>
          {:else}
            <p class="none">
              {anyErrors
                ? 'No class has the 20 objects it takes to rank its confusions here; the matrix shows every cell.'
                : 'No class is confused with another or missed.'}
            </p>
          {/each}
          <div class="rank-foot">
            <p>{#if missed}<span class="num">{sharePct(missed.v) || '0'}%</span> of all objects missed overall.{' '}{/if}{matrixLine}</p>
            <Button class="explore" onclick={openMatrix}>Explore the matrix</Button>
          </div>
        {:else if cm}
          <!-- One layout whatever the class count (DESIGN_DIRECTION, Model detail): from the laptop tier the matrix on
               the left 8 columns (cells up to 48 px) and a sticky rail on the right 4, what goes wrong most as two
               ranked lists stacked (each line shows its cell), then how to read the matrix; under 5 classes, 6 + 6.
               Tablets: the matrix, the two lists side by side under it. Phones lead with the lists; the matrix is a
               step away. -->
          <div class="cm-layout" class:few={cm.classes.length < 5}>
            <div class="cm-main"><ConfusionMatrix bind:this={matrixView} classes={cm.classes} classIds={cm.ids} matrix={cm.matrix} label="Confusion matrix of {m.name}" minCell={32} maxCell={48} scale /></div>
            <div class="cm-rail">
              <div class="rail-in">
                {#if !phone.current}
                  {#each ranked as g (g.kind)}
                    <section class="rl" aria-labelledby="cm-rl-{g.kind}">
                      <h3 class="rl-h" id="cm-rl-{g.kind}">{g.title}</h3>
                      <ol>
                        {#each g.rows as e (`${e.r}-${e.c}`)}
                          {@const at = cellOf(e)}
                          <li>
                            <button
                              type="button"
                              class="rl-row"
                              disabled={!at}
                              title={at ? 'Show in the matrix' : undefined}
                              aria-label="{e.kind === 'missed' ? `${m.classes[e.r]} missed` : `${m.classes[e.r]} taken for ${m.classes[e.c]}`}: {num(e.count)} of {num(e.total)}, {sharePct(e.v)}%{at ? '. Show in the matrix' : ''}"
                              onclick={() => reveal(at)}
                            >
                              <span class="rl-what">
                                <span class="sw" style:background={classInk(e.r)} aria-hidden="true"></span>
                                {#if e.kind === 'missed'}<b class="truncate">{m.classes[e.r]}</b>{:else}<span class="truncate"><b>{m.classes[e.r]}</b> → <b>{m.classes[e.c]}</b></span>{/if}
                              </span>
                              <span class="rl-n num">{num(e.count)} of {num(e.total)}</span>
                              <span class="rl-pct num">{sharePct(e.v)}%</span>
                            </button>
                          </li>
                        {/each}
                      </ol>
                    </section>
                  {/each}
                {/if}
                <div class="cm-notes">
                  {#if missed && !phone.current}<p><span class="num strong">{sharePct(missed.v) || '0'}%</span> of all objects missed overall.</p>{/if}
                  <p>{matrixLine} Each row is one true class, split by what the model predicted for its objects. The diagonal
                    is right; the background column is objects it missed.</p>
                  <details class="how">
                    <summary><ChevronDown size={14} aria-hidden="true" />How the shares are computed</summary>
                    <div class="how-body">
                      <p>{matrixHow}</p>
                      <p>Each row sums to 100% of that class's ground-truth boxes, in whole percentages ("&lt;1"
                        under half a percent). A cell too small for its number shows a dot instead; hover, tap or focus any cell
                        for its count. An empty cell had no objects at all.</p>
                      <p>The background row splits the false detections (boxes on no object) by the class they were given.</p>
                      {#if cm.partial}
                        <p>The {CM_MAX} classes with the most ground-truth boxes are shown by name; the other {m.classes.length - CM_MAX} are folded into Other.</p>
                      {/if}
                    </div>
                  </details>
                </div>
              </div>
            </div>
          </div>
        {:else if evaluation.error}
          <p class="none">No evaluation available.</p>
        {:else}
          <Skeleton h="300px" />
        {/if}
      </Panel>
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
  /* 1 px gaps over a hairline-coloured background draw the dividers in any column count. */
  .kpis {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1px;
    background: var(--line-subtle);
    background-clip: padding-box;
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    overflow: hidden;
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
  /* The caption is read ("at conf 0.39 (best F1)"): secondary, like every KPI caption (ui/Stat, Run's tabs). */
  .k-sub {
    color: var(--text-secondary);
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
  /* The help toggle follows the first label, on its line: a 20 px box (the row is 16 px tall), with a 44 px target on
     touch screens. */
  .k-label {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    min-width: 0;
    overflow: visible;
  }
  /* The 24 px ⓘ sits in a 16 px label line without deepening it. */
  dt :global(.k-help),
  .k-label :global(.k-help) {
    margin: -4px 0;
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
    max-width: calc(112px + var(--space-3) + var(--measure)); /* the term column, then the definition at the measure */
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
  /* Phones, the head of the Curves section (folded to one line in the others): a compact 2 × 2 summary in the strip's
     frame, mAP50-95 first. Each cell is its label over
     its value, the cells parted by hairlines (1 px gaps over the line colour); the ⓘ sits in the top-right corner, and
     what the numbers mean opens under the grid. */
  .kpi-grid {
    position: relative;
    display: grid;
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
    overflow: hidden;
  }
  .kg {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1px;
    background: var(--line-subtle);
  }
  .kg-cell {
    display: grid;
    grid-template-rows: 16px 32px;
    gap: 2px;
    /* The phone panels' one 16 px inset, so labels line up with the sections' titles and bodies below. */
    padding: var(--space-2) var(--space-4);
    background: var(--bg-panel);
    min-width: 0;
  }
  .kg dt {
    font-size: var(--type-sm-size);
    line-height: 16px;
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .kg .hero dt {
    color: var(--text-secondary);
  }
  /* The desktop strip's hierarchy in two steps down: the headline at the KPI step (2xl), the others at lg, set 6 px
     lower in the same 32 px row so the two values in a row share a baseline. */
  .kg dd {
    text-align: left;
    padding-top: 6px;
    font-size: var(--type-lg-size);
    line-height: 26px;
    letter-spacing: var(--type-lg-tracking);
    font-weight: var(--weight-semibold);
  }
  .kg .hero dd {
    padding-top: 0;
    font-size: var(--type-2xl-size);
    line-height: var(--type-2xl-line);
    letter-spacing: var(--type-2xl-tracking);
  }
  .kg dt {
    display: flex;
    align-items: center;
    gap: var(--space-1);
  }
  .kpi-grid .k-body {
    border-top: 1px solid var(--line-subtle);
  }
  /* Folded (every phone section but Curves): one 48 px line in the grid's frame, the headline's label and value on
     the left and the verb that unfolds the four cells on the right. */
  .kpi-line {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
    width: 100%;
    min-height: 48px;
    padding: var(--space-3) var(--space-4);
    text-align: left;
  }
  .kpi-line:focus-visible {
    outline-offset: -2px;
  }
  .kl-name {
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .kl-value {
    font-size: var(--type-lg-size);
    line-height: 24px;
    letter-spacing: var(--type-lg-tracking);
    font-weight: var(--weight-semibold);
  }
  .kl-verb {
    display: inline-flex;
    align-self: center;
    align-items: center;
    gap: var(--space-1);
    margin-left: auto;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .kl-verb :global(svg) {
    color: var(--text-tertiary);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .kpi-line[aria-expanded='true'] .kl-verb :global(svg) {
    transform: rotate(180deg);
  }
  .kpi-line + .kg {
    border-top: 1px solid var(--line-subtle);
  }
  /* Phones' confusion list: two-line 56 px rows (the pair and its share, then the counts behind it). */
  .rank-h {
    display: flex;
    align-items: center;
    min-height: 32px;
    margin: 0 calc(var(--space-4) * -1);
    padding: 0 var(--space-4);
    border-top: 1px solid var(--line-subtle);
    background: var(--bg-raised);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
  }
  .rank {
    margin: 0 calc(var(--space-4) * -1);
    border-top: 1px solid var(--line-subtle);
  }
  .rank li + li {
    border-top: 1px solid var(--line-subtle);
  }
  .rk {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: baseline;
    gap: 2px var(--space-3);
    width: 100%;
    min-height: 56px;
    padding: var(--space-2) var(--space-4);
    text-align: left;
  }
  .rk:hover {
    background: var(--bg-hover);
  }
  .rk:focus-visible {
    outline-offset: -2px;
  }
  .rk-what {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    color: var(--text-secondary);
  }
  .rk-what b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .rk-pct {
    font-weight: var(--weight-medium);
  }
  .rk-sub {
    grid-column: 1 / -1;
    padding-left: calc(8px + var(--space-2));
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .rank-foot {
    display: grid;
    justify-items: start;
    gap: var(--space-3);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
    margin: 0 calc(var(--space-4) * -1);
    padding-inline: var(--space-4);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .to-list {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    min-height: 44px;
    /* A 44 px target in the 40 px heading row: it overhangs the row instead of deepening it. */
    margin: -6px calc(var(--space-1) * -1) -6px 0;
    padding: 0 var(--space-1);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
  }
  .to-list:hover {
    color: var(--text-primary);
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
  /* Under the laptop tier the matrix takes the width its cells want (up to 48 px each) and the rail wraps under it at
     full width, its two lists side by side where 560 px allows. */
  .cm-layout {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-4) var(--space-8);
    align-items: flex-start;
  }
  .cm-main {
    flex: 0 1 auto;
    min-width: 0;
    max-width: 100%;
  }
  .cm-rail {
    flex: 1 1 280px;

    position: sticky;
    top: calc(var(--size-topbar) + var(--space-4));
    min-width: 0;
    container: cm-rail / inline-size;
  }
  .rail-in {
    display: grid;
    gap: var(--space-5);
  }
  /* Room for two lists (a wide rail beside a small matrix, or the rail under the matrix): side by side, the notes
     under both. */
  @container cm-rail (min-width: 560px) {
    .rail-in {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: var(--space-4) var(--space-8);
    }
    .rail-in .cm-notes {
      grid-column: 1 / -1;
    }
  }
  /* From the laptop tier, one layout for 3 classes or 80 (round 26: a 3-class matrix beside lists spread as three
     columns over 1160 px, an 8-class one beside them stacked): the matrix on 8 of the page grid's 12 columns, the rail
     on 4, its lists stacked and the notes under them; under 5 classes the small matrix and the rail share 6 + 6. The
     rail sticks under the top bar while a tall matrix scrolls past. */
  @media (min-width: 1024px) {
    .cm-layout {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: var(--space-4);
    }
    .cm-main {
      grid-column: span 8;
    }
    .cm-rail {
      grid-column: span 4;
    }
    .cm-layout.few .cm-main,
    .cm-layout.few .cm-rail {
      grid-column: span 6;
    }
    .cm-layout .rail-in {
      grid-template-columns: minmax(0, 1fr);
      gap: var(--space-5);
    }
    .cm-layout .rail-in .cm-notes {
      grid-column: auto;
    }
  }
  /* A ranked list: 32 px rows (the pair, its counts, its share); each line shows its cell in the matrix. */
  .rl-h {
    display: flex;
    align-items: center;
    height: 28px;
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
    border-bottom: 1px solid var(--line-subtle);
  }
  .rl li + li {
    border-top: 1px solid var(--line-subtle);
  }
  .rl-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto 40px;
    align-items: center;
    gap: var(--space-3);
    width: calc(100% + var(--space-4));
    height: 32px;
    margin: 0 calc(var(--space-2) * -1);
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
    text-align: left;
    color: var(--text-secondary);
  }
  .rl-row:hover:not(:disabled) {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .rl-row:focus-visible {
    outline-offset: -2px;
  }
  .rl-row:disabled {
    cursor: default;
  }
  .rl-what {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .rl-what b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .rl-n {
    color: var(--text-tertiary);
  }
  .rl-pct {
    text-align: right;
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  @media (pointer: coarse) {
    .rl-row {
      height: 44px;
    }
  }
  .cm-notes {
    display: grid;
    gap: var(--space-2);
    max-inline-size: var(--measure);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .cm-notes .strong {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
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
  /* A value with a sentence under it (published weights and what they are): the label over it, the sentence at the
     measure, in the rail; the folded rail keeps its label column (`.two .kv` below). */
  .kv.stack {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 2px;
  }
  .kv.stack dd {
    display: grid;
    gap: 2px;
    text-align: left;
  }
  .pub {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
  }
  .about {
    max-inline-size: var(--measure);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    text-wrap: pretty;
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
  .evald {
    display: grid;
    justify-items: end;
    text-align: right;
  }
  .evald .faint {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .two .evald {
    justify-items: start;
    text-align: left;
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
  dt {
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  dd {
    min-width: 0;
    text-align: right;
    overflow-wrap: anywhere;
  }
  /* Stacked in every layout: the label is long, and the value keeps its conditions on the same line. */
  .kv.ref,
  .two .kv.ref {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 2px;
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
  }
  /* The value, then each line of its conditions as its own meta line: items wrap whole, so a fact ("RTX 4080 SUPER")
     never breaks inside (dd's overflow-wrap would split it). Figures people read are secondary (§2.3). */
  .latency {
    display: grid;
    justify-items: start;
    gap: 2px;
    text-align: left;
    overflow-wrap: normal;
  }
  .latency .faint,
  .latency :global(.figs) {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .latency :global(.figs) {
    color: var(--text-secondary);
  }
  /* Measured and Reference: a label column, the value beside it and its conditions under the value. */
  .lat-rows {
    grid-template-columns: max-content minmax(0, 1fr);
    column-gap: var(--space-3);
  }
  /* Global: the conditions line is ui/Meta's own element. */
  .lat-rows > :global(:not(.lat-k)) {
    grid-column: 2;
  }
  .lat-k {
    grid-column: 1;
    color: var(--text-tertiary);
  }
  .lat-rows > .lat-k:not(:first-child),
  .lat-rows > .lat-k:not(:first-child) + :global(*) {
    margin-top: var(--space-2);
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
  @media (max-width: 1279px) {
    .k-body {
      grid-template-columns: minmax(0, 1fr);
    }
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
    .details.two {
      grid-template-columns: minmax(0, 1fr);
    }
    .two .kv {
      grid-template-columns: 96px minmax(0, 1fr);
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
  @media (min-width: 768px) and (max-width: 1023px) {
    .file-size {
      display: none;
    }
  }
</style>
