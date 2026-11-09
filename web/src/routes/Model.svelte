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
  import { sharePct } from '$lib/charts/confusionInk';
  import { classInk } from '$lib/design/classInk.svelte';
  import { ago, atBestF1, bytes, compact, conf, date, metric, num } from '$lib/format';
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
  import Ident from '$lib/ui/Ident.svelte';
  import { cmIndex, rankConfusions, type ConfusionEntry } from './model/confusion';
  import ExportDialog from './model/ExportDialog.svelte';
  import { origin } from '$lib/lineage';
  import PrCurves from './model/PrCurves.svelte';
  import { MODEL_SECTIONS, sectionShown, type ModelSection } from './model/sections';
  import { media } from '$lib/design/tokens';

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
  let exportOpen = $state(false);
  // Below the desktop tier the rail stacks under the PR panel: the details fold into one line so the matrix follows soon.
  const narrow = new MediaQuery(media.upToLaptop, false);
  let detailsOpen = $state(false);
  // Phones: after the headline metrics, one section at a time (as on the run page) instead of a 4,000 px column.
  const phone = new MediaQuery(media.phone, false);
  let section = $state<ModelSection>('curves');
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
  // Phones: the four numbers as one compact 2 × 2 summary in every section (Curves too), so it never changes shape
  // between sections and each section starts on the first screen; what they mean is behind the ⓘ.
  const kpiGrid = $derived(phone.current);
  /** The detection head, from the file (`Model.head`; missing, from older servers, means o2m). */
  const HEAD_LABEL = { o2m: 'NMS (one-to-many)', o2o: 'End-to-end (no NMS)' } as const;
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
    <div class="detail"><Skeleton h="480px" r="var(--radius-lg)" /><Skeleton h="480px" r="var(--radius-lg)" /></div>
  </Page>
{:else}
  {@const m = model.data}
  {@const run = context.data?.run}
  {@const head = m.head ?? 'o2m'}
  {#snippet lineageNote()}
    <!-- Where the weights started: the run's `weights`, a model (linked), an imported file, or nothing. -->
    {#if lineage?.kind === 'model'}
      {@const lid = lineage.id}
      <span>fine-tuned from <Ident value={context.data?.parent?.name ?? lid} class="link" href={href(`/models/${encodeURIComponent(lid)}`)} /></span>
    {:else if lineage?.kind === 'file'}
      <span>fine-tuned from <Ident value={lineage.label} title={lineage.path} /></span>
    {:else if lineage?.kind === 'scratch'}
      <span>from scratch</span>
    {/if}
  {/snippet}
  {#snippet kHelp()}
    <!-- What the four numbers mean: one place in every layout, after the strip's first label (the strip and the phone's
         2 × 2 grid alike), so it never lands on whichever cell happens to be last or top-right. -->
    <button type="button" class="k-help" aria-label="What these metrics mean" aria-expanded={helpOpen} aria-controls="metric-help" onclick={() => (helpOpen = !helpOpen)}
      ><Info size={14} aria-hidden="true" /></button
    >
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
          {#if context.data?.dataset}The validation split of {context.data.dataset.name}{:else}The validation split of the model's dataset{/if}{ev
            ? ` (${num(ev.images)} images)`
            : ''}, at {m.imgsz} px, {head === 'o2o' ? 'end to end (the one-to-one head, no NMS)' : 'with NMS IoU 0.7'}.
        </dd>
      </div>
    </dl>
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
          <span class="truncate"><Ident value={m.arch} /> · {m.classes.length} classes · {m.imgsz} px</span>
          <ChevronDown size={14} aria-hidden="true" />
        </button>
        {#if metaOpen}
          <div class="meta-more" id="model-meta">
            <span class="num">{bytes(m.bytes, prefs.units)}</span>
            {@render provenance()}
          </div>
        {/if}
      {:else}
        <Ident value={m.arch} />
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
         the ⓘ opens what they mean. Phones: a 2 × 2 summary of label over value, the headline first. -->
    {#if kpiGrid}
      <section class="kpi-grid num" aria-label="Validation metrics">
        <dl class="kg">
          {#each kpis(m) as k, i (k.label)}
            <div class="kg-cell" class:hero={i === 0}>
              <dt>{k.label}{#if i === 0}{@render kHelp()}{/if}</dt>
              <dd>{metric(k.value)}</dd>
            </div>
          {/each}
        </dl>
        {#if helpOpen}{@render metricHelp()}{/if}
      </section>
    {:else}
    <section class="kpis" aria-label="Validation metrics">
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

    {#if phone.current}
      <!-- Stays reachable under the top bar while a section scrolls. -->
      <div class="phone-pick">
        <Segmented label="Show" bind:value={section} options={MODEL_SECTIONS} />
      </div>
    {/if}

    <!-- Evidence beside reference: the curves (8 columns) and the Details rail (4) stretch to one height, the plot
         taking what the rail adds; the per-class table, the curves' legend, follows in a row of its own. Up to 1199 px
         the table follows the curves and the details fold into one line under it. Both PR panels share the traced and
         pointed-at classes. -->
    <div class="detail" class:off={!showing('curves') && !showing('classes') && !showing('details')}>
      <div class="main-col" class:off={!showing('curves')}>
        <Panel title="Precision–recall" meta="precision by recall · IoU 0.5" pad="none" id="pr">
          {#if ev}
            <PrCurves
              classes={m.classes}
              {ev}
              map50={m.metrics?.map50 ?? null}
              view="curves"
              fill={!narrow.current}
              bind:selected={prSelected}
              bind:hovered={prHovered}
              onback={backTo !== null ? backToClasses : undefined}
            />
          {:else if evaluation.error}
            <ErrorState error={evaluation.error} retry={() => evaluation.refresh()} compact title="Could not load the evaluation" />
          {:else}
            <div class="sk"><Skeleton h="24px" w="200px" /><Skeleton h="300px" /></div>
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
                <dd>{m.trained_with.amp ? 'Mixed precision (f16)' : 'f32'}</dd>
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
                    <Ident value={context.data?.parent?.name ?? lineage.id} class="link" href={href(`/models/${encodeURIComponent(lineage.id)}`)} />
                  {:else if lineage.kind === 'file'}
                    <Ident value={lineage.label} title={lineage.path} />
                  {:else}
                    <span class="muted">From scratch</span>
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
            <!-- The architecture's number (docs/BENCHMARKS.md), never this model's timing: the label says whose it is,
                 and the conditions sit beside the value. Last, apart from the facts about this model. -->
            <div class="kv wide ref">
              <dt>Architecture reference latency</dt>
              <dd class="latency">
                {#if latency(m) !== undefined}
                  <span class="num">{latency(m)!.toFixed(2)} ms</span>
                  <span class="faint"><Ident value={m.arch} /> · 640 px · RTX 4080 SUPER · end to end</span>
                {:else}
                  <span class="faint">Not measured for <Ident value={m.arch} /> yet</span>
                {/if}
              </dd>
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
                <Ident value={m.arch} /><span class="sep" aria-hidden="true">·</span>{m.imgsz} px<span class="sep" aria-hidden="true">·</span
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
      <Panel
        title={phone.current && !cmExplore ? 'Confusions and misses' : 'Confusion matrix'}
        meta={phone.current && !cmExplore
          ? 'largest share of a class first'
          : cm?.partial
            ? phone.current
              ? `${CM_MAX} most common + Other`
              : `${CM_MAX} most common classes + Other + background`
            : 'share of each true class'}
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
            {#if missed}<p><span class="num">{sharePct(missed.v) || '0'}%</span> of all objects missed overall, at IoU 0.5.</p>{/if}
            <Button class="explore" onclick={openMatrix}>Explore the matrix</Button>
          </div>
        {:else if cm}
          {#if phone.current}
            <button type="button" class="to-list" onclick={closeMatrix}><ArrowLeft size={14} aria-hidden="true" />Confusions and misses</button>
          {/if}
          <!-- The matrix at its own size (cells up to 48 px) with a sticky rail beside it taking the rest of the width:
               what goes wrong most, ranked (each line shows its cell), and how to read the matrix. With under 300 px
               left beside the matrix the rail wraps under it; a rail 560 px or wider sets its two lists side by side.
               Phones lead with the matrix alone: the lists are the step back. -->
          <div class="cm-layout">
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
                  {#if missed && !phone.current}<p><span class="num strong">{sharePct(missed.v) || '0'}%</span> of all objects missed overall, at IoU 0.5.</p>{/if}
                  <p>Each row is one true class, split by what the model predicted for its objects. The diagonal is right;
                    the background column is objects it missed.</p>
                  <details class="how">
                    <summary><ChevronDown size={14} aria-hidden="true" />How the shares are computed</summary>
                    <div class="how-body">
                      <p>Each row sums to 100% of that class's ground-truth boxes at IoU 0.5, in whole percentages ("&lt;1"
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
  .k-help {
    position: relative;
    flex-shrink: 0;
    display: inline-grid;
    place-items: center;
    width: 20px;
    height: 20px;
    margin: -2px 0;
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
  /* Phones, every section: a compact 2 × 2 summary in the strip's frame, mAP50-95 first. Each cell is its label over
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
    grid-template-rows: 16px 28px;
    gap: 2px;
    padding: var(--space-2) var(--space-3);
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
  .kg dd {
    text-align: left;
    font-size: var(--type-xl-size);
    line-height: 28px;
    letter-spacing: var(--type-xl-tracking);
    font-weight: var(--weight-semibold);
  }
  .kg dt {
    display: flex;
    align-items: center;
    gap: var(--space-1);
  }
  .kpi-grid .k-body {
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
    color: var(--text-tertiary);
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
    margin: calc(var(--space-2) * -1) 0 var(--space-1);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
  }
  .to-list:hover {
    color: var(--text-primary);
  }
  /* Curves (8) and Details (4) in one row, stretched to one height (DESIGN_DIRECTION §2.2); the per-class table under
     both. Each column's single panel fills its grid cell. */
  .detail {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    grid-template-areas: 'pr rail' 'cls cls';
    gap: var(--space-4);
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
  /* The matrix takes the width its cells want (up to 48 px each); the rail follows 32 px after it and takes the rest,
     sticking under the top bar while a tall matrix scrolls past. Where under 300 px would be left beside the matrix
     (many classes at the laptop tier, phones), the rail wraps under it at full width instead of squeezing the cells. */
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
    flex: 1 1 300px;
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
    max-width: 60ch;
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
  .d-sum :global(.ident) {
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
  /* Stacked in every layout: the label is long, and the value keeps its conditions on the same line. */
  .kv.ref,
  .two .kv.ref {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 2px;
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
  }
  .latency {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0 var(--space-2);
    text-align: left;
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
  @media (max-width: 1279px) {
    .k-body {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  /* Tablets and phones: the rail stacks under the main column (as on Run and New run). */
  @media (max-width: 1023px) {
    .detail {
      grid-template-columns: minmax(0, 1fr);
      grid-template-areas: 'pr' 'cls' 'rail';
    }
  }
  @media (pointer: coarse) {
    .k-help::after {
      content: '';
      position: absolute;
      inset: -12px;
    }
  }
  @media (max-width: 767px) {
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
</style>
