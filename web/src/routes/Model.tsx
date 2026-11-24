// One model (`#/models/<id>`): its scores, PR curves, confusion matrix and per-class metrics
// (GET /api/models/{id}/eval), where it came from, and export. The parts are in ./model/.
import ArrowLeft from '$icons/arrow-left';
import ChevronDown from '$icons/chevron-down';
import Download from '$icons/download';
import Copy from '$icons/copy';
import ScanSearch from '$icons/scan-search';
import TriangleAlert from '$icons/triangle-alert';
import { computed, css, effect, MediaQuery, signal, tick } from 'fw';
import { copyText, pageCommands, type PageCommand } from '$lib/shell/pageCommands';
import { hub } from '$lib/api/index';
import { fromModelEval } from '$lib/api/evaluation';
import { ApiError } from '$lib/api/client';
import { refreshOnModelsChange } from '$lib/api/models';
import { Resource } from '$lib/api/resource';
import type { Dataset, Model as ModelT, Run } from '$lib/api/types';
import { fallbackNotice, sourceShort, splitWord, type EvalSource } from '$lib/evaluated';
import { ago, bytes, date, num } from '$lib/format';
import { loadThroughput } from '$lib/throughput';
import { prefs } from '$lib/prefs';
import { href, router } from '$lib/router';
import Page from '$lib/shell/Page';
import { setCrumbs } from '$lib/shell/crumbs';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Menu from '$lib/ui/Menu';
import Panel from '$lib/ui/Panel';
import Tabs from '$lib/ui/Tabs';
import Segmented from '$lib/ui/Segmented';
import Skeleton from '$lib/ui/Skeleton';
import Ident from '$lib/ui/Ident';
import ExportDialog from './model/ExportDialog';
import { FROM_SCRATCH_META, origin } from '$lib/lineage';
import Lineage from '$lib/ui/Lineage';
import Meta from '$lib/ui/Meta';
import PrCurves from './model/PrCurves';
import Callout from './run/Callout';
import { MODEL_SECTIONS, sectionShown, type ModelSection } from './model/sections';
import { media } from '$lib/design/tokens';
import KpiSummary from './model/KpiSummary';
import DetailsList, { type ModelContext, type ProvenanceErrors } from './model/DetailsList';
import { finishedEpochs, weightsEpoch } from './model/weightsEpoch';
import ConfusionPanel from './model/ConfusionPanel';
import { datasetPresence, datasetsFor } from './overview/newDatasets';

export default function Model() {
  const id = computed(() => router.route.params.id ?? '');
  const model = new Resource((s) => hub.api.model(id.get(), s));
  // A run that ends, or is deleted, changes the models: this one's metrics, or whether it still exists.
  refreshOnModelsChange(model);
  /** The model went away while shown (its run was deleted): the page says so instead of showing what it was. */
  const gone = computed(() => (model.stale instanceof ApiError && model.stale.status === 404 ? model.stale : null));

  effect(() => {
    setCrumbs({ label: 'Models', path: '/models' }, { label: model.data?.name ?? id.get() });
  });

  // The source run, its dataset and the model it was fine-tuned from (its `weights`), for the provenance links.
  const runId = computed(() => model.data?.run_id ?? null);
  // The palette's This page: the header's actions and the id.
  pageCommands((): PageCommand[] => {
    const m = model.data;
    if (!m) return [];
    return [
      { id: 'model-export', label: `Export ${m.name}`, match: ['Export model', 'ONNX'], icon: Download, run: () => exportOpen.set(true) },
      { id: 'model-try', label: `Try ${m.name} in Playground`, match: ['Try model'], icon: ScanSearch, run: () => router.go(`/playground?model=${encodeURIComponent(m.id)}`) },
      { id: 'model-copy-id', label: 'Copy model id', match: [m.id], icon: Copy, run: () => void copyText(m.id, 'model id') },
    ];
  });
  const isNotFound = (e: unknown) => e instanceof ApiError && e.status === 404;

  // `epochs`: how many the run finished (model/weightsEpoch `finishedEpochs`); null while not known.
  const context = new Resource<ModelContext>(async (s) => {
    const rid = runId.get();
    if (!rid) return { run: null, dataset: null, parent: null, epochs: null, errors: {} };
    let run: Run | null = null;
    try {
      run = await hub.api.run(rid, s);
    } catch (e) {
      if (s.aborted || (e instanceof DOMException && e.name === 'AbortError')) throw e;
      if (isNotFound(e)) return { run: null, dataset: null, parent: null, epochs: null, errors: {} };
      return { run: null, dataset: null, parent: null, epochs: null, errors: { run: e } };
    }
    const from = origin(run.config.weights);
    // Run the three independent provenance lookups in parallel (cutting 4 sequential RTTs to 2).
    const [metricsRes, datasetRes, parentRes] = await Promise.allSettled([
      hub.api.runMetrics(rid, s, Math.max(1, run.step)),
      hub.api.dataset(run.config.dataset_id, s),
      from.kind === 'model' ? hub.api.model(from.id, s) : Promise.resolve(null),
    ]);
    if (s.aborted) throw new DOMException('Aborted', 'AbortError');
    const errors: ProvenanceErrors = {};
    let epochs: number | null;
    if (metricsRes.status === 'fulfilled') {
      epochs = finishedEpochs(run, metricsRes.value.epochs);
    } else {
      epochs = finishedEpochs(run, null);
      if (!isNotFound(metricsRes.reason)) errors.metrics = metricsRes.reason;
    }
    let dataset: Dataset | null = null;
    if (datasetRes.status === 'fulfilled') {
      dataset = datasetRes.value;
    } else if (!isNotFound(datasetRes.reason)) {
      errors.dataset = datasetRes.reason;
    }
    let parent: ModelT | null = null;
    if (parentRes.status === 'fulfilled') {
      parent = parentRes.value;
    } else if (!isNotFound(parentRes.reason)) {
      errors.parent = parentRes.reason;
    }
    return { run, dataset, parent, epochs, errors };
  });
  // A run resumed elsewhere that ends changes the models: the run and its epochs are read again with them.
  refreshOnModelsChange(context);
  const lineage = computed(() => {
    const c = context.data;
    return c?.run ? origin(c.run.config.weights, c.parent ? [c.parent] : []) : null;
  });

  // PR curves, confusion matrix and per-class metrics from GET /api/models/{id}/eval.
  const evaluation = new Resource((s) => hub.api.modelEval(id.get(), s));
  // This server's measured latency for the architecture (GET /api/system/throughput), where it has predicted with it.
  const tp = new Resource((s) => loadThroughput(hub.api, s));
  /** No registered dataset has this model's classes (an imported file): nothing to evaluate on, and no retry helps. */
  const unevaluable = computed(() => evaluation.error instanceof ApiError && evaluation.error.code === 'no_matching_dataset');
  /**
   * The dataset's folder is gone (`404 path_not_found`, "the folder of dataset '…' is gone", docs/API.md): the way back,
   * in this page's terms; the server's hint names endpoints, so ErrorState keeps it with the details.
   */
  const evalHint = computed(() => {
    const e = evaluation.error;
    return e instanceof ApiError && e.code === 'path_not_found' && /\bis gone\b/.test(e.message)
      ? 'Move the dataset’s folder back where it was, then Retry.'
      : undefined;
  });
  const notEvaluated = () => (
    <EmptyState
      title="Not evaluated"
      description={`No registered dataset has this model's ${m().classes.length === 1 ? 'class' : `${num(m().classes.length)} classes`}. Register one with the same classes, in the same order, to see its curves and per-class figures.`}
    />
  );
  const ev = computed(() => {
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
  const source = computed((): EvalSource => {
    const e = ev.get();
    if (e) return e.source;
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
  // The datasets the page links to (the run's, the evaluation's) stop linking once removed: the list is read every 5 s,
  // apart from the models, while the page names one, so one removed while the page is open unlinks (./overview/newDatasets).
  // Its error page (a model not found, or gone) names none.
  const named = () => ((model.error && !model.data) || gone.get() ? [] : [context.data?.dataset?.id, source.get().datasetId]);
  const presence = datasetPresence(datasetsFor(named), named);
  const removed = (dsId: string | null | undefined) => presence(dsId) === 'removed';
  /** Said once on the page, under the headline metrics, when the numbers are not on the validation split. */
  const notice = computed(() => fallbackNotice(source.get()));
  // The curves' qualifier names what was evaluated ("fasteners · validation split · IoU 0.5"); phones keep the split.
  const prMeta = computed(() => {
    const src = source.get();
    const where = phone.current ? (src.split ? `${splitWord(src.split)} split` : null) : sourceShort(src);
    return `${where ?? 'precision by recall'} · IoU 0.5`;
  });
  /** The curves panel's band / every-curve switch (many classes). */
  const allCurves = signal(false);
  const exportOpen = signal(false);
  // Phones: after the headline metrics, one section at a time (as on the run page) instead of a 4,000 px column.
  const phone = new MediaQuery(media.phone, false);
  const section = signal<ModelSection>('curves');
  /** The sticky picker's height: a section's own sticky bar (Classes' selection) holds just under it. */
  const pickH = signal(0);
  const showing = (part: ModelSection) => sectionShown(phone.current, section.get(), part);

  const smooth = (): ScrollBehavior => (matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth');
  // Phones: a class's "View curve" (in Classes) shows its traced curve, with a way back to the class it came from.
  const backTo = signal<number | null>(null);
  // The classes traced (clicked) and pointed at in the per-class table, drawn in the curves' panel.
  const prSelected = signal<number[]>([]);
  const prHovered = signal<number | null>(null);
  effect(() => {
    if (section.get() !== 'curves') backTo.set(null);
  });
  async function viewCurve(cls: number) {
    section.set('curves');
    backTo.set(cls);
    await tick();
    document.getElementById('pr')?.scrollIntoView({ block: 'start', behavior: smooth() });
    document.querySelector<HTMLElement>('#pr .back')?.focus({ preventScroll: true });
  }
  async function backToClasses() {
    const cls = backTo.peek();
    section.set('classes');
    await tick();
    const row = document.querySelector<HTMLElement>(`#per-class .disc[data-cls="${cls}"]`);
    row?.focus({ preventScroll: true });
    row?.scrollIntoView({ block: 'center', behavior: smooth() });
  }

  // Phones: the work before the summary (DESIGN_DIRECTION 12). The section picker sits right under the header, so it
  // never moves between sections; every section, Curves included, opens on one line, the headline and "All metrics",
  // which unfolds the 2 × 2 grid, so the chart is the first thing under the picker. What the numbers mean is behind
  // the grid's ⓘ.
  const kpiOpen = signal(false);
  const kpiFolded = () => phone.current;
  /**
   * How the weights are stored (the file's folo.dtype and folo.fused, docs/API.md): "f16 · fused, inference only".
   * Null when the server does not say; a dtype this client has no name for reads as unknown, as the API asks.
   */
  const stored = computed(() => {
    const m = model.data;
    const p = m?.precision === 'f32' || m?.precision === 'f16' ? m.precision : null;
    const fused = m?.fused === true;
    if (!p && !fused) return null;
    return [p, fused ? 'fused, inference only' : null].filter(Boolean).join(' · ');
  });
  // What the four numbers mean, one tap away (the strip itself keeps to values and a short qualifier).
  const helpOpen = signal(false);
  const aboutOpen = signal(false);
  // Phones: one summary line under the top bar; the rest of the provenance behind it.
  const metaOpen = signal(false);

  /** The band / every-curve switch (Segmented's value). */
  const curvesMode = { get: () => (allCurves.get() ? 'all' : 'band') as 'all' | 'band', set: (v: 'all' | 'band') => allCurves.set(v === 'all') };

  const m = () => model.data!;
  const run = () => context.data?.run;
  const done = () => context.data?.epochs ?? null;
  const head = () => m().head ?? 'o2m';

  // Where the weights started: the run's `weights`, a model (linked), an imported file, or nothing.
  const lineageNote = () => (
    <>
      {lineage.get() && lineage.get()!.kind !== 'scratch' ? (
        <span>
          fine-tuned from <Lineage origin={lineage.get()!} />
        </span>
      ) : lineage.get()?.kind === 'scratch' ? (
        <span>{FROM_SCRATCH_META}</span>
      ) : null}
    </>
  );
  const provenance = () => (
    <>
      {m().run_id ? (
        <>
          <span>
            trained by{' '}
            <a class="link" href={href(`/runs/${m().run_id}`)}>
              {run()?.name ?? m().run_id}
            </a>
          </span>
          {weightsEpoch(m(), run(), done()) ? <span>{weightsEpoch(m(), run(), done())}</span> : null}
          {lineageNote()}
        </>
      ) : m().source === 'published' ? (
        <span title={m().about}>published weights</span>
      ) : (
        <span>imported</span>
      )}
      <span class="faint" title={date(m().created)}>
        created {ago(m().created)}
      </span>
    </>
  );
  return (
    <>
      {(model.error && !model.data) || gone.get() ? (
        <Page title="Model" actions={() => <Button icon={ArrowLeft} href={href('/models')}>All models</Button>}>
          {/* Nothing to frame: the message frameless on the content edge (DESIGN_DIRECTION 8), as on Run and Not found. */}
          <ErrorState error={gone.get() ?? model.error} retry={() => model.refresh()} title="Could not load this model" />
        </Page>
      ) : !model.data ? (
        <Page title="Model" sub={() => <Skeleton w="320px" h="14px" />}>
          <Skeleton h="106px" r="var(--radius-lg)" />
          <Skeleton h="480px" r="var(--radius-lg)" />
        </Page>
      ) : (
        <>
          <Page
            title={m().name}
            inlineActions={phone.current}
            sub={() => (
              <>
                {phone.current ? (
                  <>
                    <button type="button" class="meta-line num" aria-expanded={metaOpen.get()} aria-controls="model-meta" onclick={() => metaOpen.set(!metaOpen.get())}>
                      <Meta line>
                        <Ident value={m().arch} />
                        <span>{m().classes.length} classes</span>
                        <span>{m().imgsz} px</span>
                      </Meta>
                      <ChevronDown size={14} aria-hidden="true" />
                    </button>{' '}
                    {metaOpen.get() ? (
                      <div class="meta-more" id="model-meta">
                        <Meta>
                          <span class="num">{bytes(m().bytes, prefs.units)}</span>
                          {provenance()}
                        </Meta>
                      </div>
                    ) : null}
                  </>
                ) : (
                  // One line, its last item ("created yesterday") never alone on a second one (DESIGN_DIRECTION §10): the
                  // file size is the strip's last tile, not repeated here.
                  <Meta>
                    <Ident value={m().arch} />
                    <span class="num">{m().classes.length} classes</span>
                    <span class="num">{m().imgsz} px</span>
                    {provenance()}
                  </Meta>
                )}
              </>
            )}
            actions={() => (
              <>
                {phone.current ? (
                  <>
                    <Button variant="primary" icon={ScanSearch} aria-label="Try in Playground" href={href(`/playground?model=${encodeURIComponent(m().id)}`)}>
                      Try
                    </Button>
                    <Menu label="More model actions" items={[{ label: 'Export', icon: Download, onclick: () => exportOpen.set(true) }]} />
                  </>
                ) : (
                  <>
                    <Button icon={Download} onclick={() => exportOpen.set(true)}>
                      Export
                    </Button>
                    <Button variant="primary" icon={ScanSearch} href={href(`/playground?model=${encodeURIComponent(m().id)}`)}>
                      Try in Playground
                    </Button>
                  </>
                )}
              </>
            )}
          >
            {/* One strip of numbers in ink: the headline metric, then the ones that support it, each with a short qualifier;
                the ⓘ opens what they mean. Phones: a 2 × 2 summary of label over value, the headline first, inside the
                sections (below). */}
            {!phone.current ? <KpiSummary m={m()} source={source.get()} ev={ev.get()} head={head()} bind:helpOpen={helpOpen} /> : null}

            {notice.get() ? (
              // Once on the page, in every layout and phone section: these are not validation numbers.
              <Callout tone="warning" icon={TriangleAlert} title={notice.get()!.title} role="note" label="Evaluation split">
                {notice.get()!.why}
              </Callout>
            ) : null}

            {phone.current ? (
              // Stays reachable under the top bar while a section scrolls.
              <div class="phone-pick" bind:offsetHeight={pickH}>
                <Tabs label="Model sections" idPrefix="model-sec" bind:value={section} tabs={MODEL_SECTIONS} />
              </div>
            ) : null}

            {/* The parts, in one frame: on phones it is the sections' tab panel; wider, a column with the page's own gap. */}
            <div
              class="parts"
              id="model-sec-panel"
              role={phone.current ? 'tabpanel' : undefined}
              aria-labelledby={phone.current ? `model-sec-${section.get()}` : undefined}
              style={{ '--pick-bottom': phone.current ? `calc(var(--size-topbar) + var(--space-2) + ${pickH.get()}px)` : undefined }}
            >
              {phone.current ? (
                <KpiSummary m={m()} source={source.get()} ev={ev.get()} head={head()} grid kpiFolded={kpiFolded()} bind:kpiOpen={kpiOpen} bind:helpOpen={helpOpen} />
              ) : null}

              {/* Top to bottom (§4 Model): one Precision-recall panel, the plot with its per-class table as the legend
                  (side by side from 1024 px, the table under the plot below), the confusion matrix, then Details. Phones:
                  one section at a time, the curves and the table as two sections sharing the traced and pointed-at
                  classes. */}
              <div class="part" class:off={!showing('curves')}>
                <Panel
                  title="Precision–recall"
                  meta={prMeta.get()}
                  pad="none"
                  id="pr"
                  actions={() => (
                    // Many classes: the band / every-curve switch is the chart's one header control, so the legend row keeps
                    // one line. Phones put it on the Trace row (PrCurves), never on a header row of its own that wraps
                    // under the title.
                    <>
                      {ev.get() && m().classes.length > 10 && !phone.current ? (
                        <Segmented
                          size="sm"
                          label="Curves shown"
                          bind:value={curvesMode}
                          options={[
                            { value: 'band', label: 'Band' },
                            { value: 'all', label: 'All curves' },
                          ]}
                        />
                      ) : null}
                    </>
                  )}
                >
                  {ev.get() ? (
                    <PrCurves
                      classes={m().classes}
                      ev={ev.get()!}
                      // A file's stored AP50 (imported, published) may be on another dataset than these curves.
                      map50={m().run_id ? (m().metrics?.map50 ?? null) : null}
                      view={phone.current ? 'curves' : 'all'}
                      bind:allCurves={allCurves}
                      modeInHeader
                      bind:selected={prSelected}
                      bind:hovered={prHovered}
                      onback={backTo.get() !== null ? backToClasses : undefined}
                    />
                  ) : unevaluable.get() ? (
                    notEvaluated()
                  ) : evaluation.error ? (
                    <ErrorState error={evaluation.error} retry={() => evaluation.refresh()} compact title="Could not load the evaluation" hint={evalHint.get()} />
                  ) : (
                    <div class="sk">
                      <Skeleton h="24px" w="200px" />
                      <Skeleton h="360px" />
                    </div>
                  )}
                </Panel>
              </div>
              {phone.current ? (
                <div class="part" class:off={!showing('classes')}>
                  <Panel title="Per class" meta={`${m().classes.length} classes · IoU 0.5`} pad="none" id="per-class">
                    {ev.get() ? (
                      <PrCurves
                        classes={m().classes}
                        ev={ev.get()!}
                        map50={m().run_id ? (m().metrics?.map50 ?? null) : null}
                        view="classes"
                        bind:selected={prSelected}
                        bind:hovered={prHovered}
                        onviewcurve={viewCurve}
                      />
                    ) : unevaluable.get() ? (
                      notEvaluated()
                    ) : evaluation.error ? (
                      // The failure is said in the section the reader is on: the curves are another section here.
                      <ErrorState error={evaluation.error} retry={() => evaluation.refresh()} compact title="Could not load the evaluation" hint={evalHint.get()} />
                    ) : (
                      <div class="sk">
                        {[0, 1, 2, 3, 4, 5].map(() => (
                          <Skeleton h="16px" />
                        ))}
                      </div>
                    )}
                  </Panel>
                </div>
              ) : null}
              <div class="part" class:off={!showing('confusion')}>
                <ConfusionPanel m={m()} ev={ev.get()} source={source.get()} phone={phone} bind:section={section} evaluation={evaluation} />
              </div>
              {/* Every fact about the model in one full-width panel, three columns (two on tablets). Phones: its own
                  section, no title repeating the Details picker above it. */}
              <div class="part" class:off={!showing('details')}>
                <Panel title={phone.current ? undefined : 'Details'} id="details">
                  <DetailsList
                    m={m()}
                    tp={tp}
                    run={run()}
                    context={context}
                    source={source.get()}
                    ev={ev.get()}
                    lineage={lineage.get()}
                    unevaluated={unevaluable.get()}
                    stored={stored.get()}
                    head={head()}
                    removed={removed}
                    bind:aboutOpen={aboutOpen}
                  />
                </Panel>
              </div>
            </div>
          </Page>
          <ExportDialog model={m()} bind:open={exportOpen} />
        </>
      )}
    </>
  );
}

css`
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
  .part {
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
  @media (max-width: 767px) {
    /* "View curve" and "Back to Classes" land the curves panel under the top bar and the section picker. */
    .part :global(#pr) {
      scroll-margin-top: calc(var(--size-topbar) + var(--space-16));
    }
  }
`;
