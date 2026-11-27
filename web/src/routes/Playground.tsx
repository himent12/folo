// Playground (`#/playground`): a model's detections on an image, run on the server or in this browser through
// folo-wasm, with the thresholds, the detections and their timing. The parts are in ./playground/.
import Box from '$icons/box';
import RotateCcw from '$icons/rotate-ccw';
import { computed, css, effect, MediaQuery, onCleanup, onWindow, signal, untrack, type Signal } from 'fw';
import type { Prediction } from '$lib/api/types';
import { media } from '$lib/design/tokens';
import { filterDetections } from '$lib/detect/nms';
import { BrowserEngine, type Backend } from '$lib/inference/browser';
import { sentence } from '$lib/format';
import { isOom } from '$lib/oom';
import { router } from '$lib/router';
import Page from '$lib/shell/Page';
import { setCrumbs } from '$lib/shell/crumbs';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Panel from '$lib/ui/Panel';

import ComparePickers from './playground/ComparePickers';
import CompareStage from './playground/CompareStage';
import CompareSummaryPanel from './playground/CompareSummary';
import DatasetPicker from './playground/DatasetPicker';
import MediaCaptureDialog from './playground/MediaCaptureDialog';
import PlaygroundActions from './playground/PlaygroundActions';
import PlaygroundTiming from './playground/PlaygroundTiming';
import { formatSignedDelta } from './playground/compare';
import { runJsonExport, runPngExport } from './playground/exportActions';
import { looksLikeFolo } from './playground/folo';
import { holdNoticeFocus } from './playground/noticeFocus';
import ModelPickers from './playground/ModelPickers';
import ImagePanel from './playground/ImagePanel';
import ResultsPanel from './playground/ResultsPanel';
import ClassList from './playground/ClassList';
import PhoneDetails from './playground/PhoneDetails';
import SidePanels from './playground/SidePanels';
import Rail from './playground/Rail';
import DisplayToggles from './playground/DisplayToggles';
import { MoreBody, RailThresholds, SecondarySliders } from './playground/Thresholds';
import { DEFAULTS, type RunOn } from './playground/predict';
import { ACCEPT } from './playground/intake';
import { keepSession } from './playground/keepSession';
import { sampleImages } from './playground/samples';
import { playgroundPath } from './playground/session';
import { createCompareController } from './playground/useCompare';
import { useModelCatalog } from './playground/useModelCatalog';
import { useSinglePredict } from './playground/useSinglePredict';
import { usePlaygroundSource } from './playground/useSource';
import { useWasmAvailability } from './playground/useWasmProbe';
import {
  areDefaultThresholds,
  classCounts,
  detectionIndex,
  nextRowIndex,
  predictionJson,
  scrollRowIntoView,
  smallerModelOf,
  sourceDims,
  sourceFullName,
  sourceShortName,
  visibleClasses,
  visibleRows,
  whereLabel,
  type Det,
} from './playground/viewState';
import FirstStepAction from './firstStep';

export default function Playground() {
  setCrumbs({ label: 'Playground' });

  const phone = new MediaQuery(media.phone, false);
  const narrow = new MediaQuery(media.upToLaptop, false);
  const tablet = computed(() => narrow.current && !phone.current);
  const moreOpen = signal(false);
  const sheetEl = signal<HTMLElement>();
  const moreBtn = signal<HTMLElement>();
  const pickerOpen = signal(false);
  const captureOpen = signal(false);
  const captureVideo = signal<File | null>(null);
  let fileInput!: HTMLInputElement;

  const pred = signal<Prediction | null>(null);
  const runOn = signal<RunOn>('server');
  const engine = new BrowserEngine();
  onCleanup(() => engine.dispose());
  const { webgpu, browserOff, runOnNote } = useWasmAvailability(runOn);

  const localWeights = signal(new Map<string, File>());
  const weightsMissing = signal(false);
  const weightsTarget = signal<'a' | 'b'>('a');
  let weightsInput!: HTMLInputElement;
  const where = computed(() => whereLabel(pred.get()));
  const badFile = signal<{ name: string; why: string | null } | null>(null);
  const badFileB = signal<{ name: string; why: string | null } | null>(null);

  const predicting = signal(false);
  const error = signal<unknown>(null);
  const retryTick = signal(0);

  const conf = signal(DEFAULTS.conf);
  const iou = signal(DEFAULTS.iou);
  const maxDet = signal(DEFAULTS.maxDet);
  const labels = signal(true);
  const scores = signal(true);
  const truth = signal(false);
  const hidden = signal<Set<number>>(new Set());
  const hover = signal<Det | null>(null);
  const picked = signal<Det | null>(null);
  const tbody = signal<HTMLTableSectionElement>();
  const tab = signal<'detections' | 'classes' | 'json'>('detections');
  const exportingPng = signal(false);

  let syncModelUrl: ((id: string) => void) | null = null;
  let clearCompare: () => void = () => {};

  const {
    models,
    noDatasets,
    choices,
    ranked,
    queryModel,
    pickedModel,
    model,
    modelId,
    o2o,
    pickedMeta,
    chooseModel,
  } = useModelCatalog((id) => {
    if (syncModelUrl) syncModelUrl(id);
    else router.go(playgroundPath({ model: id, source: source.peek(), conf: conf.peek(), iou: iou.peek(), maxDet: maxDet.peek() }), { replace: true });
  });

  const {
    source,
    src,
    seg,
    dragging,
    imageError,
    retryImage,
    choose,
    setSource,
    setFile,
    clear,
    onSeg,
    onpaste,
    ondrop,
    ondragover,
    ondragleave,
    isDestroyed,
  } = usePlaygroundSource({
    engine,
    hasModels: () => ranked.peek().length > 0,
    onResetState: () => {
      pred.set(null);
      error.set(null);
      weightsMissing.set(false);
      hover.set(null);
      picked.set(null);
      hidden.set(new Set());
      clearCompare();
    },
    onClearExtra: () => moreOpen.set(false),
    onDropVideo: (videoFile) => {
      captureVideo.set(videoFile);
      captureOpen.set(true);
    },
    openFilePicker: () => fileInput.click(),
    pickerOpen,
  });

  const sheetShown = computed(() => moreOpen.get() && narrow.current && !!source.get());
  const sheetTitle = computed(() => (tablet.get() ? 'More settings' : 'Thresholds'));

  effect(() => {
    if (sheetShown.get()) sheetEl.get()?.focus();
  });
  effect(() => {
    if (!narrow.current) untrack(() => moreOpen.set(false));
  });
  effect(() => {
    if (!sheetShown.get() || !tablet.get()) return;
    const away = (e: PointerEvent) => {
      const t = e.target as Node;
      if (!sheetEl.peek()?.contains(t) && !moreBtn.peek()?.contains(t)) moreOpen.set(false);
    };
    document.addEventListener('pointerdown', away);
    return () => document.removeEventListener('pointerdown', away);
  });
  function closeSheet() {
    moreOpen.set(false);
    moreBtn.peek()?.querySelector<HTMLElement>('button')?.focus({ preventScroll: true });
  }

  const samples = sampleImages(model);
  const weightPickSeq = new Map<string, number>();

  async function pickWeights(e: Event & { currentTarget: HTMLInputElement }) {
    const input = e.currentTarget;
    const f = input.files?.[0];
    input.value = '';
    const side = weightsTarget.peek();
    const id = side === 'b' ? cmp.modelBId.peek() : modelId.peek();
    if (!f || !id) return;
    const seq = (weightPickSeq.get(id) ?? 0) + 1;
    weightPickSeq.set(id, seq);
    if (!(await looksLikeFolo(f))) {
      if (!isDestroyed() && weightPickSeq.get(id) === seq) {
        if (side === 'b' && cmp.modelBId.peek() === id) badFileB.set({ name: f.name, why: null });
        else if (side === 'a' && modelId.peek() === id) badFile.set({ name: f.name, why: null });
      }
      return;
    }
    if (isDestroyed() || weightPickSeq.get(id) !== seq) return;
    if (side === 'b' && cmp.modelBId.peek() === id) badFileB.set(null);
    else if (side === 'a' && modelId.peek() === id) badFile.set(null);
    localWeights.set(new Map(localWeights.peek()).set(id, f));
  }

  function dropLocal(id: string, f: File, e: unknown, targetBad: Signal<{ name: string; why: string | null } | null> = badFile) {
    const cur = localWeights.peek();
    if (cur.get(id) === f) {
      const next = new Map(cur);
      next.delete(id);
      localWeights.set(next);
    }
    targetBad.set({ name: f.name, why: e instanceof Error ? sentence(e.message) : null });
  }

  const result = computed(() => {
    const p = pred.get();
    if (!p) return null;
    const t0 = performance.now();
    // An end-to-end (o2o) model predicts without NMS (docs/MODEL.md §6.6): an IoU of 1 suppresses nothing, as the server.
    const boxes = filterDetections(p.boxes, { conf: conf.get(), iou: o2o.get() ? 1 : iou.get(), maxDet: maxDet.get() });
    return { boxes, ms: performance.now() - t0 };
  });
  const kept = computed(() => result.get()?.boxes ?? []);
  const classes = computed(() => model.get()?.classes ?? []);
  const only = computed(() => visibleClasses(classes.get(), hidden.get()));
  const rows = computed(() => visibleRows(kept.get(), hidden.get()));
  const counts = computed(() => classCounts(kept.get()));
  const maxCount = computed(() => Math.max(1, ...counts.get().map((c) => c[1])));

  // Compare A/B controller: strictly inert (zero network calls) while Compare mode is off.
  const cmp = createCompareController({
    ranked: () => ranked.get(),
    choices: () => choices.get(),
    modelA: () => model.get(),
    modelAId: () => modelId.get(),
    chooseModelA: chooseModel,
    runOnA: runOn,
    predA: pred,
    hoverA: hover,
    pickedA: picked,
    resultA: () => result.get(),
    rowsA: () => rows.get(),
    o2oA: () => o2o.get(),
    source,
    conf,
    iou,
    maxDet,
    hidden,
    classesA: () => classes.get(),
    browserOff: () => browserOff.get(),
    localWeights,
    dropLocalB: (id, f, e) => dropLocal(id, f, e, badFileB),
    badFileB,
    engine,
  });
  clearCompare = cmp.clearResults;

  // The image, its result and the thresholds outlive the page for the tab; a dataset image and the thresholds are in
  // the URL (./playground/session).
  const restored = keepSession({
    source,
    pred,
    runOn: runOn as Signal<string>,
    conf,
    iou,
    maxDet,
    labels,
    scores,
    truth,
    hidden,
    compare: cmp.compare,
    modelB: () => cmp.modelBId.get() ?? cmp.pickedModelB.get(),
    pickModelB: (id) => cmp.pickedModelB.set(id),
    compareView: cmp.compareView,
    setSource,
    choose,
    model: () => modelId.get() ?? queryModel.get() ?? null,
    pickModel: (id) => pickedModel.set(id),
    onModelUrl: (write) => (syncModelUrl = write),
  });

  let lastModelId: string | null = null;
  effect(() => {
    const id = modelId.get();
    if (id !== null && lastModelId !== null && id !== lastModelId) untrack(() => hidden.set(new Set()));
    if (id !== null) lastModelId = id;
  });

  const cut = useSinglePredict({
    modelId: () => modelId.get(),
    source,
    runOn,
    localWeights,
    retryTick,
    restored,
    consumeReuse: cmp.consumeReuseA,
    pred,
    predicting,
    error,
    weightsMissing,
    hidden,
    hover,
    picked,
    conf,
    engine,
    dropLocal: (id, f, e) => dropLocal(id, f, e, badFile),
  });

  // Out of GPU memory on the server (507): this browser has its own device, and a smaller model of the same classes
  // may fit where this one does not.
  const oom = computed(() => runOn.get() === 'server' && isOom(error.get()));
  const smaller = computed(() => smallerModelOf(model.get(), ranked.get()));
  const localBackend: Backend = webgpu ? 'webgpu' : 'cpu';

  // What stands between the image and its boxes, shown in a strip under the image (which stays in view).
  const notice = computed(() => (!source.get() || predicting.get() ? null : weightsMissing.get() ? 'file' : error.get() && !pred.get() ? 'error' : null));
  effect(() => {
    void modelId.get();
    void runOn.get();
    badFile.set(null);
  });
  // Focus moves to the notice's heading as it appears, so it is announced and its actions are next in tab order. The
  // move never scrolls the page, and the heading keeps focus while the notice stays (./playground/noticeFocus).
  const noticeEl = signal<HTMLElement>();
  effect(() => {
    const el = noticeEl.get();
    if (el) return holdNoticeFocus(el);
  });

  const selectedIdx = computed(() => detectionIndex(picked.get(), kept.get(), hidden.get()));
  const activeIdx = computed(() => detectionIndex(hover.get(), kept.get(), hidden.get()) ?? selectedIdx.get());
  /** In Compare mode, whether Model B is currently shown in the Detections table. */
  const showingB = computed(() => cmp.compare.get() && cmp.inspectSide.get() === 'b');
  const tableResult = computed(() => (showingB.get() ? cmp.resultB.get() : result.get()));
  const tableRows = computed(() => (showingB.get() ? cmp.rowsB.get() : rows.get()));
  const tableSelectedIdx = computed(() => (showingB.get() ? cmp.selectedIdxB.get() : selectedIdx.get()));
  const tableActiveIdx = computed(() => (showingB.get() ? cmp.activeIdxB.get() : activeIdx.get()));
  /** The row that takes Tab focus (roving tabindex): the selected one, else the first. */
  const tabRow = computed(() => tableSelectedIdx.get() ?? tableRows.get()[0]?.i ?? null);

  const rowEl = (i: number) => tbody.peek()?.querySelector<HTMLTableRowElement>(`tr[data-det="${i}"]`) ?? null;

  /** A box picked on the image: bring its row into view inside the list without moving the page. */
  function pickFromImage(i: number) {
    const b = kept.peek()[i];
    if (!b) return;
    const switched = (cmp.compare.peek() && cmp.inspectSide.peek() !== 'a') || tab.peek() !== 'detections';
    if (cmp.compare.peek() && cmp.inspectSide.peek() !== 'a') cmp.inspectSide.set('a');
    const next = picked.peek() === b ? null : b;
    picked.set(next);
    if (next) {
      if (tab.peek() !== 'detections') tab.set('detections');
      if (switched) queueMicrotask(() => scrollRowIntoView(rowEl(i), narrow.peek()));
      else scrollRowIntoView(rowEl(i), narrow.peek());
    }
  }

  function pickFromImageB(i: number) {
    const b = cmp.keptB.peek()[i];
    if (!b) return;
    const switched = cmp.inspectSide.peek() !== 'b' || tab.peek() !== 'detections';
    if (cmp.inspectSide.peek() !== 'b') cmp.inspectSide.set('b');
    const next = cmp.pickedB.peek() === b ? null : b;
    cmp.pickedB.set(next);
    if (next) {
      if (tab.peek() !== 'detections') tab.set('detections');
      if (switched) queueMicrotask(() => scrollRowIntoView(rowEl(i), narrow.peek()));
      else scrollRowIntoView(rowEl(i), narrow.peek());
    }
  }

  function onrowkey(e: KeyboardEvent, k: number) {
    const list = tableRows.peek();
    const to = nextRowIndex(e.key, k, list.length);
    if (to === null) return;
    e.preventDefault();
    const r = list[to];
    if (r) rowEl(r.i)?.focus();
  }

  function onwindowkey(e: KeyboardEvent) {
    if (e.defaultPrevented || e.key !== 'Escape' || pickerOpen.peek() || captureOpen.peek()) return;
    if (document.querySelector('dialog[open]')) return;
    if (sheetShown.peek()) {
      e.preventDefault();
      e.stopPropagation();
      closeSheet();
    } else if (picked.peek() || cmp.pickedB.peek()) {
      picked.set(null);
      cmp.pickedB.set(null);
    }
  }

  function onwindowdown(e: PointerEvent) {
    if ((picked.peek() || cmp.pickedB.peek()) && !(e.target as Element | null)?.closest?.('[data-det], [data-box], .zoom-hud')) {
      picked.set(null);
      cmp.pickedB.set(null);
    }
  }

  onWindow('paste', onpaste);
  onWindow('keydown', onwindowkey);
  onWindow('pointerdown', onwindowdown);

  const dims = computed(() => sourceDims(pred.get(), source.get()));
  const sourceName = computed(() => sourceFullName(source.get()));
  const shortName = computed(() => sourceShortName(source.get()));
  const truthBoxes = computed(() => {
    const s = source.get();
    return truth.get() && s?.kind === 'dataset' ? s.img.boxes : [];
  });
  /** Dataset ground-truth visibility set resolved by class name against `source.ds.classes`. */
  const truthOnly = computed(() => {
    const s = source.get();
    if (s?.kind !== 'dataset') return null;
    const names = cmp.compare.get()
      ? cmp.allHiddenNames.get()
      : new Set([...hidden.get()].map((i) => classes.get()[i] ?? `class ${i}`));
    if (!names.size) return null;
    return new Set(s.ds.classes.map((name, idx) => (names.has(name) ? -1 : idx)).filter((idx) => idx >= 0));
  });

  const singleJson = computed(() => {
    const h = hidden.get();
    const cls = classes.get();
    const hiddenClasses = h.size ? [...h].sort((a, b) => a - b).map((c) => cls[c] ?? `class ${c}`) : undefined;
    return predictionJson(
      pred.get(),
      !!result.get(),
      rows.get().map((r) => r.b),
      {
        conf: conf.get(),
        iou: iou.get(),
        maxDet: maxDet.get(),
        o2o: o2o.get(),
        hiddenClasses,
      },
    );
  });
  const activeJson = computed(() => (cmp.compare.get() ? cmp.compareJson.get() : singleJson.get()));
  const canExport = computed(() => !!source.get() && !!dims.get() && (cmp.compare.get() ? !!result.get() || !!cmp.resultB.get() : !!result.get()));
  const effectiveO2o = computed(() => (cmp.compare.get() ? o2o.get() && cmp.o2oB.get() : o2o.get()));

  async function onExportPng() {
    const s = source.peek();
    const d = dims.peek();
    if (!s || !d || exportingPng.peek()) return;
    exportingPng.set(true);
    try {
      await runPngExport({
        compare: cmp.compare.peek(),
        compareView: cmp.compareView.peek(),
        splitPct: cmp.splitPct.peek(),
        viewportW: cmp.zoom.viewportW.peek(),
        viewportH: cmp.zoom.viewportH.peek(),
        zoomZ: cmp.zoom.z.peek(),
        zoomNx: cmp.zoom.nx.peek(),
        source: s,
        src: src.peek(),
        dims: d,
        shortName: shortName.peek(),
        modelA: model.peek(),
        modelB: cmp.modelB.peek(),
        predA: pred.peek(),
        predB: cmp.predB.peek(),
        boxesA: rows.peek().map((r) => r.b),
        boxesB: cmp.rowsB.peek().map((r) => r.b),
        classesA: classes.peek(),
        classesB: cmp.classesB.peek(),
        truthOnly: truthOnly.peek(),
        truthBoxes: truthBoxes.peek(),
        labels: labels.peek(),
        scores: scores.peek(),
      });
    } finally {
      exportingPng.set(false);
    }
  }

  function onExportJson() {
    runJsonExport({
      shortName: shortName.peek(),
      compare: cmp.compare.peek(),
      modelAName: model.peek()?.name,
      modelBName: cmp.modelB.peek()?.name,
      payload: activeJson.peek(),
    });
  }

  function toggleClass(c: number) {
    const next = new Set(hidden.peek());
    if (next.has(c)) next.delete(c);
    else next.add(c);
    hidden.set(next);
  }

  function resetThresholds() {
    conf.set(DEFAULTS.conf);
    iou.set(DEFAULTS.iou);
    maxDet.set(DEFAULTS.maxDet);
  }
  const isDefault = computed(() => areDefaultThresholds(conf.get(), iou.get(), maxDet.get(), effectiveO2o.get()));

  // ---------- snippets ----------
  const displayToggles = () => <DisplayToggles bind:labels={labels} bind:scores={scores} bind:truth={truth} dataset={source.get()?.kind === 'dataset'} />;

  const secondarySliders = (inline: boolean) => <SecondarySliders inline={inline} o2o={effectiveO2o.get()} bind:iou={iou} bind:maxDet={maxDet} />;
  const railThresholds = () => <RailThresholds bind:conf={conf} bind:iou={iou} bind:maxDet={maxDet} o2o={effectiveO2o.get()} />;
  const moreBody = () => <MoreBody tablet={tablet.get()} o2o={effectiveO2o.get()} bind:conf={conf} bind:iou={iou} bind:maxDet={maxDet} toggles={displayToggles} />;

  const resetButton = () => (
    <Button size="sm" variant="ghost" icon={RotateCcw} disabled={isDefault.get()} onclick={resetThresholds}>
      Reset
    </Button>
  );

  function clearAllHidden() {
    hidden.set(new Set());
    cmp.hiddenBNames.set(new Set());
  }

  const showAllButton = () => (
    <>
      {hidden.get().size || (cmp.compare.get() && cmp.hiddenBNames.get().size) ? (
        <Button size="sm" variant="ghost" onclick={clearAllHidden}>
          Show all
        </Button>
      ) : null}
    </>
  );

  function toggleClassB(cB: number) {
    const nameB = cmp.classesB.peek()[cB] ?? `class ${cB}`;
    const idxA = classes.peek().indexOf(nameB);
    if (idxA >= 0) toggleClass(idxA);
    else cmp.toggleHiddenBName(nameB);
  }

  const classList = () => (
    <ClassList
      result={showingB.get() ? cmp.resultB.get() : result.get()}
      counts={showingB.get() ? cmp.countsB.get() : counts.get()}
      pred={showingB.get() ? cmp.predB.get() : pred.get()}
      hidden={showingB.get() ? cmp.hiddenB.get() : hidden.get()}
      classes={showingB.get() ? cmp.classesB.get() : classes.get()}
      maxCount={showingB.get() ? cmp.maxCountB.get() : maxCount.get()}
      toggleClass={showingB.get() ? toggleClassB : toggleClass}
    />
  );

  // Stable snippet reference so `<CompareStage>` never remounts when Model A's `pred` or `dims` object identity changes.
  const compareStage = () => (
    <CompareStage
      source={source.get()!}
      src={src.get()}
      dims={dims.get()!}
      sourceName={sourceName.get()}
      onImageLoad={(ok) => imageError.set(!ok)}
      view={cmp.compareView.get()}
      bind:splitPct={cmp.splitPct}
      zoom={cmp.zoom}
      labels={labels.get()}
      scores={scores.get()}
      truthBoxes={truthBoxes.get()}
      truthOnly={truthOnly.get()}
      modelA={model.get()}
      predA={pred.get()}
      keptA={kept.get()}
      visibleCountA={rows.get().length}
      classesA={classes.get()}
      onlyA={only.get()}
      activeIdxA={activeIdx.get()}
      selectedIdxA={selectedIdx.get()}
      predictingA={predicting.get()}
      errorA={error.get()}
      onHoverA={(b) => hover.set(b)}
      onSelectA={pickFromImage}
      modelB={cmp.modelB.get()}
      predB={cmp.predB.get()}
      keptB={cmp.keptB.get()}
      visibleCountB={cmp.rowsB.get().length}
      classesB={cmp.classesB.get()}
      onlyB={cmp.onlyB.get()}
      activeIdxB={cmp.activeIdxB.get()}
      selectedIdxB={cmp.selectedIdxB.get()}
      predictingB={cmp.predictingB.get()}
      errorB={cmp.errorB.get()}
      weightsMissingB={cmp.weightsMissingB.get()}
      badFileB={badFileB.get()}
      runOnB={cmp.runOnB.get()}
      onHoverB={(b) => cmp.hoverB.set(b)}
      onSelectB={pickFromImageB}
      onRetryB={() => cmp.retryTickB.set((t) => t + 1)}
      onChooseWeightsB={() => {
        weightsTarget.set('b');
        weightsInput.click();
      }}
      onServerB={() => cmp.runOnB.set('server')}
      engine={engine}
    />
  );

  const timingBody = () => (
    <PlaygroundTiming
      compare={cmp.compare.get()}
      narrow={narrow.current}
      modelA={model.get()}
      modelB={cmp.modelB.get()}
      predA={pred.get()}
      predB={cmp.predB.get()}
      predictingA={predicting.get()}
      predictingB={cmp.predictingB.get()}
      resultMsA={result.get()?.ms ?? null}
      adapter={engine.info?.adapter}
    />
  );

  const intro = () => <span>Run a model on any image. Thresholds re-filter instantly in the browser.</span>;

  // The model and where it runs: the rail's first section from 1024 px up, a toolbar row under the title below that.
  const pickers = (inRail: boolean) =>
    cmp.compare.get() ? (
      <ComparePickers
        rail={inRail}
        toolbar={!inRail}
        ranked={ranked.get()}
        choices={choices.get()}
        modelAId={modelId.get()}
        modelBId={cmp.modelBId.get()}
        modelA={model.get()}
        modelB={cmp.modelB.get()}
        metaA={pickedMeta.get()}
        metaB={cmp.metaB.get()}
        chooseModelA={chooseModel}
        chooseModelB={(id) => cmp.pickedModelB.set(id)}
        swapModels={cmp.swapModels}
        bind:runOnA={runOn}
        bind:runOnB={cmp.runOnB}
        bind:compareView={cmp.compareView}
        webgpu={webgpu}
        browserOff={browserOff.get()}
      />
    ) : (
      <ModelPickers
        rail={inRail}
        toolbar={!inRail}
        ranked={ranked.get()}
        modelId={modelId.get()}
        choices={choices.get()}
        chooseModel={chooseModel}
        model={model.get()}
        pickedMeta={pickedMeta.get()}
        bind:runOn={runOn}
        webgpu={webgpu}
        browserOff={browserOff.get()}
        runOnNote={runOnNote.get()}
      />
    );

  const compareSummary = (inRail: boolean) => (
    <CompareSummaryPanel
      rail={inRail}
      summary={cmp.summary.get()}
      modelAName={model.get()?.name ?? 'Model A'}
      modelBName={cmp.modelB.get()?.name ?? 'Model B'}
      hasAnyResult={!!result.get() || !!cmp.resultB.get()}
      predictingA={predicting.get()}
      predictingB={cmp.predictingB.get()}
      inspectSide={cmp.inspectSide.get()}
      onInspectSide={(side) => cmp.inspectSide.set(side)}
    />
  );

  // The detections: a panel under the image below 1024 px, the rail's Detections section (with a Classes tab) above.
  const results = (rail: boolean) => (
    <ResultsPanel
      rail={rail}
      classCount={showingB.get() ? cmp.countsB.get().length : counts.get().length}
      classList={classList}
      showAllButton={showAllButton}
      onShowAll={clearAllHidden}
      result={tableResult.get()}
      bind:tab={tab}
      rows={tableRows.get()}
      predicting={showingB.get() ? cmp.predictingB.get() : predicting.get()}
      notice={showingB.get() ? (cmp.weightsMissingB.get() ? 'file' : cmp.errorB.get() && !cmp.predB.get() ? 'error' : null) : notice.get()}
      json={activeJson.get()}
      pred={showingB.get() ? cmp.predB.get() : pred.get()}
      bind:conf={conf}
      maxDet={maxDet.get()}
      hidden={showingB.get() ? cmp.hiddenB.get() : hidden.get()}
      bind:tbody={tbody}
      tabRow={tabRow.get()}
      activeIdx={tableActiveIdx.get()}
      selectedIdx={tableSelectedIdx.get()}
      hover={showingB.get() ? cmp.hoverB.get() : hover.get()}
      picked={showingB.get() ? cmp.pickedB.get() : picked.get()}
      onHover={(b) => (showingB.peek() ? cmp.hoverB.set(b) : hover.set(b))}
      onPick={(b) => (showingB.peek() ? cmp.pickedB.set(b) : picked.set(b))}
      onrowkey={onrowkey}
    />
  );

  return (
    <>
      <input
        ref={(e) => (weightsInput = e)}
        type="file"
        accept=".folo"
        class="sr-only"
        tabindex={-1}
        aria-hidden="true"
        onchange={pickWeights}
      />

      <input
        ref={(e) => (fileInput = e)}
        type="file"
        accept={ACCEPT}
        class="sr-only"
        tabindex={-1}
        aria-hidden="true"
        onchange={(e) => {
          const f = e.currentTarget.files?.[0];
          if (f) void setFile(f);
          e.currentTarget.value = '';
        }}
      />

      <Page
        title="Playground"
        wide
        sub={source.get() && phone.current ? undefined : intro}
        actions={() => (
          <PlaygroundActions
            hasModels={ranked.get().length > 0}
            phone={phone.current}
            narrow={narrow.current}
            compare={cmp.compare.get()}
            canExport={canExport.get()}
            exportingPng={exportingPng.get()}
            onToggleCompare={() => cmp.toggleCompare()}
            onExportPng={() => void onExportPng()}
            onExportJson={onExportJson}
            onOpenImage={() => fileInput.click()}
          />
        )}
      >
        {narrow.current && ranked.get().length ? pickers(false) : null}

        {models.error && !models.data ? (
          <Panel>
            <ErrorState error={models.error} retry={() => models.refresh()} title="Could not load models" compact />
          </Panel>
        ) : models.data && models.data.length === 0 ? (
          <Panel>
            <EmptyState
              icon={Box}
              title="No models to run"
                description={
                  noDatasets.get()
                    ? 'Models come from training runs, and a run needs a dataset first: add one, then start a run.'
                    : 'Train a model first; its best weights show up here when the run finishes.'
                }
                actions={() => <FirstStepAction noDatasets={noDatasets.get()} />}
            />
          </Panel>
        ) : (
          /* The image and the 320 px rail; in one column (below 1024 px) the canvas takes the image's aspect (see .layout.fit). */
          <div class="layout" class:fit={!!source.get() && !!dims.get()}>
            <div class="main">
              <ImagePanel
                notice={notice.get()}
                bind:noticeEl={noticeEl}
                badFile={badFile.get()}
                model={model.get()}
                chooseWeights={() => {
                  weightsTarget.set('a');
                  weightsInput.click();
                }}
                browse={() => fileInput.click()}
                bind:runOn={runOn}
                error={error.get()}
                bind:retryTick={retryTick}
                oom={oom.get()}
                browserOff={browserOff.get()}
                smaller={smaller.get()}
                localBackend={localBackend}
                chooseModel={chooseModel}
                ondragover={ondragover}
                ondragleave={ondragleave}
                ondrop={ondrop}
                source={source.get()}
                bind:seg={seg}
                onSeg={onSeg}
                runOnNote={runOnNote.get()}
                sourceName={sourceName.get()}
                shortName={shortName.get()}
                dims={dims.get()}
                bind:pickerOpen={pickerOpen}
                clear={clear}
                dragging={dragging.get()}
                samples={samples.get()}
                setSource={setSource}
                src={src.get()}
                imageError={imageError.get()}
                onImageLoad={(ok) => imageError.set(!ok)}
                onImageRetry={retryImage}
                pred={pred.get()}
                cut={cut()}
                kept={kept.get()}
                truthBoxes={truthBoxes.get()}
                truthOnly={truthOnly.get()}
                classes={classes.get()}
                labels={labels.get()}
                scores={scores.get()}
                only={only.get()}
                activeIdx={activeIdx.get()}
                selectedIdx={selectedIdx.get()}
                bind:hover={hover}
                pickFromImage={pickFromImage}
                predicting={predicting.get()}
                engine={engine}
                sheetShown={sheetShown.get()}
                tablet={tablet.get()}
                o2o={effectiveO2o.get()}
                rail={!narrow.current}
                bind:conf={conf}
                bind:iou={iou}
                result={result.get()}
                rows={rows.get()}
                hidden={hidden.get()}
                maxDet={maxDet.get()}
                weightsMissing={weightsMissing.get()}
                bind:moreBtn={moreBtn}
                closeSheet={closeSheet}
                bind:moreOpen={moreOpen}
                sheetTitle={sheetTitle.get()}
                bind:sheetEl={sheetEl}
                resetButton={resetButton}
                moreBody={moreBody}
                onCapture={() => {
                  captureVideo.set(null);
                  captureOpen.set(true);
                }}
                compareFacts={
                  cmp.compare.get() && (result.get() || cmp.resultB.get())
                    ? {
                        countA: cmp.summary.get().countA,
                        countB: cmp.summary.get().countB,
                        deltaText: cmp.summary.get().countDelta !== null ? formatSignedDelta(cmp.summary.get().countDelta) : null,
                      }
                    : null
                }
                compareStage={cmp.compare.get() && !!source.get() && !!dims.get() ? compareStage : null}
              />

              {narrow.current && cmp.compare.get() ? compareSummary(false) : null}

              {narrow.current ? results(false) : null}

              {source.get() && phone.current ? (
                <PhoneDetails
                  result={showingB.get() ? cmp.resultB.get() : result.get()}
                  counts={showingB.get() ? cmp.countsB.get() : counts.get()}
                  hidden={showingB.get() ? cmp.hiddenB.get() : hidden.get()}
                  pred={showingB.get() ? cmp.predB.get() : pred.get()}
                  where={cmp.compare.get() ? whereLabel(showingB.get() ? cmp.predB.get() : pred.get()) : where.get()}
                  showAllButton={showAllButton}
                  classList={classList}
                  timingBody={timingBody}
                />
              ) : null}
            </div>

            {narrow.current ? (
              <SidePanels
                source={source.get()}
                result={showingB.get() ? cmp.resultB.get() : result.get()}
                counts={showingB.get() ? cmp.countsB.get() : counts.get()}
                pred={showingB.get() ? cmp.predB.get() : pred.get()}
                where={cmp.compare.get() ? whereLabel(showingB.get() ? cmp.predB.get() : pred.get()) : where.get()}
                bind:conf={conf}
                resetButton={resetButton}
                secondarySliders={secondarySliders}
                displayToggles={displayToggles}
                showAllButton={showAllButton}
                classList={classList}
                timingBody={timingBody}
              />
            ) : (
              <Rail
                pickers={() => pickers(true)}
                thresholds={railThresholds}
                resetButton={resetButton}
                toggles={source.get() ? displayToggles : null}
                compareSummary={cmp.compare.get() ? () => compareSummary(true) : null}
                detections={() => results(true)}
                timing={timingBody}
                where={cmp.compare.get() ? null : pred.get() ? where.get() : null}
              />
            )}
          </div>
        )}
      </Page>

      <DatasetPicker
        bind:open={pickerOpen}
        prefer={classes.get()}
        onpick={(ds, img, at) => setSource({ kind: 'dataset', ds, img, at })}
        onclose={() => seg.set(source.peek()?.kind ?? 'file')}
      />

      <MediaCaptureDialog
        bind:open={captureOpen}
        initialVideo={captureVideo.get()}
        onCapture={(blob, name) => void setFile(blob, name)}
        onClose={() => captureVideo.set(null)}
      />
    </>
  );
}

css`
  /* The image beside the 320 px rail, the same with or without an image. One screen: the layout is the window less
     the top bar, the page's padding (24 above, 48 below) and its header with the gap under it (76), so neither the
     image nor the rail scrolls the page; the canvas is that less its 40 px header and the panel's edges. */
  .layout {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 320px;
    gap: var(--space-4);
    align-items: start;
    --fit-h: max(420px, calc(100dvh - var(--size-topbar) - 148px));
    --stage-h: calc(var(--fit-h) - 42px);
  }
  .main {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: var(--space-4);
    min-width: 0;
  }

  @media (max-width: 1023px) {
    .layout {
      grid-template-columns: minmax(0, 1fr);
      --stage-h: max(360px, calc(100dvh - 220px));
    }
    /* One column: interleave the panels so the controls sit right under the image. */
    .main {
      display: contents;
    }
  }
  @media (min-width: 768px) and (max-width: 1023px) {
    /* With an image: the canvas at most half the viewport (a square photo letterboxed on bg.inset), Confidence and
       Overlap in one strip right under it, the rest in the sheet; the Thresholds panel steps out and Classes sits
       beside Timing. */
    .layout.fit {
      --stage-h: 50dvh;
    }
  }
  @media (max-width: 767px) {
    /* The open sheet covers the lower part of the screen: leave room to scroll the detections above it. */
    .layout:has(:global(.sheet)) {
      padding-bottom: 45dvh;
    }
  }
`;
