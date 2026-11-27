// Playground (`#/playground`): a model's detections on an image, run on the server or in this browser through
// folo-wasm, with the thresholds, the detections and their timing. The parts are in ./playground/.
import Box from '$icons/box';
import RotateCcw from '$icons/rotate-ccw';
import { computed, css, effect, MediaQuery, onCleanup, signal, untrack, type Signal } from 'fw';
import type { Prediction } from '$lib/api/types';
import { media } from '$lib/design/tokens';
import { filterDetections } from '$lib/detect/nms';
import { BrowserEngine, type Backend } from '$lib/inference/browser';
import { isOom } from '$lib/oom';
import { router } from '$lib/router';
import Page from '$lib/shell/Page';
import { setCrumbs } from '$lib/shell/crumbs';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Panel from '$lib/ui/Panel';

import BatchFilmstrip from './playground/BatchFilmstrip';
import CompareStage from './playground/CompareStage';
import CompareSummaryPanel from './playground/CompareSummary';
import DatasetPicker from './playground/DatasetPicker';
import InferenceOptions from './playground/InferenceOptions';
import LiveMediaCanvas from './playground/LiveMediaCanvas';
import MediaBar from './playground/MediaBar';
import MediaInputs from './playground/MediaInputs';
import PlaygroundActions from './playground/PlaygroundActions';
import PlaygroundPickers from './playground/PlaygroundPickers';
import PlaygroundTiming from './playground/PlaygroundTiming';
import { formatCompareTimingMeta, formatSignedDelta } from './playground/compare';
import { holdNoticeFocus } from './playground/noticeFocus';
import ImagePanel from './playground/ImagePanel';
import ResultsPanel from './playground/ResultsPanel';
import ClassList from './playground/ClassList';
import PerClassThresholds from './playground/PerClassThresholds';
import PhoneDetails from './playground/PhoneDetails';
import SidePanels from './playground/SidePanels';
import Rail from './playground/Rail';
import DisplayToggles from './playground/DisplayToggles';
import { MoreBody, RailThresholds, SecondarySliders } from './playground/Thresholds';
import { hasPerClassConf, mapPerClassConfToClasses, normalizePerClassConf, type PerClassConfMap } from './playground/perClassConf';
import { DEFAULTS, type RunOn } from './playground/predict';
import { keepSession } from './playground/keepSession';
import { sampleImages } from './playground/samples';
import { playgroundPath } from './playground/session';
import { createCompareController } from './playground/useCompare';
import { useBatchSession } from './playground/useBatchSession';
import { usePlaygroundExport } from './playground/useExport';
import { useInferenceOptions } from './playground/useInferenceOptions';
import { useLocalWeights } from './playground/useLocalWeights';
import { useMediaSession } from './playground/useMediaSession';
import { useModelCatalog } from './playground/useModelCatalog';
import { usePlaygroundSelection } from './playground/useSelection';
import { usePlaygroundSheet } from './playground/useSheetState';
import { useSinglePredict } from './playground/useSinglePredict';
import { usePlaygroundSource } from './playground/useSource';
import { useTracking, type TrackingController } from './playground/useTracking';
import { useWasmAvailability } from './playground/useWasmProbe';
import {
  areDefaultThresholds,
  classCounts,
  detectionIndex,
  predictionJson,
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
  const pickerOpen = signal(false);
  let fileInput!: HTMLInputElement;
  let videoInput!: HTMLInputElement;

  const pred = signal<Prediction | null>(null);
  const predSourceKey = signal<string | null>(null);
  const runOn = signal<RunOn>('server');
  const engine = new BrowserEngine();
  onCleanup(() => engine.dispose());
  const mediaActive = signal(false);
  const { webgpu, browserOff, trackerBrowserOff, runOnNote, serverPredictOptions } = useWasmAvailability(runOn, () => mediaActive.get());
  let getCompareRunOn: () => RunOn | null = () => null;
  let getComparePredB: () => Prediction | null = () => null;
  const inf = useInferenceOptions({
    runOn,
    compareRunOn: () => getCompareRunOn(),
    activePredictions: () => [pred.get(), getComparePredB()],
    engine,
    browserOff: () => browserOff.get(),
    serverPredictOptions: () => serverPredictOptions.get(),
  });

  let weightsInput!: HTMLInputElement;
  const where = computed(() => whereLabel(pred.get()));

  const predicting = signal(false);
  const error = signal<unknown>(null);
  const retryTick = signal(0);

  const conf = signal(DEFAULTS.conf);
  const iou = signal(DEFAULTS.iou);
  const maxDet = signal(DEFAULTS.maxDet);
  const perClassConf = signal<PerClassConfMap>({});
  const labels = signal(true);
  const scores = signal(true);
  const truth = signal(false);
  const hidden = signal<Set<number>>(new Set());
  const hover = signal<Det | null>(null);
  const picked = signal<Det | null>(null);
  const tbody = signal<HTMLTableSectionElement>();
  const tab = signal<'detections' | 'classes' | 'json'>('detections');

  let syncModelUrl: ((id: string) => void) | null = null;
  let clearCompare: () => void = () => {};
  let stopMedia: () => void = () => {};
  let clearBatch: () => void = () => {};
  let loadBatchFiles: (files: File[]) => void = () => {};
  let loadVideoFile: (f: File) => void = () => {};
  let startWebcamFn: () => void = () => {};
  let isLiveStreamFn: () => boolean = () => false;
  let isUnsyncedVideoFn: () => boolean = () => false;

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
    setBatchFrame,
    prepareMediaSource,
    setLiveFrame,
    setFiles,
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
      predSourceKey.set(null);
      error.set(null);
      weightsMissing.set(false);
      hover.set(null);
      picked.set(null);
      hidden.set(new Set());
      clearCompare();
    },
    onResetForBatchFrame: (cached, frameKey) => {
      pred.set(cached);
      predSourceKey.set(frameKey);
      error.set(null);
      weightsMissing.set(false);
      hover.set(null);
      picked.set(null);
      clearCompare();
    },
    onClearExtra: () => moreOpen.set(false),
    onDropVideo: (videoFile) => loadVideoFile(videoFile),
    onOpenVideo: () => videoInput.click(),
    onStartWebcam: () => startWebcamFn(),
    onSwitchAwayFromMedia: () => stopMedia(),
    onClearBatch: () => clearBatch(),
    onLoadBatch: (files) => loadBatchFiles(files),
    openFilePicker: () => fileInput.click(),
    pickerOpen,
  });

  const { tablet, moreOpen, sheetEl, moreBtn, sheetShown, sheetTitle, closeSheet } = usePlaygroundSheet(narrow, phone, () => !!source.get());

  let getModelBId: () => string | null = () => null;
  const { localWeights, weightsMissing, weightsTarget, badFile, badFileB, pickWeights, dropLocal } = useLocalWeights({
    modelAId: () => modelId.peek(),
    modelBId: () => getModelBId(),
    isDestroyed,
  });

  const samples = sampleImages(model);

  let trackingRef: TrackingController | null = null;
  let cancelSinglePredict: () => void = () => {};
  const result = computed(() => {
    const p = pred.get();
    if (!p) return null;
    const t0 = performance.now();
    const tracked = trackingRef?.boxesFor(p) ?? null;
    const f = { conf: conf.get(), iou: o2o.get() ? 1 : iou.get(), maxDet: maxDet.get(), perClassConf: perClassConf.get() };
    // ByteTrack output already enforces global + per-class thresholds while keeping stage-2 low-confidence recovery.
    const boxes = tracked ? filterDetections(tracked, { conf: 0, iou: f.iou, maxDet: f.maxDet }) : filterDetections(p.boxes, f);
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
    ranked: () => ranked.get(), choices: () => choices.get(), modelA: () => model.get(), modelAId: () => modelId.get(),
    chooseModelA: chooseModel, runOnA: runOn, predA: pred, predSourceKeyA: predSourceKey, hoverA: hover, pickedA: picked,
    resultA: () => result.get(), rowsA: () => rows.get(), o2oA: () => o2o.get(), source, conf, iou, maxDet, perClassConf,
    hidden, classesA: () => classes.get(), browserOff: () => browserOff.get(), localWeights,
    dropLocalB: (id, f, e) => dropLocal(id, f, e, badFileB), badFileB, engine,
    inferenceFor: inf.effectiveFor, precisionFor: inf.precisionFor,
    isLiveStream: () => isLiveStreamFn(), isUnsyncedVideoPreview: () => isUnsyncedVideoFn(),
  });
  clearCompare = cmp.clearResults;
  getModelBId = () => cmp.modelBId.peek();
  getCompareRunOn = () => (cmp.compare.get() ? cmp.runOnB.get() : null);
  getComparePredB = () => (cmp.compare.get() ? cmp.predB.get() : null);

  const mediaSession = useMediaSession({
    onFrame: (f) => setLiveFrame(f.blob, f.name, f.width, f.height, mediaSession.mode.peek() === 'webcam' ? 'webcam' : 'video'),
    isPredicting: () => predicting.get() || (cmp.compare.get() && cmp.predictingB.get()),
    isInferenceBlocked: () => !modelId.get() || !!error.get() || weightsMissing.get() || (cmp.compare.get() && (!cmp.modelBId.get() || !!cmp.errorB.get() || cmp.weightsMissingB.get())),
    hasValidResult: () => !!pred.peek() && (!cmp.compare.peek() || !!cmp.predB.peek()),
    lastLatencyMs: () => pred.peek()?.timing.total_ms ?? null,
    onStopped: () => seg.set(source.peek()?.kind ?? 'file'),
    onTrackReset: () => { cancelSinglePredict(); trackingRef?.reset(); },
  });
  const tracking = (trackingRef = useTracking({
    engine, browserOff: () => trackerBrowserOff.get(), mediaMode: () => mediaSession.mode.get(),
    mediaKey: () => `${mediaSession.mode.get()}:${mediaSession.sessionSeq.get()}`,
    frameKey: () => (mediaSession.mode.get() === 'none' ? null : `${mediaSession.mode.get()}:${mediaSession.sessionSeq.get()}:${mediaSession.frameSeq.get()}`),
    frameMeta: () => mediaSession.lastFrameMeta.get(),
    sourceName: () => (mediaSession.mode.get() === 'video' ? mediaSession.videoName.get() || 'video' : 'webcam'),
    modelId: () => modelId.get(), modelName: () => model.get()?.name ?? null, classes: () => classes.get(), pred: () => pred.get(),
    conf: () => conf.get(), perClassConf: () => perClassConf.get(), iou: () => iou.get(), maxDet: () => maxDet.get(), o2o: () => o2o.get(),
  }));
  const visibleTrails = computed(() => {
    if (!tracking.enabled.get()) return undefined;
    const list = tracking.trails.get().filter((t) => !hidden.get().has(t.class));
    return list.length ? list : undefined;
  });
  stopMedia = () => mediaSession.stopAll(false);
  loadVideoFile = (f) => {
    prepareMediaSource('video');
    mediaSession.loadVideo(f);
  };
  startWebcamFn = () => {
    prepareMediaSource('webcam');
    void mediaSession.startWebcam();
  };
  isLiveStreamFn = () => mediaSession.liveStreaming.get();
  isUnsyncedVideoFn = () => mediaSession.mode.get() === 'video' && !mediaSession.frameSynced.get();
  effect(() => {
    mediaActive.set(mediaSession.mode.get() !== 'none');
  });

  let predMatchesSourceFn: () => boolean = () => true;

  // The image, its result and the thresholds outlive the page for the tab; a dataset image and the thresholds are in
  // the URL (./playground/session).
  let lastModelId: string | null = null;
  let lastModelClasses: string[] = [];
  const restored = keepSession({
    source, pred, predMatchesSource: () => predMatchesSourceFn(), runOn: runOn as Signal<string>,
    inference: { config: () => inf.config.get(), restore: inf.restore },
    conf, iou, maxDet, perClassConf, classes: () => classes.get(),
    onRestorePerClassOrigin: (origin) => {
      if (origin.modelId) {
        lastModelId = origin.modelId;
        lastModelClasses = origin.classes;
      }
    },
    labels, scores, truth, hidden,
    compare: cmp.compare, modelB: () => cmp.modelBId.get() ?? cmp.pickedModelB.get(),
    pickModelB: (id) => cmp.pickedModelB.set(id), compareView: cmp.compareView,
    setSource, choose, model: () => modelId.get() ?? queryModel.get() ?? null,
    pickModel: (id) => pickedModel.set(id), onModelUrl: (write) => (syncModelUrl = write),
  });

  effect(() => {
    const id = modelId.get();
    const curClasses = classes.get();
    if (id !== null && lastModelId !== null && id !== lastModelId) {
      untrack(() => {
        hidden.set(new Set());
        const curPcc = perClassConf.peek();
        if (Object.keys(curPcc).length) perClassConf.set(mapPerClassConfToClasses(curPcc, lastModelClasses, curClasses));
        const curB = cmp.hiddenBNames.peek();
        if (curB.size) {
          const nextA = new Set(curClasses);
          const keptB = new Set([...curB].filter((name) => !nextA.has(name)));
          if (keptB.size !== curB.size) cmp.hiddenBNames.set(keptB);
        }
      });
    }
    if (id !== null) {
      lastModelId = id;
      lastModelClasses = curClasses;
    }
  });

  const batch = useBatchSession({
    engine, model: () => model.peek(), modelId: () => modelId.get(), runOn,
    inference: () => inf.effective.get(), precision: () => inf.precision.get(),
    localWeights, compare: cmp.compare, o2o: () => o2o.get(), conf, iou, maxDet, perClassConf,
    hidden, classes: () => classes.get(), source, pred, predSourceKey, predicting, error, weightsMissing,
    hover, picked, choose, cancelSinglePredict: () => cancelSinglePredict(),
    onSingleFile: (s) => setSource(s), onActivateFrame: (s, cached, bump) => setBatchFrame(s, cached, bump),
    onEnterBatch: () => { stopMedia(); hidden.set(new Set()); },
    dropLocal: (id, f, e) => dropLocal(id, f, e, badFile),
  });
  clearBatch = batch.clear;
  loadBatchFiles = (files) => void batch.loadFiles(files);

  const cut = useSinglePredict({
    modelId: () => modelId.get(), source, runOn, localWeights, retryTick, restored,
    consumeReuse: cmp.consumeReuseA, pred, predSourceKey, predicting, error, weightsMissing,
    hidden, hover, picked, conf, perClassConf, engine,
    inference: () => inf.effective.get(), precision: () => inf.precision.get(),
    dropLocal: (id, f, e) => dropLocal(id, f, e, badFile),
    isLiveStream: () => mediaSession.liveStreaming.get(),
    isBatchRunningAll: () => batch.runningAll.get(), getCachedBatchPred: batch.getCachedPred,
    wasBatchStopped: batch.wasStoppedFor, onBatchResult: batch.recordSingleResult,
    onBatchBusy: batch.recordSingleBusy, onRegisterAbort: (fn) => (cancelSinglePredict = fn),
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
  const { showingB, tableResult, tableRows, tableSelectedIdx, tableActiveIdx, tabRow, pickFromImage, pickFromImageB, onrowkey } = usePlaygroundSelection({
    compare: cmp.compare,
    inspectSide: cmp.inspectSide,
    tab,
    tbody,
    keptA: () => kept.peek(),
    keptB: () => cmp.keptB.peek(),
    pickedA: picked,
    pickedB: cmp.pickedB,
    rowsA: () => rows.get(),
    rowsB: () => cmp.rowsB.get(),
    resultA: () => result.get(),
    resultB: () => cmp.resultB.get(),
    selectedIdxA: () => selectedIdx.get(),
    selectedIdxB: () => cmp.selectedIdxB.get(),
    activeIdxA: () => activeIdx.get(),
    activeIdxB: () => cmp.activeIdxB.get(),
    isNarrow: () => narrow.peek(),
    pickerOpen,
    sheetShown: () => sheetShown.peek(),
    closeSheet,
    onpaste,
  });

  const dims = computed(() => sourceDims(pred.get(), source.get()) ?? mediaSession.camDims.get() ?? mediaSession.videoDims.get());
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
    if (isUnsyncedVideoFn()) return null;
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
        perClassConf: perClassConf.get(),
      },
    );
  });
  const activeJson = computed(() => (cmp.compare.get() ? cmp.compareJson.get() : singleJson.get()));
  const { exportingPng, predAMatches, predBMatches, canExport, onExportPng, onExportCompareSideBySidePng, onExportJson } = usePlaygroundExport({
    source,
    src,
    dims: () => dims.get(),
    shortName: () => shortName.peek(),
    modelA: () => model.peek(),
    modelB: () => cmp.modelB.peek(),
    predA: pred,
    predB: cmp.predB,
    predSourceKeyA: predSourceKey,
    predSourceKeyB: cmp.predBSourceKey,
    resultA: () => result.get(),
    resultB: () => cmp.resultB.get(),
    rowsA: () => rows.peek(),
    rowsB: () => cmp.rowsB.peek(),
    classesA: () => classes.peek(),
    classesB: () => cmp.classesB.peek(),
    truthOnly: () => truthOnly.peek(),
    truthBoxes: () => truthBoxes.peek(),
    labels,
    scores,
    predictingA: predicting,
    predictingB: cmp.predictingB,
    errorA: error,
    errorB: cmp.errorB,
    weightsMissingA: weightsMissing,
    weightsMissingB: cmp.weightsMissingB,
    compare: cmp.compare,
    compareView: cmp.compareView,
    splitPct: cmp.splitPct,
    zoom: cmp.zoom,
    activeJson: () => activeJson.peek(),
    trailsA: () => (isUnsyncedVideoFn() ? undefined : visibleTrails.peek()),
    isUnsyncedVideoPreview: () => mediaSession.mode.get() === 'video' && !mediaSession.frameSynced.get(),
    isLiveStreaming: () => mediaSession.liveStreaming.get(),
  });
  predMatchesSourceFn = () => predAMatches.get();
  const effectiveO2o = computed(() => (cmp.compare.get() ? o2o.get() && cmp.o2oB.get() : o2o.get()));

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
    perClassConf.set({});
  }
  const isDefault = computed(() => areDefaultThresholds(conf.get(), iou.get(), maxDet.get(), effectiveO2o.get(), perClassConf.get()));

  function setClassConf(classId: number, value: number) {
    perClassConf.set((cur) => normalizePerClassConf({ ...cur, [classId]: value }));
  }
  function clearClassConf(classId: number) {
    perClassConf.set((cur) => {
      if (!Object.prototype.hasOwnProperty.call(cur, classId)) return cur;
      const next = { ...cur };
      delete next[classId];
      return next;
    });
  }

  // ---------- snippets ----------
  const displayToggles = () => <DisplayToggles bind:labels={labels} bind:scores={scores} bind:truth={truth} dataset={source.get()?.kind === 'dataset'} />;
  const inferenceOpts = (inline: boolean) => <InferenceOptions ctrl={inf} inline={inline} />;
  const perClassThresholds = (inline: boolean) => (
    <PerClassThresholds
      classes={classes.get()}
      globalConf={conf.get()}
      perClassConf={perClassConf.get()}
      modelId={modelId.get()}
      modelName={model.get()?.name ?? null}
      inline={inline}
      onSetClassConf={setClassConf}
      onClearClassConf={clearClassConf}
      onResetAll={() => perClassConf.set({})}
      onImport={({ globalConf, perClassConf: next }) => {
        if (globalConf !== null) conf.set(globalConf);
        perClassConf.set(normalizePerClassConf(next));
      }}
    />
  );
  const secondarySliders = (inline: boolean) => <SecondarySliders inline={inline} o2o={effectiveO2o.get()} bind:iou={iou} bind:maxDet={maxDet} />;
  const railThresholds = () => <RailThresholds bind:conf={conf} bind:iou={iou} bind:maxDet={maxDet} o2o={effectiveO2o.get()} perClassThresholds={() => perClassThresholds(false)} />;
  const moreBody = () => <MoreBody tablet={tablet.get()} o2o={effectiveO2o.get()} bind:conf={conf} bind:iou={iou} bind:maxDet={maxDet} toggles={displayToggles} perClassThresholds={() => perClassThresholds(true)} inferenceOptions={() => inferenceOpts(true)} />;

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
    if (cmp.hiddenBNames.peek().has(nameB)) {
      cmp.toggleHiddenBName(nameB);
      return;
    }
    const idxA = classes.peek().indexOf(nameB);
    if (idxA >= 0) toggleClass(idxA);
    else cmp.toggleHiddenBName(nameB);
  }

  const unsyncedPreviewState = (): 'ready' | 'loading' | 'error' | null =>
    !isUnsyncedVideoFn() ? null : mediaSession.videoErr.get() ? 'error' : mediaSession.videoReady.get() && !mediaSession.seeking.get() ? 'ready' : 'loading';
  const classList = () => {
    const u = isUnsyncedVideoFn();
    return (
      <ClassList
        result={u ? null : showingB.get() ? cmp.resultB.get() : result.get()}
        counts={u ? [] : showingB.get() ? cmp.countsB.get() : counts.get()}
        pred={u ? null : showingB.get() ? cmp.predB.get() : pred.get()}
        hidden={showingB.get() ? cmp.hiddenB.get() : hidden.get()}
        classes={showingB.get() ? cmp.classesB.get() : classes.get()}
        maxCount={showingB.get() ? cmp.maxCountB.get() : maxCount.get()}
        perClassConf={showingB.get() ? cmp.perClassConfB.get() : perClassConf.get()}
        toggleClass={showingB.get() ? toggleClassB : toggleClass}
      />
    );
  };

  const zoomResetKey = computed(() => {
    const m = mediaSession.mode.get();
    if (m === 'video') return `video:${mediaSession.videoUrl.get() ?? ''}`;
    if (m === 'webcam') return 'webcam';
    return undefined;
  });

  // Stable snippet reference so `<CompareStage>` never remounts when Model A's `pred` or `dims` object identity changes.
  const compareStage = () => (
    <CompareStage
      source={source.get()!} src={src.get()} dims={dims.get()!} sourceName={sourceName.get()}
      onImageLoad={(ok) => imageError.set(!ok)} view={cmp.compareView.get()} bind:splitPct={cmp.splitPct}
      zoom={cmp.zoom} zoomResetKey={zoomResetKey.get()} keepPrevious={mediaSession.mode.get() !== 'none'}
      labels={labels.get()} scores={scores.get()} truthBoxes={truthBoxes.get()} truthOnly={truthOnly.get()}
      modelA={model.get()} predA={predAMatches.get() ? pred.get() : null} keptA={predAMatches.get() ? kept.get() : []}
      visibleCountA={predAMatches.get() ? rows.get().length : 0} classesA={classes.get()} onlyA={only.get()}
      activeIdxA={activeIdx.get()} selectedIdxA={selectedIdx.get()} predictingA={predicting.get()} errorA={error.get()}
      onHoverA={(b) => hover.set(b)} onSelectA={pickFromImage}
      modelB={cmp.modelB.get()} predB={predBMatches.get() ? cmp.predB.get() : null} keptB={predBMatches.get() ? cmp.keptB.get() : []}
      visibleCountB={predBMatches.get() ? cmp.rowsB.get().length : 0} classesB={cmp.classesB.get()} onlyB={cmp.onlyB.get()}
      activeIdxB={cmp.activeIdxB.get()} selectedIdxB={cmp.selectedIdxB.get()} predictingB={cmp.predictingB.get()}
      errorB={cmp.errorB.get()} weightsMissingB={cmp.weightsMissingB.get()} badFileB={badFileB.get()} runOnB={cmp.runOnB.get()}
      onHoverB={(b) => cmp.hoverB.set(b)} onSelectB={pickFromImageB} onRetryB={() => cmp.retryTickB.set((t) => t + 1)}
      onChooseWeightsB={() => { weightsTarget.set('b'); weightsInput.click(); }}
      onServerB={() => cmp.runOnB.set('server')} engine={engine}
    />
  );

  const timingBody = () => (
    <PlaygroundTiming
      compare={cmp.compare.get()} narrow={narrow.current} modelA={model.get()} modelB={cmp.modelB.get()}
      predA={isUnsyncedVideoFn() ? null : pred.get()} predB={isUnsyncedVideoFn() ? null : cmp.predB.get()}
      predictingA={predicting.get()} predictingB={cmp.predictingB.get()}
      resultMsA={isUnsyncedVideoFn() ? null : (result.get()?.ms ?? null)}
      adapter={engine.info?.adapter} engine={engine} engineInfo={engine.info}
    />
  );

  const useLiveCanvas = computed(() => {
    const m = mediaSession.mode.get();
    if (m === 'webcam') return !cmp.compare.get() || !!mediaSession.camErr.get() || mediaSession.camStarting.get() || !source.get();
    if (m === 'video') {
      if (mediaSession.videoErr.get() || !source.get()) return true;
      if (!mediaSession.frameSynced.get()) return true;
      return !cmp.compare.get() && mediaSession.playing.get();
    }
    return false;
  });
  const liveStage = () => {
    const isCam = mediaSession.mode.get() === 'webcam';
    const isPreview = !isCam && !mediaSession.frameSynced.get() && !mediaSession.playing.get();
    const showBoxes = isCam ? predAMatches.get() || mediaSession.camActive.get() : mediaSession.frameSynced.get();
    return (
      <LiveMediaCanvas
        video={mediaSession.videoEl.get()} mode={isCam ? 'webcam' : 'video'}
        active={isCam ? mediaSession.camActive.get() : mediaSession.playing.get()}
        starting={isCam && mediaSession.camStarting.get()} paused={isCam && mediaSession.camPaused.get()}
        preview={isPreview} renderTick={mediaSession.previewTick.get()}
        error={isCam ? mediaSession.camErr.get() : mediaSession.videoErr.get()}
        boxes={showBoxes ? rows.get().map((r) => r.b) : []} trails={showBoxes ? visibleTrails.get() : undefined}
        classes={classes.get()} labels={labels.get()} scores={scores.get()}
        fps={mediaSession.fps.get()} latencyMs={mediaSession.latencyMs.get()}
        onRetry={() => void mediaSession.startWebcam()} onDismiss={() => mediaSession.stopWebcam()}
      />
    );
  };
  const mediaBar = () => <MediaBar media={mediaSession} predicting={predicting.get() || (cmp.compare.get() && cmp.predictingB.get())} tracking={tracking} />;
  const batchBar = () => <BatchFilmstrip batch={batch} hasModel={!!modelId.get()} compare={cmp.compare.get()} />;

  const intro = () => <span>Run a model on any image, video, or live camera. Thresholds re-filter instantly in the browser.</span>;

  // The model and where it runs: the rail's first section from 1024 px up, a toolbar row under the title below that.
  const pickers = (inRail: boolean) => (
    <PlaygroundPickers
      compare={cmp.compare.get()}
      inRail={inRail}
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
      runOnA={runOn}
      runOnB={cmp.runOnB}
      compareView={cmp.compareView}
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
      hasAnyResult={!isUnsyncedVideoFn() && (predAMatches.get() || predBMatches.get())}
      predictingA={predicting.get()}
      predictingB={cmp.predictingB.get()}
      inspectSide={cmp.inspectSide.get()}
      onInspectSide={(side) => cmp.inspectSide.set(side)}
      canExport={canExport.get()}
      exportingPng={exportingPng.get()}
      onExportSideBySidePng={() => void onExportCompareSideBySidePng()}
      onExportCompareJson={onExportJson}
    />
  );

  // The detections: a panel under the image below 1024 px, the rail's Detections section (with a Classes tab) above.
  const results = (rail: boolean) => {
    const unsynced = isUnsyncedVideoFn();
    return (
      <ResultsPanel
        rail={rail}
        unsyncedPreview={unsyncedPreviewState()}
        classCount={unsynced ? 0 : showingB.get() ? cmp.countsB.get().length : counts.get().length}
        classList={classList}
        showAllButton={showAllButton}
        onShowAll={clearAllHidden}
        result={unsynced ? null : tableResult.get()}
        bind:tab={tab}
        rows={unsynced ? [] : tableRows.get()}
        predicting={showingB.get() ? cmp.predictingB.get() : predicting.get()}
        notice={showingB.get() ? (cmp.weightsMissingB.get() ? 'file' : cmp.errorB.get() && !cmp.predB.get() ? 'error' : null) : notice.get()}
        json={activeJson.get()}
        pred={unsynced ? null : showingB.get() ? cmp.predB.get() : pred.get()}
        bind:conf={conf}
        maxDet={maxDet.get()}
        hasPerClassOverrides={hasPerClassConf(showingB.get() ? cmp.perClassConfB.get() : perClassConf.get())}
        onResetPerClass={() => perClassConf.set({})}
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
  };

  return (
    <>
      <MediaInputs
        media={mediaSession}
        onWeightsRef={(e) => (weightsInput = e)}
        onFileRef={(e) => (fileInput = e)}
        onVideoRef={(e) => (videoInput = e)}
        onPickWeights={pickWeights}
        onPickFiles={setFiles}
        onPickVideo={loadVideoFile}
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
                onSeg={(v) => onSeg(v, mediaSession.mode.peek())}
                phone={phone.current}
                mediaActive={mediaSession.mode.get() !== 'none'}
                mediaLabel={mediaSession.mode.get() === 'webcam' ? 'Live camera' : mediaSession.videoName.get()}
                liveStage={useLiveCanvas.get() ? liveStage : null}
                mediaBar={mediaSession.mode.get() !== 'none' ? mediaBar : null}
                batchBar={batch.active.get() ? batchBar : null}
                zoomResetKey={zoomResetKey.get()}
                keepPrevious={mediaSession.mode.get() !== 'none'}
                onOpenVideo={() => videoInput.click()}
                onStartWebcam={startWebcamFn}
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
                pred={isUnsyncedVideoFn() ? null : pred.get()}
                cut={cut()}
                kept={isUnsyncedVideoFn() ? [] : kept.get()}
                trails={isUnsyncedVideoFn() ? undefined : visibleTrails.get()}
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
                unsyncedPreview={unsyncedPreviewState()}
                engine={engine}
                sheetShown={sheetShown.get()}
                tablet={tablet.get()}
                o2o={effectiveO2o.get()}
                rail={!narrow.current}
                bind:conf={conf}
                bind:iou={iou}
                result={isUnsyncedVideoFn() ? null : result.get()}
                rows={isUnsyncedVideoFn() ? [] : rows.get()}
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
                compareFacts={
                  cmp.compare.get() && !isUnsyncedVideoFn() && (predAMatches.get() || predBMatches.get())
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
                  result={isUnsyncedVideoFn() ? null : showingB.get() ? cmp.resultB.get() : result.get()}
                  counts={isUnsyncedVideoFn() ? [] : showingB.get() ? cmp.countsB.get() : counts.get()}
                  hidden={showingB.get() ? cmp.hiddenB.get() : hidden.get()}
                  pred={isUnsyncedVideoFn() ? null : showingB.get() ? cmp.predB.get() : pred.get()}
                  where={cmp.compare.get() ? '' : where.get()}
                  timingMeta={isUnsyncedVideoFn() ? undefined : cmp.compare.get() ? formatCompareTimingMeta(pred.get(), cmp.predB.get()) : undefined}
                  showAllButton={showAllButton}
                  classList={classList}
                  timingBody={timingBody}
                />
              ) : null}
            </div>

            {narrow.current ? (
              <SidePanels
                source={source.get()}
                result={isUnsyncedVideoFn() ? null : showingB.get() ? cmp.resultB.get() : result.get()}
                counts={isUnsyncedVideoFn() ? [] : showingB.get() ? cmp.countsB.get() : counts.get()}
                pred={isUnsyncedVideoFn() ? null : showingB.get() ? cmp.predB.get() : pred.get()}
                where={cmp.compare.get() ? '' : where.get()}
                bind:conf={conf}
                resetButton={resetButton}
                secondarySliders={secondarySliders}
                perClassThresholds={() => perClassThresholds(true)}
                inferenceOptions={() => inferenceOpts(true)}
                displayToggles={displayToggles}
                showAllButton={showAllButton}
                classList={classList}
                timingBody={timingBody}
              />
            ) : (
              <Rail
                pickers={() => pickers(true)}
                inferenceOptions={() => inferenceOpts(false)}
                thresholds={railThresholds}
                resetButton={resetButton}
                toggles={source.get() ? displayToggles : null}
                compareSummary={cmp.compare.get() ? () => compareSummary(true) : null}
                detections={() => results(true)}
                timing={timingBody}
                where={cmp.compare.get() || isUnsyncedVideoFn() ? null : pred.get() ? where.get() : null}
              />
            )}
          </div>
        )}
      </Page>

      <DatasetPicker
        bind:open={pickerOpen}
        prefer={classes.get()}
        onpick={(ds, img, at) => setSource({ kind: 'dataset', ds, img, at })}
        onclose={() => {
          const m = mediaSession.mode.peek();
          seg.set(m !== 'none' ? m : (source.peek()?.kind ?? 'file'));
        }}
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
