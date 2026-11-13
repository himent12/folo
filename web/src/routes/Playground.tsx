import Box from '$icons/box';
import Check from '$icons/check';
import Plus from '$icons/plus';
import RotateCcw from '$icons/rotate-ccw';
import { computed, css, effect, MediaQuery, onCleanup, onWindow, signal, untrack, type Signal } from 'fw';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { modelsResource, refreshOnModelsChange } from '$lib/api/models';
import { Resource } from '$lib/api/resource';
import type { Dataset, Model, Prediction, Run } from '$lib/api/types';
import { betterModel, modelChoices, scoreContext } from '$lib/scoreSummary';
import { media } from '$lib/design/tokens';
import { filterDetections } from '$lib/detect/nms';
import { BrowserEngine, hasWebGpu, NO_WASM, probeWasm, servedBy, wasmFromSystem, type Backend } from '$lib/inference/browser';
import { sentence } from '$lib/format';
import { isOom } from '$lib/oom';
import { href, router } from '$lib/router';
import Page from '$lib/shell/Page';
import { setCrumbs } from '$lib/shell/crumbs';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Panel from '$lib/ui/Panel';
import Skeleton from '$lib/ui/Skeleton';
import Slider from '$lib/ui/Slider';
import { toast } from '$lib/ui/toast';
import DatasetPicker from './playground/DatasetPicker';
import { looksLikeFolo } from './playground/folo';
import { holdNoticeFocus } from './playground/noticeFocus';
import TimingBar from './playground/TimingBar';
import ModelPickers from './playground/ModelPickers';
import ImagePanel from './playground/ImagePanel';
import ResultsPanel from './playground/ResultsPanel';
import ClassList from './playground/ClassList';
import PhoneDetails from './playground/PhoneDetails';
import SidePanels from './playground/SidePanels';
import { ask, BACKEND_NAME, DEFAULTS, FETCH, noServerWeights, type RunOn } from './playground/predict';
import { askAllWhenCut } from './playground/allCandidates';
import { ACCEPT, droppedImages, takeImage } from './playground/intake';
import { keepSession } from './playground/keepSession';
import { sampleImages } from './playground/samples';
import { playgroundPath } from './playground/session';
import type { Source } from './playground/source';
import { followVanished } from './playground/vanished';

type Seg = 'file' | 'dataset' | 'webcam';
type Det = Prediction['boxes'][number];

/** Model sizes, smallest first: a smaller model of the same classes may fit where this one does not. */
const ARCH_ORDER = ['folo-n', 'folo-s', 'folo-m', 'folo-l', 'folo-x'];
const NMS_TIP = 'Non-max suppression: a lower-scoring box of the same class is dropped when it overlaps a kept one by more than this.';
const round = (v: number) => String(Math.round(v));

export default function Playground() {
  setCrumbs({ label: 'Playground' });

  // Phones with an image loaded: no intro line, the image full-bleed with Confidence under it. IoU, max detections
  // and the display toggles open in a non-modal bottom sheet, so the image stays in view (and live) while tuning;
  // classes and timing fold into two disclosures under the detections. Bottom sheets are for phones only
  // (DESIGN_DIRECTION 9): a tablet's "More settings" is a popover anchored under its button.
  const phone = new MediaQuery(media.phone, false);
  // Below 1024 px the model and Run on leave the title row for a toolbar row of their own under it, full width.
  const narrow = new MediaQuery(media.upToLaptop, false);
  // Tablets with an image: the canvas capped at half the viewport, Confidence and Overlap in a strip right under it,
  // Max detections and the display toggles in the same sheet, so the image and the controls that redraw it share
  // the first screen.
  const tablet = computed(() => narrow.current && !phone.current);
  const moreOpen = signal(false);
  const sheetEl = signal<HTMLElement>();
  const moreBtn = signal<HTMLElement>();

  // ---------- image source ----------
  const source = signal<Source | null>(null);
  const sheetShown = computed(() => moreOpen.get() && narrow.current && !!source.get());
  // Phones' sheet holds every threshold; a tablet's, what its strip leaves out.
  const sheetTitle = computed(() => (tablet.get() ? 'More settings' : 'Thresholds'));
  const src = signal<string | null>(null);
  const seg = signal<Seg>('file');
  const pickerOpen = signal(false);
  let fileInput!: HTMLInputElement;
  const dragging = signal(false);

  // Focus moves into the sheet (or the tablet's popover) as it opens and back to its button as it closes.
  effect(() => {
    if (sheetShown.get()) sheetEl.get()?.focus();
  });
  // The tablet's popover is transient: a press outside it (and outside its button) closes it, focus staying put.
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
    moreBtn.peek()?.querySelector('button')?.focus();
  }
  // Dragging the sheet's header down closes it; a shorter pull springs back (ui/sheetDrag, as every sheet).

  // ---------- model ----------
  // Kept current (lib/api/models): a deleted model leaves the picker, a finished run's joins it.
  const models = modelsResource();
  // What each model's score is on (lib/scoreSummary): the picker groups models by dataset and split, best first within
  // a group (on a tie the earlier checkpoint), and never ranks one group against another.
  const runs = new Resource((s) => hub.api.runs(s));
  refreshOnModelsChange(runs);
  const datasets = new Resource((s) => hub.api.datasets(s));
  const choices = computed(() => {
    // Until the runs and datasets answer (or fail) the splits are unknown: wait, so the default never switches.
    if (!models.data || (!runs.data && !runs.error) || (!datasets.data && !datasets.error)) return [];
    const runById = new Map<string, Run>((runs.data ?? []).map((r) => [r.id, r]));
    const dsById = new Map<string, Dataset>((datasets.data ?? []).map((d) => [d.id, d]));
    return modelChoices(models.data, (m) => scoreContext(m, runById, dsById));
  });
  const ranked = computed(() => choices.get().map((c) => c.model));
  /** What the page opens on: the best model (lib/scoreSummary `betterModel`, Desktop's rule), not the newest. */
  const best = computed(() => [...ranked.get()].sort(betterModel)[0] ?? null);
  const queryModel = computed(() => router.route.query.get('model'));
  const pickedModel = signal<string | null>(null);
  const wanted = computed(() => pickedModel.get() ?? queryModel.get());
  const model = computed((): Model | null => ranked.get().find((m) => m.id === wanted.get()) ?? best.get());
  const modelId = computed(() => model.get()?.id ?? null);
  /** The chosen model's score and what it is on, under the select: "mAP50-95 0.606 · shapes-hard, validation split". */
  const pickedMeta = computed(() => choices.get().find((c) => c.model.id === modelId.get())?.meta ?? null);

  function chooseModel(id: string) {
    pickedModel.set(id);
    router.go(playgroundPath({ model: id, source: source.peek(), conf: conf.peek(), iou: iou.peek(), maxDet: maxDet.peek() }), { replace: true });
  }
  followVanished(() => model.get(), () => ranked.get(), () => wanted.get(), chooseModel);

  effect(() => {
    const s = source.get();
    if (!s) {
      src.set(null);
      return;
    }
    if (s.kind === 'file') {
      src.set(s.url);
      return;
    }
    let live = true;
    src.set(null);
    hub.api.imageUrl(s.ds.id, s.img.id).then(
      (u) => {
        if (live) src.set(u);
      },
      () => {},
    );
    return () => {
      live = false;
    };
  });

  // Decodes and linked images arrive later: only the latest choice may show, and nothing shows after destroy.
  let decodeSeq = 0;
  let destroyed = false;
  function choose() {
    const seq = ++decodeSeq;
    return () => seq === decodeSeq && !destroyed;
  }

  function setSource(s: Source) {
    decodeSeq++;
    toast.clear('image');
    const old = source.peek();
    if (old?.kind === 'file') URL.revokeObjectURL(old.url);
    source.set(s);
    seg.set(s.kind);
    pred.set(null);
    error.set(null);
    hover.set(null);
    picked.set(null);
    hidden.set(new Set());
  }

  async function setFile(file: File | Blob, name = file instanceof File ? file.name : 'pasted image') {
    const live = choose();
    // An image the server cannot read (GIF, AVIF...) comes back as a PNG (./playground/intake).
    const t = await takeImage(file, name);
    if (!live()) return;
    if ('error' in t) {
      // Until the next image is taken (setSource): it goes then, not 20 s later over the results.
      toast.error(new Error(t.error), 'Unsupported image', { key: 'image' });
      return;
    }
    setSource({ kind: 'file', file: t.file, name, url: URL.createObjectURL(t.file), width: t.width, height: t.height });
  }

  function clear() {
    decodeSeq++;
    const old = source.peek();
    if (old?.kind === 'file') URL.revokeObjectURL(old.url);
    source.set(null);
    pred.set(null);
    error.set(null);
    seg.set('file');
  }

  onCleanup(() => {
    destroyed = true;
    decodeSeq++;
    const old = source.peek();
    if (old?.kind === 'file') URL.revokeObjectURL(old.url);
  });

  function onSeg(v: Seg) {
    if (v === 'file') fileInput.click();
    else if (v === 'dataset') pickerOpen.set(true);
    // The control reflects where the current image came from; it changes when a new image arrives.
    seg.set(source.peek()?.kind ?? 'file');
  }

  function onpaste(e: ClipboardEvent) {
    const item = [...(e.clipboardData?.items ?? [])].find((i) => i.kind === 'file' && i.type.startsWith('image/'));
    const file = item?.getAsFile();
    if (!file) return;
    e.preventDefault();
    void setFile(file, 'pasted image');
  }

  function ondrop(e: DragEvent) {
    e.preventDefault();
    dragging.set(false);
    const [file, ...rest] = droppedImages(e.dataTransfer?.files);
    if (!file) return;
    void setFile(file);
    // One image runs at a time: the others are not dropped without a word.
    if (rest.length) toast.info(`Using ${file.name}`, `The Playground runs one image at a time, so the other ${rest.length === 1 ? 'image was' : `${rest.length} images were`} left out.`);
  }

  function ondragover(e: DragEvent) {
    if (!e.dataTransfer?.types.includes('Files')) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    dragging.set(true);
  }

  function ondragleave(e: DragEvent) {
    const next = e.relatedTarget as Node | null;
    if (!next || !(e.currentTarget as HTMLElement).contains(next)) dragging.set(false);
  }

  const samples = sampleImages(model);

  // ---------- where inference runs ----------
  // On the server (POST /api/predict), or in this browser through folo-wasm on WebGPU or the CPU (docs/WASM.md).
  // In the browser the model's .folo comes from the server (GET /api/models/{id}/weights); a server without that
  // route gets the file from the user instead.
  const pred = signal<Prediction | null>(null);
  const runOn = signal<RunOn>('server');
  const webgpu = hasWebGpu();
  const engine = new BrowserEngine();
  // Whether this build has the in-browser runtime (folo-wasm): what /api/system says, else one probe of its files.
  // The demo simulates the browser path, so there only its server's answer counts. Unknown (null) leaves both on.
  const system = new Resource((s) => hub.api.system(s));
  // The probe is asked on each visit, again when the server reconnects (`system` is read again) and when the browser
  // comes back online: an unknown answer (offline) is never kept (lib/inference `probeWasm`).
  const probed = signal<boolean | null>(null);
  /** The server's word on its own build counts for this page only when it served it (the demo's server always does). */
  const fromServer = computed(() => wasmFromSystem(system.data, hub.api.mock || servedBy(hub.serverUrl)));
  effect(() => {
    if (hub.api.mock || fromServer.get() !== null) return;
    let live = true;
    const ask = () => void probeWasm().then((v) => live && probed.set(v));
    ask();
    window.addEventListener('online', ask);
    return () => {
      live = false;
      window.removeEventListener('online', ask);
    };
  });
  const wasm = computed(() => fromServer.get() ?? (hub.api.mock ? (system.data ? false : null) : probed.get()));
  /** Why the browser backends are off, or null when they are offered. */
  const browserOff = computed(() => (wasm.get() === false ? NO_WASM : null));
  effect(() => {
    if (browserOff.get() && untrack(() => runOn.get()) !== 'server') runOn.set('server');
  });
  // A build without the runtime offers the server alone, so there is no choice to explain (ModelPickers hides it).
  const runOnNote = computed(() => (!browserOff.get() && !webgpu ? 'No WebGPU in this browser' : null));
  /** .folo files the user picked, by model id. */
  const localWeights = signal(new Map<string, File>());
  /** The server cannot send this model's file: ask for it. */
  const weightsMissing = signal(false);
  let weightsInput!: HTMLInputElement;
  const where = computed(() => {
    const p = pred.get();
    return p?.backend ? `browser · ${BACKEND_NAME[p.backend]}` : 'server';
  });
  const deviceText = computed(() => {
    const p = pred.get();
    return p?.backend ? `${BACKEND_NAME[p.backend]}${p.backend === 'webgpu' && engine.info?.adapter ? ` · ${engine.info.adapter}` : ''}` : (p?.device ?? '');
  });

  /** A picked file that was refused as weights (not a .folo), or that the runtime could not load; cleared on a new pick. */
  const badFile = signal<{ name: string; why: string | null } | null>(null);

  async function pickWeights(e: Event & { currentTarget: HTMLInputElement }) {
    const input = e.currentTarget;
    const f = input.files?.[0];
    input.value = '';
    const id = modelId.peek();
    if (!f || !id) return;
    // Refused before it becomes this model's weights: the server's (or the request for a file) stay in place.
    if (!(await looksLikeFolo(f))) {
      badFile.set({ name: f.name, why: null });
      return;
    }
    badFile.set(null);
    localWeights.set(new Map(localWeights.peek()).set(id, f));
  }

  /** Drops a picked file that failed to load, so the model goes back to the server's weights. */
  function dropLocal(id: string, f: File, e: unknown) {
    const next = new Map(localWeights.peek());
    next.delete(id);
    localWeights.set(next);
    badFile.set({ name: f.name, why: e instanceof Error ? sentence(e.message) : null });
  }

  // ---------- prediction ----------
  const predicting = signal(false);
  const error = signal<unknown>(null);
  const retryTick = signal(0);

  // ---------- client-side filtering ----------
  const conf = signal(DEFAULTS.conf);
  const iou = signal(DEFAULTS.iou);
  const maxDet = signal(DEFAULTS.maxDet);
  const labels = signal(true);
  const scores = signal(true);
  const truth = signal(false);
  const hidden = signal<Set<number>>(new Set());
  // Inspection: hovering a box or row is transient; click, tap or keyboard focus picks one until Esc or a click
  // elsewhere. Tracked by object, since filtering re-indexes the kept boxes.
  const hover = signal<Det | null>(null);
  const picked = signal<Det | null>(null);
  const tbody = signal<HTMLTableSectionElement>();
  const tab = signal<'detections' | 'json'>('detections');

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
    setSource,
    choose,
    model: () => modelId.get() ?? queryModel.get() ?? null,
    pickModel: (id) => pickedModel.set(id),
  });

  effect(() => {
    const id = modelId.get();
    const s = source.get();
    const on = runOn.get();
    const file = localWeights.get().get(id ?? '');
    void retryTick.get();
    if (!id || !s || untrack(() => restored(id, s, on))) return;
    const ctrl = new AbortController();
    untrack(() => {
      predicting.set(true);
      error.set(null);
      weightsMissing.set(false);
      // Results belong to a model: never show another model's boxes while (or if) this one fails.
      const p = pred.get();
      if (p && p.model_id !== id) {
        pred.set(null);
        hidden.set(new Set());
        hover.set(null);
        picked.set(null);
      }
    });
    const run = ask({ api: hub.api, engine, model: id, source: s, on, file: file ?? null, options: FETCH, signal: ctrl.signal });
    run.then(
      (p) => {
        if (ctrl.signal.aborted) return;
        pred.set(p);
        predicting.set(false);
      },
      (e: unknown) => {
        if (ctrl.signal.aborted) return;
        predicting.set(false);
        // A picked file the runtime could not load is not kept as this model's weights.
        if (file && on !== 'server' && engine.phase !== 'ready') {
          dropLocal(id, file, e);
          return;
        }
        // A server without the weights route (or one that cannot read this model's file): ask for the file.
        if (on !== 'server' && !file && noServerWeights(e)) {
          weightsMissing.set(true);
          return;
        }
        // Said once, in place above the image (with its ways out): no toast repeats it, and no earlier result stays on
        // screen as if it were this run's.
        pred.set(null);
        error.set(e);
      },
    );
    return () => ctrl.abort();
  });

  // A busy image's answer is cut at FETCH.max_det candidates: a confidence below the cut asks for all of them.
  const cut = askAllWhenCut({
    pred,
    conf: () => conf.get(),
    model: () => modelId.get(),
    source: () => source.get(),
    runOn: () => runOn.get(),
    file: (id) => localWeights.get().get(id) ?? null,
    busy: () => predicting.get(),
    engine,
    // A picked .folo is not kept with the session: a kept answer asks for it again, as a first one does.
    weightsMissing: () => weightsMissing.set(true),
  });

  // Out of GPU memory on the server (507): this browser has its own device, and a smaller model of the same classes
  // may fit where this one does not.
  const oom = computed(() => runOn.get() === 'server' && isOom(error.get()));
  const smaller = computed(() => {
    const m = model.get();
    if (!m) return null;
    const rank = (x: Model) => ARCH_ORDER.indexOf(x.arch);
    return ranked.get().find((x) => x.id !== m.id && rank(x) < rank(m) && x.classes.join('|') === m.classes.join('|')) ?? null;
  });
  const localBackend: Backend = webgpu ? 'webgpu' : 'cpu';

  // What stands between the image and its boxes, shown in a strip under the image (which stays in view).
  const notice = computed(() => (!source.get() || predicting.get() ? null : weightsMissing.get() ? 'file' : error.get() && !pred.get() ? 'error' : null));
  effect(() => {
    void modelId.get();
    badFile.set(null);
  });
  // Focus moves to the notice's heading as it appears, so it is announced and its actions are next in tab order. The
  // move never scrolls the page, and the heading keeps focus while the notice stays (./playground/noticeFocus).
  const noticeEl = signal<HTMLElement>();
  effect(() => {
    const el = noticeEl.get();
    if (el) return holdNoticeFocus(el);
  });

  const result = computed(() => {
    const p = pred.get();
    if (!p) return null;
    const t0 = performance.now();
    // An end-to-end (o2o) model predicts without NMS (docs/MODEL.md §6.6): an IoU of 1 suppresses nothing, as the server.
    const boxes = filterDetections(p.boxes, { conf: conf.get(), iou: model.get()?.head === 'o2o' ? 1 : iou.get(), maxDet: maxDet.get() });
    return { boxes, ms: performance.now() - t0 };
  });
  const kept = computed(() => result.get()?.boxes ?? []);
  const classes = computed(() => model.get()?.classes ?? []);
  const only = computed(() => {
    const h = hidden.get();
    return h.size ? new Set(classes.get().map((_, i) => i).filter((i) => !h.has(i))) : null;
  });
  const rows = computed(() => {
    const h = hidden.get();
    return kept
      .get()
      .map((b, i) => ({ b, i }))
      .filter((r) => !h.has(r.b.class));
  });
  const counts = computed(() => {
    const m = new Map<number, number>();
    for (const b of kept.get()) m.set(b.class, (m.get(b.class) ?? 0) + 1);
    return [...m.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  });
  const maxCount = computed(() => Math.max(1, ...counts.get().map((c) => c[1])));

  const at = (b: Det | null) => {
    const i = b && !hidden.get().has(b.class) ? kept.get().indexOf(b) : -1;
    return i < 0 ? null : i;
  };
  const selectedIdx = computed(() => at(picked.get()));
  const activeIdx = computed(() => at(hover.get()) ?? selectedIdx.get());
  /** The row that takes Tab focus (roving tabindex): the selected one, else the first. */
  const tabRow = computed(() => selectedIdx.get() ?? rows.get()[0]?.i ?? null);

  const rowEl = (i: number) => tbody.peek()?.querySelector<HTMLTableRowElement>(`tr[data-det="${i}"]`) ?? null;

  /** A box picked on the image: bring its row into view inside the list without moving the page. */
  function pickFromImage(i: number) {
    const b = kept.peek()[i];
    if (!b) return;
    const next = picked.peek() === b ? null : b;
    picked.set(next);
    const row = next && rowEl(i);
    const box = row?.closest<HTMLElement>('.scroll');
    if (!row || !box) return;
    const head = box.querySelector('thead')?.offsetHeight ?? 0;
    if (row.offsetTop - head < box.scrollTop) box.scrollTop = row.offsetTop - head;
    else if (row.offsetTop + row.offsetHeight > box.scrollTop + box.clientHeight) box.scrollTop = row.offsetTop + row.offsetHeight - box.clientHeight;
  }

  function onrowkey(e: KeyboardEvent, k: number) {
    const list = rows.peek();
    const to = e.key === 'ArrowDown' ? k + 1 : e.key === 'ArrowUp' ? k - 1 : e.key === 'Home' ? 0 : e.key === 'End' ? list.length - 1 : null;
    if (to === null) return;
    e.preventDefault();
    const r = list[Math.max(0, Math.min(list.length - 1, to))];
    if (r) rowEl(r.i)?.focus();
  }

  function onwindowkey(e: KeyboardEvent) {
    if (e.key !== 'Escape' || pickerOpen.peek()) return;
    if (sheetShown.peek()) closeSheet();
    else if (picked.peek()) picked.set(null);
  }

  function onwindowdown(e: PointerEvent) {
    if (picked.peek() && !(e.target as Element | null)?.closest?.('[data-det], [data-box]')) picked.set(null);
  }

  onWindow('paste', onpaste);
  onWindow('keydown', onwindowkey);
  onWindow('pointerdown', onwindowdown);

  const dims = computed(() => {
    const p = pred.get();
    if (p) return p.image;
    const s = source.get();
    if (s?.kind === 'file') return { width: s.width, height: s.height };
    if (s?.kind === 'dataset') return { width: s.img.width, height: s.img.height };
    return null;
  });
  const sourceName = computed(() => {
    const s = source.get();
    return s ? (s.kind === 'file' ? s.name : `${s.ds.name} / ${s.img.file}`) : '';
  });
  const shortName = computed(() => {
    const s = source.get();
    return s ? (s.kind === 'file' ? s.name : (s.img.file.split('/').pop() ?? s.img.file)) : '';
  });
  const truthBoxes = computed(() => {
    const s = source.get();
    return truth.get() && s?.kind === 'dataset' ? s.img.boxes : [];
  });

  const json = computed(() => {
    const p = pred.get();
    return p && result.get()
      ? {
          model_id: p.model_id,
          image: p.image,
          boxes: kept.get().map((b) => ({ class: b.class, name: b.name, score: b.score, x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2 })),
          timing: p.timing,
          device: p.device,
          filter: { conf: conf.get(), iou: iou.get(), max_det: maxDet.get() },
        }
      : null;
  });

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
  const isDefault = computed(() => conf.get() === DEFAULTS.conf && iou.get() === DEFAULTS.iou && maxDet.get() === DEFAULTS.maxDet);

  // ---------- snippets ----------
  const toggle = (label: string, on: () => boolean, set: (v: boolean) => void) => (
    <button type="button" class="tog" class:on={on()} aria-pressed={on()} onclick={() => set(!on())}>
      <span class="tog-box" aria-hidden="true">
        {on() ? <Check size={10} strokeWidth={3} /> : null}
      </span>
      {label}
    </button>
  );

  const displayToggles = () => (
    <>
      {toggle('Labels', () => labels.get(), (v) => labels.set(v))}
      {toggle('Scores', () => scores.get(), (v) => scores.set(v))}
      {source.get()?.kind === 'dataset' ? toggle('Ground truth', () => truth.get(), (v) => truth.set(v)) : null}
    </>
  );

  const secondarySliders = (inline: boolean) => (
    <>
      <div title={NMS_TIP}>
        <Slider inline={inline} label="Overlap (IoU)" bind:value={iou} min={0.05} max={0.95} step={0.05} hint="Suppress same-class overlaps above this." />
      </div>
      <Slider inline={inline} label="Max detections" bind:value={maxDet} min={1} max={300} step={1} format={round} hint="Keep at most this many boxes." />
    </>
  );

  const moreBody = () => (
    <div class="sheet-body">
      <div class="more-sliders">
        {tablet.get() ? (
          <Slider inline label="Max detections" bind:value={maxDet} min={1} max={300} step={1} format={round} />
        ) : (
          <>
            <Slider inline label="Confidence" bind:value={conf} min={0.01} max={0.95} step={0.01} />
            {secondarySliders(true)}
          </>
        )}
      </div>
      <div class="more-toggles">{displayToggles()}</div>
    </div>
  );

  const resetButton = () => (
    <Button size="sm" variant="ghost" icon={RotateCcw} disabled={isDefault.get()} onclick={resetThresholds}>
      Reset
    </Button>
  );

  const showAllButton = () => (
    <>
      {hidden.get().size ? (
        <Button size="sm" variant="ghost" onclick={() => hidden.set(new Set())}>
          Show all
        </Button>
      ) : null}
    </>
  );

  const classList = () => (
    <ClassList
      result={result.get()}
      counts={counts.get()}
      pred={pred.get()}
      hidden={hidden.get()}
      classes={classes.get()}
      maxCount={maxCount.get()}
      toggleClass={toggleClass}
    />
  );

  const timingBody = () => (
    <>
      {pred.get() ? (
        <TimingBar timing={pred.get()!.timing} device={deviceText.get()} cold={pred.get()!.cold} />
      ) : predicting.get() ? (
        <div class="t-sk">
          <Skeleton h="28px" w="50%" />
          <Skeleton h="8px" />
          <Skeleton h="14px" />
          <Skeleton h="14px" />
        </div>
      ) : (
        <p class="hint">Timing appears after a run.</p>
      )}
    </>
  );

  const intro = () => <span>Run a model on any image. Thresholds re-filter instantly in the browser.</span>;

  // The model and where it runs: on the title row from 1024 px up, a toolbar row under the title below that.
  const pickers = () => (
    <ModelPickers
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
        actions={narrow.current || !ranked.get().length ? undefined : pickers}
      >
        {narrow.current && ranked.get().length ? (
          <ModelPickers
            toolbar
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
        ) : null}

        {models.error && !models.data ? (
          <Panel>
            <ErrorState error={models.error} retry={() => models.refresh()} title="Could not load models" compact />
          </Panel>
        ) : models.data && models.data.length === 0 ? (
          <Panel>
            <EmptyState
              icon={Box}
              title="No models to run"
              description="Train a model first; its best weights show up here when the run finishes."
              actions={() => (
                <Button icon={Plus} href={href('/runs/new')}>
                  New run
                </Button>
              )}
            />
          </Panel>
        ) : (
          /* 8 + 4 columns; in one column (below 1024 px) the canvas takes the image's aspect (see .layout.fit). */
          <div class="layout" class:fit={!!source.get() && !!dims.get()}>
            <div class="main">
              <ImagePanel
                notice={notice.get()}
                bind:noticeEl={noticeEl}
                badFile={badFile.get()}
                model={model.get()}
                chooseWeights={() => weightsInput.click()}
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
                pred={pred.get()}
                cut={cut()}
                kept={kept.get()}
                truthBoxes={truthBoxes.get()}
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
              />

              <ResultsPanel
                result={result.get()}
                bind:tab={tab}
                rows={rows.get()}
                predicting={predicting.get()}
                notice={notice.get()}
                json={json.get()}
                pred={pred.get()}
                bind:conf={conf}
                hidden={hidden.get()}
                bind:tbody={tbody}
                tabRow={tabRow.get()}
                activeIdx={activeIdx.get()}
                selectedIdx={selectedIdx.get()}
                bind:hover={hover}
                bind:picked={picked}
                onrowkey={onrowkey}
              />

              {source.get() && phone.current ? (
                <PhoneDetails
                  result={result.get()}
                  counts={counts.get()}
                  hidden={hidden.get()}
                  pred={pred.get()}
                  where={where.get()}
                  showAllButton={showAllButton}
                  classList={classList}
                  timingBody={timingBody}
                />
              ) : null}
            </div>

            <SidePanels
              source={source.get()}
              result={result.get()}
              counts={counts.get()}
              pred={pred.get()}
              where={where.get()}
              bind:conf={conf}
              resetButton={resetButton}
              secondarySliders={secondarySliders}
              displayToggles={displayToggles}
              showAllButton={showAllButton}
              classList={classList}
              timingBody={timingBody}
            />
          </div>
        )}
      </Page>

      <DatasetPicker
        bind:open={pickerOpen}
        prefer={classes.get()}
        onpick={(ds, img, at) => setSource({ kind: 'dataset', ds, img, at })}
        onclose={() => seg.set(source.peek()?.kind ?? 'file')}
      />
    </>
  );
}

css`
  /* The page grid's 8 + 4 split (the rail is always 4 columns), the same with or without an image. */
  .layout {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    gap: var(--space-4);
    align-items: start;
    --stage-h: max(360px, calc(100dvh - 220px));
  }
  .main {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: var(--space-4);
    min-width: 0;
  }
  /* The first toggle's box lines up with the slider labels above. */
  :global(.show) .tog:first-child {
    margin-left: calc(-1 * var(--space-2));
  }
  .tog {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: var(--control-sm);
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    white-space: nowrap;
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .tog:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .tog.on {
    color: var(--text-primary);
  }
  .tog-box {
    display: grid;
    place-items: center;
    width: 12px;
    height: 12px;
    border-radius: 3px;
    /* The empty box's edge is line.control (>= 3:1); checked is neutral (lime is for actions, focus and live). */
    box-shadow: inset 0 0 0 1px var(--line-control);
  }
  .tog.on .tog-box {
    background: var(--text-primary);
    color: var(--text-inverse);
    box-shadow: none;
  }
  .hint {
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .t-sk {
    display: grid;
    gap: var(--space-3);
  }
  .sheet-body {
    overflow-y: auto;
    overscroll-behavior: contain;
    padding: 0 var(--space-4) var(--space-3);
  }
  /* One label column and one value column: every track (the pinned Confidence, the sheet's three) starts and ends at
     the same x. */
  .more-sliders :global(.slider.inline) {
    grid-template-columns: 112px minmax(0, 1fr) 40px;
  }
  .more-sliders :global(.slider.inline) {
    min-height: 44px;
  }
  .more-toggles {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 2px;
    min-height: 44px;
    margin-left: calc(-1 * var(--space-2));
  }
  @media (pointer: coarse) {
    .more-toggles .tog {
      height: 44px;
    }
  }
  @media (max-width: 1023px) {
    .layout {
      grid-template-columns: minmax(0, 1fr);
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
