<script lang="ts">
  import ArrowUpRight from '@lucide/svelte/icons/arrow-up-right';
  import Box from '@lucide/svelte/icons/box';
  import FileUp from '@lucide/svelte/icons/file-up';
  import Check from '@lucide/svelte/icons/check';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import ChevronUp from '@lucide/svelte/icons/chevron-up';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import ImageIcon from '@lucide/svelte/icons/image';
  import Images from '@lucide/svelte/icons/images';
  import Plus from '@lucide/svelte/icons/plus';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import Upload from '@lucide/svelte/icons/upload';
  import X from '@lucide/svelte/icons/x';
  import { onDestroy, untrack } from 'svelte';
  import { MediaQuery } from 'svelte/reactivity';
  import { ApiError } from '$lib/api/client';
  import { hub } from '$lib/api/index.svelte';
  import { MockApi } from '$lib/api/mock';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Dataset, DatasetImage as Img, Model, PredictInput, Prediction } from '$lib/api/types';
  import { classColor, media } from '$lib/design/tokens';
  import DetectionView from '$lib/detect/DetectionView.svelte';
  import { filterDetections } from '$lib/detect/nms';
  import { BrowserEngine, hasWebGpu, NO_WASM, PHASE_TEXT, probeWasm, servedBy, wasmFromSystem, type Backend } from '$lib/inference/browser.svelte';
  import { metric, ms, num, sentence } from '$lib/format';
  import { isOom } from '$lib/oom';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import SheetHandle from '$lib/ui/SheetHandle.svelte';
  import { sheetDrag } from '$lib/ui/sheetDrag';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import Slider from '$lib/ui/Slider.svelte';
  import Tabs from '$lib/ui/Tabs.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import DatasetPicker from './playground/DatasetPicker.svelte';
  import DropZone from './playground/DropZone.svelte';
  import { looksLikeFolo } from './playground/folo';
  import JsonView from './playground/JsonView.svelte';
  import { holdNoticeFocus } from './playground/noticeFocus';
  import TimingBar from './playground/TimingBar.svelte';

  setCrumbs({ label: 'Playground' });

  // The server is asked once with loose settings and no suppression (IoU 1 keeps every candidate); the controls below
  // re-filter and run NMS in the browser. Suppressing on the server first would lose boxes a looser IoU should keep.
  const FETCH = { conf: 0.01, iou: 1, max_det: 1000 };
  const DEFAULTS = { conf: 0.25, iou: 0.7, maxDet: 300 };

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
  const tablet = $derived(narrow.current && !phone.current);
  let moreOpen = $state(false);
  let sheetEl = $state<HTMLElement>();
  let moreBtn = $state<HTMLElement>();

  // Focus moves into the sheet (or the tablet's popover) as it opens and back to its button as it closes.
  $effect(() => {
    if (sheetShown) sheetEl?.focus();
  });
  // The tablet's popover is transient: a press outside it (and outside its button) closes it, focus staying put.
  $effect(() => {
    if (!sheetShown || !tablet) return;
    const away = (e: PointerEvent) => {
      const t = e.target as Node;
      if (!sheetEl?.contains(t) && !moreBtn?.contains(t)) moreOpen = false;
    };
    document.addEventListener('pointerdown', away);
    return () => document.removeEventListener('pointerdown', away);
  });
  function closeSheet() {
    moreOpen = false;
    moreBtn?.querySelector('button')?.focus();
  }
  // Dragging the sheet's header down closes it; a shorter pull springs back (ui/sheetDrag, as every sheet).

  // ---------- model ----------
  const models = new Resource((s) => hub.api.models(s));
  // Best first; on a tie the earlier checkpoint (a run's best weights are saved before its last).
  const ranked = $derived(
    [...(models.data ?? [])].sort((a, b) => (b.metrics?.map50_95 ?? -1) - (a.metrics?.map50_95 ?? -1) || a.created.localeCompare(b.created)),
  );
  const queryModel = $derived(router.route.query.get('model'));
  let pickedModel = $state<string | null>(null);
  const model = $derived.by((): Model | null => {
    const list = ranked;
    const want = pickedModel ?? queryModel;
    return list.find((m) => m.id === want) ?? list[0] ?? null;
  });
  const modelId = $derived(model?.id ?? null);

  function chooseModel(id: string) {
    pickedModel = id;
    router.go(`/playground?model=${encodeURIComponent(id)}`, { replace: true });
  }

  // ---------- image source ----------
  type Source = { kind: 'file'; file: Blob; name: string; url: string; width: number; height: number } | { kind: 'dataset'; ds: Dataset; img: Img };
  let source = $state.raw<Source | null>(null);
  const sheetShown = $derived(moreOpen && narrow.current && !!source);
  // Phones' sheet holds every threshold; a tablet's, what its strip leaves out.
  const sheetTitle = $derived(tablet ? 'More settings' : 'Thresholds');
  let src = $state<string | null>(null);
  let seg = $state<'file' | 'dataset' | 'webcam'>('file');
  let pickerOpen = $state(false);
  let fileInput: HTMLInputElement;
  let dragging = $state(false);

  $effect(() => {
    const s = source;
    if (!s) {
      src = null;
      return;
    }
    if (s.kind === 'file') {
      src = s.url;
      return;
    }
    let live = true;
    src = null;
    hub.api.imageUrl(s.ds.id, s.img.id).then(
      (u) => {
        if (live) src = u;
      },
      () => {},
    );
    return () => {
      live = false;
    };
  });

  function setSource(s: Source) {
    decodeSeq++;
    if (source?.kind === 'file') URL.revokeObjectURL(source.url);
    source = s;
    seg = s.kind;
    pred = null;
    error = null;
    hover = null;
    picked = null;
    hidden = new Set();
  }

  // Decodes finish asynchronously: only the latest request may commit, and nothing commits after destroy.
  let decodeSeq = 0;
  let destroyed = false;

  async function setFile(file: File | Blob, name = file instanceof File ? file.name : 'pasted image') {
    const seq = ++decodeSeq;
    if (file.type && !file.type.startsWith('image/')) {
      toast.error(new Error(`${name} is not an image.`), 'Unsupported file');
      return;
    }
    let width = 0;
    let height = 0;
    try {
      const bmp = await createImageBitmap(file);
      width = bmp.width;
      height = bmp.height;
      bmp.close();
    } catch {
      if (seq === decodeSeq && !destroyed) toast.error(new Error('That file could not be decoded. Use a JPEG, PNG or WebP.'), 'Unsupported image');
      return;
    }
    if (seq !== decodeSeq || destroyed) return;
    setSource({ kind: 'file', file, name, url: URL.createObjectURL(file), width, height });
  }

  function clear() {
    decodeSeq++;
    if (source?.kind === 'file') URL.revokeObjectURL(source.url);
    source = null;
    pred = null;
    error = null;
    seg = 'file';
  }

  onDestroy(() => {
    destroyed = true;
    decodeSeq++;
    if (source?.kind === 'file') URL.revokeObjectURL(source.url);
  });

  function onSeg(v: 'file' | 'dataset' | 'webcam') {
    if (v === 'file') fileInput.click();
    else if (v === 'dataset') pickerOpen = true;
    // The control reflects where the current image came from; it changes when a new image arrives.
    seg = source?.kind ?? 'file';
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
    dragging = false;
    const file = e.dataTransfer?.files[0];
    if (file) void setFile(file);
  }

  function ondragover(e: DragEvent) {
    if (!e.dataTransfer?.types.includes('Files')) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    dragging = true;
  }

  function ondragleave(e: DragEvent) {
    const next = e.relatedTarget as Node | null;
    if (!next || !(e.currentTarget as HTMLElement).contains(next)) dragging = false;
  }

  // Sample images for the empty state: a few from each dataset, the selected model's own dataset first.
  const samplePool = new Resource(async (s) => {
    const list = await hub.api.datasets(s);
    const per = await Promise.all(
      list.slice(0, 4).map((d) =>
        hub.api
          .datasetImages(d.id, { split: d.splits.val ? 'val' : 'train', limit: 3 }, s)
          .then((r) => r.images.map((img) => ({ ds: d, img })))
          .catch(() => []),
      ),
    );
    return per;
  });
  const samples = $derived.by(() => {
    if (!samplePool.data) return samplePool.error ? [] : undefined;
    const key = model?.classes.join('|');
    const groups = [...samplePool.data].sort((a, b) => Number(b[0]?.ds.classes.join('|') === key) - Number(a[0]?.ds.classes.join('|') === key));
    return groups.flat().slice(0, 6);
  });

  // ---------- where inference runs ----------
  // On the server (POST /api/predict), or in this browser through folo-wasm on WebGPU or the CPU (docs/WASM.md).
  // In the browser the model's .folo comes from the server (GET /api/models/{id}/weights); a server without that
  // route gets the file from the user instead.
  let pred = $state.raw<Prediction | null>(null);
  type RunOn = 'server' | Backend;
  let runOn = $state<RunOn>('server');
  const webgpu = hasWebGpu();
  const engine = new BrowserEngine();
  // Whether this build has the in-browser runtime (folo-wasm): what /api/system says, else one probe of its files.
  // The demo simulates the browser path, so there only its server's answer counts. Unknown (null) leaves both on.
  const system = new Resource((s) => hub.api.system(s));
  // The probe is asked on each visit, again when the server reconnects (`system` is read again) and when the browser
  // comes back online: an unknown answer (offline) is never kept (lib/inference `probeWasm`).
  let probed = $state<boolean | null>(null);
  /** The server's word on its own build counts for this page only when it served it. */
  const fromServer = $derived(wasmFromSystem(system.data, servedBy(hub.serverUrl)));
  $effect(() => {
    if (hub.api.mock || fromServer !== null) return;
    let live = true;
    const ask = () => void probeWasm().then((v) => live && (probed = v));
    ask();
    window.addEventListener('online', ask);
    return () => {
      live = false;
      window.removeEventListener('online', ask);
    };
  });
  const wasm = $derived(fromServer ?? (hub.api.mock ? (system.data ? false : null) : probed));
  /** Why the browser backends are off, or null when they are offered. */
  const browserOff = $derived(wasm === false ? NO_WASM : null);
  $effect(() => {
    if (browserOff && untrack(() => runOn) !== 'server') runOn = 'server';
  });
  const runOnNote = $derived(browserOff ? 'No in-browser runtime' : !webgpu ? 'No WebGPU in this browser' : null);
  /** .folo files the user picked, by model id. */
  let localWeights = $state.raw(new Map<string, File>());
  /** The server cannot send this model's file: ask for it. */
  let weightsMissing = $state(false);
  let weightsInput: HTMLInputElement;
  const BACKEND_NAME: Record<Backend, string> = { webgpu: 'WebGPU', cpu: 'CPU' };
  const where = $derived(pred?.backend ? `browser · ${BACKEND_NAME[pred.backend]}` : 'server');
  const deviceText = $derived(
    pred?.backend ? `${BACKEND_NAME[pred.backend]}${pred.backend === 'webgpu' && engine.info?.adapter ? ` · ${engine.info.adapter}` : ''}` : (pred?.device ?? ''),
  );

  /** A picked file that was refused as weights (not a .folo), or that the runtime could not load; cleared on a new pick. */
  let badFile = $state<{ name: string; why: string | null } | null>(null);

  async function pickWeights(e: Event) {
    const f = (e.currentTarget as HTMLInputElement).files?.[0];
    (e.currentTarget as HTMLInputElement).value = '';
    const id = modelId;
    if (!f || !id) return;
    // Refused before it becomes this model's weights: the server's (or the request for a file) stay in place.
    if (!(await looksLikeFolo(f))) {
      badFile = { name: f.name, why: null };
      return;
    }
    badFile = null;
    localWeights = new Map(localWeights).set(id, f);
  }

  /** Drops a picked file that failed to load, so the model goes back to the server's weights. */
  function dropLocal(id: string, f: File, e: unknown) {
    const next = new Map(localWeights);
    next.delete(id);
    localWeights = next;
    badFile = { name: f.name, why: e instanceof Error ? sentence(e.message) : null };
  }

  /** The demo has no real networks: the browser path is simulated with the documented speeds (docs/WASM.md). */
  function simulate(p: Prediction, backend: Backend): Prediction {
    const t = backend === 'webgpu' ? { preprocess_ms: 2.5, inference_ms: 11.1, postprocess_ms: 0.2 } : { preprocess_ms: 6, inference_ms: 477, postprocess_ms: 8 };
    return { ...p, backend, device: backend === 'webgpu' ? 'gpu:0' : 'cpu', timing: { ...t, total_ms: t.preprocess_ms + t.inference_ms + t.postprocess_ms } };
  }

  async function imageBlob(s: Source, signal: AbortSignal): Promise<Blob> {
    if (s.kind === 'file') return s.file;
    const res = await fetch(await hub.api.imageUrl(s.ds.id, s.img.id), { signal });
    if (!res.ok) throw new Error(`Could not read ${s.img.file} (HTTP ${res.status}).`);
    return res.blob();
  }

  // ---------- prediction ----------
  let predicting = $state(false);
  let error = $state.raw<unknown>(null);
  let retryTick = $state(0);

  $effect(() => {
    const id = modelId;
    const s = source;
    const on = runOn;
    const file = localWeights.get(id ?? '');
    void retryTick;
    if (!id || !s) return;
    const input: PredictInput = s.kind === 'file' ? { kind: 'file', file: s.file } : { kind: 'dataset', dataset_id: s.ds.id, image_id: s.img.id };
    const ctrl = new AbortController();
    untrack(() => {
      predicting = true;
      error = null;
      weightsMissing = false;
      // Results belong to a model: never show another model's boxes while (or if) this one fails.
      if (pred && pred.model_id !== id) {
        pred = null;
        hidden = new Set();
        hover = null;
        picked = null;
      }
    });
    const run =
      on === 'server'
        ? hub.api.predict(input, { model_id: id, ...FETCH }, ctrl.signal)
        : hub.api instanceof MockApi
          ? hub.api.predictInPage(input, { model_id: id, ...FETCH }).then((p) => simulate(p, on))
          : (file ? Promise.resolve(null) : hub.api.modelWeightsTag(id, ctrl.signal)).then((tag) => {
              // Superseded while asking for the ETag: do not start loading or running a model nobody waits for.
              if (ctrl.signal.aborted) throw new DOMException('Superseded', 'AbortError');
              return engine.predict({
                modelId: id,
                file: file ?? null,
                download: () => hub.api.modelWeights(id),
                tag,
                backend: on,
                image: () => imageBlob(s, ctrl.signal),
                options: { conf: FETCH.conf, iou: FETCH.iou, maxDet: FETCH.max_det },
              });
            });
    run.then(
      (p) => {
        if (ctrl.signal.aborted) return;
        pred = p;
        predicting = false;
      },
      (e: unknown) => {
        if (ctrl.signal.aborted) return;
        predicting = false;
        // A picked file the runtime could not load is not kept as this model's weights.
        if (file && on !== 'server' && engine.phase !== 'ready') {
          dropLocal(id, file, e);
          return;
        }
        // A server without the weights route (or one that cannot read this model's file): ask for the file.
        if (on !== 'server' && !file && e instanceof ApiError && (e.status === 404 || e.status === 405 || e.status === 501) && e.code !== 'model_not_found') {
          weightsMissing = true;
          return;
        }
        error = e;
        // Out of GPU memory is answered in place, with its ways out; a toast would only repeat it.
        if (!(on === 'server' && isOom(e))) toast.error(e, 'Prediction failed');
      },
    );
    return () => ctrl.abort();
  });

  // Out of GPU memory on the server (507): this browser has its own device, and a smaller model of the same classes
  // may fit where this one does not.
  const oom = $derived(runOn === 'server' && isOom(error));
  const ARCH_ORDER = ['folo-n', 'folo-s', 'folo-m', 'folo-l', 'folo-x'];
  const smaller = $derived.by(() => {
    const m = model;
    if (!m) return null;
    const rank = (x: Model) => ARCH_ORDER.indexOf(x.arch);
    return ranked.find((x) => x.id !== m.id && rank(x) < rank(m) && x.classes.join('|') === m.classes.join('|')) ?? null;
  });
  const localBackend: Backend = webgpu ? 'webgpu' : 'cpu';

  // What stands between the image and its boxes, shown in a strip under the image (which stays in view).
  const notice = $derived(!source || predicting ? null : weightsMissing ? 'file' : error && !pred ? 'error' : null);
  $effect(() => {
    void modelId;
    badFile = null;
  });
  // Focus moves to the notice's heading as it appears, so it is announced and its actions are next in tab order. The
  // move never scrolls the page, and the heading keeps focus while the notice stays (./playground/noticeFocus).
  let noticeEl = $state<HTMLElement>();
  $effect(() => {
    const el = noticeEl;
    if (el) return holdNoticeFocus(el);
  });

  // ---------- client-side filtering ----------
  let conf = $state(DEFAULTS.conf);
  let iou = $state(DEFAULTS.iou);
  let maxDet = $state(DEFAULTS.maxDet);
  let labels = $state(true);
  let scores = $state(true);
  let truth = $state(false);
  let hidden = $state<Set<number>>(new Set());
  // Inspection: hovering a box or row is transient; click, tap or keyboard focus picks one until Esc or a click
  // elsewhere. Tracked by object, since filtering re-indexes the kept boxes.
  type Det = Prediction['boxes'][number];
  let hover = $state.raw<Det | null>(null);
  let picked = $state.raw<Det | null>(null);
  let tbody = $state<HTMLTableSectionElement>();
  let tab = $state<'detections' | 'json'>('detections');

  const result = $derived.by(() => {
    if (!pred) return null;
    const t0 = performance.now();
    const boxes = filterDetections(pred.boxes, { conf, iou, maxDet });
    return { boxes, ms: performance.now() - t0 };
  });
  const kept = $derived(result?.boxes ?? []);
  const classes = $derived(model?.classes ?? []);
  const only = $derived(hidden.size ? new Set(classes.map((_, i) => i).filter((i) => !hidden.has(i))) : null);
  const rows = $derived(kept.map((b, i) => ({ b, i })).filter((r) => !hidden.has(r.b.class)));
  const counts = $derived.by(() => {
    const m = new Map<number, number>();
    for (const b of kept) m.set(b.class, (m.get(b.class) ?? 0) + 1);
    return [...m.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  });
  const maxCount = $derived(Math.max(1, ...counts.map((c) => c[1])));

  const at = (b: Det | null) => {
    const i = b && !hidden.has(b.class) ? kept.indexOf(b) : -1;
    return i < 0 ? null : i;
  };
  const selectedIdx = $derived(at(picked));
  const activeIdx = $derived(at(hover) ?? selectedIdx);
  /** The row that takes Tab focus (roving tabindex): the selected one, else the first. */
  const tabRow = $derived(selectedIdx ?? rows[0]?.i ?? null);

  const rowEl = (i: number) => tbody?.querySelector<HTMLTableRowElement>(`tr[data-det="${i}"]`) ?? null;

  /** A box picked on the image: bring its row into view inside the list without moving the page. */
  function pickFromImage(i: number) {
    const b = kept[i];
    if (!b) return;
    picked = picked === b ? null : b;
    const row = picked && rowEl(i);
    const box = row?.closest<HTMLElement>('.scroll');
    if (!row || !box) return;
    const head = box.querySelector('thead')?.offsetHeight ?? 0;
    if (row.offsetTop - head < box.scrollTop) box.scrollTop = row.offsetTop - head;
    else if (row.offsetTop + row.offsetHeight > box.scrollTop + box.clientHeight) box.scrollTop = row.offsetTop + row.offsetHeight - box.clientHeight;
  }

  function onrowkey(e: KeyboardEvent, k: number) {
    const to = e.key === 'ArrowDown' ? k + 1 : e.key === 'ArrowUp' ? k - 1 : e.key === 'Home' ? 0 : e.key === 'End' ? rows.length - 1 : null;
    if (to === null) return;
    e.preventDefault();
    const r = rows[Math.max(0, Math.min(rows.length - 1, to))];
    if (r) rowEl(r.i)?.focus();
  }

  function onwindowkey(e: KeyboardEvent) {
    if (e.key !== 'Escape' || pickerOpen) return;
    if (sheetShown) closeSheet();
    else if (picked) picked = null;
  }

  function onwindowdown(e: PointerEvent) {
    if (picked && !(e.target as Element | null)?.closest?.('[data-det], [data-box]')) picked = null;
  }

  const dims = $derived.by(() => {
    if (pred) return pred.image;
    if (source?.kind === 'file') return { width: source.width, height: source.height };
    if (source?.kind === 'dataset') return { width: source.img.width, height: source.img.height };
    return null;
  });
  const sourceName = $derived(source ? (source.kind === 'file' ? source.name : `${source.ds.name} / ${source.img.file}`) : '');
  const shortName = $derived(source ? (source.kind === 'file' ? source.name : (source.img.file.split('/').pop() ?? source.img.file)) : '');
  const truthBoxes = $derived(truth && source?.kind === 'dataset' ? source.img.boxes : []);

  const json = $derived(
    pred && result
      ? {
          model_id: pred.model_id,
          image: pred.image,
          boxes: kept.map((b) => ({ class: b.class, name: b.name, score: b.score, x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2 })),
          timing: pred.timing,
          device: pred.device,
          filter: { conf, iou, max_det: maxDet },
        }
      : null,
  );

  function toggleClass(c: number) {
    const next = new Set(hidden);
    if (next.has(c)) next.delete(c);
    else next.add(c);
    hidden = next;
  }

  function resetThresholds() {
    conf = DEFAULTS.conf;
    iou = DEFAULTS.iou;
    maxDet = DEFAULTS.maxDet;
  }
  const isDefault = $derived(conf === DEFAULTS.conf && iou === DEFAULTS.iou && maxDet === DEFAULTS.maxDet);
  const px = (v: number) => Math.round(v);
  /** Scores read the same everywhere here (table, the picked box's line, the label chips on the image): 2 decimals. */
  const score = (v: number) => v.toFixed(2);
</script>

<svelte:window {onpaste} onkeydown={onwindowkey} onpointerdown={onwindowdown} />

<input bind:this={weightsInput} type="file" accept=".folo" class="sr-only" tabindex="-1" aria-hidden="true" onchange={pickWeights} />

<input
  bind:this={fileInput}
  type="file"
  accept="image/*"
  class="sr-only"
  tabindex="-1"
  aria-hidden="true"
  onchange={(e) => {
    const f = e.currentTarget.files?.[0];
    if (f) void setFile(f);
    e.currentTarget.value = '';
  }}
/>

{#snippet toggle(label: string, on: boolean, set: (v: boolean) => void)}
  <button type="button" class="tog" class:on aria-pressed={on} onclick={() => set(!on)}>
    <span class="tog-box" aria-hidden="true">{#if on}<Check size={10} strokeWidth={3} />{/if}</span>{label}
  </button>
{/snippet}

{#snippet displayToggles()}
  {@render toggle('Labels', labels, (v) => (labels = v))}
  {@render toggle('Scores', scores, (v) => (scores = v))}
  {#if source?.kind === 'dataset'}{@render toggle('Ground truth', truth, (v) => (truth = v))}{/if}
{/snippet}

{#snippet secondarySliders(inline: boolean)}
  <div title="Non-max suppression: a lower-scoring box of the same class is dropped when it overlaps a kept one by more than this.">
    <Slider {inline} label="Overlap (IoU)" bind:value={iou} min={0.05} max={0.95} step={0.05} hint="Suppress same-class overlaps above this." />
  </div>
  <Slider {inline} label="Max detections" bind:value={maxDet} min={1} max={300} step={1} format={(v) => String(Math.round(v))} hint="Keep at most this many boxes." />
{/snippet}

{#snippet moreBody()}
  <div class="sheet-body">
    <div class="more-sliders">
      {#if tablet}
        <Slider inline label="Max detections" bind:value={maxDet} min={1} max={300} step={1} format={(v) => String(Math.round(v))} />
      {:else}
        <Slider inline label="Confidence" bind:value={conf} min={0.01} max={0.95} step={0.01} />
        {@render secondarySliders(true)}
      {/if}
    </div>
    <div class="more-toggles">{@render displayToggles()}</div>
  </div>
{/snippet}

{#snippet resetButton()}
  <Button size="sm" variant="ghost" icon={RotateCcw} disabled={isDefault} onclick={resetThresholds}>Reset</Button>
{/snippet}

{#snippet showAllButton()}
  {#if hidden.size}<Button size="sm" variant="ghost" onclick={() => (hidden = new Set())}>Show all</Button>{/if}
{/snippet}

{#snippet classList()}
  {#if !result}
    <p class="hint">Per-class counts, click one to hide it.</p>
  {:else if counts.length === 0}
    <p class="hint">{pred?.boxes.length ? 'Nothing above the confidence threshold.' : 'The model found nothing in this image.'}</p>
  {:else}
    <ul class="counts" aria-label="Detected classes">
      {#each counts as [c, n] (c)}
        {@const off = hidden.has(c)}
        <li>
          <button type="button" class:off aria-pressed={!off} onclick={() => toggleClass(c)} aria-label="{off ? 'Show' : 'Hide'} {classes[c] ?? `class ${c}`}">
            <span class="sw" style:background={classColor(c)} aria-hidden="true"></span>
            <span class="c-name truncate">{classes[c] ?? `class ${c}`}</span>
            <span class="c-bar" aria-hidden="true"><span style:transform="scaleX({n / maxCount})" style:background={classColor(c)}></span></span>
            <span class="c-n num">{n}</span>
          </button>
        </li>
      {/each}
    </ul>
  {/if}
{/snippet}

{#snippet timingBody()}
  {#if pred}
    <TimingBar timing={pred.timing} device={deviceText} />
  {:else if predicting}
    <div class="t-sk"><Skeleton h="28px" w="50%" /><Skeleton h="8px" /><Skeleton h="14px" /><Skeleton h="14px" /></div>
  {:else}
    <p class="hint">Timing appears after a run.</p>
  {/if}
{/snippet}

{#snippet noticeStrip()}
  <!-- What went wrong and the ways out, in one block above the image, where the result would be, at every width: the
     image stays whole, and the failure and its recovery are in the first screen. -->
  {#if notice}
    {#key notice}
      <div class="notice" bind:this={noticeEl}>
        {#if notice === 'file'}
          <div class="need-file" role="alert">
            <FileUp size={16} strokeWidth={1.75} aria-hidden="true" />
            <div class="nf-body">
              {#if badFile}
                <p class="n-title bad">{badFile.why ? `${badFile.name} could not be loaded` : `${badFile.name} is not a .folo weights file`}</p>
                <p class="nf-text">
                  {#if badFile.why}{badFile.why} {/if}Choose {model?.name}'s <span class="mono">weights/best.folo</span> from its run's folder, or run it on the
                  server.
                </p>
              {:else}
                <p class="n-title">This server cannot send model files yet</p>
                <p class="nf-text">
                  To run {model?.name} in the browser, choose its <span class="mono">.folo</span> file (<span class="mono">weights/best.folo</span>
                  in the run's folder), or run it on the server.
                </p>
              {/if}
              <div class="nf-actions">
                <Button size="sm" variant="primary" icon={FileUp} onclick={() => weightsInput.click()}>Choose .folo file</Button>
                <Button size="sm" onclick={() => (runOn = 'server')}>Run on the server</Button>
              </div>
            </div>
          </div>
        {:else}
          <ErrorState
            {error}
            title="Prediction failed"
            retry={() => retryTick++}
            compact
            hint={!oom ? undefined : browserOff ? (smaller ? 'Use a smaller model.' : undefined) : smaller ? 'Run it in this browser, or use a smaller model.' : 'Run it in this browser instead.'}
          >
            {#snippet actions()}
              {#if oom && !browserOff}
                <Button size="sm" variant="primary" onclick={() => (runOn = localBackend)}>Run in this browser</Button>
                {#if smaller}<Button size="sm" title="{smaller.name} is {smaller.arch}: a smaller model of the same classes" onclick={() => chooseModel(smaller.id)}>Use {smaller.name}</Button>{/if}
              {/if}
            {/snippet}
          </ErrorState>
        {/if}
      </div>
    {/key}
  {/if}
{/snippet}

{#snippet intro()}
  <span>Run a model on any image. Thresholds re-filter instantly in the browser.</span>
{/snippet}

<!-- The model and where it runs: on the title row from 1024 px up, a toolbar row under the title below that. -->
{#snippet pickers()}
  {#if ranked.length}
    <div class="model-pick">
      <span class="pick-label">Model</span>
      <Select
        label="Model"
        value={modelId ?? ''}
        options={ranked.map((m) => ({ value: m.id, label: `${m.name}${m.metrics ? ` · mAP ${metric(m.metrics.map50_95)}` : ''}` }))}
        onchange={chooseModel}
      />
      {#if model}<Button icon={ArrowUpRight} href={href(`/models/${model.id}`)} aria-label="Open {model.name}" />{/if}
    </div>
    <div class="runon" role="group" aria-label="Run on" aria-describedby={runOnNote ? 'runon-note' : undefined}>
      <span class="pick-label" aria-hidden="true">Run on</span>
      <!-- A choice that is off says why: on the title row in a line under the control, from its left edge; in the
           toolbar row beside it. -->
      <span class="runon-seg">
        <Segmented
          label="Run the model on"
          size="sm"
          bind:value={runOn}
          options={[
            { value: 'server', label: 'Server' },
            { value: 'webgpu', label: 'WebGPU', disabled: !webgpu || !!browserOff, tip: browserOff ?? 'No WebGPU in this browser' },
            { value: 'cpu', label: 'CPU', disabled: !!browserOff, tip: browserOff ?? undefined },
          ]}
        />
        {#if runOnNote}<span class="runon-note" id="runon-note">{runOnNote}</span>{/if}
      </span>
    </div>
  {/if}
{/snippet}

<Page title="Playground" wide sub={source && phone.current ? undefined : intro} actions={narrow.current || !ranked.length ? undefined : pickers}>
  {#if narrow.current && ranked.length}<div class="pg-tools">{@render pickers()}</div>{/if}

  {#if models.error && !models.data}
    <Panel><ErrorState error={models.error} retry={() => models.refresh()} title="Could not load models" compact /></Panel>
  {:else if models.data && models.data.length === 0}
    <Panel>
      <EmptyState icon={Box} title="No models to run" description="Train a model first; its best weights show up here when the run finishes.">
        {#snippet actions()}<Button icon={Plus} href={href('/runs/new')}>New run</Button>{/snippet}
      </EmptyState>
    </Panel>
  {:else}
    <!-- 8 + 4 columns; in one column (below 1024 px) the canvas takes the image's aspect (see .layout.fit). -->
    <div class="layout" class:fit={!!source && !!dims}>
      <div class="main">
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div class="viewer" class:loaded={!!source} {ondragover} {ondragleave} {ondrop}>
          <Panel pad="none">
            <div class="vbar">
              <span class="seg">
                <Segmented
                  label="Image source"
                  size="sm"
                  bind:value={seg}
                  onchange={onSeg}
                  options={[
                    { value: 'file', label: 'Image', icon: ImageIcon },
                    { value: 'dataset', label: 'Dataset', icon: Images },
                  ]}
                />
              </span>
              {#if source}
                <span class="src-name mono truncate" title={sourceName}>{shortName}</span>
                {#if dims}<span class="src-dims num">{dims.width}×{dims.height}</span>{/if}
                <span class="vbar-gap"></span>
                <Button size="sm" variant="ghost" class="replace" aria-label="Replace image" icon={source.kind === 'dataset' ? Images : Upload} onclick={() => (source?.kind === 'dataset' ? (pickerOpen = true) : fileInput.click())}>Replace</Button>
                <Button size="sm" variant="ghost" class="clear" icon={X} aria-label="Clear image" onclick={clear} />
              {/if}
            </div>
            {@render noticeStrip()}
            <div class="stage" class:empty={!source} style:--ar={dims ? `${dims.width} / ${dims.height}` : undefined}>
              {#if !source}
                <DropZone {dragging} {samples} onbrowse={() => fileInput.click()} onpick={() => (pickerOpen = true)} onsample={(ds, img) => setSource({ kind: 'dataset', ds, img })} />
              {:else if dims}
                <DetectionView
                  {src}
                  width={dims.width}
                  height={dims.height}
                  boxes={pred ? kept : []}
                  truth={truthBoxes}
                  {classes}
                  {labels}
                  {scores}
                  {only}
                  active={activeIdx}
                  selected={selectedIdx}
                  onhover={(i) => (hover = i === null ? null : (kept[i] ?? null))}
                  onselect={pickFromImage}
                  alt={sourceName}
                />
                {#if notice === 'error'}
                  <!-- A failed prediction: the image steps back and says, in its own frame, why it has no boxes (the
                       notice above has the cause and the ways out), so it never reads as "nothing found". -->
                  <div class="failed-veil"><span>No boxes: the prediction failed</span></div>
                {/if}
                {#if predicting}
                  <div class="scan" aria-hidden="true"><span></span></div>
                  <span class="busy" role="status">{runOn === 'server' || hub.api.mock ? `Detecting with ${model?.name}…` : PHASE_TEXT[engine.phase] || 'Starting…'}</span>
                {/if}
                {#if dragging}<div class="drop-hint"><span>Drop to replace</span></div>{/if}
              {/if}
            </div>
            {#if source}
              <!-- Phones: the threshold tuned most sits right under the image; the rest is one tap away below it. Tablets:
                   Confidence and Overlap side by side, the less-used settings one tap away. -->
              <div class="qconf" class:covered={sheetShown && !tablet}>
                <Slider inline label="Confidence" bind:value={conf} min={0.01} max={0.95} step={0.01} />
                {#if tablet}
                  <div class="qiou" title="Non-max suppression: a lower-scoring box of the same class is dropped when it overlaps a kept one by more than this.">
                    <Slider inline label="Overlap (IoU)" bind:value={iou} min={0.05} max={0.95} step={0.05} />
                  </div>
                {/if}
              </div>
            {/if}
            {#if source && result}
              <!-- Phones: the picked detection in one line under the image, so its numbers are readable without the table. -->
              {@const sel = selectedIdx !== null ? kept[selectedIdx] : undefined}
              <p class="pick-line num" class:empty={!sel} aria-live="polite">
                {#if sel}
                  <span class="sw" style:background={classColor(sel.class)} aria-hidden="true"></span>
                  <b>{sel.name}</b> {score(sel.score)}<span class="sep" aria-hidden="true">·</span>{px(sel.x2 - sel.x1)}×{px(sel.y2 - sel.y1)} px
                {/if}
              </p>
            {/if}
            {#if source}
              <div class="vfoot num">
                {#if result}
                  <!-- Each item after the first draws its own separator, so hiding one never leaves a stray dot. -->
                  <span class="fi"><b>{rows.length}</b> shown <span class="wide">of {num(pred?.boxes.length)} candidates</span></span>
                  <span class="fi wide">conf ≥ {conf.toFixed(2)}</span>
                  <span class="fi" class:wide={iou === DEFAULTS.iou}>IoU {iou.toFixed(2)}</span>
                  {#if maxDet < DEFAULTS.maxDet}<span class="fi">max {Math.round(maxDet)}</span>{/if}
                  {#if hidden.size}<span class="fi">{hidden.size} {hidden.size === 1 ? 'class' : 'classes'} hidden</span>{/if}
                  <span class="grow"></span>
                  {#if selectedIdx === null}<span class="tap">Tap a box to inspect</span>{/if}
                  <span class="faint">filtered in {result.ms < 0.01 ? '<0.01' : result.ms.toFixed(2)} ms</span>
                {:else if notice === 'error'}
                  <!-- The notice says what failed; the caption only adds what it does not: which model, where. -->
                  <span class="faint truncate">{model?.name} on {runOn === 'server' ? 'the server' : `this browser · ${BACKEND_NAME[runOn]}`}</span>
                  <span class="grow"></span>
                {:else}
                  <span class="faint">{predicting ? 'Running inference…' : weightsMissing ? 'Waiting for the model file' : 'Waiting for a model'}</span>
                  <span class="grow"></span>
                {/if}
                <span class="more-btn" bind:this={moreBtn}>
                  <Button size="sm" variant="ghost" iconRight={tablet ? ChevronDown : ChevronUp} aria-expanded={sheetShown} aria-controls="pg-more" onclick={() => (sheetShown ? closeSheet() : (moreOpen = true))}>{sheetTitle}</Button>
                </span>
              </div>
              {#if sheetShown && tablet}
                <!-- Tablets: a popover anchored under its button (bg.overlay, shadow-lg), over the panels below,
                     never a bottom sheet clipped at the viewport. Escape returns focus to the button. -->
                <div class="pop" id="pg-more" role="dialog" aria-modal="false" aria-labelledby="pg-more-title" tabindex="-1" bind:this={sheetEl}>
                  <div class="pop-head">
                    <h2 id="pg-more-title">{sheetTitle}</h2>
                    {@render resetButton()}
                  </div>
                  {@render moreBody()}
                </div>
              {:else if sheetShown}
                <!-- Phones. Non-modal: no backdrop, the image above stays visible and keeps re-filtering as the sliders move. -->
                <div
                  class="sheet"
                  id="pg-more"
                  role="dialog"
                  aria-modal="false"
                  aria-labelledby="pg-more-title"
                  tabindex="-1"
                  bind:this={sheetEl}
                >
                  <div class="sheet-head" use:sheetDrag={{ sheet: () => sheetEl, close: closeSheet }}>
                    <SheetHandle />
                    <h2 id="pg-more-title">{sheetTitle}</h2>
                    {@render resetButton()}
                    <Button size="sm" variant="ghost" icon={X} aria-label="Close {sheetTitle.toLowerCase()}" onclick={closeSheet} />
                  </div>
                  {@render moreBody()}
                </div>
              {/if}
            {/if}
          </Panel>
        </div>

        <div class="results">
          <Panel pad="none">
            <div class="tabs-row">
              <Tabs
                label="Result views"
                idPrefix="pg"
                bind:value={tab}
                tabs={[
                  { value: 'detections', label: 'Detections', meta: result ? String(rows.length) : undefined },
                  { value: 'json', label: 'JSON' },
                ]}
              />
            </div>
            <div id="pg-panel" role="tabpanel" aria-labelledby="pg-{tab}">
              {#if !result}
                <p class="placeholder">
                  {predicting ? 'Running inference…' : notice === 'error' ? 'No detections: the last prediction failed.' : 'Detections appear here once you run the model on an image.'}
                </p>
              {:else if tab === 'json'}
                <div class="json-wrap"><JsonView value={json} label="Prediction as JSON" /></div>
              {:else if !pred?.boxes.length}
                <!-- Candidates are fetched down to the lowest threshold (FETCH): none means no threshold would show more. -->
                <p class="placeholder">The model found nothing in this image, even at conf ≥ {FETCH.conf}.</p>
              {:else if rows.length === 0}
                <p class="placeholder">
                  No detections at conf ≥ {conf.toFixed(2)}{hidden.size ? ' in the visible classes' : ''}.
                  {#if conf > 0.05}<button type="button" class="inline" onclick={() => (conf = Math.max(0.01, +(conf / 2).toFixed(2)))}>Lower the threshold</button>{/if}
                </p>
              {:else}
                <div class="scroll">
                  <table class="table dense">
                    <thead>
                      <tr>
                        <th scope="col" class="r idx">#</th>
                        <th scope="col">Class</th>
                        <th scope="col">Score</th>
                        <th scope="col" class="r hide-sm coord" title="Box left edge, in image pixels">x1</th>
                        <th scope="col" class="r hide-sm coord" title="Box top edge, in image pixels">y1</th>
                        <th scope="col" class="r hide-sm coord" title="Box right edge, in image pixels">x2</th>
                        <th scope="col" class="r hide-sm coord" title="Box bottom edge, in image pixels">y2</th>
                        <th scope="col" class="r">Size (px)</th>
                      </tr>
                    </thead>
                    <tbody bind:this={tbody}>
                      {#each rows as { b, i }, k (i)}
                        <tr
                          data-det={i}
                          tabindex={i === tabRow ? 0 : -1}
                          class:hot={activeIdx === i}
                          class:on={selectedIdx === i}
                          aria-selected={selectedIdx === i ? 'true' : undefined}
                          aria-label="{b.name}, score {score(b.score)}, {px(b.x2 - b.x1)} by {px(b.y2 - b.y1)} pixels"
                          onpointerenter={(e) => {
                            // A tap is not a hover: on touch the selection alone marks the row.
                            if (e.pointerType !== 'touch') hover = b;
                          }}
                          onpointerleave={() => (hover = null)}
                          onfocus={() => (picked = b)}
                          onclick={() => (picked = b)}
                          onkeydown={(e) => onrowkey(e, k)}
                        >
                          <td class="r idx faint">{i + 1}</td>
                          <td class="lead">
                            <span class="cls">
                              <span class="sw" style:background={classColor(b.class)} aria-hidden="true"></span>
                              <span class="truncate">{b.name}</span>
                            </span>
                          </td>
                          <td>
                            <span class="score">
                              <span class="s-track" aria-hidden="true"><span style:transform="scaleX({b.score})"></span></span>
                              <span class="num">{score(b.score)}</span>
                            </span>
                          </td>
                          <td class="r muted hide-sm coord">{px(b.x1)}</td>
                          <td class="r muted hide-sm coord">{px(b.y1)}</td>
                          <td class="r muted hide-sm coord">{px(b.x2)}</td>
                          <td class="r muted hide-sm coord">{px(b.y2)}</td>
                          <td class="r muted">{px(b.x2 - b.x1)}×{px(b.y2 - b.y1)}</td>
                        </tr>
                      {/each}
                    </tbody>
                  </table>
                </div>
              {/if}
            </div>
          </Panel>
        </div>

        {#if source && phone.current}
          <!-- Phones: classes and timing as two disclosures under the detections, folded until asked for. -->
          <div class="discs">
            <details class="disc">
              <summary>
                <ChevronRight size={14} aria-hidden="true" />
                <span class="disc-title">Classes</span>
                {#if result}<span class="disc-meta num">{counts.length} found{hidden.size ? ` · ${hidden.size} hidden` : ''}</span>{/if}
              </summary>
              <div class="disc-body">
                {#if hidden.size}<div class="disc-act">{@render showAllButton()}</div>{/if}
                {@render classList()}
              </div>
            </details>
            <details class="disc">
              <summary>
                <ChevronRight size={14} aria-hidden="true" />
                <span class="disc-title">Timing</span>
                {#if pred}<span class="disc-meta num">{ms(pred.timing.total_ms)} · {where}</span>{/if}
              </summary>
              <div class="disc-body">{@render timingBody()}</div>
            </details>
          </div>
        {/if}
      </div>

      <!-- Phones with an image: Thresholds open as a sheet from the image's footer, the rest folds under the detections. -->
      <div class="aside" class:tucked={!!source}>
        <div class="thresholds">
          <Panel title="Thresholds" divided>
            {#snippet actions()}{@render resetButton()}{/snippet}
            <div class="sliders">
              <Slider label="Confidence" bind:value={conf} min={0.01} max={0.95} step={0.01} hint="Hide boxes scored below this." />
              {@render secondarySliders(false)}
            </div>
            <!-- What is drawn on the image: here rather than on the image's toolbar, which is only as wide as the image
                 (on phones the same toggles sit in the Thresholds sheet). -->
            {#if source}<div class="show" role="group" aria-label="Show on the image">{@render displayToggles()}</div>{/if}
          </Panel>
        </div>

        <div class="classes">
          <Panel title="Classes" meta={result ? `${counts.length} found` : undefined} divided>
            {#snippet actions()}{@render showAllButton()}{/snippet}
            {@render classList()}
          </Panel>
        </div>

        <div class="timing">
          <Panel title="Timing" meta={pred ? where : undefined} divided>
            {@render timingBody()}
          </Panel>
        </div>
      </div>
    </div>
  {/if}
</Page>


<DatasetPicker bind:open={pickerOpen} prefer={classes} onpick={(ds, img) => setSource({ kind: 'dataset', ds, img })} onclose={() => (seg = source?.kind ?? 'file')} />

<style>
  .model-pick {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 400px;
  }
  /* Below 1024 px: one full-width toolbar row under the title; the model select takes what Run on leaves, and the
     WebGPU note sits beside the control (wrapping under it when the row is full). */
  .pg-tools {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2) var(--space-6);
  }
  .pg-tools .model-pick {
    flex: 1 1 320px;
    width: auto;
    min-width: 0;
  }
  .pg-tools .runon {
    flex-wrap: wrap;
    row-gap: var(--space-1);
  }
  .pg-tools .runon-seg {
    display: contents;
  }
  .pg-tools .runon-note {
    position: static;
  }
  .model-pick :global(.select) {
    flex: 1;
  }
  .runon {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .pick-label {
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  /* Under the segmented control, out of the row's flow (the title row keeps its height); as wide as the control. */
  .runon-seg {
    position: relative;
    display: flex;
  }
  .runon-note {
    position: absolute;
    top: calc(100% + var(--space-1));
    left: 0;
    right: 0;
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  /* The page grid's 8 + 4 split (the rail is always 4 columns), the same with or without an image. */
  .layout {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    gap: var(--space-4);
    align-items: start;
    --stage-h: max(360px, calc(100dvh - 220px));
  }
  .main,
  .aside {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: var(--space-4);
    min-width: 0;
  }
  .vbar {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-height: 44px;
    padding: var(--space-2) var(--space-2) var(--space-2) var(--space-3);
    border-bottom: 1px solid var(--line-subtle);
  }
  .seg {
    flex-shrink: 0;
    display: flex;
  }
  .src-name {
    min-width: 0;
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
  }
  .src-dims {
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
    flex-shrink: 0;
  }
  .vbar-gap {
    flex: 1;
  }
  .show {
    display: flex;
    flex-wrap: wrap;
    gap: 2px;
    margin-top: var(--space-4);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
  }
  /* The first toggle's box lines up with the slider labels above. */
  .show .tog:first-child {
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
  /* The canvas takes the viewport minus the chrome above and below it, empty or not; an image is shown whole in it
     (contained, centred on the inset well). */
  .stage {
    position: relative;
    height: var(--stage-h);
    background: var(--bg-inset);
  }
  /* Empty, the canvas is as tall as its content (the drop target, 280–480 px, and the samples; DropZone), never held
     at the viewport's height: a sparse state gets a compact composition (DESIGN_DIRECTION 8). */
  .stage.empty {
    height: auto;
    display: flex;
    flex-direction: column;
    border-radius: 0 0 calc(var(--radius-lg) - 1px) calc(var(--radius-lg) - 1px);
    overflow: hidden;
  }
  .scan {
    position: absolute;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
  }
  .scan span {
    position: absolute;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--accent-line);
    box-shadow: 0 0 12px var(--accent-line);
    opacity: 0.7;
    animation: scan 1.2s var(--ease-standard) infinite alternate;
  }
  @keyframes scan {
    from {
      top: 0;
    }
    to {
      top: calc(100% - 2px);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .scan span {
      display: none;
    }
  }
  .busy {
    position: absolute;
    top: var(--space-3);
    left: var(--space-3);
    padding: 2px 8px;
    border-radius: var(--radius-sm);
    background: var(--bg-overlay);
    box-shadow: var(--shadow-md);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  /* A strip between the toolbar and the image: the image stays whole, the notice reads on the panel, not a photo. */
  .notice {
    padding: 0 var(--space-4);
    border-bottom: 1px solid var(--line-subtle);
  }
  .need-file {
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
    padding: var(--space-3) 0;
  }
  .need-file > :global(svg) {
    flex-shrink: 0;
    margin-top: 2px;
    color: var(--text-tertiary);
  }
  .nf-body {
    display: grid;
    gap: 2px;
    min-width: 0;
  }
  .n-title {
    font-weight: var(--weight-medium);
  }
  .n-title.bad {
    color: var(--danger-text);
  }
  .n-title:focus,
  .notice :global(.title:focus) {
    outline: none;
  }
  .nf-text {
    color: var(--text-secondary);
  }
  .nf-actions {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    margin-top: var(--space-2);
  }
  .failed-veil {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    padding: var(--space-4);
    background: color-mix(in srgb, var(--bg-inset) 72%, transparent);
    pointer-events: none;
  }
  .failed-veil span {
    padding: var(--space-1) var(--space-3);
    border-radius: var(--radius-sm);
    background: var(--bg-overlay);
    box-shadow: var(--shadow-md);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    text-align: center;
  }
  .drop-hint {
    position: absolute;
    inset: var(--space-3);
    display: grid;
    place-items: center;
    border-radius: var(--radius-md);
    border: 1px dashed var(--accent-line);
    background: color-mix(in srgb, var(--bg-inset) 70%, transparent);
    font-weight: var(--weight-medium);
  }
  .vfoot {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: 32px;
    padding: 0 var(--space-3);
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
  }
  .vfoot b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .vfoot .truncate {
    min-width: 0;
  }
  .fi + .fi::before {
    content: '·';
    margin-right: var(--space-2);
    color: var(--text-tertiary);
  }
  .qconf,
  .tap {
    display: none;
  }
  .qconf :global(.slider) {
    flex: 1;
    min-width: 0;
  }
  .grow {
    flex: 1;
  }
  .tabs-row {
    padding: 0 var(--space-4);
  }
  /* A state in the frame (DESIGN_DIRECTION 8): one line with the panel's 16 px padding, on its left edge, never a
     centred block held open. */
  .placeholder {
    padding: var(--space-4);
    color: var(--text-tertiary);
  }

  /* An inline action is a link: text.primary, underlined on hover, never lime (DESIGN_DIRECTION §2.6). */
  .inline {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .inline:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
  }
  .json-wrap {
    padding: var(--space-3);
  }
  .scroll {
    overflow: auto;
    max-height: 420px;
  }
  .scroll thead th {
    position: sticky;
    top: 0;
    background: var(--bg-panel);
    z-index: 1;
  }
  .idx {
    width: 40px;
  }
  /* x1 y1 x2 y2 read as one group: tighter gaps between the four coordinate columns. */
  .table .coord + .coord {
    padding-left: var(--space-2);
  }
  .table .coord {
    font-variant-numeric: tabular-nums;
  }
  tbody tr {
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  tbody tr {
    cursor: pointer;
  }
  /* Hover (the pointer, or the box under it on the image) is a light step, only where a pointer can hover, and never
     on the selected row, which keeps the selection's fill (base.css .on): one highlight per meaning. */
  @media (hover: hover) {
    tbody tr:hover:not(.on),
    tbody tr.hot:not(.on) {
      background: var(--bg-hover);
    }
  }
  .cls {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    max-width: 180px;
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  /* Class marks keep the palette hue in both themes; a hairline edge holds pale hues on light surfaces. */
  :global([data-theme='light']) .sw,
  :global([data-theme='light']) .c-bar span {
    box-shadow: inset 0 0 0 1px #0a0e1433;
  }
  .score {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .s-track,
  .c-bar {
    position: relative;
    width: 72px;
    height: 4px;
    border-radius: var(--radius-full);
    background: var(--bg-active);
    overflow: hidden;
  }
  .s-track span,
  .c-bar span {
    position: absolute;
    inset: 0;
    transform-origin: left;
    border-radius: inherit;
  }
  /* The table's one inline bar is neutral ink (the hue is on the swatch); the hovered or selected row's in primary. */
  .s-track span {
    background: var(--text-tertiary);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  tbody tr:hover .s-track span,
  tbody tr.hot .s-track span,
  tbody tr.on .s-track span {
    background: var(--text-primary);
  }
  .sliders {
    display: grid;
    gap: var(--space-5);
  }
  /* Threshold values are numbers, not code: sans with tabular numerals. */
  .sliders :global(output.val),
  .qconf :global(output.val) {
    font-family: var(--font-sans);
    font-variant-numeric: tabular-nums;
  }
  .pick-line {
    display: none;
  }
  .hint {
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .counts {
    display: grid;
    margin: 0 calc(-1 * var(--space-2));
  }
  .counts button {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 100%;
    height: 28px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      opacity var(--dur-fast) var(--ease-standard);
  }
  .counts button:hover {
    background: var(--bg-hover);
  }
  .counts button.off {
    opacity: 0.4;
  }
  .counts button.off .c-name {
    text-decoration: line-through;
  }
  .counts button:focus-visible {
    outline-offset: -2px;
  }
  .c-name {
    flex: 1;
    min-width: 0;
    text-align: left;
  }
  .c-bar {
    width: 64px;
  }
  .c-n {
    min-width: 24px;
    text-align: right;
    font-size: var(--type-sm-size);
  }
  .t-sk {
    display: grid;
    gap: var(--space-3);
  }
  /* Phones: the threshold sheet. Slider labels share one width so the tracks line up with Confidence. */
  .sheet {
    position: fixed;
    left: 0;
    right: 0;
    bottom: calc(var(--tabbar-h) + env(safe-area-inset-bottom));
    z-index: var(--z-overlay);
    display: flex;
    flex-direction: column;
    max-height: 45dvh;
    background: var(--bg-overlay);
    border-top: 1px solid var(--line-default);
    border-radius: var(--radius-xl) var(--radius-xl) 0 0;
    box-shadow: var(--shadow-lg);
    animation: sheet-in var(--dur-slow) var(--ease-out);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .sheet:focus {
    outline: none;
  }
  @keyframes sheet-in {
    from {
      transform: translateY(100%);
    }
  }
  .sheet-head {
    position: relative;
    display: flex;
    align-items: center;
    gap: var(--space-1);
    min-height: 48px;
    padding: var(--space-2) var(--space-2) 0 var(--space-4);
    touch-action: none;
  }
  .sheet-head h2 {
    flex: 1;
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    letter-spacing: var(--type-md-tracking);
    font-weight: var(--weight-medium);
  }
  .sheet-body {
    overflow-y: auto;
    overscroll-behavior: contain;
    padding: 0 var(--space-4) var(--space-3);
  }
  .discs {
    display: grid;
    gap: var(--space-2);
    order: 3;
  }
  .disc {
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
  }
  .disc summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-height: 44px;
    padding: 0 var(--space-4);
    list-style: none;
    cursor: pointer;
    border-radius: inherit;
  }
  .disc summary::-webkit-details-marker {
    display: none;
  }
  .disc summary :global(svg) {
    flex-shrink: 0;
    color: var(--text-tertiary);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .disc[open] summary :global(svg) {
    transform: rotate(90deg);
  }
  .disc summary:focus-visible {
    outline-offset: -2px;
  }
  .disc-title {
    font-weight: var(--weight-medium);
  }
  .disc-meta {
    margin-left: auto;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .disc-body {
    padding: 0 var(--space-4) var(--space-4);
  }
  .disc-act {
    display: flex;
    justify-content: flex-end;
    margin-bottom: var(--space-1);
  }
  /* One label column and one value column: every track (the pinned Confidence, the sheet's three) starts and ends at
     the same x. */
  .qconf :global(.slider.inline),
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
  .more-btn {
    display: none;
    flex-shrink: 0;
  }
  .more-btn :global(svg) {
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .more-btn :global([aria-expanded='true'] svg) {
    transform: rotate(180deg);
  }
  @media (pointer: coarse) {
    .table.dense td {
      height: 44px;
    }
    .counts button,
    .more-toggles .tog {
      height: 44px;
    }
  }
  @media (max-width: 1023px) {
    .layout {
      grid-template-columns: minmax(0, 1fr);
    }
    /* One column: the canvas takes the image's aspect at the full width, up to the canvas height, so the image fills
       it edge to edge; empty, it is as tall as its content (the drop target and the samples, DropZone), never held at
       the viewport's height: a sparse state gets a compact composition, so the samples sit in the first screen. */
    .layout.fit .stage {
      height: auto;
      aspect-ratio: var(--ar);
      max-height: var(--stage-h);
    }
    .stage.empty {
      height: auto;
      min-height: 0;
    }
    /* The toolbar row takes the title row's place for the controls; no extra gap under the title. */
    .pg-tools {
      margin-top: calc(-1 * var(--space-1));
    }
    /* One column: interleave the panels so the controls sit right under the image. */
    .main,
    .aside {
      display: contents;
    }
    .viewer {
      order: 1;
    }
    .thresholds {
      order: 2;
    }
    .results {
      order: 3;
    }
    .classes {
      order: 4;
    }
    .timing {
      order: 5;
    }
  }
  /* Tablets: the rail is a two-column block under the image, Thresholds beside Classes over Timing, then the
     detections full width. */
  @media (min-width: 768px) and (max-width: 1023px) {
    .aside {
      order: 2;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      grid-template-rows: auto 1fr;
      gap: var(--space-4);
    }
    .thresholds {
      grid-row: span 2;
    }
    .thresholds {
      display: grid;
    }
    /* Content-sized: Timing sits right under Classes and is as tall as what it holds (one line before a run), never
       stretched to Thresholds' height. */
    .timing {
      align-self: start;
    }
    /* With an image: the canvas at most half the viewport (a square photo letterboxed on bg.inset), Confidence and
       Overlap in one strip right under it, the rest in the sheet; the Thresholds panel steps out and Classes sits
       beside Timing. */
    .layout.fit {
      --stage-h: 50dvh;
    }
    .aside.tucked .thresholds {
      display: none;
    }
    .aside.tucked {
      grid-template-rows: auto;
    }
    /* Side by side, the two panels end together (DESIGN_DIRECTION 2). */
    .aside.tucked .classes,
    .aside.tucked .timing {
      display: grid;
      align-self: stretch;
    }
    .qconf {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: var(--space-6);
      align-items: center;
      min-height: 48px;
      padding: 0 var(--space-4);
      border-top: 1px solid var(--line-subtle);
    }
    .qconf :global(.slider.inline) {
      grid-template-columns: auto minmax(0, 1fr) 40px;
    }
    .qconf :global(output.val) {
      font-family: var(--font-sans);
      font-variant-numeric: tabular-nums;
    }
    .more-btn {
      display: inline-flex;
    }
    .vfoot .wide {
      display: none;
    }
    /* The settings the strip leaves out: a popover anchored under its button, its right edge on the button's, over
       the panels below (never the image or the strip). Transient settings are popovers above 767 px (DESIGN_DIRECTION
       9): no handle, no scrim, no edge it can be clipped at. */
    .viewer {
      position: relative;
    }
    .pop {
      position: absolute;
      top: calc(100% + 6px);
      right: var(--space-2);
      z-index: var(--z-overlay);
      width: 360px;
      background: var(--bg-overlay);
      border: 1px solid var(--line-default);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow-lg);
      color: var(--text-primary);
      font-size: var(--type-base-size);
      white-space: normal;
      animation: pop-in var(--dur-fast) var(--ease-out);
    }
    .pop:focus {
      outline: none;
    }
    .pop-head {
      display: flex;
      align-items: center;
      gap: var(--space-1);
      min-height: 40px;
      padding: var(--space-1) var(--space-2) 0 var(--space-4);
    }
    .pop-head h2 {
      flex: 1;
      font-size: var(--type-md-size);
      line-height: var(--type-md-line);
      font-weight: var(--weight-medium);
    }
    @keyframes pop-in {
      from {
        opacity: 0;
        transform: translateY(-2px);
      }
    }
  }
  @media (max-width: 767px) {
    /* Phones: two rows, the model, then Run on with its control on the right; the note a row of its own under them. */
    .pg-tools {
      flex-direction: column;
      align-items: stretch;
      gap: var(--space-2);
      margin-top: 0;
    }
    .pg-tools .model-pick {
      flex: none;
    }
    .pg-tools .runon {
      justify-content: space-between;
    }
    .pg-tools .runon-note {
      flex-basis: 100%;
    }
    /* The ways out: the primary full width, the others in one row of secondary buttons under it, 44 px each. */
    .notice :global(.acts),
    .nf-actions {
      gap: var(--space-2);
    }
    .notice :global(.acts .btn),
    .nf-actions :global(.btn) {
      --h: 44px;
      flex: 1 1 auto;
    }
    .notice :global(.acts .btn:first-child),
    .nf-actions :global(.btn:first-child) {
      flex-basis: 100%;
    }
    /* The image goes edge to edge; its bars keep the 16 px gutter inside. */
    .viewer.loaded {
      margin: 0 calc(-1 * var(--space-4));
    }
    .viewer.loaded :global(.panel) {
      border-left: 0;
      border-right: 0;
      border-radius: 0;
    }
    .viewer.loaded .vbar,
    .viewer.loaded .vfoot,
    .viewer.loaded .notice,
    .viewer.loaded .pick-line,
    .viewer.loaded .qconf {
      padding-left: var(--space-4);
      padding-right: var(--space-4);
    }
    .viewer.loaded .vbar {
      padding-right: var(--space-2);
      gap: var(--space-2);
    }
    .viewer.loaded .vfoot {
      height: auto;
      min-height: 40px;
      padding-right: var(--space-2);
    }
    .more-btn {
      display: inline-flex;
    }
    /* The open sheet covers the lower part of the screen: leave room to scroll the detections above it. */
    .layout:has(.sheet) {
      padding-bottom: 45dvh;
    }
    .aside.tucked,
    .vbar .src-dims {
      display: none;
    }
    .vbar .src-name {
      display: block;
      flex: 1;
    }
    .vbar-gap {
      display: none;
    }
    .vbar :global(.replace .label) {
      display: none;
    }
    .vbar :global(.replace) {
      width: var(--control-sm);
      padding: 0;
    }
    .vfoot .wide {
      display: none;
    }
    .tap {
      display: inline;
      color: var(--text-tertiary);
    }
    .pick-line {
      display: flex;
      align-items: center;
      gap: 6px;
      min-height: 32px;
      padding: 0 var(--space-3);
      border-top: 1px solid var(--line-subtle);
      font-size: var(--type-sm-size);
      color: var(--text-secondary);
      white-space: nowrap;
      overflow: hidden;
    }
    /* Nothing picked: the live region stays in the tree but takes no room. */
    .pick-line.empty {
      min-height: 0;
      border-top: 0;
    }
    .pick-line b {
      color: var(--text-primary);
      font-weight: var(--weight-medium);
    }
    .pick-line .sep {
      color: var(--text-tertiary);
    }
    /* The open sheet holds Confidence as its first row; the pinned copy under it keeps its room but steps out. */
    .qconf.covered {
      visibility: hidden;
    }
    .qconf {
      display: flex;
      align-items: center;
      gap: var(--space-3);
      min-height: 48px;
      padding: 0 var(--space-3);
      border-top: 1px solid var(--line-subtle);
    }
    .stage,
    .layout.fit .stage {
      height: auto;
      aspect-ratio: var(--ar, 4 / 3);
      max-height: 70vh;
    }
    .stage.empty {
      aspect-ratio: auto;
      max-height: none;
      min-height: 0;
    }
    .vfoot .faint {
      display: none;
    }
  }
</style>
