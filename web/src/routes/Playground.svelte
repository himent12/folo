<script lang="ts">
  import Box from '@lucide/svelte/icons/box';
  import Check from '@lucide/svelte/icons/check';
  import Plus from '@lucide/svelte/icons/plus';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import { onDestroy, untrack } from 'svelte';
  import { MediaQuery } from 'svelte/reactivity';
  import { ApiError } from '$lib/api/client';
  import { hub } from '$lib/api/index.svelte';
  import { MockApi } from '$lib/api/mock';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Dataset, DatasetImage as Img, Model, PredictInput, Prediction, Run } from '$lib/api/types';
  import { modelChoices, scoreContext } from '$lib/scoreSummary';
  import { media } from '$lib/design/tokens';
  import { filterDetections } from '$lib/detect/nms';
  import { BrowserEngine, hasWebGpu, NO_WASM, probeWasm, servedBy, wasmFromSystem, type Backend } from '$lib/inference/browser.svelte';
  import { sentence } from '$lib/format';
  import { isOom } from '$lib/oom';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Button from '$lib/ui/Button.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import Slider from '$lib/ui/Slider.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import DatasetPicker from './playground/DatasetPicker.svelte';
  import { looksLikeFolo } from './playground/folo';
  import { holdNoticeFocus } from './playground/noticeFocus';
  import TimingBar from './playground/TimingBar.svelte';
  import ModelPickers from './playground/ModelPickers.svelte';
  import ImagePanel from './playground/ImagePanel.svelte';
  import ResultsPanel from './playground/ResultsPanel.svelte';
  import ClassList from './playground/ClassList.svelte';
  import PhoneDetails from './playground/PhoneDetails.svelte';
  import SidePanels from './playground/SidePanels.svelte';
  import { BACKEND_NAME, DEFAULTS, FETCH, simulate } from './playground/predict';
  import { imageBlob, type Source } from './playground/source';

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
  // What each model's score is on (lib/scoreSummary): the picker groups models by dataset and split, best first within
  // a group (on a tie the earlier checkpoint), and never ranks one group against another.
  const runs = new Resource((s) => hub.api.runs(s));
  const datasets = new Resource((s) => hub.api.datasets(s));
  const choices = $derived.by(() => {
    // Until the runs and datasets answer (or fail) the splits are unknown: wait, so the default never switches.
    if (!models.data || (!runs.data && !runs.error) || (!datasets.data && !datasets.error)) return [];
    const runById = new Map<string, Run>((runs.data ?? []).map((r) => [r.id, r]));
    const dsById = new Map<string, Dataset>((datasets.data ?? []).map((d) => [d.id, d]));
    return modelChoices(models.data, (m) => scoreContext(m, runById, dsById));
  });
  /** The chosen model's score and what it is on, under the select: "mAP50-95 0.606 · shapes-hard, validation split". */
  const pickedMeta = $derived(choices.find((c) => c.model.id === modelId)?.meta ?? null);
  const ranked = $derived(choices.map((c) => c.model));
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
  /** The server's word on its own build counts for this page only when it served it (the demo's server always does). */
  const fromServer = $derived(wasmFromSystem(system.data, hub.api.mock || servedBy(hub.serverUrl)));
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
    // An end-to-end (o2o) model predicts without NMS (docs/MODEL.md §6.6): an IoU of 1 suppresses nothing, as the server.
    const boxes = filterDetections(pred.boxes, { conf, iou: model?.head === 'o2o' ? 1 : iou, maxDet });
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
  <ClassList {result} {counts} {pred} {hidden} {classes} {maxCount} {toggleClass} />
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

{#snippet intro()}
  <span>Run a model on any image. Thresholds re-filter instantly in the browser.</span>
{/snippet}

<!-- The model and where it runs: on the title row from 1024 px up, a toolbar row under the title below that. -->
{#snippet pickers()}
  <ModelPickers {ranked} {modelId} {choices} {chooseModel} {model} {pickedMeta} bind:runOn {webgpu} {browserOff} {runOnNote} />
{/snippet}

<Page title="Playground" wide sub={source && phone.current ? undefined : intro} actions={narrow.current || !ranked.length ? undefined : pickers}>
  {#if narrow.current && ranked.length}<ModelPickers toolbar {ranked} {modelId} {choices} {chooseModel} {model} {pickedMeta} bind:runOn {webgpu} {browserOff} {runOnNote} />{/if}

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
        <ImagePanel
          {notice}
          bind:noticeEl
          {badFile}
          {model}
          chooseWeights={() => weightsInput.click()}
          browse={() => fileInput.click()}
          bind:runOn
          {error}
          bind:retryTick
          {oom}
          {browserOff}
          {smaller}
          {localBackend}
          {chooseModel}
          {ondragover}
          {ondragleave}
          {ondrop}
          {source}
          bind:seg
          {onSeg}
          {runOnNote}
          {sourceName}
          {shortName}
          {dims}
          bind:pickerOpen
          {clear}
          {dragging}
          {samples}
          {setSource}
          {src}
          {pred}
          {kept}
          {truthBoxes}
          {classes}
          {labels}
          {scores}
          {only}
          {activeIdx}
          {selectedIdx}
          bind:hover
          {pickFromImage}
          {predicting}
          {engine}
          {sheetShown}
          {tablet}
          bind:conf
          bind:iou
          {result}
          {rows}
          {hidden}
          {maxDet}
          {weightsMissing}
          bind:moreBtn
          {closeSheet}
          bind:moreOpen
          {sheetTitle}
          bind:sheetEl
          {resetButton}
          {moreBody}
        />

        <ResultsPanel
          {result}
          bind:tab
          {rows}
          {predicting}
          {notice}
          {json}
          {pred}
          bind:conf
          {hidden}
          bind:tbody
          {tabRow}
          {activeIdx}
          {selectedIdx}
          bind:hover
          bind:picked
          {onrowkey}
        />

        {#if source && phone.current}
          <PhoneDetails {result} {counts} {hidden} {pred} {where} {showAllButton} {classList} {timingBody} />
        {/if}
      </div>

      <SidePanels
        {source}
        {result}
        {counts}
        {pred}
        {where}
        bind:conf
        {resetButton}
        {secondarySliders}
        {displayToggles}
        {showAllButton}
        {classList}
        {timingBody}
      />
    </div>
  {/if}
</Page>


<DatasetPicker bind:open={pickerOpen} prefer={classes} onpick={(ds, img) => setSource({ kind: 'dataset', ds, img })} onclose={() => (seg = source?.kind ?? 'file')} />

<style>
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
</style>
