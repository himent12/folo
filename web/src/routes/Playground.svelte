<script lang="ts">
  import ArrowUpRight from '@lucide/svelte/icons/arrow-up-right';
  import Box from '@lucide/svelte/icons/box';
  import Check from '@lucide/svelte/icons/check';
  import ImageIcon from '@lucide/svelte/icons/image';
  import Images from '@lucide/svelte/icons/images';
  import Plus from '@lucide/svelte/icons/plus';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import Upload from '@lucide/svelte/icons/upload';
  import Video from '@lucide/svelte/icons/video';
  import X from '@lucide/svelte/icons/x';
  import { onDestroy, untrack } from 'svelte';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Dataset, DatasetImage as Img, Model, PredictInput, Prediction } from '$lib/api/types';
  import { classColor } from '$lib/design/tokens';
  import DetectionView from '$lib/detect/DetectionView.svelte';
  import { filterDetections } from '$lib/detect/nms';
  import { metric, num } from '$lib/format';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
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
  import JsonView from './playground/JsonView.svelte';
  import TimingBar from './playground/TimingBar.svelte';

  setCrumbs({ label: 'Playground' });

  // The server is asked once with these loose settings; the controls below re-filter its answer in the browser.
  const FETCH = { conf: 0.01, iou: 0.95, max_det: 1000 };
  const DEFAULTS = { conf: 0.25, iou: 0.7, maxDet: 300 };

  // ---------- model ----------
  const models = new Resource((s) => hub.api.models(s));
  const ranked = $derived([...(models.data ?? [])].sort((a, b) => (b.metrics?.map50_95 ?? -1) - (a.metrics?.map50_95 ?? -1)));
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
    if (source?.kind === 'file') URL.revokeObjectURL(source.url);
    source = s;
    seg = s.kind;
    pred = null;
    error = null;
    active = null;
    hidden = new Set();
  }

  async function setFile(file: File | Blob, name = file instanceof File ? file.name : 'pasted image') {
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
      toast.error(new Error('That file could not be decoded. Use a JPEG, PNG or WebP.'), 'Unsupported image');
      return;
    }
    setSource({ kind: 'file', file, name, url: URL.createObjectURL(file), width, height });
  }

  function clear() {
    if (source?.kind === 'file') URL.revokeObjectURL(source.url);
    source = null;
    pred = null;
    error = null;
    seg = 'file';
  }

  onDestroy(() => {
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

  // ---------- prediction ----------
  let pred = $state.raw<Prediction | null>(null);
  let predicting = $state(false);
  let error = $state.raw<unknown>(null);
  let retryTick = $state(0);

  $effect(() => {
    const id = modelId;
    const s = source;
    void retryTick;
    if (!id || !s) return;
    const input: PredictInput = s.kind === 'file' ? { kind: 'file', file: s.file } : { kind: 'dataset', dataset_id: s.ds.id, image_id: s.img.id };
    const ctrl = new AbortController();
    untrack(() => {
      predicting = true;
      error = null;
    });
    hub.api.predict(input, { model_id: id, ...FETCH }, ctrl.signal).then(
      (p) => {
        if (ctrl.signal.aborted) return;
        pred = p;
        predicting = false;
      },
      (e: unknown) => {
        if (ctrl.signal.aborted) return;
        predicting = false;
        error = e;
        toast.error(e, 'Prediction failed');
      },
    );
    return () => ctrl.abort();
  });

  // ---------- client-side filtering ----------
  let conf = $state(DEFAULTS.conf);
  let iou = $state(DEFAULTS.iou);
  let maxDet = $state(DEFAULTS.maxDet);
  let labels = $state(true);
  let scores = $state(true);
  let truth = $state(false);
  let hidden = $state<Set<number>>(new Set());
  let active = $state<number | null>(null);
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
</script>

<svelte:window {onpaste} />

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

<Page title="Playground" wide>
  {#snippet sub()}
    <span>Run a model on any image. Thresholds re-filter instantly in the browser.</span>
  {/snippet}
  {#snippet actions()}
    {#if ranked.length}
      <div class="model-pick">
        <span class="pick-label">Model</span>
        <Select
          label="Model"
          value={modelId ?? ''}
          options={ranked.map((m) => ({ value: m.id, label: `${m.name} · ${m.arch}${m.metrics ? ` · ${metric(m.metrics.map50_95)}` : ''}` }))}
          onchange={chooseModel}
        />
        {#if model}<Button icon={ArrowUpRight} href={href(`/models/${model.id}`)} aria-label="Open {model.name}" />{/if}
      </div>
    {/if}
  {/snippet}

  {#if models.error && !models.data}
    <Panel><ErrorState error={models.error} retry={() => models.refresh()} title="Could not load models" compact /></Panel>
  {:else if models.data && models.data.length === 0}
    <Panel>
      <EmptyState icon={Box} title="No models to run" description="Train a model first; its best weights show up here when the run finishes.">
        {#snippet actions()}<Button variant="primary" icon={Plus} href={href('/runs/new')}>New run</Button>{/snippet}
      </EmptyState>
    </Panel>
  {:else}
    <div class="layout">
      <div class="main">
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div class="viewer" {ondragover} {ondragleave} {ondrop}>
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
                    { value: 'webcam', label: 'Webcam', icon: Video, meta: 'soon', disabled: true },
                  ]}
                />
              </span>
              {#if source}
                <span class="src-name mono truncate" title={sourceName}>{shortName}</span>
                {#if dims}<span class="src-dims mono">{dims.width}×{dims.height}</span>{/if}
                <span class="vbar-gap"></span>
                <span class="toggles">
                  {@render toggle('Labels', labels, (v) => (labels = v))}
                  {@render toggle('Scores', scores, (v) => (scores = v))}
                  {#if source.kind === 'dataset'}{@render toggle('Ground truth', truth, (v) => (truth = v))}{/if}
                </span>
                <Button size="sm" variant="ghost" class="replace" aria-label="Replace image" icon={source.kind === 'dataset' ? Images : Upload} onclick={() => (source?.kind === 'dataset' ? (pickerOpen = true) : fileInput.click())}>Replace</Button>
                <Button size="sm" variant="ghost" class="clear" icon={X} aria-label="Clear image" onclick={clear} />
              {/if}
            </div>
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
                  {active}
                  onhover={(i) => (active = i)}
                  alt={sourceName}
                />
                {#if predicting}
                  <div class="scan" aria-hidden="true"><span></span></div>
                  <span class="busy" role="status">Detecting with {model?.name}…</span>
                {:else if error && !pred}
                  <div class="stage-err"><ErrorState {error} title="Prediction failed" retry={() => retryTick++} compact /></div>
                {/if}
                {#if dragging}<div class="drop-hint"><span>Drop to replace</span></div>{/if}
              {/if}
            </div>
            {#if source}
              <div class="vfoot mono">
                {#if result}
                  <span><b>{kept.length}</b> detections</span>
                  <span class="sep">/</span>
                  <span>{num(pred?.boxes.length)} candidates</span>
                  <span class="sep">·</span>
                  <span class="wide">conf ≥ {conf.toFixed(2)}</span>
                  <span class="wide">IoU {iou.toFixed(2)}</span>
                  <span class="grow"></span>
                  <span class="faint">filtered in {result.ms < 0.01 ? '<0.01' : result.ms.toFixed(2)} ms</span>
                {:else}
                  <span class="faint">{predicting ? 'Running inference…' : 'Waiting for a model'}</span>
                {/if}
              </div>
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
                <p class="placeholder">{predicting ? 'Running inference…' : 'Detections appear here once you run the model on an image.'}</p>
              {:else if tab === 'json'}
                <div class="json-wrap"><JsonView value={json} label="Prediction as JSON" /></div>
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
                        <th scope="col" class="r hide-sm">Box (x1, y1, x2, y2)</th>
                        <th scope="col" class="r">Size</th>
                      </tr>
                    </thead>
                    <tbody>
                      {#each rows as { b, i } (i)}
                        <tr class:hot={active === i} onpointerenter={() => (active = i)} onpointerleave={() => (active = null)}>
                          <td class="r idx faint">{i + 1}</td>
                          <td>
                            <span class="cls">
                              <span class="sw" style:background={classColor(b.class)} aria-hidden="true"></span>
                              <span class="truncate">{b.name}</span>
                            </span>
                          </td>
                          <td>
                            <span class="score">
                              <span class="s-track" aria-hidden="true"><span style:transform="scaleX({b.score})" style:background={classColor(b.class)}></span></span>
                              <span class="mono">{b.score.toFixed(3)}</span>
                            </span>
                          </td>
                          <td class="r mono muted hide-sm">{px(b.x1)}, {px(b.y1)}, {px(b.x2)}, {px(b.y2)}</td>
                          <td class="r mono muted">{px(b.x2 - b.x1)}×{px(b.y2 - b.y1)}</td>
                        </tr>
                      {/each}
                    </tbody>
                  </table>
                </div>
              {/if}
            </div>
          </Panel>
        </div>
      </div>

      <div class="aside">
        <div class="thresholds">
          <Panel title="Thresholds" divided>
            {#snippet actions()}
              <Button size="sm" variant="ghost" icon={RotateCcw} disabled={isDefault} onclick={resetThresholds}>Reset</Button>
            {/snippet}
            <div class="sliders">
              <Slider label="Confidence" bind:value={conf} min={0.01} max={0.95} step={0.01} hint="Hide boxes scored below this." />
              <Slider label="IoU (NMS)" bind:value={iou} min={0.05} max={0.95} step={0.05} hint="Merge same-class boxes overlapping more than this." />
              <Slider label="Max detections" bind:value={maxDet} min={1} max={300} step={1} format={(v) => String(Math.round(v))} />
            </div>
          </Panel>
        </div>

        <div class="classes">
          <Panel title="Classes" meta={result ? `${counts.length} found` : undefined} divided>
            {#snippet actions()}
              {#if hidden.size}<Button size="sm" variant="ghost" onclick={() => (hidden = new Set())}>Show all</Button>{/if}
            {/snippet}
            {#if !result}
              <p class="hint">Per-class counts, click one to hide it.</p>
            {:else if counts.length === 0}
              <p class="hint">Nothing above the confidence threshold.</p>
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
          </Panel>
        </div>

        <div class="timing">
          <Panel title="Timing" meta={pred ? 'server' : undefined} divided>
            {#if pred}
              <TimingBar timing={pred.timing} device={pred.device} />
            {:else if predicting}
              <div class="t-sk"><Skeleton h="28px" w="50%" /><Skeleton h="8px" /><Skeleton h="14px" /><Skeleton h="14px" /></div>
            {:else}
              <p class="hint">Preprocess, inference and postprocess time of the last prediction.</p>
            {/if}
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
  .model-pick :global(.select) {
    flex: 1;
  }
  .pick-label {
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .layout {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 320px;
    gap: var(--space-4);
    align-items: start;
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
    font-size: var(--type-xs-size);
  }
  .src-dims {
    color: var(--text-tertiary);
    font-size: var(--type-xs-size);
    flex-shrink: 0;
  }
  .vbar-gap {
    flex: 1;
  }
  .toggles {
    display: flex;
    gap: 2px;
    flex-shrink: 0;
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
    box-shadow: inset 0 0 0 1px var(--line-strong);
  }
  .tog.on .tog-box {
    background: var(--accent-solid);
    color: var(--accent-on);
    box-shadow: none;
  }
  .stage {
    position: relative;
    height: clamp(380px, 62vh, 660px);
    background: var(--bg-inset);
  }
  .stage.empty {
    border-radius: 0 0 calc(var(--radius-lg) - 1px) calc(var(--radius-lg) - 1px);
    overflow: hidden;
    height: auto;
    min-height: clamp(380px, 62vh, 660px);
    display: flex;
    flex-direction: column;
  }
  .stage.empty > :global(*) {
    flex: 1;
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
    background: var(--accent-solid);
    box-shadow: 0 0 12px var(--accent-solid);
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
    font-size: var(--type-xs-size);
    color: var(--text-secondary);
  }
  .stage-err {
    position: absolute;
    left: var(--space-4);
    right: var(--space-4);
    bottom: var(--space-4);
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
    font-size: var(--type-xs-size);
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
  }
  .vfoot b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .sep {
    color: var(--text-tertiary);
  }
  .grow {
    flex: 1;
  }
  .tabs-row {
    padding: 0 var(--space-4);
  }
  .placeholder {
    padding: var(--space-8) var(--space-4);
    text-align: center;
    color: var(--text-tertiary);
  }
  .inline {
    color: var(--accent-text);
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
  tbody tr {
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  tbody tr:hover,
  tbody tr.hot {
    background: var(--bg-hover);
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
  .sliders {
    display: grid;
    gap: var(--space-5);
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
    font-family: var(--font-mono);
    font-size: var(--type-xs-size);
  }
  .t-sk {
    display: grid;
    gap: var(--space-3);
  }
  @media (max-width: 1199px) {
    .src-name,
    .src-dims {
      display: none;
    }
    .vbar-gap {
      flex: 1;
    }
  }
  @media (max-width: 1023px) {
    .layout {
      grid-template-columns: minmax(0, 1fr);
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
  @media (max-width: 767px) {
    .model-pick {
      width: calc(100vw - 2 * var(--space-4));
    }
    .vbar {
      flex-wrap: wrap;
      row-gap: var(--space-1);
      gap: var(--space-1);
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
    .toggles {
      order: 3;
      flex: 1;
    }
    .vbar :global(.clear) {
      order: 4;
    }
    .stage {
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
    .hide-sm {
      display: none;
    }
  }
</style>
