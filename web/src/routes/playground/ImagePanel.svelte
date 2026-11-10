<script lang="ts">
  /**
   * The image and its boxes: where it came from (the toolbar), what stands between it and its boxes (the notice),
   * the canvas, the thresholds pinned under it on phones and tablets, and its footer with the settings sheet (phones)
   * or popover (tablets).
   */
  import type { Snippet } from 'svelte';
  import FileUp from '@lucide/svelte/icons/file-up';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import ChevronUp from '@lucide/svelte/icons/chevron-up';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import ImageIcon from '@lucide/svelte/icons/image';
  import Images from '@lucide/svelte/icons/images';
  import Upload from '@lucide/svelte/icons/upload';
  import X from '@lucide/svelte/icons/x';
  import { hub } from '$lib/api/index.svelte';
  import type { Dataset, DatasetImage as Img, Model, Prediction } from '$lib/api/types';
  import { classColor } from '$lib/design/tokens';
  import DetectionView from '$lib/detect/DetectionView.svelte';
  import { PHASE_TEXT, type Backend, type BrowserEngine } from '$lib/inference/browser.svelte';
  import { num } from '$lib/format';
  import SheetHandle from '$lib/ui/SheetHandle.svelte';
  import { sheetDrag } from '$lib/ui/sheetDrag';
  import Button from '$lib/ui/Button.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Slider from '$lib/ui/Slider.svelte';
  import DropZone from './DropZone.svelte';
  import { BACKEND_NAME, DEFAULTS, px, score } from './predict';
  import type { Source } from './source';
  import Swatch from './Swatch.svelte';

  type Det = Prediction['boxes'][number];

  let {
    notice,
    noticeEl = $bindable(),
    badFile,
    model,
    chooseWeights,
    browse,
    runOn = $bindable(),
    error,
    retryTick = $bindable(),
    oom,
    browserOff,
    smaller,
    localBackend,
    chooseModel,
    ondragover,
    ondragleave,
    ondrop,
    source,
    seg = $bindable(),
    onSeg,
    runOnNote,
    sourceName,
    shortName,
    dims,
    pickerOpen = $bindable(),
    clear,
    dragging,
    samples,
    setSource,
    src,
    pred,
    kept,
    truthBoxes,
    classes,
    labels,
    scores,
    only,
    activeIdx,
    selectedIdx,
    hover = $bindable(),
    pickFromImage,
    predicting,
    engine,
    sheetShown,
    tablet,
    conf = $bindable(),
    iou = $bindable(),
    result,
    rows,
    hidden,
    maxDet,
    weightsMissing,
    moreBtn = $bindable(),
    closeSheet,
    moreOpen = $bindable(),
    sheetTitle,
    sheetEl = $bindable(),
    resetButton,
    moreBody,
  }: {
    notice: 'file' | 'error' | null;
    noticeEl: HTMLElement | undefined;
    badFile: { name: string; why: string | null } | null;
    model: Model | null;
    /** Asks for the model's .folo file (the page's hidden file input). */
    chooseWeights: () => void;
    /** Asks for an image file (the page's hidden file input). */
    browse: () => void;
    runOn: 'server' | Backend;
    error: unknown;
    retryTick: number;
    oom: boolean;
    browserOff: string | null;
    smaller: Model | null;
    localBackend: Backend;
    chooseModel: (id: string) => void;
    ondragover: (e: DragEvent) => void;
    ondragleave: (e: DragEvent) => void;
    ondrop: (e: DragEvent) => void;
    source: Source | null;
    seg: 'file' | 'dataset' | 'webcam';
    onSeg: (v: 'file' | 'dataset' | 'webcam') => void;
    runOnNote: string | null;
    sourceName: string;
    shortName: string;
    dims: { width: number; height: number } | null;
    pickerOpen: boolean;
    clear: () => void;
    dragging: boolean;
    samples: { ds: Dataset; img: Img }[] | undefined;
    setSource: (s: Source) => void;
    src: string | null;
    pred: Prediction | null;
    kept: Det[];
    truthBoxes: Img['boxes'];
    classes: string[];
    labels: boolean;
    scores: boolean;
    only: Set<number> | null;
    activeIdx: number | null;
    selectedIdx: number | null;
    hover: Det | null;
    pickFromImage: (i: number) => void;
    predicting: boolean;
    engine: BrowserEngine;
    sheetShown: boolean;
    tablet: boolean;
    conf: number;
    iou: number;
    result: { boxes: Det[]; ms: number } | null;
    rows: { b: Det; i: number }[];
    hidden: Set<number>;
    maxDet: number;
    weightsMissing: boolean;
    moreBtn: HTMLElement | undefined;
    closeSheet: () => void;
    moreOpen: boolean;
    sheetTitle: string;
    sheetEl: HTMLElement | undefined;
    /** Reset of the thresholds, as the Thresholds panel's. */
    resetButton: Snippet;
    /** The sheet's and the popover's settings. */
    moreBody: Snippet;
  } = $props();
</script>

{#snippet noticeStrip()}
  <!-- What went wrong and the ways out, in one block above the image, where the result would be, at every width: the
     image stays whole, and the failure and its recovery are in the first screen. -->
  {#if notice}
    {#key notice}
      <div class="notice cta-row" bind:this={noticeEl}>
        {#if notice === 'file'}
          <div class="need-file" class:bad={!!badFile} role="alert">
            <!-- A file that failed says so through its icon; the title stays ink (DESIGN_DIRECTION §2.6). -->
            {#if badFile}<TriangleAlert size={16} strokeWidth={1.75} aria-hidden="true" />{:else}<FileUp size={16} strokeWidth={1.75} aria-hidden="true" />{/if}
            <div class="nf-body">
              {#if badFile}
                <p class="n-title">{badFile.why ? `${badFile.name} could not be loaded` : `${badFile.name} is not a .folo weights file`}</p>
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
              <div class="nf-actions cta-stack">
                <Button size="sm" variant="primary" icon={FileUp} onclick={chooseWeights}>Choose .folo file</Button>
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
      {#if runOnNote && !source}<span class="vbar-note" aria-hidden="true">{runOnNote}</span>{/if}
      {#if source}
        <span class="src-name mono truncate" title={sourceName}>{shortName}</span>
        {#if dims}<span class="src-dims num">{dims.width}×{dims.height}</span>{/if}
        <span class="vbar-gap"></span>
        <Button size="sm" variant="ghost" class="replace" aria-label="Replace image" icon={source.kind === 'dataset' ? Images : Upload} onclick={() => (source?.kind === 'dataset' ? (pickerOpen = true) : browse())}>Replace</Button>
        <Button size="sm" variant="ghost" class="clear" icon={X} aria-label="Clear image" onclick={clear} />
      {/if}
    </div>
    {@render noticeStrip()}
    <div class="stage" class:empty={!source} style:--ar={dims ? `${dims.width} / ${dims.height}` : undefined}>
      {#if !source}
        <DropZone {dragging} {samples} onbrowse={() => browse()} onpick={() => (pickerOpen = true)} onsample={(ds, img) => setSource({ kind: 'dataset', ds, img })} />
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
          <Swatch color={classColor(sel.class)} />
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

<style>
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
  /* Why a Run on choice is off, said once on the image panel's status line, at its right end, before an image is in
     (its tip says it at the control). */
  /* It yields to the control beside it: on a phone under a coarse pointer the 44 px segments leave it less than a line,
     so it wraps (right-aligned, at a space) rather than run past the panel's edge. */
  .vbar-note {
    min-width: 0;
    margin-left: auto;
    padding-right: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    text-align: right;
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
  .need-file.bad > :global(svg) {
    color: var(--danger-text);
  }
  .nf-body {
    display: grid;
    gap: 2px;
    min-width: 0;
  }
  .n-title {
    font-weight: var(--weight-medium);
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
  /* Threshold values are numbers, not code: sans with tabular numerals. */
  .qconf :global(output.val) {
    font-family: var(--font-sans);
    font-variant-numeric: tabular-nums;
  }
  .pick-line {
    display: none;
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
  /* One label column and one value column: every track (the pinned Confidence, the sheet's three) starts and ends at
     the same x. */
  .qconf :global(.slider.inline) {
    grid-template-columns: 112px minmax(0, 1fr) 40px;
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
  @media (max-width: 1023px) {
    /* One column: the canvas takes the image's aspect at the full width, up to the canvas height, so the image fills
       it edge to edge; empty, it is as tall as its content (the drop target and the samples, DropZone), never held at
       the viewport's height: a sparse state gets a compact composition, so the samples sit in the first screen. */
    :global(.layout.fit) .stage {
      height: auto;
      aspect-ratio: var(--ar);
      max-height: var(--stage-h);
      /* Edge to edge, inset by the boxes' halo (lib/detect/DetectionView): a box on the image's edge keeps its outer
         stroke inside the frame, never cut by the viewport or the panel's edge. */
      padding: var(--space-1);
    }
    .stage.empty {
      height: auto;
      min-height: 0;
    }
    .viewer {
      order: 1;
    }
  }
  @media (min-width: 768px) and (max-width: 1023px) {
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
    :global(.layout.fit) .stage {
      height: auto;
      aspect-ratio: var(--ar, 4 / 3);
      max-height: 70vh;
    }
    /* Full bleed: the image inset by the boxes' halo, so a box on its edge keeps its outer stroke on screen. */
    .stage:not(.empty) {
      padding: var(--space-1);
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
