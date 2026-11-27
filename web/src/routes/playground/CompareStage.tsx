/**
 * Visual A/B comparison viewport for the Playground image canvas.
 * Renders Model A and Model B either side-by-side or in an interactive split-overlay curtain view, both sharing one
 * synchronized `ZoomState` anchored in image space so zooming and panning stay locked across both models and layouts.
 */
import ArrowLeftRight from '$icons/arrow-left-right';
import FileUp from '$icons/file-up';
import { bindable, css, effect, onCleanup, onWindow, signal } from 'fw';
import { hub } from '$lib/api/index';
import type { DatasetImage as Img, Model, Prediction } from '$lib/api/types';
import DetectionView from '$lib/detect/DetectionView';
import { ms } from '$lib/format';
import { PHASE_TEXT, type BrowserEngine } from '$lib/inference/browser';
import Button from '$lib/ui/Button';
import { errorDetail } from '$lib/ui/toast';
import type { CompareViewMode } from './compare';
import type { RunOn } from './predict';
import { sourceImageKey, type Source } from './source';
import type { Det } from './viewState';
import ZoomHud from './ZoomHud';
import ZoomStage, { type ZoomState } from './ZoomStage';

export interface CompareStageProps {
  source: Source;
  src: string | null;
  dims: { width: number; height: number };
  sourceName: string;
  onImageLoad?: (ok: boolean) => void;
  view: CompareViewMode;
  splitPct?: number;
  zoom: ZoomState;
  /** Optional stable zoom reset key (e.g. per video file or webcam session) so live frames preserve zoom/pan. */
  zoomResetKey?: string;
  /** Keep previous frame opaque across rapid local frame swaps (avoids blank-frame flicker). */
  keepPrevious?: boolean;
  labels: boolean;
  scores: boolean;
  truthBoxes: Img['boxes'];
  /** Visibility filter in dataset class-index space (`source.ds.classes`), shared by both panes. */
  truthOnly: Set<number> | null;
  // Model A
  modelA: Model | null;
  predA: Prediction | null;
  keptA: Det[];
  visibleCountA: number;
  classesA: string[];
  onlyA: Set<number> | null;
  activeIdxA: number | null;
  selectedIdxA: number | null;
  predictingA: boolean;
  errorA: unknown;
  onHoverA: (b: Det | null) => void;
  onSelectA: (i: number) => void;
  // Model B
  modelB: Model | null;
  predB: Prediction | null;
  keptB: Det[];
  visibleCountB: number;
  classesB: string[];
  onlyB: Set<number> | null;
  activeIdxB: number | null;
  selectedIdxB: number | null;
  predictingB: boolean;
  errorB: unknown;
  weightsMissingB: boolean;
  badFileB: { name: string; why: string | null } | null;
  runOnB: RunOn;
  onHoverB: (b: Det | null) => void;
  onSelectB: (i: number) => void;
  onRetryB: () => void;
  onChooseWeightsB: () => void;
  onServerB: () => void;
  engine: BrowserEngine;
}

const clampPct = (v: number) => Math.max(0, Math.min(100, Math.round(v * 10) / 10));

export default function CompareStage(props: CompareStageProps) {
  const splitPct = bindable(props, 'splitPct', 50);
  const splitRoot = signal<HTMLDivElement | undefined>(undefined);
  const draggingSplit = signal(false);

  let dragPointer: number | null = null;
  let dragTarget: HTMLElement | null = null;
  let windowMoveCleanup: (() => void) | null = null;

  function stopSplitDrag() {
    if (dragPointer !== null && dragTarget) {
      try {
        dragTarget.releasePointerCapture?.(dragPointer);
      } catch {
        // Ignore if capture was already released.
      }
    }
    dragPointer = null;
    dragTarget = null;
    draggingSplit.set(false);
    windowMoveCleanup?.();
    windowMoveCleanup = null;
  }

  onCleanup(stopSplitDrag);
  onWindow('blur', stopSplitDrag);
  effect(() => {
    void props.view;
    stopSplitDrag();
  });

  function updateSplitFromClientX(clientX: number) {
    const el = splitRoot.peek();
    if (!el) return;
    const r = el.getBoundingClientRect();
    if (r.width <= 0) return;
    splitPct.set(clampPct(((clientX - r.left) / r.width) * 100));
  }

  function onDividerDown(e: PointerEvent) {
    if (e.button !== 0 || dragPointer !== null) return;
    e.preventDefault();
    e.stopPropagation();
    dragPointer = e.pointerId;
    dragTarget = e.currentTarget as HTMLElement | null;
    draggingSplit.set(true);
    updateSplitFromClientX(e.clientX);
    try {
      dragTarget?.setPointerCapture?.(e.pointerId);
    } catch {
      // Synthetic events in tests may not have an active pointerId.
    }
    const onMove = (ev: PointerEvent) => {
      if (ev.pointerId !== dragPointer) return;
      updateSplitFromClientX(ev.clientX);
    };
    const onUp = (ev: PointerEvent) => {
      if (ev.pointerId !== dragPointer) return;
      stopSplitDrag();
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    windowMoveCleanup = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
    };
  }

  function onDividerKey(e: KeyboardEvent) {
    const step = e.shiftKey ? 10 : 2;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
      e.preventDefault();
      splitPct.set(clampPct(splitPct.peek() - step));
    } else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
      e.preventDefault();
      splitPct.set(clampPct(splitPct.peek() + step));
    } else if (e.key === 'PageDown') {
      e.preventDefault();
      splitPct.set(clampPct(splitPct.peek() - 10));
    } else if (e.key === 'PageUp') {
      e.preventDefault();
      splitPct.set(clampPct(splitPct.peek() + 10));
    } else if (e.key === 'Home') {
      e.preventDefault();
      splitPct.set(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      splitPct.set(100);
    }
  }

  const paneCaption = (side: 'A' | 'B', model: Model | null, count: number, pred: Prediction | null, predicting: boolean, err: unknown) => (
    <div class="pane-badge num" class:b={side === 'B'}>
      <span class="side-dot" class:b={side === 'B'}>
        {side}
      </span>
      <span class="pane-model truncate" title={model?.name ?? `Model ${side}`}>
        {model?.name ?? `Model ${side}`}
      </span>
      <span class="pane-stat">
        {predicting ? '…' : err ? 'Failed' : pred ? `${count} · ${ms(pred.timing.total_ms)}` : '—'}
      </span>
    </div>
  );

  const resetKey = () => props.zoomResetKey ?? sourceImageKey(props.source);

  const viewA = () => (
    <DetectionView
      src={props.src}
      width={props.dims.width}
      height={props.dims.height}
      boxes={props.predA ? props.keptA : []}
      truth={props.truthBoxes}
      truthOnly={props.truthOnly}
      classes={props.classesA}
      labels={props.labels}
      scores={props.scores}
      only={props.onlyA}
      active={props.activeIdxA}
      selected={props.selectedIdxA}
      onhover={(i) => props.onHoverA(i === null ? null : (props.keptA[i] ?? null))}
      onselect={props.onSelectA}
      onload={props.onImageLoad}
      keepPrevious={props.keepPrevious}
      alt={`${props.sourceName} — Model A (${props.modelA?.name ?? 'A'})`}
    />
  );

  const viewB = () => (
    <DetectionView
      src={props.src}
      width={props.dims.width}
      height={props.dims.height}
      boxes={props.predB ? props.keptB : []}
      truth={props.truthBoxes}
      truthOnly={props.truthOnly}
      classes={props.classesB}
      labels={props.labels}
      scores={props.scores}
      only={props.onlyB}
      active={props.activeIdxB}
      selected={props.selectedIdxB}
      onhover={(i) => props.onHoverB(i === null ? null : (props.keptB[i] ?? null))}
      onselect={props.onSelectB}
      keepPrevious={props.keepPrevious}
      alt={`${props.sourceName} — Model B (${props.modelB?.name ?? 'B'})`}
    />
  );

  const bStatusOverlay = (floating: boolean) => (
    <>
      {props.predictingB ? (
        <span class="busy b" role="status">
          {props.runOnB === 'server' || hub.api.mock ? `Detecting with ${props.modelB?.name}…` : PHASE_TEXT[props.engine.phase] || 'Starting…'}
        </span>
      ) : props.errorB && !props.predB ? (
        <div class={floating ? 'b-banner' : 'b-veil'} role="alert">
          <span>Model B failed: {errorDetail(props.errorB)}</span>
          <div class="b-veil-actions">
            <Button size="sm" variant="secondary" onclick={props.onRetryB}>
              Retry B
            </Button>
          </div>
        </div>
      ) : !props.predB && (props.weightsMissingB || props.badFileB) ? (
        <div class={floating ? 'b-banner' : 'b-veil'} role="alert">
          <span>
            {props.badFileB
              ? props.badFileB.why
                ? `${props.badFileB.name} could not be loaded: ${props.badFileB.why}`
                : `${props.badFileB.name} is not a .folo weights file.`
              : `Model B (${props.modelB?.name ?? 'B'}) needs its .folo weights file.`}
          </span>
          <div class="b-veil-actions">
            <Button size="sm" variant="primary" icon={FileUp} onclick={props.onChooseWeightsB}>
              Choose .folo
            </Button>
            <Button size="sm" onclick={props.onServerB}>
              Run on server
            </Button>
          </div>
        </div>
      ) : null}
    </>
  );

  return (
    <>
      {props.view === 'side-by-side' ? (
        <div class="compare-grid" data-compare-view="side-by-side">
          <div class="compare-pane" data-compare-side="a">
            <ZoomStage resetKey={resetKey()} imageWidth={props.dims.width} imageHeight={props.dims.height} zoom={props.zoom} hideHud>
              {viewA()}
            </ZoomStage>
            {paneCaption('A', props.modelA, props.visibleCountA, props.predA, props.predictingA, props.errorA)}
          </div>
          <div class="compare-pane" data-compare-side="b">
            <ZoomStage resetKey={resetKey()} imageWidth={props.dims.width} imageHeight={props.dims.height} zoom={props.zoom} hideHud>
              {viewB()}
            </ZoomStage>
            {paneCaption('B', props.modelB, props.visibleCountB, props.predB, props.predictingB, props.errorB)}
            {bStatusOverlay(false)}
          </div>
          <ZoomHud zoom={props.zoom} />
        </div>
      ) : (
        <div class="split-root" class:dragging={draggingSplit.get()} ref={splitRoot} data-compare-view="split">
          <div class="split-layer back" data-compare-side="b">
            <ZoomStage resetKey={resetKey()} imageWidth={props.dims.width} imageHeight={props.dims.height} zoom={props.zoom} hideHud>
              {viewB()}
            </ZoomStage>
          </div>
          <div
            class="split-layer front"
            data-compare-side="a"
            style={{ clipPath: `inset(0 ${100 - splitPct.get()}% 0 0)` }}
          >
            <ZoomStage resetKey={resetKey()} imageWidth={props.dims.width} imageHeight={props.dims.height} zoom={props.zoom} hideHud>
              {viewA()}
            </ZoomStage>
          </div>
          <div class="split-badge-left">
            {paneCaption('A', props.modelA, props.visibleCountA, props.predA, props.predictingA, props.errorA)}
          </div>
          <div class="split-badge-right">
            {paneCaption('B', props.modelB, props.visibleCountB, props.predB, props.predictingB, props.errorB)}
          </div>
          <span class="split-line" aria-hidden="true" style={{ left: `${splitPct.get()}%` }}></span>
          <div
            class="split-divider"
            role="slider"
            tabindex={0}
            aria-label="Compare split position"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(splitPct.get())}
            aria-valuetext={`${Math.round(splitPct.get())}% Model A, ${100 - Math.round(splitPct.get())}% Model B`}
            style={{ left: `${splitPct.get()}%` }}
            onpointerdown={onDividerDown}
            onlostpointercapture={stopSplitDrag}
            onkeydown={onDividerKey}
          >
            <span class="split-handle" aria-hidden="true">
              <ArrowLeftRight size={14} strokeWidth={2} />
            </span>
          </div>
          <ZoomHud zoom={props.zoom} />
          {bStatusOverlay(true)}
        </div>
      )}
    </>
  );
}

css`
  .compare-grid {
    position: relative;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 2px;
    width: 100%;
    height: 100%;
    background: var(--line-default);
  }
  .compare-pane {
    position: relative;
    min-width: 0;
    min-height: 0;
    width: 100%;
    height: 100%;
    background: var(--bg-inset);
    overflow: hidden;
  }
  .pane-badge {
    position: absolute;
    top: var(--space-2);
    left: var(--space-2);
    z-index: 2;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    max-width: calc(100% - var(--space-4));
    padding: 2px var(--space-2) 2px 4px;
    border-radius: var(--radius-sm);
    background: color-mix(in srgb, var(--bg-overlay) 90%, transparent);
    border: 1px solid var(--line-default);
    box-shadow: var(--shadow-sm);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-primary);
    pointer-events: none;
  }
  .side-dot {
    display: inline-grid;
    place-items: center;
    width: 18px;
    height: 18px;
    border-radius: var(--radius-xs);
    background: var(--accent-solid);
    color: var(--accent-on);
    font-weight: var(--weight-semibold);
    flex-shrink: 0;
  }
  .side-dot.b {
    background: var(--info-solid);
    color: var(--text-inverse);
  }
  .pane-model {
    min-width: 0;
    font-weight: var(--weight-medium);
  }
  .pane-stat {
    color: var(--text-secondary);
    white-space: nowrap;
    flex-shrink: 0;
  }
  .split-root {
    position: relative;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: var(--bg-inset);
  }
  .split-root.dragging {
    user-select: none;
    cursor: ew-resize;
  }
  .split-layer {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }
  .split-layer.front {
    z-index: 1;
  }
  .split-badge-left,
  .split-badge-right {
    position: absolute;
    top: var(--space-2);
    z-index: 3;
    max-width: calc(50% - var(--space-4));
    pointer-events: none;
  }
  .split-badge-left {
    left: var(--space-2);
  }
  .split-badge-right {
    right: var(--space-2);
  }
  .split-badge-left .pane-badge,
  .split-badge-right .pane-badge {
    position: static;
    max-width: 100%;
  }
  .split-line {
    position: absolute;
    top: 0;
    bottom: 0;
    z-index: 3;
    width: 2px;
    margin-left: -1px;
    background: var(--accent-line);
    box-shadow: 0 0 0 1px var(--class-halo);
    pointer-events: none;
  }
  .split-divider {
    position: absolute;
    top: 40px;
    bottom: 64px;
    z-index: 4;
    width: 28px;
    margin-left: -14px; /* grid-ok: centers 28px grab target on split line */
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: ew-resize;
    touch-action: none;
  }
  .split-divider:focus-visible {
    outline: none;
  }
  .split-divider:focus-visible .split-handle {
    outline: 2px solid var(--focus);
    outline-offset: 2px;
  }
  .split-handle {
    position: relative;
    display: grid;
    place-items: center;
    width: 28px;
    height: 28px;
    border-radius: var(--radius-full);
    background: var(--bg-overlay);
    border: 1px solid var(--accent-line);
    color: var(--text-primary);
    box-shadow: var(--shadow-md);
  }
  .busy.b {
    position: absolute;
    bottom: var(--space-2);
    left: var(--space-2);
    z-index: 5;
    padding: 2px var(--space-2);
    border-radius: var(--radius-sm);
    background: var(--bg-overlay);
    box-shadow: var(--shadow-md);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .b-veil {
    position: absolute;
    inset: 0;
    z-index: 3;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-2);
    padding: var(--space-4);
    background: color-mix(in srgb, var(--bg-inset) 82%, transparent);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
    text-align: center;
  }
  .b-banner {
    position: absolute;
    right: var(--space-2);
    top: 44px;
    z-index: 5;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    max-width: min(320px, calc(100% - var(--space-4)));
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-overlay);
    border: 1px solid var(--line-default);
    box-shadow: var(--shadow-md);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .b-veil-actions {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: var(--space-2);
  }
  @media (max-width: 767px) {
    .compare-grid {
      grid-template-columns: minmax(0, 1fr);
      grid-template-rows: repeat(2, minmax(180px, 1fr));
    }
    .split-divider {
      width: 44px;
      margin-left: -22px; /* grid-ok: centers 44px touch target on split line */
    }
    .split-handle {
      width: 32px;
      height: 32px;
    }
  }
`;
