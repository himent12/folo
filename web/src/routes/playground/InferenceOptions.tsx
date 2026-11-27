/**
 * Compact collapsible Inference Options panel for Playground: Standard, SAHI tiled inference (tile size, overlap,
 * full-image pass), and Test-Time Augmentation (horizontal flip, scale preset), with honest availability messaging
 * when the active backend or WASM bundle does not support a mode.
 */
import ChevronRight from '$icons/chevron-right';
import { computed, css, signal } from 'fw';
import { TILE_OVERLAP_OPTIONS, TILE_SIZE_OPTIONS, type InferenceMode, type TtaScalePreset } from '$lib/inference/browser';
import Segmented from '$lib/ui/Segmented';
import Switch from '$lib/ui/Switch';
import type { InferenceOptionsController } from './useInferenceOptions';

export interface InferenceOptionsProps {
  ctrl: InferenceOptionsController;
  /** Compact inline style when embedded inside a sheet or popover. */
  inline?: boolean;
}

const TILE_SIZE_SEG = TILE_SIZE_OPTIONS.map((px) => ({ value: String(px), label: `${px}` }));
const OVERLAP_SEG = TILE_OVERLAP_OPTIONS.map((v) => ({ value: String(v), label: `${Math.round(v * 100)}%` }));
const TTA_SCALE_SEG: { value: TtaScalePreset; label: string }[] = [
  { value: 'multi', label: '3 scales' },
  { value: 'single', label: '1.0× only' },
];

export default function InferenceOptions(props: InferenceOptionsProps) {
  const open = signal(false);

  const modeValue = {
    get: () => props.ctrl.displayMode.get(),
    set: (v: InferenceMode) => props.ctrl.setMode(v),
  };

  const tileSizeValue = {
    get: () => String(props.ctrl.tileSize.get()),
    set: (v: string) => props.ctrl.tileSize.set(Number(v)),
  };

  const overlapValue = {
    get: () => String(props.ctrl.overlap.get()),
    set: (v: string) => props.ctrl.overlap.set(Number(v)),
  };

  const scaleValue = {
    get: () => props.ctrl.scalePreset.get(),
    set: (v: TtaScalePreset) => props.ctrl.scalePreset.set(v),
  };

  const modeOptions = computed(() => {
    const a = props.ctrl.availability.get();
    return [
      { value: 'standard' as const, label: 'Standard' },
      { value: 'sahi' as const, label: 'SAHI', disabled: !a.sahi, tip: a.sahiReason ?? undefined },
      { value: 'tta' as const, label: 'TTA', disabled: !a.tta, tip: a.ttaReason ?? undefined },
    ];
  });

  const activeBadge = computed(() => props.ctrl.badge.get());
  const availNote = computed(() => props.ctrl.availability.get().note);

  return (
    <div class="inf-opt" class:inline={!!props.inline} data-inference-options>
      <button
        type="button"
        class="inf-toggle"
        aria-expanded={open.get()}
        onclick={() => open.set(!open.peek())}
      >
        <ChevronRight size={14} class={open.get() ? 'chev open' : 'chev'} aria-hidden="true" />
        <span class="inf-title">Inference options</span>
        <span class="inf-badge num" class:active={!!activeBadge.get()}>
          {activeBadge.get() ?? 'Standard'}
        </span>
      </button>
      {open.get() ? (
        <div class="inf-body">
          <div class="inf-row">
            <span class="inf-label">Mode</span>
            <Segmented label="Inference mode" size="sm" bind:value={modeValue} options={modeOptions.get()} />
          </div>
          {props.ctrl.displayMode.get() === 'sahi' ? (
            <div class="inf-group" role="group" aria-label="SAHI tiled inference options">
              <div class="inf-row">
                <span class="inf-label">Tile size (px)</span>
                <Segmented label="Tile size" size="sm" bind:value={tileSizeValue} options={TILE_SIZE_SEG} />
              </div>
              <div class="inf-row">
                <span class="inf-label">Overlap</span>
                <Segmented label="Tile overlap" size="sm" bind:value={overlapValue} options={OVERLAP_SEG} />
              </div>
              <Switch
                label="Full-image pass"
                hint="Run one whole-image pass alongside tiles so large objects stay intact."
                bind:checked={props.ctrl.fullImage}
              />
            </div>
          ) : props.ctrl.displayMode.get() === 'tta' ? (
            <div class="inf-group" role="group" aria-label="TTA options">
              <div class="inf-row">
                <span class="inf-label">Scales</span>
                <Segmented label="TTA scales" size="sm" bind:value={scaleValue} options={TTA_SCALE_SEG} />
              </div>
              <Switch
                label="Horizontal flip"
                hint="Mirror each scale and fuse overlapping detections."
                bind:checked={props.ctrl.hflip}
              />
            </div>
          ) : null}
          {availNote.get() ? <p class="inf-note">{availNote.get()}</p> : null}
        </div>
      ) : null}
    </div>
  );
}

css`
  .inf-opt {
    display: grid;
    gap: var(--space-2);
    min-width: 0;
  }
  .inf-opt.inline {
    margin-top: var(--space-3);
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
  }
  .inf-toggle {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    width: 100%;
    min-height: var(--control-sm);
    padding: 0;
    color: var(--text-primary);
    text-align: left;
  }
  .inf-toggle:hover .inf-title {
    color: var(--text-primary);
  }
  .inf-toggle :global(svg.chev) {
    flex-shrink: 0;
    color: var(--text-tertiary);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .inf-toggle :global(svg.chev.open) {
    transform: rotate(90deg);
  }
  .inf-title {
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  .inf-badge {
    margin-left: auto;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .inf-badge.active {
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    background: var(--accent-muted);
    color: var(--accent-text);
    font-weight: var(--weight-medium);
  }
  .inf-body {
    display: grid;
    gap: var(--space-3);
    padding-top: var(--space-1);
  }
  .inf-group {
    display: grid;
    gap: var(--space-3);
  }
  .inf-row {
    display: grid;
    gap: var(--space-1);
    min-width: 0;
  }
  .inf-label {
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .inf-note {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
`;
