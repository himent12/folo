// Reactive controller for Playground inference options (Standard, SAHI tiled inference, and TTA):
// tracks capability availability for both browser folo-wasm (`detectTiled` / `detectTta`) and live servers
// (`predict_options`), exposes honest disabled reasons, and resolves the effective config per backend.
import { computed, signal, type Signal } from 'fw';
import { hub } from '$lib/api/index';
import {
  DEFAULT_INFERENCE_CONFIG,
  DEFAULT_TILE_OPTIONS,
  DEFAULT_TTA_OPTIONS,
  inferenceConfigKey,
  inferenceMetaOf,
  inferenceShortTag,
  normalizeBrowserPrecision,
  scalesForPreset,
  ttaScalePreset,
  type BrowserEngine,
  type BrowserPrecision,
  type DetectorInfo,
  type InferenceConfig,
  type InferenceMode,
  type TtaScalePreset,
} from '$lib/inference/browser';
import type { RunOn } from './predict';

export interface InferenceAvailability {
  sahi: boolean;
  tta: boolean;
  sahiReason: string | null;
  ttaReason: string | null;
  /** General availability note shown under the mode selector when SAHI or TTA is unavailable. */
  note: string | null;
}

export interface UseInferenceOptionsInput {
  runOn: Signal<RunOn>;
  /** Optional Model B backend while Compare A/B is active so heterogeneous A/B backends stay honest. */
  compareRunOn?: () => RunOn | null;
  engine: BrowserEngine;
  browserOff: () => string | null;
  serverPredictOptions: () => boolean;
}

/** Resolves honest SAHI / TTA availability and user-facing disabled reasons for a given backend. */
export function resolveInferenceAvailability(opts: {
  runOn: RunOn;
  isMock: boolean;
  browserOff: boolean;
  serverPredictOptions: boolean;
  wasmTiled: boolean | null;
  wasmTta: boolean | null;
}): InferenceAvailability {
  if (opts.isMock) {
    const reason = 'Demo mode simulates single-pass detections. Connect to a live server for SAHI and TTA.';
    return { sahi: false, tta: false, sahiReason: reason, ttaReason: reason, note: reason };
  }
  if (opts.runOn === 'server') {
    if (opts.serverPredictOptions) {
      return { sahi: true, tta: true, sahiReason: null, ttaReason: null, note: null };
    }
    const reason = opts.browserOff
      ? 'This server does not advertise SAHI or TTA support (predict_options).'
      : 'This server does not support SAHI or TTA. Switch Run on to WebGPU or CPU, or upgrade folo serve.';
    return { sahi: false, tta: false, sahiReason: reason, ttaReason: reason, note: reason };
  }
  const sahi = opts.wasmTiled !== false;
  const tta = opts.wasmTta !== false;
  const sahiReason = sahi ? null : 'This browser WASM build lacks detectTiled. Rebuild folo-wasm to enable SAHI.';
  const ttaReason = tta ? null : 'This browser WASM build lacks detectTta. Rebuild folo-wasm to enable TTA.';
  let note: string | null = null;
  if (!sahi && !tta) {
    note = 'This browser WASM build lacks SAHI and TTA (detectTiled / detectTta). Rebuild folo-wasm to enable them.';
  } else if (!sahi) {
    note = sahiReason;
  } else if (!tta) {
    note = ttaReason;
  }
  return { sahi, tta, sahiReason, ttaReason, note };
}

export function useInferenceOptions(input: UseInferenceOptionsInput) {
  const mode = signal<InferenceMode>('standard');
  const tileSize = signal<number>(DEFAULT_TILE_OPTIONS.tileSize);
  const overlap = signal<number>(DEFAULT_TILE_OPTIONS.overlap);
  const fullImage = signal<boolean>(DEFAULT_TILE_OPTIONS.fullImage);
  const hflip = signal<boolean>(DEFAULT_TTA_OPTIONS.hflip);
  const scalePreset = signal<TtaScalePreset>(ttaScalePreset(DEFAULT_TTA_OPTIONS.scales));
  const precision = signal<BrowserPrecision>('auto');

  const config = computed<InferenceConfig>(() => {
    const p = precision.get();
    return {
      mode: mode.get(),
      tile: {
        tileSize: tileSize.get(),
        overlap: overlap.get(),
        fullImage: fullImage.get(),
      },
      tta: {
        hflip: hflip.get(),
        scales: scalesForPreset(scalePreset.get()),
      },
      ...(p !== 'auto' ? { precision: p } : {}),
    };
  });

  const hasBrowserBackend = computed(() => {
    if (input.browserOff()) return false;
    if (input.runOn.get() !== 'server') return true;
    const onB = input.compareRunOn?.() ?? null;
    return onB !== null && onB !== 'server';
  });

  const hasWebGpuBackend = computed(() => {
    if (input.browserOff()) return false;
    if (input.runOn.get() === 'webgpu') return true;
    return (input.compareRunOn?.() ?? null) === 'webgpu';
  });

  const engineInfo = computed<DetectorInfo | null>(() => input.engine.info);

  function precisionFor(on: RunOn): BrowserPrecision {
    return on === 'server' ? 'auto' : precision.get();
  }

  function availabilityFor(on: RunOn): InferenceAvailability {
    const caps = input.engine.capabilities;
    return resolveInferenceAvailability({
      runOn: on,
      isMock: hub.api.mock || hub.serverMock === true,
      browserOff: !!input.browserOff(),
      serverPredictOptions: input.serverPredictOptions(),
      wasmTiled: caps.tiled,
      wasmTta: caps.tta,
    });
  }

  const availability = computed<InferenceAvailability>(() => {
    const a = availabilityFor(input.runOn.get());
    const onB = input.compareRunOn?.() ?? null;
    if (!onB || onB === input.runOn.get()) return a;
    const b = availabilityFor(onB);
    const sahi = a.sahi || b.sahi;
    const tta = a.tta || b.tta;
    const sahiReason = sahi ? null : (a.sahiReason ?? b.sahiReason);
    const ttaReason = tta ? null : (a.ttaReason ?? b.ttaReason);
    let note: string | null = null;
    if (!sahi && !tta) {
      note = a.note ?? b.note;
    } else if (a.sahi !== b.sahi || a.tta !== b.tta) {
      const limited = !a.sahi || !a.tta ? 'Model A' : 'Model B';
      note = `${limited}'s backend does not support SAHI or TTA and will run Standard.`;
    }
    return { sahi, tta, sahiReason, ttaReason, note };
  });

  /** Effective config sent for `on`: degrades cleanly to `standard` whenever the chosen mode is unavailable on `on`. */
  function effectiveFor(on: RunOn): InferenceConfig {
    const cfg = config.get();
    if (cfg.mode === 'standard') return cfg;
    const avail = availabilityFor(on);
    if (cfg.mode === 'sahi' && !avail.sahi) return DEFAULT_INFERENCE_CONFIG;
    if (cfg.mode === 'tta' && !avail.tta) return DEFAULT_INFERENCE_CONFIG;
    return cfg;
  }

  const effective = computed(() => effectiveFor(input.runOn.get()));
  const effectiveKey = computed(() => inferenceConfigKey(effective.get()));
  const effectiveMeta = computed(() => inferenceMetaOf(effective.get()));
  const badge = computed(() => {
    const onB = input.compareRunOn?.() ?? null;
    const meta = effectiveMeta.get() ?? (onB ? inferenceMetaOf(effectiveFor(onB)) : undefined);
    const modeTag = inferenceShortTag(meta);
    const prec = hasBrowserBackend.get() ? precision.get() : 'auto';
    const precTag = prec === 'int8' ? 'INT8' : prec !== 'auto' ? prec : null;
    if (modeTag && precTag) return `${modeTag} · ${precTag}`;
    return modeTag ?? precTag;
  });

  /** Active mode shown in the segmented selector (reads `standard` if unsupported on every active backend). */
  const displayMode = computed<InferenceMode>(() => {
    const cfg = config.get();
    if (cfg.mode === 'standard') return 'standard';
    const avail = availability.get();
    if (cfg.mode === 'sahi' && !avail.sahi) return 'standard';
    if (cfg.mode === 'tta' && !avail.tta) return 'standard';
    return cfg.mode;
  });

  function setMode(next: InferenceMode) {
    const avail = availability.peek();
    if (next === 'sahi' && !avail.sahi) return;
    if (next === 'tta' && !avail.tta) return;
    mode.set(next);
  }

  function restore(saved: InferenceConfig) {
    mode.set(saved.mode);
    tileSize.set(saved.tile.tileSize);
    overlap.set(saved.tile.overlap);
    fullImage.set(saved.tile.fullImage);
    hflip.set(saved.tta.hflip);
    scalePreset.set(ttaScalePreset(saved.tta.scales));
    precision.set(normalizeBrowserPrecision(saved.precision));
  }

  function reset() {
    mode.set('standard');
    tileSize.set(DEFAULT_TILE_OPTIONS.tileSize);
    overlap.set(DEFAULT_TILE_OPTIONS.overlap);
    fullImage.set(DEFAULT_TILE_OPTIONS.fullImage);
    hflip.set(DEFAULT_TTA_OPTIONS.hflip);
    scalePreset.set(ttaScalePreset(DEFAULT_TTA_OPTIONS.scales));
    precision.set('auto');
  }

  return {
    mode,
    displayMode,
    tileSize,
    overlap,
    fullImage,
    hflip,
    scalePreset,
    precision,
    hasBrowserBackend,
    hasWebGpuBackend,
    engineInfo,
    precisionFor,
    config,
    availability,
    effective,
    effectiveKey,
    effectiveMeta,
    badge,
    availabilityFor,
    effectiveFor,
    setMode,
    restore,
    reset,
  };
}

export type InferenceOptionsController = ReturnType<typeof useInferenceOptions>;
