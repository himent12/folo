// Detects whether the current Studio build ships the folo-wasm in-browser runtime and keeps `runOn` on the server
// whenever the runtime is absent.
import { computed, effect, signal, untrack, type Signal } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import { hasWebGpu, NO_WASM, probeWasm, servedBy, wasmFromFeatures, wasmFromSystem } from '$lib/inference/browser';
import type { RunOn } from './predict';

export function useWasmAvailability(runOn: Signal<RunOn>, needRealWasm?: () => boolean) {
  const webgpu = hasWebGpu();
  const system = new Resource((s) => hub.api.system(s));
  const probed = signal<boolean | null>(null);
  const servedHere = computed(() => hub.api.mock || servedBy(hub.serverUrl));
  const fromServer = computed(
    () =>
      wasmFromSystem(system.data, servedHere.get()) ??
      (hub.api.mock ? null : wasmFromFeatures(hub.features, servedHere.get())),
  );

  effect(() => {
    if (hub.mode === 'connecting') return;
    if (hub.api.mock) {
      if (!needRealWasm?.()) return;
    } else {
      if (fromServer.get() !== null) return;
      if (servedHere.get() && system.loading) return;
    }
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
  const browserOff = computed(() => (wasm.get() === false ? NO_WASM : null));
  // Demo mode simulates browser CPU/WebGPU inference, but ByteTrack Track IDs requires the real folo-wasm asset.
  const trackerBrowserOff = computed(() => {
    const base = browserOff.get();
    if (base) return base;
    if (hub.api.mock && needRealWasm?.() && probed.get() !== true) return NO_WASM;
    return null;
  });

  effect(() => {
    if (browserOff.get() && untrack(() => runOn.get()) !== 'server') runOn.set('server');
  });

  const runOnNote = computed(() => (!browserOff.get() && !webgpu ? 'No WebGPU in this browser' : null));
  const serverPredictOptions = computed(() => {
    if (hub.api.mock || hub.serverMock === true || hub.offline) return false;
    // `/api/health`'s feature list (`hub.features`) is refreshed on every reconnect and cleared on outage:
    // treat it as authoritative when present so stale `/api/system` data never survives a server downgrade.
    const healthFeatures = hub.features;
    if (healthFeatures !== null) return healthFeatures.includes('predict_options');
    return !!system.data?.features?.includes('predict_options');
  });
  return { webgpu, browserOff, trackerBrowserOff, runOnNote, serverPredictOptions };
}
