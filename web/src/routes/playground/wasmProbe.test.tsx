// @vitest-environment happy-dom
// Playground same-origin folo-wasm detection (#309): verifies zero redundant HEAD /folo-wasm/* probes when
// same-origin `/api/health` (`hub.features`) or `/api/system` already settles `web_wasm`, and fallback probing for legacy servers.
import { flush as flushSync } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';

vi.mock('$lib/inference/browser', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/inference/browser')>();
  return { ...actual, servedBy: () => true };
});

import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { resetWasmProbe } from '$lib/inference/browser';
import { parse, router } from '$lib/router';
import Playground from '../Playground';

describe('Playground same-origin wasm feature detection (#309)', () => {
  let target: HTMLElement | undefined;
  let c: ReturnType<typeof render> | undefined;
  const savedApi = hub.api;
  const savedMode = hub.mode;
  const savedFeatures = hub.features;

  afterEach(() => {
    c?.unmount();
    c = undefined;
    target?.remove();
    target = undefined;
    hub.api = savedApi;
    hub.mode = savedMode;
    hub.features = savedFeatures;
    vi.unstubAllGlobals();
    resetWasmProbe();
  });

  async function waitFor<T>(get: () => T | null | undefined | false, what: string): Promise<T> {
    let v: T | null | undefined | false = undefined;
    await vi.waitFor(
      () => {
        flushSync();
        v = get();
        if (!v) throw new Error(`waiting for ${what}`);
      },
      { timeout: 4000, interval: 10 },
    );
    return v as T;
  }

  it('skips HEAD /folo-wasm/* probes when same-origin hub.features already settles web_wasm', async () => {
    resetWasmProbe();
    const fetchSpy = vi.fn(async () => new Response(null, { status: 404 }));
    vi.stubGlobal('fetch', fetchSpy);

    const api = new MockApi('normal', 0);
    Object.defineProperty(api, 'mock', { value: false });
    const origSys = api.system.bind(api);
    api.system = async () => ({ ...(await origSys()), features: ['run_cli'] });

    hub.api = api;
    hub.features = ['run_cli'];
    hub.mode = 'live';

    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_shapes_s');
    c = render(() => <Playground />, target);

    await waitFor(() => target!.querySelector('.model-pick'), 'model picker');
    await waitFor(() => !target!.querySelector('.runon'), 'Run on hidden');
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('waits for /api/system before probing on a same-origin server whose /api/health lacked features', async () => {
    resetWasmProbe();
    const fetchSpy = vi.fn(async () => new Response(null, { status: 404 }));
    vi.stubGlobal('fetch', fetchSpy);

    let resolveSys!: (v: Awaited<ReturnType<MockApi['system']>>) => void;
    const api = new MockApi('normal', 0);
    Object.defineProperty(api, 'mock', { value: false });
    const realSys = await api.system();
    api.system = () => new Promise((r) => (resolveSys = r));

    hub.api = api;
    hub.features = null;
    hub.mode = 'live';

    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_shapes_s');
    c = render(() => <Playground />, target);

    await waitFor(() => target!.querySelector('.model-pick'), 'model picker');
    // `/api/system` is still in flight: must not prematurely probe `/folo-wasm/*`.
    expect(fetchSpy).not.toHaveBeenCalled();

    // Legacy server without `run_cli` or `web_wasm`: falls back to `probeWasm()` once `/api/system` settles.
    resolveSys({ ...realSys, features: ['queue_control'] });
    await waitFor(() => fetchSpy.mock.calls.length >= 2, 'fallback wasm HEAD probes');
    expect(fetchSpy).toHaveBeenCalledWith(expect.stringContaining('folo-wasm/folo_wasm.js'), { method: 'HEAD' });
    expect(fetchSpy).toHaveBeenCalledWith(expect.stringContaining('folo-wasm/folo_wasm_bg.wasm'), { method: 'HEAD' });
  });
});
