// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { prefs } from '$lib/prefs';
import { parse, router } from '$lib/router';
import { buildPaletteItems } from '$lib/shell/commandPalette/providers';
import { installEntryMemory, restoringScroll } from '$lib/shell/entryMemory';
import { focusRoute } from '$lib/shell/routeFocus';
import { tabHistory } from '$lib/tabHistory.testing';
import { undescribedFields } from '$lib/ui/fieldA11y';
import { render } from '../../framework/testing';
import Settings from './Settings';
import Throughput from './settings/Throughput';

/** Port of `mount(Component, { target, props })`: renders into `target`; `unmount(c)` is `c.unmount()`. */
const mount = (C: typeof Settings | typeof Throughput, o: { target: HTMLElement; props?: { benchmark?: boolean } }) =>
  render(() => (C === Throughput ? <Throughput {...(o.props ?? {})} /> : <Settings />), o.target);
const unmount = (c: { unmount(): void }) => c.unmount();

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

async function until<T>(get: () => T | null | undefined | false, what = 'the condition'): Promise<T> {
  let v: T | null | undefined | false = undefined;
  await vi.waitFor(
    () => {
      flushSync();
      v = get();
      if (!v) throw new Error(`still waiting for ${what}`);
    },
    { timeout: 3000, interval: 10 },
  );
  return v as T;
}

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
  history.replaceState(history.state, '', '#/settings');
  router.route = parse('#/settings');
  hub.mode = 'mock';
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
});

describe('Settings', () => {
  it('lists Cloud after Connection only on a server with the cloud feature', async () => {
    hub.api = new MockApi('cloud-off', 0);
    c = mount(Settings, { target });
    const jump = target.querySelector<HTMLElement>('nav.jump')!;
    await until(() => jump.textContent?.includes('System'), 'the System section');
    expect([...jump.querySelectorAll('a')].map((a) => a.textContent)).not.toContain('Cloud');
    expect(target.querySelector('#cloud')).toBeNull();
  });

  it('says when the system information failed, with Retry, instead of leaving System out (shell F11c)', async () => {
    const api = new MockApi('normal', 0);
    const real = api.system.bind(api);
    let fail = true;
    vi.spyOn(api, 'system').mockImplementation(() => (fail ? Promise.reject(new ApiError(500, 'internal', 'boom', null, 'GET /api/system')) : real()));
    hub.api = api;
    c = mount(Settings, { target });
    const panel = await until(() => target.querySelector('#system .err'), 'the error');
    expect(panel.querySelector('.title')?.textContent).toBe('Could not load the system information');
    expect([...target.querySelectorAll<HTMLElement>('nav.jump a')].map((a) => a.textContent)).toContain('System');
    fail = false;
    [...panel.querySelectorAll('button')].find((b) => b.textContent?.includes('Retry'))!.click();
    await until(() => target.querySelector('#system') && !target.querySelector('#system .err'), 'the System section');
  });

  it('describes the server URL field by its hint', () => {
    hub.api = new MockApi('normal', 0);
    c = mount(Settings, { target });
    flushSync();
    expect(target.querySelector('#server-url')!.getAttribute('aria-describedby')).toBe('server-url-hint');
    expect(undescribedFields(target)).toEqual([]);
  });

  // The field is a row (its name and hint on the left, the input on the right): it is labelled by its name, and an
  // address that will not do is said under the row, as its description, and marks the input invalid.
  it('labels the server URL by its row and describes it by the error while the address will not do', () => {
    hub.api = new MockApi('normal', 0);
    c = mount(Settings, { target });
    flushSync();
    const input = target.querySelector<HTMLInputElement>('#server-url')!;
    expect(target.querySelector('label[for="server-url"]')?.textContent).toBe('Server URL');
    input.value = 'ftp://x';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    const err = target.querySelector('#server-url-error')!;
    expect(err.getAttribute('role')).toBe('alert');
    expect(err.textContent).toBe('The server URL must start with http:// or https://.');
    expect(input.getAttribute('aria-describedby')).toBe('server-url-error');
    expect(input.getAttribute('aria-invalid')).toBe('true');
  });

  it('jumps to a section from the strip below 1024 px: every section listed, focus moved onto its heading', async () => {
    hub.api = new MockApi('normal', 0);
    c = mount(Settings, { target });
    const jump = target.querySelector<HTMLElement>('nav.jump')!;
    await until(() => jump.textContent?.includes('System'), 'the System section');
    expect([...jump.querySelectorAll('a')].map((a) => a.textContent)).toEqual([
      'Connection',
      'Cloud',
      'Appearance',
      'Units',
      'Keyboard shortcuts',
      'System',
      'Throughput',
      'Diagnostics',
      'About',
    ]);
    const scroll = vi.fn();
    const system = target.querySelector<HTMLElement>('#system')!;
    system.scrollIntoView = scroll;
    const heading = target.querySelector<HTMLElement>('#system-title')!;
    const focus = vi.spyOn(heading, 'focus');
    // Taken with the mouse (a click with a pointer's detail): the heading is focused without a ring.
    [...jump.querySelectorAll('a')].find((a) => a.textContent === 'System')!.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, detail: 1 }));
    flushSync();
    expect(scroll).toHaveBeenCalledWith({ block: 'start' });
    expect(document.activeElement).toBe(heading);
    expect(heading.getAttribute('tabindex')).toBe('-1');
    expect(focus).toHaveBeenLastCalledWith({ preventScroll: true, focusVisible: false });
    // Taken with Enter (a click with no pointer detail): the browser decides, so the ring shows as after a Tab.
    [...jump.querySelectorAll('a')].find((a) => a.textContent === 'System')!.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, detail: 0 }));
    flushSync();
    expect(focus).toHaveBeenLastCalledWith({ preventScroll: true });
    expect(jump.querySelector('a.on')?.textContent).toBe('System');
    expect(jump.querySelector('a.on')?.getAttribute('aria-current')).toBe('location');
    expect(location.hash).toBe('#/settings?section=system');
    expect(router.route.query.get('section')).toBe('system');
  });

  it('syncs ?section= in place when a jump link is explicitly activated, and leaves the URL alone on passive scroll', async () => {
    hub.api = new MockApi('normal', 0);
    history.replaceState(history.state, '', '#/settings?section=cloud');
    router.route = parse('#/settings?section=cloud');
    const entryBefore = router.entry;
    const go = vi.spyOn(router, 'go');
    try {
      c = mount(Settings, { target });
      const jump = target.querySelector<HTMLElement>('nav.jump')!;
      await until(() => jump.textContent?.includes('Cloud'), 'the Cloud section');
      const about = target.querySelector<HTMLElement>('#about')!;
      const aboutHeading = target.querySelector<HTMLElement>('#about-title')!;
      const scroll = vi.fn();
      about.scrollIntoView = scroll;
      const focus = vi.spyOn(aboutHeading, 'focus');

      // Explicit click on About updates ?section=about in place, scrolls once, and focuses #about-title.
      [...jump.querySelectorAll('a')].find((a) => a.textContent === 'About')!.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, detail: 1 }));
      flushSync();
      await new Promise((r) => requestAnimationFrame(r));
      flushSync();
      expect(location.hash).toBe('#/settings?section=about');
      expect(router.route.query.get('section')).toBe('about');
      expect(router.entry).toBe(entryBefore);
      expect(go).toHaveBeenCalledTimes(1);
      expect(go).toHaveBeenCalledWith('/settings?section=about', { replace: true });
      expect(scroll).toHaveBeenCalledTimes(1);
      expect(scroll).toHaveBeenCalledWith({ block: 'start' });
      expect(document.activeElement).toBe(aboutHeading);
      expect(focus).toHaveBeenLastCalledWith({ preventScroll: true, focusVisible: false });

      // Passive scroll releases scroll-spy without rewriting the URL.
      window.dispatchEvent(new Event('wheel'));
      window.dispatchEvent(new Event('scroll'));
      await new Promise((r) => requestAnimationFrame(r));
      flushSync();
      expect(go).toHaveBeenCalledTimes(1);
      expect(location.hash).toBe('#/settings?section=about');

      // Keyboard activation (Enter) on another section preserves keyboard focus ring (no focusVisible: false) after RAF.
      const unitsHeading = target.querySelector<HTMLElement>('#units-title')!;
      const unitsEl = target.querySelector<HTMLElement>('#units')!;
      const unitsScroll = vi.fn();
      unitsEl.scrollIntoView = unitsScroll;
      const unitsFocus = vi.spyOn(unitsHeading, 'focus');
      [...jump.querySelectorAll('a')].find((a) => a.textContent === 'Units')!.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, detail: 0 }));
      flushSync();
      await new Promise((r) => requestAnimationFrame(r));
      flushSync();
      expect(location.hash).toBe('#/settings?section=units');
      expect(document.activeElement).toBe(unitsHeading);
      expect(unitsFocus).toHaveBeenCalledTimes(1);
      expect(unitsFocus).toHaveBeenLastCalledWith({ preventScroll: true });

      // After scrolling away from Units, picking Units again in the command palette still jumps back to it.
      window.dispatchEvent(new Event('wheel'));
      unitsHeading.blur();
      const paletteUnits = buildPaletteItems({
        pageCommands: [],
        runs: [],
        datasets: [],
        models: [],
        benches: [],
        benchesOk: true,
        labelJobs: [],
        loops: [],
        cloud: true,
        lacksLabel: false,
        touchOnly: false,
        onshortcuts: () => {},
      }).find((item) => item.id === 'settings-units')!;
      paletteUnits.run();
      await new Promise((r) => requestAnimationFrame(r));
      flushSync();
      expect(unitsScroll).toHaveBeenCalledTimes(2);
      expect(document.activeElement).toBe(unitsHeading);
    } finally {
      go.mockRestore();
    }
  });

  it('does not overwrite a pending navigation when a jump link is clicked while Settings is held on display', async () => {
    hub.api = new MockApi('normal', 0);
    c = mount(Settings, { target });
    const jump = target.querySelector<HTMLElement>('nav.jump')!;
    await until(() => jump.textContent?.includes('About'), 'the About section');
    const release = router.holdWhile((to) => to.name !== 'settings');
    try {
      router.go('/runs');
      expect(router.route.name).toBe('settings');
      expect(router.location.name).toBe('runs');
      expect(location.hash).toBe('#/runs');

      const go = vi.spyOn(router, 'go');
      try {
        [...jump.querySelectorAll('a')].find((a) => a.textContent === 'About')!.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, detail: 1 }));
        flushSync();
        expect(go).not.toHaveBeenCalled();
        expect(router.location.name).toBe('runs');
        expect(location.hash).toBe('#/runs');
      } finally {
        go.mockRestore();
      }
    } finally {
      release();
    }
  });

  it('keeps a chosen section current until the person scrolls, and opens on ?section=', async () => {
    hub.api = new MockApi('normal', 0);
    router.route = parse('#/settings?section=system');
    c = mount(Settings, { target });
    const jump = target.querySelector<HTMLElement>('nav.jump')!;
    // The URL's section, once the server's capabilities add it.
    await until(() => jump.querySelector('a.on')?.textContent === 'System', 'System chosen by the URL');
    expect(jump.querySelector('a.on')?.getAttribute('href')).toBe('#/settings?section=system');
    // A scroll the page makes itself (a resize, a full-page capture) keeps it.
    window.dispatchEvent(new Event('scroll'));
    await new Promise((r) => requestAnimationFrame(r));
    flushSync();
    expect(jump.querySelector('a.on')?.textContent).toBe('System');
    // The person's own scroll releases it to the geometry (no layout here: every top is 0, so the last section).
    window.dispatchEvent(new Event('wheel'));
    await until(() => jump.querySelector('a.on')?.textContent !== 'System', 'the scroll-spy again');
  });

  it('opens the same ?section= again after the bare page, focusing its heading without a ring', async () => {
    // Round 26 review: #/settings?section=system, then #/settings, then the same section URL again stayed at the top.
    hub.api = new MockApi('normal', 0);
    const jumps: string[] = [];
    const real = HTMLElement.prototype.scrollIntoView;
    HTMLElement.prototype.scrollIntoView = function (this: HTMLElement) {
      jumps.push(this.id);
    };
    const focus = vi.spyOn(HTMLElement.prototype, 'focus');
    try {
      router.route = parse('#/settings?section=system');
      c = mount(Settings, { target });
      await until(() => jumps.length === 1, 'the first jump');
      expect(jumps).toEqual(['system']);
      // Opened from the URL: on its heading, no focus ring (nothing was tabbed to).
      const heading = target.querySelector<HTMLElement>('#system-title')!;
      expect(document.activeElement).toBe(heading);
      expect(focus.mock.contexts.at(-1)).toBe(heading);
      expect(focus).toHaveBeenLastCalledWith({ preventScroll: true, focusVisible: false });
      heading.blur();
      router.route = parse('#/settings');
      flushSync();
      await new Promise((r) => requestAnimationFrame(r));
      expect(jumps).toHaveLength(1);
      router.route = parse('#/settings?section=system');
      await until(() => jumps.length === 2, 'the second jump to the same section');
      expect(jumps).toEqual(['system', 'system']);
      expect(document.activeElement).toBe(heading);
    } finally {
      HTMLElement.prototype.scrollIntoView = real;
      focus.mockRestore();
    }
  });

  // v-shell N6: the palette's "Units" from Runs opened Settings at the top: the new page's route focus (App, a frame
  // after the page renders) went to the title and scrolled to the top after the section's own jump.
  it('lands on the section ?section= names when it is opened as a new page', async () => {
    hub.api = new MockApi('normal', 0);
    const top = vi.spyOn(window, 'scrollTo').mockImplementation(() => {});
    router.route = parse('#/settings?section=units');
    c = mount(Settings, { target });
    flushSync();
    const heading = target.querySelector<HTMLElement>('#units-title')!;
    expect(heading.hasAttribute('data-route-landing')).toBe(true);
    // What App does for a new page.
    expect(focusRoute(target)).toBe(heading);
    expect(document.activeElement).toBe(heading);
    expect(top).not.toHaveBeenCalled();
    top.mockRestore();
  });

  // review studio-2 R5: Back to a Settings entry opened on ?section=units, scrolled on since, returns to that scroll;
  // the section's hold (it keeps a section at the top while what is above it loads) is only for a new navigation, so a
  // column that grows late, after the restore has let go, does not take the page back to Units.
  it('gives the position Back restores precedence over the section the URL names, also when content above comes late', async () => {
    hub.api = new MockApi('normal', 0);
    vi.useFakeTimers();
    sessionStorage.clear();
    const tab = tabHistory();
    router.go('/runs', { replace: true });
    let y = 0;
    const scroll = (to: number) => {
      y = to;
      window.dispatchEvent(new Event('scroll'));
    };
    vi.spyOn(window, 'scrollY', 'get').mockImplementation(() => y);
    vi.spyOn(window, 'scrollTo').mockImplementation(((_x: number, to: number) => scroll(to)) as typeof window.scrollTo);
    vi.spyOn(document.documentElement, 'scrollHeight', 'get').mockImplementation(() => 5000);
    // Units sits at 800; a jump to it scrolls there.
    const jumps: string[] = [];
    const real = HTMLElement.prototype.scrollIntoView;
    HTMLElement.prototype.scrollIntoView = function (this: HTMLElement) {
      jumps.push(this.id);
      scroll(800);
    };
    // The column's size changes when the test says so (what is above Units arriving).
    const observers = new Set<() => void>();
    vi.stubGlobal(
      'ResizeObserver',
      class {
        #cb: () => void;
        constructor(cb: () => void) {
          this.#cb = cb;
        }
        observe() {
          observers.add(this.#cb);
        }
        disconnect() {
          observers.delete(this.#cb);
        }
      },
    );
    const grow = () => [...observers].forEach((cb) => cb());
    const uninstall = installEntryMemory();
    try {
      // A new navigation to the section: it is jumped to, and held while the column grows.
      router.go('/settings?section=units');
      c = mount(Settings, { target });
      await vi.advanceTimersByTimeAsync(50);
      flushSync();
      expect(jumps).toEqual(['units']);
      grow();
      expect(jumps).toEqual(['units', 'units']);
      // The person scrolls on, down to About, and leaves for Runs.
      window.dispatchEvent(new Event('wheel'));
      scroll(2000);
      router.go('/runs');
      unmount(c);
      c = undefined;

      tab.back();
      await vi.advanceTimersByTimeAsync(1);
      expect(router.route.query.get('section')).toBe('units');
      c = mount(Settings, { target });
      // The restore lets go once the height has been steady for its settling window.
      await vi.advanceTimersByTimeAsync(1000);
      flushSync();
      expect([restoringScroll(), y]).toEqual([false, 2000]);
      // What is above Units arrives after that: the page stays where the person was.
      grow();
      await vi.advanceTimersByTimeAsync(50);
      expect([jumps, y]).toEqual([['units', 'units'], 2000]);
    } finally {
      uninstall();
      HTMLElement.prototype.scrollIntoView = real;
      vi.unstubAllGlobals();
      vi.restoreAllMocks();
      vi.useRealTimers();
    }
  });

  // design/UNIFIED.md §4: Theme is a row with a Dark / Light / System segmented control (the preview cards went).
  it('sets the theme from one row: Dark, Light or System, the current one checked, and says what System follows', () => {
    hub.api = new MockApi('normal', 0);
    prefs.theme = 'light';
    c = mount(Settings, { target });
    flushSync();
    const group = target.querySelector<HTMLElement>('#appearance [role="radiogroup"][aria-label="Theme"]')!;
    const radios = () => [...group.querySelectorAll<HTMLElement>('[role="radio"]')];
    expect(radios().map((r) => r.textContent?.trim())).toEqual(['Dark', 'Light', 'System']);
    expect(radios().map((r) => r.getAttribute('aria-checked'))).toEqual(['false', 'true', 'false']);
    expect(target.querySelector('#appearance .s-name')?.textContent).toBe('Theme');
    expect(target.querySelector('#appearance')?.textContent).toContain('Dark or light. The top bar and the command palette switch it too.');
    radios()[2]!.click();
    flushSync();
    expect(prefs.theme).toBe('system');
    expect(target.querySelector('#appearance')?.textContent).toContain(`Follows your operating system, currently ${prefs.resolvedTheme}.`);
    // Changed elsewhere (the top bar, the palette): the control follows.
    prefs.theme = 'dark';
    flushSync();
    expect(radios().map((r) => r.getAttribute('aria-checked'))).toEqual(['true', 'false', 'false']);
    expect(target.querySelector('input[name="theme"], .theme .preview')).toBeNull();
  });

  it('gives each fix one remedy: the shell, the command on one unwrapped line, the driver report folded', async () => {
    hub.api = new MockApi('no-tensor-cores', 0);
    c = mount(Settings, { target });
    const fixes = await until(() => {
      const f = [...target.querySelectorAll('#system .cap-fix')];
      return f.length ? f : null;
    }, 'the System panel');
    const tc = fixes.find((f) => f.textContent?.includes('--tensor-cores'))!;
    expect(tc.querySelector('pre.line')!.textContent).toBe("$env:FOLO_GPU_BACKEND='vulkan'; folo serve --tensor-cores");
    expect(tc.querySelector('button[aria-label="Copy command"]')).not.toBeNull();
    const report = target.querySelector<HTMLDetailsElement>('#system details.report')!;
    expect(report.open).toBe(false);
    // Copy sits on the heading's row, outside the report, which takes the full width: one block per record, the
    // newlines kept in its text for a copied selection.
    expect(report.querySelector('.copy-at button[aria-label="Copy driver report"]')).not.toBeNull();
    const pre = report.querySelector('pre')!;
    expect(pre.querySelector('button')).toBeNull();
    const recs = [...pre.querySelectorAll('.rec')];
    expect(recs.length).toBeGreaterThan(1);
    expect(pre.textContent).toBe(recs.map((r) => r.textContent).join('\n'));
  });
});

describe("Settings › System: FOLO's GPU memory (gpu_pool)", () => {
  it('shows what FOLO holds on the GPU, in use, cached and the peak, in the one size format', async () => {
    hub.api = new MockApi('normal', 0);
    prefs.units = 'binary';
    c = mount(Settings, { target });
    const row = await until(() => [...target.querySelectorAll('#system .s-row')].find((r) => r.querySelector('.s-name')?.textContent === "FOLO's GPU memory"), 'the GPU memory row');
    // One meta item each (ui/Meta draws the separators); under 1 GiB the format steps down to MiB, as everywhere.
    expect([...row.querySelectorAll('.pool .ui-meta > span')].map((x) => x.textContent)).toEqual([
      '4.6 GiB in use',
      '901.1 MiB cached',
      'peak 5.3 GiB',
    ]);
  });

  it("shows the units another tab picked in the control too, not only in the sizes (review of #220)", async () => {
    hub.api = new MockApi('normal', 0);
    prefs.units = 'binary';
    c = mount(Settings, { target });
    await until(() => target.querySelector('#system .pool'), 'the GPU memory row');
    const picked = () => target.querySelector('#units [aria-checked="true"], #units [aria-pressed="true"]')?.textContent;
    expect(picked()).toMatch(/^Binary/);
    localStorage.setItem('folo.prefs', JSON.stringify({ ...JSON.parse(localStorage.getItem('folo.prefs')!), units: 'metric' }));
    window.dispatchEvent(new StorageEvent('storage', { key: 'folo.prefs' }));
    flushSync();
    expect(picked()).toMatch(/^Metric/);
    expect(target.querySelector('#system .pool')?.textContent).toContain('GB');
    prefs.units = 'binary';
  });

  it('shows nothing where the server reports no pool', async () => {
    hub.api = new MockApi('empty', 0);
    c = mount(Settings, { target });
    await until(() => target.querySelector('#system'), 'the System panel');
    expect(target.textContent).not.toContain("FOLO's GPU memory");
  });
});

describe('Settings › Throughput', () => {
  it('lists f16 with and without tensor cores as two rows, each saying which', async () => {
    const api = new MockApi('normal', 0);
    const table = await api.throughput();
    const row = table.inference.find((r) => r.device === table.device && r.precision === 'f32')!;
    const f16 = { ...row, arch: 'folo-n' as const, imgsz: 640, precision: 'f16' as const, source: 'measured' as const, samples: 3 };
    api.throughput = async () => ({ ...table, training: [], inference: [{ ...f16, tensor_cores: false }, { ...f16, tensor_cores: true, latency_ms: 2.5 }] });
    hub.api = api;
    c = mount(Throughput, { target });
    // Only predictions measured: the panel opens on Inference.
    const rows = await until(() => {
      const r = [...target.querySelectorAll('table[aria-label="Predictions, end to end"] .tp-row')];
      return r.length ? r : null;
    }, 'the prediction rows');
    const col = (name: string) => [...target.querySelectorAll('#throughput thead th')].findIndex((th) => th.textContent?.trim() === name);
    const cells = (name: string) => rows.map((r) => r.querySelectorAll('td')[col(name)]?.textContent?.trim());
    expect(cells('Precision')).toEqual(['f16', 'f16 · tensor cores']);
    expect(cells('Latency')).toEqual([expect.any(String), '2.50 ms']);
  });

  it('clears the previous server’s System, GPU memory, and Throughput panels after switching to a refused server (playtest R17)', async () => {
    vi.useFakeTimers();
    const prevUrl = prefs.serverUrl;
    const prevApi = hub.api;
    const mockA = new MockApi('normal', 0);
    const sysA = await mockA.system();
    const tpA = await mockA.throughput();
    const HEALTH = { status: 'ok', version: '0.1.0', api: 'v1', mock: false };

    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url.startsWith('http://127.0.0.1:17879')) {
        if (url.endsWith('/api/health')) return Response.json(HEALTH);
        return Response.json({ error: { code: 'unauthorized', message: 'this server needs a token', hint: null } }, { status: 401 });
      }
      if (url.endsWith('/api/health')) return Response.json(HEALTH);
      if (init?.method === 'HEAD') return new Response(null, { status: 200 });
      if (url.endsWith('/api/system')) return Response.json(sysA);
      if (url.endsWith('/api/system/throughput')) return Response.json(tpA);
      if (url.endsWith('/api/cloud/providers')) return Response.json({ providers: [] });
      if (url.endsWith('/api/cloud/connections')) return Response.json({ connections: [] });
      if (url.endsWith('/api/cloud/instances')) return Response.json({ instances: [] });
      if (url.endsWith('/api/runs')) return Response.json({ runs: [] });
      return Response.json({});
    });

    try {
      prefs.serverUrl = 'http://127.0.0.1:17878';
      await hub.connect();
      c = mount(Settings, { target });
      await vi.advanceTimersByTimeAsync(0);
      flushSync();
      expect(target.querySelector('#system')).not.toBeNull();
      expect(target.querySelector('#system .pool')).not.toBeNull();
      expect(target.querySelector('#throughput')).not.toBeNull();

      // Switch Server URL to the token-walled server on 17879 and click Save and reconnect.
      const input = target.querySelector<HTMLInputElement>('#server-url')!;
      input.value = 'http://127.0.0.1:17879';
      input.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
      const saveBtn = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Save and reconnect'))!;
      saveBtn.click();
      await vi.advanceTimersByTimeAsync(0);
      flushSync();

      expect(hub.mode).toBe('connecting');
      expect(target.querySelector('#connection .s-title')?.textContent).toContain('Not connected');
      // Server A's GPU readings, System capabilities, Cloud, and Throughput must be gone.
      expect(target.querySelector('#system')).toBeNull();
      expect(target.querySelector('#throughput')).toBeNull();
      expect(target.querySelector('#cloud')).toBeNull();
      expect(target.querySelector('#diagnostics')?.textContent).toContain('Not loaded');
    } finally {
      prefs.serverUrl = prevUrl;
      hub.api = prevApi;
      vi.unstubAllGlobals();
      vi.useRealTimers();
      hub.mode = 'mock';
    }
  });
});
