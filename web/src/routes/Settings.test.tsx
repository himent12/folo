// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { prefs } from '$lib/prefs';
import { parse, router } from '$lib/router';
import { focusRoute } from '$lib/shell/routeFocus';
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

  it('jumps to a section from the strip below 1024 px: every section listed, focus moved onto its heading', async () => {
    hub.api = new MockApi('normal', 0);
    c = mount(Settings, { target });
    const jump = target.querySelector<HTMLElement>('nav.jump')!;
    await until(() => jump.textContent?.includes('System'), 'the System section');
    expect([...jump.querySelectorAll('a')].map((a) => a.textContent)).toEqual(['Connection', 'Cloud', 'Appearance', 'Units', 'Keyboard shortcuts', 'System', 'About']);
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

  it('says which theme is current in its own card and in its radio’s name', () => {
    hub.api = new MockApi('normal', 0);
    prefs.theme = 'light';
    c = mount(Settings, { target });
    flushSync();
    const radios = [...target.querySelectorAll<HTMLInputElement>('input[name="theme"]')];
    expect(radios.map((r) => r.getAttribute('aria-label'))).toEqual(['Dark', 'Light, current', 'System']);
    // "Current" sits inside the chosen card's label.
    expect(target.querySelector('.theme .current')!.closest('label')?.querySelector('input')?.value).toBe('light');
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
    const row = await until(() => [...target.querySelectorAll('#system .setting')].find((r) => r.querySelector('p')?.textContent === "FOLO's GPU memory"), 'the GPU memory row');
    // One meta item each (ui/Meta draws the separators); under 1 GiB the format steps down to MiB, as everywhere.
    expect([...row.querySelectorAll('.pool .ui-meta > span')].map((x) => x.textContent)).toEqual([
      '4.6 GiB in use',
      '901.1 MiB cached',
      'peak 5.3 GiB',
    ]);
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
    const rows = await until(() => {
      const r = [...target.querySelectorAll('ul[aria-label="Predictions, end to end"] .tp-row')];
      return r.length ? r : null;
    }, 'the prediction rows');
    expect(rows.map((r) => r.querySelector('.tp-set')?.textContent)).toEqual(['640 px · f16', '640 px · f16 · tensor cores']);
    expect(rows.map((r) => r.querySelector('.tp-val p')?.textContent)).toEqual([expect.any(String), '2.50 ms']);
  });
});
