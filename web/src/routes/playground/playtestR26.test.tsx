// @vitest-environment happy-dom
// Regression tests for Live Studio Playtest Round 26 findings:
// - P2-1: DatasetCurateCli phone-viewport width constraints (grid-template-columns: minmax(0, 1fr) + min-width/max-width)
// - P2-2: PerClassThresholds hidden file input excluded from keyboard Tab order (`tabindex="-1"`, `aria-hidden="true"`)
// - P2-3: ClassList & formatConfThreshold preserve sub-0.01 precision (`0.005` -> `≥0.005`, not `≥0.01`)
// - P2-4: Mock Playground probes real folo-wasm for Track IDs upfront (`trackerBrowserOff`) while keeping demo browser inference enabled
// - P3-1: PerClassThresholds clears import confirmation status note on model change or manual threshold edits
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRoot, flush as flushSync, signal } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { resetWasmProbe } from '$lib/inference/browser';
import type { Dataset, DatasetStats } from '$lib/api/types';
import DatasetCurateCli from '../dataset/DatasetCurateCli';
import ClassList from './ClassList';
import PerClassThresholds from './PerClassThresholds';
import { formatConfThreshold, type PerClassConfMap } from './perClassConf';
import type { RunOn } from './predict';
import { useWasmAvailability } from './useWasmProbe';

let target: HTMLElement | undefined;
let mounted: Rendered | undefined;

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  target?.remove();
  target = undefined;
  hub.features = null;
  hub.serverMock = null;
  resetWasmProbe();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('Playtest R26 regressions', () => {
  it('P2-1: constrains DatasetCurateCli grid/flex tracks to 100% width so Export COCO never overflows phones', () => {
    const ds: Dataset = {
      id: 'ds_1',
      name: 'shapes-playtest',
      path: 'D:/folo-build/test-tmp/playtest-r26/datasets/shapes-playtest',
      format: 'yolo',
      classes: ['circle', 'square'],
      created: '2026-11-27T00:00:00Z',
      splits: { train: 12, val: 6, test: 0 },
    };
    const stats: DatasetStats = {
      boxes: 40,
      empty_images: 0,
      broken_labels: 0,
      class_counts: [20, 20],
      box_size_hist: { edges: [0, 1], counts: [40] },
      boxes_per_image_hist: { edges: [0, 2, 4], counts: [0, 18] },
    };
    target = document.createElement('div');
    document.body.appendChild(target);
    mounted = render(() => <DatasetCurateCli dataset={ds} stats={stats} activeSplit="val" />, target);
    flushSync();

    const exportBtn = [...target.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find(
      (b) => b.textContent?.trim() === 'Export COCO',
    )!;
    exportBtn.click();
    flushSync();

    const src = readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), '../dataset/DatasetCurateCli.tsx'), 'utf8');
    expect(src).toContain('grid-template-columns: minmax(0, 1fr)');
    expect(src).toContain('max-width: 100%');
    expect(target.querySelector('[data-coco-export-options]')).not.toBeNull();
    expect(target.querySelector('button[aria-label="Copy command"]')).not.toBeNull();
  });

  it('P2-2 & P3-1: excludes hidden upload input from Tab order and clears import feedback on modelId switch, global conf change, or manual edits', () => {
    const modelId = signal('m_shapes_a');
    const modelName = signal('best.folo');
    const classes = signal(['circle', 'square', 'triangle']);
    const globalConf = signal(0.25);
    const pcc = signal<PerClassConfMap>({});

    target = document.createElement('div');
    document.body.appendChild(target);
    mounted = render(
      () => (
        <PerClassThresholds
          modelId={modelId.get()}
          modelName={modelName.get()}
          classes={classes.get()}
          globalConf={globalConf.get()}
          perClassConf={pcc.get()}
          onSetClassConf={(id, v) => pcc.set({ ...pcc.peek(), [id]: v })}
          onClearClassConf={(id) => {
            const next = { ...pcc.peek() };
            delete next[id];
            pcc.set(next);
          }}
          onResetAll={() => pcc.set({})}
          onImport={(res) => {
            if (res.globalConf !== null) globalConf.set(res.globalConf);
            pcc.set(res.perClassConf);
          }}
        />
      ),
      target,
    );
    flushSync();

    // P2-2: hidden file input must not trap keyboard focus before the disclosure button.
    const hiddenFile = target.querySelector<HTMLInputElement>('input[type="file"]')!;
    expect(hiddenFile).not.toBeNull();
    expect(hiddenFile.getAttribute('tabindex')).toBe('-1');
    expect(hiddenFile.getAttribute('aria-hidden')).toBe('true');

    // Invalid JSON error clears as soon as user manually edits a per-class slider.
    target.querySelector<HTMLButtonElement>('button.pc-toggle')!.click();
    flushSync();
    target.querySelector<HTMLButtonElement>('[data-per-class-import-btn]')!.click();
    flushSync();
    const ta = target.querySelector<HTMLTextAreaElement>('[data-per-class-import] textarea')!;
    ta.value = '{not valid json';
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    target.querySelector<HTMLButtonElement>('[data-per-class-apply]')!.click();
    flushSync();
    expect(target.querySelector('[data-per-class-error]')).not.toBeNull();

    const circleRange = target.querySelector<HTMLInputElement>('[data-class-threshold-row="0"] input[type="range"]')!;
    circleRange.value = '0.5';
    circleRange.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(target.querySelector('[data-per-class-error]')).toBeNull();

    // Valid import shows status note; changing global confidence externally clears it cleanly.
    ta.value = JSON.stringify({
      global: { best_conf: 0.3 },
      per_class: [
        { class: 0, best_conf: 0.8 },
        { class: 1, best_conf: 0.4 },
      ],
    });
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    target.querySelector<HTMLButtonElement>('[data-per-class-apply]')!.click();
    flushSync();
    expect(target.querySelector('[data-per-class-status]')?.textContent).toBe('Applied global 0.30, 2 class overrides.');

    globalConf.set(0.45);
    flushSync();
    expect(target.querySelector('[data-per-class-status]')).toBeNull();

    // Re-import and switch to a distinct modelId with the SAME name & classes -> clears stale status note.
    target.querySelector<HTMLButtonElement>('[data-per-class-import-btn]')!.click();
    flushSync();
    const ta2 = target.querySelector<HTMLTextAreaElement>('[data-per-class-import] textarea')!;
    ta2.value = JSON.stringify({ global: { best_conf: 0.3 }, per_class: [{ class: 0, best_conf: 0.8 }] });
    ta2.dispatchEvent(new Event('input', { bubbles: true }));
    target.querySelector<HTMLButtonElement>('[data-per-class-apply]')!.click();
    flushSync();
    expect(target.querySelector('[data-per-class-status]')).not.toBeNull();

    modelId.set('m_shapes_b');
    flushSync();
    expect(target.querySelector('[data-per-class-status]')).toBeNull();
  });

  it('P2-3: formats sub-0.01 and 3-4 decimal thresholds accurately in formatConfThreshold and ClassList badges', () => {
    expect(formatConfThreshold(0.4)).toBe('0.40');
    expect(formatConfThreshold(0.25)).toBe('0.25');
    expect(formatConfThreshold(0)).toBe('0.00');
    expect(formatConfThreshold(0.005)).toBe('0.005');
    expect(formatConfThreshold(0.375)).toBe('0.375');
    expect(formatConfThreshold(0.0025)).toBe('0.0025');

    target = document.createElement('div');
    document.body.appendChild(target);
    mounted = render(
      () => (
        <ClassList
          result={{ boxes: [{ class: 0, name: 'circle', score: 0.0075, x1: 0, y1: 0, x2: 10, y2: 10 }], ms: 0.1 }}
          counts={[[0, 1]]}
          pred={null}
          hidden={new Set()}
          classes={['circle']}
          maxCount={1}
          perClassConf={{ 0: 0.005 }}
          toggleClass={() => {}}
        />
      ),
      target,
    );
    flushSync();

    const badge = target.querySelector<HTMLElement>('[data-class-conf-override="0"]')!;
    expect(badge.textContent?.trim()).toBe('≥0.005');
    expect(badge.getAttribute('title')).toBe('Custom confidence threshold 0.005');
  });

  it('P2-4: disables Track IDs upfront in mock mode when folo-wasm assets are absent while keeping demo browser inference on', async () => {
    const api = new MockApi('normal', 0);
    hub.api = api;
    hub.mode = 'mock';
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('<html></html>', { status: 200, headers: { 'content-type': 'text/html' } })),
    );

    const runOn = signal<RunOn>('server');
    const inMedia = signal(false);
    let avail!: ReturnType<typeof useWasmAvailability>;
    const dispose = createRoot((d) => {
      avail = useWasmAvailability(runOn, () => inMedia.get());
      return d;
    });
    flushSync();
    // Still-image demo mode does not probe network unnecessarily.
    expect(globalThis.fetch).not.toHaveBeenCalled();

    // Entering media mode in mock immediately gates Track IDs until real folo-wasm is confirmed, and stays off when absent.
    inMedia.set(true);
    flushSync();
    expect(avail.trackerBrowserOff.get()).toBeTruthy();
    await vi.waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
    flushSync();
    expect(avail.trackerBrowserOff.get()).toBeTruthy();
    // Demo CPU/WebGPU inference controls remain available in mock mode, while real-WASM Track IDs is honestly disabled upfront.
    expect(avail.browserOff.get()).toBeNull();
    dispose();

    // Positive path: when real folo-wasm assets ARE present in mock mode, trackerBrowserOff clears to null.
    resetWasmProbe();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) =>
        new Response(null, {
          status: 200,
          headers: { 'content-type': String(url).endsWith('.wasm') ? 'application/wasm' : 'text/javascript' },
        }),
      ),
    );
    const inMedia2 = signal(true);
    let avail2!: ReturnType<typeof useWasmAvailability>;
    const dispose2 = createRoot((d) => {
      avail2 = useWasmAvailability(runOn, () => inMedia2.get());
      return d;
    });
    await vi.waitFor(() => {
      flushSync();
      expect(avail2.trackerBrowserOff.get()).toBeNull();
    });
    dispose2();
  });
});
