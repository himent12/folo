// @vitest-environment happy-dom
// Regression tests for Live Studio Playtest Round 27 findings:
// 1. Playground Confidence slider preserves 3-4 decimal calibrated global confidence in `<output>` and `aria-valuetext`
//    across all 4 slider homes (desktop rail, narrow side panel, tablet/phone pinned bar, phone sheet).
// 2. BenchCompare SideBySide wraps panel export actions for 44px phone tap targets, and `compareMarkdown` guards non-finite condition values.
// 3. Model Detail clears cached `serverCal` when `hub.epoch` increments (in-place server switch).
import { flush as flushSync, signal } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Bench, BenchEntry } from '$lib/api/types';
import { parse, router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import { compareMarkdown } from '../bench/export';
import ModelPage from '../Model';
import Playground from '../Playground';
import { consumeThresholdHandoff, session } from './session';
import SidePanels from './SidePanels';
import { MoreBody, RailThresholds } from './Thresholds';

let target: HTMLElement;
let view: Rendered | undefined;

async function until<T>(get: () => T | null | undefined | false, what = 'condition'): Promise<T> {
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

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  toast.items = [];
  session.set(null);
  consumeThresholdHandoff(null);
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  view?.unmount();
  view = undefined;
  target.remove();
  session.set(null);
  consumeThresholdHandoff(null);
  toast.items = [];
  vi.restoreAllMocks();
});

describe('Playtest R27 regressions', () => {
  it('displays exact calibrated 3-decimal global confidence in output and aria-valuetext across all Playground Confidence sliders', () => {
    const conf = signal(0.453);
    const iou = signal(0.7);
    const maxDet = signal(300);

    // 1. Desktop rail slider
    view = render(() => <RailThresholds bind:conf={conf} bind:iou={iou} bind:maxDet={maxDet} o2o={false} />, target);
    flushSync();
    expect(target.querySelector('output.val')?.textContent?.trim()).toBe('0.453');
    expect(target.querySelector('input[type="range"]')?.getAttribute('aria-valuetext')).toBe('0.453');

    // Standard 2-decimal values continue formatting cleanly as 0.25 / 0.40
    conf.set(0.4);
    flushSync();
    expect(target.querySelector('output.val')?.textContent?.trim()).toBe('0.40');

    // 2. Phone sheet slider (`MoreBody`)
    view.unmount();
    conf.set(0.384);
    view = render(
      () => <MoreBody tablet={false} bind:conf={conf} bind:iou={iou} bind:maxDet={maxDet} toggles={() => <span />} />,
      target,
    );
    flushSync();
    expect(target.querySelector('output.val')?.textContent?.trim()).toBe('0.384');
    expect(target.querySelector('input[type="range"]')?.getAttribute('aria-valuetext')).toBe('0.384');

    // 3. Narrow side panel slider (`SidePanels`)
    view.unmount();
    conf.set(0.005);
    view = render(
      () => (
        <SidePanels
          source={null}
          result={null}
          counts={[]}
          pred={null}
          where="server"
          bind:conf={conf}
          resetButton={() => <span />}
          secondarySliders={() => <span />}
          displayToggles={() => <span />}
          showAllButton={() => <span />}
          classList={() => <span />}
          timingBody={() => <span />}
        />
      ),
      target,
    );
    flushSync();
    expect(target.querySelector('output.val')?.textContent?.trim()).toBe('0.005');
    expect(target.querySelector('input[type="range"]')?.getAttribute('aria-valuetext')).toBe('0.005');
  });

  it('preserves exact 3-decimal calibrated global confidence end-to-end (rail + pinned ImagePanel bar) when handing off from Model Detail to Playground', async () => {
    vi.spyOn(router, 'go').mockImplementation((path) => {
      router.route = parse(`#${path}`);
    });
    const origVal = hub.api.valModel!.bind(hub.api);
    vi.spyOn(hub.api, 'valModel').mockImplementation(async (body, sig) => {
      const res = await origVal(body, sig);
      return {
        ...res,
        thresholds: {
          ...res.thresholds!,
          global: { ...res.thresholds!.global, best_conf: 0.453 },
        },
      };
    });

    router.route = parse('#/models/m_fast_n');
    view = render(() => <ModelPage />, target);
    const calBtn = await until(() => target.querySelector<HTMLButtonElement>('[data-calibrate-thresholds]'), 'calibrate button');
    calBtn.click();
    await until(() => target.querySelector('[data-model-thresholds="calibrated"]'), 'calibrated state');

    target.querySelector<HTMLButtonElement>('[data-apply-playground]')!.click();
    flushSync();
    view.unmount();

    // Seed a dataset image in session so ImagePanel's pinned `.qconf` slider renders too.
    const ds = (await hub.api.datasets())[0]!;
    const img = (await hub.api.datasetImages(ds.id, { split: 'val', limit: 1 })).images[0]!;
    session.set({
      source: { kind: 'dataset', ds, img, at: 0 },
      pred: null,
      runOn: 'server',
      conf: 0.25,
      iou: 0.7,
      maxDet: 300,
      labels: true,
      scores: true,
      truth: false,
      hidden: [],
    });

    view = render(() => <Playground />, target);
    const qconfOut = await until(() => target.querySelector('.qconf output.val'), 'pinned qconf readout');
    expect(qconfOut.textContent?.trim()).toBe('0.453');
    expect(target.querySelector('.qconf input[type="range"]')?.getAttribute('aria-valuetext')).toBe('0.453');
    const allOutputs = [...target.querySelectorAll('output.val')].map((el) => el.textContent?.trim());
    expect(allOutputs.filter((v) => v === '0.453').length).toBeGreaterThanOrEqual(2);
  });

  it('clears cached server calibration on in-place hub.epoch switch and guards non-finite condition cells in compareMarkdown', async () => {
    router.route = parse('#/models/m_fast_n');
    view = render(() => <ModelPage />, target);
    const calBtn = await until(() => target.querySelector<HTMLButtonElement>('[data-calibrate-thresholds]'), 'calibrate button');
    calBtn.click();
    await until(() => target.querySelector('[data-model-thresholds="calibrated"]'), 'calibrated bar');

    // Simulate in-place server switch (`hub.epoch` increment) -> cached calibration clears cleanly.
    hub.epoch += 1;
    await until(() => target.querySelector('[data-model-thresholds="operating-point"]'), 'reset to operating-point');

    const bench = (await hub.api.bench('bn_3c90')) as Bench;
    const done = (await hub.api.benchEntries('bn_3c90')).filter((e) => e.status === 'done' && e.score);
    const md = compareMarkdown(
      bench,
      [done[0] as BenchEntry, done[1] as BenchEntry],
      [{ key: 'score', label: 'Score', values: [70, 75], fmt: (v) => v.toFixed(1), d: 1 }],
      [false, false],
      () => new Set(),
      [],
      [{ key: 'blur', name: 'Motion blur', values: [0.8, Number.NaN] }],
    );
    expect(md).toContain('| Motion blur | 80% | — | — |');
  });
});
