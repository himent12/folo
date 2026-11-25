// @vitest-environment happy-dom
// Singular/plural optimizer step wording in New run's EstimateFigures schedule and short-run warnings.
import { flush as flushSync, signal } from 'fw';
import { afterEach, describe, expect, it } from 'vitest';
import { render } from '../../../framework/testing';
import EstimateFigures, { type EstimateFiguresProps } from './EstimateFigures';

let target: HTMLElement | undefined;
let mounted: { unmount(): void } | undefined;

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  target?.remove();
  target = undefined;
});

function mountFigures(stepsPerEpoch: number, totalSteps: number, epochCount: number, warmup = 0) {
  target = document.createElement('div');
  document.body.appendChild(target);
  const stepsOpen = signal(false);
  const shortWhy = signal(true);
  const epochs = signal(epochCount);
  const props: EstimateFiguresProps = {
    where: 'rail',
    blocked: false,
    estReady: true,
    sched: { stepsPerEpoch, accumulate: 4, steps: totalSteps, warmup },
    batchN: 16,
    est: {
      stepsPerEpoch: Math.max(1, stepsPerEpoch),
      imgsPerS: 100,
      epochSeconds: 1,
      totalSeconds: 7,
      lowSeconds: 5,
      highSeconds: 10,
      vramBytes: 1024 ** 3,
      phases: [{ phase: 'mosaic', epochs: Math.max(1, epochCount), train_imgs_per_s: 100, epoch_s: 1 }],
      valImages: 4,
      valImgsPerS: 200,
      startupS: 6,
    },
    stepsOpen: stepsOpen.get(),
    shown: {
      lo: 5,
      hi: 10,
      rough: false,
      basis: 'Reference',
      how: 'Reference timing.',
      updated: null,
      phases: [{ phase: 'mosaic', epochs: Math.max(1, epochCount), train_imgs_per_s: 100, epoch_s: 1 }],
      val: { images: 4, rate: 200, reference: true },
      startup: 6,
      vram: 1024 ** 3,
      vramMeasured: false,
    },
    estStale: false,
    estimating: false,
    memShown: false,
    vram: 0,
    onCloud: false,
    cloudPick: { offer: null, hours: 0 } as unknown as EstimateFiguresProps['cloudPick'],
    cloudCost: null,
    gpuMem: null,
    scale: 1,
    onCpu: false,
    vramLevel: null,
    vramPct: 0,
    noFit: false,
    fix: null,
    fixLoading: false,
    short: {
      stepsPerEpoch,
      accumulate: 4,
      steps: totalSteps,
      warmup,
      scratch: true,
      warmupHeavy: warmup * 2 > totalSteps,
      epochs: 200,
    },
    shortWhy: shortWhy.get(),
    epochs: epochs.get(),
  };
  mounted = render(
    () => <EstimateFigures {...props} bind:stepsOpen={stepsOpen} bind:shortWhy={shortWhy} bind:epochs={epochs} />,
    target,
  );
  flushSync();
  return target;
}

describe('EstimateFigures optimizer step pluralization', () => {
  it('uses singular "optimizer step" for 1 and plural "optimizer steps" for 0 and 2+', () => {
    for (const [spe, total, ep, expectedSched, expectedCallout, expectedDetail] of [
      [0, 0, 0, '0 optimizer steps / epoch', 'Too short to learn: 0 optimizer steps.', '0 epochs × 0 optimizer steps.'],
      [1, 1, 1, '1 optimizer step / epoch', 'Too short to learn: 1 optimizer step.', '1 epoch × 1 optimizer step.'],
      [2, 4, 2, '2 optimizer steps / epoch', 'Too short to learn: 4 optimizer steps.', '2 epochs × 2 optimizer steps.'],
    ] as const) {
      const el = mountFigures(spe, total, ep);
      expect(el.querySelector('dd.sched button.steps')?.textContent).toContain(expectedSched);
      expect(el.querySelector('.short p.callout span')?.textContent?.trim()).toBe(expectedCallout);
      expect(el.querySelector('.short-why p')?.textContent?.replace(/\s+/g, ' ').trim()).toContain(expectedDetail);
      mounted?.unmount();
      mounted = undefined;
      target?.remove();
      target = undefined;
    }
  });
});
