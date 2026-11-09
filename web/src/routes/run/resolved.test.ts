import { describe, expect, it } from 'vitest';
import type { ResolvedTrain, TrainConfig } from '$lib/api/types';
import { resolveTrain } from '$lib/api/mock/runs';
import { batchSetting, decaySetting, lrSetting, momentumSetting, optimizerSetting, settingText, warmupSetting } from './resolved';

const config = { optimizer: 'auto', lr0: 0.01, batch: 8, epochs: 30 } as TrainConfig;

describe('the trainer resolution the mock reports (folo-vision resolved_train)', () => {
  it('matches the trainer test: auto on a short run is AdamW at 0.002 · 5 / (4 + 3)', () => {
    // 1000 images, batch 8, 30 epochs, 3 classes; 16 optimizer steps an epoch (folo-vision's test).
    const r = resolveTrain({ ...config, imgsz: 640 } as TrainConfig, 1000, 3, 16);
    expect(r).toMatchObject({ optimizer: 'adamw', lr0: 0.001429, momentum: 0.9, betas: [0.9, 0.999], warmup_bias_lr: 0 });
    expect(r).toMatchObject({ batch: 8, accumulate: 8, nbs: 64, optimizer_steps_per_epoch: 16, warmup_iters: 100, lrf: 0.01, cos_lr: false });
    expect(r.weight_decay).toBeCloseTo(0.0005, 12);
  });

  it('keeps a named optimizer at the requested LR with the recipe momentum, SGD without betas', () => {
    const r = resolveTrain({ ...config, optimizer: 'sgd', lr0: 0.02 } as TrainConfig, 1000, 3, 16);
    expect(r).toMatchObject({ optimizer: 'sgd', lr0: 0.02, momentum: 0.937, betas: null });
  });
});

describe('Run › Config wording', () => {
  const r: ResolvedTrain = resolveTrain({ ...config } as TrainConfig, 1000, 3, 16);

  it('shows what the run trains with, the request beside it', () => {
    expect(optimizerSetting(config, r)).toEqual({ value: 'AdamW', ident: true, note: '(auto)' });
    expect(settingText(lrSetting(config, r))).toBe('0.001429 (requested 0.01)');
    expect(batchSetting(config, r).value).toBe('8 × 8 = 64');
    expect(settingText(warmupSetting(r))).toBe('100 steps (6.3 epochs)');
    expect(decaySetting(r).value).toBe('Linear to 1%');
    expect(momentumSetting(r)).toEqual(['Betas', { value: '0.9 · 0.999' }]);
  });

  it('names a requested optimizer without "(auto)", and an LR that did not change once', () => {
    const sgd = { ...config, optimizer: 'sgd' } as TrainConfig;
    const rs = resolveTrain(sgd, 1000, 3, 16);
    expect(optimizerSetting(sgd, rs)).toEqual({ value: 'SGD', ident: true, note: undefined });
    expect(lrSetting(sgd, rs)).toEqual({ value: '0.01' });
    expect(momentumSetting(rs)).toEqual(['Momentum', { value: '0.937' }]);
  });

  it('keeps the request as it is without a resolution (queued runs, older servers)', () => {
    expect(optimizerSetting(config, null)).toEqual({ value: 'auto', ident: false });
    expect(lrSetting(config, undefined)).toEqual({ value: '0.01' });
    expect(batchSetting(config, null)).toEqual({ value: '8' });
  });
});
