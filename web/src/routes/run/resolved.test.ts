import { describe, expect, it } from 'vitest';
import type { ResolvedTrain, TrainConfig } from '$lib/api/types';
import { resolveTrain } from '$lib/api/mock/runs';
import { accumulationText, batchSetting, closeMosaicSetting, decaySetting, effectiveCloseMosaic, lrSetting, momentumSetting, optimizerSetting, settingText, effectiveBatchSetting, warmupSetting } from './resolved';

const config = { optimizer: 'auto', lr0: 0.01, batch: 8, epochs: 30, augment: { close_mosaic: 10 } } as TrainConfig;

describe('the trainer resolution the mock reports (folo-vision resolved_train)', () => {
  it('resolves auto as a new run does (rule 5): MuSGD at 0.01, momentum 0.937, bias warmup from 0.1, no warmup floor', () => {
    // 1000 images, batch 8, 30 epochs, 3 classes; 16 optimizer steps an epoch (folo-vision's test); 3 warmup epochs.
    const r = resolveTrain({ ...config, imgsz: 640 } as TrainConfig, 1000, 3, 16);
    expect(r).toMatchObject({ optimizer: 'musgd', lr0: 0.01, momentum: 0.937, betas: null, warmup_bias_lr: 0.1 });
    expect(r).toMatchObject({ batch: 8, accumulate: 8, nbs: 64, optimizer_steps_per_epoch: 16, warmup_iters: 48, lrf: 0.01, cos_lr: false });
    expect(r.weight_decay).toBeCloseTo(0.0005, 12);
  });

  it('reports close mosaic after the cap at half the run, as the server does', () => {
    expect(resolveTrain(config, 1000, 3, 16).close_mosaic).toBe(10);
    const short = { ...config, epochs: 15 } as TrainConfig;
    expect(resolveTrain(short, 1000, 3, 16).close_mosaic).toBe(7);
    expect(effectiveCloseMosaic({ config: short, resolved: resolveTrain(short, 1000, 3, 16) })).toBe(7);
  });

  it('keeps a named optimizer at the requested LR with the recipe momentum, SGD without betas', () => {
    const r = resolveTrain({ ...config, optimizer: 'sgd', lr0: 0.02 } as TrainConfig, 1000, 3, 16);
    expect(r).toMatchObject({ optimizer: 'sgd', lr0: 0.02, momentum: 0.937, betas: null });
  });
});

describe('Run › Config wording', () => {
  const r: ResolvedTrain = resolveTrain({ ...config } as TrainConfig, 1000, 3, 16);

  it('shows what the run trains with, the request beside it', () => {
    expect(optimizerSetting(config, r)).toEqual({ value: 'MuSGD', note: '(auto)' });
    expect(settingText(lrSetting(config, r))).toBe('0.01');
    expect(settingText(lrSetting({ ...config, lr0: 0.002 } as TrainConfig, r))).toBe('0.01 (requested 0.002)');
    expect(batchSetting(config, r).value).toBe('8');
    expect(settingText(effectiveBatchSetting(r))).toBe('64 (8 × 8)');
    expect(accumulationText(4, 16)).toBe('4 batches per optimizer step · effective batch 64');
    expect(settingText(warmupSetting(r))).toBe('48 optimizer steps (3 epochs)');
    expect(settingText(warmupSetting({ ...r, warmup_iters: 100 }))).toBe('100 optimizer steps (6.3 epochs)');
    expect(decaySetting(r).value).toBe('Linear to 1%');
    expect(momentumSetting(r)).toEqual(['Momentum', { value: '0.937' }]);
    expect(momentumSetting(resolveTrain({ ...config, optimizer: 'adamw' } as TrainConfig, 1000, 3, 16))).toEqual(['Betas', { value: '0.937 · 0.999' }]);
  });

  it('names a requested optimizer without "(auto)", and an LR that did not change once', () => {
    const sgd = { ...config, optimizer: 'sgd' } as TrainConfig;
    const rs = resolveTrain(sgd, 1000, 3, 16);
    expect(optimizerSetting(sgd, rs)).toEqual({ value: 'SGD', note: undefined });
    expect(lrSetting(sgd, rs)).toEqual({ value: '0.01' });
    expect(momentumSetting(rs)).toEqual(['Momentum', { value: '0.937' }]);
  });

  it('keeps the request as it is without a resolution (queued runs, older servers)', () => {
    // A value cell holds one short value; what Auto picks goes under the label.
    expect(optimizerSetting(config, null)).toEqual({ value: 'Auto', about: 'MuSGD for every run' });
    expect(lrSetting(config, undefined)).toEqual({ value: '0.01' });
    expect(batchSetting(config, null)).toEqual({ value: '8' });
  });
});

describe('close mosaic as the trainer runs it (DOGFOOD-3 S2)', () => {
  const run = (epochs: number, close: number, resolved: Partial<ResolvedTrain> | null = null) => ({
    config: { epochs, augment: { close_mosaic: close } } as TrainConfig,
    resolved: resolved as ResolvedTrain | null,
  });

  it('caps the request at half the run: 15 epochs asked for 10 close for 7', () => {
    expect(effectiveCloseMosaic(run(15, 10))).toBe(7);
    expect(effectiveCloseMosaic(run(100, 10))).toBe(10);
    expect(effectiveCloseMosaic(run(1, 10))).toBe(0);
    expect(effectiveCloseMosaic(run(20, 0))).toBe(0);
  });

  it("prefers the server's figure where it sends one", () => {
    expect(effectiveCloseMosaic(run(15, 10, { close_mosaic: 6 }))).toBe(6);
    expect(effectiveCloseMosaic(run(15, 10, {}))).toBe(7);
  });

  it('says the request where the trainer capped it', () => {
    expect(settingText(closeMosaicSetting(run(15, 10)))).toBe('last 7 epochs (requested 10)');
    expect(settingText(closeMosaicSetting(run(100, 10)))).toBe('last 10 epochs');
    expect(settingText(closeMosaicSetting(run(2, 1)))).toBe('last 1 epoch');
    expect(settingText(closeMosaicSetting(run(1, 1)))).toBe('never (requested 1)');
    expect(settingText(closeMosaicSetting(run(20, 0)))).toBe('never');
  });
});
