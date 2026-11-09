import { describe, expect, it } from 'vitest';
import type { ResolvedTrain, TrainConfig } from '$lib/api/types';
import { resolveTrain } from '$lib/api/mock/runs';
import { accumulationText, batchSetting, closeMosaicSetting, decaySetting, effectiveCloseMosaic, headSetting, lrSetting, momentumSetting, optimizerSetting, precisionSetting, settingText, stepSetting, warmupSetting } from './resolved';

const config = { optimizer: 'auto', lr0: 0.01, batch: 8, epochs: 30, augment: { close_mosaic: 10 } } as TrainConfig;

describe('the trainer resolution the mock reports (folo-vision resolved_train)', () => {
  it('matches the trainer test: auto on a short run is AdamW at 0.002 · 5 / (4 + 3)', () => {
    // 1000 images, batch 8, 30 epochs, 3 classes; 16 optimizer steps an epoch (folo-vision's test).
    const r = resolveTrain({ ...config, imgsz: 640 } as TrainConfig, 1000, 3, 16);
    expect(r).toMatchObject({ optimizer: 'adamw', lr0: 0.001429, momentum: 0.9, betas: [0.9, 0.999], warmup_bias_lr: 0 });
    expect(r).toMatchObject({ batch: 8, accumulate: 8, nbs: 64, optimizer_steps_per_epoch: 16, warmup_iters: 100, lrf: 0.01, cos_lr: false });
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
    expect(optimizerSetting(config, r)).toEqual({ value: 'AdamW', note: '(auto)' });
    expect(settingText(lrSetting(config, r))).toBe('0.001429 (requested 0.01)');
    expect(batchSetting(config, r).value).toBe('8');
    expect(stepSetting(r).value).toBe('8 batches · effective batch 64');
    expect(accumulationText(4, 16)).toBe('4 batches per optimizer step · effective batch 64');
    expect(settingText(warmupSetting(r))).toBe('100 optimizer steps (6.3 epochs)');
    expect(decaySetting(r).value).toBe('Linear to 1%');
    expect(momentumSetting(r)).toEqual(['Betas', { value: '0.9 · 0.999' }]);
  });

  it('names a requested optimizer without "(auto)", and an LR that did not change once', () => {
    const sgd = { ...config, optimizer: 'sgd' } as TrainConfig;
    const rs = resolveTrain(sgd, 1000, 3, 16);
    expect(optimizerSetting(sgd, rs)).toEqual({ value: 'SGD', note: undefined });
    expect(lrSetting(sgd, rs)).toEqual({ value: '0.01' });
    expect(momentumSetting(rs)).toEqual(['Momentum', { value: '0.937' }]);
  });

  it('keeps the request as it is without a resolution (queued runs, older servers)', () => {
    expect(optimizerSetting(config, null)).toEqual({ value: 'Auto (AdamW for short runs, else SGD)' });
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

describe('precision and head (DOGFOOD-3 S4)', () => {
  it("in Model › Details' words", () => {
    expect(precisionSetting({ amp: true } as TrainConfig).value).toBe('Mixed precision (f16)');
    expect(precisionSetting({} as TrainConfig).value).toBe('f32');
    expect(headSetting({ head: 'o2o' } as TrainConfig).value).toBe('End-to-end (no NMS)');
    expect(headSetting({} as TrainConfig).value).toBe('NMS (one-to-many)');
  });
});
