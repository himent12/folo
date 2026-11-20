// A live run's 0–1 chart axis (RunModel.unitFit): held within an epoch, except that a fit made while every value was
// zero (the 0–1 fallback) gives way as soon as a non-zero validation arrives (issue #382, seen in #377).
import { signal } from 'fw';
import { describe, expect, it } from 'vitest';
import { AppendLog } from '$lib/api/appendLog';
import { MockApi } from '$lib/api/mock';
import type { Epoch, Run, Step } from '$lib/api/types';
import { RunModel } from './runModel';

const step = (n: number): Step => ({
  epoch: Math.ceil(n / 50),
  step: n,
  lr: 1e-3,
  loss: { box: 1, cls: 1, dfl: 1, total: 3 },
  imgs_per_s: 40,
  eta_s: 200,
  steps_per_epoch: 50,
  epochs: 5,
});
const epoch = (n: number, map: number): Epoch => ({
  epoch: n,
  train_loss: { box: 1, cls: 1, dfl: 1, total: 3 },
  val: { map50: map, map50_95: map, precision: map, recall: map, best_f1_conf: 0.3, per_class: [] },
  duration_s: 10,
});

async function live() {
  const base = await new MockApi('normal', 0).run('run_7d1e');
  const run: Run = { ...base, status: 'running', epoch: 3, epochs: 5, step: 115, steps_per_epoch: 50, best: null };
  const steps = new AppendLog<Step>();
  const epochs = new AppendLog<Epoch>();
  steps.push(step(115));
  epochs.push(epoch(1, 0));
  const version = signal(1);
  const model = new RunModel({
    run,
    steps,
    epochs,
    get version() {
      return version.get();
    },
  });
  const fit = model.unitFit(() => model.mapSeries.get(), () => 240);
  /** A validation that arrives within the epoch in progress (the steps stay at 2.3 epochs). */
  const validate = (e: Epoch) => {
    epochs.push(e);
    version.set(version.get() + 1);
  };
  return { fit, validate };
}

describe('a live run’s 0–1 axis', () => {
  it('fits a late first non-zero validation at once instead of holding the all-zero 0–1 axis', async () => {
    const { fit, validate } = await live();
    expect(fit.get()?.domain).toEqual([0, 1]);
    validate(epoch(2, 0.0015));
    expect(fit.get()?.domain).toEqual([0, 0.0016]);
  });
});
