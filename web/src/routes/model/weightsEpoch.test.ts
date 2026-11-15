import { describe, expect, it } from 'vitest';
import type { Run } from '$lib/api/types';
import { finishedEpochs, trainedFor, weightsEpoch } from './weightsEpoch';

const run = (status: Run['status'], best: number | null, epochs = 5, epoch = epochs) => ({
  id: 'run_a',
  status,
  epoch,
  epochs,
  best: best === null ? null : { epoch: best, map50_95: 0.4 },
});

describe('weightsEpoch', () => {
  it('names the epoch the weights are from, and the same weights when best is last', () => {
    expect(weightsEpoch({ id: 'm_a', run_id: 'run_a' }, run('completed', 1))).toBe('best epoch 1 of 5');
    expect(weightsEpoch({ id: 'm_a-last', run_id: 'run_a' }, run('completed', 1))).toBe('last epoch, 5 of 5');
    expect(weightsEpoch({ id: 'm_a', run_id: 'run_a' }, run('completed', 5))).toBe('best epoch 5 of 5, same weights as last');
    expect(weightsEpoch({ id: 'm_a-last', run_id: 'run_a' }, run('completed', 5))).toBe('last epoch, 5 of 5, same weights as best');
  });

  // s4-models-1: a run stopped after epoch 1 of 50 said only "last finished epoch".
  it('numbers the last finished epoch of a run that ended early', () => {
    expect(weightsEpoch({ id: 'm_a-last', run_id: 'run_a' }, run('stopped', 1, 50, 1))).toBe('last finished epoch, 1 of 50');
    expect(weightsEpoch({ id: 'm_a-last', run_id: 'run_a' }, run('failed', 2, 50, 3))).toBe('last finished epoch, 2 of 50');
    expect(weightsEpoch({ id: 'm_a-last', run_id: 'run_a' }, run('failed', null, 50, 1))).toBe('last finished epoch');
  });

  it('says nothing it does not know', () => {
    expect(weightsEpoch({ id: 'm_a', run_id: 'run_a' }, null)).toBeNull();
    expect(weightsEpoch({ id: 'm_a', run_id: 'run_b' }, run('completed', 1))).toBeNull();
    expect(weightsEpoch({ id: 'm_a', run_id: 'run_a' }, run('completed', null))).toBeNull();
  });
});

describe('finishedEpochs', () => {
  it('reads `epoch` as the API defines it per status', () => {
    expect(finishedEpochs({ status: 'completed', epoch: 5 })).toBe(5);
    expect(finishedEpochs({ status: 'stopped', epoch: 1 })).toBe(1);
    expect(finishedEpochs({ status: 'failed', epoch: 3 })).toBe(2);
    expect(finishedEpochs({ status: 'running', epoch: 0 })).toBe(0);
  });
});

describe('trainedFor', () => {
  const tw = (epochs: number) => ({ run_id: 'run_a', trained_with: { amp: false, imgsz: 320, epochs } });

  // s4-models-1: "Trained for 50 epochs" for a run stopped after one; trained_with.epochs is the planned count.
  it('says how many of the planned epochs a run that ended early trained', () => {
    expect(trainedFor(tw(50), run('stopped', 1, 50, 1))).toBe('1 of 50 epochs at 320 px');
    expect(trainedFor(tw(50), run('failed', 2, 50, 4))).toBe('3 of 50 epochs at 320 px');
  });

  it('prints the planned epochs when the run completed or is not known', () => {
    expect(trainedFor(tw(50), run('completed', 50, 50))).toBe('50 epochs at 320 px');
    expect(trainedFor(tw(1), null)).toBe('1 epoch at 320 px');
    expect(trainedFor({ ...tw(50), run_id: 'run_b' }, run('stopped', 1, 50, 1))).toBe('50 epochs at 320 px');
    expect(trainedFor({ run_id: null, trained_with: null }, null)).toBeNull();
  });
});
