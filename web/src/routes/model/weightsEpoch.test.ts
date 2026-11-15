import { describe, expect, it } from 'vitest';
import { finishedEpochs, trainedFor, weightsEpoch } from './weightsEpoch';

const run = (best: number | null, epochs = 5) => ({ id: 'run_a', epochs, best: best === null ? null : { epoch: best, map50_95: 0.4 } });
const best = { id: 'm_a', run_id: 'run_a' };
const last = { id: 'm_a-last', run_id: 'run_a' };

describe('weightsEpoch', () => {
  it('names the epoch the weights are from, and the same weights when best is last', () => {
    expect(weightsEpoch(best, run(1), 5)).toBe('best epoch 1 of 5');
    expect(weightsEpoch(last, run(1), 5)).toBe('last epoch, 5 of 5');
    expect(weightsEpoch(best, run(5), 5)).toBe('best epoch 5 of 5, same weights as last');
    expect(weightsEpoch(last, run(5), 5)).toBe('last epoch, 5 of 5, same weights as best');
  });

  // s4-models-1: a run stopped after epoch 1 of 50 said only "last finished epoch".
  it('numbers the last finished epoch of a run that ended early', () => {
    expect(weightsEpoch(last, run(1, 50), 1)).toBe('last finished epoch, 1 of 50, same weights as best');
    expect(weightsEpoch(last, run(1, 50), 2)).toBe('last finished epoch, 2 of 50');
    expect(weightsEpoch(best, run(1, 50), 2)).toBe('best epoch 1 of 50');
  });

  // Review round 1: a completed run that stopped early (patience) is not at its planned end.
  it('says a completed run that stopped early ended before its planned end', () => {
    expect(weightsEpoch(last, run(1, 200), 101)).toBe('last finished epoch, 101 of 200');
    expect(weightsEpoch(best, run(1, 200), 101)).toBe('best epoch 1 of 200');
  });

  it('says nothing it does not know', () => {
    expect(weightsEpoch(best, null, 5)).toBeNull();
    expect(weightsEpoch(best, { ...run(1), id: 'run_b' }, 5)).toBeNull();
    expect(weightsEpoch(best, run(null), 5)).toBeNull();
    expect(weightsEpoch(last, run(null, 50), null)).toBe('last finished epoch');
    expect(weightsEpoch(best, run(5), null)).toBe('best epoch 5 of 5');
  });
});

describe('finishedEpochs', () => {
  const h = (n: number) => Array.from({ length: n }, (_, i) => ({ epoch: i + 1 }));
  // Re-review 1: a crash after `last.folo` of epoch 3 was saved, before its history line, leaves the history at epoch
  // 2; the server brings the stopped run's `epoch` up to the checkpoint (runs/history.rs reconcile), so that counts.
  it('takes a completed or stopped run’s epoch, which the checkpoint backs, over a history that lags it', () => {
    expect(finishedEpochs({ status: 'stopped', epoch: 3 }, h(2))).toBe(3);
    expect(finishedEpochs({ status: 'stopped', epoch: 3 }, null)).toBe(3);
    // Review round 1: an early-stopped completed run has epoch < epochs.
    expect(finishedEpochs({ status: 'completed', epoch: 101 }, h(101))).toBe(101);
    expect(finishedEpochs({ status: 'stopped', epoch: 0 }, [])).toBeNull();
  });

  // Review round 1: a failed resume that never trained keeps epoch 2, a queued resume keeps epoch 3.
  it('takes any other run’s history when its epoch is not past it', () => {
    expect(finishedEpochs({ status: 'failed', epoch: 2 }, h(2))).toBe(2);
    expect(finishedEpochs({ status: 'queued', epoch: 3 }, h(3))).toBe(3);
    expect(finishedEpochs({ status: 'running', epoch: 2 }, h(2))).toBe(2);
  });

  // Past it, the run may be in that epoch, have failed in it, or have saved its checkpoint and crashed (or failed to
  // write its samples) before the history line: not known.
  it('does not know when another run’s epoch is past its history, nor without the history', () => {
    expect(finishedEpochs({ status: 'failed', epoch: 3 }, h(2))).toBeNull();
    expect(finishedEpochs({ status: 'queued', epoch: 3 }, h(2))).toBeNull();
    expect(finishedEpochs({ status: 'running', epoch: 3 }, h(2))).toBeNull();
    expect(finishedEpochs({ status: 'failed', epoch: 2 }, null)).toBeNull();
    expect(finishedEpochs({ status: 'queued', epoch: 0 }, [])).toBeNull();
  });
});

describe('trainedFor', () => {
  const tw = (epochs: number) => ({ run_id: 'run_a', trained_with: { amp: false, imgsz: 320, epochs } });

  // s4-models-1: "Trained for 50 epochs" for a run stopped after one; trained_with.epochs is the planned count.
  it('says how many of the planned epochs a run that ended early trained', () => {
    expect(trainedFor(tw(50), run(1, 50), 1)).toBe('1 of 50 epochs at 320 px');
    expect(trainedFor(tw(50), run(2, 50), 2)).toBe('2 of 50 epochs at 320 px');
    expect(trainedFor(tw(200), run(1, 200), 101)).toBe('101 of 200 epochs at 320 px');
  });

  it('prints the planned epochs when the run trained them all', () => {
    expect(trainedFor(tw(50), run(50, 50), 50)).toBe('50 epochs at 320 px');
    expect(trainedFor(tw(1), run(1, 1), 1)).toBe('1 epoch at 320 px');
  });

  // Review round 1: while the history loads, when it or the run failed, or for a copied checkpoint, the count is the
  // plan, and says so.
  it('labels the plan as planned while what the run trained is not known', () => {
    expect(trainedFor(tw(50), run(1, 50), null)).toBe('50 epochs planned at 320 px');
    expect(trainedFor(tw(50), null, null)).toBe('50 epochs planned at 320 px');
    expect(trainedFor({ ...tw(50), run_id: null }, null, null)).toBe('50 epochs planned at 320 px');
    expect(trainedFor({ ...tw(50), run_id: 'run_b' }, run(1, 50), 1)).toBe('50 epochs planned at 320 px');
    expect(trainedFor({ run_id: null, trained_with: null }, null, null)).toBeNull();
  });
});
