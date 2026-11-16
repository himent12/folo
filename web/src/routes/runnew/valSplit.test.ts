import { describe, expect, it } from 'vitest';
import { noValNote } from './valSplit';

const ds = (train: number, val: number) => ({ splits: { train, val, test: 0 } });

describe('noValNote', () => {
  it('says a train-only dataset is scored on its training images (round s2-runs-6)', () => {
    expect(noValNote(ds(192, 0))).toMatch(/^No validation images: each epoch is scored on the training images/);
  });

  it('says nothing for a dataset with a val split, one with no images, or none picked', () => {
    expect(noValNote(ds(192, 48))).toBeNull();
    expect(noValNote(ds(0, 0))).toBeNull();
    expect(noValNote(undefined)).toBeNull();
  });
});
