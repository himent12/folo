import { describe, expect, it } from 'vitest';
import { pretrainedOn, publishedSummary } from './published';

const ABOUT = 'folo-n trained from scratch on COCO train2017 at 640 px (EMA weights, f32, unfused: fine-tune or predict); examples/coco';

describe('published weights', () => {
  it('names the dataset they were trained on, and none it was not told', () => {
    expect(publishedSummary({ about: ABOUT, classes: Array.from({ length: 80 }, (_, i) => `c${i}`) })).toBe('Pretrained on COCO (80 classes) · good for fine-tuning');
    expect(pretrainedOn('weights from somewhere')).toBeNull();
    expect(publishedSummary({ about: undefined, classes: ['a'] })).toBe('Pretrained (1 class) · good for fine-tuning');
  });
});
