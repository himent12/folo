import { describe, expect, it } from 'vitest';
import type { Model } from '$lib/api/types';
import { startHintOf } from './weights';

const model = (m: Partial<Model>) => ({ id: 'm', name: 'm', arch: 'folo-n', classes: [], metrics: null, source: 'run', ...m }) as unknown as Model;
const coco = model({ id: 'folo-n-coco-v2', source: 'published', about: 'folo-n from scratch on COCO train2017', classes: Array.from({ length: 80 }, (_, i) => `c${i}`) });
const none = () => ({ split: null, dataset: null }) as never;
const bccd = { name: 'bccd', classes: ['RBC', 'WBC', 'Platelets'] };

describe('Start from: the hint', () => {
  it('says the class outputs start fresh when the weights were trained on other classes', () => {
    expect(startHintOf(coco, none, 'folo-n', 1, [], bccd)).toBe(
      "Pretrained on COCO (80 classes) · good for fine-tuning. Its classes are not bccd's: the class outputs start fresh, the rest is kept.",
    );
    // A model trained on the same classes, in the same order, keeps them: nothing to say.
    const same = model({ classes: ['RBC', 'WBC', 'Platelets'] });
    expect(startHintOf(same, none, 'folo-n', 1, [], bccd)).not.toContain('start fresh');
    // The same names in another order are another class list.
    const shuffled = model({ classes: ['WBC', 'RBC', 'Platelets'] });
    expect(startHintOf(shuffled, none, 'folo-n', 1, [], bccd)).toContain("Its classes are not bccd's");
    // Unknown classes (an older model file): nothing it cannot know.
    expect(startHintOf(model({}), none, 'folo-n', 1, [], bccd)).toBe('Fine-tuning from a trained model converges much faster.');
  });
});
