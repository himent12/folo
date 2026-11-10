import { describe, expect, it } from 'vitest';
import { MODEL_SECTIONS, sectionShown } from './sections';

describe('sectionShown', () => {
  it('shows every part on wider screens, and only the picked section on a phone', () => {
    for (const { value } of MODEL_SECTIONS) expect(sectionShown(false, 'curves', value)).toBe(true);
    const shown = MODEL_SECTIONS.filter(({ value }) => sectionShown(true, 'confusion', value)).map((s) => s.value);
    expect(shown).toEqual(['confusion']);
  });
});

