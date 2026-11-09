import { describe, expect, it } from 'vitest';
import { MODEL_SECTIONS, sectionShown } from './sections';

describe('sectionShown', () => {
  it('shows every part on wider screens', () => {
    for (const { value } of MODEL_SECTIONS) expect(sectionShown(false, 'curves', value)).toBe(true);
  });
  it('shows only the picked section on a phone', () => {
    const shown = MODEL_SECTIONS.filter(({ value }) => sectionShown(true, 'confusion', value)).map((s) => s.value);
    expect(shown).toEqual(['confusion']);
  });
});

