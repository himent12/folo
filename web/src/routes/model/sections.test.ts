import { describe, expect, it } from 'vitest';
import { MODEL_SECTIONS, prView, sectionShown } from './sections';

describe('sectionShown', () => {
  it('shows every part on wider screens', () => {
    for (const { value } of MODEL_SECTIONS) expect(sectionShown(false, 'curves', value)).toBe(true);
  });
  it('shows only the picked section on a phone', () => {
    const shown = MODEL_SECTIONS.filter(({ value }) => sectionShown(true, 'confusion', value)).map((s) => s.value);
    expect(shown).toEqual(['confusion']);
  });
});

describe('prView', () => {
  it('keeps the curves and the table together on wider screens', () => {
    expect(prView(false, 'classes')).toBe('all');
  });
  it('splits them into the Curves and Classes sections on a phone', () => {
    expect(prView(true, 'curves')).toBe('curves');
    expect(prView(true, 'classes')).toBe('classes');
    // The PR panel is hidden in the other sections; whatever it renders there is not shown.
    expect(prView(true, 'details')).toBe('all');
  });
});
