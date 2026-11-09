import { describe, expect, it } from 'vitest';
import { NAME_RE, suggestName } from './names';

describe('suggestName', () => {
  it('turns what was typed into the nearest valid name', () => {
    expect(suggestName('my dataset!')).toBe('my-dataset');
    expect(suggestName('  _photos (2024)/  ')).toBe('photos-2024');
    expect(suggestName('Fasteners v2')).toBe('Fasteners-v2');
  });
  it('keeps every suggestion valid, and offers none when nothing usable is left', () => {
    for (const raw of ['a'.repeat(80) + '!', 'x y z', '-a-', 'ok.name_1']) expect(NAME_RE.test(suggestName(raw)!)).toBe(true);
    expect(suggestName('!!!')).toBeNull();
    expect(suggestName('')).toBeNull();
  });
});
