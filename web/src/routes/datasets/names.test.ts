import { describe, expect, it } from 'vitest';
import { cleanPath, isFullPath, NAME_RE, suggestName } from './names';

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

describe('cleanPath', () => {
  it('drops the quotes of a pasted "Copy as path" and stray whitespace', () => {
    expect(cleanPath('"D:\\folo-data\\medical-pills"')).toBe('D:\\folo-data\\medical-pills');
    expect(cleanPath("  'D:/data/my set'  ")).toBe('D:/data/my set');
    expect(cleanPath(' D:/data/x \n')).toBe('D:/data/x');
    // Only a matching pair: a quote inside the path stays.
    expect(cleanPath('"D:/data/x')).toBe('"D:/data/x');
    expect(cleanPath("D:/it's")).toBe("D:/it's");
  });
});

describe('isFullPath', () => {
  it('accepts drives, shares and absolute POSIX paths, and nothing relative', () => {
    for (const p of ['D:\\data', 'd:/data', '\\\\nas\\sets', '/home/me/data']) expect(isFullPath(p)).toBe(true);
    for (const p of ['bccd', 'data/x', './x', 'D:', 'D:data', '~/data']) expect(isFullPath(p)).toBe(false);
  });
});
