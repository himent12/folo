import { describe, expect, it } from 'vitest';
import { cleanPath, isFullPath, NAME_RE, nameFromPath, suggestName, uniqueName } from './names';

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

describe('nameFromPath', () => {
  it('makes a tidy name from the folder, never from a file name or its extension', () => {
    expect(nameFromPath('D:\\folo-ux\\work-datasets\\blood cells copy (2)')).toBe('blood-cells-copy-2');
    expect(nameFromPath('D:/data/my set - v2/')).toBe('my-set-v2');
    expect(nameFromPath('D:\\folo-data\\finetune\\bccd\\annotations\\instances_test.json')).toBe('bccd');
    expect(nameFromPath('D:/data/coco/instances_val.json')).toBe('instances_val');
    expect(nameFromPath('/home/me/fasteners/data.yaml')).toBe('fasteners');
    expect(nameFromPath('D:/data/shapes')).toBe('shapes');
    expect(nameFromPath('')).toBe('');
  });
});

describe('uniqueName', () => {
  it('numbers a name that is taken, ignoring case, and stays within 64 characters', () => {
    const taken = new Set(['shapes-synth', 'shapes-synth-2', 'bccd']);
    expect(uniqueName('shapes-synth', taken)).toBe('shapes-synth-3');
    expect(uniqueName('BCCD', taken)).toBe('BCCD-2');
    expect(uniqueName('fasteners', taken)).toBe('fasteners');
    const long = 'a'.repeat(64);
    expect(uniqueName(long, new Set([long]))).toBe(`${'a'.repeat(62)}-2`);
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
