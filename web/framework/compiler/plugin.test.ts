import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { fw } from './index';

describe('fw Vite plugin', () => {
  const dir = mkdtempSync(join(tmpdir(), 'fw-plugin-'));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));
  writeFileSync(join(dir, 'field.ts'), 'export {};');
  writeFileSync(join(dir, 'Field.tsx'), 'export {};');
  const caseInsensitive = existsSync(join(dir, 'FIELD.TS'));
  const plugin = fw();
  const resolveId = plugin.resolveId as unknown as (this: unknown, id: string, importer: string, options: object) => Promise<unknown>;
  const ctx = (resolved: string) => ({
    resolve: async () => ({ id: resolved, external: false }),
    error: (msg: string) => {
      throw new Error(msg);
    },
  });

  it.runIf(caseInsensitive)('rejects an import that the disk resolves to a file of another case', async () => {
    // What Vite returns on Windows for `./Field`: `.ts` is tried before `.tsx`, and `Field.ts` "exists".
    const importer = join(dir, 'x.tsx').replace(/\\/g, '/');
    await expect(resolveId.call(ctx(join(dir, 'Field.ts').replace(/\\/g, '/')), './Field', importer, {})).rejects.toThrow(/resolves to field\.ts/);
  });

  it('accepts an import whose case matches the file', async () => {
    const importer = join(dir, 'x.tsx').replace(/\\/g, '/');
    const target = join(dir, 'Field.tsx').replace(/\\/g, '/');
    await expect(resolveId.call(ctx(target), './Field.tsx', importer, {})).resolves.toEqual({ id: target, external: false });
  });

  it('puts the CSS import after the module’s own imports (child CSS first, as Svelte)', () => {
    const file = join(dir, 'Comp.tsx').replace(/\\/g, '/');
    const src = "import Child from './Child';\nimport { css } from 'fw';\nexport default () => <Child />;\ncss`.a { color: red }`;\n";
    const transform = plugin.transform as unknown as (code: string, id: string) => { code: string };
    const out = transform.call({}, src, file).code;
    expect(out.indexOf("from './Child'")).toBeLessThan(out.indexOf(`${file}.fw.css`));
    expect(out.trimEnd().endsWith(`import ${JSON.stringify(file + '.fw.css')};`)).toBe(true);
  });
});
