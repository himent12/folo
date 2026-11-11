// The fw Vite plugin: compiles JSX in `.tsx` files (templates + fine-grained bindings), scopes each file's
// `css\`...\`` as Svelte 5 would and serves it as a CSS module the file imports, and serves lucide icons as
// `$icons/<name>` modules (lucide's data, vendored in ../icons/lucide.json).

import { existsSync, readFileSync as read, realpathSync } from 'node:fs';
import { basename, isAbsolute, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Plugin } from 'vite';
import { scopeCss } from './css.ts';
import { transformJsx } from './jsx.ts';

export { scopeCss, specificity } from './css.ts';
export { transformJsx } from './jsx.ts';

/** The prefix of every scope class (`fw-1x2y3z`). */
export const SCOPE_PREFIX = 'fw-';

/** The scope class of a file: a short, stable hash of its path relative to the app root. */
export function scopeClass(file: string, root: string): string {
  const rel = relative(root, file).replace(/\\/g, '/');
  let h = 0x811c9dc5;
  for (let i = 0; i < rel.length; i++) {
    h ^= rel.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return SCOPE_PREFIX + (h >>> 0).toString(36);
}

const realCache = new Map<string, string | null>();
/** The file's name as it is on disk (its real case), or null if it cannot be read. */
function realCase(file: string): string | null {
  let name = realCache.get(file);
  if (name === undefined) {
    try {
      name = basename(realpathSync.native(file));
    } catch {
      name = null;
    }
    realCache.set(file, name);
  }
  return name;
}

/** A file's scoped CSS is the module `<file>.fw.css` (virtual; a query on a .tsx id would be taken for TSX). */
const CSS_QUERY = '.fw.css';
const ICON = '\0fw-icon:';

export interface FwOptions {
  /** Files to compile (default: every `.tsx` outside node_modules). */
  include?: (id: string) => boolean;
}

export function fw(options: FwOptions = {}): Plugin {
  let root = process.cwd();
  const css = new Map<string, string>();
  const include = options.include ?? ((id: string) => id.endsWith('.tsx') && !id.includes('/node_modules/'));
  /** The vendored lucide icons (../icons/lucide.json; `node framework/tools/icons.mjs --add` vendors a new one). */
  let icons: Record<string, unknown> | undefined;

  /** The .tsx file a `<file>.fw.css` id belongs to. The import is absolute; a server may hand it back root-relative. */
  const cssFile = (id: string) => {
    const file = id.slice(0, -CSS_QUERY.length);
    if (isAbsolute(file) && existsSync(file)) return file;
    return resolve(root, '.' + file).split('\\').join('/');
  };

  const compile = (code: string, file: string) => {
    const scope = scopeClass(file, root);
    const out = transformJsx(code, file, { scope });
    if (out.css !== undefined) css.set(file, scopeCss(out.css, scope));
    else css.delete(file);
    return out;
  };

  return {
    name: 'fw',
    enforce: 'pre',
    configResolved(config) {
      root = config.root;
    },
    async resolveId(id, importer, options) {
      if (id.startsWith('$icons/')) return ICON + id.slice('$icons/'.length);
      if (id.endsWith(CSS_QUERY)) return cssFile(id) + CSS_QUERY;
      // On a case-insensitive disk `$lib/ui/Field` can resolve to `field.ts` (`.ts` is tried before `.tsx`): an import
      // must name the file it gets, case included, or it is an error (and on Linux it would be another module).
      if (!importer || importer.includes('/node_modules/') || !(id.startsWith('./') || id.startsWith('../') || id.startsWith('$lib'))) return undefined;
      const r = await this.resolve(id, importer, { ...options, skipSelf: true });
      if (!r || r.external) return r;
      const file = r.id.split('?')[0]!;
      const real = realCase(file);
      const want = basename(id);
      if (real !== null && basename(file) !== real && real.toLowerCase() === basename(file).toLowerCase()) {
        const ext = real.slice(real.lastIndexOf('.'));
        this.error(
          `fw: "${id}" in ${relative(root, importer)} resolves to ${real} (the disk ignores case).` +
            ` Import the file you mean with its extension (e.g. "${id.slice(0, id.length - want.length)}${want}.tsx"), or "${id.slice(0, id.length - want.length)}${real.slice(0, -ext.length)}" for ${real}.`,
        );
      }
      return r;
    },
    load(id) {
      if (id.startsWith(ICON)) {
        const name = id.slice(ICON.length);
        icons ??= JSON.parse(read(fileURLToPath(new URL('../icons/lucide.json', import.meta.url)), 'utf8')) as Record<string, unknown>;
        const data = icons[name];
        if (data === undefined) throw new Error(`fw: the lucide icon "${name}" is not vendored: run \`node framework/tools/icons.mjs --add\``);
        return `import { createIcon } from 'fw/icon';\nexport default /*#__PURE__*/ createIcon(${JSON.stringify(data)});\n`;
      }
      if (id.endsWith(CSS_QUERY)) {
        const file = cssFile(id);
        if (!css.has(file)) compile(read(file, 'utf8'), file);
        return css.get(file) ?? '';
      }
      return undefined;
    },
    handleHotUpdate(ctx) {
      // An edited .tsx also changes its CSS module.
      if (!include(ctx.file)) return undefined;
      const mod = ctx.server.moduleGraph.getModuleById(ctx.file + CSS_QUERY);
      return mod ? [...ctx.modules, mod] : undefined;
    },
    transform(code, id) {
      const file = id.split('?')[0]!;
      if (file !== id || !include(file)) return undefined;
      const out = compile(code, file);
      if (out.css === undefined) return { code: out.code, map: out.map };
      // After the module's own imports, as vite-plugin-svelte does: the CSS of the components a file imports comes
      // first in the bundle, so the file's own rules win ties of specificity over theirs (the cascade Svelte gives).
      return { code: `${out.code}\nimport ${JSON.stringify(file + CSS_QUERY)};\n`, map: out.map };
    },
  };
}
