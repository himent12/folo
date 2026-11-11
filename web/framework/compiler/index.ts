// The fw Vite plugin: compiles JSX in `.tsx` files (templates + fine-grained bindings), scopes each file's
// `css\`...\`` as Svelte 5 would and serves it as a CSS module the file imports, and serves lucide icons as
// `$icons/<name>` modules.

import { createRequire } from 'node:module';
import { existsSync, readFileSync as read } from 'node:fs';
import { dirname, isAbsolute, join, relative, resolve } from 'node:path';
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
  const require = createRequire(import.meta.url);
  let lucideDir: string | undefined;

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
    resolveId(id) {
      if (id.startsWith('$icons/')) return ICON + id.slice('$icons/'.length);
      if (id.endsWith(CSS_QUERY)) return cssFile(id) + CSS_QUERY;
      return undefined;
    },
    load(id) {
      if (id.startsWith(ICON)) {
        const name = id.slice(ICON.length);
        lucideDir ??= join(dirname(require.resolve('@lucide/svelte')), 'icons');
        // `icons/<name>.js` re-exports the icon's .svelte file (an alias points at another icon's).
        const entry = read(join(lucideDir, `${name}.js`), 'utf8').match(/from "\.\/([\w-]+)\.svelte"/);
        if (!entry) throw new Error(`fw: no lucide icon "${name}"`);
        const src = read(join(lucideDir, `${entry[1]}.svelte`), 'utf8');
        const m = src.match(/export const __iconData = (\{.*?\});\r?\n/);
        if (!m) throw new Error(`fw: no icon data in @lucide/svelte/icons/${name}`);
        return `import { createIcon } from 'fw/icon';\nexport default /*#__PURE__*/ createIcon(${m[1]});\n`;
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
      return { code: `import ${JSON.stringify(file + CSS_QUERY)};\n${out.code}`, map: shiftMap(out.map) };
    },
  };
}

/** The map of code that got one line prepended. */
function shiftMap<M extends { mappings: string }>(map: M): M {
  return { ...map, mappings: ';' + map.mappings };
}
