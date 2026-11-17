// The built-in colour themes of design/tokens.json (design/THEMES.md), resolved for scripts/tokens.mjs: each theme is
// its mode's base palette (color.dark or color.light) with the theme's `color` merged over it key by key, arrays
// replaced whole. Anything malformed throws with the token path, so `npm run tokens` fails loudly instead of writing
// a palette with a hole in it. Also the one rule that names a colour token's CSS custom property: tokens.mjs writes it
// into tokens.ts (colorVars, which Studio's runtime reads) and into index.html's first-paint data.

const HEX = /^#([0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$/;
export const MODES = /** @type {const} */ (['dark', 'light']);
const isMeta = (/** @type {string} */ k) => k.startsWith('$');
const isObj = (/** @type {unknown} */ v) => v !== null && typeof v === 'object' && !Array.isArray(v);

/**
 * The CSS custom property of a colour token path: `bg.app` -> `--bg-app`, `accent.solid_hover` ->
 * `--accent-solid-hover`, `chart.series.0` -> `--chart-1`, `chart.fill.0` -> `--chart-fill-1`, `chart.text.0` ->
 * `--chart-text-1`, `focus` -> `--focus`.
 * @param {string} path
 */
export function cssVar(path) {
  const m = /^chart\.(series|fill|text)\.(\d+)$/.exec(path);
  if (m) return `--chart-${m[1] === 'series' ? '' : `${m[1]}-`}${Number(m[2]) + 1}`;
  return `--${path.replace(/[_.]/g, '-')}`;
}

/**
 * Every leaf of a palette in its key order, as [token path, value]: arrays by index (`chart.series.0`).
 * @param {Record<string, unknown>} color
 * @returns {[string, string | number][]}
 */
export function leaves(color, prefix = '') {
  /** @type {[string, string | number][]} */
  const out = [];
  for (const [k, v] of Object.entries(color)) {
    if (isMeta(k)) continue;
    const path = prefix ? `${prefix}.${k}` : k;
    if (Array.isArray(v)) v.forEach((x, i) => out.push([`${path}.${i}`, x]));
    else if (isObj(v)) out.push(...leaves(/** @type {Record<string, unknown>} */ (v), path));
    else out.push([path, /** @type {string | number} */ (v)]);
  }
  return out;
}

/**
 * `over` merged into a copy of `base` key by key; arrays replace whole. Every key of `over` must be one of `base`'s,
 * with the same shape (a group, a colour, a number, an array of the same length); `where` names the path in errors.
 * @param {Record<string, unknown>} base
 * @param {Record<string, unknown>} over
 * @param {string} where
 * @returns {Record<string, unknown>}
 */
export function mergePalette(base, over, where) {
  /** @type {Record<string, unknown>} */
  const out = {};
  for (const [k, v] of Object.entries(base)) if (!isMeta(k)) out[k] = Array.isArray(v) ? [...v] : isObj(v) ? mergePalette(/** @type {any} */ (v), {}, '') : v;
  for (const [k, v] of Object.entries(over)) {
    if (isMeta(k)) continue;
    const path = `${where}.${k}`;
    if (!(k in out)) throw new Error(`${path}: not a colour token`);
    const b = out[k];
    if (isObj(b)) {
      if (!isObj(v)) throw new Error(`${path}: expected a group of colour tokens`);
      out[k] = mergePalette(/** @type {any} */ (b), /** @type {any} */ (v), path);
    } else if (Array.isArray(b)) {
      if (!Array.isArray(v) || v.length !== b.length) throw new Error(`${path}: expected an array of ${b.length} colours`);
      v.forEach((x, i) => {
        if (typeof x !== 'string' || !HEX.test(x)) throw new Error(`${path}[${i}]: expected #RRGGBB or #RRGGBBAA, got ${JSON.stringify(x)}`);
      });
      out[k] = [...v];
    } else if (typeof b === 'number') {
      if (typeof v !== 'number' || !(v >= 0 && v <= 1)) throw new Error(`${path}: expected a number from 0 to 1, got ${JSON.stringify(v)}`);
      out[k] = v;
    } else {
      if (typeof v !== 'string' || !HEX.test(v)) throw new Error(`${path}: expected #RRGGBB or #RRGGBBAA, got ${JSON.stringify(v)}`);
      out[k] = v;
    }
  }
  return out;
}

/**
 * The built-in themes of `tokens` in their listed order, each with its resolved palette. Throws on a theme with an
 * unknown mode, a missing or empty name, an override that is not a colour token of its base, or a default (the first
 * theme of each mode, which must be the mode's own id) with overrides.
 * @param {any} tokens
 * @returns {{ id: string, name: string, mode: 'dark' | 'light', description: string, color: Record<string, unknown> }[]}
 */
export function builtinThemes(tokens) {
  if (!isObj(tokens.themes)) throw new Error('themes: expected an object of themes');
  /** @type {{ id: string, name: string, mode: 'dark' | 'light', description: string, color: Record<string, unknown> }[]} */
  const out = [];
  /** @type {Set<string>} */
  const seen = new Set();
  for (const [id, th] of Object.entries(tokens.themes)) {
    if (isMeta(id)) continue;
    const where = `themes.${id}`;
    if (!/^[a-z0-9][a-z0-9-]{0,39}$/.test(id)) throw new Error(`${where}: an id is 1-40 of a-z, 0-9 and -, starting with a letter or digit`);
    if (id === 'system') throw new Error(`${where}: "system" is reserved (Desktop's --theme system is the mode)`);
    if (!isObj(th)) throw new Error(`${where}: expected an object`);
    const t = /** @type {Record<string, unknown>} */ (th);
    for (const k of Object.keys(t)) if (!isMeta(k) && !['name', 'mode', 'description', 'color'].includes(k)) throw new Error(`${where}.${k}: unknown key`);
    const mode = t.mode;
    if (mode !== 'dark' && mode !== 'light') throw new Error(`${where}.mode: expected "dark" or "light", got ${JSON.stringify(mode)}`);
    if (typeof t.name !== 'string' || !t.name.trim()) throw new Error(`${where}.name: expected a name`);
    if (t.description !== undefined && typeof t.description !== 'string') throw new Error(`${where}.description: expected a string`);
    const over = t.color ?? {};
    if (!isObj(over)) throw new Error(`${where}.color: expected an object`);
    const first = !seen.has(mode);
    seen.add(mode);
    if (first && id !== mode) throw new Error(`${where}: the first ${mode} theme must be "${mode}", the mode's default`);
    if (first && leaves(/** @type {any} */ (over)).length) throw new Error(`${where}.color: the default ${mode} theme has no overrides`);
    const base = tokens.color?.[mode];
    if (!isObj(base)) throw new Error(`color.${mode}: missing`);
    out.push({ id, name: t.name.trim(), mode, description: typeof t.description === 'string' ? t.description : '', color: mergePalette(base, /** @type {any} */ (over), `${where}.color`) });
  }
  for (const m of MODES) if (!seen.has(m)) throw new Error(`themes: no ${m} theme`);
  return out;
}
