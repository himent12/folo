// Colour themes (design/THEMES.md): the built-in ones from tokens.ts, custom theme files checked exactly as the spec
// says (every error names its token path), a theme resolved to its palette and to what <html> needs to show it
// (`data-theme-id` and, for a custom theme, its overrides as inline custom properties), and each mode's list. Pure:
// prefs.ts keeps the choice and the installed files, App.tsx applies the result, index.html's first-paint script does
// the same before any of this has loaded (its rules are tested against these, firstPaint.test.ts).

import { colorVars, themeIds, themes as builtins, type BuiltinTheme, type Palette, type ThemeId } from './tokens';

export type Mode = 'dark' | 'light';

/** A custom theme file as checked and stored (`folo.themes`): comments (`$` keys) dropped, the name trimmed. */
export interface CustomThemeFile {
  folo_theme: 1;
  id: string;
  name: string;
  mode: Mode;
  base?: ThemeId;
  description?: string;
  /** Any subset of the palette, with the paths of `color.dark` in tokens.json. */
  color?: Record<string, unknown>;
}

/** A theme ready to show: built-in or custom. */
export interface Theme {
  id: string;
  name: string;
  mode: Mode;
  description: string;
  custom: boolean;
  /** The built-in theme whose CSS block it uses (`data-theme-id`): itself, or a custom theme's base. */
  base: ThemeId;
  /** The resolved palette (canvas and SVG code read colours here, never by mode). */
  color: Palette;
  /** A custom theme's own colours as [CSS custom property, value], set inline on <html>; none for a built-in. */
  vars: [string, string][];
}

/** The format version this Studio reads. */
export const THEME_FORMAT = 1;
const HEX = /^#(?:[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$/;
const ID = /^[a-z0-9-]{1,40}$/;
const KEYS = ['folo_theme', 'id', 'name', 'mode', 'base', 'description', 'color'];
const NAME_MAX = 40;
const DESCRIPTION_MAX = 120;

const isObj = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === 'object' && !Array.isArray(v);
const isMeta = (k: string) => k.startsWith('$');
const VAR = new Map(colorVars.map(([path, name]) => [path, name]));

/** The mode's default theme: `dark` or `light`, with no overrides. */
export const defaultThemeId = (mode: Mode): ThemeId => mode;

export function isBuiltinId(id: string): id is ThemeId {
  return (themeIds as readonly string[]).includes(id);
}

function fromBuiltin(t: BuiltinTheme): Theme {
  return { id: t.id, name: t.name, mode: t.mode, description: t.description, custom: false, base: t.id, color: t.color, vars: [] };
}

/** The built-in theme `id`. */
export function builtinTheme(id: ThemeId): Theme {
  return fromBuiltin(builtins[id]);
}

/**
 * `over` checked against `base` (one level of the palette) and merged into a copy of it: every key a token of the
 * base, of its shape; arrays replace whole and keep the base's length. Throws `<path>: <why>`.
 */
function merge(base: Record<string, unknown>, over: unknown, path: string): Record<string, unknown> {
  if (!isObj(over)) throw new Error(`${path}: expected an object of colour tokens`);
  const out: Record<string, unknown> = { ...base };
  for (const [k, v] of Object.entries(over)) {
    if (isMeta(k)) continue;
    const at = `${path}.${k}`;
    if (!Object.hasOwn(base, k)) throw new Error(`${at}: not a colour token`);
    const b = base[k];
    if (isObj(b)) out[k] = merge(b, v, at);
    else if (Array.isArray(b)) {
      if (!Array.isArray(v) || v.length !== b.length) throw new Error(`${at}: expected an array of ${b.length} colours`);
      v.forEach((x, i) => {
        if (typeof x !== 'string' || !HEX.test(x)) throw new Error(`${at}[${i}]: not a colour (#RRGGBB or #RRGGBBAA)`);
      });
      out[k] = [...v];
    } else if (typeof b === 'number') {
      if (typeof v !== 'number' || !(v >= 0 && v <= 1)) throw new Error(`${at}: expected a number from 0 to 1`);
      out[k] = v;
    } else {
      if (typeof v !== 'string' || !HEX.test(v)) throw new Error(`${at}: not a colour (#RRGGBB or #RRGGBBAA)`);
      out[k] = v;
    }
  }
  return out;
}

/** `color` with its comments dropped, at every level. */
function strip(v: unknown): unknown {
  if (Array.isArray(v)) return [...v];
  if (!isObj(v)) return v;
  return Object.fromEntries(Object.entries(v).filter(([k]) => !isMeta(k)).map(([k, x]) => [k, strip(x)]));
}

/**
 * Checks a custom theme file's content (already parsed from JSON) per design/THEMES.md, Custom themes. The first
 * error names its path (`color.bg.ap: not a colour token`); a file with one is not installed.
 */
export function checkCustomTheme(raw: unknown): { ok: true; theme: CustomThemeFile } | { ok: false; error: string } {
  try {
    return { ok: true, theme: check(raw) };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
}

function check(raw: unknown): CustomThemeFile {
  if (!isObj(raw)) throw new Error('expected a theme object ({ "folo_theme": 1, ... })');
  for (const k of Object.keys(raw)) if (!isMeta(k) && !KEYS.includes(k)) throw new Error(`${k}: unknown key`);
  const need = (k: string) => {
    if (raw[k] === undefined) throw new Error(`${k}: required`);
    return raw[k];
  };
  const format = need('folo_theme');
  if (format !== THEME_FORMAT) throw new Error(`folo_theme: expected ${THEME_FORMAT}, got ${JSON.stringify(format)}`);
  const id = need('id');
  if (typeof id !== 'string' || !ID.test(id)) throw new Error('id: expected 1-40 characters of a-z, 0-9 and -');
  if (isBuiltinId(id)) throw new Error(`id: "${id}" is a built-in theme`);
  const name = need('name');
  if (typeof name !== 'string') throw new Error('name: expected a string');
  const trimmed = name.trim();
  if (!trimmed || trimmed.length > NAME_MAX) throw new Error(`name: expected 1-${NAME_MAX} characters`);
  const description = raw.description;
  if (description !== undefined && typeof description !== 'string') throw new Error('description: expected a string');
  if (typeof description === 'string' && description.length > DESCRIPTION_MAX) throw new Error(`description: at most ${DESCRIPTION_MAX} characters`);
  const mode = need('mode');
  if (mode !== 'dark' && mode !== 'light') throw new Error('mode: expected "dark" or "light"');
  const base = raw.base;
  if (base !== undefined) {
    if (typeof base !== 'string' || !isBuiltinId(base)) throw new Error(`base: ${JSON.stringify(base)} is not a built-in theme`);
    if (builtins[base].mode !== mode) throw new Error(`base: "${base}" is a ${builtins[base].mode} theme, this one is ${mode}`);
  }
  const color = raw.color;
  if (color !== undefined) merge(builtins[(base as ThemeId | undefined) ?? defaultThemeId(mode)].color as unknown as Record<string, unknown>, color, 'color');
  const out: CustomThemeFile = { folo_theme: 1, id, name: trimmed, mode };
  if (base !== undefined) out.base = base as ThemeId;
  if (description !== undefined) out.description = description;
  if (color !== undefined) out.color = strip(color) as Record<string, unknown>;
  return out;
}

/** Parses and checks a file's text; a file that is not JSON says so. */
export function parseThemeFile(text: string): { ok: true; theme: CustomThemeFile } | { ok: false; error: string } {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return { ok: false, error: 'not a JSON file' };
  }
  return checkCustomTheme(raw);
}

/** Every leaf of a (partial) palette as [token path, value], arrays by index (`chart.series.0`). */
function leaves(color: Record<string, unknown>, prefix = ''): [string, unknown][] {
  const out: [string, unknown][] = [];
  for (const [k, v] of Object.entries(color)) {
    if (isMeta(k)) continue;
    const path = prefix ? `${prefix}.${k}` : k;
    if (Array.isArray(v)) v.forEach((x, i) => out.push([`${path}.${i}`, x]));
    else if (isObj(v)) out.push(...leaves(v, path));
    else out.push([path, v]);
  }
  return out;
}

/** A (partial) palette as CSS custom properties, written as tokens.css writes them (lower-case hex). */
export function cssVarsOf(color: Record<string, unknown>): [string, string][] {
  return leaves(color).flatMap(([path, v]): [string, string][] => {
    const name = VAR.get(path);
    if (!name) return [];
    return [[name, typeof v === 'number' ? String(v) : String(v).toLowerCase()]];
  });
}

/** A checked custom theme, resolved: its base's palette with its colours over it. */
export function customTheme(file: CustomThemeFile): Theme {
  const base = file.base ?? defaultThemeId(file.mode);
  const over = file.color ?? {};
  return {
    id: file.id,
    name: file.name,
    mode: file.mode,
    description: file.description ?? '',
    custom: true,
    base,
    color: merge(builtins[base].color as unknown as Record<string, unknown>, over, 'color') as unknown as Palette,
    vars: cssVarsOf(over),
  };
}

/**
 * The theme `id` names for `mode` among the built-ins and `customs`; the mode's default when it names none of that
 * mode (a removed custom theme, a typo in a hand edit, a theme of the other mode). The stored id is not changed, so
 * installing that theme again brings it back.
 */
export function resolveTheme(id: string, mode: Mode, customs: readonly CustomThemeFile[]): Theme {
  if (isBuiltinId(id) && builtins[id].mode === mode) return builtinTheme(id);
  const c = customs.find((t) => t.id === id && t.mode === mode);
  return c ? customTheme(c) : builtinTheme(defaultThemeId(mode));
}

/** A theme of either mode by id, or undefined: what the palette's "Theme: …" commands and Settings' selects pick. */
export function findTheme(id: string, customs: readonly CustomThemeFile[]): Theme | undefined {
  if (isBuiltinId(id)) return builtinTheme(id);
  const c = customs.find((t) => t.id === id);
  return c ? customTheme(c) : undefined;
}

/** The themes of `mode`: the built-ins in their order, then the custom ones by name. */
export function themesFor(mode: Mode, customs: readonly CustomThemeFile[]): Theme[] {
  const own = themeIds.filter((id) => builtins[id].mode === mode).map(builtinTheme);
  const mine = customs
    .filter((t) => t.mode === mode)
    .map(customTheme)
    .sort((a, b) => a.name.localeCompare(b.name) || a.id.localeCompare(b.id));
  return [...own, ...mine];
}

/** What an import did with one file. */
export type ImportResult = { file: string; ok: true; id: string; name: string; replaced: boolean } | { file: string; ok: false; error: string };

/**
 * Adds the files that pass the check to `installed` (a theme whose id is installed already replaces it, in place) and
 * says per file what happened. Files are read in order, so of two with one id the later wins.
 */
export function importThemes(installed: readonly CustomThemeFile[], files: readonly { name: string; text: string }[]): { themes: CustomThemeFile[]; results: ImportResult[] } {
  const themes = [...installed];
  const results: ImportResult[] = files.map((f) => {
    const r = parseThemeFile(f.text);
    if (!r.ok) return { file: f.name, ok: false, error: r.error };
    const at = themes.findIndex((t) => t.id === r.theme.id);
    if (at >= 0) themes[at] = r.theme;
    else themes.push(r.theme);
    return { file: f.name, ok: true, id: r.theme.id, name: r.theme.name, replaced: at >= 0 };
  });
  return { themes, results };
}

/** The stored list (`folo.themes`) with every entry that no longer passes the check left out (a hand edit). */
export function readStoredThemes(raw: unknown): CustomThemeFile[] {
  if (!Array.isArray(raw)) return [];
  const out: CustomThemeFile[] = [];
  for (const x of raw) {
    const r = checkCustomTheme(x);
    if (r.ok && !out.some((t) => t.id === r.theme.id)) out.push(r.theme);
  }
  return out;
}
