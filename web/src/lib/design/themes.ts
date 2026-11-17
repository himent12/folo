// Colour themes (design/THEMES.md): the built-in ones from tokens.ts, custom theme files checked exactly as the spec
// says (every error names its token path), a theme resolved to its palette and to what <html> needs to show it
// (`data-theme-id` and, for a custom theme, its overrides as inline custom properties), and each mode's list. Pure:
// prefs.ts keeps the choice and the installed files, App.tsx applies the result, index.html's first-paint script does
// the same before any of this has loaded (its rules are tested against these, firstPaint.test.ts). `readThemeFile` is
// the one impure part: it reads a picked File (Blob API only).

// The JSON Desktop's serde_json refuses and JSON.parse accepts (depth, lone surrogate escapes, numbers out of range),
// shared with design/contrast.mjs so the two JS readers can't drift.
import { jsonProblem } from '../../../../design/json-scan.mjs';
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
const ID = /^[a-z0-9][a-z0-9-]{0,39}$/;
const KEYS = ['folo_theme', 'id', 'name', 'mode', 'base', 'description', 'color'];
const NAME_MAX = 40;
const DESCRIPTION_MAX = 120;
/** Not a custom theme id: Desktop's `--theme system` means the mode. */
export const RESERVED_ID = 'system';
/** The largest theme file, 1 MiB (1 048 576 bytes): every reader refuses a larger one (a theme is a few kB). */
export const THEME_FILE_MAX = 1 << 20;
/** The errors of a file that can't be a theme, the same in every reader (design/THEMES.md, Custom themes). */
export const TOO_LARGE = '(file): larger than 1 MB; not a theme';
export const NOT_UTF8 = '(file): not UTF-8 text';
/** Strict UTF-8: invalid bytes are an error, not U+FFFD; a byte order mark is left for `parseThemeBytes` to strip once. */
const utf8 = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });

/** Trims Unicode white space (the White_Space property, as Rust's `str::trim`; not String.prototype.trim's set). */
const trim = (s: string) => s.replace(/^\p{White_Space}+|\p{White_Space}+$/gu, '');
/** Length in code points, as Rust's `chars().count()` (not UTF-16 units). */
const length = (s: string) => [...s].length;

/** Orders two strings by code point (as Rust compares strings), not by UTF-16 unit or locale. */
export function compareCodePoints(a: string, b: string): number {
  const x = a[Symbol.iterator]();
  const y = b[Symbol.iterator]();
  for (;;) {
    const p = x.next();
    const q = y.next();
    if (p.done || q.done) return (p.done ? 0 : 1) - (q.done ? 0 : 1);
    const d = p.value.codePointAt(0)! - q.value.codePointAt(0)!;
    if (d) return d;
  }
}

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
  if (!isObj(over)) throw new Error(path === 'color' ? 'color: must be an object' : `${path}: must be an object of colour tokens`);
  const out: Record<string, unknown> = { ...base };
  const notColour = (at: string) => new Error(`${at}: not a #RRGGBB or #RRGGBBAA colour`);
  // Depth first in the file's order: the first error is the first a reader of the file meets.
  for (const [k, v] of Object.entries(over)) {
    if (isMeta(k)) continue;
    const at = `${path}.${k}`;
    if (!Object.hasOwn(base, k)) throw new Error(`${at}: not a colour token`);
    const b = base[k];
    if (isObj(b)) out[k] = merge(b, v, at);
    else if (Array.isArray(b)) {
      if (!Array.isArray(v) || v.length !== b.length) throw new Error(`${at}: must be a list of ${b.length} colours`);
      v.forEach((x, i) => {
        if (typeof x !== 'string' || !HEX.test(x)) throw notColour(`${at}.${i}`);
      });
      out[k] = [...v];
    } else if (typeof b === 'number') {
      if (typeof v !== 'number' || !(v >= 0 && v <= 1)) throw new Error(`${at}: must be a number from 0 to 1`);
      out[k] = v;
    } else {
      if (typeof v !== 'string' || !HEX.test(v)) throw notColour(at);
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

/**
 * The checks in the order every validator runs them (design/theme-cases.json, which design/contrast.mjs and Desktop
 * replay too): unknown top-level keys, then folo_theme, id, name, description, mode, base, then `color` depth first.
 */
function check(raw: unknown): CustomThemeFile {
  if (!isObj(raw)) throw new Error('(file): must be a JSON object');
  for (const k of Object.keys(raw)) if (!isMeta(k) && !KEYS.includes(k)) throw new Error(`${k}: unknown key`);
  const need = (k: string) => {
    if (raw[k] === undefined) throw new Error(`${k}: required`);
    return raw[k];
  };
  // Any JSON number equal to 1: JSON.parse reads 1.0 and 1e0 as 1.
  if (need('folo_theme') !== THEME_FORMAT) throw new Error(`folo_theme: must be ${THEME_FORMAT}`);
  const id = need('id');
  if (typeof id !== 'string' || !ID.test(id)) throw new Error('id: must be 1 to 40 characters of a-z, 0-9 and -, starting with a letter or digit');
  if (isBuiltinId(id)) throw new Error(`id: "${id}" is a built-in theme`);
  if (id === RESERVED_ID) throw new Error(`id: "${RESERVED_ID}" is reserved`);
  const name = need('name');
  const trimmed = typeof name === 'string' ? trim(name) : '';
  if (!trimmed || length(trimmed) > NAME_MAX) throw new Error(`name: must be 1 to ${NAME_MAX} characters`);
  const rawDescription = raw.description;
  const description = typeof rawDescription === 'string' ? trim(rawDescription) : undefined;
  if (rawDescription !== undefined && (description === undefined || length(description) > DESCRIPTION_MAX))
    throw new Error(`description: must be at most ${DESCRIPTION_MAX} characters`);
  const mode = need('mode');
  if (mode !== 'dark' && mode !== 'light') throw new Error('mode: must be "dark" or "light"');
  const base = raw.base;
  if (base !== undefined) {
    if (typeof base !== 'string' || !isBuiltinId(base)) throw new Error('base: not a built-in theme');
    if (builtins[base].mode !== mode) throw new Error(`base: "${base}" is a ${builtins[base].mode} theme, not ${mode}`);
  }
  const color = raw.color;
  if (color !== undefined) merge(builtins[(base as ThemeId | undefined) ?? defaultThemeId(mode)].color as unknown as Record<string, unknown>, color, 'color');
  const out: CustomThemeFile = { folo_theme: 1, id, name: trimmed, mode };
  if (base !== undefined) out.base = base as ThemeId;
  if (description !== undefined) out.description = description;
  if (color !== undefined) out.color = strip(color) as Record<string, unknown>;
  return out;
}

/**
 * Parses and checks a file's text (decoded, its byte order mark gone: a U+FEFF still at the start is not JSON); a file
 * that is not JSON says so. So does one Desktop's serde_json refuses while parsing where JSON.parse would accept it
 * (design/json-scan.mjs, checked first): nesting past serde_json's limit, a lone surrogate escape in any string (a key
 * too, or a member a later duplicate replaces), a number out of a double's range. Duplicate members are otherwise
 * allowed, the last one winning, as in serde_json.
 */
export function parseThemeFile(text: string): { ok: true; theme: CustomThemeFile } | { ok: false; error: string } {
  const problem = jsonProblem(text);
  if (problem) return { ok: false, error: `(file): not valid JSON (${problem})` };
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    return { ok: false, error: `(file): not valid JSON (${(e as Error).message})` };
  }
  return checkCustomTheme(raw);
}

/**
 * A theme file's bytes, checked as every reader does (design/THEMES.md): exactly one leading UTF-8 byte order mark
 * (EF BB BF) is stripped, the rest decoded strictly as UTF-8 (`(file): not UTF-8 text` otherwise), then parsed.
 */
export function parseThemeBytes(bytes: Uint8Array): { ok: true; theme: CustomThemeFile } | { ok: false; error: string } {
  const body = bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf ? bytes.subarray(3) : bytes;
  let text: string;
  try {
    text = utf8.decode(body);
  } catch {
    return { ok: false, error: NOT_UTF8 };
  }
  return parseThemeFile(text);
}

/** A file picked for import, read and checked: its result alone (the file's bytes and text are not kept). */
export type ParsedFile = { file: string; ok: true; theme: CustomThemeFile } | { file: string; ok: false; error: string };

/**
 * Reads one picked file and checks it (`parseThemeBytes`). One over THEME_FILE_MAX is refused by its size, unread, and
 * the limit holds on what is read too (a file that grew after it was picked). It never rejects: anything that throws
 * while one file is read or checked is that file's error, so the rest of a pick still goes ahead.
 */
export async function readThemeFile(f: File): Promise<ParsedFile> {
  const file = f.name;
  let bytes: Uint8Array;
  try {
    if (f.size > THEME_FILE_MAX) return { file, ok: false, error: TOO_LARGE };
    bytes = new Uint8Array(await f.arrayBuffer());
  } catch (e) {
    return { file, ok: false, error: `(file): couldn't read (${messageOf(e)})` };
  }
  if (bytes.length > THEME_FILE_MAX) return { file, ok: false, error: TOO_LARGE };
  try {
    return { file, ...parseThemeBytes(bytes) };
  } catch (e) {
    return { file, ok: false, error: `(file): couldn't check (${messageOf(e)})` };
  }
}

/** What a thrown value says (an Error's message; anything else as text). */
const messageOf = (e: unknown) => (e instanceof Error ? e.message : String(e));

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

/** The themes of `mode`: the built-ins in their order, then the custom ones by name (`themeCatalog`). */
export function themesFor(mode: Mode, customs: readonly CustomThemeFile[]): readonly Theme[] {
  return themeCatalog(customs)[mode];
}

/** Every theme resolved once: each mode's list (as `themesFor` orders it), the custom ones and every theme by id. */
export interface ThemeCatalog {
  readonly dark: readonly Theme[];
  readonly light: readonly Theme[];
  /** The custom themes of both modes in the lists' order (by name): Settings lists the installed ones so, as Desktop. */
  readonly custom: readonly Theme[];
  readonly byId: ReadonlyMap<string, Theme>;
}

/**
 * The catalog of the built-ins and `customs`, each resolved once. Custom themes follow the built-ins of their mode in
 * the order of every list (design/THEMES.md, and Desktop's): by lower-cased name, then by id, both by code point (no
 * locale collation, so the two apps list them alike).
 */
export function themeCatalog(customs: readonly CustomThemeFile[]): ThemeCatalog {
  const byId = new Map<string, Theme>(themeIds.map((id) => [id, builtinTheme(id)]));
  const mine = customs.map((f) => ({ t: customTheme(f), key: f.name.toLowerCase() }));
  mine.sort((a, b) => compareCodePoints(a.key, b.key) || compareCodePoints(a.t.id, b.t.id));
  for (const { t } of mine) if (!byId.has(t.id)) byId.set(t.id, t);
  const custom = mine.map(({ t }) => t).filter((t) => byId.get(t.id) === t);
  const list = (mode: Mode) => [...themeIds.filter((id) => builtins[id].mode === mode).map((id) => byId.get(id)!), ...custom.filter((t) => t.mode === mode)];
  return { dark: list('dark'), light: list('light'), custom, byId };
}

/** What an import did with one file. */
export type ImportResult = { file: string; ok: true; id: string; name: string; replaced: boolean } | { file: string; ok: false; error: string };

/**
 * Adds the files that passed the check (`readThemeFile`) to `installed` (a theme whose id is installed already
 * replaces it, in place) and says per file what happened. Files go in order, so of two with one id the later wins.
 * `accepted`: the files that passed, in order (installing them one by one over `installed` gives `themes`).
 */
export function importThemes(
  installed: readonly CustomThemeFile[],
  files: readonly ParsedFile[],
): { themes: CustomThemeFile[]; accepted: CustomThemeFile[]; results: ImportResult[] } {
  const themes = [...installed];
  const accepted: CustomThemeFile[] = [];
  const results: ImportResult[] = files.map((r) => {
    if (!r.ok) return r;
    accepted.push(r.theme);
    const at = themes.findIndex((t) => t.id === r.theme.id);
    if (at >= 0) themes[at] = r.theme;
    else themes.push(r.theme);
    return { file: r.file, ok: true, id: r.theme.id, name: r.theme.name, replaced: at >= 0 };
  });
  return { themes, accepted, results };
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
