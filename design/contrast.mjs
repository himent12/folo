// Colour checks for tokens.json, in every theme (design/THEMES.md): WCAG 2 contrast for every pairing
// design/README.md documents, colour-vision separation for colours it documents as told apart, elevation by lightness,
// and the stricter floors of the high-contrast themes.
//   node design/contrast.mjs                  check every built-in theme and every design/themes/*.json, and replay the
//                                             format cases in theme-cases.json; exit 1 on any failure
//   node design/contrast.mjs --theme <file>   validate one custom theme file (THEMES.md, Custom themes), resolve it
//                                             on its base and check it; exit 1 if it is invalid or fails a check
//   --all                                     also print every measured pair
// Text needs 4.5:1; non-text marks (focus rings, dots, chart lines, bars) and large text need 3:1. High contrast themes
// (contrast-*, and custom themes based on one) need 7:1 for text and 4.5:1 for marks, and line.default 3:1 on bg.app.
// Translucent colours (tints, overlays, lines) are measured composited over each surface they sit on. Run by
// `npm run check` in web/.

import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { resolve as resolvePath } from 'node:path';
import { fileURLToPath } from 'node:url';

const tokens = JSON.parse(readFileSync(new URL('./tokens.json', import.meta.url), 'utf8'));
const TEXT = 4.5;
const MARK = 3;
const HC_TEXT = 7;
const HC_MARK = 4.5;
const SURFACES = ['bg.app', 'bg.panel', 'bg.raised', 'bg.overlay', 'bg.inset'];
const STATUS = ['success', 'warning', 'danger', 'info'];
const MODES = ['dark', 'light'];

// ---------------------------------------------------------------------------------------------------------------
// Themes: resolution (mode base + overrides, arrays whole) and validation of the custom format.

const isComment = (k) => k.startsWith('$');
const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const COLOUR = /^#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$/;

/** `over` merged over `base` key by key; arrays and scalars replace whole; `$` keys are skipped. */
function merge(base, over) {
  const out = structuredClone(base);
  for (const [k, v] of Object.entries(over ?? {})) {
    if (isComment(k)) continue;
    out[k] = isObject(v) && isObject(out[k]) ? merge(out[k], v) : structuredClone(v);
  }
  return out;
}

/** The built-in themes in listing order: { id, name, mode, description, color }. */
export const builtIns = Object.entries(tokens.themes)
  .filter(([id]) => !isComment(id))
  .map(([id, t]) => ({ id, ...t }));
const builtInById = new Map(builtIns.map((t) => [t.id, t]));

/** A built-in theme's full palette: its mode's base with its overrides merged over it. */
export function builtInPalette(id) {
  const t = builtInById.get(id);
  if (!t) throw new Error(`no built-in theme ${id}`);
  return merge(tokens.color[t.mode], t.color);
}

/**
 * Checks a `color` override object against the token shape (`color.dark`'s paths). Returns the first error as
 * `path: message`, or null. `base` is the palette it merges over (array lengths come from it).
 */
function validateColor(color, base, path) {
  if (!isObject(color)) return `${path}: must be an object`;
  for (const [k, v] of Object.entries(color)) {
    if (isComment(k)) continue;
    const p = `${path}.${k}`;
    if (!Object.hasOwn(base, k)) return `${p}: not a colour token`;
    const shape = base[k];
    if (isObject(shape)) {
      if (!isObject(v)) return `${p}: must be an object of colour tokens`;
      const e = validateColor(v, shape, p);
      if (e) return e;
    } else if (Array.isArray(shape)) {
      if (!Array.isArray(v) || v.length !== shape.length) return `${p}: must be a list of ${shape.length} colours`;
      const bad = v.findIndex((c) => typeof c !== 'string' || !COLOUR.test(c));
      if (bad >= 0) return `${p}.${bad}: not a #RRGGBB or #RRGGBBAA colour`;
    } else if (typeof shape === 'number') {
      if (typeof v !== 'number' || !Number.isFinite(v) || v < 0 || v > 1) return `${p}: must be a number from 0 to 1`;
    } else if (typeof v !== 'string' || !COLOUR.test(v)) {
      return `${p}: not a #RRGGBB or #RRGGBBAA colour`;
    }
  }
  return null;
}

const HEADER = ['folo_theme', 'id', 'name', 'mode', 'base', 'description', 'color'];

/**
 * Validates a parsed custom theme file (THEMES.md, Custom themes). Returns { error } with the first error as
 * `path: message`, or { theme: { id, name, mode, base, description, palette } } with its resolved palette.
 * Order: unknown top-level keys, folo_theme, id, name, description, mode, base, then color (depth first, file order).
 */
export function validateTheme(json) {
  const fail = (error) => ({ error });
  if (!isObject(json)) return fail('(file): must be a JSON object');
  for (const k of Object.keys(json)) if (!isComment(k) && !HEADER.includes(k)) return fail(`${k}: unknown key`);
  if (!('folo_theme' in json)) return fail('folo_theme: required');
  if (json.folo_theme !== 1) return fail('folo_theme: must be 1');
  if (!('id' in json)) return fail('id: required');
  if (typeof json.id !== 'string' || !/^[a-z0-9-]{1,40}$/.test(json.id)) {
    return fail('id: must be 1 to 40 characters of a-z, 0-9 and -');
  }
  if (builtInById.has(json.id)) return fail(`id: "${json.id}" is a built-in theme`);
  if (!('name' in json)) return fail('name: required');
  if (typeof json.name !== 'string' || json.name.trim().length < 1 || json.name.trim().length > 40) {
    return fail('name: must be 1 to 40 characters');
  }
  if ('description' in json && (typeof json.description !== 'string' || json.description.length > 120)) {
    return fail('description: must be at most 120 characters');
  }
  if (!('mode' in json)) return fail('mode: required');
  if (!MODES.includes(json.mode)) return fail('mode: must be "dark" or "light"');
  let base = json.mode;
  if ('base' in json) {
    const b = typeof json.base === 'string' ? builtInById.get(json.base) : undefined;
    if (!b) return fail('base: not a built-in theme');
    if (b.mode !== json.mode) return fail(`base: "${json.base}" is a ${b.mode} theme, not ${json.mode}`);
    base = b.id;
  }
  const basePalette = builtInPalette(base);
  if ('color' in json) {
    const e = validateColor(json.color, basePalette, 'color');
    if (e) return fail(e);
  }
  return {
    theme: {
      id: json.id,
      name: json.name.trim(),
      mode: json.mode,
      base,
      description: json.description,
      palette: merge(basePalette, json.color),
    },
  };
}

// ---------------------------------------------------------------------------------------------------------------
// Colour maths.

const hex = (h) => {
  const m = /^#([0-9a-f]{6})([0-9a-f]{2})?$/i.exec(h);
  if (!m) throw new Error(`not a #RRGGBB(AA) colour: ${h}`);
  const n = parseInt(m[1], 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255, a: m[2] ? parseInt(m[2], 16) / 255 : 1 };
};
const over = (top, under) => {
  const t = hex(top);
  const u = hex(under);
  const mix = (k) => Math.round(t[k] * t.a + u[k] * (1 - t.a));
  return `#${[mix('r'), mix('g'), mix('b')].map((v) => v.toString(16).padStart(2, '0')).join('')}`;
};
const luminance = (h) => {
  const lin = (v) => ((v /= 255) <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
  const { r, g, b } = hex(h);
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
};
export const contrast = (a, b) => {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
};

// Colours documented as told apart stay apart under colour-vision deficiency: OKLab distance (×100) under normal
// vision and Machado et al. (2009) full-severity protan / deutan / tritan simulation, in linear RGB.
const SEPARATION = 7;
const CVD = {
  normal: null,
  protan: [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
  deutan: [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.01182, 0.04294, 0.968881]],
  tritan: [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.3039]],
};
const toLinear = (h) => {
  const { r, g, b } = hex(h);
  return [r, g, b].map((v) => ((v /= 255) <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
};
const oklab = ([r, g, b]) => {
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
};
const seen = (h, mat) => {
  const c = toLinear(h);
  return mat ? mat.map((row) => Math.min(1, Math.max(0, row[0] * c[0] + row[1] * c[1] + row[2] * c[2]))) : c;
};
const distance = (a, b, mat) => 100 * Math.hypot(...oklab(seen(a, mat)).map((v, i) => v - oklab(seen(b, mat))[i]));

// ---------------------------------------------------------------------------------------------------------------
// The checks.

/**
 * Every documented pairing: `fg` on each of `on` (token paths; `[tint, surface]` is a tint over a surface). `kind`
 * says which floor applies: 'text' (4.5:1, 7:1 in high contrast) or 'mark' (3:1, 4.5:1 in high contrast).
 */
function rules(palette, hc) {
  const tinted = (tint) => SURFACES.map((s) => [tint, s]);
  return [
    // Colour: text tones meet AA on every surface.
    ...['text.primary', 'text.secondary', 'text.tertiary'].map((fg) => ({ fg, on: SURFACES, kind: 'text' })),
    // Accent: text in accent.text; accent.on is the label on a lime fill (and on accent.line).
    { fg: 'accent.text', on: SURFACES, kind: 'text' },
    { fg: 'accent.on', on: ['accent.solid', 'accent.solid_hover'], kind: 'text' },
    // accent.on on accent.line (a step number on an olive dot in light). In a high contrast light theme accent.line
    // must be 4.5:1 on white and near-white surfaces, which leaves no shade a near-black label also reaches 4.5:1 on;
    // the mark wins there and this label holds 3:1, the one documented pairing a theme may hold at a lower floor.
    { fg: 'accent.on', on: ['accent.line'], kind: hc ? 'mark' : 'text', min: hc ? MARK : undefined },
    // Non-text marks that must be seen against the surface: accent.line and focus rings.
    { fg: 'accent.line', on: SURFACES, kind: 'mark' },
    { fg: 'focus', on: SURFACES, kind: 'mark' },
    // Control boundaries (WCAG 1.4.11): switch tracks, slider rails, checkbox / radio and input edges.
    { fg: 'line.control', on: SURFACES, kind: 'mark' },
    // High contrast: panel edges stand out against the app.
    ...(hc ? [{ fg: 'line.default', on: ['bg.app'], kind: 'mark', min: MARK }] : []),
    // Status: text on every surface and on its own tint (badges, banners); solid dots and icons are marks.
    ...STATUS.map((s) => ({ fg: `${s}.text`, on: [...SURFACES, ...tinted(`${s}.muted`)], kind: 'text' })),
    ...STATUS.map((s) => ({ fg: `${s}.solid`, on: SURFACES, kind: 'mark' })),
    // The solid destructive button (ui/Button `destructive`, a dialog's confirm): its label on its fill, at rest and on
    // hover (round 24: white on danger.solid was 4.37:1 in light, and the hover mix made it worse).
    { fg: 'text.inverse', on: ['danger.fill', 'danger.fill_hover'], kind: 'text' },
    // Waiting states: text.secondary on bg.active.
    { fg: 'text.secondary', on: tinted('bg.active'), kind: 'text' },
    // Charts: tick labels are text on both plot surfaces; series and context curves are lines.
    { fg: 'chart.label', on: ['bg.panel', 'bg.inset'], kind: 'text' },
    ...palette.chart.series.map((_, i) => ({ fg: `chart.series.${i}`, on: ['bg.panel', 'bg.inset'], kind: 'mark' })),
    { fg: 'chart.context', on: ['bg.panel'], kind: 'mark' },
    // Series-coloured text (direct labels, JSON tokens, any 12-13 px text in a series hue) is chart.text, text on
    // every surface: a series' line ink is a mark (3:1) and too light for small text in light.
    ...palette.chart.text.map((_, i) => ({ fg: `chart.text.${i}`, on: SURFACES, kind: 'text' })),
  ].map((r) => ({ ...r, min: r.min ?? (r.kind === 'text' ? (hc ? HC_TEXT : TEXT) : hc ? HC_MARK : MARK) }));
}

/** Sets whose members must be told apart: neighbouring chart series, and the loss components (box, cls, dfl). */
const SETS = [
  ...[0, 1, 2, 3, 4].map((i) => ({ what: `chart.series ${i} / ${i + 1}`, paths: [`chart.series.${i}`, `chart.series.${i + 1}`] })),
  { what: 'loss box / cls / dfl', paths: ['chart.series.1', 'chart.series.4', 'chart.series.3'] },
];

/** Elevation by lightness (THEMES.md): each pair is [lower, higher, strict]. */
const ELEVATION = {
  dark: [['bg.inset', 'bg.app', true], ['bg.app', 'bg.panel', true], ['bg.panel', 'bg.raised', false], ['bg.raised', 'bg.overlay', false]],
  light: [['bg.inset', 'bg.app', true], ['bg.app', 'bg.raised', false], ['bg.raised', 'bg.panel', false], ['bg.raised', 'bg.overlay', false], ['bg.overlay', 'bg.panel', false]],
};

function get(palette, path) {
  let v = palette;
  for (const k of path.split('.')) v = v?.[k];
  if (typeof v !== 'string') throw new Error(`no colour at color.${path}`);
  return v;
}

/**
 * Runs every check on one resolved theme. Returns { checked, failures, passes }, each line naming the pair, its
 * ratio and its floor.
 */
export function checkTheme({ mode, palette, highContrast }) {
  const failures = [];
  const passes = [];
  let checked = 0;
  const record = (ok, line) => {
    checked++;
    (ok ? passes : failures).push(line);
  };
  for (const { fg, on, min } of rules(palette, highContrast)) {
    for (const bg of on) {
      const [tint, surface] = Array.isArray(bg) ? bg : [null, bg];
      const ground = tint ? over(get(palette, tint), get(palette, surface)) : get(palette, surface);
      const ink = get(palette, fg);
      const ratio = contrast(hex(ink).a < 1 ? over(ink, ground) : ink, ground);
      const where = tint ? `${tint} over ${surface}` : surface;
      record(ratio >= min, `${fg.padEnd(18)} on ${where.padEnd(30)} ${ratio.toFixed(2).padStart(5)} (min ${min})`);
    }
  }
  for (const [lo, hi, strict] of ELEVATION[mode]) {
    const [a, b] = [luminance(get(palette, lo)), luminance(get(palette, hi))];
    const ok = strict ? a < b : a <= b;
    const line = `elevation: ${lo} ${get(palette, lo)} ${strict ? '<' : '<='} ${hi} ${get(palette, hi)} in lightness`;
    record(ok, line);
  }
  for (const { what, paths } of SETS) {
    for (let i = 0; i < paths.length; i++) {
      for (let j = i + 1; j < paths.length; j++) {
        for (const [vision, mat] of Object.entries(CVD)) {
          const d = distance(get(palette, paths[i]), get(palette, paths[j]), mat);
          const line = `${what}: ${paths[i]} vs ${paths[j]} (${vision})`.padEnd(72) + `ΔE ${d.toFixed(1).padStart(5)} (min ${SEPARATION})`;
          record(d >= SEPARATION, line);
        }
      }
    }
  }
  return { checked, failures, passes };
}

const isHighContrast = (id) => id.startsWith('contrast-');

// ---------------------------------------------------------------------------------------------------------------
// Command line.

function readThemeFile(file) {
  let json;
  try {
    json = JSON.parse(readFileSync(file, 'utf8'));
  } catch (e) {
    return { error: `(file): ${e.code === 'ENOENT' ? 'not found' : `not valid JSON (${e.message})`}` };
  }
  return validateTheme(json);
}

function main(argv) {
  const all = argv.includes('--all');
  const at = argv.indexOf('--theme');
  const targets = [];
  let invalid = 0;
  let replayed = 0;
  if (at >= 0) {
    const file = argv[at + 1];
    if (!file) {
      console.error('contrast: --theme needs a file');
      return 2;
    }
    const { error, theme } = readThemeFile(resolvePath(file));
    if (error) {
      console.error(`contrast: ${file}: ${error}`);
      return 1;
    }
    targets.push({ label: `${theme.id} (${file}, ${theme.mode} on ${theme.base})`, ...theme, highContrast: isHighContrast(theme.base) });
  } else {
    for (const t of builtIns) {
      const e = MODES.includes(t.mode) ? validateColor(t.color, tokens.color[t.mode], `themes.${t.id}.color`) : `themes.${t.id}.mode: must be "dark" or "light"`;
      if (e) {
        console.error(`contrast: tokens.json: ${e}`);
        invalid++;
        continue;
      }
      targets.push({ label: `${t.id} (${t.mode})`, id: t.id, mode: t.mode, palette: builtInPalette(t.id), highContrast: isHighContrast(t.id) });
    }
    // The shared format cases (theme-cases.json): the validator still refuses and accepts what the other apps' do.
    const cases = JSON.parse(readFileSync(new URL('./theme-cases.json', import.meta.url), 'utf8'));
    replayed = cases.invalid.length + cases.valid.length;
    for (const { what, theme, error } of cases.invalid) {
      const got = validateTheme(theme).error;
      if (got !== error) {
        console.error(`contrast: theme-cases.json, invalid "${what}": expected ${JSON.stringify(error)}, got ${JSON.stringify(got)}`);
        invalid++;
      }
    }
    for (const { what, theme } of cases.valid) {
      const got = validateTheme(theme).error;
      if (got) {
        console.error(`contrast: theme-cases.json, valid "${what}": refused with ${JSON.stringify(got)}`);
        invalid++;
      }
    }
    const dir = fileURLToPath(new URL('./themes/', import.meta.url));
    const files = existsSync(dir) ? readdirSync(dir).filter((f) => f.endsWith('.json')).sort() : [];
    for (const f of files) {
      const { error, theme } = readThemeFile(resolvePath(dir, f));
      if (error) {
        console.error(`contrast: design/themes/${f}: ${error}`);
        invalid++;
        continue;
      }
      targets.push({ label: `${theme.id} (themes/${f}, ${theme.mode} on ${theme.base})`, ...theme, highContrast: isHighContrast(theme.base) });
    }
  }

  let checked = 0;
  let failed = 0;
  for (const t of targets) {
    const r = checkTheme(t);
    checked += r.checked;
    failed += r.failures.length;
    const floors = t.highContrast ? `text ${HC_TEXT}:1, marks ${HC_MARK}:1` : `text ${TEXT}:1, marks ${MARK}:1`;
    if (r.failures.length) {
      console.error(`FAIL ${t.label}: ${r.failures.length} of ${r.checked} checks below the minimum (${floors})`);
      for (const f of r.failures) console.error(`       ${f}`);
    } else {
      console.log(`ok   ${t.label}: ${r.checked} checks (${floors}, CVD ΔE ${SEPARATION})`);
    }
    if (all) for (const p of r.passes) console.log(`       ${p}`);
  }
  if (failed || invalid) {
    const parts = [failed && `${failed} of ${checked} checks below the minimum`, invalid && `${invalid} invalid theme${invalid > 1 ? 's' : ''} or format case${invalid > 1 ? 's' : ''}`];
    console.error(`contrast: ${parts.filter(Boolean).join(', ')} (design/README.md, Colour; design/THEMES.md)`);
    return 1;
  }
  const formats = replayed ? `; the ${replayed} format cases of theme-cases.json hold` : '';
  console.log(`contrast: all ${checked} colour checks pass in ${targets.length} theme${targets.length > 1 ? 's' : ''}${formats}`);
  return 0;
}

if (process.argv[1] && resolvePath(process.argv[1]) === fileURLToPath(import.meta.url)) process.exitCode = main(process.argv.slice(2));
