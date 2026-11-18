// Generates web/src/lib/design/tokens.css and tokens.ts from design/tokens.json, and the data line (`const T = ...;`)
// of index.html's first-paint script.
//   node scripts/tokens.mjs          write the files
//   node scripts/tokens.mjs --check  exit 1 if the committed files are stale
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { THEME_ID, builtinThemes, cssVar, leaves } from './themes.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const designDir = resolve(root, '..', 'design');
const outDir = resolve(root, 'src', 'lib', 'design');
const cssPath = resolve(outDir, 'tokens.css');
const tsPath = resolve(outDir, 'tokens.ts');
const indexPath = resolve(root, 'index.html');

const t = JSON.parse(readFileSync(resolve(designDir, 'tokens.json'), 'utf8'));
/** The built-in themes, resolved (base palette + overrides); throws on a malformed one. */
const themes = builtinThemes(t);
const HEX = /^#([0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$/;

const isMeta = (k) => k.startsWith('$');
const kebab = (s) => String(s).replace(/[_.]/g, '-');
const px = (n) => (n === 0 ? '0' : `${n}px`);

function hex(v, where) {
  if (typeof v !== 'string' || !HEX.test(v)) throw new Error(`${where}: expected #RRGGBB or #RRGGBBAA, got ${v}`);
  return v.toUpperCase();
}

/** #RRGGBBAA -> CSS. Modern browsers accept 8-digit hex directly. */
const css = (v, where) => hex(v, where).toLowerCase();

function shadow(layers, where) {
  return layers
    .map((l) => `${px(l.x)} ${px(l.y)} ${px(l.blur)} ${px(l.spread)} ${css(l.color, where)}`)
    .join(', ');
}

/** A palette's custom properties, in its key order (scripts/themes.mjs, cssVar). */
function colorVars(color, where) {
  return leaves(color).map(([path, v]) => [cssVar(path), typeof v === 'number' ? String(v) : css(v, `${where}.${path}`)]);
}

function themeVars(name) {
  const out = colorVars(t.color[name], `color.${name}`);
  // The mode's default area opacity: a theme's chart.area_opacity scales the charts' area fills relative to it
  // (LineChart, Sparkline), so the default themes draw them exactly as before. Not a colour token: no theme sets it.
  const area = t.color[name].chart?.area_opacity;
  if (typeof area !== 'number' || !(area > 0)) throw new Error(`color.${name}.chart.area_opacity: expected a number above 0`);
  out.push(['--chart-area-opacity-default', String(area)]);
  for (const [k, layers] of Object.entries(t.shadow[name])) {
    if (isMeta(k)) continue;
    out.push([`--shadow-${k}`, shadow(layers, `shadow.${name}.${k}`)]);
  }
  return out;
}

function staticVars() {
  const out = [];
  const fam = (f) => [`"${f.family}"`, ...f.fallback.map((x) => (/\s/.test(x) ? `"${x}"` : x))].join(', ');
  out.push(['--font-sans', fam(t.font.sans)], ['--font-mono', fam(t.font.mono)]);
  for (const [k, v] of Object.entries(t.font.weight)) out.push([`--weight-${k}`, String(v)]);
  for (const [k, s] of Object.entries(t.type)) {
    if (isMeta(k)) continue;
    out.push(
      [`--type-${k}-size`, px(s.size)],
      [`--type-${k}-line`, px(s.line)],
      [`--type-${k}-tracking`, s.tracking === 0 ? '0' : `${s.tracking}em`],
      [`--type-${k}-weight`, String(s.weight)],
    );
  }
  for (const [k, v] of Object.entries(t.space)) out.push([`--space-${kebab(k)}`, px(v)]);
  for (const [k, v] of Object.entries(t.radius)) out.push([`--radius-${k}`, px(v)]);
  for (const [k, v] of Object.entries(t.size)) {
    if (isMeta(k)) continue;
    if (typeof v === 'object') for (const [k2, v2] of Object.entries(v)) out.push([`--${kebab(k)}-${k2}`, px(v2)]);
    else out.push([`--size-${kebab(k)}`, px(v)]);
  }
  for (const [k, v] of Object.entries(t.motion.duration)) out.push([`--dur-${k}`, `${v}ms`]);
  for (const [k, v] of Object.entries(t.motion.easing)) out.push([`--ease-${k}`, `cubic-bezier(${v.join(', ')})`]);
  for (const [k, v] of Object.entries(t.z)) out.push([`--z-${k}`, String(v)]);
  t.classes.palette.forEach((c, i) => out.push([`--class-${i}`, css(c, `classes.palette[${i}]`)]));
  out.push(
    ['--class-chip-text', css(t.classes.chip_text, 'classes.chip_text')],
    ['--class-halo', css(t.classes.halo, 'classes.halo')],
    ['--class-fill-opacity', String(t.classes.fill_opacity)],
  );
  return out;
}

const block = (sel, vars) => `${sel} {\n${vars.map(([k, v]) => `  ${k}: ${v};`).join('\n')}\n}\n`;

function fontFaces() {
  const rel = relative(outDir, designDir).replace(/\\/g, '/');
  const faces = [];
  for (const f of [t.font.sans, t.font.mono]) {
    for (const w of f.weights) {
      const file = f.files[String(w)];
      if (!file) throw new Error(`font ${f.family}: no file for weight ${w}`);
      if (!existsSync(resolve(designDir, `${file}.woff2`))) throw new Error(`missing ${file}.woff2`);
      faces.push(
        `@font-face {\n  font-family: "${f.family}";\n  font-style: normal;\n  font-weight: ${w};\n  font-display: swap;\n  src: url("${rel}/${file}.woff2") format("woff2");\n}\n`,
      );
    }
  }
  return faces.join('');
}

function buildCss() {
  const dark = themeVars('dark');
  const light = themeVars('light');
  const reduced = Object.keys(t.motion.duration).map((k) => [`--dur-${k}`, '0ms']);
  return [
    '/* Generated by web/scripts/tokens.mjs from design/tokens.json. Do not edit by hand. */\n',
    fontFaces(),
    block(':root', staticVars()),
    block(':root,\n:root[data-theme="dark"]', [['color-scheme', 'dark'], ...dark]),
    block(':root[data-theme="light"]', [['color-scheme', 'light'], ...light]),
    // Each non-default theme's whole palette, after the mode blocks so it wins: `data-theme` stays the mode (its
    // shadows and every [data-theme='light'] rule), `data-theme-id` names the theme (design/THEMES.md).
    ...themes.filter((th) => th.id !== th.mode).map((th) => block(`:root[data-theme-id="${th.id}"]`, colorVars(th.color, `themes.${th.id}.color`))),
    `@media (prefers-reduced-motion: reduce) {\n  :root {\n${reduced.map(([k, v]) => `    ${k}: ${v};`).join('\n')}\n  }\n}\n`,
  ].join('\n');
}

function buildTs() {
  const data = {
    classPalette: t.classes.palette.map((c) => hex(c, 'palette')),
    classChipText: hex(t.classes.chip_text, 'chip_text'),
    classHalo: hex(t.classes.halo, 'halo'),
    classFillOpacity: t.classes.fill_opacity,
    duration: t.motion.duration,
    easing: t.motion.easing,
    size: Object.fromEntries(Object.entries(t.size).filter(([k]) => !isMeta(k))),
  };
  const bp = t.breakpoint;
  const tiers = Object.entries(bp).filter(([k]) => !isMeta(k));
  if (tiers.map(([k]) => k).join() !== 'tablet,laptop,desktop' || !tiers.every(([, v], i) => i === 0 || v > tiers[i - 1][1]))
    throw new Error('breakpoint: expected ascending tablet, laptop, desktop');
  data.breakpoint = Object.fromEntries(tiers);
  // The tier queries, written exactly as CSS writes them (custom media cannot use variables; lint-spacing.mjs accepts
  // only these widths). `upTo*` is everything below that tier, `*Up` that tier and wider.
  data.media = {
    phone: `(max-width: ${bp.tablet - 1}px)`,
    tabletUp: `(min-width: ${bp.tablet}px)`,
    upToLaptop: `(max-width: ${bp.laptop - 1}px)`,
    laptopUp: `(min-width: ${bp.laptop}px)`,
    upToDesktop: `(max-width: ${bp.desktop - 1}px)`,
    desktop: `(min-width: ${bp.desktop}px)`,
  };
  const body = Object.entries(data)
    .map(([k, v]) => `export const ${k} = ${JSON.stringify(v, null, 2)} as const;\n`)
    .join('\n');
  return `// Generated by web/scripts/tokens.mjs from design/tokens.json. Do not edit by hand.\n\n${body}
/** Colour for detection class \`i\` (wraps around the 16-colour palette). */
export function classColor(i: number): string {
  const n = classPalette.length;
  return classPalette[((i % n) + n) % n] ?? classPalette[0];
}

${themesTs()}`;
}

/** The palette's TypeScript shape, from the dark base (both bases have the same keys; contrast.mjs checks that). */
function paletteType(color, indent = '  ') {
  const lines = [];
  for (const [k, v] of Object.entries(color)) {
    if (isMeta(k)) continue;
    if (Array.isArray(v)) lines.push(`${indent}readonly ${k}: readonly string[];`);
    else if (v !== null && typeof v === 'object') lines.push(`${indent}readonly ${k}: {\n${paletteType(v, `${indent}  `)}\n${indent}};`);
    else lines.push(`${indent}readonly ${k}: ${typeof v === 'number' ? 'number' : 'string'};`);
  }
  return lines.join('\n');
}

/** A palette as TS, one line per group, so seven palettes stay a short read. */
function paletteTs(color, indent) {
  const lines = Object.entries(color).map(([k, v]) => `${indent}  ${k}: ${JSON.stringify(v)},`);
  return `{\n${lines.join('\n')}\n${indent}}`;
}

/** tokens.ts's themes: the palette type, the ids in order, every built-in theme resolved, and the token -> CSS table. */
function themesTs() {
  const ids = themes.map((th) => th.id);
  const entries = themes
    .map(
      (th) =>
        `  ${JSON.stringify(th.id)}: {\n    id: ${JSON.stringify(th.id)},\n    name: ${JSON.stringify(th.name)},\n    mode: ${JSON.stringify(th.mode)},\n` +
        `    description: ${JSON.stringify(th.description)},\n    color: ${paletteTs(th.color, '    ')},\n  },`,
    )
    .join('\n');
  const vars = leaves(t.color.dark)
    .map(([path]) => `  [${JSON.stringify(path)}, ${JSON.stringify(cssVar(path))}],`)
    .join('\n');
  return `/** A theme's colours: the shape of \`color.dark\` in tokens.json (design/THEMES.md). */
export interface Palette {
${paletteType(t.color.dark)}
}

/** The built-in themes' ids, in the order Settings and the palette list them; the first of each mode is its default. */
export const themeIds = ${JSON.stringify(ids)} as const;
export type ThemeId = (typeof themeIds)[number];

/** A theme id's pattern (design/THEMES.md, \`id\`), as a RegExp source: index.html's first-paint script has the same. */
export const themeIdPattern = ${JSON.stringify(THEME_ID.source)} as const;

export interface BuiltinTheme {
  readonly id: ThemeId;
  readonly name: string;
  readonly mode: 'dark' | 'light';
  readonly description: string;
  /** The resolved palette: the mode's base with the theme's overrides. */
  readonly color: Palette;
}

/** Every built-in theme, resolved. */
export const themes: Readonly<Record<ThemeId, BuiltinTheme>> = {
${entries}
};

/** Every colour token path (arrays by index) and its CSS custom property, in tokens.css's order. */
export const colorVars: readonly (readonly [string, string])[] = [
${vars}
];
`;
}
/**
 * What index.html's first-paint script needs to check a stored custom theme as src/lib/design/themes.ts does and set
 * only colour token properties: the built-in ids of each mode, the id pattern (themes.mjs's THEME_ID), the palette's
 * shape with every colour token as its CSS custom property (a list token: one per entry), and the number tokens'
 * properties.
 */
function firstPaintData() {
  /** @type {string[]} */
  const number = [];
  /** @returns {Record<string, unknown>} */
  const shape = (/** @type {Record<string, unknown>} */ color, /** @type {string} */ prefix) =>
    Object.fromEntries(
      Object.entries(color)
        .filter(([k]) => !isMeta(k))
        .map(([k, v]) => {
          const path = prefix ? `${prefix}.${k}` : k;
          if (Array.isArray(v)) return [k, v.map((_, i) => cssVar(`${path}.${i}`))];
          if (v !== null && typeof v === 'object') return [k, shape(/** @type {Record<string, unknown>} */ (v), path)];
          if (typeof v === 'number') number.push(cssVar(path));
          return [k, cssVar(path)];
        }),
    );
  const color = shape(t.color.dark, '');
  const ids = (/** @type {string} */ mode) => themes.filter((th) => th.mode === mode).map((th) => th.id);
  return { dark: ids('dark'), light: ids('light'), id: THEME_ID.source, color, number };
}

/** index.html with its first-paint data line regenerated. */
function buildIndex() {
  const html = readFileSync(indexPath, 'utf8');
  const line = /(<script id="first-paint">[\s\S]*?\n[ \t]*const T = )[^\n]*;\n/;
  if (!line.test(html)) throw new Error('index.html: the first-paint script has no `const T = ...;` line');
  return html.replace(line, (_, head) => `${head}${JSON.stringify(firstPaintData())};\n`);
}

const files = [
  [cssPath, buildCss()],
  [tsPath, buildTs()],
  [indexPath, buildIndex()],
];

if (process.argv.includes('--check')) {
  const stale = files.filter(([p, s]) => !existsSync(p) || readFileSync(p, 'utf8') !== s);
  if (stale.length) {
    console.error(`stale generated tokens: ${stale.map(([p]) => relative(root, p)).join(', ')}; run npm run tokens`);
    process.exit(1);
  }
  console.log('tokens up to date');
} else {
  for (const [p, s] of files) writeFileSync(p, s);
  console.log(`wrote ${files.map(([p]) => relative(root, p)).join(', ')}`);
}
