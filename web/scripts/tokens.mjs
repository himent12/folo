// Generates web/src/lib/design/tokens.css and tokens.ts from design/tokens.json.
//   node scripts/tokens.mjs          write the files
//   node scripts/tokens.mjs --check  exit 1 if the committed files are stale
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const designDir = resolve(root, '..', 'design');
const outDir = resolve(root, 'src', 'lib', 'design');
const cssPath = resolve(outDir, 'tokens.css');
const tsPath = resolve(outDir, 'tokens.ts');

const t = JSON.parse(readFileSync(resolve(designDir, 'tokens.json'), 'utf8'));
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

function themeVars(name) {
  const c = t.color[name];
  const out = [];
  for (const [group, val] of Object.entries(c)) {
    if (isMeta(group)) continue;
    if (typeof val === 'string') {
      out.push([`--${kebab(group)}`, css(val, `color.${name}.${group}`)]);
      continue;
    }
    for (const [k, v] of Object.entries(val)) {
      if (isMeta(k)) continue;
      const where = `color.${name}.${group}.${k}`;
      if (group === 'chart' && k === 'series') {
        v.forEach((s, i) => out.push([`--chart-${i + 1}`, css(s, `${where}[${i}]`)]));
      } else if (group === 'chart' && k === 'fill') {
        v.forEach((s, i) => out.push([`--chart-fill-${i + 1}`, css(s, `${where}[${i}]`)]));
      } else if (typeof v === 'number') {
        out.push([`--${kebab(group)}-${kebab(k)}`, String(v)]);
      } else {
        out.push([`--${kebab(group)}-${kebab(k)}`, css(v, where)]);
      }
    }
  }
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
    `@media (prefers-reduced-motion: reduce) {\n  :root {\n${reduced.map(([k, v]) => `    ${k}: ${v};`).join('\n')}\n  }\n}\n`,
  ].join('\n');
}

function buildTs() {
  const theme = (name) => {
    const c = t.color[name];
    return {
      series: c.chart.series.map((s) => hex(s, 'series')),
      grid: hex(c.chart.grid, 'grid'),
      axis: hex(c.chart.axis, 'axis'),
      label: hex(c.chart.label, 'label'),
      areaOpacity: c.chart.area_opacity,
    };
  };
  const data = {
    classPalette: t.classes.palette.map((c) => hex(c, 'palette')),
    classChipText: hex(t.classes.chip_text, 'chip_text'),
    classHalo: hex(t.classes.halo, 'halo'),
    classFillOpacity: t.classes.fill_opacity,
    chart: { dark: theme('dark'), light: theme('light') },
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
`;
}

const files = [
  [cssPath, buildCss()],
  [tsPath, buildTs()],
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
