// design/tokens.json -> CSS custom properties, with the same variable names as Studio's web/src/lib/design/tokens.css
// (web/scripts/tokens.mjs). The site differs in one way: with no saved choice it follows the system theme.
import { existsSync, readFileSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import { REPO_DIR } from './config.ts';

const DESIGN_DIR = resolve(REPO_DIR, 'design');
const HEX = /^#([0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$/;

type Json = Record<string, unknown>;
interface Layer {
  x: number;
  y: number;
  blur: number;
  spread: number;
  color: string;
}
interface FontSpec {
  family: string;
  fallback: string[];
  weights: number[];
  files: Record<string, string>;
}
interface TypeStep {
  size: number;
  line: number;
  tracking: number;
  weight: number;
}

const isMeta = (k: string) => k.startsWith('$');
const kebab = (s: string) => s.replace(/[_.]/g, '-');
const px = (n: number) => (n === 0 ? '0' : `${n}px`);

function hex(v: unknown, where: string): string {
  if (typeof v !== 'string' || !HEX.test(v)) throw new Error(`tokens.json ${where}: expected #RRGGBB or #RRGGBBAA, got ${v}`);
  return v.toLowerCase();
}

function entries(o: unknown): [string, unknown][] {
  return Object.entries(o as Json).filter(([k]) => !isMeta(k));
}

function themeVars(t: Json, name: string): [string, string][] {
  const out: [string, string][] = [];
  for (const [group, val] of entries((t.color as Json)[name])) {
    if (typeof val === 'string') {
      out.push([`--${kebab(group)}`, hex(val, `color.${name}.${group}`)]);
      continue;
    }
    for (const [k, v] of entries(val)) {
      const where = `color.${name}.${group}.${k}`;
      if (group === 'chart' && k === 'series') {
        (v as string[]).forEach((s, i) => out.push([`--chart-${i + 1}`, hex(s, `${where}[${i}]`)]));
      } else if (group === 'chart' && k === 'fill') {
        (v as string[]).forEach((s, i) => out.push([`--chart-fill-${i + 1}`, hex(s, `${where}[${i}]`)]));
      } else if (typeof v === 'number') {
        out.push([`--${kebab(group)}-${kebab(k)}`, String(v)]);
      } else {
        out.push([`--${kebab(group)}-${kebab(k)}`, hex(v, where)]);
      }
    }
  }
  for (const [k, layers] of entries((t.shadow as Json)[name])) {
    const css = (layers as Layer[])
      .map((l) => `${px(l.x)} ${px(l.y)} ${px(l.blur)} ${px(l.spread)} ${hex(l.color, `shadow.${name}.${k}`)}`)
      .join(', ');
    out.push([`--shadow-${k}`, css]);
  }
  return out;
}

function staticVars(t: Json): [string, string][] {
  const font = t.font as { sans: FontSpec; mono: FontSpec; weight: Record<string, number> };
  const fam = (f: FontSpec) => [`"${f.family}"`, ...f.fallback.map((x) => (/\s/.test(x) ? `"${x}"` : x))].join(', ');
  const out: [string, string][] = [
    ['--font-sans', fam(font.sans)],
    ['--font-mono', fam(font.mono)],
  ];
  for (const [k, v] of Object.entries(font.weight)) out.push([`--weight-${k}`, String(v)]);
  for (const [k, s] of entries(t.type) as [string, TypeStep][]) {
    out.push(
      [`--type-${k}-size`, px(s.size)],
      [`--type-${k}-line`, px(s.line)],
      [`--type-${k}-tracking`, s.tracking === 0 ? '0' : `${s.tracking}em`],
      [`--type-${k}-weight`, String(s.weight)],
    );
  }
  for (const [k, v] of entries(t.space)) out.push([`--space-${kebab(k)}`, px(v as number)]);
  for (const [k, v] of entries(t.radius)) out.push([`--radius-${k}`, px(v as number)]);
  for (const [k, v] of entries(t.size)) {
    if (typeof v === 'object' && v) for (const [k2, v2] of entries(v)) out.push([`--${kebab(k)}-${k2}`, px(v2 as number)]);
    else out.push([`--size-${kebab(k)}`, px(v as number)]);
  }
  const motion = t.motion as { duration: Record<string, number>; easing: Record<string, number[]> };
  for (const [k, v] of entries(motion.duration)) out.push([`--dur-${k}`, `${v}ms`]);
  for (const [k, v] of entries(motion.easing)) out.push([`--ease-${k}`, `cubic-bezier(${(v as number[]).join(', ')})`]);
  for (const [k, v] of entries(t.z)) out.push([`--z-${k}`, String(v)]);
  return out;
}

function fontFaces(t: Json, cssDir: string): string {
  const font = t.font as { sans: FontSpec; mono: FontSpec };
  const rel = relative(cssDir, DESIGN_DIR).replace(/\\/g, '/');
  const faces: string[] = [];
  for (const f of [font.sans, font.mono]) {
    for (const w of f.weights) {
      const file = f.files[String(w)];
      if (!file) throw new Error(`tokens.json font ${f.family}: no file for weight ${w}`);
      if (!existsSync(resolve(DESIGN_DIR, `${file}.woff2`))) throw new Error(`design/${file}.woff2 is missing`);
      faces.push(
        `@font-face {\n  font-family: "${f.family}";\n  font-style: normal;\n  font-weight: ${w};\n  font-display: swap;\n  src: url("${rel}/${file}.woff2") format("woff2");\n}\n`,
      );
    }
  }
  return faces.join('');
}

const block = (sel: string, vars: [string, string][], indent = '') =>
  `${indent}${sel} {\n${vars.map(([k, v]) => `${indent}  ${k}: ${v};`).join('\n')}\n${indent}}\n`;

/** The tokens stylesheet, with font URLs relative to `cssDir` (where the file is written). */
export function tokensCss(cssDir: string): string {
  const t = JSON.parse(readFileSync(resolve(DESIGN_DIR, 'tokens.json'), 'utf8')) as Json;
  const dark: [string, string][] = [['color-scheme', 'dark'], ...themeVars(t, 'dark')];
  const light: [string, string][] = [['color-scheme', 'light'], ...themeVars(t, 'light')];
  const durations = (t.motion as { duration: Json }).duration;
  const reduced: [string, string][] = entries(durations).map(([k]) => [`--dur-${k}`, '0ms']);
  return [
    '/* Generated by site/scripts/tokens.ts from design/tokens.json. Do not edit. */\n',
    fontFaces(t, cssDir),
    block(':root', staticVars(t)),
    block(':root,\n:root[data-theme="dark"]', dark),
    `@media (prefers-color-scheme: light) {\n${block(':root:not([data-theme="dark"])', light, '  ')}}\n`,
    block(':root[data-theme="light"]', light),
    `@media (prefers-reduced-motion: reduce) {\n${block(':root', reduced, '  ')}}\n`,
  ].join('\n');
}
