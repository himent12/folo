// Spacing lint: every px value of a spacing declaration sits on the grid.
//   node scripts/lint-spacing.mjs      (part of `npm run check`)
// Scans the <style> blocks of src/**/*.svelte and src/lib/design/base.css for padding / margin / gap / inset /
// top / right / bottom / left declarations whose px values are off the allowed set below. The sign is ignored, and so
// are px values inside calc() / min() / max() / clamp() that combine with a var() (e.g. `calc(var(--space-4) - 1px)`).
// Type sizes come from the scale: a `font-size` in px is flagged (use var(--type-*-size), or em for a size relative to
// its text). A deliberate exception carries `/* grid-ok */` on its line.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ALLOWED = new Set([0, 1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 44, 48, 56, 64]);
const PROP = /^(?:padding|margin|inset)(?:-[a-z-]+)?$|^(?:gap|row-gap|column-gap|top|right|bottom|left)$/;

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const src = join(root, 'src');

function* walk(dir) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) yield* walk(p);
    else if (name.endsWith('.svelte')) yield p;
  }
}

/** Removes calc()/min()/max()/clamp() groups that contain a var(), keeping the rest of the value. */
function dropVarMath(value) {
  let out = '';
  let i = 0;
  while (i < value.length) {
    const m = /^(?:calc|min|max|clamp)\(/.exec(value.slice(i));
    if (!m) {
      out += value[i++];
      continue;
    }
    let depth = 0;
    let j = i + m[0].length - 1;
    for (; j < value.length; j++) {
      if (value[j] === '(') depth++;
      else if (value[j] === ')' && --depth === 0) break;
    }
    const group = value.slice(i, j + 1);
    out += group.includes('var(') ? ' ' : group;
    i = j + 1;
  }
  return out;
}

/**
 * Lints CSS text whose first line is line `startLine` of `file`; returns the problems found. Comments are blanked
 * first (keeping line breaks, so locations stay right) and declarations are read whole, so a value on the next line
 * is still checked and commented-out code is not. A declaration touching a line with `grid-ok` is exempt.
 */
export function lintCss(css, file, startLine = 1) {
  const problems = [];
  const lines = css.split('\n');
  const exempt = new Set(lines.flatMap((l, k) => (l.includes('grid-ok') ? [k] : [])));
  const code = css.replace(/\/\*[\s\S]*?\*\//g, (c) => c.replace(/[^\n]/g, ' '));
  const lineAt = (i) => code.slice(0, i).split('\n').length - 1;
  for (const m of code.matchAll(/(?:^|[;{\s])([a-z-]+)\s*:\s*([^;{}]+)/g)) {
    const [, prop, value] = m;
    const size = prop === 'font-size';
    if (!PROP.test(prop) && !size) continue;
    const first = lineAt(m.index + m[0].indexOf(prop));
    const last = lineAt(m.index + m[0].length);
    let skip = false;
    for (let k = first; k <= last; k++) if (exempt.has(k)) skip = true;
    if (skip) continue;
    if (size) {
      if (/\d*\.?\d+px\b/.test(dropVarMath(value))) {
        const where = file ? `${relative(root, file).replaceAll('\\', '/')}:${startLine + first}` : `line ${startLine + first}`;
        problems.push(`${where}  font-size: ${value.trim()}  (use a --type-*-size token)`);
      }
      continue;
    }
    for (const px of dropVarMath(value).matchAll(/(-?\d*\.?\d+)px\b/g)) {
      const v = Math.abs(Number(px[1]));
      const where = file ? `${relative(root, file).replaceAll('\\', '/')}:${startLine + first}` : `line ${startLine + first}`;
      if (!ALLOWED.has(v)) problems.push(`${where}  ${prop}: ${value.trim().replace(/\s+/g, ' ')}  (${v}px is off the grid)`);
    }
  }
  return problems;
}

/** Lints every Svelte style block under src/ and base.css. */
export function lintAll() {
  const problems = [];
  for (const file of walk(src)) {
    const text = readFileSync(file, 'utf8').replace(/\r\n/g, '\n');
    for (const m of text.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/g)) {
      const startLine = text.slice(0, m.index + m[0].indexOf('>') + 1).split('\n').length;
      problems.push(...lintCss(m[1], file, startLine));
    }
  }
  const base = join(src, 'lib/design/base.css');
  problems.push(...lintCss(readFileSync(base, 'utf8').replace(/\r\n/g, '\n'), base, 1));
  return problems;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const problems = lintAll();
  if (problems.length) {
    console.error(`Spacing or type off the scale (spacing px: ${[...ALLOWED].join(', ')}; mark a deliberate exception /* grid-ok */):`);
    for (const p of problems) console.error(`  ${p}`);
    process.exit(1);
  }
  console.log('spacing and type: every padding / margin / gap / inset px value is on the grid, every font-size a token');
}
