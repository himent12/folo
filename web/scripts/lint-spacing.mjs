// Spacing, type and layout lint: the grid, the type scale, the layout tiers and the text-size floor.
//   node scripts/lint-spacing.mjs      (part of `npm run check`)
// Scans the css`` blocks of src/**/*.tsx (each component's styles) and src/lib/design/base.css for padding / margin / gap / inset /
// top / right / bottom / left declarations whose px values are off the allowed set below. The sign is ignored, and so
// are px values inside calc() / min() / max() / clamp() that combine with a var() (e.g. `calc(var(--space-4) - 1px)`).
// Type sizes come from the scale: a `font-size` in px is flagged (use var(--type-*-size), or em for a size relative to
// its text). A deliberate exception carries `/* grid-ok */` on its line.
//
// Layout tiers (design/tokens.json `breakpoint`, design/README.md): every width in an `@media` query is a tier
// boundary written one way: `(min-width: 768px)` / `(max-width: 767px)`, and the same for 1024 and 1280. Any other
// width, unit or the range syntax (`width < 768px`) fails. Script never spells a width query (matchMedia, MediaQuery):
// it uses `media` from $lib/design/tokens, generated from the same tokens. A component that needs its own threshold
// uses a container query (not linted) or an intrinsic layout, with a comment saying why.
//
// Text-size floor: text is at least `type.sm` (12 px). `--type-xs-*` (11 px) is for chart tick labels only and
// `--type-micro-*` (10 px) for label chips drawn on images and keycaps' secondary marks. Every use of either step's
// size or line height, in CSS, markup or TS, carries an annotation on its line or the line above saying which:
// `small: chart tick` (xs only), `small: image chip` (xs or micro) or `small: keycap mark` (micro). Unannotated uses
// fail. A font size in px outside a stylesheet (an inline style, a canvas font) fails like one inside.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ALLOWED = new Set([0, 1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 44, 48, 56, 64]);
const PROP = /^(?:padding|margin|inset)(?:-[a-z-]+)?$|^(?:gap|row-gap|column-gap|top|right|bottom|left)$/;

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const src = join(root, 'src');

const tokens = JSON.parse(readFileSync(resolve(root, '..', 'design', 'tokens.json'), 'utf8'));
/** Tier boundaries: a tier starts at `min-width: N`; everything below it is `max-width: N - 1`. */
const TIERS = Object.entries(tokens.breakpoint)
  .filter(([k]) => !k.startsWith('$'))
  .map(([, v]) => v);
const MIN_OK = new Set(TIERS);
const MAX_OK = new Set(TIERS.map((v) => v - 1));
/** The small steps each annotation allows. */
const SMALL_REASONS = { 'chart tick': ['xs'], 'image chip': ['xs', 'micro'], 'keycap mark': ['micro'] };

const locate = (file, line) => (file ? `${relative(root, file).replaceAll('\\', '/')}:${line}` : `line ${line}`);

function* walk(dir, ext = ['.tsx']) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) yield* walk(p, ext);
    else if (ext.some((e) => name.endsWith(e)) && !/\.test\.tsx$|\.testing\.tsx$/.test(name)) yield p;
  }
}

/** Blanks comments, keeping line breaks: block and HTML comments, and with `js` also line comments. */
function blankComments(text, js = false) {
  const blank = (c) => c.replace(/[^\n]/g, ' ');
  const out = text.replace(/\/\*[\s\S]*?\*\/|<!--[\s\S]*?-->/g, blank);
  return js ? out.replace(/(^|[^:'"`\\])(\/\/[^\n]*)/gm, (_, pre, c) => pre + blank(c)) : out;
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
  const code = blankComments(css);
  const lineAt = (i) => code.slice(0, i).split('\n').length - 1;
  for (const m of code.matchAll(/(?:^|[;{\s])([a-z-]+)\s*:\s*([^;{}]+)/g)) {
    const [, prop, value] = m;
    // The `font` shorthand sets a size too (`font: 10px Geist`).
    const size = prop === 'font-size' || prop === 'font';
    if (!PROP.test(prop) && !size) continue;
    const first = lineAt(m.index + m[0].indexOf(prop));
    const last = lineAt(m.index + m[0].length);
    let skip = false;
    for (let k = first; k <= last; k++) if (exempt.has(k)) skip = true;
    if (skip) continue;
    const where = locate(file, startLine + first);
    if (size) {
      if (/\d*\.?\d+px\b/.test(dropVarMath(value))) problems.push(`${where}  ${prop}: ${value.trim()}  (use a --type-*-size token)`);
      continue;
    }
    for (const px of dropVarMath(value).matchAll(/(-?\d*\.?\d+)px\b/g)) {
      const v = Math.abs(Number(px[1]));
      if (!ALLOWED.has(v)) problems.push(`${where}  ${prop}: ${value.trim().replace(/\s+/g, ' ')}  (${v}px is off the grid)`);
    }
  }
  return problems;
}

/** The width features of one media query that are not tier boundaries, or null when all are. */
function offTier(query) {
  const bad = [];
  for (const f of query.matchAll(/\(([^()]*\bwidth\b[^()]*)\)/g)) {
    const feature = f[1].trim().replace(/\s+/g, ' ');
    const m = /^(min|max)-width: ?(\d+)px$/.exec(feature);
    if (!m || !(m[1] === 'min' ? MIN_OK : MAX_OK).has(Number(m[2]))) bad.push(`(${feature})`);
  }
  return bad.length ? bad.join(', ') : null;
}

const SMALL_HINT =
  'xs (11 px) only with /* small: chart tick */ or /* small: image chip */, micro (10 px) only with /* small: image chip */ or /* small: keycap mark */';
const TIER_HINT = `the tiers are ${TIERS.map((v) => `(min-width: ${v}px)`).join(' / ')} and ${TIERS.map((v) => `(max-width: ${v - 1}px)`).join(' / ')}; a component's own threshold is a container query`;

/** Lints the `@media` queries of CSS text whose first line is `startLine` of `file`. */
export function lintMedia(css, file, startLine = 1) {
  const problems = [];
  const code = blankComments(css);
  for (const m of code.matchAll(/@media\b([^{;]*)/g)) {
    const bad = offTier(m[1]);
    const line = startLine + code.slice(0, m.index).split('\n').length - 1;
    if (bad) problems.push(`${locate(file, line)}  @media ${m[1].trim()}  (${bad} is not a tier boundary)`);
  }
  return problems;
}

/** Lints the media-query strings of script (matchMedia and friends) by the same rule. */
export function lintScriptMedia(text, file) {
  const problems = [];
  const code = blankComments(text, true);
  for (const m of code.matchAll(/(['"`])([^'"`\n]*\b(?:min-|max-)?width\s*[:<>=][^'"`\n]*)\1/g)) {
    // A query in parentheses anywhere, or the bare feature Svelte's MediaQuery also takes ('max-width: 767px').
    const bare = /MediaQuery\(\s*$/.test(code.slice(0, m.index));
    if (!bare && !/\(\s*(?:min-|max-)?width\s*[:<>=]/.test(m[2])) continue;
    const bad = offTier(bare && !m[2].includes('(') ? `(${m[2]})` : m[2]);
    const line = code.slice(0, m.index).split('\n').length;
    // Script names the tiers rather than spelling them, so a tier change is one edit in tokens.json.
    problems.push(`${locate(file, line)}  '${m[2]}'  (${bad ? `${bad} is not a tier boundary; ` : ''}use \`media\` from $lib/design/tokens)`);
  }
  return problems;
}

/**
 * Lints the text-size floor over a whole file (a stylesheet with `css`, else a Svelte component or TS): every
 * `--type-xs-*` / `--type-micro-*` size or line height needs `small: <reason>` on its line or the line above, with a
 * reason that allows that step. Outside a stylesheet a font size in px is flagged too (lintCss covers stylesheets).
 */
export function lintSmallText(text, file, { css = false } = {}) {
  const problems = [];
  const raw = text.split('\n');
  const code = blankComments(text, !css).split('\n');
  const style = css ? null : blankStyles(text).split('\n');
  const reasons = (k) =>
    [raw[k], k > 0 ? raw[k - 1] : ''].flatMap((l) => [...l.matchAll(/small:\s*([a-z]+(?: [a-z]+)*)/g)].map((r) => r[1]));
  code.forEach((line, k) => {
    for (const m of line.matchAll(/--type-(xs|micro)-(?:size|line)\b/g)) {
      const step = m[1];
      const given = reasons(k);
      if (given.some((r) => SMALL_REASONS[r]?.includes(step))) continue;
      const why = given.length ? `"small: ${given.join('", "')}" does not allow ${step}` : `unannotated ${step === 'xs' ? 11 : 10} px text`;
      problems.push(`${locate(file, k + 1)}  ${m[0]}  (${why})`);
    }
    const outside = style ? blankComments(style[k], true) : '';
    // Each font declaration on its own (a token elsewhere on the line exempts nothing): its value runs to the next
    // `;`, quote or line end.
    for (const d of outside.matchAll(/font(?:-size)?\s*[=:]\s*["'`]?([^;\n"'`]*)/g))
      if (/\b\d+(?:\.\d+)?px/.test(d[1]))
        problems.push(`${locate(file, k + 1)}  ${d[0].trim()}  (a font size in px outside a stylesheet; use a --type-*-size token)`);
  });
  return problems;
}

/** A component's styles: its top-level css`...` block, and (in a test's sample) a `<style>` element. */
const STYLE = /(^|\n)css`((?:[^`\\]|\\[\s\S])*)`;|<style[^>]*>([\s\S]*?)<\/style>/g;

/** `text` with its styles blanked (line breaks kept). */
function blankStyles(text) {
  return text.replace(STYLE, (c) => c.replace(/[^\n]/g, ' '));
}

/** Lints every component under src/ (its css`` block, script and markup), every TS module and base.css. */
export function lintAll() {
  const problems = [];
  for (const file of walk(src)) {
    const text = readFileSync(file, 'utf8').replace(/\r\n/g, '\n');
    for (const m of text.matchAll(STYLE)) {
      if (m[2] === undefined) continue;
      const startLine = text.slice(0, m.index + (m[1] ?? '').length).split('\n').length;
      // The template's own escapes undone (framework/PORTING.md §7), so the CSS is what was written.
      const styles = m[2].replace(/\\(`|\\|\$\{)/g, '$1');
      problems.push(...lintCss(styles, file, startLine), ...lintMedia(styles, file, startLine));
    }
    problems.push(...lintScriptMedia(blankStyles(text), file), ...lintSmallText(text, file));
  }
  for (const file of walk(src, ['.ts', '.css'])) {
    // Generated tokens define the steps; tests quote them.
    if (/\.test\.ts$|[\\/]tokens\.(?:ts|css)$/.test(file)) continue;
    const text = readFileSync(file, 'utf8').replace(/\r\n/g, '\n');
    if (file.endsWith('.css')) problems.push(...lintCss(text, file), ...lintMedia(text, file), ...lintSmallText(text, file, { css: true }));
    else problems.push(...lintScriptMedia(text, file), ...lintSmallText(text, file));
  }
  return problems;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const problems = lintAll();
  if (problems.length) {
    console.error(
      `Off the system. Spacing px: ${[...ALLOWED].join(', ')} (a deliberate exception is /* grid-ok */). Media widths: ${TIER_HINT}. Text: 12 px or more; ${SMALL_HINT}:`,
    );
    for (const p of problems) console.error(`  ${p}`);
    process.exit(1);
  }
  console.log(
    'spacing, type and layout: spacing on the grid, font sizes from the scale, media widths on the tiers, every 10–11 px text annotated',
  );
}
