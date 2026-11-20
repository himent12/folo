// Writes src/theme/name_chars.rs (run from apps/desktop: `node tools/name-chars.mjs`): the code point ranges of
// \p{M} and \p{Cf} a custom theme's name is checked against, from this Node's Unicode data.
import { writeFileSync } from 'node:fs';

function ranges(re) {
  const out = [];
  let lo = -1;
  for (let c = 0; c <= 0x110000; c++) {
    const hit = c <= 0x10ffff && !(c >= 0xd800 && c <= 0xdfff) && re.test(String.fromCodePoint(c));
    if (hit && lo < 0) lo = c;
    if (!hit && lo >= 0) {
      out.push([lo, c - 1]);
      lo = -1;
    }
  }
  return out;
}

const hex = (n) => `0x${n.toString(16).toUpperCase()}`;
function table(name, doc, rs) {
  const pairs = rs.map(([a, b]) => `(${hex(a)}, ${hex(b)})`);
  const lines = [];
  for (let i = 0; i < pairs.length; i += 6) lines.push(`    ${pairs.slice(i, i + 6).join(', ')},`);
  return `${doc}\n#[rustfmt::skip]\npub(super) const ${name}: [(u32, u32); ${rs.length}] = [\n${lines.join('\n')}\n];\n`;
}

const marks = ranges(/^\p{M}$/u);
const format = ranges(/^\p{Cf}$/u);
const out = `//! The Unicode general categories a custom theme's name is checked against (\`custom::name_problem\`,
//! design/THEMES.md): Mark (M) and Format (Cf), as code point ranges, inclusive and in order.
//! Generated from Unicode ${process.versions.unicode} (Node ${process.versions.node}'s \`\\p{M}\` and \`\\p{Cf}\`), which the
//! JavaScript validators read through their own engine: \`node tools/name-chars.mjs\` writes this file again.

${table('MARKS', '/// General category M: nonspacing, spacing and enclosing marks (combining accents, variation selectors).', marks)}
${table('FORMAT', '/// General category Cf: invisible format characters (zero-width space and joiners, bidi controls, U+FEFF).', format)}`;
writeFileSync(process.argv[2] ?? 'src/theme/name_chars.rs', out);
console.log(`marks ${marks.length} ranges, format ${format.length} ranges`);
