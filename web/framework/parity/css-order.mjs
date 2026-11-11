// Checks that the built fw app keeps the legacy Svelte app's cascade order between components: each component's
// styles (a scope class: `svelte-*` in index.html's CSS, `fw-*` in next.html's) are matched across the two bundles by
// their rules (scoping leaves declarations unchanged), then the order of the matched components is compared. Order only
// matters between two components when one of them has a rule that reaches outside its own elements (a `:global`
// part): fully scoped rules of two files never match the same element, so their relative order changes nothing (and
// the bundler may put a component shared with the lazy screens in a CSS chunk of its own, linked before the entry's).
// Run `npm run build` first.
//   node framework/parity/css-order.mjs
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import postcss from 'postcss';

const web = fileURLToPath(new URL('../..', import.meta.url));
const read = (name) => readFileSync(join(web, 'dist', name), 'utf8');
/** The CSS a page links, in order. */
const cssOf = (page) => [...read(page).matchAll(/href="\/assets\/([^"]+\.css)"/g)].map((m) => read(join('assets', m[1]))).join('\n');

/** fw scope classes with a rule that can match another file's elements: a compound without the scope class. */
const reaching = new Set();

/** Per scope class, in order of first appearance: the set of its rules (selector without scope + declarations). */
function components(css, prefix) {
  const out = new Map();
  const re = new RegExp(`\\.(${prefix}-[a-z0-9]+)`);
  postcss.parse(css).walkRules((r) => {
    const m = r.selector.match(re);
    if (!m) return;
    const parts = r.selector.split(',').flatMap((s) => s.trim().split(/\s*[\s>+~]\s*/));
    if (prefix === 'fw' && parts.some((part) => part && !part.includes(m[1]))) reaching.add(m[1]);
    const decls = (r.nodes ?? []).filter((n) => n.type === 'decl').map((n) => `${n.prop}:${n.value}`).join(';');
    const sel = r.selector.replace(new RegExp(`(:where\\()?\\.${prefix}-[a-z0-9]+\\)?`, 'g'), '').replace(/\s+/g, ' ');
    if (!out.has(m[1])) out.set(m[1], new Set());
    out.get(m[1]).add(`${sel}{${decls}}`);
  });
  return out;
}

const legacy = components(cssOf('index.html'), 'svelte');
const fw = components(cssOf('next.html'), 'fw');
const legacyOrder = [...legacy.keys()];
// Each fw component's twin: the legacy component sharing the most rules (and at least half of them).
const pairs = [];
for (const [cls, rules] of fw) {
  let best = null;
  let score = 0;
  for (const [h, lr] of legacy) {
    let n = 0;
    for (const r of rules) if (lr.has(r)) n++;
    if (n > score) {
      score = n;
      best = h;
    }
  }
  if (best && score * 2 >= rules.size) pairs.push([cls, best]);
}
let inversions = 0;
for (let i = 1; i < pairs.length; i++) {
  const a = legacyOrder.indexOf(pairs[i - 1][1]);
  const b = legacyOrder.indexOf(pairs[i][1]);
  if (b < a && (reaching.has(pairs[i - 1][0]) || reaching.has(pairs[i][0]))) {
    inversions++;
    console.log(`  out of order: ${pairs[i - 1][0]} (legacy #${a}) before ${pairs[i][0]} (legacy #${b})`);
  }
}
console.log(`components with styles in next.html: ${fw.size}; matched to legacy: ${pairs.length}; out of legacy order: ${inversions}`);
if (inversions) process.exitCode = 1;
