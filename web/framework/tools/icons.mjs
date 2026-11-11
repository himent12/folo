// The lucide icons the app draws, vendored: framework/icons/lucide.json holds each icon's data (name, size, node,
// aliases) as lucide publishes it, for every `$icons/<name>` the app and the framework import. The fw Vite plugin
// builds `$icons/<name>` from it, so the build needs no icon package.
//
//   node framework/tools/icons.mjs            checks that every imported icon is vendored (exit 1 if not)
//   node framework/tools/icons.mjs --add      vendors the missing ones (and drops the ones no longer imported), from the
//                                             plain `lucide` package at LUCIDE (fetched with `npm pack` into a temp dir)
//
// Data comes from lucide's ESM build: an icon's node list, its canonical name (an alias import such as
// `triangle-alert` resolves to `triangle-exclamation-point`, whose name is the icon's `lucide-<name>` class). Lucide's
// published package does not carry the `aliases` list @lucide/svelte added as classes (`lucide-check-circle-2`); the
// icons vendored from @lucide/svelte 1.70.0 keep theirs, a newly added one has none.
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/** The lucide release the vendored icons come from. */
const LUCIDE = '1.70.0';

const web = fileURLToPath(new URL('../..', import.meta.url));
const file = resolve(web, 'framework/icons/lucide.json');

/** Every `$icons/<name>` imported under src/ and framework/. */
function imported() {
  const set = new Set();
  const walk = (d) => {
    for (const name of readdirSync(d)) {
      const p = join(d, name);
      if (statSync(p).isDirectory()) {
        if (name !== 'node_modules' && !name.startsWith('dist')) walk(p);
      } else if (/\.(tsx?|mjs)$/.test(name)) for (const m of readFileSync(p, 'utf8').matchAll(/['"]\$icons\/([a-z0-9-]+)['"]/g)) set.add(m[1]);
    }
  };
  walk(resolve(web, 'src'));
  walk(resolve(web, 'framework'));
  return [...set].sort();
}

const vendored = JSON.parse(readFileSync(file, 'utf8'));
const wanted = imported();
const missing = wanted.filter((n) => !(n in vendored));
const unused = Object.keys(vendored).filter((n) => !wanted.includes(n));

if (!process.argv.includes('--add')) {
  if (missing.length) {
    console.error(`icons imported but not vendored: ${missing.join(', ')}\nrun: node framework/tools/icons.mjs --add`);
    process.exit(1);
  }
  console.log(`${wanted.length} icons, all vendored${unused.length ? ` (${unused.length} vendored and unused: ${unused.join(', ')})` : ''}`);
  process.exit(0);
}

const out = {};
for (const n of wanted) if (n in vendored) out[n] = vendored[n];
if (missing.length) {
  const tmp = mkdtempSync(join(tmpdir(), 'lucide-'));
  try {
    execFileSync('npm', ['pack', `lucide@${LUCIDE}`, '--silent'], { cwd: tmp, shell: process.platform === 'win32' });
    execFileSync('tar', ['-xzf', `lucide-${LUCIDE}.tgz`], { cwd: tmp });
    const pkg = join(tmp, 'package/dist/esm');
    const aliases = new Map();
    for (const m of readFileSync(join(pkg, 'iconsAndAliases.mjs'), 'utf8').matchAll(/export \{([^}]*)\} from '\.\/icons\/([\w-]+)\.mjs';/g)) {
      for (const part of m[1].split(',')) {
        const as = part.trim().match(/as (\w+)$/);
        if (as) aliases.set(as[1], m[2]);
      }
    }
    const pascal = (k) => k.split('-').map((p) => p[0].toUpperCase() + p.slice(1)).join('');
    for (const n of missing) {
      const canonical = aliases.get(pascal(n));
      if (!canonical) throw new Error(`no lucide icon "${n}" in lucide ${LUCIDE}`);
      const mod = readFileSync(join(pkg, `icons/${canonical}.mjs`), 'utf8');
      const lit = mod.match(/const \w+ = (\[[\s\S]*?\]);\n/)?.[1];
      if (!lit) throw new Error(`no icon data in lucide's icons/${canonical}.mjs`);
      out[n] = { name: canonical, size: 24, node: new Function(`return ${lit}`)() };
    }
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
}
const sorted = Object.fromEntries(Object.keys(out).sort().map((k) => [k, out[k]]));
writeFileSync(file, JSON.stringify(sorted, null, 1) + '\n');
console.log(`${Object.keys(sorted).length} icons vendored (${missing.length} added, ${unused.length} dropped)`);
