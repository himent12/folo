// Pixel parity of two shoots: is the port (next.html) a 1:1 copy of the legacy app (/) on every screen and state?
//   node scripts/parity.mjs <dirA> <dirB> [--out <dir>] [--only a,b]
//       compares every shot by name, pixel for pixel; writes <out>/parity.md, parity.json and diff/<shot>.png
//       (differences in magenta over a dimmed copy of A). --out defaults to the folder holding <dirB>.
//   node scripts/parity.mjs --run [--only runs,run-live] [--out D:/folo-build/ui-fast-runtime/parity] [--dev]
//       [--baseline <legacy shots>] [--jobs N] [--retries 1]
//       serves the app once (the production build of both entries; --dev: the dev server), shoots both apps with
//       `shoot.mjs --exact` into <out>/legacy and <out>/next at the same time, and compares them. A state that differs
//       is shot again in both apps (--retries rounds, default 1, at most 150 states): it passes when any legacy shot
//       and any next shot of it match, and the report flags it when the legacy app's own shots differ (a timing
//       residue of the legacy app, not a porting bug). --baseline compares against stored legacy shots (e.g.
//       D:/folo-build/ui-fast-runtime/baseline) instead of shooting the legacy app again. --self shoots the legacy app
//       as B too: the determinism check (every state must match itself).
// Per shot: identical, N differing pixels and their bounding box, a size mismatch, or missing in A or B (with the
// shoot's reason: next.html not served yet, a scene's setup that timed out). Per shot, too, the first difference of the
// two normalized DOM dumps (`<shot>.dom.txt`, see shoot.mjs): why the pixels differ, usually. Exits 1 if any shot
// differs or is missing.
import { createServer, preview } from 'vite';
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { availableParallelism } from 'node:os';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodePng, encodePng } from './png.mjs';
import { buildApps } from './entries.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const VALUED = ['--only', '--out', '--baseline', '--jobs', '--retries'];
const opt = (/** @type {string} */ flag) => {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
};
const positional = args.filter((a, i) => !a.startsWith('--') && !VALUED.includes(args[i - 1]));
const only = opt('--only');
const onlyParts = only ? only.split(',').map((s) => s.trim()).filter(Boolean) : [];
const picked = (/** @type {string} */ name) => !onlyParts.length || onlyParts.some((p) => name.includes(p));
const started = Date.now();

/** Where `--run` puts its shots and report: off the repo (a full set is ~1 GB), on D: on the dev machine. */
const DEFAULT_OUT = existsSync('D:/') ? 'D:/folo-build/ui-fast-runtime/parity' : resolve(root, 'shots/parity');
/** States shot again at most (a port that differs everywhere is not worth a second full shoot). */
const MAX_RETRY = 150;

const run = args.includes('--run');
/** --self: B is the legacy app as well. */
const bApp = args.includes('--self') ? 'legacy' : 'next';
/** @type {string} */
let dirA;
/** @type {string} */
let dirB;
/** @type {string} */
let outDir;
if (run) {
  outDir = resolve(opt('--out') ?? DEFAULT_OUT);
  const baseline = opt('--baseline');
  dirA = baseline ? resolve(baseline) : resolve(outDir, 'legacy');
  dirB = resolve(outDir, 'next');
} else {
  if (positional.length !== 2) {
    console.error(
      'usage: node scripts/parity.mjs <dirA> <dirB> [--out <dir>] [--only a,b]\n' +
        '       node scripts/parity.mjs --run [--only a,b] [--out <dir>] [--dev] [--baseline <dir>] [--jobs N] [--retries 1] [--self]',
    );
    process.exit(2);
  }
  dirA = resolve(positional[0]);
  dirB = resolve(positional[1]);
  outDir = resolve(opt('--out') ?? dirname(dirB));
}

/** Per shoot, how long it took (s). @type {Record<string, number>} */
const shootSecs = {};

/** @type {{ base: string, close(): Promise<void> } | null} */
let served = null;
/** Why next.html did not build, or null: then every next state is missing for that reason. @type {string | null} */
let buildNote = null;

/** Serves the app once: the production build with both entries, or (--dev) the dev server. */
async function serve() {
  const listen = { host: '127.0.0.1', port: 5198, strictPort: false };
  const t0 = Date.now();
  /** @type {{ resolvedUrls: import('vite').ResolvedServerUrls | null, close(): Promise<void> }} */
  let server;
  if (args.includes('--dev')) {
    const dev = await createServer({ root, logLevel: 'error', server: listen });
    await dev.listen();
    server = dev;
  } else {
    const dist = resolve(root, 'node_modules/.parity-dist');
    const { input, nextError } = await buildApps(root, dist);
    buildNote = nextError;
    server = await preview({ root, logLevel: 'error', build: { outDir: dist }, preview: listen });
    console.log(`built ${Object.keys(input).join(' + ')} in ${Math.round((Date.now() - t0) / 1000)} s`);
  }
  return { base: (server.resolvedUrls?.local[0] ?? 'http://127.0.0.1:5198/').replace(/\/$/, ''), close: () => server.close() };
}

/**
 * Shoots the given apps into their folders at the same time, each with its share of the state pool.
 * @param {{ app: 'legacy' | 'next', dir: string }[]} jobs
 * @param {string | undefined} filter shoot.mjs's --only
 * @param {string} label for shootSecs and the log
 */
async function shootAll(jobs, filter, label) {
  served ??= await serve();
  const base = served.base;
  const pool = Number(opt('--jobs')) || Math.max(2, Math.min(16, availableParallelism() - 4));
  const each = Math.max(1, Math.floor(pool / jobs.length));
  await Promise.all(
    jobs.map(async ({ app, dir }) => {
      rmSync(dir, { recursive: true, force: true });
      const t = Date.now();
      const code = await new Promise((done) => {
        const child = spawn(
          process.execPath,
          [resolve(root, 'scripts/shoot.mjs'), '--exact', '--app', app, '--base', base, '--out', dir, '--jobs', String(each), ...(filter ? ['--only', filter] : [])],
          { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] },
        );
        const tag = (/** @type {Buffer} */ b) => b.toString().split(/\r?\n/).filter(Boolean).map((l) => `[${label} ${app}] ${l}`).join('\n');
        child.stdout.on('data', (b) => console.log(tag(b)));
        child.stderr.on('data', (b) => console.error(tag(b)));
        child.on('close', done);
      });
      shootSecs[`${label} ${basename(dir)}`] = Math.round((Date.now() - t) / 1000);
      if (code !== 0) console.warn(`[${label} ${app}] shoot exited ${code} (failed states are reported as missing)`);
    }),
  );
}

/** Shot names in a folder (without .png), and the shoot's record of states that failed. */
function listShots(/** @type {string} */ dir) {
  const names = existsSync(dir) ? readdirSync(dir).filter((f) => f.endsWith('.png')).map((f) => f.slice(0, -4)) : [];
  /** @type {{ failed?: Record<string, string>, missing?: string | null }} */
  let meta = {};
  const file = resolve(dir, 'shoot.json');
  if (existsSync(file)) meta = JSON.parse(readFileSync(file, 'utf8'));
  return { names: new Set(names), failed: meta.failed ?? {}, missing: meta.missing ?? null };
}

/**
 * Compares two shots pixel for pixel; `diff` is A dimmed with the differing pixels magenta (over the union of sizes).
 * @returns {{ pixels: number, total: number, bbox: number[] | null, sizeA: string, sizeB: string, diff: Buffer | null }}
 */
function comparePngs(/** @type {Buffer} */ a, /** @type {Buffer} */ b, wantDiff = true) {
  const A = decodePng(a);
  const B = decodePng(b);
  const w = Math.max(A.width, B.width);
  const h = Math.max(A.height, B.height);
  const out = wantDiff ? new Uint8Array(w * h * 4) : null;
  const a32 = new Uint32Array(A.data.buffer, A.data.byteOffset, A.width * A.height);
  const b32 = new Uint32Array(B.data.buffer, B.data.byteOffset, B.width * B.height);
  let pixels = 0;
  let x0 = w;
  let y0 = h;
  let x1 = -1;
  let y1 = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const ia = y * A.width + x;
      const same = x < A.width && y < A.height && x < B.width && y < B.height && a32[ia] === b32[y * B.width + x];
      const o = (y * w + x) * 4;
      if (same) {
        if (out) {
          // A dimmed to a quarter of its luminance, so the magenta reads on light and dark shots alike.
          const p = ia * 4;
          out[o] = out[o + 1] = out[o + 2] = (A.data[p] * 0.299 + A.data[p + 1] * 0.587 + A.data[p + 2] * 0.114) * 0.25 + 24;
          out[o + 3] = 255;
        }
        continue;
      }
      pixels++;
      if (x < x0) x0 = x;
      if (y < y0) y0 = y;
      if (x > x1) x1 = x;
      if (y > y1) y1 = y;
      if (out) {
        out[o] = 255;
        out[o + 1] = 0;
        out[o + 2] = 255;
        out[o + 3] = 255;
      }
    }
  }
  return {
    pixels,
    total: w * h,
    bbox: pixels ? [x0, y0, x1 - x0 + 1, y1 - y0 + 1] : null,
    sizeA: `${A.width}x${A.height}`,
    sizeB: `${B.width}x${B.height}`,
    diff: pixels && out ? encodePng({ width: w, height: h, data: out }) : null,
  };
}

/** Whether two folders' shots of `name` are the same pixels (byte-equal files are; Chrome encodes deterministically). */
function samePixels(/** @type {string} */ da, /** @type {string} */ db, /** @type {string} */ name) {
  const fa = resolve(da, `${name}.png`);
  const fb = resolve(db, `${name}.png`);
  if (!existsSync(fa) || !existsSync(fb)) return false;
  const a = readFileSync(fa);
  const b = readFileSync(fb);
  return a.equals(b) || comparePngs(a, b, false).pixels === 0;
}

/** A DOM dump line's element, short: `div#app.shell` (text lines as they are). */
function short(/** @type {string} */ line) {
  const t = line.trim();
  if (!t.startsWith('<')) return t.length > 60 ? `${t.slice(0, 57)}…"` : t;
  const tag = t.match(/^<([\w-]+)/)?.[1] ?? '?';
  const id = t.match(/ id="([^"]*)"/)?.[1];
  const cls = t.match(/ class="([^"]*)"/)?.[1]?.split(' ').slice(0, 2).join('.');
  return `${tag}${id ? `#${id}` : ''}${cls ? `.${cls}` : ''}`;
}

/**
 * The first line where two DOM dumps differ, with the path of elements down to it.
 * @returns {{ line: number, path: string, a: string, b: string, linesA: number, linesB: number } | null}
 */
function domDiff(/** @type {string} */ a, /** @type {string} */ b) {
  if (a === b) return null;
  const la = a.split('\n');
  const lb = b.split('\n');
  let i = 0;
  while (i < la.length && i < lb.length && la[i] === lb[i]) i++;
  const depth = (/** @type {string} */ l) => l.length - l.trimStart().length;
  const path = [];
  const at = la[i] ?? lb[i] ?? '';
  for (let k = i - 1, d = depth(at); k >= 0 && d > 0; k--) {
    if (depth(la[k]) < d) {
      path.unshift(short(la[k]));
      d = depth(la[k]);
    }
  }
  const clip = (/** @type {string | undefined} */ l) => (l === undefined ? '(end)' : l.trim().length > 240 ? `${l.trim().slice(0, 240)}…` : l.trim());
  return { line: i + 1, path: path.join(' > '), a: clip(la[i]), b: clip(lb[i]), linesA: la.length, linesB: lb.length };
}

/**
 * @typedef {{ name: string, result: 'identical' | 'differs' | 'size' | 'missing-a' | 'missing-b', pixels?: number, share?: number,
 *   bbox?: number[] | null, sizeA?: string, sizeB?: string, why?: string, dom?: ReturnType<typeof domDiff> | 'missing',
 *   retried?: 'matched' | 'still differs', legacyFlaky?: boolean }} Row
 */

/**
 * Compares one state's shots in A and B (writing its diff image).
 * @returns {Row}
 */
function compareOne(/** @type {string} */ name, /** @type {ReturnType<typeof listShots>} */ A, /** @type {ReturnType<typeof listShots>} */ B) {
  const inA = A.names.has(name);
  const inB = B.names.has(name);
  /** @type {Row} */
  let row;
  if (!inA || !inB) {
    row = { name, result: !inA ? 'missing-a' : 'missing-b', why: (!inA ? A.failed[name] : B.failed[name]) ?? (!inA ? A.missing : B.missing) ?? 'no shot' };
  } else {
    const a = readFileSync(resolve(dirA, `${name}.png`));
    const b = readFileSync(resolve(dirB, `${name}.png`));
    // Chrome encodes the same pixels to the same bytes: most shots never need decoding.
    if (a.equals(b)) row = { name, result: 'identical', pixels: 0 };
    else {
      const c = comparePngs(a, b);
      if (c.diff) writeFileSync(resolve(diffDir, `${name}.png`), c.diff);
      row = { name, result: c.sizeA !== c.sizeB ? 'size' : c.pixels ? 'differs' : 'identical', pixels: c.pixels, share: c.pixels / c.total, bbox: c.bbox, sizeA: c.sizeA, sizeB: c.sizeB };
    }
  }
  const da = resolve(dirA, `${name}.dom.txt`);
  const db = resolve(dirB, `${name}.dom.txt`);
  if (existsSync(da) && existsSync(db)) row.dom = domDiff(readFileSync(da, 'utf8'), readFileSync(db, 'utf8'));
  else if (inA && inB) row.dom = 'missing';
  return row;
}

const baseline = run && !!opt('--baseline');
const diffDir = resolve(outDir, 'diff');
let A = listShots(dirA);
let B = listShots(dirB);
/** @type {Row[]} */
let rows = [];
try {
  if (run) {
    await shootAll(
      [...(baseline ? [] : [{ app: /** @type {const} */ ('legacy'), dir: dirA }]), { app: /** @type {'legacy' | 'next'} */ (bApp), dir: dirB }],
      only,
      'shoot',
    );
  }
  A = listShots(dirA);
  B = listShots(dirB);
  if (buildNote && bApp === 'next' && B.missing) {
    for (const k of Object.keys(B.failed)) if (B.failed[k] === B.missing) B.failed[k] = buildNote;
    B.missing = buildNote;
  }
  const all = [...new Set([...A.names, ...B.names, ...Object.keys(A.failed), ...Object.keys(B.failed)])].filter(picked).sort();
  rmSync(diffDir, { recursive: true, force: true });
  mkdirSync(diffDir, { recursive: true });
  rows = all.map((name) => compareOne(name, A, B));

  // Shot again: a state whose shots differ passes when any legacy shot and any next shot of it match.
  const retries = run ? Number(opt('--retries') ?? 1) : 0;
  for (let round = 1; round <= retries; round++) {
    const again = rows.filter((r) => r.result === 'differs' || r.result === 'size').map((r) => r.name);
    if (!again.length) break;
    if (again.length > MAX_RETRY) {
      console.log(`${again.length} states differ: not shooting them again (more than ${MAX_RETRY})`);
      break;
    }
    console.log(`shooting the ${again.length} state(s) that differ again (round ${round})`);
    const la = resolve(outDir, `legacy-retry${round}`);
    const nb = resolve(outDir, `next-retry${round}`);
    await shootAll([{ app: 'legacy', dir: la }, { app: /** @type {'legacy' | 'next'} */ (bApp), dir: nb }], again.join(','), `retry${round}`);
    const legacyDirs = [dirA, ...Array.from({ length: round }, (_, i) => resolve(outDir, `legacy-retry${i + 1}`))];
    const nextDirs = [dirB, ...Array.from({ length: round }, (_, i) => resolve(outDir, `next-retry${i + 1}`))];
    for (const r of rows) {
      if (!again.includes(r.name)) continue;
      const matched = legacyDirs.some((da) => nextDirs.some((db) => samePixels(da, db, r.name)));
      r.retried = matched ? 'matched' : 'still differs';
      r.legacyFlaky = !samePixels(dirA, la, r.name);
      if (matched) {
        r.result = 'identical';
        rmSync(resolve(diffDir, `${r.name}.png`), { force: true });
      }
    }
  }
} finally {
  // Assigned inside shootAll, which TypeScript cannot see here.
  await /** @type {{ close(): Promise<void> } | null} */ (served)?.close();
}

const count = (/** @type {Row['result']} */ r) => rows.filter((x) => x.result === r).length;
const bad = rows.filter((r) => r.result !== 'identical');
const domDiffers = rows.filter((r) => r.dom && r.dom !== 'missing').length;
const flaky = rows.filter((r) => r.legacyFlaky);
const secs = Math.round((Date.now() - started) / 1000);
const summary = {
  a: dirA,
  b: dirB,
  only: only ?? null,
  shots: rows.length,
  identical: count('identical'),
  differs: count('differs'),
  size: count('size'),
  missingA: count('missing-a'),
  missingB: count('missing-b'),
  domDiffers,
  matchedOnRetry: rows.filter((r) => r.retried === 'matched').length,
  legacyFlaky: flaky.map((r) => r.name),
  secs,
  shootSecs,
};
writeFileSync(resolve(outDir, 'parity.json'), JSON.stringify({ ...summary, rows }, null, 1));

const fmtBox = (/** @type {number[] | null | undefined} */ b) => (b ? `${b[2]}x${b[3]} at ${b[0]},${b[1]}` : '');
const cell = (/** @type {string} */ s) => s.replace(/\|/g, '\\|').replace(/\n/g, ' ');
const domText = (/** @type {Row} */ r) =>
  r.result.startsWith('missing') ? '' : !r.dom ? 'same' : r.dom === 'missing' ? '(no dump)' : `line ${r.dom.line} in \`${r.dom.path || 'body'}\`: A \`${r.dom.a}\` / B \`${r.dom.b}\``;
const md = [
  '# Parity',
  '',
  `A: \`${dirA}\`  `,
  `B: \`${dirB}\`${only ? `  \nonly: \`${only}\`` : ''}`,
  '',
  `**${summary.identical} / ${summary.shots} identical**${summary.matchedOnRetry ? ` (${summary.matchedOnRetry} on a second shot)` : ''}; ${summary.differs} differ, ` +
    `${summary.size} differ in size, ${summary.missingA} missing in A, ${summary.missingB} missing in B. DOM differs in ${domDiffers}. ` +
    `${secs} s${Object.keys(shootSecs).length ? ` (shoots: ${Object.entries(shootSecs).map(([k, v]) => `${k} ${v} s`).join(', ')})` : ''}.`,
  '',
];
if (A.missing) md.push(`Every state missing in A: ${A.missing}`, '');
if (B.missing) md.push(`Every state missing in B: ${B.missing}`, '');
const listed = bad.filter((r) => !(r.result === 'missing-a' && A.missing) && !(r.result === 'missing-b' && B.missing));
if (listed.length) {
  md.push('| shot | result | pixels | box | first DOM difference |', '|---|---|---:|---|---|');
  for (const r of listed) {
    const res =
      (r.result === 'size' ? `size ${r.sizeA} vs ${r.sizeB}` : r.result.startsWith('missing') ? `${r.result}: ${r.why}` : 'differs') +
      (r.legacyFlaky ? ' (legacy differs from itself too)' : '');
    const px = r.pixels ? `${r.pixels} (${((r.share ?? 0) * 100).toFixed(2)}%)` : '';
    md.push(`| ${r.pixels ? `[${r.name}](diff/${r.name}.png)` : r.name} | ${cell(res)} | ${px} | ${fmtBox(r.bbox)} | ${cell(domText(r))} |`);
  }
  md.push('');
}
if (flaky.length) md.push(`Legacy shots that differ between two shoots (timing residue in the legacy app): ${flaky.map((r) => r.name).join(', ')}`, '');
const domOnly = rows.filter((r) => r.result === 'identical' && r.dom && r.dom !== 'missing');
if (domOnly.length) {
  md.push('Identical pixels, different DOM:', '');
  for (const r of domOnly) md.push(`- ${r.name}: ${cell(domText(r))}`);
  md.push('');
}
writeFileSync(resolve(outDir, 'parity.md'), md.join('\n'));

// A whole app missing (next.html not served yet) is one line, not one per state.
if (A.missing) console.log(`  every state missing in ${basename(dirA)}: ${A.missing}`);
if (B.missing) console.log(`  every state missing in ${basename(dirB)}: ${B.missing}`);
for (const r of listed.slice(0, 40)) {
  const what = r.result.startsWith('missing') ? `${r.result} (${r.why})` : r.result === 'size' ? `size ${r.sizeA} vs ${r.sizeB}, ${r.pixels} px` : `${r.pixels} px in ${fmtBox(r.bbox)}`;
  const dom = r.dom && r.dom !== 'missing' ? `\n    DOM line ${r.dom.line} ${r.dom.path}\n      A ${r.dom.a}\n      B ${r.dom.b}` : '';
  console.log(`  ${r.name}: ${what}${r.legacyFlaky ? ' (legacy differs from itself too)' : ''}${dom}`);
}
if (listed.length > 40) console.log(`  … and ${listed.length - 40} more`);
console.log(
  `parity: ${summary.identical}/${summary.shots} identical${summary.matchedOnRetry ? ` (${summary.matchedOnRetry} on a second shot)` : ''}, ${summary.differs} differ, ${summary.size} size, ` +
    `${summary.missingA} missing in ${basename(dirA)}, ${summary.missingB} missing in ${basename(dirB)}; DOM differs in ${domDiffers} (${secs} s) -> ${resolve(outDir, 'parity.md')}`,
);
if (bad.length) process.exitCode = 1;
