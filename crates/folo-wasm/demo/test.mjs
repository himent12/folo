// End-to-end check of folo-wasm in a real browser: serves the repository, opens the demo page in headless Chrome
// (Playwright), runs a model on WebGPU and on the CPU, prints the latency and compares the boxes with the native
// predictor's (optional `--expect`, the JSON of `folo predict --json`).
//
//   sh scripts/build-wasm.sh
//   node crates/folo-wasm/demo/test.mjs --model shapes-n.folo --image img.jpg [--expect native.json] [--runs 20]
//
// Options: --conf 0.25   --imgsz <n>   --backends webgpu,cpu   --chrome <chrome executable>   --headed   --screenshots <dir>
// playwright-core comes from web/node_modules (`npm ci` in web/) or anywhere Node resolves it.
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { existsSync, readFileSync, statSync } from "node:fs";
import { extname, join, normalize, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, all) => {
    if (a.startsWith("--")) acc.push([a.slice(2), all[i + 1] && !all[i + 1].startsWith("--") ? all[i + 1] : "true"]);
    return acc;
  }, []),
);
if (!args.model || !args.image) {
  console.error("usage: node test.mjs --model <file.folo> --image <image> [--imgsz 640] [--expect native.json] [--runs 20]");
  process.exit(2);
}
const runs = Number(args.runs ?? 20);
const conf = Number(args.conf ?? 0.25);
const backends = (args.backends ?? "webgpu,cpu").split(",");

let playwright;
try {
  playwright = createRequire(join(root, "web", "package.json"))("playwright-core");
} catch {
  playwright = await import("playwright-core");
}

const chromeCandidates = [
  args.chrome,
  process.env.CHROME_PATH,
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "/usr/bin/google-chrome",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].filter(Boolean);
const chrome = chromeCandidates.find((p) => existsSync(p));

const MIME = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm",
  ".json": "application/json", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp" };
const server = createServer((req, res) => {
  const url = new URL(req.url, "http://x");
  let file;
  if (url.pathname === "/__model") file = resolve(args.model);
  else if (url.pathname === "/__image") file = resolve(args.image);
  else {
    file = normalize(join(root, decodeURIComponent(url.pathname)));
    if (!file.startsWith(root)) { res.writeHead(403).end(); return; }
    if (existsSync(file) && statSync(file).isDirectory()) file = join(file, "index.html");
  }
  if (!existsSync(file)) { res.writeHead(404).end(); return; }
  res.writeHead(200, { "content-type": MIME[extname(file).toLowerCase()] ?? "application/octet-stream" });
  res.end(readFileSync(file));
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}`;

const browser = await playwright.chromium.launch({
  executablePath: chrome,
  headless: args.headed !== "true",
  args: ["--enable-unsafe-webgpu", "--enable-gpu", "--ignore-gpu-blocklist"],
});

function iou(a, b) {
  const w = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const h = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const i = w * h;
  return i / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - i);
}

/** Every expected box has a same-class match with IoU >= 0.98 and a score within 0.01, and the counts agree. */
function compare(got, want) {
  const problems = [];
  if (got.length !== want.length) problems.push(`${got.length} boxes, native has ${want.length}`);
  const used = new Set();
  for (const w of want) {
    let best = -1, bestIou = 0;
    got.forEach((g, i) => {
      if (used.has(i) || g.class !== w.class) return;
      const v = iou(g, w);
      if (v > bestIou) { bestIou = v; best = i; }
    });
    if (best < 0 || bestIou < 0.98 || Math.abs(got[best].score - w.score) > 0.01) {
      problems.push(`no match for ${w.name} ${w.score.toFixed(3)} (best IoU ${bestIou.toFixed(3)})`);
    } else used.add(best);
  }
  return problems;
}

let expected = null;
if (args.expect) {
  // `folo predict --json` prints one JSON object per line; take the record with boxes.
  const lines = readFileSync(args.expect, "utf8").split(/\r?\n/).filter((l) => l.trim().startsWith("{"));
  const rec = lines.map((l) => JSON.parse(l)).find((r) => Array.isArray(r.boxes));
  expected = rec?.boxes ?? null;
}

const summary = { chrome: chrome ?? "playwright default", results: [] };
let failed = false;
for (const backend of backends) {
  const page = await browser.newPage();
  const logs = [];
  page.on("console", (m) => logs.push(`${m.type()}: ${m.text()}`));
  page.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`));
  const imgsz = args.imgsz ? `&imgsz=${args.imgsz}` : "";
  await page.goto(`${base}/crates/folo-wasm/demo/?model=/__model&image=/__image${imgsz}`);
  await page.waitForFunction(() => window.foloDemoReady === true, null, { timeout: 120_000 });
  let r;
  try {
    r = await page.evaluate(([b, n, c]) => window.foloDemo.bench(b, n, 3, c), [backend, runs, conf]);
  } catch (e) {
    console.error(`[${backend}] failed: ${e.message}\n${logs.join("\n")}`);
    failed = true;
    await page.close();
    continue;
  }
  const entry = {
    requested: backend,
    backend: r.backend,
    adapter: r.info.adapter,
    fallback: r.info.fallback,
    simd128: r.info.simd128,
    imgsz: r.info.imgsz,
    image: r.last.image,
    boxes: r.last.boxes.length,
    load_ms: +r.load_ms.toFixed(1),
    first_call_ms: +r.first_ms.toFixed(1),
    wall_ms_median: +r.wall_ms_median.toFixed(2),
    wall_ms_min: +r.wall_ms_min.toFixed(2),
    timing_last: r.last.timing,
    hasWebGpu: await page.evaluate(() => window.foloDemo.hasWebGpu()),
  };
  if (r.backend !== backend) { entry.error = `ran on ${r.backend}`; failed = true; }
  if (expected) {
    const problems = compare(r.last.boxes, expected);
    entry.matches_native = problems.length === 0;
    if (problems.length) { entry.problems = problems; failed = true; }
  }
  if (args.screenshots) await page.screenshot({ path: join(resolve(args.screenshots), `folo-wasm-${backend}.png`) });
  summary.results.push(entry);
  await page.close();
}
await browser.close();
server.close();
console.log(JSON.stringify(summary, null, 2));
process.exit(failed ? 1 : 0);
