# FOLO Studio

The web app of FOLO: datasets, training runs, models and a detection playground. TypeScript + Vite on fw, our own UI
framework (`framework/`: JSX compiled to cloned DOM templates and fine-grained signals, scoped CSS), with our own
design system (no UI kit, no CSS framework). It talks only to the HTTP API in `docs/API.md`.

```sh
npm ci
npm run dev       # http://localhost:5173, proxies /api to FOLO_API (default http://127.0.0.1:7878)
npm run check     # generated tokens are fresh, contrast, icons vendored, tsc (app, framework, scripts), spacing lint
npm test          # vitest (logic, components, the framework)
npm run build     # -> dist/, served by `folo serve`; fails if a built stylesheet lost an unprefixed declaration
npm run shoot r1  # screenshots of every screen/state -> shots/r1/ (uses the installed Chrome)
npm run qa        # the shoot's layout checks on the production build at tablet/phone sizes, no shots; fails on any (CI)
FOLO_API=http://127.0.0.1:7979 npm run smoke   # every route against a running `folo serve`, reports errors
```

Data source: the server Studio was served from, or in dev `/api` proxied to `FOLO_API` (default `folo serve` on
`127.0.0.1:7878`; try `folo serve --mock`). Studio's built-in mock (`src/lib/api/mock`) is used only when it is asked
for: `?mock=1` in the address (what the screenshot, QA, parity and perf scripts open, so they are deterministic), or
Settings › Connection › Use demo data, which reloads with it. A server that is slow to answer or not there is never
replaced by demo data: Studio says "Connecting…", then "Server offline" with "Can't reach folo serve. Retrying…", and
keeps checking (`src/lib/api/index.ts`) until it answers. So `npm run dev` without a server shows that offline state:
start `folo serve`, or open `http://localhost:5173/?mock=1`.

- Design tokens come from `../design/tokens.json` (see `../design/README.md`). After editing it, run `npm run tokens`.
- `src/lib/design/` generated tokens + base CSS · `src/lib/ui/` components · `src/lib/shell/` app chrome ·
  `src/lib/charts/` SVG/canvas charts · `src/lib/api/` typed client + dev mock · `src/routes/` screens (each a chunk of
  its own, `src/routes.ts`).
- Routing is hash based (`#/runs/run_7d1e`) so any static server works. Screens read `router.route`, the route on
  display: while the next screen's chunk loads, the screen still shown keeps its own route (`router.location` is the
  URL's); a chunk that fails to load shows an error with Retry (`src/ScreenError.tsx`).
- Components are `.tsx` files: JSX with HTML attribute names, a `css\`...\`` block of scoped styles. How to write one:
  `framework/README.md` (design, API) and `framework/PORTING.md` (every construct, with its Svelte origin).
- Icons: `import X from '$icons/x'` (lucide). Their data is vendored in `framework/icons/lucide.json`;
  `node framework/tools/icons.mjs --add` vendors a newly imported one (`npm run check` fails until it is).

## Screenshot parity and performance

Studio moved from Svelte 5 to fw with every screen and state pixel-identical (CHANGELOG, "Studio on fw"). The
same tools keep later changes honest: shoot every state, compare it with a stored shoot, and measure against another
version's build.

```sh
node scripts/parity.mjs --run --only runs     # shoot the app (exact mode) and diff with the stored baseline shoot
node scripts/parity.mjs --run --baseline <dir> --out <dir>   # against another stored shoot (a folder of --exact shots)
node scripts/parity.mjs --run --self          # shoot it twice: the determinism check
node scripts/parity.mjs <dirA> <dirB>         # compare two shot folders
node scripts/shoot.mjs --exact --prod --out <dir>            # a stored shoot of this version, to compare with later
node scripts/perf.mjs --runs 5                # load, memory and CPU of this version
node scripts/perf.mjs --ref origin/main       # against another git ref (its web/ built in a temporary worktree)
node scripts/perf.mjs --baseline-dist <dir>   # against a stored build (e.g. D:/folo-build/ui-fast-runtime/legacy-dist)
```

- `shoot.mjs --exact` (what parity uses) makes a shot reproducible to the pixel: the clock paused as for `--qa`,
  `Math.random` constant (the mock's latency jitter), real-time work (blobs, image loads and decodes, chunks, fonts)
  awaited before and after every clock step, Chrome drawing in software, animations frozen at the shot. It also writes a
  normalized DOM (`<shot>.dom.txt`) and `shoot.json` (failed states and why). `--out`, `--base <url>`, `--prod`,
  `--jobs N`; `--only a,b` takes a comma list. Scope classes `fw-*` (`FOLO_SCOPE_PREFIX`) and `svelte-*` (so the
  Svelte app's stored shoots stay comparable) are left out of the DOM dumps.
- `parity.mjs` reports per shot: identical, N differing pixels + bounding box, size mismatch, or missing (a setup that
  timed out); `parity.md`, `parity.json` and `diff/<shot>.png` (magenta over a dimmed copy) in the out dir, with the
  first difference of the two DOM dumps. A state that differs is shot again (`--retries 1`) and passes if any of its
  shots matches. The default baseline is `D:/folo-build/ui-fast-runtime/baseline`, the Svelte app's last shoot, whose
  DOM differs from fw's only where the Svelte app had bugs (its palette kept a page's commands after leaving it) or was
  not deterministic (the Overview's live numbers, the phone label grid). Exit 1 on any difference.
- `perf.mjs` measures the production build (built into `<out>/.dist`, `vite preview`, Chrome + CDP): the bundle, cold
  loads of the main screens (FCP, content shown, script/task/layout/style time, long tasks, heap after GC, nodes,
  listeners), hash navigation cycles (with the heap growth), 15 s live on the Run page and Overview (busy %, rAF/s),
  and the dataset grid scrolled end to end. With a baseline (`--baseline-dist`, `--ref`) both are measured,
  alternated, and compared. `--api <url>` uses a running `folo serve --mock` instead of `?mock=1`;
  `--only bundle,cold,nav,live,large`. Writes `perf.md` (median, min–max, change %) and `perf.json` (every sample).
