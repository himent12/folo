# FOLO Studio

The web app of FOLO: datasets, training runs, models and a detection playground. Svelte 5 (runes) + TypeScript +
Vite, with our own design system (no UI kit, no CSS framework). It talks only to the HTTP API in `docs/API.md`.

```sh
npm ci
npm run dev       # http://localhost:5173, proxies /api to FOLO_API (default http://127.0.0.1:7878)
npm run check     # generated tokens are fresh + svelte-check + tsc
npm test          # vitest (logic)
npm run build     # -> dist/, served by `folo serve`
npm run shoot r1  # screenshots of every screen/state -> shots/r1/ (uses the installed Chrome)
npm run qa        # the shoot's layout checks on the production build at tablet/phone sizes, no shots; fails on any (CI)
FOLO_API=http://127.0.0.1:7979 npm run smoke   # every route against a running `folo serve`, reports errors
```

Data source: in dev, `/api` is proxied to `FOLO_API` (default `folo serve` on `127.0.0.1:7878`; try
`folo serve --mock`). If no server answers, Studio falls back to its built-in mock (`src/lib/api/mock`), which is also
what `?mock=1` forces and what screenshots use, so they are deterministic.

- Design tokens come from `../design/tokens.json` (see `../design/README.md`). After editing it, run `npm run tokens`.
- `src/lib/design/` generated tokens + base CSS · `src/lib/ui/` components · `src/lib/shell/` app chrome ·
  `src/lib/charts/` SVG/canvas charts · `src/lib/api/` typed client + dev mock · `src/routes/` screens.
- Routing is hash based (`#/runs/run_7d1e`) so any static server works.
- Being ported to fw, our own JSX + signals framework (`framework/`, its README and PORTING.md): `next.html` is the fw
  app (`src/main.tsx`), built beside `index.html`; ported components sit beside their `.svelte` originals as `.tsx`.

## Port parity and performance (legacy `/` vs `next.html`)

```sh
node scripts/shoot.mjs --app next r1          # the shoot of next.html (default --app legacy: /); same scenes and URLs
node scripts/parity.mjs --run --only runs     # shoot both apps (exact mode) and diff -> D:/folo-build/ui-fast-runtime/parity
node scripts/parity.mjs --run --only runs --baseline D:/folo-build/ui-fast-runtime/baseline   # next only, vs stored legacy shots
node scripts/parity.mjs <dirA> <dirB>         # compare two shot folders
node scripts/perf.mjs --app both --runs 5     # load, memory and CPU, A/B alternated -> D:/folo-build/ui-fast-runtime/perf/<time>
```

- `shoot.mjs --exact` (what parity uses) makes a shot reproducible to the pixel: the clock paused as for `--qa`,
  `Math.random` constant (the mock's latency jitter), real-time work (blobs, image loads and decodes, chunks, fonts)
  awaited before and after every clock step, Chrome drawing in software, animations frozen at the shot. It also writes a
  normalized DOM (`<shot>.dom.txt`) and `shoot.json` (failed states and why). `--out`, `--base <url>`, `--prod`,
  `--jobs N`; `--only a,b` takes a comma list. Scope classes `svelte-*` and `f-*` (`FOLO_SCOPE_PREFIX`) are ignored.
- `parity.mjs` reports per shot: identical, N differing pixels + bounding box, size mismatch, or missing (next.html not
  built or served, a setup that timed out); `parity.md`, `parity.json` and `diff/<shot>.png` (magenta over a dimmed copy)
  in the out dir, with the first difference of the two DOM dumps. A state that differs is shot again in both apps
  (`--retries 1`) and passes if any pair matches. `--self` shoots legacy twice (the determinism check). Exit 1 on any
  difference.
- `perf.mjs` measures the production build (`node_modules/.perf-dist`, `vite preview`, Chrome + CDP): bundle per entry,
  cold loads of the main screens (FCP, content shown, script/task/layout/style time, long tasks, heap after GC, nodes,
  listeners), hash navigation cycles (with the heap growth), 15 s live on the Run page and Overview (busy %, rAF/s),
  and the dataset grid scrolled end to end. `--api <url>` uses a running `folo serve --mock` instead of `?mock=1`;
  `--only bundle,cold,nav,live,large`. Writes `perf.md` (median, min–max, change %) and `perf.json` (every sample).
