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
npm run qa        # the shoot's blank-section and overflow checks at tablet/phone sizes, no shots; fails on any (CI)
FOLO_API=http://127.0.0.1:7979 npm run smoke   # every route against a running `folo serve`, reports errors
```

Data source: in dev, `/api` is proxied to `FOLO_API` (default `folo serve` on `127.0.0.1:7878`; try
`folo serve --mock`). If no server answers, Studio falls back to its built-in mock (`src/lib/api/mock`), which is also
what `?mock=1` forces and what screenshots use, so they are deterministic.

- Design tokens come from `../design/tokens.json` (see `../design/README.md`). After editing it, run `npm run tokens`.
- `src/lib/design/` generated tokens + base CSS · `src/lib/ui/` components · `src/lib/shell/` app chrome ·
  `src/lib/charts/` SVG/canvas charts · `src/lib/api/` typed client + dev mock · `src/routes/` screens.
- Routing is hash based (`#/runs/run_7d1e`) so any static server works.
