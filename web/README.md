# FOLO Studio

The web app of FOLO: datasets, labeling, training runs, models, vision benchmarks and a detection playground.
TypeScript + Vite on fw, our own UI framework (`framework/`: JSX compiled to cloned DOM templates and fine-grained
signals, scoped CSS), with our own design system (no UI kit, no CSS framework). It talks only to the HTTP API in
[docs/API.md](../docs/API.md), through the typed client in `src/lib/api/`, and has an in-browser mock of that API for
development, screenshots and tests.

Related: the [root README](../README.md), [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) (how Studio fits with the
server and the engine), [framework/README.md](framework/README.md) (fw), [design/README.md](../design/README.md)
(tokens, type, colour) and [FOLO Desktop](../apps/desktop/README.md), the native app with the same design language
and main screens (Overview, Datasets, Runs, Models, Benchmarks, Playground and Settings; agent labeling, run compare,
benchmark compare and importing external predictions stay in Studio).

## Commands

```sh
npm ci
npm run dev       # http://localhost:5173, proxies /api to FOLO_API (default http://127.0.0.1:7878)
npm run check     # generated tokens are fresh, contrast (and its tests), icons vendored, tsc (app, framework, scripts), spacing lint
npm test          # vitest (logic, components, the framework, the scripts' own tests)
npm run build     # -> dist/, served by `folo serve`; fails if a built stylesheet lost an unprefixed declaration
npm run preview   # serves dist/ (vite preview)
npm run tokens    # regenerates src/lib/design/tokens.css and tokens.ts from ../design/tokens.json
npm run shoot r1  # screenshots of every screen/state -> shots/r1/ (uses the installed Chrome); -- --theme paper: one theme
npm run qa        # the shoot's layout checks on the production build at tablet/phone sizes, no shots, then
                  # scripts/load-faults.mjs (a stylesheet or the mock failing to load); fails on any (CI)
FOLO_API=http://127.0.0.1:7979 npm run smoke   # core routes (excludes Labeling and Benchmarks) against a running `folo serve`, reports errors
npm run live-smoke -- <out dir>                # a short training run through New run, watched (usage: its header)
npm run label-smoke -- <out dir> --images <dir> # agent labeling end to end with a real agent CLI (spends tokens)
npm run parity    # scripts/parity.mjs --run (Screenshot parity and performance, below)
npm run perf      # scripts/perf.mjs
node scripts/release-smoke.mjs --folo <binary> --weights <model.folo>   # a release build's embedded Studio
```

Each script under `scripts/` starts with a comment giving its usage and options.

CI (`.github/workflows/web.yml`) runs on pull requests that touch `web/`, `design/` or the workflow, and on every push
to `main`: `npm ci`, `npm run check` and `npm test` in one job, and `npm run qa` split over three runners
(`FOLO_QA_SHARD`).

`npm run build` copies `public/` into `dist/`. The Playground's in-browser inference needs folo-wasm there:
`sh ../scripts/build-wasm.sh` (or `../scripts/build-wasm.ps1`) writes it to `public/folo-wasm/` (git-ignored) before
the build (docs/WASM.md). Without it the Playground's Run on choices WebGPU and CPU are off: models run on the server.

## Data source

The server Studio was served from, or in dev `/api` proxied to `FOLO_API` (default `folo serve` on `127.0.0.1:7878`;
try `folo serve --mock`). The proxy is `server.proxy` in `vite.config.ts`: the client's requests are same-origin
`/api/...`, and Vite forwards them. Settings › Connection can name another server (`prefs.serverUrl`). Studio's
built-in mock (`src/lib/api/mock`) is used only when it is asked for: `?mock=1` in the address (what the screenshot,
QA, parity and perf scripts open, so they are deterministic; `?scenario=` picks a state, listed at the top of
`src/lib/api/mock/index.ts`), or Settings › Connection › Use demo data, which reloads with it. A server that is slow to
answer or not there is never replaced by demo data: Studio says "Connecting…", then "Server offline" with "Can't reach
folo serve. Retrying…", and keeps checking (`src/lib/api/index.ts`) until it answers with FOLO's health payload. An
address that answers with something else says what came back ("This address answered, but not as a FOLO server",
"This server asks for sign-in (401)", `src/lib/api/handshake.ts`) and is checked again the same way; so does a FOLO
server started with `--token`, whose health is open but whose runs list answers 401 ("This server needs a token
(folo serve --token), which Studio can't send"). That token check (`HEAD /api/runs`) is asked after the health has
passed; a copy of it also goes out with the health, a round trip less on a remote server, and its answer is used only
when it is a FOLO server's (401: the wall; 200: none). Anything else (another status, no answer, or none within
`TOKEN_GRACE`, 250 ms, of the health's) is set aside and the check after the health decides, as it always did: the
table in `#reachServer`. A server that starts refusing Studio while it is open (a proxy that sent the token stops)
is said in the same words, kept apart from whether it answers at all (`src/lib/api/reach.ts`, unchanged by it):
`src/lib/api/access.ts` changes only on the token check's own answer (401 or 403: refused; 2xx: let in; anything else:
no change). A 401 or 403 on any `/api/` request only asks for that check (one at a time, and one more when the
refused request went out after the check out had started), and while refused the check repeats until Studio is let in
again, which brings it back by itself. Settings › Connection › Test asks both as well: no answer, a refusal (red), a
token check that cannot tell (neither red nor green), or a server Studio can use (green). So `npm run dev` without a
server shows that offline state: start `folo serve`, or open `http://localhost:5173/?mock=1`.

## Where things are

| Path | What it holds |
|---|---|
| `index.html` | The page: applies the saved mode and theme (custom ones included) before first paint, loads `src/main.tsx` |
| `src/main.tsx` | Entry: loads the tokens and base CSS, renders `App` into `#app` |
| `src/firstScreen.ts` | Starts fetching the first screen's chunk before the rest of the app has run |
| `src/App.tsx` | The shell (sidebar, top bar, tab bar, palette, shortcuts, toasts, run watch) and the current screen |
| `src/routes.ts` | The route table: which file in `src/routes/` renders each route; each screen is a chunk of its own |
| `src/ScreenError.tsx` | A screen whose chunk failed to load: Retry, or "Studio was updated" with Reload; `ScreenCrash`, a screen that threw while created |
| `src/routes/` | The screens (table below); each screen's parts in the folder beside it |
| `src/lib/api/` | `index.ts` (`hub`: live server or mock), `client.ts` (the `Api` interface and `HttpApi`), `types.ts` + `types/` (docs/API.md's types, by resource), `resource.ts`, `sse.ts`, `live*.ts`, `reach.ts`, `handshake.ts`, `upload.ts`, `mock/` |
| `src/lib/router.ts` | The hash router (`router`, `href`, `RouteName`) |
| `src/lib/shell/` | App chrome: `Sidebar`, `Topbar`, `TabBar`, `CommandPalette` (parts in `commandPalette/`: `index.tsx`, `scoring.ts`, `providers.ts`, `data.ts`, `types.ts`), `GChordIndicator`, `Page` (the page frame), `nav.ts`, `crumbs.ts`, `shortcuts.ts`, `RunWatch` |
| `src/lib/ui/` | The design system's components (`Button`, `Panel`, `Dialog`, `Menu`, `Segmented`, `KpiStrip`, `Stat`, `StatusBadge`, `Field`, ...) and their helpers (`focus.ts`, `motion.ts`, `toast.ts`) |
| `src/lib/charts/` | SVG/canvas charts (`LineChart`, `Sparkline`, `Histogram`, `ConfusionMatrix`, `XYChart`) and their geometry |
| `src/lib/detect/` | Images with detection boxes (`DetectionView`, `DatasetImage`), label chips, NMS run in the browser (the Playground filters the server's candidates with it) |
| `src/lib/inference/` | `browser.ts`: FOLO models in this browser through folo-wasm, on WebGPU or the wasm CPU |
| `src/lib/bench/` | Vision benchmarks' rules, score functions, and external prediction validation (`external.ts`, `externalParse.ts`, `externalWorker.ts`), shared by the screens and the mock |
| `src/lib/design/` | `tokens.css` and `tokens.ts` (generated from `design/tokens.json`, never edited by hand: every built-in theme's palette too), `themes.ts` (custom theme files checked, themes resolved and listed), `applyTheme.ts` (a theme onto `<html>`), `base.css`, `classInk.ts` |
| `src/lib/*.ts` | Shared rules and wording: `failedStyles.ts` (stylesheets that failed to load, loaded again before the next chunk), `format.ts`, `prefs.ts` (preferences in localStorage: the mode, each mode's theme, the custom themes in `folo.themes`), `keys.ts`, `trainCli.ts`, `throughput.ts`, `cloud.ts`, `label.ts`, ... |
| `framework/` | fw, the UI framework: [framework/README.md](framework/README.md) |
| `public/` | Copied as is into `dist/`: `favicon.svg`, and `folo-wasm/` once `scripts/build-wasm.sh` has built it |
| `scripts/` | `tokens.mjs` with `themes.mjs` (the built-in themes resolved), `lint-spacing.mjs`, `builtCss.mjs` (the build's CSS check), `shoot.mjs` with `scenes.mjs` and `scenes/` (screenshots and QA), `load-faults.mjs` (recovery from a stylesheet or the mock that failed to load), `parity.mjs`, `perf.mjs` with `perfWire.mjs` (what went over the network by a moment), `entries.mjs`, `png.mjs`, the smoke tests |
| `shots/`, `dist/` | Output of `npm run shoot` and `npm run build`; git-ignored |
| `notes/` | The design review record: the rubric each round was scored against, and the log of the rounds |
| `DESIGN_DIRECTION.md` | The design rules Studio's code follows |
| `vite.config.ts` | The fw plugin, the `$lib` and `fw` aliases, the dev proxy, the vitest settings |
| `tsconfig.json`, `tsconfig.node.json` | Types of the app and the framework runtime; of `vite.config.ts` and `scripts/` |

Screens (`src/routes/`):

| Section | Routes | Screen files | Parts in |
|---|---|---|---|
| Overview | `#/` | `Overview.tsx` | `overview/` |
| Datasets | `#/datasets`, `#/datasets/<id>` | `Datasets.tsx`, `Dataset.tsx` | `datasets/`, `dataset/` |
| Labeling | `#/datasets/label`, `#/datasets/label/<id>` | `LabelJobs.tsx`, `LabelJob.tsx` | `label/` |
| Runs | `#/runs`, `#/runs/new`, `#/runs/compare`, `#/runs/<id>` | `Runs.tsx`, `RunNew.tsx`, `RunCompare.tsx`, `Run.tsx` | `runs/`, `runnew/` (`state.ts`, `SummaryRail.tsx`, sections, estimation & validation), `compare/` (`config.ts`, `state.ts`, `ComparePlot.tsx`, `RunChips.tsx`, `AddRun.tsx`, `ManageSheet.tsx`, `CompareTable.tsx`), `run/` |
| Models | `#/models`, `#/models/<id>` | `Models.tsx`, `Model.tsx` | `models/`, `model/` |
| Benchmarks | `#/benchmarks`, `#/benchmarks/<id>`, `…/<id>/entries/<entryId>`, `…/<id>/compare`, `…/<id>/loops/<loopId>` | `Benchmarks.tsx`, `Benchmark.tsx`, `BenchEntry.tsx`, `BenchCompare.tsx`, `BenchLoop.tsx` | `bench/` |
| Playground | `#/playground` | `Playground.tsx` | `playground/` |
| Settings | `#/settings` | `Settings.tsx` | `settings/` |
| (any other) | | `NotFound.tsx`, rendered by `App` | |

Tests sit next to what they test (`*.test.ts`, `*.test.tsx`); tests of a whole screen are `src/routes/*.test.tsx`.

## Routing

Routing is hash based (`#/runs/run_7d1e`) so any static server works; a path URL is moved to its hash route on
load. `src/lib/router.ts` holds the `RouteName` union, the table of path patterns and their params, `section()` (which
sidebar entry a route highlights) and `router`. Screens read `router.route`, the route on display: while the next
screen's chunk loads, the screen still shown keeps its own route (`router.location` is the URL's); a chunk that fails
to load shows an error with Retry (`src/ScreenError.tsx`), or, when the server's index.html now names another entry
script (Studio was updated under the open page), "Studio was updated" with Reload; a screen that throws while it is
created shows "This page could not be shown" with Reload in its place (`ScreenCrash`), never the previous screen under
the new URL. `src/routes.ts` maps each route name to its file in `src/routes/` (`ROUTES`), and `import.meta.glob`
makes each of those files a chunk. A screen's first load fetches three scripts and their stylesheets: the entry (the
shell and what it imports), `shared` (every module two or more chunks use) and the screen's own chunk
(`build.rolldownOptions` in `vite.config.ts`); later screens need only their own. Keep it that way: `folo serve` speaks
HTTP/1.1, where a browser opens six connections to a server, so on a remote one every six more files cost another
round trip. The bundler's preload helper never asks for a stylesheet twice in a page, so one that failed to load (a
dropped connection) would stay missing until a reload: when a screen's chunk or the demo's mock module loads, each
stylesheet that failed earlier is asked for again (`src/lib/failedStyles.ts`), in the place of its failed link and with
its attributes, so the stylesheets keep a clean load's order. Nothing waits for it: the screen loads as it would
without it (a screen whose own stylesheet failed still says it could not load, and shows without it after Retry while
the stylesheet is gone, as before), and the stylesheet applies when it arrives. `scripts/load-faults.mjs` (in
`npm run qa`) holds that. The sidebar, the phone tab bar, the palette's navigation and the `G` shortcuts all come from
`mainNav` and `settingsNav` in `src/lib/shell/nav.ts`. A screen sets its breadcrumbs with `setCrumbs`
(`src/lib/shell/crumbs.ts`).

## State and live updates

- **In a component:** fw's `signal`, `computed` and `effect` ([framework/README.md](framework/README.md), API).
- **App-wide:** a few module singletons: `hub` (`src/lib/api/index.ts`: the API in use, `mode`, `offline`), `router`,
  `prefs` (`src/lib/prefs.ts`, kept in localStorage) and the page's crumbs (`src/lib/shell/crumbs.ts`).
- **Server data:** `new Resource((signal) => hub.api.x(signal), { poll? })` in a component body
  (`src/lib/api/resource.ts`). It fetches again when a reactive value it read changes, drops its data when the hub
  reconnects (`hub.epoch`), and retries or refreshes when the server answers again after an outage (`hub.back`).
  Identical reads issued in the same turn are one request (`src/lib/api/share.ts`). `startFrom` names another Resource
  that reads the same thing: the first fetch takes the answer it is loading, or one under 2 s old, and asks for itself
  when that read fails or has not answered by its first poll (Overview's run list from the shell, the machine foot's
  system info from Overview).
- **Live updates (SSE):** `openEventStream` (`src/lib/api/sse.ts`) reads the stream with `fetch` (EventSource hides
  the server's 15 s keep-alive comments) and reopens, with exponential backoff, a stream that ended, failed, was refused
  or passed no byte for `STREAM_SILENCE_MS` (45 s: three keep-alives missed); each reopening asks its view to read the
  history again (`reconnect`). `LiveRun` (`live.ts`, a run's
  `GET /api/runs/{id}/events`), `LiveLabelJob` (`liveLabel.ts`) and `LiveBench` (`liveBench.ts`) subscribe before they
  fetch history and merge the two; `LiveRun` publishes new steps once per animation frame (`version`).
- **Run list:** the shell polls it every 2 s (`src/lib/shell/RunWatch.tsx`, `runList.ts`) for the sidebar's live row
  and the toast when a run ends.
- **Offline:** `src/lib/api/reach.ts` hears every request and stream and sets `hub.offline`; screens then show what
  they had as a last known state (`src/lib/api/freshness.ts`).

## Common changes

**Add a screen.** Add the name to `RouteName` and its pattern to the table in `src/lib/router.ts` (and to `section()`
when it belongs under another section), map it to a file in `ROUTES` in `src/routes.ts` (the `Record` type makes tsc
fail until every route has one), and write `src/routes/<Name>.tsx` with a default export that renders `Page`
(`src/lib/shell/Page.tsx`) and calls `setCrumbs`: `src/routes/NotFound.tsx` is the smallest example. A top-level
section also gets an entry in `mainNav` (`src/lib/shell/nav.ts`). Add its states to the shoot
(`scripts/scenes/`, each file listed in `scripts/scenes.mjs`) so `npm run qa` checks its layout, and its path to
`scripts/smoke.mjs`.

**Add an API call.** The endpoint is specified in [docs/API.md](../docs/API.md) first (its server side is
`crates/folo-server/src/routes/` with types in `crates/folo-api`). In Studio: add the types to the matching file in
`src/lib/api/types/` (it mirrors docs/API.md; change both together), the method to the `Api` interface and to `HttpApi`
in `src/lib/api/client.ts` (as `throughput()`: `this.#req<ThroughputTable>('GET', '/system/throughput', undefined,
signal)`), and the
same method to the mock: `src/lib/api/mock/server.ts` (system, datasets, models), `trainingApi.ts` (runs, cloud) or
`index.ts` (labeling, benchmarks, inference), answering through `this.call(...)` as `MockServer.throughput()` does.
`MockApi` implements `Api`, so tsc fails until the mock has the method too. A GET method also goes into `READS` in
`src/lib/api/share.ts`; without it, identical requests issued together are not shared.

**Add a component.** A `.tsx` file in `src/lib/ui/` (or in the screen's folder when only that screen uses it) with a
default-exported function and one `css\`...\`` block of scoped styles that uses the tokens' CSS variables
(`var(--radius-sm)`, `var(--type-sm-size)`); `src/lib/ui/Badge.tsx` is a short example, and `framework/PORTING.md`
lists more. `scripts/lint-spacing.mjs` (in `npm run check`) rejects px values off the grid and font sizes off the
scale. A test is `<Name>.test.tsx` beside it, starting with `// @vitest-environment happy-dom` and using `render` and
`flush` from `framework/testing.ts` (as `src/lib/ui/Segmented.test.tsx`).

**Add or change a theme.** The built-in themes are `themes` in `design/tokens.json` (`design/THEMES.md`: a mode
and overrides over that mode's base; the first of each mode is its default and has none). Edit it there, run `npm run
tokens` (it writes one `:root[data-theme-id="<id>"]` block per non-default theme into `tokens.css` and every theme's
resolved palette, `themeIds` and `colorVars` into `tokens.ts`, and fails naming the path of a malformed theme), and
`npm run check` (`design/contrast.mjs` holds every theme to its mode's pairings). `npm run tokens` also rewrites the
data line (`const T = …`) of `index.html`'s first-paint script: the built-in ids, the id pattern (`THEME_ID` in
`scripts/themes.mjs`, also `themeIdPattern` in `tokens.ts`, which `themes.ts` checks ids with) and every colour token's
custom property, which it checks a stored custom theme against (`npm run check` fails while it is stale). The rules of a
custom theme file are `checkCustomTheme` in `src/lib/design/themes.ts`, and a picked file's bytes go through
`readThemeFile` (size, one byte order mark, strict UTF-8, `design/json-scan.mjs`'s check of what Desktop's JSON
parser refuses, JSON; it never rejects, so one bad file can't stop a pick), both held to `design/theme-cases.json` (the same
cases `contrast.mjs` and Desktop replay) by `themes.test.ts`; the first-paint script repeats the entry rules, and
`firstPaint.test.ts` holds it to the app on every one of those cases; change them together. Settings › Appearance is
`src/routes/settings/Appearance.tsx`, the palette's "Theme: …" commands `src/lib/shell/themeCommands.ts`. Colours reach
components only as CSS variables, so a theme repaints everything; canvas or SVG code that needs a colour as a value
reads `prefs.activeTheme.color`, never `color.dark` / `color.light` by mode. `node scripts/shoot.mjs --theme <id>`
shoots a theme (`settings-appearance-custom` imports the example `design/themes/night-shift.json`).

**Add or change a token.** Edit `design/tokens.json` (format rules in `design/README.md`), run `npm run tokens`, and use
the new CSS variable in a `css` block, or import it from `$lib/design/tokens` in canvas or SVG code. `npm run check`
fails while the generated files are stale. A new colour pairing gets a rule in `design/contrast.mjs`. Desktop and the
docs site generate their theme from the same file on their next build (`design/README.md`).

## Conventions

- Design tokens come from `../design/tokens.json` (see `../design/README.md`). After editing it, run `npm run tokens`.
- The look is one with FOLO Desktop's: `../design/UNIFIED.md` says screen by screen what each shows, and
  `DESIGN_DIRECTION.md` the rules Studio's code follows. Screens build from the shared parts, never their own copies:
  `shell/Page` (title, status `badge`, the facts line), `ui/Panel`, `ui/KpiStrip` and `ui/Stat`, `ui/StatusBadge`,
  `ui/Segmented`, `ui/SortHeader`, `ui/Ident` (names in sans, ids and paths in mono) and the `.table` class.
- Components are `.tsx` files: JSX with HTML attribute names, a `css\`...\`` block of scoped styles. How to write one:
  `framework/README.md` (design, API) and `framework/PORTING.md` (every construct, with its Svelte origin).
- Imports: `$lib/...` is `src/lib/`, `fw` is the framework runtime (`vite.config.ts`, `tsconfig.json`).
- Icons: `import X from '$icons/x'` (lucide). Their data is vendored in `framework/icons/lucide.json`;
  `node framework/tools/icons.mjs --add` vendors a newly imported one (`npm run check` fails until it is).
- A source file starts with a comment saying what it holds.

## Screenshot parity and performance

Studio moved from Svelte 5 to fw with every screen and state pixel-identical (CHANGELOG, "Studio on fw"); the DOM
of that comparison differed only where the Svelte app had bugs (its palette kept a page's commands after leaving it)
or was not deterministic (the Overview's live numbers, the phone label grid). The same tools keep later changes
honest: shoot every state, compare it with a stored shoot, and measure against another version's build.

```sh
node scripts/shoot.mjs --exact --prod --out <dir>            # a stored shoot of this version, to compare with later
node scripts/parity.mjs --run --baseline <dir> --only runs   # shoot the app (exact mode) and diff with a stored shoot
node scripts/parity.mjs --run --baseline <dir> --out <dir>   # all states, results in --out
node scripts/parity.mjs --run --self          # shoot it twice: the determinism check
node scripts/parity.mjs <dirA> <dirB>         # compare two shot folders
node scripts/perf.mjs --runs 5                # load, memory and CPU of this version
node scripts/perf.mjs --ref origin/main       # against another git ref (its web/ built in a temporary worktree)
node scripts/perf.mjs --baseline-dist <dir>   # against a stored build (another version's dist/)
node scripts/perf.mjs --folo <folo binary> --net 50,20 --only cold,warm   # served by folo serve, a remote server's network (Chrome's network throttling)
```

- `shoot.mjs --exact` (what parity uses) makes a shot reproducible to the pixel: the clock paused as for `--qa`,
  `Math.random` constant (the mock's latency jitter), real-time work (blobs, image loads and decodes, chunks, fonts)
  awaited before and after every clock step (`--qa` does that too, so a loaded machine cannot run a scene's clock out
  before its page is in), Chrome drawing in software, animations frozen at the shot. It also writes a
  normalized DOM (`<shot>.dom.txt`) and `shoot.json` (failed states and why). `--out`, `--base <url>`, `--prod`,
  `--jobs N`; `--only a,b` takes a comma list. Scope classes `fw-*` (`FOLO_SCOPE_PREFIX` names another prefix) and
  `svelte-*` (so the Svelte app's stored shoots stay comparable) are left out of the DOM dumps.
- `parity.mjs` reports per shot: identical, N differing pixels + bounding box, size mismatch, or missing (a setup that
  timed out); `parity.md`, `parity.json` and `diff/<shot>.png` (magenta over a dimmed copy) in the out dir, with the
  first difference of the two DOM dumps. A state that differs is shot again (`--retries 1`) and passes if any of its
  shots matches. Pass `--baseline <dir>` (a folder of `shoot.mjs --exact` shots) or `--self`: the built-in default
  is the maintainers' stored shoot of the last Svelte build, which is not in the repository. Exit 1 on any difference.
- `perf.mjs` measures the production build (built into `<out>/.dist`, `vite preview`, Chrome + CDP): the bundle, cold
  loads of the main screens (FCP, content shown, script/task/layout/style time, long tasks, heap after GC, nodes,
  listeners), hash navigation cycles (with the heap growth), 15 s live on the Run page and Overview (busy %, rAF/s),
  and the dataset grid scrolled end to end. With a baseline (`--baseline-dist`, `--ref`) both are measured,
  alternated, and compared. `--api <url>` uses a running `folo serve --mock` instead of `?mock=1`; `--folo <binary>`
  serves each build with that binary (`folo serve --mock --web-dir`, so the server's own headers and compression are in
  the numbers; `--base-folo` serves the baseline with another one), and `--net <rtt ms>,<Mbit/s>` emulates a network
  for the cold loads and the `warm` ones with Chrome's network throttling (a delay per request and a bandwidth cap; it
  does not model TCP connection set-up or slow start, which cost one large file more round trips than many small
  ones, so its numbers are that emulation's, not a real link's). `--served base=<url>,app=<url>` measures builds a
  server already serves there instead (another machine, or a relay over a link with real delay); the bundle figures
  still come from the builds on disk, so each served `index.html` must be the one on disk and every file a page loads
  must be in it, or the run fails. A load's requests and KB are counted from CDP's Network events, each at its own
  time, cut at the moment the content showed (the page's mark, on CDP's clock through the document's navigation
  start): the HTTP requests sent by then (each redirect hop one, one still waiting for its first byte too) and the
  bytes received by then as Chrome counts them on the wire (encoded; every response's headers and body, a redirect
  hop's too; a response still arriving with what had arrived; the document's own body, which Chrome reports only
  whole, when it finished). `scripts/perfWire.mjs` does the counting (`perfWire.test.mjs`). `warm` is a reload of the
  same build from a filled cache, after the first load went quiet (no request but an event stream out, nothing on the
  wire for 500 ms): the requests and KB that still went out (revalidations count, what the cache answered does not),
  not a return visit after a release, where a new build renames almost every chunk (on main too).
  `--only bundle,cold,warm,nav,live,large` (default all but warm). Writes `perf.md` (median, min–max, change %) and `perf.json` (every sample).
