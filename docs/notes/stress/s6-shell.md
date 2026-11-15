# Stress round, lane s6-shell: FOLO Studio's shell

What a user meets around every screen: routing, deep links, Back and Forward, the command palette, the global keys,
Settings, preferences in localStorage, layout at extreme widths, and every screen when the server misbehaves.

## How it was tested

- `folo serve` (shared prebuilt binary, `--device cpu`) on a copy of the seed data, serving this branch's `web/dist`;
  a second one with `--token` on its own data dir. Studio driven in headless Chrome (playwright-core), screenshots viewed.
- A mutating proxy in front of the server (lane script `proxy.mjs`, switched per request with `/__mode?m=`): 500, 404,
  truncated JSON, HTML instead of JSON, every array replaced by an object (`wrongtype`), every top-level field dropped
  (`dropall`) or set to null (`nullall`), an empty object, a 10 s delay. Every list and detail route walked through each
  mode, checked for page errors and for a visible error with Retry.
- Deep links to every route, unknown ids on every `:id` route, malformed percent-encoding, `#//`, a 5000-character hash,
  traversal-looking ids. Corrupt, wrong-typed, array, string and 4 MB `folo.prefs`.
- Back and Forward hammered (60 random steps, no waiting, two seeds). Palette: Ctrl+K 21 times, garbage queries,
  Enter with no results, Ctrl+K over the shortcuts dialog, focus return. Single keys typed inside the Settings field.
- Server killed mid-session on Runs, then Models, Settings, Datasets, Overview visited, then the server restarted.
- 320×640 and 3840×2160 through every screen.

## Findings

| id | severity | what | status |
| --- | --- | --- | --- |
| s6-shell-1 | blocker | `folo.prefs` with a non-string `serverUrl` (`42`, `null`): Studio offline for good, Settings never opens (`e.trim is not a function`), `{"theme":"neon"}` applied as is | fixed: `web/src/lib/prefs.ts` `sanitize()` puts values of the wrong type back to their defaults; test `prefs.test.ts` |
| s6-shell-2 | blocker | a screen that throws while created leaves the previous screen on display under the new URL, no error (seen with s6-shell-1's state: URL `#/settings`, Overview shown) | fixed: `web/src/App.tsx` `show()` catches it and shows `ScreenCrash` (`ScreenError.tsx`) with Reload; test `App.test.tsx` |
| s6-shell-3 | annoying | a list answer of the wrong shape (array missing, null, an object): Benchmarks and Labeling jobs skeleton forever with `(s.data ?? []).filter is not a function`; with the key missing, lists read as empty (wrong data shown as right) | fixed: `web/src/lib/api/client.ts` list getters go through `#list`, which throws `bad_response`; every screen shows its error with Retry; test `client.test.ts` |
| s6-shell-4 | annoying | proxy `nullall`, `#/datasets/<id>`: skeleton forever, no error | deferred: `routes/Dataset.tsx`, lane s1-datasets' area |
| s6-shell-5 | polish | `#//` said "There is no page at ." | fixed: `web/src/lib/router.ts`, `#//` is the Overview; test `router.test.ts` |
| s6-shell-6 | polish | a palette pick made over the shortcuts dialog navigates but leaves that dialog open on the new page | deferred: Ctrl+K over a dialog is deliberate (`lib/shell/shortcuts.ts:104`); one Escape closes it |
| s6-shell-7 | polish | Overview at 320 px: Recent runs cuts run names to one letter | deferred: a layout decision for phones |

## Checked and fine

- Every route deep-linked; unknown ids give a clear "not found"; malformed percent-encoding and long hashes give Not
  found, never a crash. Corrupt JSON and 4 MB prefs load with defaults.
- Back/Forward hammering ends with screen, heading and sidebar highlight matching the URL.
- Palette: fast toggling, garbage queries, Escape and focus return work. Single keys in a text field type, never navigate.
- Server gone: every screen says so, with Retry; server back: screens recover by themselves within seconds.
- A `--token` server: every screen names the token wall (documented: Studio cannot send a token).
- 500, 404, truncated JSON, HTML, a 10 s delay through the proxy: every screen shows an error it can recover from.

## Not reached

SSE streams that stall or send garbage on a live run's page, 200% zoom and reduced motion, the accessibility tree of
each screen beyond the dialogs' focus, the Cloud panel against `folo serve --mock`, two tabs changing prefs at once.

## Gates

`npm run check` passed; `npm test` 1742 of 1744 passed, the two others (`LineChart.perf.test.tsx` memoPathsOf,
`Logs.test.tsx` streaming) timed out at 5 s on the loaded PC and pass alone with `--testTimeout 60000`; neither touches
these changes. `npm run build` passed. `tools/file-size.sh` passed. No Rust was touched.
