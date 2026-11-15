# Stress round 4, lane d4-bench-settings: Desktop Benchmarks, Settings and the cloud dialogs

FOLO Desktop driven off screen (`FOLO_UI_OFFSCREEN=1`, window messages only, shared prebuilt binaries) against a
real `folo serve` on the CPU (port 18090, seed data dir), a second empty server (18091) and `folo serve --mock`
(18092) for the cloud. Every number checked against the API JSON; every finding reproduced twice.

## What was tested
- **Benchmarks, real server.** Two benchmarks made through the API with their own small plans (24 shapes images:
  haze 1, 5 + sensor noise 3; 32 COCO photos with a unicode, 90-character name: low light 2 + JPEG 1, 5). Model
  entries run on the CPU (folo-n-coco, the seed's shapes models) watched live over the stream; leaving and
  returning mid-run; the server killed mid-run and restarted (the entry becomes Failed, "the server stopped before
  the entry finished"); Run pressed twice; Stop pressed as the entry finished; delete with its second click.
- **Many and odd entries.** 26 external entries through `POST …/entries` (unicode, 100-character, HTML-like and
  padded names; unscored; partial; unchanged images only; zero scores tied): leaderboard order, ranks, every score
  and axis against the API, "–" for an axis not measured, partial badges.
- **Forms and keys.** New benchmark, New loop, Design with an agent, the delete confirmations: Esc, Enter,
  Ctrl+Enter. Every shortcut Settings › Keyboard lists, on Benchmarks, Playground and Settings, and the bindings it
  does not list (`bind_keys`).
- **Settings.** Server address changed to garbage, an unreachable host, the second live server and back (also into
  a benchmark id the other server lacks); the demo-data switch both ways (the status line says "Built-in demo",
  real data comes back); Device throughput's Measure now pressed twice; the window at its minimum size.
- **Cloud, mock only.** Connect (empty key, unicode key, Enter through the fields), Check, Disconnect, Terminate
  confirm, every dialog cancelled with Esc; the `+trouble` (revoked key, 502 banner) and `+empty` scenarios.

## Findings

| id | severity | what | status |
|---|---|---|---|
| d4-bench-settings-1 | annoying | Benchmarks' inline forms (New benchmark, New loop, Design) ignore Esc, Enter and Ctrl+Enter; Esc on a benchmark page leaves it and drops the form | fixed |
| d4-bench-settings-2 | polish | A long benchmark (or run, model) name in the title bar runs under the search field with no "…"; on an entry page the entry's own name is pushed out of sight | fixed |
| d4-bench-settings-3 | annoying | The Plan tab calls any plan no agent designed "The default plan", also a two-condition one written by hand (Studio says "Written by hand.") | fixed |
| d4-bench-settings-4 | annoying | A server address that is not a URL ("abc def", "http://[::") reads "Could not reach folo serve at abc def" with a hint to start a server | fixed (message); the retry banner stays |

### d4-bench-settings-1: Esc, Enter and Ctrl+Enter on the Benchmarks forms
Repro: Ctrl+6, New benchmark; the name field has focus. Enter: nothing. Ctrl+Enter: nothing (Settings › Keyboard
lists it). Esc: the form stays. On a benchmark with Loops › New loop open, Esc leaves the page for the list and the
form is lost. Cause: `dismiss` (`src/app/overlay_state.rs`) knew no bench form and fell through to the parent
route; the Submit action (`src/app/render.rs`) handled only the dialogs. Fix: `src/bench/keys.rs` names the form on
screen; Esc stands a delete's second click down first, then closes the form (Cancel, Close and Discard too) and
gives the focus back to the button that opened it, then goes back; Ctrl+Enter (and Enter in the New benchmark name)
does what Create, Start loop or Ask does, never while the command palette is open over the form and never while that
button is off (no agent installed, a pool with no images: the form then says why). One exception on purpose: with
Design's proposal on screen Ctrl+Enter does nothing, as using it replaces the plan and is left to "Use this plan"
after a look. The forms and Settings › Connection show an error's hint under its words. The Keyboard list's line now
reads "Submit the open dialog or form". Tests: `controls/key_tests.rs`
(`esc_closes_the_new_benchmark_form_and_stays_on_the_list`, `enter_or_ctrl_enter_in_the_new_benchmark_form_creates_it`,
`esc_on_a_benchmark_stands_a_delete_and_the_new_loop_form_down_before_going_back`,
`ctrl_enter_under_the_palette_creates_no_benchmark`, `ctrl_enter_on_a_design_proposal_neither_uses_it_nor_asks_again`).

### d4-bench-settings-2: long names in the title bar
Repro: a benchmark named "COCO photos 🐟 ünïcødé 名前 long name with spaces and more text to see truncation ok", open
it at 1280x800: the crumb is cut mid-letter by the search field; open one of its entries: the entry's name is gone.
Cause: `src/app/titlebar.rs` crumbs were `whitespace_nowrap` without `min_w_0`. Fix: every crumb after the section
shrinks and ends in "…" (`ellipsis()`); the section's name stays whole. Checked with a headless shot.

### d4-bench-settings-3: "The default plan"
Repro: `POST /api/bench/benchmarks` with `plan: {conditions: [{condition: "haze", severities: [1, 5]},
{condition: "sensor_noise", severities: [3]}]}`, open it, Plan. Cause: `src/screens/bench/plan.rs` said "The default
plan" whenever `designed_by` was empty. Fix: `plan_origin` (`src/screens/bench/format.rs`) uses Studio's rule
(`PlanTab.tsx` `isDefault`): the catalog's default (every condition at 1, 3 and 5, weight 1), no conditions, or
written by hand. A plan of the default's shape is called neither while the catalog loads or after it failed (one
condition alone at 1, 3 and 5 is not the default). The Plan tab's line is built by `plan_intro`, which the test
checks. Test: `only_the_default_plan_is_called_the_default_one`.

### d4-bench-settings-4: an address that is not one
Repro: Settings › Connection, type `abc def`, Connect. Cause: `src/api/http/transport.rs` `unreachable` said the
same for every failed request. Fix: an address is judged as typed, before a scheme is added or a `/` trimmed
(`http://` used to become the host "http"): unless it is http or https with a host and, when one is typed, a port
from 1 to 65535 (`foo://some:42`, `ftp://…`, `http://`, `http://:7878`, `http://127.0.0.1:99999`, `localhost:abc`
are not; ureq sent the last two to port 80, review round 2), nothing is sent, also by the Connect probe and it says "abc def is not a server address" with "Type it as
http://host:port, like http://127.0.0.1:7878.", shown under the status in Settings; the sidebar's connection line
says "Not a server address" instead of "Server unreachable". Test:
`an_address_that_is_not_one_says_so_instead_of_blaming_the_server`. Deferred: the lost-connection banner still
retries such an address (the retry loop is the connection lane's, SIBLING).

## Checked and fine
Leaderboard numbers and order against the API (63.4602 shows 63.5; axes match); ties to the tenth share a rank, as in
Studio; live progress over the stream; leave and return mid-run; the server killed mid-run; switching servers with
a benchmark open ("Benchmark not found" on the server that lacks it); an unreachable host; the demo switch; Run's
in-flight guard (two presses 420 ms apart start two entries, both on purpose); Measure now's guard; an empty server's
New benchmark ("Add a dataset first", Create disabled); mock cloud Connect / Check / Disconnect / Terminate and their
Esc; `+trouble` and `+empty`; the minimum window size on the leaderboard and Settings.

## Not done
Design with an agent end to end (no agent CLI may be used here); loops run for real (training is off limits on this
PC beyond tiny CPU runs); reduce motion toggled while a chart animates (only the theme toggle was).
