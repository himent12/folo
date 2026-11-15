# Stress round, lane d5-window-input: FOLO Desktop's window, text input, focus, palette and command line

Base `1436ad68`. Everything ran off screen (`FOLO_UI_OFFSCREEN=1`, driven with posted window messages,
never the clipboard), on the shared prebuilt `folo-desktop`, with the lane's own `FOLO_HOME` and
`FOLO_DESKTOP_STATE`, `--mock` or a lane server on port 18101.

## What was tested, and how

- **Command line** (a matrix script, every case twice, exit code + stderr + the PNG): `--width` / `--height`
  of 1, 0, -5, NaN, inf, 1e30, 20000, `abc`, a missing value, with `--shot` and live; `--scale` inf, NaN, 0,
  1e30, 0.0001; `--shot-time` -1, NaN, 1e12; `--shot` to `C:\Windows`, to a directory, under a file, to a
  missing drive; `--theme x` / `LIGHT`; unknown flags and typos (`--sever`); `--open` with unknown ids and
  no value; `--scenario` unknown and without a value; `--bench-frames` 0, -1, `abc`, huge; `--help` and
  `--version` with junk around them.
- **The state file**: corrupt JSON, wrong types, a window mode from a "newer" build, width 0 / -500 / 100000,
  x 1e30 and 3.4e38 (far off every display), a read-only file, a directory in its place, a missing parent
  folder with unicode and spaces.
- **Text input** (the palette's field and the settings fields): 10,000 characters typed as WM_CHAR, held
  Left / Backspace on it (autorepeat lParam), emoji, a ZWJ family, combining marks, Hebrew and Arabic, CJK,
  lone surrogates and control characters, Home / End, word jumps and word deletes, keyboard selection,
  bare and partial IME composition messages.
- **Palette**: every group, garbage and regex-looking queries, Ctrl+K hammered 201 times, a character and
  Enter posted back to back, Esc with nothing open.
- **Focus**: Tab / Shift+Tab / F6 on the pages, where focus lands after the new-run dialog closes.
- **Toasts**: fifty at once (layout, handles).

## Findings

| id | sev | what | status |
|---|---|---|---|
| d5-window-input-1 | annoying | A flagless `WM_IME_COMPOSITION` (an IME ending a composition) with nothing composed deleted the selected text: `ui/src/platform/windows/keyboard.rs` cleared `replace_text_in_range(None, "")`, which falls back to the selection. | fixed: only composed text is dropped; tests `an_ended_composition_drops_only_composed_text` and, on a hidden window through the message handler, `typed_text_is_windows_translation` |
| d5-window-input-2 | annoying | Shift+Home, Shift+End, Ctrl+Shift+Left/Right and Ctrl+Delete did nothing in every text field (`src/input.rs` bound none of them). | fixed: five bindings + actions; tests in `src/app/text_input_tests.rs` |
| d5-window-input-3 | annoying | `--shot … --width inf` aborted ("memory allocation of 13743895344000 bytes failed"), `--scale inf` panicked ("capacity overflow", exit 101), `--shot-time 1e12` never exited; 0 / NaN / negative failed late and vaguely. `ui/src/app/shot.rs` and `bench.rs` checked only `> 0`. | fixed: `check_size` (finite, 1 px, 1 to 16384 device px a side as the framebuffer rounds them, so `--scale 0.0001` too, scale in (0, 8]) and a 10-minute `--shot-time` cap, refused up front; tests `refuses_sizes_and_times_it_cannot_render` (shot) and `refuses_sizes_it_cannot_render` (bench) |
| d5-window-input-4 | polish | One field `desktop.json` couldn't read (a window `mode` from a newer build, a wrong type) threw the whole file away: the theme came back dark and was written back so. | fixed: `window_state::parse` reads field by field; test `a_bad_field_drops_only_itself` |
| d5-window-input-5 | polish | An unknown `--scenario` wrote an Overview shot with exit 0, `--theme x` was silently dark, unknown flags were ignored, `--scenario --shot P` took `--shot` as the name. | fixed: `arg_warnings` names each on stderr (the app still opens); a bad `--theme` falls back to the remembered theme, else dark (`launch_mode`); test `bad_arguments_are_named_not_ignored` |
| d5-window-input-6 | polish | Palette results showed a group heading twice: the group name is part of the matched text, so scores of groups interleaved (`run` gave Runs / Actions / Runs). | fixed: `palette::rank` takes the 40 best matches, then keeps each group together, ordered by its best match; tests `a_group_shows_once_with_its_best_match_first`, `the_cut_keeps_the_best_matches_of_every_group` |
| d5-window-input-7 | polish | Paste into a single-line field replaced only LF: a Windows CRLF left an invisible CR in the value (server address, run name). Found by reading the code; the lane may not touch the clipboard. | fixed: `one_line` maps CRLF / CR / LF / tab to a space and drops other control characters; test `pasted_text_becomes_one_line_without_control_characters` |

### Checked, not bugs

- `desktop.json` corrupt, width 0 / -500 / 100000, x 1e30 / 3.4e38: the window opens at the minimum or the
  display's size, on a display. A directory or a read-only file in its place: one stderr line
  ("couldn't remember the window …: Access is denied"), the app runs, exit 0.
- Live `--width inf|NaN|-5|1|20000|1e30`: the OS clamps, no crash. `--open run:nope`: the "Run not found"
  page. `--shot` to an unwritable place: a clear error, exit 1. `--bench-frames 0|abc`: a clear error, exit 2.
- 10,000 characters: about 0.4 s of CPU to type, caret and scrolling right; 100 held Left / Backspace on it
  about 0.6 s. Every script listed above shapes and carets right; lone surrogates and control characters are
  dropped.
- Palette: no stale action list in a char-then-Enter race; 201 Ctrl+K then typing is fine.
- Fifty toasts: the stack keeps three by design with "Show N earlier"; handle count steady.
- Focus returns to the opener when the new-run dialog closes; F6 cycles title bar search and page.

## Review round 1

Three reviewers found: the palette grouped before its 40-item cut (a big group of weak matches pushed "New
training run" out), `--theme x` did not fall back to the remembered theme as its warning said, a scale that
rounds a side to 0 device pixels passed `check_size`, and the IME and bench fixes lacked tests that fail
without them. All fixed in the round 1 commit, each with a test.

## Not reached

Resizing to the minimum / huge, maximize / restore / F11 on a sandbox desktop; tooltips at window edges;
nested scroll areas with wheel and thumb drag; mouse drag selection; WM_DPICHANGED; two instances writing
the state file at once (code read: the write is a temp file renamed over, last one wins).
