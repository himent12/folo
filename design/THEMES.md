# Themes

FOLO Studio and FOLO Desktop share one theme system. A **theme** is a named colour palette for one **mode** (dark
or light). The user picks the mode (Dark, Light or System) and, separately, which theme each mode uses, so "System"
can follow the operating system between, say, Midnight at night and Paper by day. Everything that is not a colour
(type, space, radii, sizes, motion, the class palette) is the same in every theme.

`tokens.json` is still the single source of truth: `color.dark` and `color.light` are the two base palettes, and
`themes` lists the named themes, each a mode plus overrides merged over that mode's base. The default theme of each
mode (`dark`, `light`) has no overrides, so the apps look exactly as before this system existed until the user picks
another theme.

## Built-in themes

Listed in this order everywhere (Settings, the command palette); the first of each mode is its default.

| id | Name | Mode | Character |
|---|---|---|---|
| `dark` | Dark | dark | FOLO's own dark: near-black neutrals (unchanged). |
| `midnight` | Midnight | dark | Deep blue-black surfaces (hue ≈ 220–230), cool greys for text and lines. |
| `dim` | Dim | dark | Lifted charcoal surfaces (app ≈ L 12 %), lower glare for long sessions; text still AA. |
| `contrast-dark` | High contrast dark | dark | Black surfaces, white text, edges and controls that stand out (AAA text). |
| `light` | Light | light | FOLO's own light: white panels on light grey (unchanged). |
| `paper` | Paper | light | Warm off-white surfaces (hue ≈ 40–50, low chroma), warm greys for text and lines. |
| `contrast-light` | High contrast light | light | White surfaces, black text, edges and controls that stand out (AAA text). |

Rules every theme keeps (checked by `contrast.mjs`, see below):
- **Elevation by lightness** holds, by relative luminance: `bg.inset` < `bg.app` < `bg.panel` ≤ `bg.raised` ≤
  `bg.overlay` in dark themes; `bg.inset` < `bg.app` ≤ `bg.raised` ≤ `bg.panel` and `bg.raised` ≤ `bg.overlay` ≤
  `bg.panel` in light ones (panels white-most).
- **The accent stays the signal lime** in every built-in theme: it means focus, live state and the primary action,
  and the chart series exclude it. A theme may tune `accent.text`, `accent.line`, `accent.muted`, `bg.selected` and
  `focus` (the shades that must hold contrast on its surfaces), not the hue.
- Status hues keep their meaning (success green, warning amber, danger red, info blue); a theme only shifts their
  shades to hold contrast on its surfaces.
- Chart series keep their fixed order and hue families (sky, amber, magenta, teal, violet, orange); a theme may
  shift shades for contrast on its plot surfaces.
- Every pairing in `design/README.md` → Colour holds on every surface of every theme; status and accent text also
  on hovered and selected rows (`bg.hover` / `bg.selected` over `bg.panel`) and on their tint over each of those.
  High contrast themes hold
  stricter floors: text (primary, secondary, tertiary, status text) ≥ 7:1, `line.control` and other non-text marks
  ≥ 4.5:1, and `line.default` (panel edges) ≥ 3:1 against `bg.app`. A custom theme whose base is a `contrast-*`
  theme is checked as a high contrast theme too. One exception: `accent.on` on `accent.line` holds 3:1, not 4.5:1
  (in High contrast light the olive mark cannot be dark enough for white surfaces and light enough for a near-black
  label at once); text on an `accent.line` fill in a light theme uses `text.inverse` instead.
- The class palette (detection boxes on photos) is not part of a theme.

## Custom themes

A custom theme is a small JSON file with the same shape as an entry of `themes`, plus a header:

```json
{
  "folo_theme": 1,
  "id": "night-shift",
  "name": "Night shift",
  "mode": "dark",
  "base": "midnight",
  "description": "Optional, one line.",
  "color": {
    "bg": { "app": "#0B0D12", "panel": "#11141B" },
    "text": { "secondary": "#A3AAB6" },
    "chart": { "series": ["#5CC8FF", "#F2C14E", "#E879F9", "#2DD4BF", "#A78BFA", "#FB923C"] }
  }
}
```

- `folo_theme`: the format version, `1`: any JSON number equal to 1 (`1`, `1.0`, `1e0`; JSON.parse can't tell them
  apart); a string is an error. Required.
- `id`: `[a-z0-9][a-z0-9-]{0,39}`: 1–40 characters of a-z, 0-9 and `-`, starting with a letter or digit (so
  Desktop's `--theme <id>` never reads as a flag: `--help` and `-x` are not ids); not a built-in id, and not
  `system` (reserved: Desktop's `--theme system` is the mode). A custom theme with the id of one already installed replaces it: in Studio the later import wins; in
  Desktop, of two files in the folder with one id, the later file name (in name order) wins. Required.
- `name`: 1–40 characters after trimming. Required. `description`: optional, at most 120 characters after trimming.
  Both are stored trimmed. Lengths count Unicode code points (an emoji is one), and trimming removes Unicode white
  space (the White_Space property) at both ends: the same in every validator. Studio's import also refuses a name
  that holds a control character (a newline, a tab), a line or paragraph separator, or an invisible format character
  (a zero-width space, a bidi override; the zero-width joiner and non-joiner are allowed), "name: must not hold control
  or invisible characters", and one with no letter, digit, punctuation or symbol (the blank ones, the Hangul fillers
  U+115F, U+1160, U+3164, U+FFA0 and the blank Braille pattern U+2800, do not count), "name: must have a visible
  character"; a theme stored before that rule stays installed. Desktop and `design/theme-cases.json` take this rule
  in a later change.
- `mode`: `"dark"` or `"light"`. Required.
- `base`: optional, a **built-in** theme id of the same mode; default: the mode's default (`dark` or `light`). The
  palette is the base theme's resolved palette with `color` merged over it.
- `color`: optional; any subset of the colour tokens, with the same paths as `color.dark` in `tokens.json`, one
  level per object: `{"bg": {"app": …}}`, never a dotted key (`{"bg.app": …}` is `color.bg.app: not a colour token`).
  Colours are `#RRGGBB` or `#RRGGBBAA` (either case). Arrays (`chart.series`, `chart.fill`, `chart.text`) replace
  whole and must have the base's length (6). `chart.area_opacity` is a number from 0 to 1 that scales chart area
  fills; 0 removes them in both apps, and the built-in values look as before. Desktop multiplies it by each chart's
  own factor (the line chart's fill alpha is 1.6 × it, the sparkline's 1.4 ×, at most 1). Studio scales its own fill
  opacities (0.08 for the line chart's gradient top, 0.12 for the sparkline) by it relative to the mode's default
  (`--chart-area-opacity-default`, 0.14 dark, 0.12 light).
- Anything else is an error, so a typo never fails silently: an unknown key or token path, a wrong type, a bad
  colour, a base of the other mode. A file with an error is not installed, and the app says which file and why
  (the first error is enough; give the token path, e.g. `color.bg.ap: not a colour token`). Keys starting with `$`
  are comments and are ignored, as in `tokens.json`. When several keys are wrong, which one a reader names first may
  differ (JavaScript lists an object's integer-like keys, such as `"0"`, before the others; Desktop keeps the file's
  order), but every reader refuses the file.
- A key may appear twice in one object (JSON allows it): the last value wins in every reader, as in serde_json. The
  earlier value is still parsed, so it can still make the file not JSON (below).
- A file is UTF-8 JSON of at most 1 MiB (1 048 576 bytes); a larger one is `(file): larger than 1 MB; not a theme`.
  Studio refuses it by the picked file's size, unread, and checks the length it read again. `contrast.mjs` refuses
  it by the size of the file it opened, unread, and its read stops one byte past the limit, as Desktop's does (Desktop
  reads at most the limit + 1 byte and refuses it then), so a file that grew or was replaced since costs no more. A
  file that can't be read is `(file): couldn't read (…)`, with the system's reason, in every reader.
- Every reader treats a file's bytes alike: exactly one leading UTF-8 byte order mark (EF BB BF, which PowerShell 5.1
  and some editors write) is stripped, then the rest is decoded strictly as UTF-8: invalid bytes (Windows-1252 from
  PowerShell 5.1's `Set-Content`, UTF-16 from its `>`) are `(file): not UTF-8 text`, never replaced. A second byte
  order mark is a U+FEFF before the JSON, which is not valid JSON. Desktop's serde_json refuses three more things
  while parsing that `JSON.parse` would accept, and they are not valid JSON in every reader: nesting deeper than 127
  arrays and objects (the top-level object counted; 128 is refused); a lone surrogate escape (`\uD800` without its
  pair, or a pair in the wrong order) in any string, keys and comments included, even in a member a later duplicate
  replaces; and a number outside a double's range (`1e400`; one too small, `1e-400`, reads as 0). Every reader
  rounds a decimal number to the nearest double (Desktop builds serde_json with `float_roundtrip`, as `Number()`
  rounds), so numbers read alike to the last bit, at the range's edge too: `1.7976931348623158e308` is the largest
  double and `1.7976931348623159e308` is out of range; `folo_theme: 1.00000000000000011` is 1 and
  `0.99999999999999994` is not; `area_opacity: 1.0000000000000001110223` is 1. The JS readers
  find them with one shared scan of the text before `JSON.parse` (`design/json-scan.mjs`, iterative, so no depth
  overflows it). `(file): not valid JSON` is followed by each parser's own detail in parentheses (serde_json's line
  and column, V8's message, or the scan's `nested too deep`, `a lone surrogate in a string`, `a number out of
  range`). In Studio, anything that goes wrong with one file of a pick is that file's result; the others still
  import.
- `design/theme-cases.json` is the shared conformance table: invalid files with their exact first error, and valid
  ones with their resolved result, each parsed (`theme`), as the file's raw text (`text`: number spellings and dotted
  keys survive only that way) or as the file's bytes in hex (`hex`: byte order marks, invalid UTF-8, surrogate
  escapes). `contrast.mjs` and both apps' tests replay it, `text` and `hex` cases through each one's own file reader
  (a temporary file for the CLI and Desktop, a `File` for Studio), so the three validators agree; a change to the
  format changes that file first.
- The apps do not refuse a custom theme for low contrast (it is the user's choice), but `node design/contrast.mjs
  --theme <file>` checks one against the same rules as the built-ins, so theme authors can.

`design/themes/` holds example files (`night-shift.json`, `sepia.json`); `contrast.mjs` checks them too, so they
stay valid examples.

Where custom themes live:
- **Studio**: Settings → Appearance → Custom themes → *Import theme…* reads a `.json` file (a file input; several
  at once is fine). The files of a pick are read one at a time and each is checked as soon as it is read, keeping
  only its result (one over 1 MiB is refused by its size, unread); picks are imported in the order made, a second
  waiting for the first. Imported themes are stored in `localStorage` under `folo.themes` (an array of the validated
  files, stored without their `$` comments and with the name and description trimmed) and listed under the row by
  name, as the selects order them (name; mode · based on its base · id), each with *Remove*. Another tab of the same
  Studio follows changes (the `storage` event), as preferences already do. When the browser refuses the write
  (storage full or blocked), the change stays in this tab, over what other tabs store, until a write succeeds, and
  the row says so ("Couldn't save to this browser's storage; your custom themes last until you close this tab.").
  When storage can't even be read (blocked, `localStorage` null, or touching `localStorage` at all throws, as on a
  sandboxed page's opaque origin: Studio still starts), changes go over the list last read or written,
  so an import still installs for the tab; once a read works again, they go over what it reads.
- **Desktop**: every `*.json` in `<FOLO_HOME or ~/.folo>/themes/` is read at start and again on *Reload* in
  Settings → Appearance → Custom themes. The row shows that folder (mono, in the platform's form) and, under the row, the
  themes it loaded (name; mode · based on its base · file name, as Studio lists them) and then the files that failed,
  each with its error. There is no *Remove*: deleting the file and *Reload* removes a theme, as the row says. Desktop
  never opens a file picker or Explorer for this.

## Choosing a theme

Settings → Appearance holds four rows (`UNIFIED.md` §4, Settings):

1. **Theme**: a Dark / Light / System segmented control, as before (Desktop gains System). With System, its
   description says what it follows now ("Follows your operating system, currently dark.").
2. **Dark theme**: a select of every dark theme, built-ins in the order above, then custom themes by name: by the
   lower-cased name, then by id, both compared by code point (no locale collation, so both apps list them alike).
   Description: "Used when the theme is Dark, or System while your system is dark." A closed select cannot wrap: in
   Studio, when the chosen theme's name is wider than the select, the description ends with "Chosen: <the whole name>".
3. **Light theme**: the same for light themes.
4. **Custom themes**: as above, per app.

Picking a different theme in a slot's select applies it at once: choosing a dark theme while the mode is Light
switches the mode to Dark (and the other way round); with System, the mode stays System and the choice applies
whenever that mode is in effect. Re-picking the theme a select already shows (its slot's theme, or the default it
shows for a stored id that names no theme) changes nothing, in both apps and by every input (pointer, Enter, Space):
the mode stays, and so does the stored id. To use the other mode's theme, switch the mode (the Theme row, the top
bar's button, Ctrl Shift L) or use its palette command.

The command palette has one command per theme, "Theme: Midnight" and so on (custom themes too), besides the existing
toggle. A command names a theme, not a slot, so it always applies its theme: it fills the slot and switches an
explicit mode to the theme's, also when the slot has that theme already. They are offered for a query only ("theme", "midnight", "light theme"), never in
the empty palette, so seven or more commands don't crowd it. The top bar's theme button (Studio) and Ctrl Shift L
(Desktop, and Studio where it has it) keep switching the mode, not the theme. On Desktop, Ctrl Shift L under System
leaves System for the explicit opposite of the mode shown (System showing dark: Light).

A stored theme id that no longer exists (a removed custom theme, a typo in a hand edit) shows the mode's default,
and the stored id stays, so installing that theme again brings it back. Selects show the default as selected then.

## Storage

- Studio: `folo.prefs` gains `darkTheme` and `lightTheme` (theme ids; default `"dark"` / `"light"`); `theme` stays
  the mode preference (`"dark" | "light" | "system"`). `index.html`'s first-paint script applies the resolved theme
  (custom themes included) so there is no flash of the default, accepting exactly what the app does: the mode from a
  valid legacy `folo.theme`, else `folo.prefs`' `theme`, else dark; a stored custom theme only when the whole entry
  passes the check, setting only colour token properties. Its data (the built-in ids, each colour token's custom
  property) is generated by `npm run tokens`, and `npm run check` fails when it is stale.
- Desktop: `desktop.json` keeps `theme` as the mode (`"dark" | "light" | "system"`, the old file's values read the
  same) and gains `dark_theme` and `light_theme`. `--theme` on the command line takes a mode or a theme id
  (`--theme midnight` means mode Dark with Midnight), for screenshots and tests.

## How each app applies a theme

- **Studio**: `<html data-theme="dark|light">` stays the resolved **mode** (every `[data-theme='light']` rule keeps
  working); `<html data-theme-id="<id>">` names the theme. `npm run tokens` writes one CSS block per non-default
  built-in theme, `:root[data-theme-id="<id>"] { … }`, after the mode blocks, holding the full resolved palette.
  A custom theme sets `data-theme-id` to its base and its overrides as inline custom properties on `<html>`.
  `tokens.ts` exports every built-in theme's resolved palette (canvas and SVG code reads colours from the active
  theme, never from `color.dark` / `color.light` by mode alone).
- **Desktop**: `build.rs` generates one `Colors` constructor per built-in theme and a `THEMES` table (id, name,
  mode, description, constructor), plus `Colors::set(&mut self, path: &str, value) -> bool` for custom overrides by
  token path (`bg.app`, `chart.series.0`). `Theme` keeps `mode` (so `is_dark()` and the mode's shadows work) and
  gains `id`. System mode follows the OS appearance live (Win32: `WM_SETTINGCHANGE` "ImmersiveColorSet";
  winit: its theme-changed event); the headless platform reports dark until a test changes it
  (`HeadlessApp::set_system_appearance`, delivered through the same window callback). The window frame (DWM's caption and border
  on Windows) follows the mode shown, not the OS: a Light theme draws a light frame on a dark system.

## Checks

- `node design/contrast.mjs` checks every built-in theme and every file in `design/themes/` (each theme against its
  mode's rules, plus the high-contrast floors for `contrast-*`), and `--theme <file>` checks one custom file and
  prints its result without failing on others. `npm run check` runs it, as before.
- Both apps test: parsing (every error case above names its path), resolution (base + overrides, unknown stored id →
  default), the mode/slot rules of *Choosing a theme*, and persistence.
- The default themes stay pixel-identical: shots of the default Dark and Light before and after this change match
  on every channel, except the Settings page, where the Appearance panel itself changed (its new rows) and what
  follows it moved down. Above that panel, Studio's Settings matches too; Desktop's has the same colours, but its
  content sits up to 2 device pixels lower (at scale 1.5), because folo-ui's layout of that page depends on the
  height of what follows (the same build moves it alike with the panel left out).
