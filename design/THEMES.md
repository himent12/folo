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
- Every pairing in `design/README.md` → Colour holds on every surface of every theme. High contrast themes hold
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

- `folo_theme`: the format version, `1`. Required.
- `id`: `[a-z0-9-]`, 1–40 characters, not a built-in id. A custom theme with the id of one already installed replaces
  it. Required.
- `name`: 1–40 characters after trimming. Required. `description`: optional, at most 120 characters.
- `mode`: `"dark"` or `"light"`. Required.
- `base`: optional, a **built-in** theme id of the same mode; default: the mode's default (`dark` or `light`). The
  palette is the base theme's resolved palette with `color` merged over it.
- `color`: optional; any subset of the colour tokens, with the same paths as `color.dark` in `tokens.json`. Colours
  are `#RRGGBB` or `#RRGGBBAA` (either case). Arrays (`chart.series`, `chart.fill`, `chart.text`) replace whole and
  must have the base's length (6). `chart.area_opacity` is a number from 0 to 1.
- Anything else is an error, so a typo never fails silently: an unknown key or token path, a wrong type, a bad
  colour, a base of the other mode. A file with an error is not installed, and the app says which file and why
  (the first error is enough; give the token path, e.g. `color.bg.ap: not a colour token`). Keys starting with `$`
  are comments and are ignored, as in `tokens.json`.
- `design/theme-cases.json` is the shared conformance table: invalid files with their exact first error, and valid
  ones with their resolved result. `contrast.mjs` and both apps' tests replay it, so the three validators agree;
  a change to the format changes that file first.
- The apps do not refuse a custom theme for low contrast (it is the user's choice), but `node design/contrast.mjs
  --theme <file>` checks one against the same rules as the built-ins, so theme authors can.

`design/themes/` holds example files (`night-shift.json`, `sepia.json`); `contrast.mjs` checks them too, so they
stay valid examples.

Where custom themes live:
- **Studio**: Settings → Appearance → Custom themes → *Import theme…* reads a `.json` file (a file input; several
  at once is fine). Imported themes are stored in `localStorage` under `folo.themes` (an array of the validated
  files) and listed under the row, each with *Remove*. Another tab of the same Studio follows changes (the `storage`
  event), as preferences already do.
- **Desktop**: every `*.json` in `<FOLO_HOME or ~/.folo>/themes/` is read at start and again on *Reload* in
  Settings → Appearance → Custom themes. The row shows that folder (mono) and how many themes it loaded; files that
  failed are listed under it with their error. Desktop never opens a file picker or Explorer for this.

## Choosing a theme

Settings → Appearance holds four rows (`UNIFIED.md` §4, Settings):

1. **Theme**: a Dark / Light / System segmented control, as before (Desktop gains System). With System, its
   description says what it follows now ("Follows your operating system, currently dark.").
2. **Dark theme**: a select of every dark theme, built-ins in the order above, then custom themes by name.
   Description: "Used when the theme is Dark, or System while your system is dark."
3. **Light theme**: the same for light themes.
4. **Custom themes**: as above, per app.

Picking a theme applies it at once: choosing a dark theme while the mode is Light switches the mode to Dark (and the
other way round); with System, the mode stays System and the choice applies whenever that mode is in effect.

The command palette has one command per theme, "Theme: Midnight" and so on (custom themes too), with the same
effect, besides the existing toggle. The top bar's theme button (Studio) and Ctrl Shift L (Desktop, and Studio
where it has it) keep switching the mode, not the theme.

A stored theme id that no longer exists (a removed custom theme, a typo in a hand edit) shows the mode's default,
and the stored id stays, so installing that theme again brings it back. Selects show the default as selected then.

## Storage

- Studio: `folo.prefs` gains `darkTheme` and `lightTheme` (theme ids; default `"dark"` / `"light"`); `theme` stays
  the mode preference (`"dark" | "light" | "system"`). `index.html`'s first-paint script applies the resolved theme
  (custom themes included) so there is no flash of the default.
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
  winit: its theme-changed event); the headless platform reports dark.

## Checks

- `node design/contrast.mjs` checks every built-in theme and every file in `design/themes/` (each theme against its
  mode's rules, plus the high-contrast floors for `contrast-*`), and `--theme <file>` checks one custom file and
  prints its result without failing on others. `npm run check` runs it, as before.
- Both apps test: parsing (every error case above names its path), resolution (base + overrides, unknown stored id →
  default), the mode/slot rules of *Choosing a theme*, and persistence.
- The default themes stay pixel-identical: shots of the default Dark and Light before and after this change match.
