// Shows a theme on <html> (design/THEMES.md, How each app applies a theme): `data-theme` is the mode (every
// [data-theme='light'] rule and the mode's shadows), `data-theme-id` the built-in palette whose block tokens.css holds
// (a custom theme's base), and a custom theme's own colours inline over it. index.html's first-paint script does the
// same before the app has loaded; this one then takes over and corrects anything it set from a stale entry.

import type { Theme } from './themes';
import { colorVars } from './tokens';

export function applyTheme(root: HTMLElement, theme: Theme) {
  root.dataset.theme = theme.mode;
  root.dataset.themeId = theme.base;
  const keep = new Map(theme.vars);
  // Only the colour properties: <html> carries others inline too (the New run dock's height, shell/dock.ts).
  for (const [, name] of colorVars) if (!keep.has(name)) root.style.removeProperty(name);
  for (const [name, value] of theme.vars) if (root.style.getPropertyValue(name) !== value) root.style.setProperty(name, value);
}
