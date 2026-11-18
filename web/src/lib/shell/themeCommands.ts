// The command palette's "Theme: <name>" commands (design/THEMES.md, Choosing a theme): one per theme of either mode,
// custom themes included, in Settings' order. Each is offered for a query only ("theme", "midnight", "light theme"),
// so they never crowd the open palette; the theme on screen says so. Choosing one is `prefs.useTheme`: it takes its
// mode's slot and, under Dark or Light, switches the mode to its own.

import SunMoon from '$icons/sun-moon';
import { prefs } from '$lib/prefs';

export function themeCommands() {
  const active = prefs.activeTheme.id;
  return (['dark', 'light'] as const).flatMap((mode) =>
    prefs.themesFor(mode).map((t) => ({
      id: `theme-${t.id}`,
      group: 'Actions',
      label: `Theme: ${t.name}`,
      match: [t.name, `${t.name} theme`, `${mode} theme`],
      hint: [mode === 'dark' ? 'Dark' : 'Light', ...(t.custom ? ['custom'] : []), ...(t.id === active ? ['in use'] : [])],
      scoped: true,
      icon: SunMoon,
      run: () => void prefs.useTheme(t.id),
    })),
  );
}
