// Keyboard helpers shared by the shortcut registry, the palette and <Kbd>.

export const isApple = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);

const glyphs: Record<string, string> = {
  Mod: isApple ? '⌘' : 'Ctrl',
  Shift: isApple ? '⇧' : 'Shift',
  Alt: isApple ? '⌥' : 'Alt',
  Enter: '↵',
  Escape: 'Esc',
  ArrowUp: '↑',
  ArrowDown: '↓',
  ArrowLeft: '←',
  ArrowRight: '→',
};

/** "Mod+Shift+K" → ["⌘", "⇧", "K"]; "G R" (a sequence) → ["G", "R"]; "+ -" → ["+", "-"]. */
export function formatKeys(keys: string): string[] {
  return keys
    .split(' ')
    .flatMap((t) => (t === '+' ? [t] : t.split('+')))
    .filter(Boolean)
    .map((k) => glyphs[k] ?? (k.length === 1 ? k.toUpperCase() : k));
}

/** True when the event comes from a text field, where single-key shortcuts must not fire. */
export function isTyping(e: KeyboardEvent): boolean {
  const el = e.target as HTMLElement | null;
  if (!el) return false;
  return el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName);
}
