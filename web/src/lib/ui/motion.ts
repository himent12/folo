// Durations for JS-driven transitions (svelte/transition). The CSS reduced-motion rule (base.css, tokens.css) cannot
// reach them, so they ask here: with `prefers-reduced-motion: reduce`, every duration is 0.

/** True when the user asked for reduced motion. */
export function reducedMotion(): boolean {
  return typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** `ms`, or 0 under reduced motion. Read when the transition starts, so a changed preference applies at once. */
export function motion(ms: number): number {
  return reducedMotion() ? 0 : ms;
}
