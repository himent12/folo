// fw twin of types.ts (which types icons as Svelte components) until the switch.
import type { IconComponent } from 'fw';

/** An icon component (lucide via `$icons/<name>`, or our own), rendered at the size the parent picks. */
export type IconType = IconComponent;
